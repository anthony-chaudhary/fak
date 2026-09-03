package codedebt

import (
	"bytes"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	DefaultFileHardMax = 1500
	DefaultFuncHardMax = 200
	DefaultTestMinFunc = 4
)

var excludedDirs = map[string]bool{
	".git":              true,
	".claude":           true,
	".fak":              true,
	".dos":              true,
	".tmp":              true,
	".head_build_check": true,
	".tmp_overlay":      true,
	"_scratch":          true,
	"node_modules":      true,
	"testdata":          true,
	"vendor":            true,
	"__pycache__":       true,
}

// ScanOptions configures the native code debt scanner.
type ScanOptions struct {
	Workspace     string
	Deterministic bool
	FileHardMax   int
	FuncHardMax   int
	TestMinFuncs  int
}

// ScanTree performs a native, deterministic code debt scan across the workspace.
func ScanTree(opts ScanOptions) (*Report, error) {
	root := opts.Workspace
	if root == "" {
		cwd, err := os.Getwd()
		if err != nil {
			return nil, err
		}
		root = cwd
	}
	root = filepath.Clean(root)

	fileHardMax := opts.FileHardMax
	if fileHardMax <= 0 {
		fileHardMax = DefaultFileHardMax
	}
	funcHardMax := opts.FuncHardMax
	if funcHardMax <= 0 {
		funcHardMax = DefaultFuncHardMax
	}
	testMinFuncs := opts.TestMinFuncs
	if testMinFuncs <= 0 {
		testMinFuncs = DefaultTestMinFunc
	}

	report := &Report{
		Timestamp:     time.Now().UTC(),
		Workspace:     root,
		Deterministic: opts.Deterministic,
		DebtByKPI:     make(map[string]int),
		DebtByCat:     make(map[Category]int),
		DebtByPkg:     make(map[string]int),
		Defects:       make([]Defect, 0),
		KPISummaries:  make(map[string]KPISummary),
	}

	allFiles, err := listGoFiles(root)
	if err != nil {
		return nil, err
	}

	var srcFiles, testFiles []string
	for _, rel := range allFiles {
		if strings.HasSuffix(rel, "_test.go") {
			testFiles = append(testFiles, rel)
		} else {
			srcFiles = append(srcFiles, rel)
		}
	}

	testedPkgs, zeroAssertDefects := scanTests(root, testFiles)
	pkgFuncCount, archDefects, formatDefects := scanSource(root, srcFiles, fileHardMax, funcHardMax)
	testDefects, nonTrivialPkgs := checkUntestedPackages(pkgFuncCount, testedPkgs, testMinFuncs)
	depDefects := checkDeps(root)
	honestyDefects := checkHonesty(root)

	allDefects := assembleDefects(archDefects, testDefects, zeroAssertDefects, formatDefects, depDefects, honestyDefects)
	report.Defects = allDefects
	report.TotalDebt = len(allDefects)

	for _, d := range allDefects {
		report.DebtByKPI[d.KPI]++
		for _, cat := range d.Categories {
			report.DebtByCat[cat]++
		}
		if d.Package != "" {
			report.DebtByPkg[d.Package]++
		}
	}

	populateKPISummaries(report, archDefects, testDefects, zeroAssertDefects, formatDefects, depDefects, honestyDefects, nonTrivialPkgs)
	calculateScoreAndGrade(report)

	return report, nil
}

func scanTests(root string, testFiles []string) (map[string]bool, []Defect) {
	testedPkgs := make(map[string]bool)
	var zeroAssertDefects []Defect

	for _, rel := range testFiles {
		fullPath := filepath.Join(root, filepath.FromSlash(rel))
		contentBytes, err := os.ReadFile(fullPath)
		if err != nil {
			continue
		}
		pkg := PackageOf(rel)

		fset := token.NewFileSet()
		fileAst, pErr := parser.ParseFile(fset, fullPath, contentBytes, 0)
		if pErr != nil {
			continue
		}

		for _, decl := range fileAst.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			fnName := fn.Name.Name
			if strings.HasPrefix(fnName, "Test") || strings.HasPrefix(fnName, "Benchmark") ||
				strings.HasPrefix(fnName, "Fuzz") || strings.HasPrefix(fnName, "Example") {
				testedPkgs[pkg] = true
			}

			if isTestFuncDecl(fn) && !funcCanFail(fn) {
				pos := fset.Position(fn.Pos())
				raw := "zero-assertion test (cannot fail): " + rel + ":" + strconv.Itoa(pos.Line) + " " + fnName
				zeroAssertDefects = append(zeroAssertDefects, Defect{
					KPI:        "assertion_strength",
					Categories: []Category{CategoryInternalCoherence},
					Raw:        raw,
					Path:       rel,
					Package:    pkg,
					Kind:       "zero-assertion-test",
					Line:       pos.Line,
				})
			}
		}
	}
	return testedPkgs, zeroAssertDefects
}

func scanSource(root string, srcFiles []string, fileHardMax, funcHardMax int) (map[string]int, []Defect, []Defect) {
	pkgFuncCount := make(map[string]int)
	var archDefects []Defect
	var formatDefects []Defect

	for _, rel := range srcFiles {
		fullPath := filepath.Join(root, filepath.FromSlash(rel))
		contentBytes, err := os.ReadFile(fullPath)
		if err != nil {
			continue
		}
		pkg := PackageOf(rel)

		normalized := bytes.ReplaceAll(contentBytes, []byte("\r\n"), []byte("\n"))
		lines := strings.Split(string(normalized), "\n")
		lineCount := len(lines)
		if lineCount > 0 && lines[lineCount-1] == "" {
			lineCount--
		}

		if lineCount > fileHardMax {
			raw := "god-file " + rel + " (" + strconv.Itoa(lineCount) + " lines > " + strconv.Itoa(fileHardMax) + ")"
			archDefects = append(archDefects, Defect{
				KPI:        "architecture",
				Categories: []Category{CategoryModularity},
				Raw:        raw,
				Path:       rel,
				Package:    pkg,
				Kind:       "god-file",
				Line:       1,
			})
		}

		formatted, fErr := format.Source(normalized)
		if fErr == nil && !bytes.Equal(formatted, normalized) {
			raw := "unformatted (run gofmt -w): " + rel
			formatDefects = append(formatDefects, Defect{
				KPI:        "format",
				Categories: []Category{CategoryInternalConsistency},
				Raw:        raw,
				Path:       rel,
				Package:    pkg,
				Kind:       "unformatted",
			})
		}

		fset := token.NewFileSet()
		fileAst, pErr := parser.ParseFile(fset, fullPath, contentBytes, 0)
		if pErr != nil {
			continue
		}

		for _, decl := range fileAst.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			pkgFuncCount[pkg]++

			startLine := fset.Position(fn.Pos()).Line
			endLine := fset.Position(fn.End()).Line
			fnLen := endLine - startLine + 1
			if fnLen > funcHardMax {
				raw := "god-function " + rel + ":" + fn.Name.Name + " (" + strconv.Itoa(fnLen) + " lines > " + strconv.Itoa(funcHardMax) + ")"
				archDefects = append(archDefects, Defect{
					KPI:        "architecture",
					Categories: []Category{CategoryModularity},
					Raw:        raw,
					Path:       rel,
					Package:    pkg,
					Kind:       "god-function",
					Line:       startLine,
				})
			}
		}
	}
	return pkgFuncCount, archDefects, formatDefects
}

func checkUntestedPackages(pkgFuncCount map[string]int, testedPkgs map[string]bool, testMinFuncs int) ([]Defect, []string) {
	var testDefects []Defect
	var nonTrivialPkgs []string
	for pkg, count := range pkgFuncCount {
		if count >= testMinFuncs {
			nonTrivialPkgs = append(nonTrivialPkgs, pkg)
			if !testedPkgs[pkg] {
				raw := "non-trivial package has no _test.go: " + pkg
				testDefects = append(testDefects, Defect{
					KPI:        "tests",
					Categories: []Category{CategoryInternalCoherence},
					Raw:        raw,
					Path:       pkg,
					Package:    pkg,
					Kind:       "untested-package",
				})
			}
		}
	}
	sort.Strings(nonTrivialPkgs)
	return testDefects, nonTrivialPkgs
}

func checkDeps(root string) []Defect {
	var depDefects []Defect
	goModPath := filepath.Join(root, "go.mod")
	if modBytes, err := os.ReadFile(goModPath); err == nil {
		lines := strings.Split(string(modBytes), "\n")
		inRequireBlock := false
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "//") {
				continue
			}
			if inRequireBlock {
				if strings.HasPrefix(line, ")") {
					inRequireBlock = false
					continue
				}
				fields := strings.Fields(line)
				if len(fields) > 0 && !strings.HasPrefix(fields[0], "//") {
					depDefects = append(depDefects, Defect{
						KPI:        "deps",
						Categories: []Category{CategoryInternalConsistency},
						Raw:        "external dependency added: " + fields[0],
						Path:       "go.mod",
						Package:    ".",
						Kind:       "external-dep",
					})
				}
				continue
			}
			if strings.HasPrefix(line, "require (") {
				inRequireBlock = true
				continue
			}
			if strings.HasPrefix(line, "require ") {
				fields := strings.Fields(line[len("require "):])
				if len(fields) > 0 {
					depDefects = append(depDefects, Defect{
						KPI:        "deps",
						Categories: []Category{CategoryInternalConsistency},
						Raw:        "external dependency added: " + fields[0],
						Path:       "go.mod",
						Package:    ".",
						Kind:       "external-dep",
					})
				}
			}
		}
	}

	goSumPath := filepath.Join(root, "go.sum")
	if _, err := os.Stat(goSumPath); err == nil {
		depDefects = append(depDefects, Defect{
			KPI:        "deps",
			Categories: []Category{CategoryInternalConsistency},
			Raw:        "go.sum exists (the zero-dep invariant broke)",
			Path:       "go.sum",
			Package:    ".",
			Kind:       "gosum-present",
		})
	}
	return depDefects
}

func checkHonesty(root string) []Defect {
	var honestyDefects []Defect
	claimsPath := filepath.Join(root, "CLAIMS.md")
	claimsBytes, err := os.ReadFile(claimsPath)
	if err != nil {
		return honestyDefects
	}

	lines := strings.Split(string(claimsBytes), "\n")
	inFence := false
	tags := []string{"[SHIPPED]", "[SIMULATED]", "[STUB]"}
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
			inFence = !inFence
			continue
		}
		if inFence || !strings.HasPrefix(trimmed, "- [") {
			continue
		}
		count := 0
		for _, tag := range tags {
			if strings.Contains(line, tag) {
				count++
			}
		}
		if count != 1 {
			honestyDefects = append(honestyDefects, Defect{
				KPI:        "honesty",
				Categories: []Category{CategoryInternalConsistency},
				Raw:        "untagged/double-tagged claim: " + trimmed,
				Path:       "CLAIMS.md",
				Package:    ".",
				Kind:       "mis-tagged-claim",
			})
		}
	}
	return honestyDefects
}

func assembleDefects(arch, tests, zeroAssert, format, deps, honesty []Defect) []Defect {
	allDefects := make([]Defect, 0, len(arch)+len(tests)+len(zeroAssert)+len(format)+len(deps)+len(honesty))
	allDefects = append(allDefects, arch...)
	allDefects = append(allDefects, tests...)
	allDefects = append(allDefects, zeroAssert...)
	allDefects = append(allDefects, format...)
	allDefects = append(allDefects, deps...)
	allDefects = append(allDefects, honesty...)

	sort.Slice(allDefects, func(i, j int) bool {
		if allDefects[i].KPI != allDefects[j].KPI {
			return allDefects[i].KPI < allDefects[j].KPI
		}
		if allDefects[i].Package != allDefects[j].Package {
			return allDefects[i].Package < allDefects[j].Package
		}
		if allDefects[i].Path != allDefects[j].Path {
			return allDefects[i].Path < allDefects[j].Path
		}
		return allDefects[i].Raw < allDefects[j].Raw
	})
	return allDefects
}

func populateKPISummaries(report *Report, arch, tests, zeroAssert, format, deps, honesty []Defect, nonTrivialPkgs []string) {
	report.KPISummaries["architecture"] = KPISummary{
		KPI:        "architecture",
		Debt:       len(arch),
		Score:      clampScore(100 - 12*len(arch)),
		Detail:     strconv.Itoa(len(arch)) + " god-file/god-function outlier(s)",
		Categories: []Category{CategoryModularity},
	}
	report.KPISummaries["tests"] = KPISummary{
		KPI:        "tests",
		Debt:       len(tests),
		Score:      clampScore(int(100.0 * float64(len(nonTrivialPkgs)-len(tests)) / float64(max(1, len(nonTrivialPkgs))))),
		Detail:     strconv.Itoa(len(nonTrivialPkgs)-len(tests)) + "/" + strconv.Itoa(len(nonTrivialPkgs)) + " non-trivial packages tested",
		Categories: []Category{CategoryInternalCoherence},
	}
	report.KPISummaries["assertion_strength"] = KPISummary{
		KPI:        "assertion_strength",
		Debt:       len(zeroAssert),
		Score:      clampScore(100 - 10*len(zeroAssert)),
		Detail:     strconv.Itoa(len(zeroAssert)) + " zero-assertion test(s)",
		Categories: []Category{CategoryInternalCoherence},
	}
	report.KPISummaries["format"] = KPISummary{
		KPI:        "format",
		Debt:       len(format),
		Score:      clampScore(100 - 12*len(format)),
		Detail:     strconv.Itoa(len(format)) + " unformatted file(s)",
		Categories: []Category{CategoryInternalConsistency},
	}
	report.KPISummaries["deps"] = KPISummary{
		KPI:        "deps",
		Debt:       len(deps),
		Score:      clampScore(100 - 25*len(deps)),
		Detail:     strconv.Itoa(len(deps)) + " external dependency defect(s)",
		Categories: []Category{CategoryInternalConsistency},
	}
	report.KPISummaries["honesty"] = KPISummary{
		KPI:        "honesty",
		Debt:       len(honesty),
		Score:      clampScore(100 - 20*len(honesty)),
		Detail:     strconv.Itoa(len(honesty)) + " claim honesty defect(s)",
		Categories: []Category{CategoryInternalConsistency},
	}
}

func calculateScoreAndGrade(report *Report) {
	weightedSum := 0.0
	totalWeight := 0.0
	weights := map[string]float64{
		"architecture": 0.25,
		"tests":        0.25,
		"format":       0.15,
		"deps":         0.15,
		"honesty":      0.20,
	}
	for k, w := range weights {
		if s, ok := report.KPISummaries[k]; ok {
			weightedSum += float64(s.Score) * w
			totalWeight += w
		}
	}
	if totalWeight > 0 {
		report.Score = weightedSum / totalWeight
	} else {
		report.Score = 100.0
	}

	switch {
	case report.Score >= 90:
		report.Grade = "A"
	case report.Score >= 80:
		report.Grade = "B"
	case report.Score >= 70:
		report.Grade = "C"
	case report.Score >= 60:
		report.Grade = "D"
	default:
		report.Grade = "F"
	}
}

func listGoFiles(root string) ([]string, error) {
	cmd := exec.Command("git", "ls-files", "--cached", "--others", "--exclude-standard", "*.go")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err == nil {
		lines := strings.Split(string(out), "\n")
		var files []string
		for _, ln := range lines {
			ln = strings.TrimSpace(filepath.ToSlash(ln))
			if ln != "" && strings.HasSuffix(ln, ".go") && !isPathExcluded(ln) {
				files = append(files, ln)
			}
		}
		sort.Strings(files)
		return files, nil
	}

	var files []string
	wErr := filepath.WalkDir(root, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		name := d.Name()
		if d.IsDir() {
			if strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") || excludedDirs[name] {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(name, ".go") {
			return nil
		}
		rel, rErr := filepath.Rel(root, path)
		if rErr != nil {
			return nil
		}
		rel = filepath.ToSlash(rel)
		if isPathExcluded(rel) {
			return nil
		}
		files = append(files, rel)
		return nil
	})
	sort.Strings(files)
	return files, wErr
}

func isPathExcluded(rel string) bool {
	parts := strings.Split(rel, "/")
	for i := 0; i < len(parts)-1; i++ {
		p := parts[i]
		if excludedDirs[p] || strings.HasPrefix(p, ".") || strings.HasPrefix(p, "_") {
			return true
		}
	}
	fileName := parts[len(parts)-1]
	return excludedDirs[fileName]
}

func isTestFuncDecl(fn *ast.FuncDecl) bool {
	if fn.Recv != nil || !strings.HasPrefix(fn.Name.Name, "Test") {
		return false
	}
	if fn.Type.Params == nil || len(fn.Type.Params.List) != 1 {
		return false
	}
	p := fn.Type.Params.List[0]
	star, ok := p.Type.(*ast.StarExpr)
	if !ok {
		return false
	}
	sel, ok := star.X.(*ast.SelectorExpr)
	if !ok || sel.Sel.Name != "T" {
		return false
	}
	return true
}

func testTParamName(fn *ast.FuncDecl) string {
	if fn.Type.Params != nil && len(fn.Type.Params.List) == 1 {
		p := fn.Type.Params.List[0]
		if len(p.Names) > 0 {
			return p.Names[0].Name
		}
	}
	return "t"
}

func funcCanFail(fn *ast.FuncDecl) bool {
	tName := testTParamName(fn)
	canFail := false

	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if canFail {
			return false
		}
		switch x := n.(type) {
		case *ast.CallExpr:
			if isAssertionCall(x, tName) {
				canFail = true
				return false
			}
		}
		return true
	})

	return canFail
}

var assertPkgs = map[string]bool{"assert": true, "require": true}
var failMethodNames = map[string]bool{
	"Error": true, "Errorf": true, "Fatal": true, "Fatalf": true,
	"Fail": true, "FailNow": true, "Run": true, "Skip": true, "Skipf": true, "SkipNow": true,
}

func isAssertionCall(call *ast.CallExpr, tName string) bool {
	switch fn := call.Fun.(type) {
	case *ast.SelectorExpr:
		if ident, ok := fn.X.(*ast.Ident); ok {
			if ident.Name == tName && failMethodNames[fn.Sel.Name] {
				return true
			}
			if assertPkgs[ident.Name] {
				return true
			}
		}
	case *ast.Ident:
		if fn.Name == "panic" {
			return true
		}
	}
	for _, arg := range call.Args {
		if ident, ok := arg.(*ast.Ident); ok && ident.Name == tName {
			return true
		}
	}
	return false
}

func clampScore(s int) int {
	if s < 0 {
		return 0
	}
	if s > 100 {
		return 100
	}
	return s
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
