package debtlane

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// UnitDetectorResult holds all detected findings and evaluated dimensions for a unit of work.
type UnitDetectorResult struct {
	Lane       string
	UnitOfWork string
	Surface    SurfaceClass
	Findings   []FindingProvenance
	Dimensions []DetectorDimension
}

// InspectUnitDetectors runs all 15 detector dimensions on a unit of work.
func InspectUnitDetectors(lane *DebtLane, unitAbsDir string, compRoots ...string) []FindingProvenance {
	if lane == nil {
		return nil
	}

	surface := classifySurface(lane.UnitOfWork)
	var findings []FindingProvenance

	// 1. Headline 5 dimensions
	if !lane.Evidence.HasTests {
		sev := "warning"
		if lane.Criticality == CriticalityCore {
			sev = "critical"
		}
		findings = append(findings, FindingProvenance{
			Dimension: string(DimTestStatus),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  sev,
			Message:   "missing unit tests: no passing *_test.go found",
		})
	}

	if lane.Evidence.ExcessComments {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimCommentHygiene),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  "warning",
			Message:   fmt.Sprintf("excess comment bloat or formulaic comment gaming (%.1f%% comments)", lane.Evidence.CommentRatio*100),
		})
	}

	if !lane.Evidence.Integrated {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimWiringStatus),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  "warning",
			Message:   "disconnected wiring: unreachable from production command graph",
		})
	}

	if !lane.Evidence.Dogfooded && (lane.Criticality == CriticalityCore || lane.Criticality == CriticalityEnabling) {
		sev := "warning"
		if lane.Criticality == CriticalityCore {
			sev = "critical"
		}
		findings = append(findings, FindingProvenance{
			Dimension: string(DimProofStatus),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  sev,
			Message:   "unproven runtime: missing real execution proof or dogfooding receipt in runtime-proofs.json",
		})
	}

	if !lane.Evidence.Benchmarked && (lane.Criticality == CriticalityCore || lane.Criticality == CriticalityEnabling) {
		sev := "warning"
		if lane.Criticality == CriticalityCore {
			sev = "critical"
		}
		findings = append(findings, FindingProvenance{
			Dimension: string(DimBenchmarkStatus),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  sev,
			Message:   "unmeasured performance: missing substantive benchmark function or authority entry",
		})
	}

	// 6. Modularity deficit
	if lane.Evidence.ModularityDeficit {
		msg := "modularity deficit: monolithic god-files (>1500 lines) or god-functions (>200 lines)"
		if len(lane.Evidence.ModularityIssues) > 0 {
			msg = strings.Join(lane.Evidence.ModularityIssues, "; ")
		}
		findings = append(findings, FindingProvenance{
			Dimension: string(DimModularityDeficit),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  "critical",
			Message:   msg,
		})
	}

	// 7. Model coupling
	if lane.Evidence.HasModelHardcoding {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimModelCoupling),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  "warning",
			Message:   "model coupling: hardcoded model family tokens outside model/ or hfhub/ packages",
		})
	}

	// 8. Blast radius
	if lane.Evidence.DependentsCount > 25 {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimBlastRadius),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  "warning",
			Message:   fmt.Sprintf("high blast radius: %d inbound dependents require strict API stability", lane.Evidence.DependentsCount),
		})
	}

	// 9-15: AST-grounded inspections for Go files
	if unitAbsDir != "" {
		astFindings := inspectGoASTDetectors(lane, unitAbsDir, surface)
		findings = append(findings, astFindings...)
	}

	// Non-Go surface-appropriate checks
	if surface == SurfaceSkills && unitAbsDir != "" {
		skillFindings := inspectSkillSurface(lane, unitAbsDir)
		findings = append(findings, skillFindings...)
	} else if surface == SurfaceWorkflows && unitAbsDir != "" {
		workflowFindings := inspectWorkflowSurface(lane, unitAbsDir)
		findings = append(findings, workflowFindings...)
	} else if surface == SurfaceDocs && unitAbsDir != "" {
		docFindings := inspectDocSurface(lane, unitAbsDir)
		findings = append(findings, docFindings...)
	}

	return findings
}

func inspectGoASTDetectors(lane *DebtLane, unitDir string, surface SurfaceClass) []FindingProvenance {
	var findings []FindingProvenance
	fset := token.NewFileSet()

	entries, err := os.ReadDir(unitDir)
	if err != nil {
		return findings
	}

	hasFuzzOrRace := false
	isProtocolOrConcurrency := strings.Contains(lane.Lane, "gateway") ||
		strings.Contains(lane.Lane, "engine") ||
		strings.Contains(lane.Lane, "sched") ||
		strings.Contains(lane.Lane, "sync") ||
		strings.Contains(lane.Lane, "protocol") ||
		strings.Contains(lane.Lane, "ctxmmu")

	// Pre-scan test files in unitDir to discover local test helper functions
	localTestHelpers := make(map[string]bool)
	testFiles := make(map[string]*ast.File)
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		path := filepath.Join(unitDir, e.Name())
		fileNode, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if err != nil {
			continue
		}
		testFiles[e.Name()] = fileNode
		for _, decl := range fileNode.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			if !isTestFunctionName(fn.Name.Name) && !strings.HasPrefix(fn.Name.Name, "Benchmark") && !strings.HasPrefix(fn.Name.Name, "Fuzz") {
				if hasTestingTParam(fn) && classifyTestAssertions(fn.Body, nil) == proofConfirmed {
					localTestHelpers[fn.Name.Name] = true
				}
			}
		}
	}
	// Second pass: resolve helpers that call other helpers
	for _, fileNode := range testFiles {
		for _, decl := range fileNode.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			if !isTestFunctionName(fn.Name.Name) && !strings.HasPrefix(fn.Name.Name, "Benchmark") && !strings.HasPrefix(fn.Name.Name, "Fuzz") {
				if hasTestingTParam(fn) && classifyTestAssertions(fn.Body, localTestHelpers) == proofConfirmed {
					localTestHelpers[fn.Name.Name] = true
				}
			}
		}
	}

	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") {
			continue
		}
		path := filepath.Join(unitDir, e.Name())
		relPath := filepath.Join(lane.UnitOfWork, e.Name())

		// Check for fuzz test files
		if strings.Contains(e.Name(), "fuzz") || strings.Contains(e.Name(), "race") {
			hasFuzzOrRace = true
		}

		// 9. Thin test detection: inspect test files for real assertions
		if strings.HasSuffix(e.Name(), "_test.go") {
			fileNode := testFiles[e.Name()]
			if fileNode == nil {
				var err error
				fileNode, err = parser.ParseFile(fset, path, nil, parser.ParseComments)
				if err != nil {
					continue
				}
			}
			for _, decl := range fileNode.Decls {
				fn, ok := decl.(*ast.FuncDecl)
				if !ok || fn.Body == nil {
					continue
				}
				if strings.HasPrefix(fn.Name.Name, "Fuzz") {
					hasFuzzOrRace = true
				}
				if isTestFunctionName(fn.Name.Name) {
					proof := classifyTestAssertions(fn.Body, localTestHelpers)
					if proof != proofConfirmed {
						startPos := fset.Position(fn.Pos())
						endPos := fset.Position(fn.End())
						span := fmt.Sprintf("%s:%d:%d-%d:%d", filepath.ToSlash(relPath), startPos.Line, startPos.Column, endPos.Line, endPos.Column)
						msg := fmt.Sprintf("thin test hazard: function %s (%s) contains zero assertions", fn.Name.Name, span)
						sev := "warning"
						if proof == proofUnknown {
							msg = fmt.Sprintf("thin test hazard: function %s (%s) assertion proof is unknown (ambiguous)", fn.Name.Name, span)
							sev = "info"
						}
						findings = append(findings, FindingProvenance{
							Dimension: string(DimThinTests),
							Surface:   string(surface),
							Lane:      lane.Lane,
							Path:      filepath.ToSlash(relPath),
							Severity:  sev,
							Message:   msg,
						})
					}
				}
			}
			continue
		}

		// Non-test Go files
		fileNode, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if err != nil {
			continue
		}

		// 10. Stub debt detection: TODO/FIXME/unimplemented panics
		stubFindings := inspectStubDebt(fset, fileNode, relPath, lane, surface)
		findings = append(findings, stubFindings...)

		// 12. Unsafe usage in non-core
		if lane.Criticality != CriticalityCore && lane.Criticality != CriticalityEnabling {
			for _, imp := range fileNode.Imports {
				if imp.Path != nil && imp.Path.Value == `"unsafe"` {
					findings = append(findings, FindingProvenance{
						Dimension: string(DimUnsafeUsage),
						Surface:   string(surface),
						Lane:      lane.Lane,
						Path:      relPath,
						Severity:  "warning",
						Message:   "unsafe usage: importing package unsafe in peripheral/stewardship unit",
					})
					break
				}
			}
		}

		// 13. Subprocess execution in internal libraries
		if surface == SurfaceInternal || surface == SurfacePkg {
			for _, imp := range fileNode.Imports {
				if imp.Path != nil && imp.Path.Value == `"os/exec"` {
					// Allowed in driver/daemon packages, flagged in pure algorithmic/compute leaves
					if !strings.Contains(lane.Lane, "dispatch") && !strings.Contains(lane.Lane, "process") &&
						!strings.Contains(lane.Lane, "flow") && !strings.Contains(lane.Lane, "bridge") {
						findings = append(findings, FindingProvenance{
							Dimension: string(DimSubprocessExec),
							Surface:   string(surface),
							Lane:      lane.Lane,
							Path:      relPath,
							Severity:  "info",
							Message:   "subprocess execution: os/exec imported in library leaf; prefer native Go abstractions",
						})
					}
					break
				}
			}
		}
	}

	// 11. Undocumented exported contracts
	if lane.Evidence.ExportedSymbols > 5 && lane.Evidence.DocumentedExports == 0 {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimUndocumentedExports),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  "info",
			Message:   fmt.Sprintf("undocumented exports: %d exported symbol(s) lack godoc documentation", lane.Evidence.ExportedSymbols),
		})
	}

	// 14. Race/Fuzz status for concurrency/protocol leaves
	if isProtocolOrConcurrency && !hasFuzzOrRace && lane.Evidence.HasTests {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimRaceFuzzStatus),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  "info",
			Message:   "race/fuzz gap: protocol or concurrency-critical package lacks fuzz or dedicated race tests",
		})
	}

	// 15. Stale or missing performance proof
	if (lane.Criticality == CriticalityCore || lane.Criticality == CriticalityEnabling) &&
		lane.Evidence.Integrated && !lane.Evidence.Benchmarked && !lane.Evidence.Dogfooded {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimStalePerfProof),
			Surface:   string(surface),
			Lane:      lane.Lane,
			Path:      lane.UnitOfWork,
			Severity:  "critical",
			Message:   "stale performance proof: core/enabling unit missing from both benchmark authority and runtime proofs",
		})
	}

	return findings
}

type testAssertionProof int

const (
	proofNone testAssertionProof = iota
	proofUnknown
	proofConfirmed
)

func isTestFunctionName(name string) bool {
	if !strings.HasPrefix(name, "Test") || name == "TestMain" || name == "TestHelperProcess" {
		return false
	}
	if len(name) > 4 && name[4] >= 'a' && name[4] <= 'z' {
		return false
	}
	return true
}

func hasTestingTParam(fn *ast.FuncDecl) bool {
	if fn == nil || fn.Type == nil || fn.Type.Params == nil {
		return false
	}
	for _, field := range fn.Type.Params.List {
		if isTestingTExpr(field.Type) {
			return true
		}
	}
	return false
}

func isTestingTExpr(expr ast.Expr) bool {
	if star, ok := expr.(*ast.StarExpr); ok {
		if sel, ok := star.X.(*ast.SelectorExpr); ok {
			if id, ok := sel.X.(*ast.Ident); ok {
				return id.Name == "testing" && (sel.Sel.Name == "T" || sel.Sel.Name == "B" || sel.Sel.Name == "TB")
			}
		}
	}
	if sel, ok := expr.(*ast.SelectorExpr); ok {
		if id, ok := sel.X.(*ast.Ident); ok {
			return id.Name == "testing" && (sel.Sel.Name == "TB" || sel.Sel.Name == "T" || sel.Sel.Name == "B")
		}
	}
	return false
}

var failureMethods = map[string]bool{
	"Error":   true,
	"Errorf":  true,
	"Fatal":   true,
	"Fatalf":  true,
	"Fail":    true,
	"FailNow": true,
	"Skip":    true,
	"Skipf":   true,
	"SkipNow": true,
	"Helper":  true,
}

var assertionMethods = map[string]bool{
	"Equal":          true,
	"NotEqual":       true,
	"True":           true,
	"False":          true,
	"Nil":            true,
	"NotNil":         true,
	"NoError":        true,
	"Error":          true,
	"Contains":       true,
	"NotContains":    true,
	"Empty":          true,
	"NotEmpty":       true,
	"Len":            true,
	"Zero":           true,
	"NotZero":        true,
	"EqualValues":    true,
	"Same":           true,
	"NotSame":        true,
	"Match":          true,
	"Panics":         true,
	"NotPanics":      true,
	"Regexp":         true,
	"Subset":         true,
	"NotSubset":      true,
	"Greater":        true,
	"Less":           true,
	"GreaterOrEqual": true,
	"LessOrEqual":    true,
	"ElementsMatch":  true,
	"Condition":      true,
	"InDelta":        true,
	"InEpsilon":      true,
	"IsType":         true,
	"JSONEq":         true,
	"YAMLEq":         true,
}

func isAssertionFuncName(name string) bool {
	lower := strings.ToLower(name)
	return strings.HasPrefix(lower, "assert") ||
		strings.HasPrefix(lower, "check") ||
		strings.HasPrefix(lower, "verify") ||
		strings.HasPrefix(lower, "require") ||
		strings.HasPrefix(lower, "expect") ||
		strings.HasPrefix(lower, "must")
}

func getCallName(fun ast.Expr) string {
	switch f := fun.(type) {
	case *ast.Ident:
		return f.Name
	case *ast.SelectorExpr:
		return f.Sel.Name
	}
	return ""
}

func classifyTestAssertions(body *ast.BlockStmt, localHelpers map[string]bool) testAssertionProof {
	if body == nil || len(body.List) == 0 {
		return proofNone
	}
	proof := proofNone
	ast.Inspect(body, func(n ast.Node) bool {
		if proof == proofConfirmed {
			return false
		}
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		switch fun := call.Fun.(type) {
		case *ast.SelectorExpr:
			name := fun.Sel.Name
			if failureMethods[name] || assertionMethods[name] {
				proof = proofConfirmed
				return false
			}
			if localHelpers != nil && localHelpers[name] {
				proof = proofConfirmed
				return false
			}
			if isAssertionFuncName(name) {
				proof = proofConfirmed
				return false
			}
		case *ast.Ident:
			if fun.Name == "panic" {
				proof = proofConfirmed
				return false
			}
			if localHelpers != nil && localHelpers[fun.Name] {
				proof = proofConfirmed
				return false
			}
			if isAssertionFuncName(fun.Name) {
				proof = proofConfirmed
				return false
			}
		}
		for _, arg := range call.Args {
			if id, ok := arg.(*ast.Ident); ok {
				if id.Name == "t" || id.Name == "b" || id.Name == "tb" {
					callName := getCallName(call.Fun)
					if isAssertionFuncName(callName) || (localHelpers != nil && localHelpers[callName]) {
						proof = proofConfirmed
						return false
					}
					if proof == proofNone {
						proof = proofUnknown
					}
				}
			}
		}
		return true
	})
	return proof
}

func hasTestAssertions(body *ast.BlockStmt) bool {
	return classifyTestAssertions(body, nil) == proofConfirmed
}

var stubCommentRegex = regexp.MustCompile(`\b(TODO|FIXME|XXX)\b`)

func isGeneratedASTFile(fileNode *ast.File) bool {
	if fileNode == nil {
		return false
	}
	var b strings.Builder
	for _, cg := range fileNode.Comments {
		b.WriteString(cg.Text())
		b.WriteByte('\n')
		for _, c := range cg.List {
			b.WriteString(c.Text)
			b.WriteByte('\n')
		}
	}
	all := b.String()
	return strings.Contains(all, "DO NOT EDIT") && (strings.Contains(all, "Code generated") || strings.Contains(all, "GENERATED"))
}

type scopeInfo struct {
	symbol    string
	startLine int
	endLine   int
}

func formatReceiver(recv *ast.FieldList) string {
	if recv == nil || len(recv.List) == 0 {
		return ""
	}
	switch t := recv.List[0].Type.(type) {
	case *ast.StarExpr:
		if id, ok := t.X.(*ast.Ident); ok {
			return "(*" + id.Name + ")"
		}
	case *ast.Ident:
		return t.Name
	case *ast.IndexExpr: // generic receiver T[P]
		if id, ok := t.X.(*ast.Ident); ok {
			return id.Name
		}
	case *ast.IndexListExpr: // generic receiver T[P, Q]
		if id, ok := t.X.(*ast.Ident); ok {
			return id.Name
		}
	}
	return ""
}

func typeDeclSymbol(decl *ast.GenDecl) string {
	if decl == nil || decl.Tok != token.TYPE {
		return ""
	}
	for _, spec := range decl.Specs {
		if ts, ok := spec.(*ast.TypeSpec); ok && ts.Name != nil {
			return ts.Name.Name
		}
	}
	return ""
}

func declRange(fset *token.FileSet, decl ast.Decl) (token.Pos, token.Pos, int, int) {
	startPos := decl.Pos()
	endPos := decl.End()
	switch d := decl.(type) {
	case *ast.GenDecl:
		if d.Doc != nil && d.Doc.Pos() < startPos {
			startPos = d.Doc.Pos()
		}
	case *ast.FuncDecl:
		if d.Doc != nil && d.Doc.Pos() < startPos {
			startPos = d.Doc.Pos()
		}
	}
	startLine := fset.Position(startPos).Line
	endLine := fset.Position(endPos).Line
	return startPos, endPos, startLine, endLine
}

func findEnclosingScope(fset *token.FileSet, fileNode *ast.File, pos token.Pos) scopeInfo {
	p := fset.Position(pos)
	line := p.Line

	for _, decl := range fileNode.Decls {
		startPos, endPos, dStart, dEnd := declRange(fset, decl)
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if (pos >= startPos && pos <= endPos) || (line >= dStart && line <= dEnd) {
				symbol := d.Name.Name
				if d.Recv != nil {
					if r := formatReceiver(d.Recv); r != "" {
						symbol = r + "." + d.Name.Name
					}
				}
				return scopeInfo{
					symbol:    symbol,
					startLine: dStart,
					endLine:   dEnd,
				}
			}
		case *ast.GenDecl:
			if (pos >= startPos && pos <= endPos) || (line >= dStart && line <= dEnd) {
				symbol := ""
				if d.Tok == token.TYPE {
					symbol = typeDeclSymbol(d)
				}
				if symbol == "" && fileNode.Name != nil {
					symbol = fileNode.Name.Name
				}
				return scopeInfo{
					symbol:    symbol,
					startLine: dStart,
					endLine:   dEnd,
				}
			}
		}
	}

	pkgName := "package"
	if fileNode.Name != nil && fileNode.Name.Name != "" {
		pkgName = fileNode.Name.Name
	}
	return scopeInfo{
		symbol:    pkgName,
		startLine: line,
		endLine:   line,
	}
}

func cleanComment(text string) string {
	s := strings.TrimSpace(text)
	if strings.HasPrefix(s, "//") {
		s = strings.TrimPrefix(s, "//")
		return strings.TrimSpace(s)
	}
	if strings.HasPrefix(s, "/*") {
		s = strings.TrimPrefix(s, "/*")
		s = strings.TrimSuffix(s, "*/")
		lines := strings.Split(s, "\n")
		for _, l := range lines {
			l = strings.TrimSpace(l)
			l = strings.TrimPrefix(l, "*")
			l = strings.TrimSpace(l)
			if stubCommentRegex.MatchString(l) {
				return l
			}
		}
		return strings.TrimSpace(lines[0])
	}
	return strings.TrimSpace(s)
}

func isStubPanic(call *ast.CallExpr) (bool, string) {
	ident, ok := call.Fun.(*ast.Ident)
	if !ok || ident.Name != "panic" || len(call.Args) == 0 {
		return false, ""
	}
	found := false
	detail := ""
	for _, arg := range call.Args {
		ast.Inspect(arg, func(n ast.Node) bool {
			if found {
				return false
			}
			lit, ok := n.(*ast.BasicLit)
			if ok && lit.Kind == token.STRING {
				val := strings.Trim(lit.Value, "`\"")
				lower := strings.ToLower(val)
				if strings.Contains(lower, "not implemented") ||
					strings.Contains(lower, "unimplemented") ||
					strings.Contains(lower, "todo") {
					found = true
					detail = val
					return false
				}
			}
			return true
		})
		if found {
			break
		}
	}
	return found, detail
}

type stubCandidate struct {
	pos    token.Pos
	line   int
	symbol string
	span   string
	detail string
}

func inspectStubDebt(fset *token.FileSet, fileNode *ast.File, relPath string, lane *DebtLane, surface SurfaceClass) []FindingProvenance {
	if isGeneratedASTFile(fileNode) {
		return nil
	}

	var candidates []stubCandidate

	// 1. AST comments
	for _, cg := range fileNode.Comments {
		for _, c := range cg.List {
			if stubCommentRegex.MatchString(c.Text) {
				scope := findEnclosingScope(fset, fileNode, c.Pos())
				line := fset.Position(c.Pos()).Line
				detail := cleanComment(c.Text)
				candidates = append(candidates, stubCandidate{
					pos:    c.Pos(),
					line:   line,
					symbol: scope.symbol,
					span:   fmt.Sprintf("%d-%d", scope.startLine, scope.endLine),
					detail: detail,
				})
			}
		}
	}

	// 2. AST panic calls
	ast.Inspect(fileNode, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if isPanic, msg := isStubPanic(call); isPanic {
			scope := findEnclosingScope(fset, fileNode, call.Pos())
			line := fset.Position(call.Pos()).Line
			detail := fmt.Sprintf("panic(%q)", msg)
			candidates = append(candidates, stubCandidate{
				pos:    call.Pos(),
				line:   line,
				symbol: scope.symbol,
				span:   fmt.Sprintf("%d-%d", scope.startLine, scope.endLine),
				detail: detail,
			})
		}
		return true
	})

	if len(candidates) == 0 {
		return nil
	}

	// Sort deterministically in source order
	sort.SliceStable(candidates, func(i, j int) bool {
		return candidates[i].pos < candidates[j].pos
	})

	// Deduplicate by detector (stub_debt), symbol, and source span
	laneName := ""
	if lane != nil {
		laneName = lane.Lane
	}

	var findings []FindingProvenance
	seen := make(map[string]bool)

	for _, c := range candidates {
		key := fmt.Sprintf("stub_debt:%s:%s", c.symbol, c.span)
		if seen[key] {
			continue
		}
		seen[key] = true
		findings = append(findings, FindingProvenance{
			Dimension: string(DimStubDebt),
			Surface:   string(surface),
			Lane:      laneName,
			Path:      relPath,
			Severity:  "warning",
			Message:   fmt.Sprintf("stub debt in %s (%s:%d): %s", c.symbol, relPath, c.line, c.detail),
		})
	}

	return findings
}

func countStubMarkers(filePath string) int {
	f, err := os.Open(filePath)
	if err != nil {
		return 0
	}
	defer f.Close()

	count := 0
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		lower := strings.ToLower(line)
		if strings.Contains(line, "TODO") || strings.Contains(line, "FIXME") || strings.Contains(line, "XXX") {
			count++
		} else if strings.Contains(lower, `panic("not implemented")`) ||
			strings.Contains(lower, `panic("unimplemented")`) ||
			strings.Contains(lower, `panic("todo")`) {
			count++
		}
	}
	return count
}

func inspectSkillSurface(lane *DebtLane, dir string) []FindingProvenance {
	var findings []FindingProvenance
	skillPath := filepath.Join(dir, "SKILL.md")
	data, err := os.ReadFile(skillPath)
	if err != nil {
		return findings
	}
	content := string(data)

	// Frontmatter check
	if !strings.HasPrefix(content, "---\n") {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimWiringStatus),
			Surface:   string(SurfaceSkills),
			Lane:      lane.Lane,
			Path:      filepath.Join(lane.UnitOfWork, "SKILL.md"),
			Severity:  "critical",
			Message:   "missing YAML frontmatter in SKILL.md",
		})
	}

	// Verification check
	if !strings.Contains(content, "## Verification") && !strings.Contains(content, "## Witness") &&
		!strings.Contains(content, "go test") && !strings.Contains(content, "python tools/") {
		findings = append(findings, FindingProvenance{
			Dimension: string(DimProofStatus),
			Surface:   string(SurfaceSkills),
			Lane:      lane.Lane,
			Path:      filepath.Join(lane.UnitOfWork, "SKILL.md"),
			Severity:  "warning",
			Message:   "missing verification or witness commands in SKILL.md",
		})
	}

	return findings
}

func inspectWorkflowSurface(lane *DebtLane, dir string) []FindingProvenance {
	var findings []FindingProvenance
	entries, err := os.ReadDir(dir)
	if err != nil {
		return findings
	}
	for _, e := range entries {
		if e.IsDir() || (!strings.HasSuffix(e.Name(), ".yml") && !strings.HasSuffix(e.Name(), ".yaml")) {
			continue
		}
		path := filepath.Join(dir, e.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		content := string(data)
		if !strings.Contains(content, "timeout-minutes:") {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimBlastRadius),
				Surface:   string(SurfaceWorkflows),
				Lane:      lane.Lane,
				Path:      filepath.Join(lane.UnitOfWork, e.Name()),
				Severity:  "warning",
				Message:   "unbounded CI job timeout: workflow job missing timeout-minutes",
			})
		}
	}
	return findings
}

func inspectDocSurface(lane *DebtLane, dir string) []FindingProvenance {
	var findings []FindingProvenance
	entries, err := os.ReadDir(dir)
	if err != nil {
		return findings
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") {
			continue
		}
		path := filepath.Join(dir, e.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		content := strings.TrimSpace(string(data))
		lines := strings.Split(content, "\n")
		if len(lines) < 10 && (strings.Contains(content, "TODO") || strings.Contains(content, "stub")) {
			findings = append(findings, FindingProvenance{
				Dimension: string(DimStubDebt),
				Surface:   string(SurfaceDocs),
				Lane:      lane.Lane,
				Path:      filepath.Join(lane.UnitOfWork, e.Name()),
				Severity:  "warning",
				Message:   "stub doc: documentation file is an unfinished placeholder stub",
			})
		}
	}
	return findings
}
