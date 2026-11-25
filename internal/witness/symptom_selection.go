package witness

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/build"
	"go/doc"
	"go/format"
	"go/parser"
	"go/token"
	"io"
	"path"
	"regexp"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

type symptomSelection struct {
	Package string
	Tests   []string
}

// discoverSymptomSelections derives the executable witness from candidate source.
// Each package owns its selector set; selectors are never unioned across packages.
// An explicit selector is only a disambiguation hint and still has to match a
// top-level Test/Example in a changed candidate file.
func discoverSymptomSelections(ctx context.Context, git Runner, repoDir, commit, parent string, tests, explicit []string) ([]symptomSelection, string) {
	if git == nil {
		git = gitRunner
	}
	type packageTests map[string]string
	candidateByPackage := map[string]packageTests{}
	changedByPackage := map[string]map[string]bool{}
	changedPackages := map[string]bool{}
	var explicitRegex []*regexp.Regexp
	for _, selector := range explicit {
		re, err := regexp.Compile(selector)
		if err != nil {
			return nil, "invalid symptom test selector"
		}
		explicitRegex = append(explicitRegex, re)
	}
	explicitMatched := make([]bool, len(explicitRegex))

	for _, rel := range tests {
		rel = strings.ReplaceAll(strings.TrimSpace(rel), "\\", "/")
		if !strings.HasSuffix(rel, "_test.go") {
			continue
		}
		changedPackages[goTestPackage(rel)] = true
		testSource, code, err := git(ctx, repoDir, "show", commit+":"+rel)
		if err != nil || code != 0 {
			continue // deleted tests cannot witness a fix
		}
		candidateFuncs, err := topLevelGoTests(testSource)
		if err != nil {
			return nil, "candidate changed test source did not parse"
		}
		pkg := goTestPackage(rel)
		if candidateByPackage[pkg] == nil {
			candidateByPackage[pkg] = packageTests{}
		}
		for name, fingerprint := range candidateFuncs {
			candidateByPackage[pkg][name] = fingerprint
		}

		parentFuncs := map[string]string{}
		if parentSource, parentCode, parentErr := git(ctx, repoDir, "show", parent+":"+rel); parentErr == nil && parentCode == 0 {
			parentFuncs, err = topLevelGoTests(parentSource)
			if err != nil {
				return nil, "parent changed test source did not parse"
			}
		}
		for name, fingerprint := range candidateFuncs {
			selected := len(explicitRegex) == 0 && parentFuncs[name] != fingerprint
			for i, re := range explicitRegex {
				if re.MatchString(name) {
					explicitMatched[i] = true
					selected = true
				}
			}
			if selected {
				if changedByPackage[pkg] == nil {
					changedByPackage[pkg] = map[string]bool{}
				}
				changedByPackage[pkg][name] = true
			}
		}
	}
	if len(explicitRegex) > 0 {
		// Exact selectors found in changed tests can use one Git search to find
		// every matching file in the changed packages. Keep the full scan for
		// broad selectors and for runners that cannot perform the search.
		paths := []string(nil)
		if exactSelectorsMatchedChangedTests(explicit, explicitMatched) {
			paths = grepExplicitTestPaths(ctx, git, repoDir, commit, explicit, changedPackages)
		}
		if paths == nil {
			tree, code, err := git(ctx, repoDir, "ls-tree", "-r", "--name-only", commit)
			if err != nil || code != 0 {
				return nil, "could not inspect candidate tests for explicit selector"
			}
			paths = strings.Split(tree, "\n")
		}
		for _, rel := range paths {
			rel = strings.ReplaceAll(strings.TrimSpace(rel), "\\", "/")
			pkg := goTestPackage(rel)
			if !changedPackages[pkg] || !strings.HasSuffix(rel, "_test.go") {
				continue
			}
			source, showCode, showErr := git(ctx, repoDir, "show", commit+":"+rel)
			if showErr != nil || showCode != 0 {
				return nil, "could not inspect candidate test for explicit selector"
			}
			funcs, parseErr := topLevelGoTests(source)
			if parseErr != nil {
				return nil, "candidate package test source did not parse"
			}
			for name := range funcs {
				for i, re := range explicitRegex {
					if !re.MatchString(name) {
						continue
					}
					explicitMatched[i] = true
					if changedByPackage[pkg] == nil {
						changedByPackage[pkg] = map[string]bool{}
					}
					changedByPackage[pkg][name] = true
				}
			}
		}
	}

	for _, matched := range explicitMatched {
		if !matched {
			return nil, "explicit symptom selector matched no changed top-level Test/Example"
		}
	}
	if len(changedByPackage) == 0 {
		if len(candidateByPackage) == 0 {
			return nil, "changed Go tests contain no top-level Test/Example"
		}
		return nil, "automatic symptom selection ambiguous: no added or modified top-level Test/Example"
	}

	packages := make([]string, 0, len(changedByPackage))
	for pkg := range changedByPackage {
		packages = append(packages, pkg)
	}
	sort.Strings(packages)
	selections := make([]symptomSelection, 0, len(packages))
	for _, pkg := range packages {
		names := make([]string, 0, len(changedByPackage[pkg]))
		for name := range changedByPackage[pkg] {
			names = append(names, name)
		}
		sort.Strings(names)
		selections = append(selections, symptomSelection{Package: pkg, Tests: names})
	}
	return selections, ""
}

func exactSelectorsMatchedChangedTests(selectors []string, matched []bool) bool {
	if len(selectors) == 0 || len(selectors) != len(matched) {
		return false
	}
	for i, selector := range selectors {
		if !matched[i] || len(selector) < 3 || selector[0] != '^' || selector[len(selector)-1] != '$' {
			return false
		}
		literal := selector[1 : len(selector)-1]
		if regexp.QuoteMeta(literal) != literal {
			return false
		}
	}
	return true
}

func grepExplicitTestPaths(ctx context.Context, git Runner, repoDir, commit string, selectors []string, changedPackages map[string]bool) []string {
	args := []string{"grep", "-l", "-F"}
	for _, selector := range selectors {
		args = append(args, "-e", selector[1:len(selector)-1])
	}
	args = append(args, commit, "--")
	packages := make([]string, 0, len(changedPackages))
	for pkg := range changedPackages {
		packages = append(packages, pkg)
	}
	sort.Strings(packages)
	args = append(args, packages...)
	out, code, err := git(ctx, repoDir, args...)
	if err != nil || code != 0 {
		return nil
	}
	var paths []string
	for _, line := range strings.Split(out, "\n") {
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, commit+":") {
			return nil
		}
		rel := strings.TrimPrefix(line, commit+":")
		if strings.HasSuffix(rel, "_test.go") && changedPackages[goTestPackage(rel)] {
			paths = append(paths, rel)
		}
	}
	return paths
}

func goTestPackage(rel string) string {
	dir := path.Dir(strings.ReplaceAll(rel, "\\", "/"))
	if dir == "." || dir == "" {
		return "./."
	}
	return "./" + strings.TrimPrefix(dir, "./")
}

func topLevelGoTests(source string) (map[string]string, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "candidate_test.go", source, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	// `go test` compiles an Example without an output comment but never runs it.
	runnableExamples := map[string]bool{}
	for _, ex := range doc.Examples(file) {
		if ex.Output != "" || ex.EmptyOutput {
			runnableExamples["Example"+ex.Name] = true
		}
	}
	out := map[string]string{}
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Recv != nil || !isGoTestEntryName(fn.Name.Name) {
			continue
		}
		if strings.HasPrefix(fn.Name.Name, "Example") {
			if !runnableExamples[fn.Name.Name] {
				continue
			}
		} else if !isRunnableGoTestFunc(fn) {
			continue
		}
		var rendered bytes.Buffer
		if err := format.Node(&rendered, fset, fn); err != nil {
			return nil, err
		}
		out[fn.Name.Name] = rendered.String()
	}
	return out, nil
}

// isRunnableGoTestFunc reports whether `go test` emits a run event for a Test*
// declaration: TestMain and any non-func(*testing.T) shape never run as a test.
func isRunnableGoTestFunc(fn *ast.FuncDecl) bool {
	if fn.Name.Name == "TestMain" || fn.Type.TypeParams != nil || fn.Type.Results.NumFields() != 0 {
		return false
	}
	params := fn.Type.Params
	if params == nil || len(params.List) != 1 || len(params.List[0].Names) > 1 {
		return false
	}
	star, ok := params.List[0].Type.(*ast.StarExpr)
	if !ok {
		return false
	}
	switch x := star.X.(type) {
	case *ast.SelectorExpr:
		return x.Sel.Name == "T"
	case *ast.Ident:
		// A dot-imported testing package spells the parameter as *T.
		return x.Name == "T"
	}
	return false
}

// dropHostExcludedSelections removes selected names declared only in changed test
// files whose build constraints exclude them on this host: `go test -run` exits 0
// without a run event for them, which would read as a selector that matched nothing.
// A file whose constraints cannot be evaluated is kept.
func dropHostExcludedSelections(ctx context.Context, git Runner, repoDir, commit string, tests, tags []string, selections []symptomSelection) ([]symptomSelection, int) {
	if git == nil {
		git = gitRunner
	}
	included := map[string]map[string]bool{}
	excluded := map[string]map[string]bool{}
	excludedFiles := 0
	for _, rel := range tests {
		rel = strings.ReplaceAll(strings.TrimSpace(rel), "\\", "/")
		if !strings.HasSuffix(rel, "_test.go") {
			continue
		}
		source, code, err := git(ctx, repoDir, "show", commit+":"+rel)
		if err != nil || code != 0 {
			continue
		}
		funcs, err := topLevelGoTests(source)
		if err != nil {
			continue
		}
		into := included
		if !hostMatchesTestFile(rel, source, tags) {
			into = excluded
			excludedFiles++
		}
		pkg := goTestPackage(rel)
		if into[pkg] == nil {
			into[pkg] = map[string]bool{}
		}
		for name := range funcs {
			into[pkg][name] = true
		}
	}
	if excludedFiles == 0 {
		return selections, 0
	}
	kept := make([]symptomSelection, 0, len(selections))
	for _, selection := range selections {
		var names []string
		for _, name := range selection.Tests {
			if excluded[selection.Package][name] && !included[selection.Package][name] {
				continue
			}
			names = append(names, name)
		}
		if len(names) > 0 {
			kept = append(kept, symptomSelection{Package: selection.Package, Tests: names})
		}
	}
	return kept, excludedFiles
}

// hostMatchesTestFile evaluates the filename suffix and //go:build line of a
// candidate blob against the host GOOS/GOARCH plus the feature tags the run will
// set. Dropping a runnable test weakens the proof, so a file is excluded only when
// no cgo setting and no toolchain release could include it: the go command's
// effective cgo and release may differ from this binary's build.Default.
func hostMatchesTestFile(rel, source string, tags []string) bool {
	everyRelease := make([]string, 0, 99)
	for minor := 1; minor <= 99; minor++ {
		everyRelease = append(everyRelease, fmt.Sprintf("go1.%d", minor))
	}
	for _, cgo := range []bool{true, false} {
		for _, release := range [][]string{build.Default.ReleaseTags, everyRelease} {
			bctx := build.Default
			bctx.CgoEnabled = cgo
			bctx.ReleaseTags = release
			bctx.BuildTags = append([]string(nil), tags...)
			bctx.JoinPath = path.Join
			bctx.OpenFile = func(string) (io.ReadCloser, error) {
				return io.NopCloser(strings.NewReader(source)), nil
			}
			match, err := bctx.MatchFile(path.Dir(rel), path.Base(rel))
			if err != nil || match {
				return true
			}
		}
	}
	return false
}

func isGoTestEntryName(name string) bool {
	for _, prefix := range []string{"Test", "Example"} {
		if !strings.HasPrefix(name, prefix) {
			continue
		}
		rest := strings.TrimPrefix(name, prefix)
		if prefix == "Example" && rest == "" {
			return true
		}
		if rest == "" {
			return false
		}
		first, _ := utf8.DecodeRuneInString(rest)
		return !unicode.IsLower(first)
	}
	return false
}

func exactTestSelectors(names []string) []string {
	selectors := make([]string, 0, len(names))
	for _, name := range names {
		selectors = append(selectors, "^"+regexp.QuoteMeta(name)+"$")
	}
	return selectors
}

// testPackages maps the changed `_test.go` paths to their parent package directories (deduped),
// the unit `go test` operates on.
func testPackages(tests []string) []string {
	seen := map[string]bool{}
	var pkgs []string
	for _, t := range tests {
		t = strings.ReplaceAll(strings.TrimSpace(t), "\\", "/")
		if !strings.HasSuffix(t, "_test.go") {
			continue
		}
		dir := path.Dir(t)
		if dir == "." || dir == "" {
			dir = "."
		}
		if !seen[dir] {
			seen[dir] = true
			pkgs = append(pkgs, "./"+strings.TrimPrefix(dir, "./"))
		}
	}
	return pkgs
}

// pythonTestFiles filters tests down to Python test files.
func pythonTestFiles(tests []string) []string {
	var pyTests []string
	for _, t := range tests {
		t = strings.ReplaceAll(strings.TrimSpace(t), "\\", "/")
		if isPythonTestFile(t) {
			pyTests = append(pyTests, t)
		}
	}
	return pyTests
}

// allTestsPass reports whether every changed-test package passes in dir, and — when the Go
// side fails — whether the failure is a committed go.work that cannot load in this scratch
// topology (#13213): no test ran at all, so the caller must ABSTAIN instead of REFUTING.
func allTestsPass(ctx context.Context, exec CommandRunner, dir string, pkgs, pyTests, tags []string) (passed bool, workspaceBroken bool) {
	if len(pkgs) > 0 {
		ok, out := goTestOutcome(ctx, exec, dir, pkgs, tags)
		if !ok {
			if isGoWorkspaceLoadFailure(out) {
				return false, true
			}
			return false, false
		}
	}
	if len(pyTests) > 0 && !pythonTestsPass(ctx, exec, dir, pyTests) {
		return false, false
	}
	return true, false
}
