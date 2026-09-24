package witness

// The symptom rung (#1326) — the test analog of dos commit-audit's diff-witness.
//
// THE GAP IT CLOSES. A bug-fix commit can pass `dos commit-audit` (verdict OK,
// diff-witnessed) while the bug is STILL FULLY LIVE. commit-audit is honest about its
// rung: it grades whether the diff did the KIND of thing claimed, NOT whether the symptom
// is gone. The worked example is the guard "stuck on login" fix (`c9cd25b2`): diff-witnessed
// OK, yet the gate used `os.ModeCharDevice`, which on Windows reports `NUL`/`</dev/null` AS a
// char device — so the headless case was treated as interactive and the gate never fired. The
// unit test passed because it ENCODED THE BUG as expected behavior. The real witness
// (`28dd3b33`) re-execs the binary headless from `os.DevNull` and asserts exit-2-within-deadline:
// it FAILS on the pre-fix code and PASSES on the fix.
//
// THE DISTINGUISHING PROPERTY. A real fix-witness REPRODUCES THE ACTUAL FAILURE CONDITION —
// a test that FAILS against the parent commit's source and PASSES at the fix. A test that
// passes against the parent is tautological: it constrains nothing about the bug.
//
// THE TWO RUNGS, ONE FAIL-CLOSED CONTRACT.
//
//	STRUCTURAL (always, flag-independent): did the fix commit add or modify a `_test.go` or Python test file?
//	  No test touched => REFUTED — a fix with no symptom witness is exactly the gap. This is
//	  the cheap, deterministic half; it needs no second worktree and never abstains on cost.
//
//	EXECUTION (red-then-green, gated behind FAK_WITNESS_SYMPTOM): overlay the ref's version of
//	  each changed test onto a PARENT scratch worktree and run it — it must FAIL (red: the test
//	  reproduces the bug against the old source) — then run it at the ref — it must PASS (green).
//	  Both hold => CONFIRMED. Passes-at-parent => REFUTED (tautological). Parent compilation or
//	  build failure => ABSTAIN (unproven: the overlaid test failed to compile against parent source,
//	  e.g. referencing an API introduced in the fix — never a false CONFIRM; #12058). Default (flag unset) =>
//	  ABSTAIN after the structural check: running an arbitrary test against an old tree is heavy,
//	  so like the RSL rung the cost is opt-in and the kernel's fail-closed default turns abstain
//	  into a deny rather than a false CONFIRM.
//
// This is the mirror of the `notests:` rung in this package: notests REFUTES a ship-commit that
// edited the very tests it must pass (reward-hack); symptom CONFIRMS a fix-commit that added a
// test which genuinely constrains the bug. Same git evidence, opposite polarity.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// SymptomFlagEnv opts the EXECUTION rung in. The structural rung runs regardless.
const SymptomFlagEnv = "FAK_WITNESS_SYMPTOM"

// SymptomExecEnabled reports whether the red-then-green execution rung is opted in. Default off.
func SymptomExecEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(SymptomFlagEnv))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// WithSymptomTags supplies EXTRA build tags for the execution rung explicitly (#13243), for the
// case where the tag derivation from the changed test files' //go:build constraints is unsafe or
// incomplete. The tags are normalized (trimmed, empties dropped, de-duplicated, sorted) and
// stored on the Resolver; resolveSymptomExec unions them with the derived set — explicit tags
// never remove a derived tag. The empty default keeps the untagged behavior byte-identical.
func (r *Resolver) WithSymptomTags(tags []string) *Resolver {
	r.symptomTags = normalizeTags(tags)
	return r
}

// WithSymptomTests narrows the Go execution rung to explicit test-name regexes.
// Changed tests are still overlaid onto the parent before the selected tests run.
func (r *Resolver) WithSymptomTests(tests []string) *Resolver {
	r.symptomTests = normalizeTags(tests)
	return r
}

// normalizeTags trims, drops empties, de-duplicates, and sorts a tag list so the Resolver's
// stored explicit set is canonical (and mergeTags is order-stable).
func normalizeTags(tags []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, t := range tags {
		t = strings.TrimSpace(t)
		if t == "" || seen[t] {
			continue
		}
		seen[t] = true
		out = append(out, t)
	}
	sort.Strings(out)
	return out
}

// mergeTags returns the UNION of a and b: trimmed, empties dropped, de-duplicated, sorted.
// Either side may be nil; both nil yields nil so the caller's untagged path stays byte-identical.
func mergeTags(a, b []string) []string {
	if len(a) == 0 && len(b) == 0 {
		return nil
	}
	return normalizeTags(append(append([]string{}, a...), b...))
}

// ResolveSymptom adjudicates a `symptom:<ref>` claim: does the fix at <ref> carry a witness that
// the symptom is gone? See the package doc for the two-rung contract.
// When mandatoryExec is true, it forces the red-then-green execution check even when
// FAK_WITNESS_SYMPTOM is not set in the environment.
func (r *Resolver) ResolveSymptom(ctx context.Context, ref string, mandatoryExec bool) abi.WitnessOutcome {
	outcome, _ := r.ResolveSymptomWithDetail(ctx, ref, mandatoryExec)
	return outcome
}

// ResolveSymptomWithDetail returns the witness outcome plus a bounded diagnostic.
// Diagnostics contain no test output or source bytes, so land receipts stay compact.
func (r *Resolver) ResolveSymptomWithDetail(ctx context.Context, ref string, mandatoryExec bool) (abi.WitnessOutcome, string) {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return abi.WitnessAbstain, "missing symptom ref"
	}
	git := r.run
	if git == nil {
		git = gitRunner
	}

	// STRUCTURAL rung: which _test.go files did the commit add or modify?
	out, code, err := git(ctx, r.dir, "show", "--name-only", "--format=", ref)
	if err != nil || code != 0 {
		return abi.WitnessAbstain, "could not inspect changed tests" // never a false CONFIRM
	}
	tests := changedTestFiles(out)
	if len(tests) == 0 {
		return abi.WitnessRefuted, "no changed test file"
	}

	// EXECUTION rung is opt-in via env or mandatoryExec; without it the structural pass is as far as we honestly go.
	if !mandatoryExec && !SymptomExecEnabled() {
		return abi.WitnessAbstain, "symptom execution disabled"
	}
	return r.resolveSymptomExec(ctx, ref, tests)
}

func (r *Resolver) resolveSymptom(ctx context.Context, ref string) abi.WitnessOutcome {
	return r.ResolveSymptom(ctx, ref, false)
}

// changedTestFiles extracts the repo-relative test paths from `git show --name-only`
// output, normalizing separators and skipping blanks. Both Go test files (`_test.go`)
// and Python test files (e.g. under `tools/` ending with `_test.py` or starting with `test_`
// and ending with `.py`) are recognized. testdata/ fixtures are excluded — a
// fixture named *_test.go or *_test.py is not a gating test.
func changedTestFiles(nameOnly string) []string {
	var tests []string
	for _, line := range strings.Split(nameOnly, "\n") {
		p := strings.ReplaceAll(strings.TrimSpace(line), "\\", "/")
		if p == "" || !isTestFile(p) {
			continue
		}
		if isUnderTestdata(p) {
			continue
		}
		tests = append(tests, p)
	}
	return tests
}

func isTestFile(p string) bool {
	p = strings.ReplaceAll(strings.TrimSpace(p), "\\", "/")
	if strings.HasSuffix(p, "_test.go") {
		return true
	}
	return isPythonTestFile(p)
}

func isPythonTestFile(p string) bool {
	p = strings.ReplaceAll(strings.TrimSpace(p), "\\", "/")
	base := path.Base(p)
	if strings.HasSuffix(base, "_test.py") && len(base) > len("_test.py") {
		return true
	}
	if strings.HasPrefix(base, "test_") && strings.HasSuffix(base, ".py") && len(base) > len("test_.py") {
		return true
	}
	return false
}

func isUnderTestdata(p string) bool {
	p = strings.ReplaceAll(strings.TrimSpace(p), "\\", "/")
	for _, seg := range strings.Split(p, "/") {
		if seg == "testdata" {
			return true
		}
	}
	return false
}

// resolveSymptomExec runs the red-then-green check: the changed test(s), taken at <ref>, must
// FAIL against the parent's source and PASS at <ref>. All git/exec goes through the injected
// runners so this is exercised without a real repo (NewWithRunners).
func (r *Resolver) resolveSymptomExec(ctx context.Context, ref string, tests []string) (abi.WitnessOutcome, string) {
	exec := r.execRun
	if exec == nil {
		exec = commandRunner
	}
	v := NewExecutionVerifierWithRunners(r.run, exec, r.dir)

	commit, ok := v.revParse(ctx, ref+"^{commit}")
	if !ok {
		return abi.WitnessAbstain, "candidate commit not found"
	}
	parent, ok := v.revParse(ctx, commit+"^")
	if !ok {
		return abi.WitnessAbstain, "parent commit not found"
	}

	pkgs := testPackages(tests)
	pyTests := pythonTestFiles(tests)
	if len(pkgs) == 0 && len(pyTests) == 0 {
		return abi.WitnessAbstain, "changed tests have no executable package"
	}
	if len(r.symptomTests) > 0 {
		if len(pkgs) == 0 {
			return abi.WitnessAbstain, "symptom test selector requires a changed Go test"
		}
		for _, selector := range r.symptomTests {
			if _, err := regexp.Compile(selector); err != nil {
				return abi.WitnessAbstain, "invalid symptom test selector"
			}
		}
	}
	selections, selectionDetail := discoverSymptomSelections(ctx, r.run, r.dir, commit, parent, tests, r.symptomTests)
	if len(pkgs) > 0 && len(selections) == 0 {
		return abi.WitnessAbstain, selectionDetail
	}

	// The build constraints the changed test files declare (#13243). A device-tagged test
	// (`//go:build vulkan`, `metal`, `cuda`, …) is excluded from a bare `go test`, so without
	// this the package passes trivially at BOTH refs and a genuine witness is falsely refuted.
	// An explicit caller hint (WithSymptomTags) is UNIONED with the derived truth — the hint
	// can add a tag the derivation missed but never remove one a test file declares.
	tags := mergeTags(resolveGoBuildTags(ctx, r.run, r.dir, commit, tests), r.symptomTags)

	// GREEN at the fix: the changed test must pass at <ref> as committed.
	commitDir, cleanupCommit, err := v.scratchWorktree(ctx, commit)
	if err != nil {
		return abi.WitnessAbstain, "candidate scratch worktree failed"
	}
	defer cleanupCommit()
	if len(selections) > 0 {
		for _, selection := range selections {
			result := runSelectedGoTests(ctx, exec, commitDir, []string{selection.Package}, tags, exactTestSelectors(selection.Tests))
			switch {
			case result.timedOut:
				return abi.WitnessAbstain, "candidate selected symptom test timed out"
			case result.runErr != nil:
				return abi.WitnessAbstain, "candidate selected test execution failed"
			case result.buildFailure:
				return abi.WitnessRefuted, "candidate selected symptom test failed"
			case !result.matched:
				return abi.WitnessRefuted, "candidate symptom selector matched no executed test"
			case !result.passed:
				return abi.WitnessRefuted, "candidate selected symptom test failed"
			}
		}
		if len(pyTests) > 0 {
			passed, buildErr := runPythonTests(ctx, exec, commitDir, pyTests)
			switch {
			case buildErr:
				return abi.WitnessRefuted, "candidate changed Python symptom test failed to execute"
			case !passed:
				return abi.WitnessRefuted, "candidate changed Python symptom test failed"
			}
		}
	} else if !allTestsPass(ctx, exec, commitDir, pkgs, pyTests, tags) {
		// The committed test does not even pass at the fix — not a usable witness; don't CONFIRM.
		return abi.WitnessRefuted, "candidate changed-test package failed"
	}

	// RED at the parent: overlay each changed test file (its <ref> content) onto the parent
	// worktree, then run it. It must FAIL — the test reproduces the bug against the old source.
	parentDir, cleanupParent, err := v.scratchWorktree(ctx, parent)
	if err != nil {
		return abi.WitnessAbstain, "parent scratch worktree failed"
	}
	defer cleanupParent()
	if !overlayTestsAtRef(ctx, r.run, r.dir, commit, parentDir, tests) {
		return abi.WitnessAbstain, "could not overlay changed tests onto parent"
	}
	if len(selections) > 0 {
		parentRed := false
		for _, selection := range selections {
			result := runSelectedGoTests(ctx, exec, parentDir, []string{selection.Package}, tags, exactTestSelectors(selection.Tests))
			switch {
			case result.timedOut:
				return abi.WitnessAbstain, "parent selected symptom test timed out"
			case result.runErr != nil:
				return abi.WitnessAbstain, "parent selected test execution failed"
			case result.buildFailure:
				return abi.WitnessAbstain, "parent selected symptom test did not build"
			case !result.matched:
				return abi.WitnessRefuted, "parent symptom selector matched no executed test"
			case !result.passed && !result.selectedFailed:
				return abi.WitnessAbstain, "parent selected test outcome obscured by unrelated failure"
			case result.selectedFailed:
				parentRed = true
			}
		}
		if len(pyTests) > 0 {
			passed, buildErr := runPythonTests(ctx, exec, parentDir, pyTests)
			switch {
			case buildErr:
				return abi.WitnessAbstain, "parent changed Python symptom test did not execute"
			case !passed:
				parentRed = true
			}
		}
		if !parentRed {
			return abi.WitnessRefuted, "selected symptom test passed at parent"
		}
		return abi.WitnessConfirmed, "selected symptom test failed at parent and passed at candidate"
	}
	passed, buildErr := runParentTests(ctx, exec, parentDir, pkgs, pyTests, tags)
	if buildErr {
		// Parent failed to compile/build (e.g. test references an API introduced by the fix) —
		// this is unproven, never a false CONFIRM of behavioral reproduction (#12058).
		return abi.WitnessAbstain, "parent changed-test package did not build"
	}
	if passed {
		// The test passes against the OLD source too: it constrains nothing about the bug.
		return abi.WitnessRefuted, "changed tests passed at parent"
	}
	return abi.WitnessConfirmed, "changed tests failed at parent and passed at candidate"
}

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
		tree, code, err := git(ctx, repoDir, "ls-tree", "-r", "--name-only", commit)
		if err != nil || code != 0 {
			return nil, "could not inspect candidate tests for explicit selector"
		}
		for _, rel := range strings.Split(tree, "\n") {
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
	out := map[string]string{}
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Recv != nil || !isGoTestEntryName(fn.Name.Name) {
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

func allTestsPass(ctx context.Context, exec CommandRunner, dir string, pkgs, pyTests, tags []string) bool {
	if len(pkgs) > 0 && !goTestPasses(ctx, exec, dir, pkgs, tags) {
		return false
	}
	if len(pyTests) > 0 && !pythonTestsPass(ctx, exec, dir, pyTests) {
		return false
	}
	return true
}

type goTestJSONEvent struct {
	Action  string `json:"Action"`
	Package string `json:"Package"`
	Test    string `json:"Test"`
}

type selectedGoTestResult struct {
	passed, matched, selectedFailed bool
	buildFailure, timedOut          bool
	runErr                          error
}

// runSelectedGoTests executes only the requested names and proves every selector
// matched a test that actually started. `go test -run` exits zero for zero matches,
// so the JSON run-event check is part of the fail-closed witness contract.
func runSelectedGoTests(ctx context.Context, run CommandRunner, dir string, pkgs, tags, selectors []string) selectedGoTestResult {
	if run == nil {
		run = commandRunner
	}
	parts := make([]string, 0, len(selectors))
	compiled := make([]*regexp.Regexp, 0, len(selectors))
	for _, selector := range selectors {
		parts = append(parts, "(?:"+selector+")")
		compiled = append(compiled, regexp.MustCompile(selector))
	}
	out, code, err := run(ctx, dir, goTestArgvSelected(pkgs, tags, strings.Join(parts, "|"))...)
	if err != nil {
		return selectedGoTestResult{timedOut: errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled), runErr: err}
	}
	seen := make([]bool, len(compiled))
	failed := make([]bool, len(compiled))
	for _, line := range strings.Split(out, "\n") {
		var event goTestJSONEvent
		if json.Unmarshal([]byte(line), &event) != nil {
			continue
		}
		if event.Test == "" {
			continue
		}
		for i, selector := range compiled {
			if selector.MatchString(event.Test) {
				switch event.Action {
				case "run":
					seen[i] = true
				case "fail":
					failed[i] = true
				}
			}
		}
	}
	for _, matched := range seen {
		if !matched {
			return selectedGoTestResult{
				passed: code == 0, buildFailure: code != 0 && isGoBuildFailure(out),
				timedOut: strings.Contains(out, "panic: test timed out"),
			}
		}
	}
	selectedFailed := false
	for _, testFailed := range failed {
		if testFailed {
			selectedFailed = true
			break
		}
	}
	return selectedGoTestResult{
		passed: code == 0, matched: true, selectedFailed: selectedFailed,
		timedOut: strings.Contains(out, "panic: test timed out"),
	}
}

// overlayTestsAtRef writes each changed test file's content AT <commit> into the corresponding
// path under destDir, so the parent worktree runs the NEW test against the OLD source. It reads
// the blob with `git show <commit>:<path>` through the injected runner.
func overlayTestsAtRef(ctx context.Context, git Runner, repoDir, commit, destDir string, tests []string) bool {
	if git == nil {
		git = gitRunner
	}
	for _, rel := range tests {
		content, code, err := git(ctx, repoDir, "show", commit+":"+rel)
		if err != nil || code != 0 {
			return false
		}
		dst := filepath.Join(destDir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return false
		}
		if err := os.WriteFile(dst, []byte(content), 0o644); err != nil {
			return false
		}
	}
	return true
}

// goTestPasses runs `go test` for the given packages in dir and reports whether it exited 0.
// tags are the build constraints derived from the changed test files (#13243): a test gated
// behind `//go:build vulkan` is excluded from a bare `go test`, so the execution rung would
// see a trivially-passing package at BOTH refs and falsely REFUTE a genuine device witness.
// A nil/empty tag set keeps the argv byte-identical to the pre-#13243 untagged form.
func goTestPasses(ctx context.Context, run CommandRunner, dir string, pkgs []string, tags []string) bool {
	if run == nil {
		run = commandRunner
	}
	_, code, err := run(ctx, dir, goTestArgv(pkgs, tags)...)
	return err == nil && code == 0
}

// goTestArgv composes the `go test` argv. The -tags flag is appended only when a tag set was
// derived, so an untagged change produces byte-identical argv to today (P1: preserved).
func goTestArgv(pkgs, tags []string) []string {
	argv := append([]string{"go", "test", "-count=1"}, pkgs...)
	if len(tags) > 0 {
		argv = append(argv, "-tags", strings.Join(tags, ","))
	}
	return argv
}

func goTestArgvSelected(pkgs, tags []string, selector string) []string {
	argv := []string{"go", "test", "-json", "-count=1", "-run", selector}
	argv = append(argv, pkgs...)
	if len(tags) > 0 {
		argv = append(argv, "-tags", strings.Join(tags, ","))
	}
	return argv
}

// buildTagsFromTestSources derives the build-constraint tag set from the bodies of the changed
// test files (#13243). It reads the leading `//go:build` expression (and the legacy `// +build`
// form), extracting each tag token and dropping the boolean operators. Unknown tokens (the
// `go1.21` release tags, `cgo`) are kept as-is: `go test -tags` accepts them, and passing an
// irrelevant tag is harmless, whereas DROPPING a real one silently excludes the witness again.
// A file with no constraint contributes nothing.
func buildTagsFromTestSources(bodies []string) []string {
	seen := map[string]bool{}
	var tags []string
	for _, body := range bodies {
		for _, tag := range parseBuildConstraints(body) {
			if seen[tag] {
				continue
			}
			seen[tag] = true
			tags = append(tags, tag)
		}
	}
	sort.Strings(tags)
	return tags
}

// parseBuildConstraints extracts the tag tokens from a source file's leading build constraints,
// stopping at the first non-comment, non-blank line (constraints must precede the package clause).
func parseBuildConstraints(body string) []string {
	var tags []string
	for _, line := range strings.Split(body, "\n") {
		s := strings.TrimSpace(line)
		if s == "" {
			continue
		}
		if !strings.HasPrefix(s, "//") {
			break // the package clause (or any code) ends the build-constraint block
		}
		expr := ""
		switch {
		case strings.HasPrefix(s, "//go:build"):
			expr = strings.TrimSpace(strings.TrimPrefix(s, "//go:build"))
		case strings.HasPrefix(s, "// +build"):
			// Legacy form: space-separated tags on the line, OPTIONS on extra lines.
			// Legacy `,` is AND and a space is OR, but for -tags purposes either way each
			// token is a tag name, so the naive split is sufficient.
			expr = strings.TrimSpace(strings.TrimPrefix(s, "// +build"))
			expr = strings.ReplaceAll(expr, ",", " ")
		default:
			continue
		}
		tags = append(tags, tokenizeConstraint(expr)...)
	}
	return tags
}

// tokenizeConstraint splits a build expression into its tag tokens, dropping the boolean
// operators/punctuation (`&&`, `||`, `!`, `(`, `)`). `!windows` keeps its `!` so `go test`
// applies the negation the file declared.
func tokenizeConstraint(expr string) []string {
	fields := strings.FieldsFunc(expr, func(r rune) bool {
		return r == '(' || r == ')' || r == '&' || r == '|'
	})
	var tags []string
	for _, f := range fields {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		tags = append(tags, f)
	}
	return tags
}

// resolveGoBuildTags reads each changed test file's content AT ref through the injected git
// runner and returns the derived tag set. Any read failure yields nil (the untagged default),
// never a guessed tag set: an unreadable constraint means we cannot know the tags, and the
// caller's fail-closed ABSTAIN path handles the untagged run's outcome.
func resolveGoBuildTags(ctx context.Context, git Runner, repoDir, ref string, tests []string) []string {
	if git == nil {
		git = gitRunner
	}
	var bodies []string
	for _, rel := range tests {
		if !strings.HasSuffix(rel, "_test.go") {
			continue
		}
		content, code, err := git(ctx, repoDir, "show", ref+":"+rel)
		if err != nil || code != 0 {
			continue
		}
		bodies = append(bodies, content)
	}
	return buildTagsFromTestSources(bodies)
}

// pythonTestsPass runs each Python test script in dir and reports whether all exited 0.
func pythonTestsPass(ctx context.Context, run CommandRunner, dir string, tests []string) bool {
	if run == nil {
		run = commandRunner
	}
	py := pythonBin()
	for _, t := range tests {
		_, code, err := run(ctx, dir, py, t)
		if err != nil || code != 0 {
			return false
		}
	}
	return true
}

func pythonBin() string {
	if env := strings.TrimSpace(os.Getenv("PYTHON")); env != "" {
		return env
	}
	if runtime.GOOS == "windows" {
		if _, err := exec.LookPath("python"); err == nil {
			return "python"
		}
		if _, err := exec.LookPath("python3"); err == nil {
			return "python3"
		}
		return "python"
	}
	if _, err := exec.LookPath("python3"); err == nil {
		return "python3"
	}
	if _, err := exec.LookPath("python"); err == nil {
		return "python"
	}
	return "python3"
}

func runParentTests(ctx context.Context, exec CommandRunner, dir string, pkgs, pyTests, tags []string) (passed bool, buildErr bool) {
	goPassed := true
	if len(pkgs) > 0 {
		var err bool
		goPassed, err = runGoTests(ctx, exec, dir, pkgs, tags)
		if err {
			return false, true
		}
	}
	pyPassed := true
	if len(pyTests) > 0 {
		var err bool
		pyPassed, err = runPythonTests(ctx, exec, dir, pyTests)
		if err {
			return false, true
		}
	}
	if goPassed && pyPassed {
		return true, false
	}
	return false, false
}

func runGoTests(ctx context.Context, run CommandRunner, dir string, pkgs, tags []string) (passed bool, buildErr bool) {
	if run == nil {
		run = commandRunner
	}
	out, code, err := run(ctx, dir, goTestArgv(pkgs, tags)...)
	if err != nil {
		return false, true
	}
	if code == 0 {
		return true, false
	}
	if isGoBuildFailure(out) {
		return false, true
	}
	return false, false
}

func runPythonTests(ctx context.Context, run CommandRunner, dir string, tests []string) (passed bool, buildErr bool) {
	if run == nil {
		run = commandRunner
	}
	py := pythonBin()
	allPassed := true
	for _, t := range tests {
		out, code, err := run(ctx, dir, py, t)
		if err != nil {
			return false, true
		}
		if code == 0 {
			continue
		}
		if isPythonBuildFailure(out) {
			return false, true
		}
		allPassed = false
	}
	return allPassed, false
}

func isGoBuildFailure(out string) bool {
	if strings.Contains(out, "[build failed]") || strings.Contains(out, "[setup failed]") {
		return true
	}
	lower := strings.ToLower(out)
	if strings.Contains(lower, "compile error") ||
		strings.Contains(lower, "compiler error") ||
		strings.Contains(lower, "build error") ||
		strings.Contains(lower, "syntax error") ||
		strings.Contains(lower, "cannot find package") ||
		strings.Contains(lower, "no go files in") {
		return true
	}
	if !strings.Contains(out, "--- FAIL:") {
		if strings.Contains(lower, "build failed") ||
			strings.Contains(lower, "setup failed") ||
			strings.Contains(lower, "undefined:") {
			return true
		}
	}
	return false
}

func isPythonBuildFailure(out string) bool {
	lower := strings.ToLower(out)
	if strings.Contains(lower, "syntaxerror:") ||
		strings.Contains(lower, "importerror:") ||
		strings.Contains(lower, "modulenotfounderror:") ||
		strings.Contains(lower, "indentationerror:") ||
		strings.Contains(lower, "taberror:") ||
		strings.Contains(lower, "compile error") ||
		strings.Contains(lower, "compiler error") ||
		strings.Contains(lower, "build error") {
		return true
	}
	return false
}
