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
	"context"
	"go/ast"
	"go/doc"
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
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return abi.WitnessAbstain
	}
	git := r.run
	if git == nil {
		git = gitRunner
	}

	// STRUCTURAL rung: which _test.go files did the commit add or modify?
	out, code, err := git(ctx, r.dir, "show", "--name-only", "--format=", ref)
	if err != nil || code != 0 {
		return abi.WitnessAbstain // bad ref / git missing — never a false CONFIRM
	}
	tests := changedTestFiles(out)
	if len(tests) == 0 {
		return abi.WitnessRefuted // a fix with no symptom witness — the whole point
	}

	// EXECUTION rung is opt-in via env or mandatoryExec; without it the structural pass is as far as we honestly go.
	if !mandatoryExec && !SymptomExecEnabled() {
		return abi.WitnessAbstain
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
func (r *Resolver) resolveSymptomExec(ctx context.Context, ref string, tests []string) abi.WitnessOutcome {
	exec := r.execRun
	if exec == nil {
		exec = commandRunner
	}
	v := NewExecutionVerifierWithRunners(r.run, exec, r.dir)

	commit, ok := v.revParse(ctx, ref+"^{commit}")
	if !ok {
		return abi.WitnessAbstain
	}
	parent, ok := v.revParse(ctx, commit+"^")
	if !ok {
		return abi.WitnessAbstain // a root commit has no parent to red-test against
	}

	pkgs := testPackages(tests)
	pyTests := pythonTestFiles(tests)
	if len(pkgs) == 0 && len(pyTests) == 0 {
		return abi.WitnessAbstain
	}
	goSelections, selectionOK := resolveGoTestSelectionsAtParent(ctx, r.run, r.dir, parent, commit, tests)

	// The build constraints the changed test files declare (#13243). A device-tagged test
	// (`//go:build vulkan`, `metal`, `cuda`, …) is excluded from a bare `go test`, so without
	// this the package passes trivially at BOTH refs and a genuine witness is falsely refuted.
	// An explicit caller hint (WithSymptomTags) is UNIONED with the derived truth — the hint
	// can add a tag the derivation missed but never remove one a test file declares.
	tags := mergeTags(resolveGoBuildTags(ctx, r.run, r.dir, commit, tests), r.symptomTags)

	// GREEN at the fix: the changed test must pass at <ref> as committed.
	commitDir, cleanupCommit, err := v.scratchWorktree(ctx, commit)
	if err != nil {
		return abi.WitnessAbstain
	}
	defer cleanupCommit()
	goPassed := true
	useSelection := selectionOK && len(goSelections) > 0
	if len(pkgs) > 0 {
		if useSelection {
			var definitive bool
			goPassed, definitive = selectedGoTestsPass(ctx, exec, commitDir, goSelections, tags)
			if !definitive {
				// A successful command with no observable test execution is not evidence.
				// Fall back to the established full-package witness at both refs.
				useSelection = false
				goPassed = goTestPasses(ctx, exec, commitDir, pkgs, tags)
			}
		} else {
			goPassed = goTestPasses(ctx, exec, commitDir, pkgs, tags)
		}
	}
	if !goPassed || (len(pyTests) > 0 && !pythonTestsPass(ctx, exec, commitDir, pyTests)) {
		// The committed test does not even pass at the fix — not a usable witness; don't CONFIRM.
		return abi.WitnessRefuted
	}

	// RED at the parent: overlay each changed test file (its <ref> content) onto the parent
	// worktree, then run it. It must FAIL — the test reproduces the bug against the old source.
	parentDir, cleanupParent, err := v.scratchWorktree(ctx, parent)
	if err != nil {
		return abi.WitnessAbstain
	}
	defer cleanupParent()
	if !overlayTestsAtRef(ctx, r.run, r.dir, commit, parentDir, tests) {
		return abi.WitnessAbstain // could not stage the red test — uncertain, never a false CONFIRM
	}
	passed, buildErr := runParentTestsSelected(ctx, exec, parentDir, pkgs, pyTests, tags, goSelections, useSelection)
	if buildErr {
		// Parent failed to compile/build (e.g. test references an API introduced by the fix) —
		// this is unproven, never a false CONFIRM of behavioral reproduction (#12058).
		return abi.WitnessAbstain
	}
	if passed {
		// The test passes against the OLD source too: it constrains nothing about the bug.
		return abi.WitnessRefuted
	}
	return abi.WitnessConfirmed
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

// goTestSelection is the exact runnable test set contributed by changed Go test
// blobs in one package. Names are resolved once from the fix commit and reused for
// both the green candidate and red parent overlay.
type goTestSelection struct {
	pkg   string
	names []string
}

// resolveGoTestSelections reads changed Go test files from the resolved commit,
// never from the caller's working tree. Selection is deliberately conservative:
// an unreadable or unparseable blob, an unsupported Test/Fuzz/Example declaration,
// or a helper-only changed file returns false and preserves the full-package path.
func resolveGoTestSelections(ctx context.Context, git Runner, repoDir, commit string, tests []string) ([]goTestSelection, bool) {
	if git == nil {
		git = gitRunner
	}
	parent, code, err := git(ctx, repoDir, "rev-parse", commit+"^")
	if err != nil || code != 0 {
		return nil, false
	}
	return resolveGoTestSelectionsAtParent(ctx, git, repoDir, strings.TrimSpace(parent), commit, tests)
}

func resolveGoTestSelectionsAtParent(ctx context.Context, git Runner, repoDir, parent, commit string, tests []string) ([]goTestSelection, bool) {
	if git == nil {
		git = gitRunner
	}
	byPackage := make(map[string]map[string]bool)
	sawGoTest := false
	for _, rel := range tests {
		rel = strings.ReplaceAll(strings.TrimSpace(rel), "\\", "/")
		if !strings.HasSuffix(rel, "_test.go") {
			continue
		}
		sawGoTest = true
		if !testFileAddedAtCommit(ctx, git, repoDir, parent, commit, rel) {
			// A modified test file may also change helpers, imports, or constraints used
			// by tests in other files. Without proving those declarations unchanged,
			// selecting only this file's runnable names could miss the real witness.
			return nil, false
		}
		body, code, err := git(ctx, repoDir, "show", commit+":"+rel)
		if err != nil || code != 0 {
			return nil, false
		}
		names, ok := runnableGoTestNames(rel, body)
		if !ok || len(names) == 0 {
			return nil, false
		}
		dir := path.Dir(rel)
		if dir == "." || dir == "" {
			dir = "."
		}
		pkg := "./" + strings.TrimPrefix(dir, "./")
		if byPackage[pkg] == nil {
			byPackage[pkg] = make(map[string]bool)
		}
		for _, name := range names {
			byPackage[pkg][name] = true
		}
	}
	if !sawGoTest {
		return nil, true
	}
	pkgs := make([]string, 0, len(byPackage))
	for pkg := range byPackage {
		pkgs = append(pkgs, pkg)
	}
	sort.Strings(pkgs)
	selections := make([]goTestSelection, 0, len(pkgs))
	for _, pkg := range pkgs {
		names := make([]string, 0, len(byPackage[pkg]))
		for name := range byPackage[pkg] {
			names = append(names, name)
		}
		sort.Strings(names)
		selections = append(selections, goTestSelection{pkg: pkg, names: names})
	}
	return selections, len(selections) > 0
}

func testFileAddedAtCommit(ctx context.Context, git Runner, repoDir, parent, commit, rel string) bool {
	out, code, err := git(ctx, repoDir, "diff", "--diff-filter=A", "--name-only", parent, commit, "--", rel)
	if err != nil || code != 0 {
		return false
	}
	found := false
	for _, line := range strings.Split(out, "\n") {
		p := strings.ReplaceAll(strings.TrimSpace(line), "\\", "/")
		if p == "" {
			continue
		}
		if p != rel || found {
			return false
		}
		found = true
	}
	return found
}

func runnableGoTestNames(filename, body string) ([]string, bool) {
	f, err := parser.ParseFile(token.NewFileSet(), filename, body, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil {
		return nil, false
	}
	seen := make(map[string]bool)
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Recv != nil {
			continue
		}
		name := fn.Name.Name
		switch {
		case name == "TestMain":
			switch {
			case validGoTestFunc(fn, "T"):
				seen[name] = true
			case validGoTestFunc(fn, "M"):
				// TestMain controls the selected run but is not itself selected by -run.
			default:
				return nil, false
			}
		case goTestName(name, "Test"):
			if !validGoTestFunc(fn, "T") {
				return nil, false
			}
			seen[name] = true
		case goTestName(name, "Fuzz"):
			if !validGoTestFunc(fn, "F") {
				return nil, false
			}
			seen[name] = true
		case goTestName(name, "Example"):
			if fn.Type.TypeParams.NumFields() > 0 || fn.Type.Params.NumFields() > 0 || fn.Type.Results.NumFields() > 0 || fn.Body == nil {
				return nil, false
			}
		}
	}
	for _, example := range doc.Examples(f) {
		if example.Output == "" && !example.EmptyOutput {
			continue
		}
		seen["Example"+example.Name] = true
	}
	names := make([]string, 0, len(seen))
	for name := range seen {
		names = append(names, name)
	}
	sort.Strings(names)
	return names, true
}

// validGoTestFunc mirrors cmd/go's conservative signature check for Test and
// Fuzz entry points without attempting to resolve import aliases.
func validGoTestFunc(fn *ast.FuncDecl, arg string) bool {
	if fn.Type.TypeParams.NumFields() > 0 || fn.Type.Results.NumFields() > 0 || fn.Type.Params.NumFields() != 1 {
		return false
	}
	field := fn.Type.Params.List[0]
	if len(field.Names) > 1 {
		return false
	}
	ptr, ok := field.Type.(*ast.StarExpr)
	if !ok {
		return false
	}
	switch x := ptr.X.(type) {
	case *ast.Ident:
		return x.Name == arg
	case *ast.SelectorExpr:
		return x.Sel.Name == arg
	default:
		return false
	}
}

func goTestName(name, prefix string) bool {
	if !strings.HasPrefix(name, prefix) {
		return false
	}
	if len(name) == len(prefix) {
		return true
	}
	r, _ := utf8.DecodeRuneInString(name[len(prefix):])
	return !unicode.IsLower(r)
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

// selectedGoTestArgv adds an exact, anchored -run expression while leaving the
// legacy goTestArgv package/tag behavior unchanged.
func selectedGoTestArgv(pkg string, tags, names []string) ([]string, bool) {
	pattern, ok := exactGoTestRunPattern(names)
	if !ok || strings.TrimSpace(pkg) == "" {
		return nil, false
	}
	argv := goTestArgv([]string{pkg}, tags)
	return append(argv, "-run", pattern), true
}

func exactGoTestRunPattern(names []string) (string, bool) {
	seen := make(map[string]bool)
	var quoted []string
	for _, name := range names {
		name = strings.TrimSpace(name)
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		quoted = append(quoted, regexp.QuoteMeta(name))
	}
	if len(quoted) == 0 {
		return "", false
	}
	sort.Strings(quoted)
	if len(quoted) == 1 {
		return "^" + quoted[0] + "$", true
	}
	return "^(?:" + strings.Join(quoted, "|") + ")$", true
}

// selectedGoTestsPass runs each package with only its changed runnable test
// names. definitive is false when go test reports no selected test execution;
// callers must then use the full-package fallback rather than accept a vacuous pass.
func selectedGoTestsPass(ctx context.Context, run CommandRunner, dir string, selections []goTestSelection, tags []string) (passed, definitive bool) {
	if run == nil {
		run = commandRunner
	}
	if len(selections) == 0 {
		return false, false
	}
	for _, selection := range selections {
		argv, ok := selectedGoTestArgv(selection.pkg, tags, selection.names)
		if !ok {
			return false, false
		}
		out, code, err := run(ctx, dir, argv...)
		if err != nil || code != 0 {
			return false, true
		}
		if noGoTestsRan(out) {
			return false, false
		}
	}
	return true, true
}

func noGoTestsRan(out string) bool {
	lower := strings.ToLower(strings.TrimSpace(out))
	return strings.Contains(lower, "no tests to run") ||
		strings.Contains(lower, "no tests were run") ||
		strings.Contains(lower, "[no test files]")
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

func runParentTestsSelected(ctx context.Context, exec CommandRunner, dir string, pkgs, pyTests, tags []string, selections []goTestSelection, useSelection bool) (passed bool, buildErr bool) {
	goPassed := true
	if len(pkgs) > 0 {
		var err bool
		if useSelection {
			goPassed, err = runSelectedGoTests(ctx, exec, dir, selections, tags)
		} else {
			goPassed, err = runGoTests(ctx, exec, dir, pkgs, tags)
		}
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
	return goPassed && pyPassed, false
}

func runSelectedGoTests(ctx context.Context, run CommandRunner, dir string, selections []goTestSelection, tags []string) (passed bool, buildErr bool) {
	if run == nil {
		run = commandRunner
	}
	if len(selections) == 0 {
		return false, true
	}
	for _, selection := range selections {
		argv, ok := selectedGoTestArgv(selection.pkg, tags, selection.names)
		if !ok {
			return false, true
		}
		out, code, err := run(ctx, dir, argv...)
		if err != nil {
			return false, true
		}
		if code == 0 {
			if noGoTestsRan(out) {
				return false, true // uncertain/vacuous parent result => ABSTAIN
			}
			continue
		}
		if isGoBuildFailure(out) {
			return false, true
		}
		return false, false
	}
	return true, false
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
