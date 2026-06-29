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
//	STRUCTURAL (always, flag-independent): did the fix commit add or modify a `_test.go`?
//	  No test touched => REFUTED — a fix with no symptom witness is exactly the gap. This is
//	  the cheap, deterministic half; it needs no second worktree and never abstains on cost.
//
//	EXECUTION (red-then-green, gated behind FAK_WITNESS_SYMPTOM): overlay the ref's version of
//	  each changed test onto a PARENT scratch worktree and run it — it must FAIL (red: the test
//	  reproduces the bug against the old source) — then run it at the ref — it must PASS (green).
//	  Both hold => CONFIRMED. Passes-at-parent => REFUTED (tautological). Default (flag unset) =>
//	  ABSTAIN after the structural check: running an arbitrary test against an old tree is heavy,
//	  so like the RSL rung the cost is opt-in and the kernel's fail-closed default turns abstain
//	  into a deny rather than a false CONFIRM.
//
// This is the mirror of the `notests:` rung in this package: notests REFUTES a ship-commit that
// edited the very tests it must pass (reward-hack); symptom CONFIRMS a fix-commit that added a
// test which genuinely constrains the bug. Same git evidence, opposite polarity.

import (
	"context"
	"os"
	"path"
	"strings"

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

// resolveSymptom adjudicates a `symptom:<ref>` claim: does the fix at <ref> carry a witness that
// the symptom is gone? See the package doc for the two-rung contract.
func (r *Resolver) resolveSymptom(ctx context.Context, ref string) abi.WitnessOutcome {
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

	// EXECUTION rung is opt-in; without it the structural pass is as far as we honestly go.
	if !SymptomExecEnabled() {
		return abi.WitnessAbstain
	}
	return r.resolveSymptomExec(ctx, ref, tests)
}

// changedTestFiles extracts the repo-relative `_test.go` paths from `git show --name-only`
// output, normalizing separators and skipping blanks. testdata/ fixtures are excluded — a
// fixture named *_test.go is not a gating test.
func changedTestFiles(nameOnly string) []string {
	var tests []string
	for _, line := range strings.Split(nameOnly, "\n") {
		p := strings.ReplaceAll(strings.TrimSpace(line), "\\", "/")
		if p == "" || !strings.HasSuffix(p, "_test.go") {
			continue
		}
		if isUnderTestdata(p) {
			continue
		}
		tests = append(tests, p)
	}
	return tests
}

func isUnderTestdata(p string) bool {
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
	if len(pkgs) == 0 {
		return abi.WitnessAbstain
	}

	// GREEN at the fix: the changed test must pass at <ref> as committed.
	commitDir, cleanupCommit, err := v.scratchWorktree(ctx, commit)
	if err != nil {
		return abi.WitnessAbstain
	}
	defer cleanupCommit()
	if !goTestPasses(ctx, exec, commitDir, pkgs) {
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
	if goTestPasses(ctx, exec, parentDir, pkgs) {
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
		dst := path.Join(strings.ReplaceAll(destDir, "\\", "/"), rel)
		if err := os.MkdirAll(path.Dir(dst), 0o755); err != nil {
			return false
		}
		if err := os.WriteFile(dst, []byte(content), 0o644); err != nil {
			return false
		}
	}
	return true
}

// goTestPasses runs `go test` for the given packages in dir and reports whether it exited 0.
func goTestPasses(ctx context.Context, run CommandRunner, dir string, pkgs []string) bool {
	if run == nil {
		run = commandRunner
	}
	argv := append([]string{"go", "test", "-count=1"}, pkgs...)
	_, code, err := run(ctx, dir, argv...)
	return err == nil && code == 0
}
