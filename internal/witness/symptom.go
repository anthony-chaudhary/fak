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
//	  e.g. referencing an API introduced in the fix — never a false CONFIRM; #12058). The caller's
//	  context bounds the whole rung: once it ends, any verdict => ABSTAIN as a timeout, because a
//	  killed `go test` is indistinguishable from a red or a zero-match run. Default (flag unset) =>
//	  ABSTAIN after the structural check: running an arbitrary test against an old tree is heavy,
//	  so like the RSL rung the cost is opt-in and the kernel's fail-closed default turns abstain
//	  into a deny rather than a false CONFIRM.
//
// This is the mirror of the `notests:` rung in this package: notests REFUTES a ship-commit that
// edited the very tests it must pass (reward-hack); symptom CONFIRMS a fix-commit that added a
// test which genuinely constrains the bug. Same git evidence, opposite polarity.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/buildwitness"
	"github.com/anthony-chaudhary/fak/internal/computebuild"
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
// never remove a derived feature tag. Platform constraints remain owned by GOOS/GOARCH.
// The empty default keeps the untagged behavior byte-identical.
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

// mergeTags unions feature tags; GOOS/GOARCH constraints must never be forced with -tags.
// Either side may be nil; both nil yields nil so the caller's untagged path stays byte-identical.
func mergeTags(a, b []string) []string {
	if len(a) == 0 && len(b) == 0 {
		return nil
	}
	var features []string
	for _, tag := range normalizeTags(append(append([]string{}, a...), b...)) {
		platform := buildwitness.FilenamePlatformTags("witness_" + tag + ".go")
		if len(platform) == 1 && platform[0] == tag {
			continue
		}
		if toolchainOwnedTag(tag) {
			continue
		}
		features = append(features, tag)
	}
	return features
}

var releaseTagPattern = regexp.MustCompile(`^go1\.\d+$`)

// toolchainOwnedTag reports tags the go command derives itself; forcing one with
// -tags redeclares stdlib files (unix on Windows) or claims a toolchain the host
// does not run, and a negated token is never a tag to set.
func toolchainOwnedTag(tag string) bool {
	switch tag {
	case "unix", "cgo", "gc", "gccgo", "ignore":
		return true
	}
	return strings.HasPrefix(tag, "!") || releaseTagPattern.MatchString(tag)
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
// A resolution whose context ended (deadline or cancel) abstains as a timeout; see
// symptomBudgetVerdict.
func (r *Resolver) ResolveSymptomWithDetail(ctx context.Context, ref string, mandatoryExec bool) (abi.WitnessOutcome, string) {
	outcome, detail := r.resolveSymptomWithDetail(ctx, ref, mandatoryExec)
	return symptomBudgetVerdict(ctx, outcome, detail)
}

// symptomBudgetVerdict refuses to trust any verdict reached after the resolution context
// ended. Every git and `go test` step runs under ctx, and a step the budget kills fails in
// the same shape as a genuine negative: a mid-compile kill leaves no `run` event (reads as
// "matched no executed test"), a killed parent run exits non-zero (reads as RED), a killed
// git read reads as a missing ref. Once ctx is done the rung cannot tell a real refutation
// or confirmation from a kill, so it abstains as a timeout — never a refutation (a worker
// land's 10-minute budget expiring mid-compile was once reported as SYMPTOM_NO_MATCH).
func symptomBudgetVerdict(ctx context.Context, outcome abi.WitnessOutcome, detail string) (abi.WitnessOutcome, string) {
	if ctx.Err() == nil {
		return outcome, detail
	}
	cause := "symptom budget expired: " + ctx.Err().Error()
	if strings.Contains(detail, "timed out") {
		return abi.WitnessAbstain, detail + " (" + cause + ")"
	}
	return abi.WitnessAbstain, "timed out before a trustworthy verdict (" + cause + "); discarded result: " + detail
}

// symptomRunInterrupted reports whether a runner result was cut short by the resolution
// context. commandRunner uses exec.CommandContext with a process-tree cancel, and Cmd.Wait
// reports the killed process as an ordinary *exec.ExitError, so the runner returns
// (partial output, non-zero code, err == nil): the same shape as a real test failure or a
// zero-match run. Only the context says the run was killed.
func symptomRunInterrupted(ctx context.Context, err error) bool {
	return ctx.Err() != nil || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled)
}

func (r *Resolver) resolveSymptomWithDetail(ctx context.Context, ref string, mandatoryExec bool) (abi.WitnessOutcome, string) {
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

// SymptomTestFiles returns the paths the symptom witness would count as changed
// tests, by the same rule it applies to a commit's name list. A fix(*) land whose
// paths yield none is refuted as "no changed test file" without building anything.
func SymptomTestFiles(paths []string) []string {
	return changedTestFiles(strings.Join(paths, "\n"))
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
	// can add a feature tag the derivation missed; GOOS/GOARCH remain target constraints.
	tags := mergeTags(resolveGoBuildTags(ctx, r.run, r.dir, commit, tests), r.symptomTags)

	if len(selections) > 0 {
		var hostExcluded int
		selections, hostExcluded = dropHostExcludedSelections(ctx, r.run, r.dir, commit, tests, tags, selections)
		if len(selections) == 0 {
			return abi.WitnessAbstain, fmt.Sprintf("symptom selector abstained: every selected changed test is excluded on %s/%s (%d host-excluded changed test file(s))", runtime.GOOS, runtime.GOARCH, hostExcluded)
		}
	}
	predicted := fixOnlyTestReferences(ctx, r.run, r.dir, commit, parent, tests)
	if len(pyTests) == 0 {
		if detail, doomed := staticParentUnbuildable(ctx, r.run, r.dir, commit, selections, predicted); doomed {
			return abi.WitnessAbstain, detail
		}
	}

	// Materialize the candidate first, preserving the established verdict order:
	// a bad candidate is reported before any parent preparation problem.
	commitDir, cleanupCommit, err := v.scratchWorktree(ctx, commit)
	if err != nil {
		return abi.WitnessAbstain, "candidate scratch worktree failed"
	}
	defer cleanupCommit()
	parentDir := ""
	vulkanEnvs := make(map[string]map[string]string)
	vulkanScratchRequired := false
	if r.realCommandExecution {
		exec = func(ctx context.Context, dir string, argv ...string) (string, int, error) {
			if env := vulkanEnvs[filepath.Clean(dir)]; len(env) > 0 {
				return commandRunnerEnv(env)(ctx, dir, argv...)
			}
			return commandRunner(ctx, dir, argv...)
		}
		if env, needed, prepErr := prepareVulkanSymptomScratch(ctx, commitDir, pkgs, tags, false); needed {
			vulkanScratchRequired = true
			if prepErr != nil {
				r.symptomExcerpt = capSymptomExcerpt("Vulkan scratch preparation: " + prepErr.Error())
				if ctx.Err() != nil {
					return abi.WitnessAbstain, "candidate Vulkan scratch preparation timed out"
				}
				return abi.WitnessRefuted, "candidate Vulkan scratch preparation failed"
			}
			vulkanEnvs[filepath.Clean(commitDir)] = env
		}
	}
	var cleanupParent func()
	defer func() {
		if cleanupParent != nil {
			cleanupParent()
		}
	}()
	prepareParent := func() string {
		parentDir, cleanupParent, err = v.scratchWorktree(ctx, parent)
		if err != nil {
			return "parent scratch worktree failed"
		}
		if r.realCommandExecution && vulkanScratchRequired {
			if env, needed, prepErr := prepareVulkanSymptomScratch(ctx, parentDir, nil, tags, true); needed {
				if prepErr != nil {
					return "parent Vulkan scratch preparation failed"
				}
				vulkanEnvs[filepath.Clean(parentDir)] = env
			}
		}
		if !overlayTestsAtRef(ctx, r.run, r.dir, commit, parentDir, tests) {
			return "could not overlay changed tests onto parent"
		}
		return ""
	}

	proofKey, parentProblem := "", ""
	cacheCandidate := r.realCommandExecution && len(selections) > 0 && len(pyTests) == 0 &&
		symptomChangedTestsCheapCacheEligible(ctx, r.run, r.dir, commit, tests)
	if cacheCandidate {
		parentProblem = prepareParent()
		if parentProblem == "" {
			proofKey, _ = r.symptomProofCacheKey(ctx, commitDir, parentDir, selections, tags)
		}
		if proofKey != "" && r.symptomProofCacheConfirmed(ctx, proofKey) {
			return abi.WitnessConfirmed, "reused confirmed selected symptom proof (parent failed; candidate passed)"
		}
	}

	// GREEN at the fix: the changed test must pass at <ref> as committed. A name red here
	// is tolerated only as pre-existing: it FAILs with the parent's own body, and its
	// candidate body FAILs at the parent with the same first diagnostic. Any other red refutes.
	var (
		tol               symptomTolerance
		passNames, ownRed [][]string
		redOut            string
		sawFail           bool
	)
	// Once a candidate red is explicit, parent or proof infrastructure trouble refutes
	// with the candidate-failed family rather than abstaining.
	refuteOrAbstain := func(abstain, refute string) (abi.WitnessOutcome, string) {
		if sawFail {
			return abi.WitnessRefuted, "candidate selected symptom test failed: " + refute
		}
		return abi.WitnessAbstain, abstain
	}
	if len(selections) > 0 {
		passNames = make([][]string, len(selections))
		pending := make([][]string, len(selections))
		pendingRes := make([]selectedGoTestResult, len(selections))
		var firstRedOut string
		goRed := false
		anyPass, exitFailed := false, false
		for i, selection := range selections {
			result := runSelectedGoTests(ctx, exec, commitDir, []string{selection.Package}, tags, exactTestSelectors(selection.Tests))
			// Base refutes at the first red selection, so a later non-clean outcome cannot win.
			if goRed && (result.timedOut || result.runErr != nil || result.workspaceBroken || result.buildFailure || !result.matched) {
				r.symptomExcerpt = symptomFailureExcerpt(firstRedOut)
				return abi.WitnessRefuted, "candidate selected symptom test failed"
			}
			switch {
			case result.timedOut:
				return abi.WitnessAbstain, "candidate selected symptom test timed out"
			case result.runErr != nil:
				return abi.WitnessAbstain, "candidate selected test execution failed"
			case result.workspaceBroken:
				// The committed go.work cannot load in this scratch topology (#13213): no
				// test ran at all, so neither REFUTE nor CONFIRM is earned.
				return abi.WitnessAbstain, "candidate selected symptom test could not load go.work"
			case result.buildFailure:
				r.symptomExcerpt = symptomFailureExcerpt(result.output)
				return abi.WitnessRefuted, "candidate selected symptom test did not build"
			case !result.matched:
				return abi.WitnessRefuted, "candidate symptom selector matched no executed test"
			}
			passNames[i] = namesByState(selection.Tests, result, symptomTestPassed)
			pending[i] = namesByState(selection.Tests, result, symptomTestFailed, symptomTestUnknown)
			pendingRes[i] = result
			if len(namesByState(selection.Tests, result, symptomTestFailed)) > 0 {
				sawFail = true
			}
			if len(passNames[i]) > 0 {
				anyPass = true
			}
			if !result.passed {
				exitFailed = true
			}
			if !goRed && (!result.passed || len(namesByState(selection.Tests, result, symptomTestFailed)) > 0) {
				goRed = true
				firstRedOut = result.output
			}
		}
		// A non-zero exit without an explicit red, or a red with no PASS, refutes as base.
		if (exitFailed && !sawFail) || (sawFail && !anyPass) {
			r.symptomExcerpt = symptomFailureExcerpt(firstRedOut)
			return abi.WitnessRefuted, "candidate selected symptom test failed"
		}

		if sawFail {
			// One rerun, only for the names that were red or unknown at the candidate. A name
			// that passes on the rerun is intermittent, which the fix may have introduced.
			stillRed := make([][]string, len(selections))
			var unresolved []string
			for i, selection := range selections {
				if len(pending[i]) == 0 {
					continue
				}
				rerun := runSelectedGoTests(ctx, exec, commitDir, []string{selection.Package}, tags, exactTestSelectors(pending[i]))
				if rerun.output != "" {
					redOut = rerun.output
				}
				if rerun.timedOut || rerun.runErr != nil || rerun.workspaceBroken {
					unresolved = append(unresolved, pending[i]...)
					continue
				}
				if len(namesByState(pending[i], rerun, symptomTestPassed)) > 0 {
					r.symptomExcerpt = symptomFailureExcerpt(rerun.output)
					return abi.WitnessRefuted, "candidate selected symptom test failed"
				}
				pendingRes[i] = rerun
				stillRed[i] = namesByState(pending[i], rerun, symptomTestFailed)
				unresolved = append(unresolved, namesByState(pending[i], rerun, symptomTestUnknown, symptomTestSkipped)...)
			}
			if len(unresolved) > 0 {
				r.symptomExcerpt = symptomFailureExcerpt(redOut)
				return abi.WitnessRefuted, "candidate selected symptom test failed: " + symptomNameList(unresolved) + " gave no explicit verdict on rerun"
			}

			// A red name the parent does not declare is a new test that regressed, so it is
			// refuted before any parent preparation.
			ownRed = make([][]string, len(selections))
			var regressed []string
			for i, selection := range selections {
				if len(stillRed[i]) == 0 {
					continue
				}
				declared := parentDeclaresTests(ctx, r.run, r.dir, commit, parent, symptomPackageTestFiles(tests, selection.Package), stillRed[i])
				for _, name := range stillRed[i] {
					if declared[name] {
						ownRed[i] = append(ownRed[i], name)
					} else {
						regressed = append(regressed, name)
					}
				}
			}
			if len(regressed) > 0 {
				r.symptomExcerpt = symptomFailureExcerpt(redOut)
				return abi.WitnessRefuted, "candidate selected symptom test regressed: " + symptomNameList(regressed) + " failed at candidate on rerun and is absent at parent"
			}
		} else {
			// Nothing red and the run exited zero: every selected name is proven at the parent,
			// as the base witness does.
			for i, selection := range selections {
				passNames[i] = selection.Tests
			}
		}

		if len(pyTests) > 0 {
			passed, buildErr := runPythonTests(ctx, exec, commitDir, pyTests)
			switch {
			case goRed && (buildErr || !passed):
				// A red Go selection already fixes the verdict; Python must not change it.
				r.symptomExcerpt = symptomFailureExcerpt(firstRedOut)
				return abi.WitnessRefuted, "candidate selected symptom test failed"
			case buildErr:
				return abi.WitnessRefuted, "candidate changed Python symptom test failed to execute"
			case !passed:
				return abi.WitnessRefuted, "candidate changed Python symptom test failed"
			}
		}

		// Each still-red name must FAIL with the parent's own body and FAIL with the
		// candidate body overlaid at the parent, printing the same first diagnostic.
		var ownRedAll []string
		for _, names := range ownRed {
			ownRedAll = append(ownRedAll, names...)
		}
		if len(ownRedAll) > 0 {
			if parentDir == "" {
				parentProblem = prepareParent()
			}
			if parentProblem != "" {
				r.symptomExcerpt = symptomFailureExcerpt(redOut)
				return abi.WitnessRefuted, "candidate selected symptom test failed: " + symptomNameList(ownRedAll) + " gave no explicit verdict at parent"
			}
			var regressedAtParent, diagMismatch, unresolvedAtParent []string
			for i, selection := range selections {
				if len(ownRed[i]) == 0 {
					continue
				}
				own, ok := r.runParentOwnTests(ctx, exec, parentDir, commit, parent, selection.Package, ownRed[i], tests, tags)
				if !ok {
					r.symptomExcerpt = symptomFailureExcerpt(redOut)
					return abi.WitnessRefuted, "candidate selected symptom test failed: " + symptomNameList(ownRed[i]) + " gave no explicit verdict at parent"
				}
				// The candidate overlay is back in place once runParentOwnTests returns.
				overlaid := runSelectedGoTests(ctx, exec, parentDir, []string{selection.Package}, tags, exactTestSelectors(ownRed[i]))
				for j, name := range ownRed[i] {
					ownState, overlayState := stateOf(own, j), stateOf(overlaid, j)
					switch {
					case ownState == symptomTestPassed || overlayState == symptomTestPassed:
						regressedAtParent = append(regressedAtParent, name)
					case ownState == symptomTestFailed && overlayState == symptomTestFailed:
						candDiag := diagFor(pending[i], pendingRes[i], name)
						if diagnosticsMatch(candDiag, diagFor(ownRed[i], overlaid, name)) {
							tol.preexisting = append(tol.preexisting, name)
						} else {
							diagMismatch = append(diagMismatch, name)
						}
					default:
						unresolvedAtParent = append(unresolvedAtParent, name)
					}
				}
			}
			if len(regressedAtParent) > 0 {
				r.symptomExcerpt = symptomFailureExcerpt(redOut)
				return abi.WitnessRefuted, "candidate selected symptom test regressed: " + symptomNameList(regressedAtParent) + " failed at candidate on rerun and passed at parent"
			}
			if len(diagMismatch) > 0 {
				r.symptomExcerpt = symptomFailureExcerpt(redOut)
				return abi.WitnessRefuted, "candidate selected symptom test regressed: " + symptomNameList(diagMismatch) + " failed at candidate on rerun and at parent with a different or missing diagnostic"
			}
			if len(unresolvedAtParent) > 0 {
				r.symptomExcerpt = symptomFailureExcerpt(redOut)
				return abi.WitnessRefuted, "candidate selected symptom test failed: " + symptomNameList(unresolvedAtParent) + " gave no explicit verdict at parent"
			}
		}
	} else {
		passed, workspaceBroken := allTestsPass(ctx, exec, commitDir, pkgs, pyTests, tags)
		if workspaceBroken {
			// The committed go.work cannot load in this scratch topology (#13213): no
			// test ran at all, so neither REFUTE nor CONFIRM is earned.
			return abi.WitnessAbstain, "candidate changed-test package could not load go.work"
		}
		if !passed {
			// The committed test does not even pass at the fix — not a usable witness; don't CONFIRM.
			return abi.WitnessRefuted, "candidate changed-test package failed"
		}
	}

	// RED at the parent: the changed tests are already overlaid onto the parent
	// worktree. They must FAIL there — the test reproduces the old behavior.
	if parentDir == "" {
		parentProblem = prepareParent()
	}
	if parentProblem != "" {
		return refuteOrAbstain(parentProblem, parentProblem)
	}
	if len(selections) > 0 {
		parentRed, excluded, ran := false, 0, 0
		for i, selection := range selections {
			// Only the candidate PASS names are proven red at the parent: a tolerated
			// red name is never the witness.
			if len(passNames[i]) == 0 {
				continue
			}
			proof := symptomSelection{Package: selection.Package, Tests: passNames[i]}
			result, dropped := r.runParentSelection(ctx, exec, parentDir, commit, parent, proof, tests, tags, predicted)
			excluded += dropped
			if result.allExcluded {
				continue
			}
			ran++
			switch {
			case result.timedOut:
				return refuteOrAbstain("parent selected symptom test timed out", "parent selected symptom test gave no explicit verdict")
			case result.runErr != nil:
				return refuteOrAbstain("parent selected test execution failed", "parent selected symptom test gave no explicit verdict")
			case result.workspaceBroken:
				// The committed go.work cannot load in this scratch topology (#13213): no
				// test ran at all, so the parent red is unproven — never a CONFIRM.
				return refuteOrAbstain("parent selected symptom test could not load go.work", "parent selected symptom test gave no explicit verdict")
			case result.buildFailure:
				return refuteOrAbstain("parent selected symptom test did not build", "parent selected symptom test gave no explicit verdict")
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
		if ran == 0 && len(pyTests) == 0 {
			return abi.WitnessAbstain, "parent selected symptom test did not build"
		}
		if !parentRed {
			return abi.WitnessRefuted, "selected symptom test passed at parent"
		}
		// A tolerated confirm must not ride on a masked failure: the candidate-PASS names
		// run alone at the candidate and must each PASS with exit 0.
		if !tol.empty() {
			for i, selection := range selections {
				if len(passNames[i]) == 0 {
					continue
				}
				again := runSelectedGoTests(ctx, exec, commitDir, []string{selection.Package}, tags, exactTestSelectors(passNames[i]))
				if again.timedOut || again.runErr != nil || again.workspaceBroken || !again.passed ||
					len(namesByState(passNames[i], again, symptomTestPassed)) != len(passNames[i]) {
					r.symptomExcerpt = symptomFailureExcerpt(again.output)
					return abi.WitnessRefuted, "candidate selected symptom test failed"
				}
			}
		}
		// A tolerated confirm is never cached: its proof depends on reruns that a later
		// run may not reproduce.
		if proofKey != "" && symptomConfirmCacheable(tol) {
			r.symptomProofCachePutConfirmed(ctx, proofKey)
		}
		return abi.WitnessConfirmed, symptomConfirmDetail(excluded, tol)
	}
	passed, buildErr, workspaceBroken := runParentTests(ctx, exec, parentDir, pkgs, pyTests, tags)
	if buildErr {
		// Parent failed to compile/build (e.g. test references an API introduced by the fix) —
		// this is unproven, never a false CONFIRM of behavioral reproduction (#12058).
		return abi.WitnessAbstain, "parent changed-test package did not build"
	}
	if workspaceBroken {
		// The committed go.work cannot load in this scratch topology (#13213): the parent
		// "failure" is a workspace-load error, not a red test — unproven, never a false CONFIRM.
		return abi.WitnessAbstain, "parent changed-test package could not load go.work"
	}
	if passed {
		// The test passes against the OLD source too: it constrains nothing about the bug.
		return abi.WitnessRefuted, "changed tests passed at parent"
	}
	return abi.WitnessConfirmed, "changed tests failed at parent and passed at candidate"
}

// vulkanSymptomCgoDisabled reports whether the selected Go toolchain's effective
// CGO setting is disabled, queried once via a bounded `go env CGO_ENABLED`
// (#13605 leaf K / #13611: an unset ambient variable is not the toolchain's
// effective setting; symptom_proof_cache already keys proofs on these same
// toolchain facts). A failed query reads as enabled, preserving the historical
// unset-env default: preparation proceeds and the cgo build itself fails loudly
// if cgo is truly unavailable.
func vulkanSymptomCgoDisabled(ctx context.Context) bool {
	qctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	out, code, err := commandRunner(qctx, "", "go", "env", "CGO_ENABLED")
	if err != nil || code != 0 {
		return false
	}
	return strings.TrimSpace(out) == "0"
}

func prepareVulkanSymptomScratch(ctx context.Context, repoRoot string, pkgs, tags []string, required bool) (map[string]string, bool, error) {
	if runtime.GOOS != "windows" && runtime.GOOS != "linux" {
		return nil, false, nil
	}
	if vulkanSymptomCgoDisabled(ctx) {
		return nil, false, nil
	}
	vulkan := false
	for _, tag := range tags {
		if tag == "vulkan" {
			vulkan = true
			break
		}
	}
	pkgDir := filepath.Join(repoRoot, "internal", "compute")
	if !vulkan || !computebuild.FileExists(filepath.Join(pkgDir, "vulkan_shim.cpp")) {
		return nil, false, nil
	}
	tc, err := computebuild.DiscoverToolchain()
	if err != nil {
		return nil, true, err
	}
	env := computebuild.SynthesizeVulkanCgoEnv(tc, pkgDir)
	env["FAK_VULKAN_SPIRV"] = ""
	if !required {
		argv := []string{"go", "list", "-test", "-deps", "-f={{if .CgoFiles}}{{.ImportPath}}{{end}}", "-tags", strings.Join(tags, ",")}
		argv = append(argv, pkgs...)
		out, code, err := commandRunnerEnv(env)(ctx, repoRoot, argv...)
		if err != nil || code != 0 {
			return nil, true, fmt.Errorf("resolving Vulkan cgo package closure: %s", capSymptomExcerpt(out))
		}
		for _, importPath := range strings.Fields(out) {
			if strings.HasSuffix(importPath, "/internal/compute") {
				required = true
				break
			}
		}
	}
	if !required {
		return nil, false, nil
	}
	if _, err := computebuild.BuildVulkanShim(ctx, tc, repoRoot, nil); err != nil {
		return nil, true, err
	}
	return env, true, nil
}

type goTestJSONEvent struct {
	Action  string `json:"Action"`
	Package string `json:"Package"`
	Test    string `json:"Test"`
	Output  string `json:"Output"`
}

// symptomTestState is the verdict go test -json gives one selected test. Only an
// explicit pass or fail event sets Passed or Failed; a started test with no final
// event is Unknown, so a sibling panic can never be read as green or red.
type symptomTestState uint8

const (
	symptomTestUnknown symptomTestState = iota
	symptomTestPassed
	symptomTestFailed
	symptomTestSkipped
)

type selectedGoTestResult struct {
	passed, matched, selectedFailed bool
	buildFailure, timedOut          bool
	workspaceBroken                 bool
	runErr                          error
	buildOutput                     string
	output                          string
	allExcluded                     bool
	// states and diag are indexed like the selectors passed to runSelectedGoTests; diag is
	// the first non-noise output line of each selected test.
	states []symptomTestState
	diag   []string
}

func stateOf(res selectedGoTestResult, i int) symptomTestState {
	if i < len(res.states) {
		return res.states[i]
	}
	return symptomTestUnknown
}

// namesByState returns the selected names (index-aligned with the selectors) whose
// state is one of want.
func namesByState(names []string, res selectedGoTestResult, want ...symptomTestState) []string {
	var out []string
	for i, name := range names {
		s := stateOf(res, i)
		for _, w := range want {
			if s == w {
				out = append(out, name)
				break
			}
		}
	}
	return out
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
	if symptomRunInterrupted(ctx, err) {
		// A mid-compile kill leaves partial output with no `run` event and no build-failure
		// marker; reading it would refute the witness as "matched no executed test".
		return selectedGoTestResult{timedOut: true, runErr: err}
	}
	if err != nil {
		return selectedGoTestResult{runErr: err}
	}
	seen := make([]bool, len(compiled))
	failed := make([]bool, len(compiled))
	passed := make([]bool, len(compiled))
	skipped := make([]bool, len(compiled))
	diag := make([]string, len(compiled))
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
				case "pass":
					passed[i] = true
				case "skip":
					skipped[i] = true
				case "output":
					if t := strings.TrimSpace(event.Output); t != "" && diag[i] == "" && !symptomExcerptNoise(t) {
						diag[i] = t
					}
				}
			}
		}
	}
	states := make([]symptomTestState, len(compiled))
	anyUnknownOrFailed := false
	for i := range compiled {
		switch {
		case failed[i]:
			states[i] = symptomTestFailed
		case passed[i]:
			states[i] = symptomTestPassed
		case skipped[i]:
			states[i] = symptomTestSkipped
		default:
			states[i] = symptomTestUnknown
		}
		if states[i] == symptomTestFailed || states[i] == symptomTestUnknown {
			anyUnknownOrFailed = true
		}
	}
	// A non-zero exit with every name green explicitly means the failure lies outside
	// the selected tests (TestMain, teardown): no per-test pass is trustworthy then.
	if code != 0 && !anyUnknownOrFailed {
		for i := range states {
			if states[i] == symptomTestPassed {
				states[i] = symptomTestUnknown
			}
		}
	}
	for _, matched := range seen {
		if !matched {
			result := selectedGoTestResult{
				passed: code == 0, buildFailure: code != 0 && isGoBuildFailure(out),
				workspaceBroken: code != 0 && isGoWorkspaceLoadFailure(out),
				timedOut:        strings.Contains(out, "panic: test timed out"),
				states:          states,
				diag:            diag,
			}
			if result.buildFailure {
				result.buildOutput = out
			}
			if code != 0 {
				result.output = out
			}
			return result
		}
	}
	selectedFailed := false
	for _, testFailed := range failed {
		if testFailed {
			selectedFailed = true
			break
		}
	}
	result := selectedGoTestResult{
		passed: code == 0, matched: true, selectedFailed: selectedFailed,
		workspaceBroken: code != 0 && isGoWorkspaceLoadFailure(out),
		timedOut:        strings.Contains(out, "panic: test timed out"),
		states:          states,
		diag:            diag,
	}
	if code != 0 {
		result.output = out
	}
	return result
}

// runParentSelection runs one selection against the parent. A changed test file
// that names a fix-introduced API cannot compile there, and because Go builds a
// test package as one unit it used to sink every sibling witness with it. Files
// predicted unbuildable, and files the compiler then blames, are restored to
// their parent version and their tests leave the parent run. Excluded tests are
// never counted as red (#12058); at least one remaining selected test must fail.
func (r *Resolver) runParentSelection(ctx context.Context, exec CommandRunner, parentDir, commit, parent string, selection symptomSelection, tests, tags []string, predicted map[string][]string) (selectedGoTestResult, int) {
	git := r.run
	if git == nil {
		git = gitRunner
	}
	pkgTests := symptomPackageTestFiles(tests, selection.Package)
	names := selection.Tests
	dropped := map[string]bool{}
	drop := func(files []string) bool {
		excluded := map[string]bool{}
		for _, rel := range files {
			if !restoreTestAtParent(ctx, git, r.dir, parent, parentDir, rel) {
				return false
			}
			dropped[rel] = true
			for name := range candidateTestNames(ctx, git, r.dir, commit, rel) {
				excluded[name] = true
			}
		}
		kept := make([]string, 0, len(names))
		for _, name := range names {
			if !excluded[name] {
				kept = append(kept, name)
			}
		}
		names = kept
		return len(names) > 0
	}
	var initial []string
	for _, rel := range pkgTests {
		if len(predicted[rel]) > 0 {
			initial = append(initial, rel)
		}
	}
	if len(initial) > 0 && !drop(initial) {
		return selectedGoTestResult{allExcluded: true}, len(dropped)
	}
	for {
		result := runSelectedGoTests(ctx, exec, parentDir, []string{selection.Package}, tags, exactTestSelectors(names))
		if !result.buildFailure || result.timedOut || result.runErr != nil {
			return result, len(dropped)
		}
		var blamed []string
		for _, rel := range pkgTests {
			if !dropped[rel] && compileErrorNamesFile(result.buildOutput, path.Base(rel)) {
				blamed = append(blamed, rel)
			}
		}
		if len(blamed) == 0 {
			return result, len(dropped)
		}
		if !drop(blamed) {
			return selectedGoTestResult{allExcluded: true}, len(dropped)
		}
	}
}

// candidateTestNames lists the top-level Test/Example names a changed test file
// declares at the candidate.
func candidateTestNames(ctx context.Context, git Runner, repoDir, commit, rel string) map[string]string {
	source, code, err := git(ctx, repoDir, "show", commit+":"+rel)
	if err != nil || code != 0 {
		return nil
	}
	funcs, err := topLevelGoTests(source)
	if err != nil {
		return nil
	}
	return funcs
}

// staticParentUnbuildable reports the fix-only names when every selected Go
// test lives in a changed file predicted not to compile at the parent, so the
// witness can abstain before paying for two scratch checkouts and compiles.
func staticParentUnbuildable(ctx context.Context, git Runner, repoDir, commit string, selections []symptomSelection, predicted map[string][]string) (string, bool) {
	if len(selections) == 0 || len(predicted) == 0 {
		return "", false
	}
	if git == nil {
		git = gitRunner
	}
	doomed := map[string]map[string]bool{}
	nameSet := map[string]bool{}
	for rel, names := range predicted {
		pkg := goTestPackage(rel)
		if doomed[pkg] == nil {
			doomed[pkg] = map[string]bool{}
		}
		for test := range candidateTestNames(ctx, git, repoDir, commit, rel) {
			doomed[pkg][test] = true
		}
		for _, name := range names {
			nameSet[name] = true
		}
	}
	for _, selection := range selections {
		for _, test := range selection.Tests {
			if !doomed[selection.Package][test] {
				return "", false
			}
		}
	}
	names := make([]string, 0, len(nameSet))
	for name := range nameSet {
		names = append(names, name)
	}
	sort.Strings(names)
	if len(names) > 8 {
		names = append(names[:8], "...")
	}
	return "parent selected symptom test did not build: every selected test names fix-introduced " + strings.Join(names, ", ") + "; add a regression that drives the bug through API the parent already has", true
}

// compileErrorNamesFile reports whether go build output carries a compiler
// position (`name:line:`) for the file with this base name.
func compileErrorNamesFile(out, base string) bool {
	return regexp.MustCompile(`(?:^|[^\w.-])` + regexp.QuoteMeta(base) + `:\d+:`).MatchString(out)
}

// restoreTestAtParent puts a changed test file back to its parent content, or
// removes it when the parent has no such file.
func restoreTestAtParent(ctx context.Context, git Runner, repoDir, parent, parentDir, rel string) bool {
	dst := filepath.Join(parentDir, filepath.FromSlash(rel))
	content, code, err := git(ctx, repoDir, "show", parent+":"+rel)
	if err != nil {
		return false
	}
	if code != 0 {
		err := os.Remove(dst)
		return err == nil || errors.Is(err, os.ErrNotExist)
	}
	return os.WriteFile(dst, []byte(content), 0o644) == nil
}

// symptomTolerance records the candidate reds the witness tolerated, each backed by
// explicit go test -json evidence; a non-empty tolerance is never cached.
type symptomTolerance struct {
	preexisting []string
}

func (t symptomTolerance) empty() bool {
	return len(t.preexisting) == 0
}

func symptomConfirmCacheable(tol symptomTolerance) bool {
	return tol.empty()
}

// diagFor returns the first diagnostic of name within res, whose selectors are names.
func diagFor(names []string, res selectedGoTestResult, name string) string {
	for j, n := range names {
		if n == name && j < len(res.diag) {
			return res.diag[j]
		}
	}
	return ""
}

// diagnosticsMatch requires both sides to print the same, non-empty first diagnostic.
func diagnosticsMatch(a, b string) bool {
	return a != "" && a == b
}

// symptomConfirmDetail keeps today's CONFIRM strings byte-identical when nothing was
// excluded or tolerated.
func symptomConfirmDetail(excluded int, tol symptomTolerance) string {
	detail := "selected symptom test failed at parent and passed at candidate"
	if excluded > 0 {
		detail += fmt.Sprintf(" (%d parent-unbuildable changed test file(s) excluded at parent)", excluded)
	}
	if tol.empty() {
		return detail
	}
	var parts []string
	if len(tol.preexisting) > 0 {
		parts = append(parts, "pre-existing red at parent "+symptomNameList(tol.preexisting))
	}
	return detail + " (tolerated at candidate: " + strings.Join(parts, "; ") + ")"
}

// symptomNameList caps long lists the way staticParentUnbuildable does, so a detail
// stays readable for an operator.
func symptomNameList(names []string) string {
	sorted := append([]string(nil), names...)
	sort.Strings(sorted)
	if len(sorted) > 8 {
		sorted = append(sorted[:8], "...")
	}
	return strings.Join(sorted, ", ")
}

// symptomPackageTestFiles returns the changed _test.go files that belong to pkg.
func symptomPackageTestFiles(tests []string, pkg string) []string {
	var out []string
	for _, rel := range tests {
		rel = strings.ReplaceAll(strings.TrimSpace(rel), "\\", "/")
		if strings.HasSuffix(rel, "_test.go") && goTestPackage(rel) == pkg {
			out = append(out, rel)
		}
	}
	return out
}

// parentDeclaresTests reports, per name, whether the parent declares it. A name no
// changed candidate file declares lives in an unchanged file, so it is declared at
// the parent too.
func parentDeclaresTests(ctx context.Context, git Runner, repoDir, commit, parent string, pkgTests, names []string) map[string]bool {
	atCandidate := map[string]bool{}
	atParent := map[string]bool{}
	for _, rel := range pkgTests {
		if source, code, err := git(ctx, repoDir, "show", commit+":"+rel); err == nil && code == 0 {
			funcs, err := topLevelGoTests(source)
			if err != nil {
				// A candidate file we cannot read may declare any name, so none is
				// provably declared at the parent: every red name refutes.
				return make(map[string]bool, len(names))
			}
			for name := range funcs {
				atCandidate[name] = true
			}
		}
		source, code, err := git(ctx, repoDir, "show", parent+":"+rel)
		if err != nil || code != 0 {
			continue
		}
		if funcs, err := topLevelGoTests(source); err == nil {
			for name := range funcs {
				atParent[name] = true
			}
		}
	}
	declared := make(map[string]bool, len(names))
	for _, name := range names {
		declared[name] = atParent[name] || !atCandidate[name]
	}
	return declared
}

// runParentOwnTests runs names at the parent's own version of the package's changed
// test files, not the candidate overlay. The whole package set is restored so the
// test package compiles as the parent has it; the candidate overlay is put back
// afterwards for the proof run. ok=false means the overlay could not be restored.
func (r *Resolver) runParentOwnTests(ctx context.Context, exec CommandRunner, parentDir, commit, parent, pkg string, names, tests, tags []string) (selectedGoTestResult, bool) {
	git := r.run
	if git == nil {
		git = gitRunner
	}
	pkgFiles := symptomPackageTestFiles(tests, pkg)
	var res selectedGoTestResult
	restored := true
	for _, rel := range pkgFiles {
		if !restoreTestAtParent(ctx, git, r.dir, parent, parentDir, rel) {
			restored = false
			break
		}
	}
	if restored {
		res = runSelectedGoTests(ctx, exec, parentDir, []string{pkg}, tags, exactTestSelectors(names))
	}
	if !overlayTestsAtRef(ctx, git, r.dir, commit, parentDir, pkgFiles) {
		return selectedGoTestResult{}, false
	}
	return res, true
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

// goTestOutcome runs `go test` for the given packages in dir and reports whether it exited 0,
// along with the full output so callers can classify a non-zero exit: a real red test versus
// a committed go.work that cannot load in this scratch topology (#13213).
func goTestOutcome(ctx context.Context, run CommandRunner, dir string, pkgs, tags []string) (passed bool, out string) {
	if run == nil {
		run = commandRunner
	}
	out, code, err := run(ctx, dir, goTestArgv(pkgs, tags)...)
	return err == nil && code == 0, out
}

// goTestPasses runs `go test` for the given packages in dir and reports whether it exited 0.
// tags are the build constraints derived from the changed test files (#13243): a test gated
// behind `//go:build vulkan` is excluded from a bare `go test`, so the execution rung would
// see a trivially-passing package at BOTH refs and falsely REFUTE a genuine device witness.
// A nil/empty tag set keeps the argv byte-identical to the pre-#13243 untagged form.
func goTestPasses(ctx context.Context, run CommandRunner, dir string, pkgs []string, tags []string) bool {
	passed, _ := goTestOutcome(ctx, run, dir, pkgs, tags)
	return passed
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

func runParentTests(ctx context.Context, exec CommandRunner, dir string, pkgs, pyTests, tags []string) (passed bool, buildErr bool, workspaceBroken bool) {
	goPassed := true
	if len(pkgs) > 0 {
		var err, broken bool
		goPassed, err, broken = runGoTests(ctx, exec, dir, pkgs, tags)
		if err {
			return false, true, false
		}
		if broken {
			return false, false, true
		}
	}
	pyPassed := true
	if len(pyTests) > 0 {
		var err bool
		pyPassed, err = runPythonTests(ctx, exec, dir, pyTests)
		if err {
			return false, true, false
		}
	}
	if goPassed && pyPassed {
		return true, false, false
	}
	return false, false, false
}

func runGoTests(ctx context.Context, run CommandRunner, dir string, pkgs, tags []string) (passed bool, buildErr bool, workspaceBroken bool) {
	if run == nil {
		run = commandRunner
	}
	out, code, err := run(ctx, dir, goTestArgv(pkgs, tags)...)
	if err != nil {
		return false, true, false
	}
	if code == 0 {
		return true, false, false
	}
	if isGoBuildFailure(out) {
		return false, true, false
	}
	if isGoWorkspaceLoadFailure(out) {
		return false, false, true
	}
	return false, false, false
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

// isGoWorkspaceLoadFailure reports whether a non-zero `go test` exit is a go.work load
// failure rather than a build or test failure: the committed workspace lists a module
// that does not resolve in this scratch topology (#13213). It is kept separate from
// isGoBuildFailure on purpose — folding it in would still refuse the candidate with
// "did not build" for a package that never got to building, and the receipt would
// describe a step that never happened.
func isGoWorkspaceLoadFailure(out string) bool {
	return strings.Contains(out, "listed in go.work file")
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
