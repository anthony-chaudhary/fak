package safecommit

import (
	"context"
	"errors"
	"strings"
)

// buildcheck.go — what `fak commit`'s COMMITTED_RED prospective-tree compile gate DID, as a
// reported value rather than a bool (#6006).
//
// The gate itself lives in cmd/fak: it needs git and the go toolchain to materialize the
// prospective committed tree and build it. What lives here is the part a caller has to be able
// to trust — the closed vocabulary for the gate's outcome, the admit/refuse verdict over it,
// and that verdict's exit class.
//
// The hole this closes: the gate failed OPEN when its `git archive` step hit the 2m deadline,
// announced the skip on stderr only, and admitted the commit. The commit then landed with a
// --json result that carried NO build-check field at all — byte-identical to a commit whose
// gate had actually compiled the prospective tree and passed, and graded A either way. A gate
// whose result cannot be told apart from a gate that never ran is not a gate. Worse, the
// timeout is correlated with exactly the conditions where a red commit costs most (a loaded
// host, several fleet workers committing into one shared trunk at once).
//
// Contract:
//   - EVERY outcome is reported. Compiled says plainly whether the prospective committed tree
//     was really built, so `passed` can never be confused with `skipped-timeout`.
//   - A timeout is not an infra shrug. An absent git/go toolchain is a static property of the
//     host and still fails open; a deadline means the check COULD have run and we stopped
//     waiting, so it refuses unless the caller opted into fail-open (allowTimeout).
//   - Admitting without a compile is recorded as FailedOpen, and that docks the commit's score
//     (score.go) — an unchecked commit is no longer graded like a checked one.

// BuildCheckOutcome is the closed vocabulary for what the compile gate did. It is a wire value
// (Result.BuildCheck.Outcome in --json), so the strings are part of the contract.
type BuildCheckOutcome string

const (
	// BuildCheckNotApplicable: the commit touches no .go file, or the packages it would build
	// no longer exist in the prospective tree. Nothing to compile, so nothing was skipped.
	BuildCheckNotApplicable BuildCheckOutcome = "not-applicable"
	// BuildCheckDisabled: --no-build-check or FAK_COMMIT_BUILD_CHECK=off. The operator asked
	// for no gate; this is the ONLY invisible-by-request state, and it is still reported.
	BuildCheckDisabled BuildCheckOutcome = "disabled"
	// BuildCheckPassed: the prospective committed tree was compiled and is green.
	BuildCheckPassed BuildCheckOutcome = "passed"
	// BuildCheckFailed: it was compiled and is red, and HEAD is not red the same way — THIS
	// commit introduces the break.
	BuildCheckFailed BuildCheckOutcome = "failed"
	// BuildCheckHeadRed: it was compiled and is red, but HEAD's committed bytes are red too.
	// The red is pre-existing, so the commit is admitted rather than wedged behind a peer.
	BuildCheckHeadRed BuildCheckOutcome = "head-red"
	// BuildCheckSkippedTimeout: the archive/extract deadline expired. The tree was NEVER
	// compiled — this is the state that used to be indistinguishable from "passed".
	BuildCheckSkippedTimeout BuildCheckOutcome = "skipped-timeout"
	// BuildCheckSkippedInfra: git or the go toolchain is unavailable or errored. Also never
	// compiled, but for a reason retrying will not clear, so it keeps failing open.
	BuildCheckSkippedInfra BuildCheckOutcome = "skipped-infra"
)

// Pre-commit gate reasons. These are NOT part of RefusalReasons(): that vocabulary is the
// executor's commit-STATE refusals, reported after staging, whereas the compile gate refuses
// before any git effect and owns its own exit-code mapping (BuildCheckExitCode).
const (
	ReasonCommittedRed      = "COMMITTED_RED"       // this commit would red the committed trunk
	ReasonBuildCheckTimeout = "BUILD_CHECK_TIMEOUT" // the gate could not finish; nobody opted into fail-open
)

// Compiled reports whether the prospective committed tree was actually built under this
// outcome. It is the single question a caller has to be able to answer, and the reason a
// timed-out gate can never masquerade as a passed one.
func (o BuildCheckOutcome) Compiled() bool {
	switch o {
	case BuildCheckPassed, BuildCheckFailed, BuildCheckHeadRed:
		return true
	case BuildCheckNotApplicable, BuildCheckDisabled, BuildCheckSkippedTimeout, BuildCheckSkippedInfra:
		return false
	default:
		return false
	}
}

// BuildCheckResult is the gate's outcome as it appears on Result.BuildCheck in --json.
type BuildCheckResult struct {
	Outcome BuildCheckOutcome `json:"outcome"`
	// Compiled mirrors Outcome.Compiled() so a caller that does not know the vocabulary (an
	// older parser, a shell reading `.build_check.compiled`) still cannot mistake a skip for a
	// pass.
	Compiled bool `json:"compiled"`
	// FailedOpen is true when the commit was admitted although the tree was never compiled.
	// It is the audit field: "this commit is unchecked, and here is why it was allowed".
	FailedOpen bool `json:"failed_open,omitempty"`
	// Detail carries the compiler transcript (red outcomes) or the error that stopped the gate
	// (skips), so the skip is diagnosable from the JSON alone rather than from stderr.
	Detail string `json:"detail,omitempty"`
}

// ClassifyBuildCheckError maps the error that aborted the gate onto the outcome it must
// report: an expired deadline is skipped-timeout, everything else (no `go` on PATH, a git
// failure, an unwritable temp dir) is skipped-infra. The string fallback catches an error that
// only *describes* a timeout after crossing a process or a %v boundary that dropped the wrapped
// context.DeadlineExceeded.
func ClassifyBuildCheckError(err error) BuildCheckOutcome {
	if err == nil {
		return BuildCheckSkippedInfra
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return BuildCheckSkippedTimeout
	}
	if msg := strings.ToLower(err.Error()); strings.Contains(msg, "timed out") ||
		strings.Contains(msg, "context deadline exceeded") ||
		strings.Contains(msg, "timeout") {
		return BuildCheckSkippedTimeout
	}
	return BuildCheckSkippedInfra
}

// DecideBuildCheck folds a raw gate outcome into the result a caller sees and the verdict the
// commit obeys. allowTimeout is the caller's EXPLICIT opt-in to fail open on a timed-out gate
// (--allow-build-check-timeout / FAK_COMMIT_BUILD_CHECK=allow-timeout); without it a timeout
// refuses, because "we stopped waiting for the check" is not evidence the tree compiles.
//
// admit=false is always accompanied by a reason from the pair above; admit=true never carries
// one. detail is passed through onto the result so the JSON is self-diagnosing.
func DecideBuildCheck(outcome BuildCheckOutcome, detail string, allowTimeout bool) (res BuildCheckResult, admit bool, reason string) {
	res = BuildCheckResult{Outcome: outcome, Compiled: outcome.Compiled(), Detail: strings.TrimSpace(detail)}
	switch outcome {
	case BuildCheckFailed:
		return res, false, ReasonCommittedRed
	case BuildCheckSkippedTimeout:
		if !allowTimeout {
			return res, false, ReasonBuildCheckTimeout
		}
		res.FailedOpen = true
		return res, true, ""
	case BuildCheckSkippedInfra:
		// Cannot check, and retrying cannot make the host grow a toolchain: admit, but say so.
		res.FailedOpen = true
		return res, true, ""
	default:
		// passed / head-red / not-applicable / disabled. head-red is admitted although the tree
		// is red, but it WAS compiled and the red is provably not this commit's. not-applicable
		// had nothing to compile. disabled is the operator's own up-front choice, already named
		// on the wire as "disabled" — FailedOpen is reserved for the case this issue is about:
		// the tooling admitting a commit it could not check, on its own initiative.
		return res, true, ""
	}
}

// BuildCheckExitCode classifies a gate refusal into its process exit code, mirroring
// RefusalExitCode's split for the executor's own vocabulary. ok is false for anything outside
// the two gate reasons.
//
// A timeout is CONTENTION (ExitLockBusy, 3): the archive lost a race with a loaded host, so the
// same command may well succeed next tick and the fleet's landers already back off on 3. An
// introduced red is a VERDICT (ExitRefused, 4): recompiling the identical tree yields the
// identical red, so a lander must fix the build instead of retrying.
func BuildCheckExitCode(reason string) (code int, ok bool) {
	switch reason {
	case ReasonBuildCheckTimeout:
		return ExitLockBusy, true
	case ReasonCommittedRed:
		return ExitRefused, true
	}
	return 0, false
}

// BuildCheckScoreNote renders the score note for a commit admitted without a compile check,
// or "" when the gate's outcome does not warrant one. Kept next to the vocabulary so a new
// outcome cannot be added without deciding how it reads on a scorecard.
func BuildCheckScoreNote(res BuildCheckResult) string {
	if !res.FailedOpen {
		return ""
	}
	switch res.Outcome {
	case BuildCheckSkippedTimeout:
		return "build-check TIMED OUT and was failed open by request: the prospective committed tree was never compiled"
	default:
		return "build-check could not run (" + string(res.Outcome) + "): the prospective committed tree was never compiled"
	}
}
