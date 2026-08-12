package safecommit

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
)

// buildcheck_test.go — the #6006 witness. The headline property is NEGATIVE: a build check
// that TIMED OUT must not be able to produce a result indistinguishable from a build check
// that ran and passed. Everything else here guards the pieces that property rests on.

// TestTimedOutGateIsDistinguishableFromPassed is the regression the issue names. It compares
// the two results the way a --json consumer actually does: over the serialized bytes, over the
// outcome token, and over the one boolean a naive parser would read.
func TestTimedOutGateIsDistinguishableFromPassed(t *testing.T) {
	timedOut, admit, reason := DecideBuildCheck(BuildCheckSkippedTimeout, "archive timed out after 2m0s", true /* operator opted into fail-open */)
	if !admit || reason != "" {
		t.Fatalf("explicit opt-in must admit; got admit=%v reason=%q", admit, reason)
	}
	passed, _, _ := DecideBuildCheck(BuildCheckPassed, "", true)

	if timedOut == passed {
		t.Fatal("a timed-out gate produced a result equal to a passed gate: the skip is invisible again")
	}
	if timedOut.Compiled {
		t.Fatal("a timed-out gate must not report the prospective tree as compiled")
	}
	if !passed.Compiled {
		t.Fatal("a passed gate must report the prospective tree as compiled")
	}
	if !timedOut.FailedOpen {
		t.Fatal("admitting a never-compiled tree must be recorded as fail-open")
	}
	if passed.FailedOpen {
		t.Fatal("a gate that ran and passed did not fail open")
	}

	// The wire form is the contract a fleet worker parses; the states must not collide there
	// either, and the timeout must be named rather than merely absent.
	a := mustJSON(t, timedOut)
	b := mustJSON(t, passed)
	if a == b {
		t.Fatalf("serialized results collide: %s", a)
	}
	if !strings.Contains(a, string(BuildCheckSkippedTimeout)) {
		t.Fatalf("timed-out result does not name its outcome on the wire: %s", a)
	}
}

// TestTimedOutGateRefusesWithoutOptIn pins the second half of the done condition: the fail-open
// is no longer the default. Without the caller's explicit opt-in a timeout is a refusal, and a
// retryable one — the archive lost a race, not an argument.
func TestTimedOutGateRefusesWithoutOptIn(t *testing.T) {
	res, admit, reason := DecideBuildCheck(BuildCheckSkippedTimeout, "archive timed out after 2m0s", false)
	if admit {
		t.Fatal("a timed-out gate must refuse unless the caller opted into fail-open")
	}
	if reason != ReasonBuildCheckTimeout {
		t.Fatalf("reason = %q, want %q", reason, ReasonBuildCheckTimeout)
	}
	if res.FailedOpen {
		t.Fatal("a refused commit did not fail open")
	}
	if res.Detail == "" {
		t.Fatal("the refusal must carry the timeout detail so --json is self-diagnosing")
	}
	code, ok := BuildCheckExitCode(reason)
	if !ok || code != ExitLockBusy {
		t.Fatalf("BuildCheckExitCode(%q) = (%d, %v), want (%d, true): a timeout is retryable contention",
			reason, code, ok, ExitLockBusy)
	}
}

// TestBuildCheckVerdicts covers the rest of the vocabulary in one table: what admits, what
// refuses, and which admissions are fail-open (an admitted commit whose tree was never built).
func TestBuildCheckVerdicts(t *testing.T) {
	cases := []struct {
		outcome    BuildCheckOutcome
		allow      bool
		wantAdmit  bool
		wantReason string
		wantOpen   bool
		wantBuilt  bool
	}{
		{BuildCheckPassed, false, true, "", false, true},
		{BuildCheckFailed, false, false, ReasonCommittedRed, false, true},
		{BuildCheckFailed, true, false, ReasonCommittedRed, false, true},
		{BuildCheckHeadRed, false, true, "", false, true},
		{BuildCheckSkippedTimeout, false, false, ReasonBuildCheckTimeout, false, false},
		{BuildCheckSkippedTimeout, true, true, "", true, false},
		{BuildCheckSkippedInfra, false, true, "", true, false},
		{BuildCheckNotApplicable, false, true, "", false, false},
		{BuildCheckDisabled, false, true, "", false, false},
	}
	for _, tc := range cases {
		t.Run(string(tc.outcome)+fmt.Sprintf("/allow=%v", tc.allow), func(t *testing.T) {
			res, admit, reason := DecideBuildCheck(tc.outcome, "detail", tc.allow)
			if admit != tc.wantAdmit || reason != tc.wantReason {
				t.Fatalf("DecideBuildCheck(%q, allow=%v) = (admit=%v, reason=%q), want (%v, %q)",
					tc.outcome, tc.allow, admit, reason, tc.wantAdmit, tc.wantReason)
			}
			if res.FailedOpen != tc.wantOpen {
				t.Fatalf("FailedOpen = %v, want %v", res.FailedOpen, tc.wantOpen)
			}
			if res.Compiled != tc.wantBuilt {
				t.Fatalf("Compiled = %v, want %v", res.Compiled, tc.wantBuilt)
			}
			if !admit && reason != "" {
				if _, ok := BuildCheckExitCode(reason); !ok {
					t.Fatalf("refusal reason %q has no exit-code class", reason)
				}
			}
			if admit && reason != "" {
				t.Fatalf("an admitted commit carried refusal reason %q", reason)
			}
		})
	}
	// COMMITTED_RED stays a verdict: recompiling the identical tree yields the identical red.
	if code, ok := BuildCheckExitCode(ReasonCommittedRed); !ok || code != ExitRefused {
		t.Fatalf("BuildCheckExitCode(COMMITTED_RED) = (%d, %v), want (%d, true)", code, ok, ExitRefused)
	}
	if _, ok := BuildCheckExitCode("NOT_A_GATE_REASON"); ok {
		t.Fatal("BuildCheckExitCode must not classify a reason outside its own pair")
	}
}

// TestClassifyBuildCheckError separates the two ways the gate can fail to produce an answer.
// The distinction is load-bearing: infra keeps failing open (retrying cannot make the host grow
// a toolchain), a deadline does not (it is exactly the loaded-trunk case).
func TestClassifyBuildCheckError(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want BuildCheckOutcome
	}{
		{"wrapped deadline", fmt.Errorf("git archive timed out after 2m0s: %w", context.DeadlineExceeded), BuildCheckSkippedTimeout},
		{"bare deadline", context.DeadlineExceeded, BuildCheckSkippedTimeout},
		{"prose only, wrapping lost", errors.New("archive timed out after 2m0s (failing open)"), BuildCheckSkippedTimeout},
		{"context deadline prose", errors.New("context deadline exceeded"), BuildCheckSkippedTimeout},
		{"toolchain missing", errors.New(`exec: "go": executable file not found in $PATH`), BuildCheckSkippedInfra},
		{"git failure", errors.New("git write-tree: exit status 128"), BuildCheckSkippedInfra},
		{"nil", nil, BuildCheckSkippedInfra},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ClassifyBuildCheckError(tc.err); got != tc.want {
				t.Fatalf("ClassifyBuildCheckError(%v) = %q, want %q", tc.err, got, tc.want)
			}
		})
	}
}

// TestScoreDocksAnUncheckedCommit pins the grading half of the complaint: the observed timeout
// landed a commit that was still graded A. A verified commit whose gate never compiled the tree
// must score below one whose gate passed, and must SAY why.
func TestScoreDocksAnUncheckedCommit(t *testing.T) {
	base := Result{Committed: true, Verified: true, SHA: "deadbeef"}

	checked := ScoreResult(withBuildCheck(base, BuildCheckPassed, true))
	if checked.Score != 100 {
		t.Fatalf("a gate-passed verified commit should keep full credit; got %d %v", checked.Score, checked.ScoreNotes)
	}
	unchecked := ScoreResult(withBuildCheck(base, BuildCheckSkippedTimeout, true))
	if unchecked.Score >= checked.Score {
		t.Fatalf("unchecked commit scored %d, want less than the checked commit's %d", unchecked.Score, checked.Score)
	}
	if len(unchecked.ScoreNotes) == 0 || !strings.Contains(strings.Join(unchecked.ScoreNotes, " "), "never compiled") {
		t.Fatalf("the score must name the missing check; got %v", unchecked.ScoreNotes)
	}
	if unchecked.Grade == checked.Grade {
		t.Fatalf("unchecked and checked commits share grade %q", unchecked.Grade)
	}
	// A caller that attaches no gate outcome at all is untouched by this change.
	if plain := ScoreResult(base); plain.Score != 100 || len(plain.ScoreNotes) != 0 {
		t.Fatalf("absent build-check must not change legacy scoring; got %d %v", plain.Score, plain.ScoreNotes)
	}
}

// withBuildCheck attaches the gate outcome to a copy of res the way cmd/fak does.
func withBuildCheck(res Result, outcome BuildCheckOutcome, allowTimeout bool) Result {
	bc, _, _ := DecideBuildCheck(outcome, "", allowTimeout)
	res.BuildCheck = &bc
	return res
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(b)
}
