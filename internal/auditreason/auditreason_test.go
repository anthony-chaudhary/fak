package auditreason

import (
	"errors"
	"strings"
	"testing"
)

// TestClassify pins the transient-vs-permanent contract for every closed
// reason: the four transient reasons are retry-eligible, every permanent reason
// (including the ReasonUnknown default) is not.
func TestClassify(t *testing.T) {
	cases := []struct {
		reason    Reason
		wantClass Class
		wantRetry bool
	}{
		// Transient — retry-eligible.
		{ReasonLockBusy, Transient, true},
		{ReasonPushRejectedRace, Transient, true},
		{ReasonRemoteUnreachable, Transient, true},
		{ReasonRebaseInFlight, Transient, true},

		// Permanent — genuine unwitnessed claim, not retryable.
		{ReasonNoCommitFound, Permanent, false},
		{ReasonSubjectDiffMismatch, Permanent, false},
		{ReasonNoWitnessArtifact, Permanent, false},

		// Conservative default.
		{ReasonUnknown, Permanent, false},
	}
	for _, c := range cases {
		if got := Classify(c.reason); got != c.wantClass {
			t.Errorf("Classify(%q) = %q, want %q", c.reason, got, c.wantClass)
		}
		if got := RetryEligible(c.reason); got != c.wantRetry {
			t.Errorf("RetryEligible(%q) = %v, want %v", c.reason, got, c.wantRetry)
		}
	}
}

// TestClassifyContract asserts the invariant that ties the two classes to
// retry eligibility, so a future reason cannot be transient-yet-non-retryable
// or permanent-yet-retryable.
func TestClassifyContract(t *testing.T) {
	all := []Reason{
		ReasonUnknown,
		ReasonLockBusy, ReasonPushRejectedRace, ReasonRemoteUnreachable, ReasonRebaseInFlight,
		ReasonNoCommitFound, ReasonSubjectDiffMismatch, ReasonNoWitnessArtifact,
	}
	for _, r := range all {
		transient := Classify(r) == Transient
		if transient != RetryEligible(r) {
			t.Errorf("reason %q: Classify transient=%v but RetryEligible=%v (must agree)", r, transient, RetryEligible(r))
		}
	}
}

// TestFromMessage maps representative raw audit-failure strings to the right
// closed reason, covering each transient and permanent bucket, and confirms an
// unrecognized message defaults to the conservative non-retryable unknown.
func TestFromMessage(t *testing.T) {
	cases := []struct {
		name string
		msg  string
		want Reason
	}{
		// Transient signatures.
		{"index lock", "fatal: Unable to create '.git/index.lock': File exists.", ReasonLockBusy},
		{"packed-refs lock", "Unable to create 'C:/work/fak/.git/packed-refs.lock': File exists", ReasonLockBusy},
		{"another git process", "Another git process seems to be running in this repository", ReasonLockBusy},
		{"non-fast-forward", "! [rejected] main -> main (non-fast-forward)", ReasonPushRejectedRace},
		{"fetch first", "error: failed to push some refs; hint: fetch first", ReasonPushRejectedRace},
		{"updates rejected", "Updates were rejected because the tip of your current branch is behind", ReasonPushRejectedRace},
		{"could not resolve host", "fatal: unable to access 'https://...': Could not resolve host: github.com", ReasonRemoteUnreachable},
		{"connection refused", "fatal: connection refused", ReasonRemoteUnreachable},
		{"rebase in progress", "fatal: It seems that there is already a rebase in progress", ReasonRebaseInFlight},
		{"merge in progress", "error: you have not concluded your merge; merge in progress", ReasonRebaseInFlight},

		// Permanent signatures.
		{"no commit found", "commit-audit: no commit found referencing issue #1809", ReasonNoCommitFound},
		{"not shipped", "dos verify: NOT_SHIPPED (no resolving commit)", ReasonNoCommitFound},
		{"subject mismatch", "subject claims fix(x) but the diff does not witness it", ReasonSubjectDiffMismatch},
		{"unwitnessed", "closure refused: claim is unwitnessed", ReasonNoWitnessArtifact},
		{"missing witness", "audit: missing witness artifact for the claim", ReasonNoWitnessArtifact},

		// Canonical token round-trips (each Reason string maps back to itself).
		{"canonical lock", "lock_busy", ReasonLockBusy},
		{"canonical race", "push_rejected_race", ReasonPushRejectedRace},
		{"canonical no-commit", "no_commit_found", ReasonNoCommitFound},

		// Unknown — conservative default.
		{"empty", "", ReasonUnknown},
		{"unrelated", "the build succeeded and everything is fine", ReasonUnknown},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := FromMessage(c.msg); got != c.want {
				t.Errorf("FromMessage(%q) = %q, want %q", c.msg, got, c.want)
			}
		})
	}
}

// TestUnknownIsConservativelyNonRetryable is the load-bearing safety property:
// an unrecognized failure must NOT be masked as retryable, or a genuine miss
// would be silently deferred forever.
func TestUnknownIsConservativelyNonRetryable(t *testing.T) {
	r := FromMessage("some brand-new failure mode we have never seen")
	if r != ReasonUnknown {
		t.Fatalf("unrecognized message mapped to %q, want ReasonUnknown", r)
	}
	if RetryEligible(r) {
		t.Fatal("ReasonUnknown must be NON-retryable so a genuine miss is not hidden as retry-later")
	}
	if Classify(r) != Permanent {
		t.Fatalf("ReasonUnknown must classify Permanent, got %q", Classify(r))
	}
}

// TestTransientMatchedBeforePermanent proves the ordering guarantee: a message
// that mentions both a passing tree condition (a lock) and a miss is treated as
// the retryable transient failure, not prematurely declared drift.
func TestTransientMatchedBeforePermanent(t *testing.T) {
	msg := "Unable to create '.git/index.lock': File exists; also no commit found"
	if got := FromMessage(msg); got != ReasonLockBusy {
		t.Errorf("mixed message = %q, want ReasonLockBusy (transient wins)", got)
	}
}

// TestFromError covers the error-typed wrapper, including the nil case.
func TestFromError(t *testing.T) {
	if got := FromError(nil); got != ReasonUnknown {
		t.Errorf("FromError(nil) = %q, want ReasonUnknown", got)
	}
	err := errors.New("! [rejected] main (non-fast-forward)")
	if got := FromError(err); got != ReasonPushRejectedRace {
		t.Errorf("FromError(race) = %q, want ReasonPushRejectedRace", got)
	}
}

// TestExamine confirms the Report fields are derived consistently from the
// message for both a transient and a permanent case.
func TestExamine(t *testing.T) {
	tr := Examine("fatal: Unable to create '.git/index.lock': File exists")
	if tr.Reason != ReasonLockBusy || tr.Class != Transient || !tr.RetryEligible {
		t.Errorf("transient Examine = %+v", tr)
	}
	if tr.Detail == "" {
		t.Error("Detail should carry the raw message")
	}

	pm := Examine("commit-audit: no commit found for the issue")
	if pm.Reason != ReasonNoCommitFound || pm.Class != Permanent || pm.RetryEligible {
		t.Errorf("permanent Examine = %+v", pm)
	}
}

// TestExamineError covers the error-typed Report wrapper, including nil.
func TestExamineError(t *testing.T) {
	if got := ExamineError(nil); got.Reason != ReasonUnknown || got.RetryEligible {
		t.Errorf("ExamineError(nil) = %+v, want unknown/non-retryable", got)
	}
	got := ExamineError(errors.New("subject claims a fix the diff does not witness"))
	if got.Reason != ReasonSubjectDiffMismatch || got.Class != Permanent || got.RetryEligible {
		t.Errorf("ExamineError(mismatch) = %+v", got)
	}
}

// TestReportString proves the rendered line names the class and retry
// eligibility for both classes, and includes the detail when present.
func TestReportString(t *testing.T) {
	tr := Examine("Another git process seems to be running")
	s := tr.String()
	if !strings.Contains(s, string(ReasonLockBusy)) {
		t.Errorf("String() %q missing reason", s)
	}
	if !strings.Contains(s, string(Transient)) {
		t.Errorf("String() %q missing class", s)
	}
	if !strings.Contains(s, "RETRYABLE") || strings.Contains(s, "NOT-RETRYABLE") {
		t.Errorf("String() %q should say RETRYABLE for a transient failure", s)
	}
	if !strings.Contains(s, "detail=") {
		t.Errorf("String() %q should carry the detail", s)
	}

	pm := Examine("no commit found for issue")
	ps := pm.String()
	if !strings.Contains(ps, "NOT-RETRYABLE") {
		t.Errorf("String() %q should say NOT-RETRYABLE for a permanent failure", ps)
	}
	if !strings.Contains(ps, string(Permanent)) {
		t.Errorf("String() %q missing permanent class", ps)
	}

	// A report with no detail omits the detail token.
	empty := Report{Reason: ReasonUnknown, Class: Permanent, RetryEligible: false}
	if strings.Contains(empty.String(), "detail=") {
		t.Errorf("String() %q should omit empty detail", empty.String())
	}
}
