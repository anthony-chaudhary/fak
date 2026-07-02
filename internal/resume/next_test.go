package resume

import (
	"strings"
	"testing"
)

// TestFoldNextActionRun: a pending session, gate open, no wall-clock limit, host admits →
// run (fire=true).
func TestFoldNextActionRun(t *testing.T) {
	v := FoldNextAction(NextInput{
		State:    ResumePending,
		Outcome:  OutcomeUnknown,
		Retry:    RetryDecision{Blocked: false, Reason: "first resume"},
		Admitted: true,
	})
	if v.Action != ActRun || !v.Fire {
		t.Fatalf("action = %q fire=%v, want run/true (%s)", v.Action, v.Fire, v.Reason)
	}
}

// TestFoldNextActionWaitReset: a pending session that crashed on a session cap and has been
// idle LESS than the 5h reset window → wait_reset, not run.
func TestFoldNextActionWaitReset(t *testing.T) {
	v := FoldNextAction(NextInput{
		State:       ResumePending,
		Retry:       RetryDecision{Blocked: false},
		LimitReason: LimitSession,
		IdleSeconds: 3600, // 1h < 5h
		Admitted:    true,
	})
	if v.Action != ActWaitReset || v.Fire {
		t.Fatalf("action = %q fire=%v, want wait_reset/false (%s)", v.Action, v.Fire, v.Reason)
	}
}

// TestFoldNextActionResetElapsedRuns: same session-cap crash, but idle PAST the 5h window →
// the reset has surely elapsed, so it is fire-eligible again (run).
func TestFoldNextActionResetElapsedRuns(t *testing.T) {
	v := FoldNextAction(NextInput{
		State:       ResumePending,
		Retry:       RetryDecision{Blocked: false},
		LimitReason: LimitSession,
		IdleSeconds: SessionLimitResetSeconds + 1,
		Admitted:    true,
	})
	if v.Action != ActRun || !v.Fire {
		t.Fatalf("action = %q, want run after the reset window elapsed (%s)", v.Action, v.Reason)
	}
}

// TestFoldNextActionUnknownIdleWaits: a wall-clock cap with UNKNOWN idle cannot be proven
// reset, so the conservative call is wait_reset (never fire on a guess).
func TestFoldNextActionUnknownIdleWaits(t *testing.T) {
	v := FoldNextAction(NextInput{
		State:       ResumePending,
		Retry:       RetryDecision{Blocked: false},
		LimitReason: LimitWeekly,
		IdleSeconds: -1,
		Admitted:    true,
	})
	if v.Action != ActWaitReset {
		t.Fatalf("action = %q, want wait_reset for unknown idle on a wall-clock cap", v.Action)
	}
}

// TestFoldNextActionRateLimitNotWallClock: a 429 rate crash is a source-burst concern, not a
// wall-clock reset — with the host admitting it should run, not wait_reset.
func TestFoldNextActionRateLimitNotWallClock(t *testing.T) {
	v := FoldNextAction(NextInput{
		State:       ResumePending,
		Retry:       RetryDecision{Blocked: false},
		LimitReason: LimitRate,
		IdleSeconds: 5,
		Admitted:    true,
	})
	if v.Action != ActRun {
		t.Fatalf("action = %q, want run — a 429 is not a wall-clock reset (%s)", v.Action, v.Reason)
	}
}

// TestFoldNextActionHoldAdmission: fire-eligible, reset elapsed, but the host source gate
// refuses → hold_admission, carrying the retry_after and the refusal reason.
func TestFoldNextActionHoldAdmission(t *testing.T) {
	v := FoldNextAction(NextInput{
		State:               ResumeReStranded,
		Retry:               RetryDecision{Blocked: false, Reason: "last resume failed recoverably; attempt 2/8"},
		Admitted:            false,
		AdmitReason:         ReasonSourceSaturated,
		AdmitRetryAfterUnix: 1234567890,
	})
	if v.Action != ActHoldAdmission || v.Fire {
		t.Fatalf("action = %q fire=%v, want hold_admission/false", v.Action, v.Fire)
	}
	if v.RetryAfterUnix != 1234567890 {
		t.Errorf("retry_after = %d, want the carried admission retry_after", v.RetryAfterUnix)
	}
	if !strings.Contains(v.Reason, ReasonSourceSaturated) {
		t.Errorf("reason %q should name the admission refusal", v.Reason)
	}
}

// TestFoldNextActionLogin: the gate is blocked AND the last outcome was an auth wall → login
// (a human must clear it), never gave_up.
func TestFoldNextActionLogin(t *testing.T) {
	v := FoldNextAction(NextInput{
		State:    ResumeGaveUp,
		Outcome:  OutcomeUnrecoverable,
		Retry:    RetryDecision{Blocked: true, Reason: "last resume hit an auth/access wall — a re-resume cannot fix it"},
		Admitted: true,
	})
	if v.Action != ActLogin || v.Fire {
		t.Fatalf("action = %q, want login for a blocked auth wall (%s)", v.Action, v.Reason)
	}
}

// TestFoldNextActionGaveUp: the gate is blocked by a spent attempt cap (not an auth wall) →
// gave_up.
func TestFoldNextActionGaveUp(t *testing.T) {
	v := FoldNextAction(NextInput{
		State:    ResumeGaveUp,
		Outcome:  OutcomeRecoverable,
		Retry:    RetryDecision{Blocked: true, Reason: "attempt cap reached (8/8)"},
		Admitted: true,
	})
	if v.Action != ActGaveUp || v.Fire {
		t.Fatalf("action = %q, want gave_up for a spent cap (%s)", v.Action, v.Reason)
	}
}

// TestFoldNextActionDone: a resume that took is blocked from re-firing and is not a wall →
// done (the quiet tail).
func TestFoldNextActionDone(t *testing.T) {
	v := FoldNextAction(NextInput{
		State:    ResumeTook,
		Outcome:  OutcomeProgressed,
		Retry:    RetryDecision{Blocked: true, Reason: "already resumed once (resume took)"},
		Admitted: true,
	})
	if v.Action != ActDone || v.Fire {
		t.Fatalf("action = %q, want done for a resume that took (%s)", v.Action, v.Reason)
	}
}

// TestResetElapsedTable exercises the reset-window helper directly across the closed limit
// vocabulary.
func TestResetElapsedTable(t *testing.T) {
	cases := []struct {
		limit string
		idle  int64
		want  bool
	}{
		{LimitSession, SessionLimitResetSeconds - 1, false},
		{LimitSession, SessionLimitResetSeconds, true},
		{LimitWeekly, WeeklyLimitResetSeconds - 1, false},
		{LimitWeekly, WeeklyLimitResetSeconds, true},
		{LimitUsage, 60, false}, // usage takes the session floor
		{LimitUsage, SessionLimitResetSeconds + 1, true},
		{LimitRate, 0, true},      // a burst has no wall-clock reset
		{"", 0, true},             // not a limit crash
		{LimitSession, -1, false}, // unknown idle never proves elapsed
	}
	for _, c := range cases {
		if got := resetElapsed(c.limit, c.idle); got != c.want {
			t.Errorf("resetElapsed(%q, %d) = %v, want %v", c.limit, c.idle, got, c.want)
		}
	}
}
