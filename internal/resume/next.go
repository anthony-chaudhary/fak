// next.go — the ACT half of the resume process: the single deterministic step that turns
// "where does this crashed session stand?" (FoldResumeState) into "what do I DO about it
// right now?". scan.go finds the sessions that need a restart; outcome.go folds each one's
// journey state; this file folds that state — plus the host's live launch-admission verdict
// and the crash's reset class — into ONE closed NextAction per session, so an agent bringing
// a dead batch back reads an ordered runbook and acts, instead of re-deriving the decision
// per session from four separate verbs.
//
// # Why a distinct fold
//
// ResumeState answers "did the resume take?" — a diagnostic. It does NOT answer "should I
// fire a new resume, and if not, why not?". Those turn on three more facts a state label
// omits: (1) the RetryGate verdict (is a new automatic resume even allowed, or is the
// attempt budget spent / the wall unrecoverable), (2) the host source-admission verdict (may
// the BOX take one more live resume right now, or is it at the 529 burst ceiling), and (3)
// the crash's reset class (a wall-clock cap must wait out its named reset before a relaunch
// can possibly stick). NextAction is the total function over those facts. It is the runbook
// the operator question "I have N dead sessions — what do I run?" actually asks for.
//
// # Content-blind and clock-free by construction
//
// Like every decision in this package, NextAction reasons only over typed facts the cmd/fak
// shell folds (the state, the outcome, the gate, the closed limit token, an idle count, and
// the shared admission verdict). It reads no clock and does no I/O — the reset-window
// comparison uses the caller-supplied idle seconds against a documented cap window, never
// time.Now — so same facts in, same action out.
package resume

import "fmt"

// NextAction is the closed vocabulary of what to do with a dead/dormant session now. It is
// emittable (the shell renders it), verifiable (each maps to a checkable precondition), and
// routable (an agent acts on it) — the same closed-set discipline the refusal vocabulary
// takes, aimed at the resume runbook.
type NextAction string

const (
	// ActRun: fire a managed restart / resume now — the gate allows it, the crash's reset
	// (if any) has elapsed, and the host admits one more live resume. This is the only
	// action that carries a launch command.
	ActRun NextAction = "run"
	// ActWaitReset: the session crashed on a wall-clock usage cap (session/weekly/usage)
	// whose reset window may not yet have elapsed — relaunching before it resets just
	// re-strands. Wait for the named reset, then it becomes ActRun.
	ActWaitReset NextAction = "wait_reset"
	// ActHoldAdmission: the session is fire-eligible, but the host is at its live-resume
	// ceiling / launch-rate / spacing floor (the per-source 529 burst wall). Defer until
	// RetryAfterUnix, then re-evaluate — a batch-wide backpressure, not a per-session block.
	ActHoldAdmission NextAction = "hold_admission"
	// ActLogin: the last attempt hit an auth/login/credit/access wall — a re-resume cannot
	// fix it. A human must clear the account (interactive /login) before it can be resumed.
	ActLogin NextAction = "login"
	// ActGaveUp: the automatic-resume attempt budget is spent. No further automatic resume
	// will fire; a human owns the decision to force one.
	ActGaveUp NextAction = "gave_up"
	// ActDone: nothing to fire — the resume took, an operator settled it, it is a fired-but-
	// unproven launch (burn-once), or the session ended cleanly. The good/quiet tail.
	ActDone NextAction = "done"
)

// Documented usage-cap reset windows, in seconds — the wall-clock windows within which the
// provider's rolling caps reset. They are the published cap semantics (a 5-hour rolling
// session cap, a 7-day weekly cap), NOT a fak measurement: they bound how long a wall-clock
// crash must wait before a relaunch can possibly stick. usage_limit has no published window,
// so it takes the session floor (the shorter, more conservative wait). A value of 0 means
// "no wall-clock reset to wait on" (a 529 burst / a non-limit crash).
const (
	// SessionLimitResetSeconds is the 5-hour rolling session-cap window.
	SessionLimitResetSeconds int64 = 5 * 3600
	// WeeklyLimitResetSeconds is the 7-day weekly-cap window.
	WeeklyLimitResetSeconds int64 = 7 * 24 * 3600
)

// wallClockResetSeconds is the reset window a crash's closed limit token must wait out, or 0
// when the crash is not a wall-clock cap (a 529 burst is a source-admission concern, not a
// reset one; a non-limit crash has no reset to wait on). usage_limit takes the session floor.
func wallClockResetSeconds(limitReason string) int64 {
	switch limitReason {
	case LimitSession, LimitUsage:
		return SessionLimitResetSeconds
	case LimitWeekly:
		return WeeklyLimitResetSeconds
	default: // LimitRate (429 burst — admission handles it) or "" (not a limit crash)
		return 0
	}
}

// resetElapsed reports whether a wall-clock cap's reset window has SURELY elapsed given the
// session's idle time. Unknown idle (< 0) is never "surely elapsed" — the conservative call
// is to wait. A crash with no wall-clock window (resetSeconds 0) is trivially elapsed (there
// is nothing to wait on). idle >= the window means the reset has definitely passed.
func resetElapsed(limitReason string, idleSeconds int64) bool {
	window := wallClockResetSeconds(limitReason)
	if window <= 0 {
		return true // no wall-clock reset to wait on
	}
	if idleSeconds < 0 {
		return false // idle unknown: cannot prove the reset passed → wait
	}
	return idleSeconds >= window
}

// NextInput is the closed set of facts NextAction folds — all shell-computed from the same
// leaf functions the status readout already runs (FoldResumeState, ClassifyOutcome,
// RetryGate) plus the shared host-admission verdict. No transcript content, no clock.
type NextInput struct {
	// State is the folded resume-journey label (FoldResumeState).
	State ResumeState `json:"state"`
	// Outcome is the terminal-turn classification of the last attempt (ClassifyOutcome) —
	// it disambiguates a blocked gate (an auth wall → login vs a spent cap → gave_up).
	Outcome Outcome `json:"outcome"`
	// Retry is the outcome-aware RetryGate verdict — the master "may a new resume fire" gate.
	Retry RetryDecision `json:"retry"`
	// LimitReason is the closed crash limit token (session_limit/weekly_limit/usage_limit/
	// rate_limited), or "" when the session did not crash on a limit.
	LimitReason string `json:"limit_reason,omitempty"`
	// IdleSeconds is how long since the session last moved (-1 unknown); it decides whether a
	// wall-clock cap's reset window has surely elapsed.
	IdleSeconds int64 `json:"idle_seconds"`
	// Admitted is the shared host source-admission verdict (AdmitSource.Admit) — the same for
	// every session in a batch, since the 529 burst wall is a per-source, not per-session, bound.
	Admitted bool `json:"admitted"`
	// AdmitReason / AdmitRetryAfterUnix carry the admission refusal detail through to the
	// hold_admission verdict so the agent sees why and until when.
	AdmitReason         string `json:"admit_reason,omitempty"`
	AdmitRetryAfterUnix int64  `json:"admit_retry_after_unix,omitempty"`
}

// NextVerdict is the per-session runbook verdict.
type NextVerdict struct {
	// Action is the closed next-action token.
	Action NextAction `json:"action"`
	// Reason is the human one-liner explaining the action.
	Reason string `json:"reason"`
	// Fire is true iff Action == ActRun — the one bit a launcher gates on: this session
	// should have a resume fired now. Every other action is a defer/settle with a reason.
	Fire bool `json:"fire"`
	// RetryAfterUnix is the earliest a deferred action (hold_admission) should be retried,
	// carried straight from the admission decision. 0 when there is no machine-checkable wait.
	RetryAfterUnix int64 `json:"retry_after_unix,omitempty"`
}

// NextAction folds the facts into the one runbook action, applying the preconditions in a
// fixed order so the verdict is deterministic and the first binding constraint wins:
//
//  1. The RetryGate blocks a new resume → we are NOT firing. Split the reason by why:
//     an auth wall (login), a spent attempt budget (gave_up), or a resume that already
//     took / is unproven / was settled (done).
//  2. The gate allows a resume (pending, or a recoverable re-strand under the cap) → this
//     session is fire-eligible. Check the preconditions before firing:
//     a. a wall-clock cap whose reset has NOT surely elapsed → wait_reset;
//     b. the host source-admission gate refuses → hold_admission (with retry_after);
//     c. otherwise → run.
//
// Total over any input: the zero RetryDecision is Blocked=false and the zero input carries
// no limit and Admitted=false, so a bare zero value folds to hold_admission — but a caller
// never constructs that: every real row carries a gate, an outcome, and the shared admission
// verdict computed from the same leaf functions.
func FoldNextAction(in NextInput) NextVerdict {
	if in.Retry.Blocked {
		switch {
		case in.Outcome == OutcomeUnrecoverable:
			return NextVerdict{Action: ActLogin,
				Reason: "last resume hit an auth/access wall — a re-resume cannot fix it; a human must /login"}
		case in.State == ResumeGaveUp:
			return NextVerdict{Action: ActGaveUp, Reason: in.Retry.Reason}
		default:
			return NextVerdict{Action: ActDone, Reason: in.Retry.Reason}
		}
	}

	if in.LimitReason != "" && !resetElapsed(in.LimitReason, in.IdleSeconds) {
		return NextVerdict{Action: ActWaitReset,
			Reason: fmt.Sprintf("crashed on a %s whose reset window may not have elapsed — wait, then resume", in.LimitReason)}
	}

	if !in.Admitted {
		reason := "host at the live-resume / burst ceiling — defer this launch"
		if in.AdmitReason != "" {
			reason = "host admission refused (" + in.AdmitReason + ") — defer this launch"
		}
		return NextVerdict{Action: ActHoldAdmission, Reason: reason, RetryAfterUnix: in.AdmitRetryAfterUnix}
	}

	return NextVerdict{Action: ActRun, Fire: true,
		Reason: "gate allows it, reset (if any) elapsed, host admits — fire the managed resume now"}
}
