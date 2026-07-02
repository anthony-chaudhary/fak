package resume

import "testing"

// mturn is a real model turn carrying a prompt of n tokens.
func mturn(n int) Event { return Event{Kind: EventRealAssistant, PromptTokens: n} }

// rl is a synthetic rate-limit refusal turn with the given closed reason.
func rl(reason string) Event { return Event{Kind: EventRateLimitError, LimitReason: reason} }

// book is an ignored bookkeeping/user record (system, last-prompt, mode, …).
var book = Event{Kind: EventOther}

// scanIn is the resume Input used across the scan tests: a 2-hour idle on a 5-minute cache,
// so a sizable resident transcript is COLD and the managed plan recommends a CUT.
func scanIn() Input {
	return Input{IdleSeconds: 7200, TTL: TTL5m, Pricing: opusPricing, HorizonTurns: 20}
}

// TestDiagnoseUnresumedRateLimit is the goal in one test: a transcript whose last real model
// turn (250k prompt) is followed by a synthetic session-limit refusal and only bookkeeping —
// the canonical rate-limited crash. It is flagged for restart, sized from the real turn (NOT
// the zero-usage refusal), and its managed plan recommends a CUT off the cold prefix.
func TestDiagnoseUnresumedRateLimit(t *testing.T) {
	events := []Event{mturn(120000), mturn(250000), rl(LimitSession), book, book}
	d := Diagnose(events, scanIn())

	if d.Crash != CrashRateLimit {
		t.Fatalf("crash = %q, want rate_limit", d.Crash)
	}
	if !d.NeedsRestart || !d.Unresumed {
		t.Errorf("NeedsRestart=%v Unresumed=%v, want both true", d.NeedsRestart, d.Unresumed)
	}
	if d.LimitReason != LimitSession {
		t.Errorf("limit reason = %q, want %q", d.LimitReason, LimitSession)
	}
	if d.ResidentTokens != 250000 {
		t.Errorf("resident = %d, want 250000 (the last REAL turn, not the zero-usage refusal)", d.ResidentTokens)
	}
	if d.RealTurns != 2 {
		t.Errorf("real turns = %d, want 2", d.RealTurns)
	}
	if d.Plan.Posture != PostureCold {
		t.Errorf("plan posture = %q, want cold", d.Plan.Posture)
	}
	if d.Plan.Recommended != StrategyCut || d.Plan.Reason != ReasonColdPrefillShed {
		t.Errorf("plan recommend = (%q,%q), want (cut, cold_prefill_shed)", d.Plan.Recommended, d.Plan.Reason)
	}
}

// TestDiagnoseRecoveredRateLimit: a rate limit that was FOLLOWED by a real model turn was
// recovered — the session came back on its own, so it is not flagged for restart.
func TestDiagnoseRecoveredRateLimit(t *testing.T) {
	events := []Event{mturn(80000), rl(LimitRate), mturn(90000)}
	d := Diagnose(events, scanIn())

	if d.Crash != CrashNone {
		t.Fatalf("crash = %q, want none (a real turn followed the rate limit)", d.Crash)
	}
	if d.NeedsRestart || d.Unresumed {
		t.Errorf("NeedsRestart=%v Unresumed=%v, want both false", d.NeedsRestart, d.Unresumed)
	}
	if d.ResidentTokens != 90000 {
		t.Errorf("resident = %d, want 90000 (the last real turn)", d.ResidentTokens)
	}
}

// TestDiagnoseCleanEnd: a transcript that ends on a real model turn is a clean end.
func TestDiagnoseCleanEnd(t *testing.T) {
	d := Diagnose([]Event{mturn(50000), mturn(60000), book}, scanIn())
	if d.Crash != CrashNone || d.NeedsRestart {
		t.Errorf("crash=%q NeedsRestart=%v, want (none,false)", d.Crash, d.NeedsRestart)
	}
}

// TestDiagnoseOtherErrorNotFlagged: an unclean end on a NON-rate error is reported as
// other_error (unresumed) but not flagged for this verb's rate-limit restart.
func TestDiagnoseOtherErrorNotFlagged(t *testing.T) {
	events := []Event{mturn(70000), {Kind: EventOtherError}, book}
	d := Diagnose(events, scanIn())
	if d.Crash != CrashOther {
		t.Fatalf("crash = %q, want other_error", d.Crash)
	}
	if d.NeedsRestart {
		t.Errorf("NeedsRestart = true, want false for a non-rate error")
	}
	if !d.Unresumed {
		t.Errorf("Unresumed = false, want true (no real turn after the error)")
	}
}

// TestDiagnoseTerminalErrorWins: when both a rate limit and a later other-error sit in the
// tail, the error CLOSEST to the end is the terminal failure. A rate limit after an
// other-error is a rate-limit crash; an other-error after a rate limit is an other crash.
func TestDiagnoseTerminalErrorWins(t *testing.T) {
	rlLast := Diagnose([]Event{mturn(40000), Event{Kind: EventOtherError}, rl(LimitWeekly)}, scanIn())
	if rlLast.Crash != CrashRateLimit || rlLast.LimitReason != LimitWeekly {
		t.Errorf("rate-limit-last: crash=(%q,%q), want (rate_limit, weekly_limit)", rlLast.Crash, rlLast.LimitReason)
	}
	otherLast := Diagnose([]Event{mturn(40000), rl(LimitSession), Event{Kind: EventOtherError}}, scanIn())
	if otherLast.Crash != CrashOther {
		t.Errorf("other-last: crash=%q, want other_error", otherLast.Crash)
	}
}

// TestDiagnosePinOverridesResident: an explicit Input.ResidentTokens pin wins over the size
// derived from the events, mirroring the single-transcript flag precedence.
func TestDiagnosePinOverridesResident(t *testing.T) {
	in := scanIn()
	in.ResidentTokens = 999000
	d := Diagnose([]Event{mturn(10), rl(LimitSession)}, in)
	if d.ResidentTokens != 999000 {
		t.Errorf("resident = %d, want the pinned 999000", d.ResidentTokens)
	}
}

// TestDiagnoseRateLimitWithNoRealTurn: a session that hit the limit before any real model
// turn (a tiny transcript) is still flagged, but has nothing to manage — resident 0, zero
// real turns, and a trivial RESUME_FULL plan.
func TestDiagnoseRateLimitWithNoRealTurn(t *testing.T) {
	d := Diagnose([]Event{book, rl(LimitSession), book}, scanIn())
	if !d.NeedsRestart {
		t.Fatalf("NeedsRestart = false, want true (it crashed on a limit)")
	}
	if d.ResidentTokens != 0 || d.RealTurns != 0 {
		t.Errorf("resident=%d realTurns=%d, want 0/0", d.ResidentTokens, d.RealTurns)
	}
	if d.Plan.Recommended != StrategyResumeFull {
		t.Errorf("plan recommend = %q, want resume_full (nothing to shed)", d.Plan.Recommended)
	}
}

// TestDiagnoseDeterministicAndTotal: identical inputs give identical diagnoses, and the empty
// event slice yields a defined clean diagnosis rather than a panic.
func TestDiagnoseDeterministicAndTotal(t *testing.T) {
	events := []Event{mturn(123456), mturn(200000), rl(LimitSession)}
	a, b := Diagnose(events, scanIn()), Diagnose(events, scanIn())
	if a.Crash != b.Crash || a.NeedsRestart != b.NeedsRestart || a.ResidentTokens != b.ResidentTokens {
		t.Errorf("Diagnose is not deterministic: %+v vs %+v", a, b)
	}
	empty := Diagnose(nil, scanIn())
	if empty.Crash != CrashNone || empty.NeedsRestart || empty.ResidentTokens != 0 {
		t.Errorf("empty diagnosis = %+v, want clean/zero", empty)
	}
}

// uturn is a user-side main-chain record (a typed prompt or a tool result) — the model
// owes a reply after it.
var uturn = Event{Kind: EventUserTurn}

// TestDiagnoseInterruptedMidTurn is the invisible-death regression (the 392f-class crash):
// a transcript whose tail is an unanswered user turn — no refusal record, no error, just
// silence — used to read as a clean end and vanish from every readout. Past the idle floor
// it must be diagnosed interrupted (unresumed), while staying OFF the rate-limit restart
// flag (scan's NeedsRestart contract is unchanged).
func TestDiagnoseInterruptedMidTurn(t *testing.T) {
	events := []Event{mturn(120000), uturn, book}
	d := Diagnose(events, scanIn()) // 2h idle — far past the floor

	if d.Crash != CrashInterrupted {
		t.Fatalf("crash = %q, want interrupted", d.Crash)
	}
	if !d.Unresumed {
		t.Error("Unresumed = false, want true (nothing answered the user turn)")
	}
	if d.NeedsRestart {
		t.Error("NeedsRestart = true, want false (interrupted is not the rate-limit restart class)")
	}
	if d.ResidentTokens != 120000 {
		t.Errorf("resident = %d, want 120000 (sized from the last real turn)", d.ResidentTokens)
	}
}

// TestDiagnoseInterruptedFirstPrompt: a session killed before its FIRST assistant reply
// (no real turn at all) is still an interrupted death, not a clean end.
func TestDiagnoseInterruptedFirstPrompt(t *testing.T) {
	d := Diagnose([]Event{uturn, book}, scanIn())
	if d.Crash != CrashInterrupted {
		t.Fatalf("crash = %q, want interrupted (unanswered first prompt)", d.Crash)
	}
}

// TestDiagnoseInterruptedIdleFloor: below the idle floor — or with idle unknown — the same
// tail may be a LIVE session still thinking, so the conservative verdict is a clean end
// (never invite a duplicate resume onto a live session).
func TestDiagnoseInterruptedIdleFloor(t *testing.T) {
	events := []Event{mturn(120000), uturn}
	for _, idle := range []int64{0, InterruptedIdleFloorSeconds - 1, -1} {
		d := Diagnose(events, Input{IdleSeconds: idle, TTL: TTL5m})
		if d.Crash != CrashNone {
			t.Errorf("idle=%d: crash = %q, want none (below the interrupted floor)", idle, d.Crash)
		}
	}
	d := Diagnose(events, Input{IdleSeconds: InterruptedIdleFloorSeconds, TTL: TTL5m})
	if d.Crash != CrashInterrupted {
		t.Errorf("idle=floor: crash = %q, want interrupted", d.Crash)
	}
}

// TestDiagnoseTailErrorBeatsUserTurn: when the tail carries BOTH an unanswered user turn
// and an error record, the error is the terminal failure regardless of their order — the
// richer verdict (with its limit reason and reset semantics) must win.
func TestDiagnoseTailErrorBeatsUserTurn(t *testing.T) {
	for name, events := range map[string][]Event{
		"error after user turn":  {mturn(50000), uturn, rl(LimitSession)},
		"error before user turn": {mturn(50000), rl(LimitSession), uturn},
	} {
		d := Diagnose(events, scanIn())
		if d.Crash != CrashRateLimit || d.LimitReason != LimitSession {
			t.Errorf("%s: crash = (%q,%q), want (rate_limit,session_limit)", name, d.Crash, d.LimitReason)
		}
	}
}

// TestDiagnoseAnsweredUserTurnIsClean: a user turn FOLLOWED by a real model turn was
// answered — the ordinary conversational rhythm, not an interruption.
func TestDiagnoseAnsweredUserTurnIsClean(t *testing.T) {
	d := Diagnose([]Event{uturn, mturn(50000), book}, scanIn())
	if d.Crash != CrashNone {
		t.Errorf("crash = %q, want none (the user turn was answered)", d.Crash)
	}
}
