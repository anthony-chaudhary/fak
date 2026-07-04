package main

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// guard_timebudget_test.go witnesses the #2229 enforcement fix: before it, no
// production code called session.Table.DecideTimeBudget, so a `fak guard
// --max-duration` envelope was armed, persisted, and displayed but NEVER acted on.
// guardTimeBudgetExhausted is that missing production caller — the seam the
// supervision-loop ticker (runGuardChildSupervisedAndReport) polls once per tick.
// These tests pin its verdict classification: a bounded run stops with the closed
// TIME_BUDGET_EXHAUSTED token, an unbounded run is untouched, and an operator-paused
// run is deferred rather than drained. The DecideTimeBudget mechanics themselves are
// pinned in internal/session/timebudget_test.go; this proves the guard-side caller
// agrees, which is the enforcement gap #2229 named.

var timeBudgetT0 = time.Unix(1_700_000_000, 0)

// TestGuardTimeBudgetExhaustedBoundedStops: a --max-duration run is left alone while
// within its envelope and stops with TIME_BUDGET_EXHAUSTED once it elapses.
func TestGuardTimeBudgetExhaustedBoundedStops(t *testing.T) {
	tbl := session.NewTable()
	tbl.StartTimeBudget("guard", 30*time.Minute, timeBudgetT0)

	if stop, reason := guardTimeBudgetExhausted(tbl, "guard", timeBudgetT0.Add(10*time.Minute)); stop {
		t.Fatalf("within a 30m envelope at +10m: got stop=%v reason=%q, want no stop", stop, reason)
	}

	stop, reason := guardTimeBudgetExhausted(tbl, "guard", timeBudgetT0.Add(31*time.Minute))
	if !stop {
		t.Fatalf("past a 30m envelope at +31m: got stop=false, want stop")
	}
	if reason != session.ReasonTimeBudgetExhausted {
		t.Fatalf("stop reason = %q, want %q", reason, session.ReasonTimeBudgetExhausted)
	}
}

// TestGuardTimeBudgetExhaustedUnboundedUntouched: a live session with no envelope
// (--max-duration 0) is never stopped, however much wall-clock elapses — the
// "unbounded session is untouched" half of the acceptance.
func TestGuardTimeBudgetExhaustedUnboundedUntouched(t *testing.T) {
	tbl := session.NewTable()
	// limit<=0 configures an unbounded-but-started budget: the clock ticks for status,
	// but the envelope is never exceeded.
	tbl.StartTimeBudget("guard", 0, timeBudgetT0)

	if stop, reason := guardTimeBudgetExhausted(tbl, "guard", timeBudgetT0.Add(1000*time.Hour)); stop {
		t.Fatalf("unbounded session at +1000h: got stop=%v reason=%q, want untouched", stop, reason)
	}
}

// TestGuardTimeBudgetExhaustedPausedDeferred: an operator-paused session is deferred,
// not drained, even when its envelope has technically elapsed (mirrors DecideTimeBudget's
// Paused gate, so a --max-duration ticker never fights an operator hold).
func TestGuardTimeBudgetExhaustedPausedDeferred(t *testing.T) {
	tbl := session.NewTable()
	tbl.StartTimeBudget("guard", time.Minute, timeBudgetT0)
	tbl.Transition("guard", session.Paused, "operator hold")

	if stop, reason := guardTimeBudgetExhausted(tbl, "guard", timeBudgetT0.Add(time.Hour)); stop {
		t.Fatalf("paused session past its envelope: got stop=%v reason=%q, want deferred", stop, reason)
	}
}

// TestGuardTimeBudgetExhaustedNilAndEmptySafe: the caller is nil-/empty-safe, so a guard
// run that never armed a session table (no serveSessions traffic) is a clean no-op.
func TestGuardTimeBudgetExhaustedNilAndEmptySafe(t *testing.T) {
	if stop, _ := guardTimeBudgetExhausted(nil, "guard", timeBudgetT0); stop {
		t.Fatalf("nil table: got stop=true, want no stop")
	}
	tbl := session.NewTable()
	tbl.StartTimeBudget("guard", time.Minute, timeBudgetT0)
	if stop, _ := guardTimeBudgetExhausted(tbl, "  ", timeBudgetT0.Add(time.Hour)); stop {
		t.Fatalf("blank trace id: got stop=true, want no stop")
	}
}
