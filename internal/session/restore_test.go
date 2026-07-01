package session

import (
	"reflect"
	"testing"
)

// TestRestoreRoundTripsSnapshotVerbatim is the load-bearing witness for the durable
// session image: a State read out via Snapshot, then Restore-d into a FRESH table,
// reproduces the exact record — every axis AND the Rev. This is what makes a session
// offloaded to another host/instance/VM and brought back resume at the drive it held,
// not at a default.
func TestRestoreRoundTripsSnapshotVerbatim(t *testing.T) {
	src := NewTable()
	// Drive a session away from every default so a silent drop on any axis is visible.
	src.Transition("gw-7", Throttled, "operator-offload")
	src.SetBudget("gw-7", Budget{TurnsLeft: 3, TokensLeft: 4096})
	src.SetPriority("gw-7", 5)
	src.SetPace("gw-7", Pace{MaxTokensPerTurn: 512, MinTurnGapMs: 100})

	snap := src.Snapshot()
	if len(snap) != 1 {
		t.Fatalf("snapshot len = %d, want 1", len(snap))
	}
	want := snap[0]

	dst := NewTable()
	got := dst.Restore("gw-7", want)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Restore returned %+v, want %+v", got, want)
	}
	// And the table now serves the restored record verbatim, Rev included.
	if reread := dst.Get("gw-7"); !reflect.DeepEqual(reread, want) {
		t.Fatalf("Get after Restore = %+v, want %+v (Rev must be preserved, not bumped)", reread, want)
	}
}

// TestRestoreReestablishesTerminalSession proves the honesty rung: a STOPPED image
// restores AS Stopped, with its reason — never silently resurrected as Running. The
// live control verbs refuse to write a terminal session; Restore is the deliberate
// exception, because resume must be faithful.
func TestRestoreReestablishesTerminalSession(t *testing.T) {
	stopped := State{TraceID: "gw-9", Run: Stopped, Reason: ReasonBudgetTurns, Rev: 42,
		Budget: Budget{TurnsLeft: 0, TokensLeft: 0}}

	dst := NewTable()
	got := dst.Restore("gw-9", stopped)
	if got.Run != Stopped {
		t.Fatalf("restored run = %s, want stopped", got.Run)
	}
	if got.Reason != ReasonBudgetTurns {
		t.Fatalf("restored reason = %q, want %q", got.Reason, ReasonBudgetTurns)
	}
	if got.Rev != 42 {
		t.Fatalf("restored Rev = %d, want 42 (a load preserves Rev)", got.Rev)
	}
	// A subsequent Decide on the restored terminal session stops cleanly with its
	// reason — it is a real Stopped session, not a phantom default Running.
	v := dst.Decide("gw-9")
	if v.Proceed || !v.Stop {
		t.Fatalf("Decide on restored Stopped = proceed=%v stop=%v, want proceed=false stop=true", v.Proceed, v.Stop)
	}
	if v.Reason != ReasonBudgetTurns {
		t.Fatalf("Decide reason = %q, want %q", v.Reason, ReasonBudgetTurns)
	}
}

// TestRestoreForcesTraceKey confirms a session can be re-homed to a new id on a new
// host: the record's TraceID is overwritten by the restore key.
func TestRestoreForcesTraceKey(t *testing.T) {
	st := State{TraceID: "old-home", Run: Running, Budget: Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded}}
	dst := NewTable()
	got := dst.Restore("new-home", st)
	if got.TraceID != "new-home" {
		t.Fatalf("restored TraceID = %q, want new-home", got.TraceID)
	}
	if dst.Get("old-home").Run != Running || dst.Len() != 1 {
		t.Fatalf("restore must key under new-home only; len=%d", dst.Len())
	}
}
