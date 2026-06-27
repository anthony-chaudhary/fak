package main

// serve_session_state_test.go — the COLD-RESUME persistence of #629: `fak serve`
// dumps the live DRIVE table on a clean shutdown and re-attaches every session at the
// budget / priority / run-state / pace it held on the next boot, not its defaults. These
// tests drive the two serve-lifecycle hooks (restoreServeSessions / dumpServeSessions)
// over a FRESH table each, so they are isolated from the process-global serveSessions and
// from each other. They cover the load-bearing fence: a STOPPED session reloads STOPPED
// with its reason, never silently RUNNING.

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// TestServeSessionStateColdResume is the round-trip: dump a table holding three sessions
// at non-default drive state, then cold-resume onto a FRESH table (the process-restart
// simulation) and assert each session re-attaches verbatim.
func TestServeSessionStateColdResume(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session-state.snap")

	src := session.NewTable()
	// A terminal session with a closed reason and a specific Rev — the fence case.
	src.Restore("sess-stopped", session.State{
		TraceID: "sess-stopped", Run: session.Stopped, Reason: session.ReasonBudgetTurns, Rev: 7,
	})
	// A throttled session holding a non-default budget.
	src.Transition("sess-throttled", session.Throttled, "operator-dial-down")
	src.SetBudget("sess-throttled", session.Budget{TurnsLeft: 3, TokensLeft: 4096})
	// A paused session.
	src.Transition("sess-paused", session.Paused, "")

	dumpServeSessions(src, path)
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("dumpServeSessions did not write %s: %v", path, err)
	}

	dst := session.NewTable()
	if err := restoreServeSessions(dst, path); err != nil {
		t.Fatalf("restoreServeSessions: %v", err)
	}

	// A STOPPED session reloads STOPPED with its reason and Rev — not silently RUNNING.
	stopped := dst.Get("sess-stopped")
	if stopped.Run != session.Stopped {
		t.Errorf("stopped session reloaded as %v, want Stopped (never silently resurrected RUNNING)", stopped.Run)
	}
	if stopped.Reason != session.ReasonBudgetTurns {
		t.Errorf("stopped reason = %q, want %q", stopped.Reason, session.ReasonBudgetTurns)
	}
	if stopped.Rev != 7 {
		t.Errorf("stopped Rev = %d, want 7 (Restore preserves the persisted Rev)", stopped.Rev)
	}

	// The throttled session re-attaches its held budget, not a default.
	thr := dst.Get("sess-throttled")
	if thr.Run != session.Throttled {
		t.Errorf("throttled session reloaded as %v, want Throttled", thr.Run)
	}
	if thr.Budget.TokensLeft != 4096 || thr.Budget.TurnsLeft != 3 {
		t.Errorf("throttled budget = %+v, want {TurnsLeft:3 TokensLeft:4096}", thr.Budget)
	}

	if dst.Get("sess-paused").Run != session.Paused {
		t.Errorf("paused session reloaded as %v, want Paused", dst.Get("sess-paused").Run)
	}
}

// TestServeSessionStateMissingFileIsCleanFirstBoot: a never-written file is a fresh
// install, not an error — the first `fak serve` has nothing to resume.
func TestServeSessionStateMissingFileIsCleanFirstBoot(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "does-not-exist.snap")
	if err := restoreServeSessions(session.NewTable(), missing); err != nil {
		t.Fatalf("a missing --session-state file must be a clean first boot, got: %v", err)
	}
}

// TestServeSessionStateCorruptFileFailsLoud: a present-but-tampered drive record fails
// closed — the persisted drive is a re-checkable fact, so a corrupt one is refused rather
// than resumed as if whole.
func TestServeSessionStateCorruptFileFailsLoud(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corrupt.snap")
	if err := os.WriteFile(path, []byte("{not a valid snapshot envelope"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := restoreServeSessions(session.NewTable(), path); err == nil {
		t.Fatal("a corrupt --session-state file must fail loud, got nil error")
	}
}

// TestServeSessionStateOffIsNoOp: the empty default is off — restore and dump touch
// nothing and never error, so the no-flag serve path is byte-for-byte today's.
func TestServeSessionStateOffIsNoOp(t *testing.T) {
	if err := restoreServeSessions(session.NewTable(), ""); err != nil {
		t.Fatalf("empty --session-state must be a no-op restore, got: %v", err)
	}
	dumpServeSessions(session.NewTable(), "") // must not panic or write anything
}

// TestServeSessionStateRewriteRoundTrip: a second shutdown overwrites the file with the
// then-current table, so a sequence of restarts always resumes the latest drive state.
func TestServeSessionStateRewriteRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.snap")

	first := session.NewTable()
	first.Transition("sess-x", session.Paused, "")
	dumpServeSessions(first, path)

	// Restore, advance, re-dump — the second boot's shutdown captures the new state.
	mid := session.NewTable()
	if err := restoreServeSessions(mid, path); err != nil {
		t.Fatalf("restore #1: %v", err)
	}
	if mid.Get("sess-x").Run != session.Paused {
		t.Fatalf("restore #1: sess-x = %v, want Paused", mid.Get("sess-x").Run)
	}
	mid.Restore("sess-x", session.State{TraceID: "sess-x", Run: session.Stopped, Reason: session.ReasonStopped, Rev: 2})
	dumpServeSessions(mid, path)

	final := session.NewTable()
	if err := restoreServeSessions(final, path); err != nil {
		t.Fatalf("restore #2: %v", err)
	}
	if got := final.Get("sess-x"); got.Run != session.Stopped || got.Reason != session.ReasonStopped {
		t.Fatalf("restore #2: sess-x = {run:%v reason:%q}, want {Stopped %q}", got.Run, got.Reason, session.ReasonStopped)
	}
}
