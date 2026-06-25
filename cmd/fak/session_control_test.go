package main

// session_control_test.go — exercises the cmd/fak closures that bind the gateway's
// /v1/fak/session control surface (#620) to a real internal/session.Table: the
// verb→table dispatch (applySessionControl), the optimistic-concurrency CAS path
// (if_rev), the terminal-refusal (ok=false), and the SessionState projection. The
// HTTP routing/validation is covered by internal/gateway/session_routes_test.go;
// this file proves the host wiring actually drives the table it owns.

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// TestApplySessionControlDispatchesEveryVerb proves each route verb lands on its
// matching Table write and the returned SessionState reflects the new drive.
func TestApplySessionControlDispatchesEveryVerb(t *testing.T) {
	tbl := session.NewTable()
	const trace = "drive-1"

	// run: throttle the session, carrying a reason.
	st, ok, err := applySessionControl(tbl, trace, "run", gateway.SessionControlRequest{
		Run: "throttled", Reason: "operator-slowdown",
	})
	if err != nil || !ok || st.Run != session.Throttled || st.Reason != "operator-slowdown" {
		t.Fatalf("run verb: st=%+v ok=%v err=%v", st, ok, err)
	}

	// budget: cut the turns allotment live.
	st, ok, err = applySessionControl(tbl, trace, "budget", gateway.SessionControlRequest{
		Budget: &gateway.SessionBudget{TurnsLeft: 4, TokensLeft: -1},
	})
	if err != nil || !ok || st.Budget.TurnsLeft != 4 || st.Budget.TokensLeft != -1 {
		t.Fatalf("budget verb: st=%+v ok=%v err=%v", st, ok, err)
	}

	// pace: tighten the per-turn cap.
	st, ok, err = applySessionControl(tbl, trace, "pace", gateway.SessionControlRequest{
		Pace: &gateway.SessionPace{MaxTokensPerTurn: 256, MinTurnGapMs: 100},
	})
	if err != nil || !ok || st.Pace.MaxTokensPerTurn != 256 || st.Pace.MinTurnGapMs != 100 {
		t.Fatalf("pace verb: st=%+v ok=%v err=%v", st, ok, err)
	}

	// priority: lower the rank so an urgent session passes.
	st, ok, err = applySessionControl(tbl, trace, "priority", gateway.SessionControlRequest{
		Priority: intPtr(3),
	})
	if err != nil || !ok || st.Priority != 3 {
		t.Fatalf("priority verb: st=%+v ok=%v err=%v", st, ok, err)
	}
	if st.Rev != 4 {
		t.Fatalf("expected Rev=4 after four writes, got %d", st.Rev)
	}

	// Unknown verb ⇒ error (the route maps this to 400).
	if _, _, err := applySessionControl(tbl, trace, "nope", gateway.SessionControlRequest{}); err == nil {
		t.Fatalf("unknown verb must return an error")
	}
	// Missing body field ⇒ error.
	if _, _, err := applySessionControl(tbl, trace, "budget", gateway.SessionControlRequest{}); err == nil {
		t.Fatalf("budget verb without a body must return an error")
	}
}

// TestApplySessionControlCAS proves if_rev is the optimistic-concurrency guard: a
// matching rev applies the write; a stale rev loses the race (ok=false).
func TestApplySessionControlCAS(t *testing.T) {
	tbl := session.NewTable()
	const trace = "cas-1"

	// Seed a budget at Rev 1.
	seed, _, _ := applySessionControl(tbl, trace, "budget", gateway.SessionControlRequest{
		Budget: &gateway.SessionBudget{TurnsLeft: 10},
	})
	if seed.Rev != 1 {
		t.Fatalf("seed Rev = %d, want 1", seed.Rev)
	}

	// A stale if_rev (0 is "no CAS"; use an obviously-wrong rev) loses the race.
	stale, ok, err := applySessionControl(tbl, trace, "budget", gateway.SessionControlRequest{
		Budget: &gateway.SessionBudget{TurnsLeft: 5}, IfRev: 999,
	})
	if err != nil || ok {
		t.Fatalf("stale CAS must refuse: st=%+v ok=%v err=%v", stale, ok, err)
	}

	// The matching if_rev applies and bumps the rev.
	good, ok, err := applySessionControl(tbl, trace, "budget", gateway.SessionControlRequest{
		Budget: &gateway.SessionBudget{TurnsLeft: 5}, IfRev: seed.Rev,
	})
	if err != nil || !ok || good.Budget.TurnsLeft != 5 || good.Rev != 2 {
		t.Fatalf("matching CAS must apply: st=%+v ok=%v err=%v", good, ok, err)
	}
}

// TestApplySessionControlTerminalRefused proves a stopped session rejects every
// control verb (ok=false) — you start a new session, you do not un-stop one.
func TestApplySessionControlTerminalRefused(t *testing.T) {
	tbl := session.NewTable()
	const trace = "term-1"

	if _, _, err := applySessionControl(tbl, trace, "run", gateway.SessionControlRequest{
		Run: "stopped", Reason: "operator-stop",
	}); err != nil {
		t.Fatalf("stop seed: %v", err)
	}
	// Every verb on the now-terminal session must refuse (ok=false, no error).
	for _, verb := range []string{"run", "budget", "pace", "priority"} {
		req := gateway.SessionControlRequest{
			Budget: &gateway.SessionBudget{TurnsLeft: 1}, Pace: &gateway.SessionPace{MaxTokensPerTurn: 1},
			Priority: intPtr(1), Run: "running",
		}
		if _, ok, err := applySessionControl(tbl, trace, verb, req); ok || err != nil {
			t.Fatalf("terminal session verb %q must refuse with ok=false,err=nil; got ok=%v err=%v", verb, ok, err)
		}
	}
}

// TestControlAndObserveRoundTrip proves the package-global closures wired into the
// gateway Config (observeSession/controlSession over serveSessions) are connected
// end to end: a control write is visible to the next observe read.
func TestControlAndObserveRoundTrip(t *testing.T) {
	const trace = "roundtrip-1"
	t.Cleanup(func() { serveSessions.Reset(trace) })

	if _, _, err := controlSession(context.Background(), trace, "run",
		gateway.SessionControlRequest{Run: "paused"}); err != nil {
		t.Fatalf("control pause: %v", err)
	}
	got := observeSession(context.Background(), trace)
	if got.Run != "paused" || got.TraceID != trace {
		t.Fatalf("observe after pause = %+v, want run=paused", got)
	}
	// An unseen trace reads its safe default (Running, unbounded), never a phantom.
	fresh := observeSession(context.Background(), "never-seen-"+trace)
	if fresh.Run != "running" || fresh.Budget.TurnsLeft != session.Unbounded {
		t.Fatalf("unseen trace = %+v, want running/unbounded default", fresh)
	}
}

// intPtr is a small helper so the pointer-typed Priority field reads cleanly.
func intPtr(v int) *int { return &v }
