package agent

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// The default task takes ~7 mock turns to complete; these tests cap it below that to
// prove the session gate ends the arm early, and confirm the no-table path is the
// historical loop.

func TestRunArmNoTableIsHistoricalLoop(t *testing.T) {
	p := NewMockPlanner("mock")
	m, err := RunArm(context.Background(), p, DefaultTask, false, 20, nil)
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if m.StoppedBySession != "" {
		t.Fatalf("no-table run set StoppedBySession=%q, want empty (historical loop untouched)", m.StoppedBySession)
	}
	if m.FinalAnswer == "" {
		t.Fatal("no-table run produced no final answer; the task should complete in <20 turns")
	}
}

func TestRunArmTurnBudgetCapsRun(t *testing.T) {
	p := NewMockPlanner("mock")
	tbl := session.NewTable()
	const trace = "arm-1"
	tbl.SetBudget(trace, session.Budget{TurnsLeft: 2, TokensLeft: session.Unbounded})

	m, err := RunArm(context.Background(), p, DefaultTask, false, 20, nil, WithSessionTable(tbl, trace))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if m.Turns != 2 {
		t.Fatalf("turn budget 2 ran %d model turns, want exactly 2", m.Turns)
	}
	if m.StoppedBySession != session.ReasonBudgetTurns {
		t.Fatalf("StoppedBySession=%q, want %s", m.StoppedBySession, session.ReasonBudgetTurns)
	}
	if m.FinalAnswer != "" {
		t.Fatal("budget-capped run reached a final answer; it should have stopped first")
	}
	// The session is now Stopped in the table — observable, not re-derived.
	if st := tbl.Get(trace); st.Run != session.Stopped {
		t.Fatalf("after budget cap session Run=%v, want Stopped", st.Run)
	}
}

func TestRunArmPausedStopsImmediately(t *testing.T) {
	p := NewMockPlanner("mock")
	tbl := session.NewTable()
	const trace = "arm-paused"
	tbl.Transition(trace, session.Paused, "")

	m, err := RunArm(context.Background(), p, DefaultTask, false, 20, nil, WithSessionTable(tbl, trace))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if m.Turns != 0 {
		t.Fatalf("paused session ran %d turns, want 0 (held at the first boundary)", m.Turns)
	}
	if m.StoppedBySession != session.ReasonPaused {
		t.Fatalf("StoppedBySession=%q, want %s", m.StoppedBySession, session.ReasonPaused)
	}
}

func TestRunArmDrainingTakenAtFirstBoundary(t *testing.T) {
	p := NewMockPlanner("mock")
	tbl := session.NewTable()
	const trace = "arm-drain"
	tbl.Transition(trace, session.Draining, "operator-stop")

	m, err := RunArm(context.Background(), p, DefaultTask, false, 20, nil, WithSessionTable(tbl, trace))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if m.Turns != 0 || m.StoppedBySession == "" {
		t.Fatalf("draining arm: Turns=%d StoppedBySession=%q, want 0 turns with a stop reason", m.Turns, m.StoppedBySession)
	}
	if st := tbl.Get(trace); st.Run != session.Stopped {
		t.Fatalf("after draining run, session Run=%v, want Stopped (taken at boundary)", st.Run)
	}
}

func TestRunArmNilTableViaOptionIsPermissive(t *testing.T) {
	// Passing the option with a nil table must degrade to the historical loop, so a
	// caller can wire the option unconditionally.
	p := NewMockPlanner("mock")
	m, err := RunArm(context.Background(), p, DefaultTask, false, 20, nil, WithSessionTable(nil, "x"))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if m.StoppedBySession != "" || m.FinalAnswer == "" {
		t.Fatalf("nil-table option changed behavior: StoppedBySession=%q final=%q", m.StoppedBySession, m.FinalAnswer)
	}
}
