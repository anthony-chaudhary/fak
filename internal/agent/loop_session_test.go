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

func TestRunArmContextBudgetStopsAtNextBoundary(t *testing.T) {
	p := NewMockPlanner("mock")
	tbl := session.NewTable()
	const trace = "arm-context"
	tbl.SetBudget(trace, session.Budget{
		TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: 1,
	})

	m, err := RunArm(context.Background(), p, DefaultTask, false, 20, nil, WithSessionTable(tbl, trace))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if m.Turns != 1 {
		t.Fatalf("context budget ran %d turns, want exactly 1 before boundary stop", m.Turns)
	}
	if m.StoppedBySession != session.ReasonBudgetContext {
		t.Fatalf("StoppedBySession=%q, want %s", m.StoppedBySession, session.ReasonBudgetContext)
	}
	if st := tbl.Get(trace); st.Run != session.Stopped || st.ContinuationID == "" {
		t.Fatalf("after context cap session = %+v, want stopped with continuation id", st)
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

// runawayToolCallPlanner announces THREE tool calls in a SINGLE turn, then a final
// answer. It is the runaway shape the tool-call floor (#2887/#5235) exists for and the
// one the per-TURN gate structurally cannot hold: the whole batch dispatches inside one
// model round-trip, so a turn budget only ever sees the boundary after all of them ran.
type runawayToolCallPlanner struct{ calls int }

func (p *runawayToolCallPlanner) Model() string { return "runaway-tool-calls" }

func (p *runawayToolCallPlanner) Complete(_ context.Context, _ []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	p.calls++
	if p.calls == 1 {
		return &Completion{
			Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{
				{ID: "call_a", Type: "function", Function: Func{Name: "search", Arguments: `{"query":"a"}`}},
				{ID: "call_b", Type: "function", Function: Func{Name: "search", Arguments: `{"query":"b"}`}},
				{ID: "call_c", Type: "function", Function: Func{Name: "search", Arguments: `{"query":"c"}`}},
			}},
			FinishReason: "tool_calls",
			Usage:        Usage{CompletionTokens: 1},
		}, nil
	}
	return &Completion{
		Message:      Message{Role: RoleAssistant, Content: "done"},
		FinishReason: "stop",
		Usage:        Usage{CompletionTokens: 1},
	}, nil
}

// TestRunArmToolCallBudgetCutsRunawayLoop is the #5235 acceptance witness: the session's
// per-CALL allotment is debited per DISPATCHED tool call and cuts the arm MID-TURN. With
// ToolCallsLeft=2 against a turn announcing three calls, exactly two reach the kernel —
// the third is never dispatched — and the arm ends carrying the closed
// BUDGET_TOOLCALLS_EXHAUSTED witness with no final answer. A turn budget cannot express
// this: all three calls live in one turn.
func TestRunArmToolCallBudgetCutsRunawayLoop(t *testing.T) {
	p := &runawayToolCallPlanner{}
	tbl := session.NewTable()
	const trace = "arm-toolcalls"
	tbl.SetBudget(trace, session.Budget{
		TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ToolCallsLeft: 2,
	})

	var log []traceEvent
	m, err := RunArm(context.Background(), p, "task", false, 20, &log, WithSessionTable(tbl, trace))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	// The cut landed INSIDE the turn: two of the three announced calls dispatched and
	// were adjudicated; the third never reached the kernel.
	if m.ToolCalls != 2 || len(log) != 2 {
		t.Fatalf("tool-call budget 2 dispatched %d calls / %d trace events, want exactly 2/2 (mid-turn cut)", m.ToolCalls, len(log))
	}
	if m.Turns != 1 {
		t.Fatalf("tool-call-capped arm ran %d turns, want exactly the one in-flight turn", m.Turns)
	}
	if m.StoppedBySession != session.ReasonBudgetToolCalls {
		t.Fatalf("StoppedBySession=%q, want %s", m.StoppedBySession, session.ReasonBudgetToolCalls)
	}
	if m.FinalAnswer != "" {
		t.Fatalf("runaway-capped run reached a final answer %q; the floor should have cut it first", m.FinalAnswer)
	}
	// The stop is observable in the drive state, not re-derived from the metrics — and
	// the stamped cap keeps "0 left with a positive cap = exhausted" distinguishable
	// from "0 = unconfigured".
	st := tbl.Get(trace)
	if st.Run != session.Stopped {
		t.Fatalf("after tool-call cap session Run=%v, want Stopped", st.Run)
	}
	if st.Budget.ToolCallsLeft != 0 || st.Budget.ToolCallsCap != 2 {
		t.Fatalf("after tool-call cap budget left=%d cap=%d, want 0 left against a stamped cap of 2",
			st.Budget.ToolCallsLeft, st.Budget.ToolCallsCap)
	}
}

func TestRunArmZeroToolCallBudgetIsPermissive(t *testing.T) {
	// The axis follows the spend/query 0=off convention, NOT the turns/tokens
	// Unbounded=-1 sentinel. A session wired with no calls= envelope must dispatch
	// freely — otherwise every existing table-wired caller would be cut at its first
	// tool call the moment this wiring landed.
	p := NewMockPlanner("mock")
	tbl := session.NewTable()
	const trace = "arm-toolcalls-off"
	tbl.SetBudget(trace, session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded})

	m, err := RunArm(context.Background(), p, DefaultTask, false, 20, nil, WithSessionTable(tbl, trace))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if m.StoppedBySession != "" {
		t.Fatalf("unconfigured tool-call axis cut the run: StoppedBySession=%q, want empty", m.StoppedBySession)
	}
	if m.FinalAnswer == "" || m.ToolCalls == 0 {
		t.Fatalf("unconfigured tool-call axis changed behavior: %d tool calls, final=%q", m.ToolCalls, m.FinalAnswer)
	}
}

// TestRunArmToolCallBudgetDoesNotUsurpDrain pins the axis boundary the per-call debit
// must respect: it cuts ONLY for its own ceiling. session.Table.DebitToolCall shares its
// run-state head with Decide, so a session drained mid-turn answers !Proceed carrying
// DRAINING — and honoring that reason at the dispatch site would convert every drain into
// a terminate, silently breaking the #2758 contract. Here the tool-call axis is CONFIGURED
// with budget to spare while the session drains mid-turn: both announced calls must still
// dispatch and the stop must be the boundary's DRAINED, not a budget cut.
func TestRunArmToolCallBudgetDoesNotUsurpDrain(t *testing.T) {
	tbl := session.NewTable()
	const trace = "arm-toolcalls-drain"
	tbl.SetBudget(trace, session.Budget{
		TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ToolCallsLeft: 10,
	})
	p := &midTurnFlipPlanner{flip: func() {
		if _, ok := tbl.Transition(trace, session.Draining, ""); !ok {
			t.Errorf("mid-turn drain write refused on a live session")
		}
	}}

	m, err := RunArm(context.Background(), p, "task", false, 3, nil, WithSessionTable(tbl, trace))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if m.ToolCalls != 2 {
		t.Fatalf("drain with a configured tool-call axis dispatched %d calls, want 2 — the budget hook must not usurp the drain contract", m.ToolCalls)
	}
	if m.StoppedBySession != session.ReasonDrained {
		t.Fatalf("StoppedBySession = %q, want %q (the boundary owns a drain, not the per-call debit)", m.StoppedBySession, session.ReasonDrained)
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
