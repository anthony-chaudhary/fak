package agent

// loop_terminate_test.go — the loop-side witness of the drain/terminate distinction
// (#2758, epic #2753). Both ops stop a running session; the DIFFERENCE is the whole
// point and is what this file proves from the loop's observable behavior:
//
//	drain (OpCancel, `fak session stop`)          — the in-flight turn runs to
//	  completion: every tool call the model already announced is dispatched, and the
//	  stop is taken at the NEXT TURN BOUNDARY with the closed DRAINING reason.
//	terminate (OpTerminate, `fak session terminate`) — the arm stops at the next SAFE
//	  POINT inside the turn: no further tool call dispatches (the announced calls are
//	  abandoned), the in-flight model call's context is cancelled, and the closed
//	  TERMINATED reason is recorded.
//
// The two runs below are byte-identical except for the run-state each flips to
// mid-turn, so any behavioral difference observed IS the drain/terminate contract.

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// midTurnFlipPlanner flips the session's run-state from INSIDE the first model call
// — after the boundary gate admitted the turn, before any tool dispatches — then
// answers with two tool calls. That timing is what makes the drain/terminate
// difference observable: a drain must still dispatch both announced calls, a
// terminate must dispatch neither. A second call (drain reaches the next boundary
// via the gate, so it should not happen) returns a plain final answer.
type midTurnFlipPlanner struct {
	flip  func()
	calls int
}

func (p *midTurnFlipPlanner) Model() string { return "mid-turn-flip" }

func (p *midTurnFlipPlanner) Complete(_ context.Context, _ []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	p.calls++
	if p.calls == 1 {
		p.flip()
		return &Completion{
			Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{
				{ID: "call_1", Type: "function", Function: Func{Name: "search", Arguments: `{"query":"first"}`}},
				{ID: "call_2", Type: "function", Function: Func{Name: "search", Arguments: `{"query":"second"}`}},
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

// runMidTurnFlip runs one arm whose session flips to `to` mid-turn and returns the
// metrics, the planner (for its call count), and the table (for the final state).
func runMidTurnFlip(t *testing.T, trace string, to session.RunState) (ArmMetrics, *midTurnFlipPlanner, *session.Table) {
	t.Helper()
	tbl := session.NewTable()
	tbl.Decide(trace) // seed a live record so the mid-turn flip is a legal transition
	p := &midTurnFlipPlanner{flip: func() {
		// An empty reason lets the boundary/safe-point finalization stamp the closed
		// stop token (DRAINING / TERMINATED) rather than echoing operator prose.
		if _, ok := tbl.Transition(trace, to, ""); !ok {
			t.Errorf("mid-turn %v write refused on a live session", to)
		}
	}}
	m, err := RunArm(context.Background(), p, "task", false, 3, nil, WithSessionTable(tbl, trace))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	return m, p, tbl
}

// TestDrainFinishesInFlightTurnTerminateDoesNot is the #2758 acceptance witness: the
// drain/terminate distinction observed loop-side under identical conditions.
func TestDrainFinishesInFlightTurnTerminateDoesNot(t *testing.T) {
	t.Run("drain", func(t *testing.T) {
		m, p, tbl := runMidTurnFlip(t, "drain-vs-term-2758-drain", session.Draining)
		// The in-flight turn's remaining work RAN: both announced tool calls
		// dispatched, and the model was never called again (the stop was taken at
		// the next boundary, not by re-entering the model).
		if m.ToolCalls != 2 {
			t.Fatalf("drain dispatched %d tool calls, want 2 — a drain must let the in-flight turn finish its announced work", m.ToolCalls)
		}
		if p.calls != 1 {
			t.Fatalf("planner called %d times, want 1 — the drain stop is taken at the boundary before the next model call", p.calls)
		}
		if m.StoppedBySession != session.ReasonDrained {
			t.Fatalf("StoppedBySession = %q, want %q", m.StoppedBySession, session.ReasonDrained)
		}
		if st := tbl.Get("drain-vs-term-2758-drain"); st.Run != session.Stopped {
			t.Fatalf("post-drain run-state = %v, want Stopped", st.Run)
		}
	})
	t.Run("terminate", func(t *testing.T) {
		m, p, tbl := runMidTurnFlip(t, "drain-vs-term-2758-term", session.Terminating)
		// The safe point is BEFORE the next tool dispatch: none of the announced
		// calls ran, and no further model call happened.
		if m.ToolCalls != 0 {
			t.Fatalf("terminate dispatched %d tool calls, want 0 — a terminated session starts no new work", m.ToolCalls)
		}
		if p.calls != 1 {
			t.Fatalf("planner called %d times, want 1 — terminate must not re-enter the model", p.calls)
		}
		if m.StoppedBySession != session.ReasonTerminated {
			t.Fatalf("StoppedBySession = %q, want %q", m.StoppedBySession, session.ReasonTerminated)
		}
		if st := tbl.Get("drain-vs-term-2758-term"); st.Run != session.Stopped {
			t.Fatalf("post-terminate run-state = %v, want Stopped (the safe point finalizes Terminating)", st.Run)
		}
	})
}

// TestTerminateCancelsInFlightModelCall proves the other half of the forceful stop:
// a model call ALREADY IN FLIGHT when the terminate lands is cancelled via its
// context — the arm does not wait for the completion to end naturally — and the
// cancellation is recorded as the typed TERMINATED stop, never surfaced as an error.
func TestTerminateCancelsInFlightModelCall(t *testing.T) {
	const trace = "drain-vs-term-2758-inflight"
	tbl := session.NewTable()
	tbl.Decide(trace)
	entered := make(chan struct{})
	p := &blockingPlanner{entered: entered}
	done := make(chan struct{})
	var m ArmMetrics
	var err error
	go func() {
		defer close(done)
		m, err = RunArm(context.Background(), p, "task", false, 3, nil, WithSessionTable(tbl, trace))
	}()
	<-entered // the model call is in flight
	if _, ok := tbl.Transition(trace, session.Terminating, ""); !ok {
		t.Fatal("terminate write refused on a live session")
	}
	<-done // must return promptly: the terminate signal cancels the call's context
	if err != nil {
		t.Fatalf("RunArm returned an error for a session terminate: %v (the op working is not a failure)", err)
	}
	if m.StoppedBySession != session.ReasonTerminated {
		t.Fatalf("StoppedBySession = %q, want %q", m.StoppedBySession, session.ReasonTerminated)
	}
	if st := tbl.Get(trace); st.Run != session.Stopped {
		t.Fatalf("post-terminate run-state = %v, want Stopped", st.Run)
	}
}

// blockingPlanner parks its Complete call on the caller's context, signaling entry —
// the "in-flight model call" a terminate must cancel rather than wait out.
type blockingPlanner struct{ entered chan struct{} }

func (p *blockingPlanner) Model() string { return "blocking" }

func (p *blockingPlanner) Complete(ctx context.Context, _ []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	close(p.entered)
	<-ctx.Done()
	return nil, ctx.Err()
}
