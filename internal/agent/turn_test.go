package agent

// turn_test.go — the #1318 acceptance witness: the suspend-and-resume turn primitive.
// Two layers: (1) the Turn primitive in isolation (suspend holds a provisional effect
// WITHOUT advancing the turn index; resume promotes on a match and squashes on a miss);
// and (2) RunArm driven through an effect-free speculated call — Speculator.Predict
// reached from a non-test caller for the first time, the turn SUSPENDS rather than
// terminates, and match->Committed / mismatch->Squashed, all within the same turn count.

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func inlineRef(s string) abi.Ref {
	return abi.Ref{Kind: abi.RefInline, Inline: []byte(s), Len: int64(len(s))}
}

func specCall(tool, args string, epoch uint64) *abi.ToolCall {
	return &abi.ToolCall{Tool: tool, Args: inlineRef(args), Spec: abi.SpeculationContext{Speculative: true, Epoch: epoch}}
}

// TestTurnSuspendResumeCommit: a suspended speculation whose authoritative next call
// MATCHES is promoted — the BufferSink holds the committed effect — and the turn index
// never moved.
func TestTurnSuspendResumeCommit(t *testing.T) {
	ctx := context.Background()
	turn := NewTurn(3)
	pred := specCall("search_direct_flight", `{"o":"SFO"}`, 7)

	sink := turn.Suspend(pred, inlineRef("flight-result"))
	if !turn.Suspended() {
		t.Fatal("Suspend must leave the turn suspended (not terminated)")
	}
	if turn.Index() != 3 {
		t.Fatalf("Suspend advanced the turn index to %d, want it pinned at 3", turn.Index())
	}
	bs := sink.(*abi.BufferSink)
	if len(bs.Committed()) != 0 {
		t.Fatal("a suspended (not-yet-resolved) effect must NOT be committed yet")
	}

	// Authoritative call matches the prediction -> Committed.
	out, err := turn.Resume(ctx, &abi.ToolCall{Tool: "search_direct_flight", Args: inlineRef(`{"o":"SFO"}`)})
	if err != nil {
		t.Fatalf("Resume: %v", err)
	}
	if out != abi.OutcomeCommitted {
		t.Fatalf("matching call: outcome = %v, want OutcomeCommitted", out)
	}
	if turn.Suspended() {
		t.Fatal("after Resume the turn must no longer be suspended")
	}
	if turn.Index() != 3 {
		t.Fatalf("Resume moved the turn index to %d, want it still 3 (one turn index across suspend/resume)", turn.Index())
	}
	if got := turn.Sink().Committed(); len(got) != 1 {
		t.Fatalf("committed effects = %d, want 1 (the promoted provisional result)", len(got))
	}
}

// TestTurnSuspendResumeSquash: a mismatching authoritative call squashes the speculation
// — the BufferSink retains nothing, the executable form of "squash undoes the effect".
func TestTurnSuspendResumeSquash(t *testing.T) {
	ctx := context.Background()
	turn := NewTurn(1)
	pred := specCall("search_direct_flight", `{"o":"SFO"}`, 9)
	turn.Suspend(pred, inlineRef("flight-result"))

	out, err := turn.Resume(ctx, &abi.ToolCall{Tool: "search_direct_flight", Args: inlineRef(`{"o":"JFK"}`)}) // different args
	if err != nil {
		t.Fatalf("Resume: %v", err)
	}
	if out != abi.OutcomeSquashed {
		t.Fatalf("mismatching call: outcome = %v, want OutcomeSquashed", out)
	}
	if got := turn.Sink().Committed(); len(got) != 0 {
		t.Fatalf("squashed speculation left %d committed effects, want 0 (squash retracts)", len(got))
	}
	if turn.Sink().PendingEpochs() != 0 {
		t.Fatalf("squashed speculation left %d pending epochs, want 0 (no leak)", turn.Sink().PendingEpochs())
	}
}

// scriptedPlanner returns a fixed sequence of completions, one per Complete call, so a
// multi-turn RunArm flow is fully deterministic regardless of message state.
type scriptedPlanner struct {
	turns []*Completion
	n     int
}

func (p *scriptedPlanner) Complete(_ context.Context, _ []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	c := p.turns[p.n]
	if p.n < len(p.turns)-1 {
		p.n++
	}
	return c, nil
}
func (p *scriptedPlanner) Model() string { return "scripted" }

func toolCallTurn(tool, args string) *Completion {
	return &Completion{Message: Message{ToolCalls: []ToolCall{{ID: "c", Function: Func{Name: tool, Arguments: args}}}}}
}

// searchPattern predicts search_direct_flight(args) after a get_user_details turn, an
// effect-free read (readOnlyHint+idempotentHint, not write-shaped) the default-deny-on-
// effects gate admits.
func searchPattern(args string) abi.SpecPattern {
	return abi.SpecPattern{
		Signature:   "get_user_details",
		PredictTool: "search_direct_flight",
		SuccessProb: 1.0,
		Meta:        map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
		DeriveArgs:  func([]*abi.Result) (abi.Ref, bool) { return inlineRef(args), true },
	}
}

// runSpeculated drives RunArm over a 3-turn script (get_user -> search -> final answer)
// with a speculator that predicts the search call after get_user. predictedArgs controls
// match vs miss against the script's actual search args.
func runSpeculated(t *testing.T, scriptArgs, predictedArgs string) ArmMetrics {
	t.Helper()
	spec := abi.NewSpeculator(0)
	spec.Learn(searchPattern(predictedArgs))
	planner := &scriptedPlanner{turns: []*Completion{
		toolCallTurn("get_user_details", `{"user_id":"u1"}`),
		toolCallTurn("search_direct_flight", scriptArgs),
		{Message: Message{Content: "done"}}, // final answer, no tool call
	}}
	m, err := RunArm(context.Background(), planner, "find me a flight", true, 10, nil, WithSpeculator(spec))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	return m
}

// TestRunArmSpeculatesAndCommitsWithinOneTurn is the #1318 integration witness: RunArm
// reaches Speculator.Predict (the first non-test caller), suspends an effect-free
// predicted call, and PROMOTES it when the model's authoritative next call matches —
// without inflating the model-turn count (speculation rides within a turn).
func TestRunArmSpeculatesAndCommitsWithinOneTurn(t *testing.T) {
	args := `{"origin":"SFO","destination":"JFK","date":"2026-07-01"}`
	m := runSpeculated(t, args, args) // predicted == script -> match

	if m.SpecIssued == 0 {
		t.Fatal("RunArm never issued a speculation — Speculator.Predict was not reached from the loop")
	}
	if m.SpecCommitted == 0 {
		t.Fatalf("a matching authoritative call must COMMIT the speculation; got issued=%d committed=%d squashed=%d", m.SpecIssued, m.SpecCommitted, m.SpecSquashed)
	}
	if m.SpecIssued != m.SpecCommitted+m.SpecSquashed {
		t.Fatalf("speculation leaked: issued=%d != committed=%d + squashed=%d", m.SpecIssued, m.SpecCommitted, m.SpecSquashed)
	}
	// The model-turn count is the 3 scripted Complete calls — speculation did NOT advance
	// it (the suspend/resume rides within a turn boundary, never as an extra turn).
	noSpec, err := RunArm(context.Background(), &scriptedPlanner{turns: []*Completion{
		toolCallTurn("get_user_details", `{"user_id":"u1"}`),
		toolCallTurn("search_direct_flight", args),
		{Message: Message{Content: "done"}},
	}}, "find me a flight", true, 10, nil)
	if err != nil {
		t.Fatalf("RunArm (no spec): %v", err)
	}
	if m.Turns != noSpec.Turns {
		t.Fatalf("speculation changed the model-turn count: with-spec=%d, no-spec=%d (must be equal)", m.Turns, noSpec.Turns)
	}
}

// TestRunArmSpeculationSquashesOnMiss: when the authoritative next call differs from the
// prediction, the speculation is SQUASHED — no commit, no leak.
func TestRunArmSpeculationSquashesOnMiss(t *testing.T) {
	m := runSpeculated(t, `{"origin":"SFO","destination":"JFK","date":"2026-07-01"}`, `{"origin":"LAX","destination":"BOS","date":"2026-08-02"}`)
	if m.SpecIssued == 0 {
		t.Fatal("RunArm never issued a speculation")
	}
	if m.SpecSquashed == 0 {
		t.Fatalf("a mismatching authoritative call must SQUASH the speculation; got issued=%d committed=%d squashed=%d", m.SpecIssued, m.SpecCommitted, m.SpecSquashed)
	}
	if m.SpecCommitted != 0 {
		t.Fatalf("a mismatch must not commit; got committed=%d", m.SpecCommitted)
	}
}
