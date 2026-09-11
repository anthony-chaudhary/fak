package agent

// governedrun_test.go — the trace-fidelity half of #12764: RunGovernedArmStream must
// deliver the same CallTrace rows the buffered RunGovernedArm does WHILE streaming
// turn content to the sink, and must never let a tool call's arguments ride the
// content stream (tool calls are held for adjudication, exactly as the buffered
// path holds them).

import (
	"context"
	"strings"
	"sync"
	"testing"
)

// traceFidelityPlanner is a separate scripted streaming planner (loop_stream_test.go's
// scriptedStreamingPlanner name is taken in this package namespace). Its
// CompleteStream calls sink synchronously BEFORE returning the turn's Completion,
// so every sink fragment is provably delivered inside the RunGovernedArmStream call.
type traceFidelityPlanner struct {
	turns []*Completion
	n     int
}

func (p *traceFidelityPlanner) Model() string            { return "trace-fidelity-model" }
func (p *traceFidelityPlanner) StreamingSupported() bool { return true }

func (p *traceFidelityPlanner) Complete(_ context.Context, _ []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	c := p.turns[p.n]
	if p.n < len(p.turns)-1 {
		p.n++
	}
	return c, nil
}

func (p *traceFidelityPlanner) CompleteStream(_ context.Context, sink StreamSink, _ []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	c := p.turns[p.n]
	if p.n < len(p.turns)-1 {
		p.n++
	}
	if sink != nil && c.Message.Content != "" {
		for _, chunk := range strings.SplitAfter(c.Message.Content, " ") {
			if err := sink(chunk); err != nil {
				return nil, err
			}
		}
	}
	return c, nil
}

// bufferedOnlyPlanner implements ONLY the Planner interface — no StreamingPlanner
// methods — so RunGovernedArmStream must fall back cleanly to the buffered path.
type bufferedOnlyPlanner struct {
	completions []*Completion
	n           int
}

func (p *bufferedOnlyPlanner) Model() string { return "buffered-only-model" }

func (p *bufferedOnlyPlanner) Complete(_ context.Context, _ []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	c := p.completions[p.n]
	if p.n < len(p.completions)-1 {
		p.n++
	}
	return c, nil
}

// sinkRecorder records every content fragment in delivery order. The planner calls
// the sink synchronously on the same goroutine that runs RunGovernedArmStream, so
// anything recorded before RunGovernedArmStream returns was delivered DURING the
// arm call — the "streamed before return" invariant is structural, not racy.
type sinkRecorder struct {
	mu        sync.Mutex
	fragments []string
}

func (s *sinkRecorder) record(delta string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.fragments = append(s.fragments, delta)
	return nil
}

func (s *sinkRecorder) joined() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return strings.Join(s.fragments, "")
}

func TestRunGovernedArmStreamStreamsAndPreservesTrace(t *testing.T) {
	const turn1Prose = "Checking user details for mia_li_3668 "
	const turn2Prose = "User Mia Li is verified with gold membership."

	planner := &traceFidelityPlanner{
		turns: []*Completion{
			{
				Message: Message{
					Role:    RoleAssistant,
					Content: turn1Prose,
					ToolCalls: []ToolCall{
						{
							ID: "call_get_user_1",
							Function: Func{
								Name:      toolGetUser,
								Arguments: `{"user_id":"mia_li_3668"}`,
							},
						},
					},
				},
			},
			{
				Message: Message{Role: RoleAssistant, Content: turn2Prose},
			},
		},
	}

	rec := &sinkRecorder{}
	m, calls, err := RunGovernedArmStream(context.Background(), planner, "Look up user mia_li_3668", 5, rec.record)
	if err != nil {
		t.Fatalf("RunGovernedArmStream failed: %v", err)
	}

	// Streaming contract: at least one fragment was delivered through the sink, and
	// the joined fragment text is EXACTLY the two turns' prose — proving (a) turn-1
	// content streamed while the arm was running (the planner calls sink
	// synchronously inside CompleteStream, which runs inside the arm call), (b) the
	// interleaved tool call's arguments never leaked onto the content stream (the
	// args JSON would otherwise appear between the two prose turns), and (c) turn-2's
	// final answer streamed after it, with nothing extra in between.
	if len(rec.fragments) == 0 {
		t.Fatal("expected the sink to receive at least one streamed fragment")
	}
	wantJoined := turn1Prose + turn2Prose
	if got := rec.joined(); got != wantJoined {
		t.Fatalf("joined fragments = %q, want %q exactly (interleaved tool-call args must NOT stream)", got, wantJoined)
	}

	// Metrics: the same turn/tool accounting the buffered arm reports.
	if m.Turns != 2 {
		t.Fatalf("metrics.Turns = %d, want 2", m.Turns)
	}
	if m.ToolCalls != 1 {
		t.Fatalf("metrics.ToolCalls = %d, want 1", m.ToolCalls)
	}
	if m.FinalAnswer != turn2Prose {
		t.Fatalf("metrics.FinalAnswer = %q, want turn-2 content %q", m.FinalAnswer, turn2Prose)
	}

	// Trace fidelity: exactly one row for the one tool call, carrying the SAME
	// verdict fields the buffered RunGovernedArm/run artifact records.
	if len(calls) != 1 {
		t.Fatalf("len(calls) = %d, want 1", len(calls))
	}
	row := calls[0]
	if row.Tool != toolGetUser {
		t.Fatalf("row.Tool = %q, want %q", row.Tool, toolGetUser)
	}
	if row.Verdict != "ALLOW" {
		t.Fatalf("row.Verdict = %q, want ALLOW", row.Verdict)
	}
	if row.Turn != 1 {
		t.Fatalf("row.Turn = %d, want 1", row.Turn)
	}
	if row.Arm != "fak" {
		t.Fatalf("row.Arm = %q, want fak", row.Arm)
	}
	if !strings.Contains(row.Args, "mia_li_3668") {
		t.Fatalf("row.Args = %q, want it to contain mia_li_3668", row.Args)
	}
}

func TestRunGovernedArmStreamFallsBackToBuffered(t *testing.T) {
	const finalProse = "Buffered planner delivered the final answer."
	planner := &bufferedOnlyPlanner{
		completions: []*Completion{
			{Message: Message{Role: RoleAssistant, Content: finalProse}},
		},
	}

	rec := &sinkRecorder{}
	m, calls, err := RunGovernedArmStream(context.Background(), planner, "Answer the question", 3, rec.record)
	if err != nil {
		t.Fatalf("RunGovernedArmStream failed: %v", err)
	}
	if m.Turns != 1 {
		t.Fatalf("metrics.Turns = %d, want 1", m.Turns)
	}
	if len(calls) != 0 {
		t.Fatalf("len(calls) = %d, want 0 (no tool calls in the script)", len(calls))
	}
	if m.FinalAnswer != finalProse {
		t.Fatalf("metrics.FinalAnswer = %q, want %q", m.FinalAnswer, finalProse)
	}
	if len(rec.fragments) != 0 {
		t.Fatalf("buffered fallback must emit nothing to the sink, got %q", strings.Join(rec.fragments, ""))
	}
}
