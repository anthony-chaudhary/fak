package agent

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

type scriptedStreamingPlanner struct {
	turns []*Completion
	n     int
}

func (p *scriptedStreamingPlanner) Model() string            { return "scripted-streaming-model" }
func (p *scriptedStreamingPlanner) StreamingSupported() bool { return true }

func (p *scriptedStreamingPlanner) Complete(_ context.Context, _ []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	c := p.turns[p.n]
	if p.n < len(p.turns)-1 {
		p.n++
	}
	return c, nil
}

func (p *scriptedStreamingPlanner) CompleteStream(ctx context.Context, sink StreamSink, msgs []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error) {
	c := p.turns[p.n]
	if p.n < len(p.turns)-1 {
		p.n++
	}
	if sink != nil && c.Message.Content != "" {
		chunks := strings.SplitAfter(c.Message.Content, " ")
		for _, chunk := range chunks {
			if err := sink(chunk); err != nil {
				return nil, err
			}
		}
	}
	return c, nil
}

func TestRunArmStreamEnvelopeBroadcast(t *testing.T) {
	planner := &scriptedStreamingPlanner{
		turns: []*Completion{
			{
				Message: Message{
					Content: "Checking user details ",
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
				Message: Message{
					Content: "User Mia Li is verified with gold membership.",
				},
			},
		},
	}

	var mu sync.Mutex
	var envelopes []harnesskit.Envelope
	sink := func(env harnesskit.Envelope) {
		mu.Lock()
		defer mu.Unlock()
		envelopes = append(envelopes, env)
	}

	metrics, err := RunArmStream(
		context.Background(),
		planner,
		"Look up user mia_li_3668",
		false,
		5,
		nil,
		nil,
		WithEnvelopeSink(sink),
	)
	if err != nil {
		t.Fatalf("RunArmStream failed: %v", err)
	}
	if metrics.Turns != 2 {
		t.Fatalf("metrics.Turns = %d, want 2", metrics.Turns)
	}
	if metrics.ToolCalls != 1 {
		t.Fatalf("metrics.ToolCalls = %d, want 1", metrics.ToolCalls)
	}

	if len(envelopes) == 0 {
		t.Fatal("expected envelopes to be broadcast, got 0")
	}

	// Verify sequential and monotonic sequence numbering starting at 1
	for i, env := range envelopes {
		expectedSeq := uint64(i + 1)
		if env.Sequence != expectedSeq {
			t.Errorf("envelope[%d].Sequence = %d, want %d", i, env.Sequence, expectedSeq)
		}
		if err := env.Validate(); err != nil {
			t.Errorf("envelope[%d] validation failed: %v", i, err)
		}
		if !env.Known() {
			t.Errorf("envelope[%d] type %q is not known", i, env.Type)
		}
		if env.Version != harnesskit.ProtocolVersion {
			t.Errorf("envelope[%d].Version = %q, want %q", i, env.Version, harnesskit.ProtocolVersion)
		}
		if env.RunID == "" {
			t.Errorf("envelope[%d].RunID is empty", i)
		}
		if env.EventID == "" {
			t.Errorf("envelope[%d].EventID is empty", i)
		}
	}

	// Verify presence and ordering of event types:
	// Turn 1 should stream deltas, then tool started, then tool completed.
	// Turn 2 should stream deltas.
	var hasDelta, hasToolStarted, hasToolCompleted bool
	var toolStartedIdx, toolCompletedIdx int
	for i, env := range envelopes {
		switch env.Type {
		case harnesskit.EventMessageDelta:
			hasDelta = true
			var payload harnesskit.MessagePayload
			if err := env.DecodePayload(&payload); err != nil {
				t.Fatalf("decode MessagePayload: %v", err)
			}
			if payload.Text == "" {
				t.Errorf("envelope[%d] MessagePayload text is empty", i)
			}
		case harnesskit.EventToolStarted:
			hasToolStarted = true
			toolStartedIdx = i
			var payload harnesskit.ToolPayload
			if err := env.DecodePayload(&payload); err != nil {
				t.Fatalf("decode ToolPayload on start: %v", err)
			}
			if payload.Name != toolGetUser {
				t.Errorf("tool started name = %q, want %q", payload.Name, toolGetUser)
			}
			if payload.CallID != "call_get_user_1" {
				t.Errorf("tool started call_id = %q, want call_get_user_1", payload.CallID)
			}
		case harnesskit.EventToolCompleted:
			hasToolCompleted = true
			toolCompletedIdx = i
			var payload harnesskit.ToolPayload
			if err := env.DecodePayload(&payload); err != nil {
				t.Fatalf("decode ToolPayload on complete: %v", err)
			}
			if payload.Name != toolGetUser {
				t.Errorf("tool completed name = %q, want %q", payload.Name, toolGetUser)
			}
			if payload.CallID != "call_get_user_1" {
				t.Errorf("tool completed call_id = %q, want call_get_user_1", payload.CallID)
			}
			if payload.Status != "ok" {
				t.Errorf("tool completed status = %q, want ok", payload.Status)
			}
		}
	}

	if !hasDelta {
		t.Error("missing EventMessageDelta envelope")
	}
	if !hasToolStarted {
		t.Error("missing EventToolStarted envelope")
	}
	if !hasToolCompleted {
		t.Error("missing EventToolCompleted envelope")
	}
	if toolStartedIdx >= toolCompletedIdx {
		t.Errorf("tool started index %d should precede completed index %d", toolStartedIdx, toolCompletedIdx)
	}

	t.Run("fak_arm", func(t *testing.T) {
		planner := &scriptedStreamingPlanner{
			turns: []*Completion{
				{
					Message: Message{
						Content: "Finding flight details ",
						ToolCalls: []ToolCall{
							{
								ID: "call_search_1",
								Function: Func{
									Name:      toolSearch,
									Arguments: `{"origin":"SFO","destination":"JFK","date":"2026-10-01"}`,
								},
							},
						},
					},
				},
				{
					Message: Message{
						Content: "Flight UA100 found.",
					},
				},
			},
		}

		var mu sync.Mutex
		var envelopes []harnesskit.Envelope
		sink := func(env harnesskit.Envelope) {
			mu.Lock()
			defer mu.Unlock()
			envelopes = append(envelopes, env)
		}

		metrics, err := RunArmStream(
			context.Background(),
			planner,
			"Search flights SFO to JFK",
			true,
			5,
			nil,
			nil,
			WithEnvelopeSink(sink),
		)
		if err != nil {
			t.Fatalf("RunArmStream failed: %v", err)
		}
		if metrics.Turns != 2 {
			t.Fatalf("metrics.Turns = %d, want 2", metrics.Turns)
		}
		if len(envelopes) == 0 {
			t.Fatal("expected envelopes to be broadcast, got 0")
		}
		for i, env := range envelopes {
			if env.Sequence != uint64(i+1) {
				t.Fatalf("envelope[%d].Sequence = %d, want %d", i, env.Sequence, i+1)
			}
			if err := env.Validate(); err != nil {
				t.Fatalf("envelope[%d].Validate: %v", i, err)
			}
		}
	})
}
