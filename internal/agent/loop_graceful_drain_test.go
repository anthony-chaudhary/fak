package agent

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

type gracefulDrainTestPlanner struct {
	calls       int
	seenTools   [][]ToolDef
	finalText   string
	failOnDrain bool
}

func (p *gracefulDrainTestPlanner) Model() string {
	return "mock-drain-model"
}

func (p *gracefulDrainTestPlanner) Complete(_ context.Context, _ []Message, tools []ToolDef, _ ...SampleOpt) (*Completion, error) {
	p.calls++
	p.seenTools = append(p.seenTools, tools)
	if len(tools) == 0 {
		if p.failOnDrain {
			return nil, errors.New("synthesis model failure")
		}
		return &Completion{
			Message: Message{
				Role:    RoleAssistant,
				Content: p.finalText,
			},
			FinishReason: "stop",
			Usage: Usage{
				PromptTokens:     12,
				CompletionTokens: 8,
			},
		}, nil
	}

	return &Completion{
		Message: Message{
			Role: RoleAssistant,
			ToolCalls: []ToolCall{
				{
					ID:   fmt.Sprintf("call_%d", p.calls),
					Type: "function",
					Function: Func{
						Name:      toolGetUser,
						Arguments: `{"user_id":"mia_li_3668"}`,
					},
				},
			},
		},
		FinishReason: "tool_calls",
		Usage: Usage{
			PromptTokens:     10,
			CompletionTokens: 5,
		},
	}, nil
}

func TestGracefulDrain_Disabled(t *testing.T) {
	p := &gracefulDrainTestPlanner{finalText: "Summary of actions taken"}
	var traceLog []traceEvent
	m, err := RunArm(context.Background(), p, "task", false, 2, &traceLog, WithGracefulDrain(false))
	if err != nil {
		t.Fatalf("RunArm failed: %v", err)
	}
	if m.GracefulDrained {
		t.Fatalf("expected GracefulDrained to be false, got true")
	}
	if m.SynthesizedFinalTurn {
		t.Fatalf("expected SynthesizedFinalTurn to be false, got true")
	}
	if m.FinalAnswer != "" {
		t.Fatalf("expected FinalAnswer to be empty, got %q", m.FinalAnswer)
	}
	if !m.HitTurnCap {
		t.Fatalf("expected HitTurnCap to be true, got false")
	}
	if p.calls != 2 {
		t.Fatalf("expected 2 planner calls, got %d", p.calls)
	}
}

func TestGracefulDrain_Default(t *testing.T) {
	p := &gracefulDrainTestPlanner{finalText: "Summary of actions taken"}
	var traceLog []traceEvent
	m, err := RunArm(context.Background(), p, "task", false, 2, &traceLog)
	if err != nil {
		t.Fatalf("RunArm failed: %v", err)
	}
	if m.GracefulDrained {
		t.Fatalf("expected GracefulDrained to be false, got true")
	}
	if m.SynthesizedFinalTurn {
		t.Fatalf("expected SynthesizedFinalTurn to be false, got true")
	}
	if m.FinalAnswer != "" {
		t.Fatalf("expected FinalAnswer to be empty, got %q", m.FinalAnswer)
	}
	if !m.HitTurnCap {
		t.Fatalf("expected HitTurnCap to be true, got false")
	}
	if p.calls != 2 {
		t.Fatalf("expected 2 planner calls, got %d", p.calls)
	}
}

func TestGracefulDrain_Enabled(t *testing.T) {
	p := &gracefulDrainTestPlanner{finalText: "Natural language synthesis"}
	var traceLog []traceEvent
	var progressEvents []ProgressEvent
	m, err := RunArm(context.Background(), p, "task", false, 2, &traceLog,
		WithGracefulDrain(true),
		WithProgressObserver(func(ev ProgressEvent) {
			progressEvents = append(progressEvents, ev)
		}),
	)
	if err != nil {
		t.Fatalf("RunArm failed: %v", err)
	}
	if !m.GracefulDrained {
		t.Fatalf("expected GracefulDrained to be true")
	}
	if !m.SynthesizedFinalTurn {
		t.Fatalf("expected SynthesizedFinalTurn to be true")
	}
	if m.FinalAnswer != "Natural language synthesis" {
		t.Fatalf("expected FinalAnswer %q, got %q", "Natural language synthesis", m.FinalAnswer)
	}
	if !m.HitTurnCap {
		t.Fatalf("expected HitTurnCap to be true")
	}
	if p.calls != 3 {
		t.Fatalf("expected 3 planner calls (2 turn calls + 1 synthesis call), got %d", p.calls)
	}
	if len(p.seenTools) != 3 {
		t.Fatalf("expected 3 seen tool entries, got %d", len(p.seenTools))
	}
	if len(p.seenTools[2]) != 0 {
		t.Fatalf("expected final synthesis turn to have nil/empty tools, got %d tools", len(p.seenTools[2]))
	}

	var sawSynthesisDone bool
	for _, ev := range progressEvents {
		if ev.Kind == ProgressTurnDone && ev.Turn == 3 {
			sawSynthesisDone = true
			break
		}
	}
	if !sawSynthesisDone {
		t.Fatalf("expected progress observer to record ProgressTurnDone for turn 3")
	}
}

func TestGracefulDrain_ArmRunnerDirect(t *testing.T) {
	cfg := runConfig{gracefulDrain: true}
	var metrics ArmMetrics
	p := &gracefulDrainTestPlanner{finalText: "Summary of actions taken"}
	r := &armRunner{
		cfg:      &cfg,
		metrics:  &metrics,
		messages: []Message{{Role: RoleUser, Content: "task"}},
		tools:    []ToolDef{{Type: "function", Function: ToolDefFunction{Name: toolGetUser}}},
		complete: func(ctx context.Context, msgs []Message, tools []ToolDef, sink StreamSink, opts ...SampleOpt) (*Completion, error) {
			return p.Complete(ctx, msgs, tools, opts...)
		},
		stopTerminated: func() bool { return false },
	}

	err := r.run(context.Background(), 2)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if !metrics.GracefulDrained {
		t.Fatalf("expected GracefulDrained to be true")
	}
	if !metrics.SynthesizedFinalTurn {
		t.Fatalf("expected SynthesizedFinalTurn to be true")
	}
	if metrics.FinalAnswer != "Summary of actions taken" {
		t.Fatalf("expected FinalAnswer %q, got %q", "Summary of actions taken", metrics.FinalAnswer)
	}
	if !metrics.HitTurnCap {
		t.Fatalf("expected HitTurnCap to be true")
	}

	if len(r.messages) == 0 {
		t.Fatalf("expected r.messages to be non-empty")
	}
	lastMsg := r.messages[len(r.messages)-1]
	if lastMsg.Role != RoleAssistant {
		t.Fatalf("expected last message role %q, got %q", RoleAssistant, lastMsg.Role)
	}
	if lastMsg.Content != "Summary of actions taken" {
		t.Fatalf("expected last message content %q, got %q", "Summary of actions taken", lastMsg.Content)
	}
}

func TestGracefulDrain_SynthesisError(t *testing.T) {
	p := &gracefulDrainTestPlanner{
		finalText:   "Synthesis",
		failOnDrain: true,
	}
	var traceLog []traceEvent
	m, err := RunArm(context.Background(), p, "task", false, 2, &traceLog, WithGracefulDrain(true))
	if err == nil {
		t.Fatalf("expected error from failed synthesis turn, got nil")
	}
	if !m.GracefulDrained {
		t.Fatalf("expected GracefulDrained to be true")
	}
	if m.SynthesizedFinalTurn {
		t.Fatalf("expected SynthesizedFinalTurn to be false")
	}
}
