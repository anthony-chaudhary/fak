package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/stopgate"
)

type stopgateTestPlanner struct {
	turn    int
	seen    []Message
	answers []Completion
}

func (p *stopgateTestPlanner) Model() string { return "stopgate-test" }
func (p *stopgateTestPlanner) Complete(_ context.Context, msgs []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	p.seen = append(p.seen, msgs...)
	idx := p.turn
	p.turn++
	if idx < len(p.answers) {
		return &p.answers[idx], nil
	}
	return &Completion{
		Message:      Message{Role: RoleAssistant, Content: "fallback done"},
		FinishReason: "stop",
		Usage:        Usage{CompletionTokens: 1},
	}, nil
}

func TestStopgateDenyAllNudgeContinuationInAgentLoop(t *testing.T) {
	// Turn 1: Model calls an unknown/denied naive tool "bad_tool"
	// Turn 2: Model attempts to stop with a text answer "I cannot proceed."
	// Stopgate catches the unchosen stop after deny-all and injects Nudge guidance
	// Turn 3: Model finishes
	p := &stopgateTestPlanner{
		answers: []Completion{
			{
				Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{
					{ID: "c1", Type: "function", Function: Func{Name: "bad_tool", Arguments: `{}`}},
				}},
				FinishReason: "tool_calls",
				Usage:        Usage{CompletionTokens: 1},
			},
			{
				Message:      Message{Role: RoleAssistant, Content: "I cannot proceed."},
				FinishReason: "stop",
				Usage:        Usage{CompletionTokens: 1},
			},
			{
				Message:      Message{Role: RoleAssistant, Content: "Work finished finally."},
				FinishReason: "stop",
				Usage:        Usage{CompletionTokens: 1},
			},
		},
	}

	ladderCfg := stopgate.DefaultLadderConfig()
	ladderCfg.WarnAt = 2
	ladderCfg.FinalAt = 3
	ladderCfg.Max = 4

	metrics, err := RunArm(context.Background(), p, "test task", false, 5, nil, WithStopLadderConfig(ladderCfg))
	if err != nil {
		t.Fatalf("RunArm failed: %v", err)
	}

	// Verify that the continuation message was injected into the conversation
	var foundNudge bool
	for _, m := range p.seen {
		if m.Role == RoleUser && strings.Contains(m.Content, "ALLOWED alternative") {
			foundNudge = true
			break
		}
	}
	if !foundNudge {
		t.Fatalf("expected deny-all continuation prompt to be spliced into planner messages, but none was found. Messages: %+v", p.seen)
	}
	if metrics.FinalAnswer != "Work finished finally." {
		t.Fatalf("expected final answer after continuation, got %q", metrics.FinalAnswer)
	}
}

func TestStopgateNoAllowedPathCleanWrapupInAgentLoop(t *testing.T) {
	// Turn 1: Tool call fails
	// Turn 2: Model acknowledges boundary cleanly with "no allowed path: TRUST_VIOLATION"
	// Stopgate recognizes clean wrap-up and allows the stop immediately
	p := &stopgateTestPlanner{
		answers: []Completion{
			{
				Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{
					{ID: "c1", Type: "function", Function: Func{Name: "bad_tool", Arguments: `{}`}},
				}},
				FinishReason: "tool_calls",
				Usage:        Usage{CompletionTokens: 1},
			},
			{
				Message:      Message{Role: RoleAssistant, Content: "Task blocked; no allowed path: TRUST_VIOLATION boundary enforced."},
				FinishReason: "stop",
				Usage:        Usage{CompletionTokens: 1},
			},
		},
	}

	metrics, err := RunArm(context.Background(), p, "test task", false, 5, nil)
	if err != nil {
		t.Fatalf("RunArm failed: %v", err)
	}

	if metrics.FinalAnswer != "Task blocked; no allowed path: TRUST_VIOLATION boundary enforced." {
		t.Fatalf("expected clean wrap-up final answer, got %q", metrics.FinalAnswer)
	}
}

func TestStopgateNoAllowedPathRequiresWitnessInAgentLoop(t *testing.T) {
	satisfied := false
	p := &stopgateTestPlanner{
		answers: []Completion{
			{
				Message:      Message{Role: RoleAssistant, Content: "Blocked; no allowed path: TRUST_VIOLATION"},
				FinishReason: "stop",
				Usage:        Usage{CompletionTokens: 1},
			},
			{
				Message:      Message{Role: RoleAssistant, Content: "Landed verified commit."},
				FinishReason: "stop",
				Usage:        Usage{CompletionTokens: 1},
			},
		},
	}

	metrics, err := RunArm(context.Background(), p, "test task", false, 5, nil, WithFinalGate(func() (bool, string) {
		if !satisfied {
			satisfied = true
			return false, "missing:stamp"
		}
		return true, ""
	}))
	if err != nil {
		t.Fatalf("RunArm failed: %v", err)
	}

	if metrics.FinalAnswer != "Landed verified commit." {
		t.Fatalf("expected loop to continue past unwitnessed no allowed path stop, got %q", metrics.FinalAnswer)
	}
}

func TestStopgateDenyAllGiveUpStandDownInAgentLoop(t *testing.T) {
	// Configure SameStop=2 so that after 2 identical consecutive denials (same tool+reason),
	// give-up stands down and allows the stop
	ladderCfg := stopgate.DefaultLadderConfig()
	ladderCfg.SameStop = 2

	// Turn 1: tool fails (sameConsecutive=1)
	// Turn 2: same tool fails (sameConsecutive=2 >= SameStop=2)
	// Turn 3: model attempts stop with text -> give-up stand-down allows stop!
	p := &stopgateTestPlanner{
		answers: []Completion{
			{
				Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{
					{ID: "c1", Type: "function", Function: Func{Name: "bad_tool_1", Arguments: `{}`}},
				}},
				FinishReason: "tool_calls",
				Usage:        Usage{CompletionTokens: 1},
			},
			{
				Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{
					{ID: "c2", Type: "function", Function: Func{Name: "bad_tool_1", Arguments: `{}`}},
				}},
				FinishReason: "tool_calls",
				Usage:        Usage{CompletionTokens: 1},
			},
			{
				Message:      Message{Role: RoleAssistant, Content: "Giving up."},
				FinishReason: "stop",
				Usage:        Usage{CompletionTokens: 1},
			},
		},
	}

	metrics, err := RunArm(context.Background(), p, "test task", false, 5, nil, WithStopLadderConfig(ladderCfg))
	if err != nil {
		t.Fatalf("RunArm failed: %v", err)
	}

	if metrics.FinalAnswer != "Giving up." {
		t.Fatalf("expected give-up stand-down final answer, got %q", metrics.FinalAnswer)
	}
}
