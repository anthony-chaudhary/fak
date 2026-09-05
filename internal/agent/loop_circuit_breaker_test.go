package agent

import (
	"context"
	"strings"
	"testing"
)

type cbTestPlanner struct {
	turn       int
	onComplete func(turn int, messages []Message) (*Completion, error)
}

func (p *cbTestPlanner) Model() string { return "mock-circuit-breaker-model" }

func (p *cbTestPlanner) Complete(ctx context.Context, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error) {
	p.turn++
	return p.onComplete(p.turn, messages)
}

func failCall(id string) []ToolCall {
	return []ToolCall{
		{
			ID: id,
			Function: Func{
				Name:      toolGetUser,
				Arguments: "{}", // missing required user_id -> failure
			},
		},
	}
}

func successCall(id string) []ToolCall {
	return []ToolCall{
		{
			ID: id,
			Function: Func{
				Name:      toolGetUser,
				Arguments: `{"user_id":"mia_li_3668"}`,
			},
		},
	}
}

func denyCall(id string) []ToolCall {
	return []ToolCall{
		{
			ID: id,
			Function: Func{
				Name:      "bad_tool_forbidden",
				Arguments: "{}",
			},
		},
	}
}

func TestAntiLoopCircuitBreaker(t *testing.T) {
	t.Run("GuidanceInjectionAtTwoConsecutiveFailures", func(t *testing.T) {
		var sawGuidanceAtTurn3 bool
		planner := &cbTestPlanner{
			onComplete: func(turn int, messages []Message) (*Completion, error) {
				switch turn {
				case 1:
					return &Completion{
						Message: Message{
							Role:      RoleAssistant,
							ToolCalls: failCall("call_1"),
						},
					}, nil
				case 2:
					return &Completion{
						Message: Message{
							Role:      RoleAssistant,
							ToolCalls: failCall("call_2"),
						},
					}, nil
				case 3:
					for _, m := range messages {
						if m.Role == RoleSystem && strings.Contains(m.Content, "[CIRCUIT BREAKER GUIDANCE]") {
							sawGuidanceAtTurn3 = true
							if !strings.Contains(m.Content, toolGetUser) {
								t.Errorf("expected guidance to mention tool name, got %q", m.Content)
							}
							if !strings.Contains(m.Content, "2 consecutive occurrences") {
								t.Errorf("expected guidance to mention 2 consecutive occurrences, got %q", m.Content)
							}
						}
					}
					return &Completion{
						Message: Message{
							Role:    RoleAssistant,
							Content: "Understood guidance; changing course.",
						},
					}, nil
				default:
					t.Fatalf("unexpected turn %d", turn)
					return nil, nil
				}
			},
		}

		m, err := RunArm(context.Background(), planner, "test guidance", false, 10, nil)
		if err != nil {
			t.Fatalf("RunArm failed: %v", err)
		}
		if !sawGuidanceAtTurn3 {
			t.Fatal("expected [CIRCUIT BREAKER GUIDANCE] in messages on turn 3, but did not find it")
		}
		if m.CircuitBreakerTripped {
			t.Fatal("expected circuit breaker NOT to trip when model stopped after guidance")
		}
		if m.FinalAnswer != "Understood guidance; changing course." {
			t.Fatalf("unexpected final answer %q", m.FinalAnswer)
		}
	})

	t.Run("TripsAtThreeConsecutiveFailures", func(t *testing.T) {
		planner := &cbTestPlanner{
			onComplete: func(turn int, messages []Message) (*Completion, error) {
				if turn > 3 {
					t.Fatalf("turn %d executed; expected circuit breaker to halt at turn 3", turn)
				}
				return &Completion{
					Message: Message{
						Role:      RoleAssistant,
						ToolCalls: failCall("call_fail"),
					},
				}, nil
			},
		}

		m, err := RunArm(context.Background(), planner, "test tripping at 3", false, 10, nil)
		if err != nil {
			t.Fatalf("RunArm failed: %v", err)
		}
		if !m.CircuitBreakerTripped {
			t.Fatal("expected CircuitBreakerTripped to be true")
		}
		if m.Turns != 3 {
			t.Fatalf("expected loop to stop after 3 turns, got %d turns", m.Turns)
		}
		if m.HitTurnCap {
			t.Fatal("expected HitTurnCap to be false when circuit breaker trips")
		}
		if !strings.Contains(m.CircuitBreakerReason, toolGetUser) || !strings.Contains(m.CircuitBreakerReason, "3 consecutive occurrences") {
			t.Fatalf("unexpected CircuitBreakerReason: %q", m.CircuitBreakerReason)
		}
		if !strings.Contains(m.FinalAnswer, "Circuit breaker tripped") {
			t.Fatalf("expected FinalAnswer to explain circuit breaker trip, got %q", m.FinalAnswer)
		}
	})

	t.Run("CustomThresholdViaOptionTripsAtTwo", func(t *testing.T) {
		planner := &cbTestPlanner{
			onComplete: func(turn int, messages []Message) (*Completion, error) {
				if turn > 2 {
					t.Fatalf("turn %d executed; expected custom threshold 2 to halt at turn 2", turn)
				}
				return &Completion{
					Message: Message{
						Role:      RoleAssistant,
						ToolCalls: failCall("call_fail_custom"),
					},
				}, nil
			},
		}

		m, err := RunArm(context.Background(), planner, "test custom threshold", false, 10, nil, WithCircuitBreakerThreshold(2))
		if err != nil {
			t.Fatalf("RunArm failed: %v", err)
		}
		if !m.CircuitBreakerTripped {
			t.Fatal("expected CircuitBreakerTripped to be true for custom threshold 2")
		}
		if m.Turns != 2 {
			t.Fatalf("expected loop to stop after 2 turns, got %d turns", m.Turns)
		}
		if m.HitTurnCap {
			t.Fatal("expected HitTurnCap to be false")
		}
		if !strings.Contains(m.CircuitBreakerReason, "2 consecutive occurrences") {
			t.Fatalf("unexpected CircuitBreakerReason: %q", m.CircuitBreakerReason)
		}
	})

	t.Run("InterveningSuccessResetsCounter", func(t *testing.T) {
		planner := &cbTestPlanner{
			onComplete: func(turn int, messages []Message) (*Completion, error) {
				switch turn {
				case 1:
					// Failure 1
					return &Completion{
						Message: Message{
							Role:      RoleAssistant,
							ToolCalls: failCall("call_fail_1"),
						},
					}, nil
				case 2:
					// Intervening success
					return &Completion{
						Message: Message{
							Role:      RoleAssistant,
							ToolCalls: successCall("call_ok"),
						},
					}, nil
				case 3:
					// Failure 1 (after reset)
					return &Completion{
						Message: Message{
							Role:      RoleAssistant,
							ToolCalls: failCall("call_fail_2"),
						},
					}, nil
				case 4:
					// Failure 2 (after reset)
					return &Completion{
						Message: Message{
							Role:      RoleAssistant,
							ToolCalls: failCall("call_fail_3"),
						},
					}, nil
				case 5:
					// Success / final answer
					return &Completion{
						Message: Message{
							Role:    RoleAssistant,
							Content: "Finished without tripping circuit breaker.",
						},
					}, nil
				default:
					t.Fatalf("unexpected turn %d", turn)
					return nil, nil
				}
			},
		}

		m, err := RunArm(context.Background(), planner, "test reset", false, 10, nil)
		if err != nil {
			t.Fatalf("RunArm failed: %v", err)
		}
		if m.CircuitBreakerTripped {
			t.Fatal("expected circuit breaker NOT to trip because intervening success reset the counter")
		}
		if m.Turns != 5 {
			t.Fatalf("expected 5 turns, got %d", m.Turns)
		}
		if m.FinalAnswer != "Finished without tripping circuit breaker." {
			t.Fatalf("unexpected final answer %q", m.FinalAnswer)
		}
	})

	t.Run("TripsAtThreeConsecutiveDenials", func(t *testing.T) {
		planner := &cbTestPlanner{
			onComplete: func(turn int, messages []Message) (*Completion, error) {
				if turn > 3 {
					t.Fatalf("turn %d executed; expected circuit breaker to halt at turn 3", turn)
				}
				return &Completion{
					Message: Message{
						Role:      RoleAssistant,
						ToolCalls: denyCall("call_deny"),
					},
				}, nil
			},
		}

		m, err := RunArm(context.Background(), planner, "test denial trip", false, 10, nil)
		if err != nil {
			t.Fatalf("RunArm failed: %v", err)
		}
		if !m.CircuitBreakerTripped {
			t.Fatal("expected CircuitBreakerTripped to be true for 3 consecutive denials")
		}
		if m.Turns != 3 {
			t.Fatalf("expected loop to stop after 3 turns, got %d turns", m.Turns)
		}
		if m.HitTurnCap {
			t.Fatal("expected HitTurnCap to be false")
		}
		if !strings.Contains(m.CircuitBreakerReason, "bad_tool_forbidden") || !strings.Contains(m.CircuitBreakerReason, "POLICY_BLOCK") {
			t.Fatalf("unexpected CircuitBreakerReason: %q", m.CircuitBreakerReason)
		}
	})
}
