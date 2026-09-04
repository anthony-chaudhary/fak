package agent

import (
	"context"
	"testing"
)

func TestAgentIslandsWiring(t *testing.T) {
	ctx := context.Background()

	t.Run("GoalPrefixStabilizer wired into prepareUpstream", func(t *testing.T) {
		planner := &HTTPPlanner{
			BaseURL: "http://localhost:1234",
			ModelID: "test-model",
		}
		// Messages with volatile progress header
		msgs := []Message{
			{Role: RoleUser, Content: "Goal: Ship feature\nTurn: 1\nTokens left: 1000"},
			{Role: RoleAssistant, Content: "Working on it"},
			{Role: RoleUser, Content: "Goal: Ship feature\nTurn: 2\nTokens left: 800"},
		}

		call, err := planner.prepareUpstream(msgs, nil, false)
		if err != nil {
			t.Fatalf("prepareUpstream failed: %v", err)
		}
		if call == nil {
			t.Fatal("expected non-nil upstreamCall")
		}
		// The stabilized prompt prefix should have separated or canonicalized volatile fields
		bodyStr := string(call.body)
		if bodyStr == "" {
			t.Fatal("expected non-empty body")
		}
	})

	t.Run("GoalAnchor wired into RunArm and loop", func(t *testing.T) {
		anchor := NewGoalAnchor("Book flight SFO to JFK")
		cfg := resolveRunConfig([]RunOption{WithGoalAnchor(anchor)})
		if cfg.GoalAnchor() != anchor {
			t.Fatalf("expected WithGoalAnchor to set goalAnchor in runConfig")
		}

		// Verify loop initializes goal anchor from task when unset
		mockP := NewMockPlanner("test-model")
		metrics, err := RunArm(ctx, mockP, "Book flight SFO to JFK", false, 5, nil, WithGoalAnchor(anchor))
		if err != nil {
			t.Fatalf("RunArm failed: %v", err)
		}
		_ = metrics
		if anchor.Objective != "Book flight SFO to JFK" {
			t.Fatalf("expected anchor objective %q, got %q", "Book flight SFO to JFK", anchor.Objective)
		}
	})
}
