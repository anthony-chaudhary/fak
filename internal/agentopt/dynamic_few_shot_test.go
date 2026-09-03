package agentopt

import (
	"context"
	"strings"
	"testing"
)

func TestDynamicFewShotSelector(t *testing.T) {
	pool := []FewShotExemplar{
		{
			ID:      "payment-canary",
			Prompt:  "Deploy payment service with canary verification.",
			Thought: "Check schema version then route 5% canary traffic to verify payment service.",
			ToolCalls: []ToolCall{
				{Name: "verify_schema", Args: map[string]any{"service": "payment"}},
				{Name: "route_traffic", Args: map[string]any{"service": "payment", "weight": 0.05}},
			},
			ToolsUsed: []string{"verify_schema", "route_traffic"},
			Output:    "Canary deployment initiated for payment service.",
		},
		{
			ID:      "db-failover",
			Prompt:  "Failover database replica under network partition.",
			Thought: "Check consensus peers and replication lag before promoting replica-b.",
			ToolCalls: []ToolCall{
				{Name: "check_consensus", Args: map[string]any{"cluster": "db-prod"}},
				{Name: "promote_replica", Args: map[string]any{"target": "replica-b"}},
			},
			ToolsUsed: []string{"check_consensus", "promote_replica"},
			Output:    "Replica-b promoted to primary.",
		},
		{
			ID:      "log-triage",
			Prompt:  "Diagnose 500 error spikes in cluster.",
			Thought: "Query logs for HTTP 500 status in last 15 minutes and isolate failing node.",
			ToolCalls: []ToolCall{
				{Name: "query_logs", Args: map[string]any{"status": 500}},
				{Name: "isolate_node", Args: map[string]any{"node": "api-03"}},
			},
			ToolsUsed: []string{"query_logs", "isolate_node"},
			Output:    "Node api-03 isolated from traffic.",
		},
		{
			ID:      "k8s-pod-restart",
			Prompt:  "Restart crashlooping pod in payment namespace.",
			Thought: "Inspect pod events and trigger rolling restart for payment worker.",
			ToolCalls: []ToolCall{
				{Name: "get_events", Args: map[string]any{"namespace": "payment"}},
				{Name: "restart_deployment", Args: map[string]any{"name": "payment-worker"}},
			},
			ToolsUsed: []string{"get_events", "restart_deployment"},
			Output:    "Rolling restart scheduled for payment worker.",
		},
	}

	t.Run("selects_relevant_by_semantic_and_tool_affinity", func(t *testing.T) {
		selector := NewDynamicFewShotSelector(DefaultSelectorConfig())
		for _, ex := range pool {
			selector.AddExemplar(ex)
		}

		req := SelectionRequest{
			Query:            "Deploy updated payment gateway and check schema",
			InputTokenBudget: 4000,
			PredictedTools:   []string{"route_traffic", "verify_schema"},
			HistoricalToolAffinity: map[string]float64{
				"verify_schema": 0.9,
				"route_traffic": 0.8,
				"query_logs":    0.1,
			},
		}

		res, err := selector.Select(context.Background(), req)
		if err != nil {
			t.Fatalf("Select returned unexpected error: %v", err)
		}

		if len(res.Selected) == 0 {
			t.Fatal("expected at least one selected exemplar, got 0")
		}
		if len(res.Selected) > 3 {
			t.Fatalf("expected at most 3 exemplars, got %d", len(res.Selected))
		}

		top := res.Selected[0]
		if top.ID != "payment-canary" {
			t.Errorf("expected top exemplar to be 'payment-canary', got %q", top.ID)
		}

		// Ensure tool affinity score is positive for top match
		if res.Scores[0].ToolAffinityScore <= 0 {
			t.Errorf("expected positive tool affinity score for top match, got %f", res.Scores[0].ToolAffinityScore)
		}
		if res.Scores[0].SemanticScore <= 0 {
			t.Errorf("expected positive semantic score for top match, got %f", res.Scores[0].SemanticScore)
		}
	})

	t.Run("strict_token_budget_bound_under_10_percent", func(t *testing.T) {
		selector := NewDynamicFewShotSelector(DefaultSelectorConfig())
		for _, ex := range pool {
			selector.AddExemplar(ex)
		}

		// Given an input budget of 500 tokens, 10% is 50 tokens
		inputBudget := 500
		req := SelectionRequest{
			Query:            "Database failover and replica health checks",
			InputTokenBudget: inputBudget,
			PredictedTools:   []string{"check_consensus"},
		}

		res, err := selector.Select(context.Background(), req)
		if err != nil {
			t.Fatalf("Select error: %v", err)
		}

		maxAllowed := int(float64(inputBudget) * 0.10)
		if res.MaxAllowedTokens != maxAllowed {
			t.Errorf("expected MaxAllowedTokens=%d, got %d", maxAllowed, res.MaxAllowedTokens)
		}
		if res.TotalTokens > maxAllowed {
			t.Errorf("demonstration tokens (%d) exceeded 10%% input budget ceiling (%d)", res.TotalTokens, maxAllowed)
		}
		if res.TokenBudgetRatio > 0.1001 {
			t.Errorf("token budget ratio %f exceeds 10%%", res.TokenBudgetRatio)
		}
	})

	t.Run("bounds_selection_to_at_most_3_exemplars", func(t *testing.T) {
		selector := NewDynamicFewShotSelector(DefaultSelectorConfig())
		// Add 10 micro exemplars with very small token counts
		for i := 0; i < 10; i++ {
			selector.AddExemplar(FewShotExemplar{
				ID:        string(rune('a' + i)),
				Prompt:    "Task step",
				Thought:   "Short thought",
				ToolsUsed: []string{"tool"},
				Output:    "Done",
			})
		}

		req := SelectionRequest{
			Query:            "Task step with tool",
			InputTokenBudget: 10000, // Large budget to allow many exemplars
			PredictedTools:   []string{"tool"},
		}

		res, err := selector.Select(context.Background(), req)
		if err != nil {
			t.Fatalf("Select error: %v", err)
		}

		if len(res.Selected) > 3 {
			t.Fatalf("expected at most 3 exemplars, got %d", len(res.Selected))
		}
		if len(res.Selected) < 1 {
			t.Fatalf("expected at least 1 exemplar, got %d", len(res.Selected))
		}
	})

	t.Run("handles_empty_pool_and_zero_budget", func(t *testing.T) {
		selector := NewDynamicFewShotSelector(DefaultSelectorConfig())

		res, err := selector.Select(context.Background(), SelectionRequest{
			Query:            "Anything",
			InputTokenBudget: 1000,
		})
		if err != nil {
			t.Fatalf("unexpected error on empty pool: %v", err)
		}
		if len(res.Selected) != 0 || res.TotalTokens != 0 {
			t.Errorf("expected 0 selected on empty pool, got %d", len(res.Selected))
		}

		// Zero token budget
		selector.AddExemplar(pool[0])
		resZero, err := selector.Select(context.Background(), SelectionRequest{
			Query:            "Payment",
			InputTokenBudget: 0,
		})
		if err != nil {
			t.Fatalf("unexpected error on zero budget: %v", err)
		}
		if len(resZero.Selected) != 0 || resZero.TotalTokens != 0 {
			t.Errorf("expected 0 selected on zero budget, got %d", len(resZero.Selected))
		}
	})

	t.Run("formats_demonstrations_correctly", func(t *testing.T) {
		selector := NewDynamicFewShotSelector(DefaultSelectorConfig())
		selector.AddExemplar(pool[0])

		res, err := selector.Select(context.Background(), SelectionRequest{
			Query:            "Deploy payment service",
			InputTokenBudget: 2000,
		})
		if err != nil {
			t.Fatalf("Select error: %v", err)
		}
		if len(res.Selected) == 0 {
			t.Fatal("expected selected exemplar")
		}

		formatted := res.FormattedDemonstrations
		if !strings.Contains(formatted, "Example 1:") ||
			!strings.Contains(formatted, "User: Deploy payment service") ||
			!strings.Contains(formatted, "Response:") {
			t.Errorf("formatted demonstrations missing standard delimiters, got:\n%s", formatted)
		}
	})
}
