package ctxmmu

import (
	"strings"
	"testing"
)

func TestPositiveCompaction(t *testing.T) {
	t.Run("BasicRejectedToolCallAndApology", func(t *testing.T) {
		turns := []TurnRecord{
			{
				Role:    "user",
				Content: "Investigate database schema and report table names",
			},
			{
				Role:         "assistant",
				Content:      "Attempting to drop production database",
				ToolCallName: "drop_database",
				ToolCallArgs: `{"target": "prod"}`,
			},
			{
				Role:         "tool",
				Content:      "[ERROR] POLICY_BLOCK: drop_database is forbidden by policy",
				ToolCallName: "drop_database",
				IsFailure:    true,
			},
			{
				Role:    "assistant",
				Content: "I apologize for the mistake. That was my error.",
			},
			{
				Role:         "assistant",
				Content:      "I will now list tables safely.",
				ToolCallName: "list_tables",
				ToolCallArgs: "{}",
			},
			{
				Role:         "tool",
				Content:      "users, orders, products",
				ToolCallName: "list_tables",
				VerifiedFact: "tables: users, orders, products",
				IsFailure:    false,
			},
			{
				Role:       "assistant",
				Content:    "Found 3 tables in the database.",
				Affordance: "query_table",
			},
		}

		res := CompactPositiveState(turns, "Investigate database schema")

		if res.OriginalGoal != "Investigate database schema" {
			t.Fatalf("unexpected original goal: %q", res.OriginalGoal)
		}
		if res.ShedTurnsCount != 3 {
			t.Fatalf("expected 3 shed turns, got %d", res.ShedTurnsCount)
		}
		if res.TotalTokensSavedEstimate <= 0 {
			t.Fatalf("expected positive tokens saved estimate, got %d", res.TotalTokensSavedEstimate)
		}
		if len(res.RetainedTurns) != 4 {
			t.Fatalf("expected 4 retained turns, got %d", len(res.RetainedTurns))
		}

		// Retained: user turn, list_tables call, list_tables result, assistant finish
		if res.RetainedTurns[0].Role != "user" {
			t.Errorf("expected retained[0] to be user, got %s", res.RetainedTurns[0].Role)
		}
		if res.RetainedTurns[1].ToolCallName != "list_tables" {
			t.Errorf("expected retained[1] tool call to be list_tables, got %s", res.RetainedTurns[1].ToolCallName)
		}
		if res.RetainedTurns[2].VerifiedFact != "tables: users, orders, products" {
			t.Errorf("expected retained[2] fact, got %s", res.RetainedTurns[2].VerifiedFact)
		}
		if res.RetainedTurns[3].Affordance != "query_table" {
			t.Errorf("expected retained[3] affordance, got %s", res.RetainedTurns[3].Affordance)
		}

		if len(res.VerifiedFacts) != 1 || res.VerifiedFacts[0] != "tables: users, orders, products" {
			t.Fatalf("unexpected verified facts: %#v", res.VerifiedFacts)
		}
		if res.LatestAffordance != "query_table" {
			t.Fatalf("expected latest affordance query_table, got %q", res.LatestAffordance)
		}
	})

	t.Run("ApologyStrippedFromRetainedTurn", func(t *testing.T) {
		turns := []TurnRecord{
			{
				Role:    "user",
				Content: "Fix typo in README",
			},
			{
				Role:       "assistant",
				Content:    "I apologize for the delay. Here is the typo fix in README.",
				Affordance: "git commit",
			},
		}

		res := CompactPositiveState(turns, "Fix typo in README")

		if res.ShedTurnsCount != 0 {
			t.Fatalf("expected 0 shed turns, got %d", res.ShedTurnsCount)
		}
		if res.TotalTokensSavedEstimate <= 0 {
			t.Fatalf("expected positive token savings from stripped apology, got %d", res.TotalTokensSavedEstimate)
		}
		if len(res.RetainedTurns) != 2 {
			t.Fatalf("expected 2 retained turns, got %d", len(res.RetainedTurns))
		}
		if strings.Contains(res.RetainedTurns[1].Content, "apologize") {
			t.Fatalf("retained content still contains apology: %q", res.RetainedTurns[1].Content)
		}
		if res.RetainedTurns[1].Content != "Here is the typo fix in README." {
			t.Fatalf("expected cleaned content, got %q", res.RetainedTurns[1].Content)
		}
		if res.LatestAffordance != "git commit" {
			t.Fatalf("expected affordance git commit, got %q", res.LatestAffordance)
		}
	})

	t.Run("CommaSeparatedApologyClause", func(t *testing.T) {
		turns := []TurnRecord{
			{
				Role:    "user",
				Content: "Update dependencies",
			},
			{
				Role:         "assistant",
				Content:      "Sorry for the confusion, I will update package.json now.",
				ToolCallName: "edit_file",
				ToolCallArgs: `{"path": "package.json"}`,
			},
			{
				Role:         "tool",
				Content:      "Updated package.json successfully",
				VerifiedFact: "deps updated",
			},
		}

		res := CompactPositiveState(turns, "Update dependencies")

		if res.ShedTurnsCount != 0 {
			t.Fatalf("expected 0 shed turns, got %d", res.ShedTurnsCount)
		}
		if res.TotalTokensSavedEstimate <= 0 {
			t.Fatalf("expected tokens saved > 0, got %d", res.TotalTokensSavedEstimate)
		}
		if res.RetainedTurns[1].Content != "I will update package.json now." {
			t.Fatalf("expected cleaned content, got %q", res.RetainedTurns[1].Content)
		}
		if len(res.VerifiedFacts) != 1 || res.VerifiedFacts[0] != "deps updated" {
			t.Fatalf("unexpected verified facts: %#v", res.VerifiedFacts)
		}
	})

	t.Run("ErrorBannersStrippingAndShedding", func(t *testing.T) {
		turns := []TurnRecord{
			{
				Role:    "user",
				Content: "Run migration",
			},
			{
				Role:         "assistant",
				Content:      "Running raw migration",
				ToolCallName: "migrate_raw",
			},
			{
				Role:    "tool",
				Content: "=== ERROR BANNER ===\nExecution failed with exit code 1\n=== END ===",
			},
			{
				Role:    "assistant",
				Content: "My apologies for that failure.",
			},
			{
				Role:         "assistant",
				Content:      "Running safe migration now",
				ToolCallName: "migrate_safe",
			},
			{
				Role:         "tool",
				Content:      "Migration complete: version 42",
				VerifiedFact: "migration version 42",
			},
		}

		res := CompactPositiveState(turns, "Run migration")

		if res.ShedTurnsCount != 3 {
			t.Fatalf("expected 3 shed turns (call, error banner, apology), got %d", res.ShedTurnsCount)
		}
		if len(res.RetainedTurns) != 3 {
			t.Fatalf("expected 3 retained turns, got %d", len(res.RetainedTurns))
		}
		if res.RetainedTurns[1].ToolCallName != "migrate_safe" {
			t.Fatalf("expected retained tool call migrate_safe, got %q", res.RetainedTurns[1].ToolCallName)
		}
		if len(res.VerifiedFacts) != 1 || res.VerifiedFacts[0] != "migration version 42" {
			t.Fatalf("unexpected verified facts: %#v", res.VerifiedFacts)
		}
	})

	t.Run("ConsecutiveRejectedAttempts", func(t *testing.T) {
		turns := []TurnRecord{
			{Role: "user", Content: "Build release"},
			{Role: "assistant", ToolCallName: "toolA"},
			{Role: "tool", Content: "Error: toolA missing", IsFailure: true},
			{Role: "assistant", ToolCallName: "toolB"},
			{Role: "tool", Content: "Error: toolB failed", IsFailure: true},
			{Role: "assistant", Content: "I apologize. I'll use toolC now.", ToolCallName: "toolC"},
			{Role: "tool", Content: "Build succeeded", VerifiedFact: "release built"},
			{Role: "assistant", Content: "Release ready.", Affordance: "ship"},
		}

		res := CompactPositiveState(turns, "Build release")

		if res.ShedTurnsCount != 4 {
			t.Fatalf("expected 4 shed turns for attempts A and B, got %d", res.ShedTurnsCount)
		}
		if len(res.RetainedTurns) != 4 {
			t.Fatalf("expected 4 retained turns, got %d", len(res.RetainedTurns))
		}
		if res.RetainedTurns[1].ToolCallName != "toolC" {
			t.Fatalf("expected retained toolCall toolC, got %q", res.RetainedTurns[1].ToolCallName)
		}
		if strings.Contains(res.RetainedTurns[1].Content, "apologize") {
			t.Fatalf("expected apology stripped from toolC call turn, got %q", res.RetainedTurns[1].Content)
		}
		if res.LatestAffordance != "ship" {
			t.Fatalf("expected latest affordance 'ship', got %q", res.LatestAffordance)
		}
	})

	t.Run("NoOpCleanHistory", func(t *testing.T) {
		turns := []TurnRecord{
			{Role: "user", Content: "Check status"},
			{Role: "assistant", Content: "Checking status", ToolCallName: "status_check"},
			{Role: "tool", Content: "All systems healthy", VerifiedFact: "status healthy"},
			{Role: "assistant", Content: "Done.", Affordance: "next"},
		}

		res := CompactPositiveState(turns, "Check status")

		if res.ShedTurnsCount != 0 {
			t.Fatalf("expected 0 shed turns, got %d", res.ShedTurnsCount)
		}
		if res.TotalTokensSavedEstimate != 0 {
			t.Fatalf("expected 0 tokens saved on clean history, got %d", res.TotalTokensSavedEstimate)
		}
		if len(res.RetainedTurns) != len(turns) {
			t.Fatalf("expected %d retained turns, got %d", len(turns), len(res.RetainedTurns))
		}
		if res.LatestAffordance != "next" {
			t.Fatalf("expected affordance 'next', got %q", res.LatestAffordance)
		}
	})

	t.Run("EmptyAndNilTurns", func(t *testing.T) {
		res1 := CompactPositiveState(nil, "Goal 1")
		if res1.OriginalGoal != "Goal 1" || res1.RetainedTurns == nil || len(res1.RetainedTurns) != 0 || res1.ShedTurnsCount != 0 {
			t.Fatalf("unexpected result on nil turns: %#v", res1)
		}

		res2 := CompactPositiveState([]TurnRecord{}, "")
		if res2.OriginalGoal != "" || res2.RetainedTurns == nil || len(res2.RetainedTurns) != 0 {
			t.Fatalf("unexpected result on empty turns: %#v", res2)
		}
	})

	t.Run("GoalInferenceFromUserTurn", func(t *testing.T) {
		turns := []TurnRecord{
			{Role: "user", Content: "Deploy staging environment"},
			{Role: "assistant", Content: "Deploying now"},
		}

		res := CompactPositiveState(turns, "")
		if res.OriginalGoal != "Deploy staging environment" {
			t.Fatalf("expected inferred goal 'Deploy staging environment', got %q", res.OriginalGoal)
		}
	})

	t.Run("MultipleVerifiedFactsDeduplication", func(t *testing.T) {
		turns := []TurnRecord{
			{Role: "user", Content: "Audit cluster"},
			{Role: "tool", Content: "fact 1", VerifiedFact: "node count is 3"},
			{Role: "tool", Content: "fact 2", VerifiedFact: "node count is 3"},
			{Role: "tool", Content: "fact 3", VerifiedFact: "k8s version 1.30"},
		}

		res := CompactPositiveState(turns, "Audit cluster")
		if len(res.VerifiedFacts) != 2 {
			t.Fatalf("expected 2 deduplicated verified facts, got %d: %#v", len(res.VerifiedFacts), res.VerifiedFacts)
		}
		if res.VerifiedFacts[0] != "node count is 3" || res.VerifiedFacts[1] != "k8s version 1.30" {
			t.Fatalf("unexpected facts order/content: %#v", res.VerifiedFacts)
		}
	})

	t.Run("PreserveUserTurnVerbatim", func(t *testing.T) {
		userPrompt := "Please write a test that verifies error handling and doesn't apologize unnecessarily."
		turns := []TurnRecord{
			{Role: "user", Content: userPrompt},
			{Role: "assistant", Content: "I will write the test.", Affordance: "run_test"},
		}

		res := CompactPositiveState(turns, "")
		if len(res.RetainedTurns) != 2 {
			t.Fatalf("expected 2 retained turns, got %d", len(res.RetainedTurns))
		}
		if res.RetainedTurns[0].Content != userPrompt {
			t.Fatalf("expected user prompt preserved verbatim, got %q", res.RetainedTurns[0].Content)
		}
	})
}
