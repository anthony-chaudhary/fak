package agent

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestGoalAnchor(t *testing.T) {
	t.Run("NewGoalAnchor", func(t *testing.T) {
		objective := "Implement payment webhook signature validation"
		before := time.Now().UTC()
		anchor := NewGoalAnchor(objective)
		after := time.Now().UTC()

		if anchor == nil {
			t.Fatal("expected non-nil GoalAnchor")
		}
		if anchor.Objective != objective {
			t.Errorf("expected Objective %q, got %q", objective, anchor.Objective)
		}
		if anchor.PinnedAt.Before(before) || anchor.PinnedAt.After(after) {
			t.Errorf("PinnedAt %v not within expected range [%v, %v]", anchor.PinnedAt, before, after)
		}
		if anchor.RecoveryTurnCount != 0 {
			t.Errorf("expected RecoveryTurnCount 0, got %d", anchor.RecoveryTurnCount)
		}
	})

	t.Run("FormatRecoveryReinforcement", func(t *testing.T) {
		objective := "Implement payment webhook signature validation"
		anchor := NewGoalAnchor(objective)
		guidance := "Signature verification failed with 401 Unauthorized; inspect crypto HMAC keys"

		got := anchor.FormatRecoveryReinforcement(guidance)
		expected := fmt.Sprintf(
			"[PRIMARY GOAL ANCHOR]: %s\n[RECOVERY GUIDANCE]: %s\n[REMINDER]: Maintain focus on the primary goal above while resolving this error.",
			objective,
			guidance,
		)

		if got != expected {
			t.Errorf("FormatRecoveryReinforcement mismatch.\nwant:\n%s\ngot:\n%s", expected, got)
		}

		// Verify on nil receiver
		var nilAnchor *GoalAnchor
		nilGot := nilAnchor.FormatRecoveryReinforcement(guidance)
		if !strings.Contains(nilGot, guidance) {
			t.Errorf("expected nil anchor format to contain guidance, got %q", nilGot)
		}
	})

	t.Run("RecordRecoveryTurn", func(t *testing.T) {
		anchor := NewGoalAnchor("Fix login loop")
		if anchor.RecoveryTurnCount != 0 {
			t.Fatalf("expected 0, got %d", anchor.RecoveryTurnCount)
		}

		anchor.RecordRecoveryTurn()
		if anchor.RecoveryTurnCount != 1 {
			t.Errorf("expected 1, got %d", anchor.RecoveryTurnCount)
		}

		anchor.RecordRecoveryTurn()
		anchor.RecordRecoveryTurn()
		if anchor.RecoveryTurnCount != 3 {
			t.Errorf("expected 3, got %d", anchor.RecoveryTurnCount)
		}

		// Nil receiver should not panic
		var nilAnchor *GoalAnchor
		nilAnchor.RecordRecoveryTurn()
	})

	t.Run("ValidateTextContainsAnchor", func(t *testing.T) {
		objective := "Refactor token bucket rate limiter"
		anchor := NewGoalAnchor(objective)

		// Exact match
		if !anchor.ValidateTextContainsAnchor(objective) {
			t.Error("expected true for exact objective in context")
		}

		// Context with reinforcement block
		reinforcement := anchor.FormatRecoveryReinforcement("Rate limit exceeded; retry after 5s")
		contextWithReinforcement := fmt.Sprintf("System message:\n%s\nTool output: error", reinforcement)
		if !anchor.ValidateTextContainsAnchor(contextWithReinforcement) {
			t.Error("expected true when reinforcement block is present in context")
		}

		// Unrelated context without anchor (drifted)
		driftedContext := "System message: Fix syntax error on line 42 in utils.go\nTool output: nil pointer"
		if anchor.ValidateTextContainsAnchor(driftedContext) {
			t.Error("expected false for drifted context missing anchor")
		}

		// Empty context
		if anchor.ValidateTextContainsAnchor("") {
			t.Error("expected false for empty context")
		}

		// Anchor with empty objective
		emptyAnchor := NewGoalAnchor("")
		if emptyAnchor.ValidateTextContainsAnchor("Some random context text") {
			t.Error("expected false when anchor objective is empty")
		}

		// Nil anchor
		var nilAnchor *GoalAnchor
		if nilAnchor.ValidateTextContainsAnchor(objective) {
			t.Error("expected false for nil anchor")
		}
	})

	t.Run("MultiTurnRecoveryDriftPrevention", func(t *testing.T) {
		primaryGoal := "Migrate schema from v1 to v2 with zero downtime"
		anchor := NewGoalAnchor(primaryGoal)

		turns := []struct {
			errorGuidance string
			simulatedBody string
		}{
			{
				errorGuidance: "Table migration failed: column already exists",
				simulatedBody: "Inspecting migration scripts in db/migrations",
			},
			{
				errorGuidance: "Rollback transaction timed out",
				simulatedBody: "Increasing transaction timeout in db pool",
			},
			{
				errorGuidance: "Lock wait timeout exceeded",
				simulatedBody: "Acquiring advisory lock with retry backoff",
			},
		}

		var fullWorkingContext string
		for i, turn := range turns {
			anchor.RecordRecoveryTurn()
			if anchor.RecoveryTurnCount != i+1 {
				t.Fatalf("turn %d: expected turn count %d, got %d", i+1, i+1, anchor.RecoveryTurnCount)
			}

			reinforcement := anchor.FormatRecoveryReinforcement(turn.errorGuidance)
			fullWorkingContext += fmt.Sprintf("\n--- Turn %d ---\n%s\nAgent action: %s\n", i+1, reinforcement, turn.simulatedBody)

			if !anchor.ValidateTextContainsAnchor(fullWorkingContext) {
				t.Fatalf("turn %d: context lost primary goal anchor!", i+1)
			}
		}

		// Now simulate context compaction or truncation that dropped the anchor
		compactedWithoutAnchor := "Agent action: Acquiring advisory lock with retry backoff\nTool: success"
		if anchor.ValidateTextContainsAnchor(compactedWithoutAnchor) {
			t.Error("expected validation failure when anchor was purged from context")
		}
	})
}
