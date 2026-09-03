package agentopt

import (
	"context"
	"strings"
	"testing"
)

func TestEarlyExitVerification(t *testing.T) {
	ctx := context.Background()

	t.Run("terminates early when all tests pass", func(t *testing.T) {
		controller := NewEarlyExitController(10, AllTestsPassedRule{})

		// Simulate reasoning cycles where tests pass at iteration 3
		stepFn := func(ctx context.Context, iter int) (IterationStep, error) {
			step := IterationStep{
				TokensConsumed: 100,
				TotalTests:     5,
			}
			if iter < 3 {
				step.PassedTests = iter
			} else {
				step.PassedTests = 5
				step.StepOutput = "optimal fix candidate"
			}
			return step, nil
		}

		rep, err := controller.Run(ctx, stepFn)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !rep.EarlyExited {
			t.Fatal("expected early exit, got false")
		}
		if rep.CompletedIterations != 3 {
			t.Fatalf("completed iterations = %d, want 3", rep.CompletedIterations)
		}
		if rep.SavedIterations != 7 {
			t.Fatalf("saved iterations = %d, want 7", rep.SavedIterations)
		}
		if rep.EstimatedSavedTokens != 700 {
			t.Fatalf("estimated saved tokens = %d, want 700", rep.EstimatedSavedTokens)
		}
		if !strings.Contains(rep.ExitReason, "all 5 tests passed") {
			t.Fatalf("exit reason = %q, want 'all 5 tests passed'", rep.ExitReason)
		}
	})

	t.Run("terminates on confidence threshold", func(t *testing.T) {
		controller := NewEarlyExitController(5, ConfidenceThresholdRule{MinConfidence: 0.95})

		stepFn := func(ctx context.Context, iter int) (IterationStep, error) {
			return IterationStep{
				ConfidenceScore: float64(iter) * 0.5,
				TokensConsumed:  50,
			}, nil
		}

		rep, err := controller.Run(ctx, stepFn)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !rep.EarlyExited || rep.CompletedIterations != 2 {
			t.Fatalf("expected early exit at iteration 2, got: %+v", rep)
		}
	})

	t.Run("runs to max iterations when condition unsatisfied", func(t *testing.T) {
		controller := NewEarlyExitController(4, AllTestsPassedRule{})

		stepFn := func(ctx context.Context, iter int) (IterationStep, error) {
			return IterationStep{
				PassedTests:    1,
				TotalTests:     3,
				TokensConsumed: 25,
			}, nil
		}

		rep, err := controller.Run(ctx, stepFn)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if rep.EarlyExited {
			t.Fatal("expected no early exit when condition never met")
		}
		if rep.CompletedIterations != 4 || rep.SavedIterations != 0 {
			t.Fatalf("expected 4 completed iterations, got: %+v", rep)
		}
	})
}
