package agentopt

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

func TestPlanAndSolveExecution(t *testing.T) {
	t.Run("unverified step blocks subsequent step execution", func(t *testing.T) {
		sm := NewDecomposedSolveMachine()

		step1Executed := false
		step2Executed := false
		step3Executed := false

		steps := []StepSpec{
			{
				StepIndex: 1,
				Title:     "Step 1 - initial setup",
				ExecuteFn: func(ctx context.Context, senv *StepEnv) (any, error) {
					step1Executed = true
					senv.Set("step1_done", true)
					return "data_1", nil
				},
				VerifyFn: func(ctx context.Context, senv *StepEnv, output any) (bool, error) {
					val, ok := output.(string)
					return ok && val == "data_1", nil
				},
			},
			{
				StepIndex: 2,
				Title:     "Step 2 - unverified execution",
				ExecuteFn: func(ctx context.Context, senv *StepEnv) (any, error) {
					step2Executed = true
					return "data_2", nil
				},
				VerifyFn: func(ctx context.Context, senv *StepEnv, output any) (bool, error) {
					// Explicit verification failure: returns false
					return false, nil
				},
			},
			{
				StepIndex: 3,
				Title:     "Step 3 - subsequent action",
				ExecuteFn: func(ctx context.Context, senv *StepEnv) (any, error) {
					step3Executed = true
					return "data_3", nil
				},
				VerifyFn: func(ctx context.Context, senv *StepEnv, output any) (bool, error) {
					return true, nil
				},
			},
		}

		report := sm.ExecuteSteps(context.Background(), steps)

		if !step1Executed {
			t.Fatalf("expected step 1 to execute")
		}
		if !step2Executed {
			t.Fatalf("expected step 2 to execute")
		}
		if step3Executed {
			t.Fatalf("expected step 3 NOT to execute because step 2 verification failed")
		}

		if report.Success {
			t.Fatalf("expected report.Success to be false, got true")
		}
		if !report.Blocked {
			t.Fatalf("expected report.Blocked to be true, got false")
		}
		if report.CompletedSteps != 1 {
			t.Fatalf("expected 1 completed step, got %d", report.CompletedSteps)
		}
		if report.TotalSteps != 3 {
			t.Fatalf("expected 3 total steps, got %d", report.TotalSteps)
		}
		if len(report.Results) != 3 {
			t.Fatalf("expected 3 results, got %d", len(report.Results))
		}

		if !report.Results[0].Verified || report.Results[0].Status != StepStatusSuccess {
			t.Fatalf("step 1 should be verified success, got %+v", report.Results[0])
		}
		if report.Results[1].Verified || report.Results[1].Status != StepStatusFailed {
			t.Fatalf("step 2 should be failed and unverified, got %+v", report.Results[1])
		}
		if report.Results[2].Status != StepStatusBlocked {
			t.Fatalf("step 3 should be marked blocked, got status %s", report.Results[2].Status)
		}

		if sm.State() != WorkflowStateBlocked {
			t.Fatalf("expected state machine state %s, got %s", WorkflowStateBlocked, sm.State())
		}
	})

	t.Run("verification error blocks subsequent steps", func(t *testing.T) {
		sm := NewDecomposedSolveMachine()

		step2Ran := false

		steps := []StepSpec{
			{
				StepIndex: 1,
				Title:     "Step 1 - verify error",
				ExecuteFn: func(ctx context.Context, senv *StepEnv) (any, error) {
					return 100, nil
				},
				VerifyFn: func(ctx context.Context, senv *StepEnv, output any) (bool, error) {
					return false, errors.New("deterministic verification checksum mismatch")
				},
			},
			{
				StepIndex: 2,
				Title:     "Step 2 - blocked",
				ExecuteFn: func(ctx context.Context, senv *StepEnv) (any, error) {
					step2Ran = true
					return 200, nil
				},
			},
		}

		report := sm.ExecuteSteps(context.Background(), steps)

		if step2Ran {
			t.Fatalf("step 2 should not run after step 1 verification error")
		}
		if report.Success {
			t.Fatalf("expected report.Success false")
		}
		if !report.Blocked {
			t.Fatalf("expected report.Blocked true")
		}
		if report.Results[1].Status != StepStatusBlocked {
			t.Fatalf("expected step 2 status blocked, got %s", report.Results[1].Status)
		}
	})

	t.Run("precondition failure blocks step execution and subsequent steps", func(t *testing.T) {
		sm := NewDecomposedSolveMachine()

		step1Ran := false
		step2Ran := false

		steps := []StepSpec{
			{
				StepIndex: 1,
				Title:     "Step 1 - failed precondition",
				Precondition: StepCondition{
					Name: "missing_context",
					Assert: func(ctx context.Context, senv *StepEnv) error {
						return fmt.Errorf("context key missing")
					},
				},
				ExecuteFn: func(ctx context.Context, senv *StepEnv) (any, error) {
					step1Ran = true
					return nil, nil
				},
			},
			{
				StepIndex: 2,
				Title:     "Step 2 - blocked by precondition",
				ExecuteFn: func(ctx context.Context, senv *StepEnv) (any, error) {
					step2Ran = true
					return nil, nil
				},
			},
		}

		report := sm.ExecuteSteps(context.Background(), steps)

		if step1Ran {
			t.Fatalf("step 1 ExecuteFn must not be called when precondition fails")
		}
		if step2Ran {
			t.Fatalf("step 2 must not be called")
		}
		if !report.Blocked {
			t.Fatalf("expected report.Blocked true")
		}
	})

	t.Run("postcondition failure blocks subsequent steps", func(t *testing.T) {
		sm := NewDecomposedSolveMachine()

		step2Ran := false

		steps := []StepSpec{
			{
				StepIndex: 1,
				Title:     "Step 1 - postcondition check",
				ExecuteFn: func(ctx context.Context, senv *StepEnv) (any, error) {
					return "ok", nil
				},
				VerifyFn: func(ctx context.Context, senv *StepEnv, output any) (bool, error) {
					return true, nil
				},
				Postcondition: StepCondition{
					Name: "state_validation",
					Assert: func(ctx context.Context, senv *StepEnv) error {
						return fmt.Errorf("postcondition unmet")
					},
				},
			},
			{
				StepIndex: 2,
				Title:     "Step 2 - blocked",
				ExecuteFn: func(ctx context.Context, senv *StepEnv) (any, error) {
					step2Ran = true
					return nil, nil
				},
			},
		}

		report := sm.ExecuteSteps(context.Background(), steps)

		if step2Ran {
			t.Fatalf("step 2 should not execute when step 1 postcondition fails")
		}
		if !report.Blocked {
			t.Fatalf("expected report.Blocked true")
		}
	})

	t.Run("all steps succeed and verify", func(t *testing.T) {
		runner := NewDecomposedSolveRunner()

		steps := []StepSpec{
			{
				StepIndex: 1,
				Title:     "Compute sum",
				ExecuteFn: func(ctx context.Context, senv *StepEnv) (any, error) {
					senv.Set("total", 42)
					return 42, nil
				},
				VerifyFn: func(ctx context.Context, senv *StepEnv, output any) (bool, error) {
					val, ok := output.(int)
					return ok && val == 42, nil
				},
				Postcondition: RequireStateKey("total"),
			},
			{
				StepIndex:    2,
				Title:        "Transform sum",
				Precondition: RequireStateKey("total"),
				ExecuteFn: func(ctx context.Context, senv *StepEnv) (any, error) {
					tot, _ := senv.Get("total")
					result := tot.(int) * 2
					senv.Set("doubled", result)
					return result, nil
				},
				VerifyFn: func(ctx context.Context, senv *StepEnv, output any) (bool, error) {
					val, ok := output.(int)
					return ok && val == 84, nil
				},
				Postcondition: RequireStateKey("doubled"),
			},
		}

		report := runner.ExecuteSteps(context.Background(), steps)

		if !report.Success {
			t.Fatalf("expected report.Success true, got false (reason: %s)", report.BlockedReason)
		}
		if report.Blocked {
			t.Fatalf("expected report.Blocked false")
		}
		if report.CompletedSteps != 2 {
			t.Fatalf("expected 2 completed steps, got %d", report.CompletedSteps)
		}
		if runner.StateMachine.State() != WorkflowStateCompleted {
			t.Fatalf("expected state %s, got %s", WorkflowStateCompleted, runner.StateMachine.State())
		}
	})

	t.Run("objective decomposition and execution", func(t *testing.T) {
		sm := NewDecomposedSolveMachine()

		objective := "1. Prepare workspace\n2. Run transformation\n3. Finalize output"
		steps, err := sm.Decompose(context.Background(), objective)
		if err != nil {
			t.Fatalf("unexpected decompose error: %v", err)
		}
		if len(steps) != 3 {
			t.Fatalf("expected 3 decomposed steps, got %d", len(steps))
		}
		if steps[0].Title != "Prepare workspace" {
			t.Fatalf("expected title 'Prepare workspace', got %q", steps[0].Title)
		}

		for i := range steps {
			idx := i + 1
			steps[i].ExecuteFn = func(ctx context.Context, senv *StepEnv) (any, error) {
				return idx * 10, nil
			}
			steps[i].VerifyFn = func(ctx context.Context, senv *StepEnv, output any) (bool, error) {
				val, ok := output.(int)
				return ok && val == idx*10, nil
			}
		}

		report := sm.ExecuteSteps(context.Background(), steps)
		if !report.Success {
			t.Fatalf("expected success on decomposed steps")
		}
		if report.CompletedSteps != 3 {
			t.Fatalf("expected 3 completed steps, got %d", report.CompletedSteps)
		}
	})
}
