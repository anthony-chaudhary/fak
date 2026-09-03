package agentopt

import (
	"bytes"
	"strings"
	"testing"
)

func TestSTaRTrajectoryFiltering(t *testing.T) {
	filter := NewSTaRTrajectoryFilter()

	t.Run("rejects self-reported success lacking external verification", func(t *testing.T) {
		unverified := ReasoningTrajectory{
			ID:                  "traj-unverified-1",
			Prompt:              "Prove Goldbach conjecture up to 100",
			ThoughtTrace:        "I will check all even numbers up to 100.",
			SelfReportedSuccess: true,
			ExternalVerification: ExternalProof{
				Verified:    false,
				Command:     "python3 verify_goldbach.py",
				ExitCode:    0,
				ProofOutput: "",
			},
		}

		if unverified.VerifiedSuccess() {
			t.Errorf("expected VerifiedSuccess() to be false for unverified trajectory")
		}

		result := filter.FilterTrajectories([]ReasoningTrajectory{unverified})
		if result.RetainedCount != 0 {
			t.Errorf("expected 0 retained trajectories, got %d", result.RetainedCount)
		}
		if result.RejectedCount != 1 {
			t.Errorf("expected 1 rejected trajectory, got %d", result.RejectedCount)
		}
		if len(result.Trajectories) != 0 {
			t.Errorf("expected empty retained slice, got %d items", len(result.Trajectories))
		}
	})

	t.Run("rejects trajectory with non-zero exit code", func(t *testing.T) {
		failedTest := ReasoningTrajectory{
			ID:                  "traj-failed-exit-code",
			Prompt:              "Write a binary search in Go",
			ThoughtTrace:        "Binary search divides range by two.",
			SelfReportedSuccess: true,
			ExternalVerification: ExternalProof{
				Verified:    true,
				Command:     "go test -v ./search",
				ExitCode:    1,
				ProofOutput: "FAIL: TestBinarySearch off-by-one",
			},
		}

		if failedTest.VerifiedSuccess() {
			t.Errorf("expected VerifiedSuccess() to be false when exit code is non-zero")
		}

		result := filter.FilterTrajectories([]ReasoningTrajectory{failedTest})
		if result.RetainedCount != 0 {
			t.Errorf("expected 0 retained trajectories, got %d", result.RetainedCount)
		}
		if result.RejectedCount != 1 {
			t.Errorf("expected 1 rejected trajectory, got %d", result.RejectedCount)
		}
	})

	t.Run("retains trajectory with verified external proof", func(t *testing.T) {
		verified := ReasoningTrajectory{
			ID:           "traj-verified-success-1",
			Prompt:       "Write a factorial function in Go",
			ThoughtTrace: "Recursive factorial with base case 0 and 1 returning 1.",
			Steps: []TrajectoryStep{
				{
					StepIndex: 1,
					Thought:   "Writing factorial implementation",
					ToolCall: ToolCall{
						Name: "WriteFile",
						Args: map[string]any{"path": "factorial.go"},
					},
					ToolResult: "ok",
				},
				{
					StepIndex: 2,
					Thought:   "Executing unit tests",
					ToolCall: ToolCall{
						Name: "RunTest",
						Args: map[string]any{"path": "factorial_test.go"},
					},
					ToolResult: "PASS",
				},
			},
			SelfReportedSuccess: true,
			ExternalVerification: ExternalProof{
				Verified:    true,
				Command:     "go test -v ./math",
				ExitCode:    0,
				ProofOutput: "PASS: TestFactorial (0.01s)",
			},
			FinalAnswer: "Implemented factorial with unit test verification.",
		}

		if !verified.VerifiedSuccess() {
			t.Errorf("expected VerifiedSuccess() to be true for verified trajectory")
		}

		result := filter.FilterTrajectories([]ReasoningTrajectory{verified})
		if result.RetainedCount != 1 {
			t.Errorf("expected 1 retained trajectory, got %d", result.RetainedCount)
		}
		if result.RejectedCount != 0 {
			t.Errorf("expected 0 rejected trajectories, got %d", result.RejectedCount)
		}
		if len(result.Trajectories) != 1 {
			t.Fatalf("expected 1 item in Trajectories, got %d", len(result.Trajectories))
		}
		if result.Trajectories[0].ID != "traj-verified-success-1" {
			t.Errorf("unexpected retained ID: %s", result.Trajectories[0].ID)
		}
	})

	t.Run("rejects trajectory when self-reported success is false", func(t *testing.T) {
		failedSelfReport := ReasoningTrajectory{
			ID:                  "traj-self-report-false",
			Prompt:              "Solve quadratic equation",
			ThoughtTrace:        "I could not find real roots.",
			SelfReportedSuccess: false,
			ExternalVerification: ExternalProof{
				Verified:    true,
				Command:     "python3 check_roots.py",
				ExitCode:    0,
				ProofOutput: "OK",
			},
		}

		if failedSelfReport.VerifiedSuccess() {
			t.Errorf("expected VerifiedSuccess() to be false when self reported success is false")
		}

		result := filter.FilterTrajectories([]ReasoningTrajectory{failedSelfReport})
		if result.RetainedCount != 0 {
			t.Errorf("expected 0 retained trajectories, got %d", result.RetainedCount)
		}
	})

	t.Run("rejects trajectory failing custom proof output validator", func(t *testing.T) {
		customFilter := STaRTrajectoryFilter{
			ProofOutputValidator: func(output string) bool {
				return strings.Contains(output, "VERIFIED_CORRECT")
			},
		}

		itemMismatch := ReasoningTrajectory{
			ID:                  "traj-oracle-mismatch",
			Prompt:              "Find shortest path in graph",
			ThoughtTrace:        "Dijkstra algorithm search.",
			SelfReportedSuccess: true,
			ExternalVerification: ExternalProof{
				Verified:    true,
				Command:     "oracle_check --graph g1",
				ExitCode:    0,
				ProofOutput: "SUBOPTIMAL_PATH_FOUND",
			},
		}

		itemMatch := ReasoningTrajectory{
			ID:                  "traj-oracle-match",
			Prompt:              "Find shortest path in graph",
			ThoughtTrace:        "Dijkstra algorithm search.",
			SelfReportedSuccess: true,
			ExternalVerification: ExternalProof{
				Verified:    true,
				Command:     "oracle_check --graph g1",
				ExitCode:    0,
				ProofOutput: "VERIFIED_CORRECT distance=42",
			},
		}

		result := customFilter.FilterTrajectories([]ReasoningTrajectory{itemMismatch, itemMatch})
		if result.RetainedCount != 1 {
			t.Errorf("expected 1 retained trajectory, got %d", result.RetainedCount)
		}
		if result.RejectedCount != 1 {
			t.Errorf("expected 1 rejected trajectory, got %d", result.RejectedCount)
		}
		if result.Trajectories[0].ID != "traj-oracle-match" {
			t.Errorf("expected traj-oracle-match retained, got %s", result.Trajectories[0].ID)
		}
	})

	t.Run("partitions batch correctly and exports JSONL", func(t *testing.T) {
		batch := []ReasoningTrajectory{
			{
				ID:                  "traj-1",
				Prompt:              "Task 1",
				ThoughtTrace:        "Trace 1",
				SelfReportedSuccess: true,
				ExternalVerification: ExternalProof{
					Verified: true,
					ExitCode: 0,
				},
			},
			{
				ID:                  "traj-2",
				Prompt:              "Task 2",
				ThoughtTrace:        "Trace 2",
				SelfReportedSuccess: true,
				ExternalVerification: ExternalProof{
					Verified: false,
				},
			},
			{
				ID:                  "traj-3",
				Prompt:              "Task 3",
				ThoughtTrace:        "Trace 3",
				SelfReportedSuccess: false,
				ExternalVerification: ExternalProof{
					Verified: true,
					ExitCode: 0,
				},
			},
			{
				ID:                  "traj-4",
				Prompt:              "Task 4",
				ThoughtTrace:        "Trace 4",
				SelfReportedSuccess: true,
				ExternalVerification: ExternalProof{
					Verified: true,
					ExitCode: 2,
				},
			},
			{
				ID:                  "traj-5",
				Prompt:              "Task 5",
				ThoughtTrace:        "Trace 5",
				SelfReportedSuccess: true,
				ExternalVerification: ExternalProof{
					Verified: true,
					ExitCode: 0,
				},
			},
		}

		dataset := filter.FilterTrajectories(batch)
		if dataset.TotalCount != 5 {
			t.Errorf("expected TotalCount 5, got %d", dataset.TotalCount)
		}
		if dataset.RetainedCount != 2 {
			t.Errorf("expected RetainedCount 2, got %d", dataset.RetainedCount)
		}
		if dataset.RejectedCount != 3 {
			t.Errorf("expected RejectedCount 3, got %d", dataset.RejectedCount)
		}
		if dataset.Len() != 2 {
			t.Errorf("expected Len() 2, got %d", dataset.Len())
		}
		if dataset.IsEmpty() {
			t.Errorf("expected IsEmpty() to be false")
		}

		prompts := dataset.ToTuningPrompts()
		if len(prompts) != 2 || prompts[0] != "Task 1" || prompts[1] != "Task 5" {
			t.Errorf("unexpected prompts: %v", prompts)
		}

		var buf bytes.Buffer
		if err := dataset.ExportJSONL(&buf); err != nil {
			t.Fatalf("unexpected ExportJSONL error: %v", err)
		}

		lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
		if len(lines) != 2 {
			t.Fatalf("expected 2 JSONL lines, got %d", len(lines))
		}
	})
}
