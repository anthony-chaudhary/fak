package tb4bench

import (
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func gutcheckSuitePath(t *testing.T) string {
	t.Helper()
	return filepath.Join("..", "..", "testdata", "tb4bench", "gutcheck_suite.json")
}

// fak-test:runtime fast est=50ms lane=default
func TestGutcheckSuiteLoadsAndValidatesBudgets(t *testing.T) {
	suite, err := LoadManifestFile(gutcheckSuitePath(t))
	if err != nil {
		t.Fatalf("failed to load gutcheck_suite.json: %v", err)
	}

	if suite.Benchmark != BenchmarkName {
		t.Errorf("Benchmark = %q, want %q", suite.Benchmark, BenchmarkName)
	}
	if suite.Version != "tb4-gutcheck-v1" {
		t.Errorf("Version = %q, want %q", suite.Version, "tb4-gutcheck-v1")
	}
	if len(suite.Tasks) != 12 {
		t.Fatalf("len(Tasks) = %d, want 12", len(suite.Tasks))
	}

	classCounts := map[string]int{}
	var totalTimeSec int
	var totalTokenBudget int

	for _, task := range suite.Tasks {
		if err := task.Validate(); err != nil {
			t.Errorf("task %s failed Validate(): %v", task.TaskID, err)
		}
		if err := task.VerifyOracleScript([]byte(task.VerificationOracle)); err != nil {
			t.Errorf("task %s oracle hash mismatch: %v", task.TaskID, err)
		}
		if task.TimeoutSeconds <= 0 || task.TimeoutSeconds > 120 {
			t.Errorf("task %s timeout_seconds = %d, want in (0, 120]", task.TaskID, task.TimeoutSeconds)
		}
		if task.BudgetTurns <= 0 || task.BudgetTurns > 10 {
			t.Errorf("task %s budget_turns = %d, want in (0, 10]", task.TaskID, task.BudgetTurns)
		}
		totalTimeSec += task.TimeoutSeconds

		if task.Metadata["suite"] != "tb4-gutcheck-v1" {
			t.Errorf("task %s metadata.suite = %q, want tb4-gutcheck-v1", task.TaskID, task.Metadata["suite"])
		}
		taskClass := task.Metadata["task_class"]
		switch taskClass {
		case "coding", "reasoning", "tool-call":
			classCounts[taskClass]++
		default:
			t.Errorf("task %s has unexpected task_class %q", task.TaskID, taskClass)
		}

		maxPrompt, err := strconv.Atoi(task.Metadata["max_prompt_tokens"])
		if err != nil || maxPrompt <= 0 {
			t.Errorf("task %s invalid max_prompt_tokens %q: %v", task.TaskID, task.Metadata["max_prompt_tokens"], err)
		}
		maxOutput, err := strconv.Atoi(task.Metadata["max_output_tokens"])
		if err != nil || maxOutput <= 0 {
			t.Errorf("task %s invalid max_output_tokens %q: %v", task.TaskID, task.Metadata["max_output_tokens"], err)
		}
		maxTotal, err := strconv.Atoi(task.Metadata["max_total_tokens"])
		if err != nil || maxTotal <= 0 {
			t.Errorf("task %s invalid max_total_tokens %q: %v", task.TaskID, task.Metadata["max_total_tokens"], err)
		}
		if maxTotal != maxPrompt+maxOutput {
			t.Errorf("task %s max_total_tokens = %d, want max_prompt_tokens(%d) + max_output_tokens(%d)",
				task.TaskID, maxTotal, maxPrompt, maxOutput)
		}
		hardTime, err := strconv.Atoi(task.Metadata["hard_time_budget_seconds"])
		if err != nil || hardTime != task.TimeoutSeconds {
			t.Errorf("task %s hard_time_budget_seconds = %q, want %d",
				task.TaskID, task.Metadata["hard_time_budget_seconds"], task.TimeoutSeconds)
		}
		totalTokenBudget += maxTotal
	}

	for _, cls := range []string{"coding", "reasoning", "tool-call"} {
		if classCounts[cls] != 4 {
			t.Errorf("task_class %q count = %d, want 4", cls, classCounts[cls])
		}
	}
	if totalTimeSec > 900 {
		t.Errorf("suite total time budget = %ds, want <= 900s", totalTimeSec)
	}
	if totalTokenBudget > 100000 {
		t.Errorf("suite total token budget = %d, want <= 100000", totalTokenBudget)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestGutcheckPairedArmCompareReportVerdicts(t *testing.T) {
	suite, err := LoadManifestFile(gutcheckSuitePath(t))
	if err != nil {
		t.Fatalf("failed to load gutcheck_suite.json: %v", err)
	}

	taskIDs := make([]string, 0, len(suite.Tasks))
	for _, tk := range suite.Tasks {
		taskIDs = append(taskIDs, tk.TaskID)
	}
	contract := DefaultRunContract("qwen3.8-coder-7b.gguf", "sha256:gutcheck-pinned", "Q4_K_M", taskIDs)

	// 1. Parity run: both arms solve all 12 tasks.
	parityA := make(map[string]*GradingReceipt, len(suite.Tasks))
	parityB := make(map[string]*GradingReceipt, len(suite.Tasks))
	for _, tk := range suite.Tasks {
		parityA[tk.TaskID] = &GradingReceipt{
			TaskID: tk.TaskID, Arm: "fak_inkernel", Verdict: "SOLVED",
			FailureReason: ReasonSolved, DurationMs: 1100,
		}
		parityB[tk.TaskID] = &GradingReceipt{
			TaskID: tk.TaskID, Arm: "opencode_llamacpp", Verdict: "SOLVED",
			FailureReason: ReasonSolved, DurationMs: 1150,
		}
	}

	parityReport, err := BuildCompareReport(
		contract,
		parityA,
		parityB,
		TelemetryTierMetrics{TotalPromptTokens: 20000, TotalCompletionTokens: 5000},
		TelemetryTierMetrics{TotalPromptTokens: 20500, TotalCompletionTokens: 5100},
		suite.Tasks,
	)
	if err != nil {
		t.Fatalf("BuildCompareReport (parity) failed: %v", err)
	}
	if parityReport.ArmAMetrics.Official.TotalTasks != 12 || parityReport.ArmBMetrics.Official.TotalTasks != 12 {
		t.Fatalf("expected 12 tasks per arm, got A=%d B=%d",
			parityReport.ArmAMetrics.Official.TotalTasks, parityReport.ArmBMetrics.Official.TotalTasks)
	}
	if parityReport.SolveRateDelta != 0 {
		t.Errorf("parity SolveRateDelta = %f, want 0", parityReport.SolveRateDelta)
	}
	if parityReport.WinTieLoss.BothSolved != 12 {
		t.Errorf("parity BothSolved = %d, want 12", parityReport.WinTieLoss.BothSolved)
	}

	// 2. Degraded run: candidate arm fails 4 tasks with closed taxonomy failure reasons.
	degradedA := make(map[string]*GradingReceipt, len(suite.Tasks))
	for i, tk := range suite.Tasks {
		if i < 4 {
			reason := ReasonTestFailed
			if i == 1 {
				reason = ReasonToolCallMalformed
			} else if i == 2 {
				reason = ReasonTimeoutAgent
			}
			if err := ValidateReason(reason); err != nil {
				t.Fatalf("invalid test failure reason %q: %v", reason, err)
			}
			degradedA[tk.TaskID] = &GradingReceipt{
				TaskID: tk.TaskID, Arm: "fak_inkernel", Verdict: "FAILED",
				FailureReason: reason, ExitCode: 1, DurationMs: 2400,
			}
		} else {
			degradedA[tk.TaskID] = &GradingReceipt{
				TaskID: tk.TaskID, Arm: "fak_inkernel", Verdict: "SOLVED",
				FailureReason: ReasonSolved, DurationMs: 1100,
			}
		}
	}

	degradedReport, err := BuildCompareReport(
		contract,
		degradedA,
		parityB,
		TelemetryTierMetrics{TotalPromptTokens: 24000, TotalCompletionTokens: 6000},
		TelemetryTierMetrics{TotalPromptTokens: 20500, TotalCompletionTokens: 5100},
		suite.Tasks,
	)
	if err != nil {
		t.Fatalf("BuildCompareReport (degraded) failed: %v", err)
	}
	if degradedReport.WinTieLoss.ArmBWins != 4 {
		t.Errorf("degraded ArmBWins = %d, want 4", degradedReport.WinTieLoss.ArmBWins)
	}
	if degradedReport.WinTieLoss.BothSolved != 8 {
		t.Errorf("degraded BothSolved = %d, want 8", degradedReport.WinTieLoss.BothSolved)
	}
	if degradedReport.SolveRateDelta >= -0.30 {
		t.Errorf("degraded SolveRateDelta = %.4f, want < -0.30", degradedReport.SolveRateDelta)
	}
	if len(degradedReport.Tasks) != 12 {
		t.Errorf("len(degradedReport.Tasks) = %d, want 12", len(degradedReport.Tasks))
	}
	for _, tc := range degradedReport.Tasks {
		if tc.ArmAVerdict == "FAILED" {
			if err := ValidateReason(tc.ArmAFailureReason); err != nil {
				t.Errorf("task %s ArmAFailureReason %q invalid: %v", tc.TaskID, tc.ArmAFailureReason, err)
			}
		}
	}

	md := degradedReport.GenerateMarkdown()
	if !strings.Contains(md, "tb4-gutcheck-01-coding-go-edgecase") {
		t.Errorf("markdown report missing gutcheck task id")
	}
	outDir := t.TempDir()
	if err := degradedReport.Save(filepath.Join(outDir, "compare.json"), filepath.Join(outDir, "compare.md")); err != nil {
		t.Fatalf("Save() failed: %v", err)
	}
}
