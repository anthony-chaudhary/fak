package nightrun

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/tb4bench"
)

// fak-test:runtime fast est=50ms lane=default
func TestGutcheckTaskHelpersAndBacklogDiscovery(t *testing.T) {
	canonical := Task{
		ID:         GutcheckTaskID,
		Title:      "Paired-arm gut-check quality benchmark",
		Run:        "fak tb4-gutcheck run --arm both",
		Source:     SourceBenchmark,
		Value:      ValueCoverage,
		Requires:   []Requirement{ReqDataset},
		Acceptance: "paired-arm CompareReport verdict within cost budget",
	}
	if !IsGutcheckTask(canonical) {
		t.Error("IsGutcheckTask should return true for canonical gutcheck task")
	}

	datasetRun := Task{
		ID:  "custom-gutcheck",
		Run: "fak bench tb4 run --dataset testdata/tb4bench/gutcheck_suite.json",
	}
	if !IsGutcheckTask(datasetRun) {
		t.Error("IsGutcheckTask should return true for task referencing gutcheck_suite.json")
	}

	other := Task{
		ID:  TB4TaskID,
		Run: "fak bench tb4 run --arm both --mock --dataset testdata/tb4bench/synthetic_suite.json",
	}
	if IsGutcheckTask(other) {
		t.Error("IsGutcheckTask should return false for non-gutcheck task")
	}

	found, ok := FindGutcheckTask([]Task{other, canonical})
	if !ok || found.ID != GutcheckTaskID {
		t.Fatalf("FindGutcheckTask = (%q, %v), want (%q, true)", found.ID, ok, GutcheckTaskID)
	}

	// Verify discovery in the real nightrun backlog built from benchcatalog.
	backlog, err := Backlog("")
	if err != nil {
		t.Fatalf("Backlog() returned error: %v", err)
	}
	bt, ok := FindGutcheckTask(backlog)
	if !ok {
		t.Fatal("Backlog() missing tb4-gutcheck task")
	}
	if bt.ID != GutcheckTaskID {
		t.Errorf("backlog task ID = %q, want %q", bt.ID, GutcheckTaskID)
	}
	if bt.Source != SourceBenchmark {
		t.Errorf("backlog task Source = %q, want %q", bt.Source, SourceBenchmark)
	}
	if bt.Value != ValueCoverage {
		t.Errorf("backlog task Value = %q, want %q", bt.Value, ValueCoverage)
	}
	if !requires(bt, ReqDataset) {
		t.Errorf("backlog task Requires = %v, want ReqDataset", bt.Requires)
	}
	if !bt.AutoRunnable() {
		t.Error("tb4-gutcheck backlog task must be AutoRunnable()")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestGutcheckNightrunParityPass(t *testing.T) {
	gutcheckRunnerMu.RLock()
	prev := gutcheckRunner
	gutcheckRunnerMu.RUnlock()
	t.Cleanup(func() { RegisterGutcheckRunner(prev) })

	RegisterGutcheckRunner(func(ctx context.Context, datasetPath string, outDir string, budget GutcheckBudget) (*GutcheckResult, error) {
		suite, err := tb4bench.LoadManifestFile(datasetPath)
		if err != nil {
			return nil, err
		}
		taskIDs := make([]string, 0, len(suite.Tasks))
		receiptsA := make(map[string]*tb4bench.GradingReceipt, len(suite.Tasks))
		receiptsB := make(map[string]*tb4bench.GradingReceipt, len(suite.Tasks))
		for _, tk := range suite.Tasks {
			taskIDs = append(taskIDs, tk.TaskID)
			receiptsA[tk.TaskID] = &tb4bench.GradingReceipt{
				TaskID: tk.TaskID, Arm: "fak_inkernel", Verdict: "SOLVED",
				FailureReason: tb4bench.ReasonSolved, DurationMs: 900,
			}
			receiptsB[tk.TaskID] = &tb4bench.GradingReceipt{
				TaskID: tk.TaskID, Arm: "opencode_llamacpp", Verdict: "SOLVED",
				FailureReason: tb4bench.ReasonSolved, DurationMs: 950,
			}
		}
		contract := tb4bench.DefaultRunContract("reference-model.gguf", "sha256:pinned", "Q4_K_M", taskIDs)
		report, err := tb4bench.BuildCompareReport(
			contract,
			receiptsA,
			receiptsB,
			tb4bench.TelemetryTierMetrics{TotalPromptTokens: 18000, TotalCompletionTokens: 4500},
			tb4bench.TelemetryTierMetrics{TotalPromptTokens: 18500, TotalCompletionTokens: 4600},
			suite.Tasks,
		)
		if err != nil {
			return nil, err
		}
		_ = report.Save(filepath.Join(outDir, "compare.json"), filepath.Join(outDir, "compare.md"))

		return &GutcheckResult{
			SuiteVersion:     suite.Version,
			TasksEvaluated:   len(suite.Tasks),
			ArmASolveRate:    report.ArmAMetrics.Official.SolveRate,
			ArmBSolveRate:    report.ArmBMetrics.Official.SolveRate,
			SolveRateDelta:   report.SolveRateDelta,
			ArmAPromptTokens: report.ArmAMetrics.Telemetry.TotalPromptTokens,
			ArmBPromptTokens: report.ArmBMetrics.Telemetry.TotalPromptTokens,
			TotalTokens:      45600,
			EstimatedCostUSD: 1.25,
			DurationSec:      180,
			MarkdownReport:   report.GenerateMarkdown(),
		}, nil
	})

	res, err := RunGutcheckNightrun(context.Background(), "", t.TempDir(), DefaultGutcheckBudget())
	if err != nil {
		t.Fatalf("RunGutcheckNightrun failed: %v", err)
	}
	if res.LogPath != "" {
		t.Cleanup(func() { _ = os.Remove(res.LogPath) })
	}

	if res.TasksEvaluated != 12 {
		t.Errorf("TasksEvaluated = %d, want 12", res.TasksEvaluated)
	}
	if res.SuiteVersion != GutcheckSuiteVersion {
		t.Errorf("SuiteVersion = %q, want %q", res.SuiteVersion, GutcheckSuiteVersion)
	}
	if res.Verdict != GutcheckVerdictParity {
		t.Errorf("Verdict = %q, want %q", res.Verdict, GutcheckVerdictParity)
	}
	if res.BudgetExceeded {
		t.Error("BudgetExceeded = true, want false")
	}
	if !res.ThresholdsMet {
		t.Errorf("ThresholdsMet = false, want true (err=%q)", res.Error)
	}
	if res.Outcome != "collected" {
		t.Errorf("Outcome = %q, want collected", res.Outcome)
	}

	logBytes, err := os.ReadFile(res.LogPath)
	if err != nil {
		t.Fatalf("failed to read gutcheck log %s: %v", res.LogPath, err)
	}
	logText := string(logBytes)
	for _, want := range []string{
		"TB4 Paired-Arm Gut-Check Nightrun Collector",
		`outcome="collected"`,
		"task_id: " + GutcheckTaskID,
		"verdict: PARITY",
		"tasks_evaluated: 12",
		"budget_exceeded: false",
		"thresholds_met: true",
	} {
		if !strings.Contains(logText, want) {
			t.Errorf("gutcheck log missing %q:\n%s", want, logText)
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestGutcheckNightrunCostBudgetGuard(t *testing.T) {
	gutcheckRunnerMu.RLock()
	prev := gutcheckRunner
	gutcheckRunnerMu.RUnlock()
	t.Cleanup(func() { RegisterGutcheckRunner(prev) })

	// Negative budget must fail before execution.
	if _, err := RunGutcheckNightrun(context.Background(), "", t.TempDir(), GutcheckBudget{MaxCostUSD: -1}); err == nil {
		t.Fatal("expected error for negative MaxCostUSD")
	}

	// Exceeding cost and token caps must trip the budget guard and return REFUSED.
	RegisterGutcheckRunner(func(ctx context.Context, datasetPath string, outDir string, budget GutcheckBudget) (*GutcheckResult, error) {
		return &GutcheckResult{
			SuiteVersion:     GutcheckSuiteVersion,
			TasksEvaluated:   12,
			ArmASolveRate:    1.0,
			ArmBSolveRate:    1.0,
			SolveRateDelta:   0.0,
			TotalTokens:      150000, // > 100000 default
			EstimatedCostUSD: 4.50,   // > 2.00 default
			DurationSec:      120,
		}, nil
	})

	res, err := RunGutcheckNightrun(context.Background(), "", t.TempDir(), DefaultGutcheckBudget())
	if res != nil && res.LogPath != "" {
		t.Cleanup(func() { _ = os.Remove(res.LogPath) })
	}
	if err == nil {
		t.Fatal("expected error when cost/token budget is exceeded")
	}
	if !res.BudgetExceeded {
		t.Error("expected BudgetExceeded = true")
	}
	if res.Verdict != GutcheckVerdictRefused {
		t.Errorf("Verdict = %q, want %q", res.Verdict, GutcheckVerdictRefused)
	}
	if res.Outcome != "failed" {
		t.Errorf("Outcome = %q, want failed", res.Outcome)
	}
	if res.ThresholdsMet {
		t.Error("ThresholdsMet = true, want false")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestGutcheckNightrunDegradedVerdictFailsGate(t *testing.T) {
	gutcheckRunnerMu.RLock()
	prev := gutcheckRunner
	gutcheckRunnerMu.RUnlock()
	t.Cleanup(func() { RegisterGutcheckRunner(prev) })

	RegisterGutcheckRunner(func(ctx context.Context, datasetPath string, outDir string, budget GutcheckBudget) (*GutcheckResult, error) {
		return &GutcheckResult{
			SuiteVersion:     GutcheckSuiteVersion,
			TasksEvaluated:   12,
			ArmASolveRate:    0.50,
			ArmBSolveRate:    0.92,
			SolveRateDelta:   -0.42,
			TotalTokens:      40000,
			EstimatedCostUSD: 1.10,
			DurationSec:      150,
		}, nil
	})

	res, err := RunGutcheckNightrun(context.Background(), "", t.TempDir(), DefaultGutcheckBudget())
	if res != nil && res.LogPath != "" {
		t.Cleanup(func() { _ = os.Remove(res.LogPath) })
	}
	if err == nil {
		t.Fatal("expected error for DEGRADED paired-arm verdict")
	}
	if res.BudgetExceeded {
		t.Error("expected BudgetExceeded = false")
	}
	if res.Verdict != GutcheckVerdictDegraded {
		t.Errorf("Verdict = %q, want %q", res.Verdict, GutcheckVerdictDegraded)
	}
	if res.Outcome != "failed" {
		t.Errorf("Outcome = %q, want failed", res.Outcome)
	}
	if res.ThresholdsMet {
		t.Error("ThresholdsMet = true, want false")
	}
}
