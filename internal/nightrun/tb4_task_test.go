package nightrun

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/tb4bench"
)

func init() {
	RegisterTB4Runner(func(ctx context.Context, datasetPath string, outDir string) (*TB4RegressionResult, error) {
		suite, err := tb4bench.LoadManifestFile(datasetPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load manifest %s: %w", datasetPath, err)
		}

		var taskIDs []string
		for _, t := range suite.Tasks {
			taskIDs = append(taskIDs, t.TaskID)
		}
		contract := tb4bench.DefaultRunContract("qwen3.8-coder-7b-q4_k_m.gguf", "sha256:pinned", "Q4_K_M", taskIDs)

		runCfg := tb4bench.RunCampaignConfig{
			Tasks:       suite.Tasks,
			Arm:         "both",
			OutDir:      outDir,
			MockMode:    true,
			Determinism: contract.Determinism,
			Contract:    contract,
		}

		if _, err := tb4bench.RunCampaign(ctx, runCfg); err != nil {
			return nil, fmt.Errorf("campaign execution failed: %w", err)
		}

		evalCfg := tb4bench.EvaluateCampaignConfig{
			RunDir:  outDir,
			Dataset: datasetPath,
			Tasks:   suite.Tasks,
		}
		if _, err := tb4bench.EvaluateCampaign(ctx, evalCfg); err != nil {
			return nil, fmt.Errorf("campaign evaluation failed: %w", err)
		}

		compCfg := tb4bench.CompareCampaignConfig{
			FakDir:      filepath.Join(outDir, "fak"),
			OpenCodeDir: filepath.Join(outDir, "opencode"),
			Dataset:     datasetPath,
			Tasks:       suite.Tasks,
			OutJSON:     filepath.Join(outDir, "compare.json"),
			OutMD:       filepath.Join(outDir, "compare.md"),
		}
		report, err := tb4bench.CompareCampaign(ctx, compCfg)
		if err != nil {
			return nil, fmt.Errorf("campaign comparison failed: %w", err)
		}

		return &TB4RegressionResult{
			ArmASolveRate:    report.ArmAMetrics.Official.SolveRate,
			ArmBSolveRate:    report.ArmBMetrics.Official.SolveRate,
			SolveRateDelta:   report.SolveRateDelta,
			ArmAPromptTokens: report.ArmAMetrics.Telemetry.TotalPromptTokens,
			ArmBPromptTokens: report.ArmBMetrics.Telemetry.TotalPromptTokens,
			VDSOHits:         report.ArmAMetrics.Telemetry.VDSOHits,
			MarkdownReport:   report.GenerateMarkdown(),
		}, nil
	})
}

func TestTB4NightrunRegressionAgainstSyntheticSuite(t *testing.T) {
	tempDir := t.TempDir()
	ctx := context.Background()

	res, err := RunTB4NightrunRegression(ctx, "", tempDir)
	if err != nil {
		t.Fatalf("RunTB4NightrunRegression failed unexpectedly: %v", err)
	}
	if res == nil {
		t.Fatal("expected non-nil TB4RegressionResult")
	}

	// 1. Verify solve rate >= 1.0 (100% on synthetic suite)
	if res.ArmASolveRate < TB4MinSolveRateArmA {
		t.Errorf("arm A solve rate = %.2f, want >= %.2f (100%%)", res.ArmASolveRate, TB4MinSolveRateArmA)
	}

	// Arm B should reflect OpenCode baseline (60% on synthetic suite)
	if res.ArmBSolveRate < 0.60 {
		t.Errorf("arm B solve rate = %.2f, expected ~0.60", res.ArmBSolveRate)
	}
	if res.SolveRateDelta <= 0 {
		t.Errorf("expected positive solve rate delta (+40%%), got %.2f", res.SolveRateDelta)
	}

	// 2. Verify token reduction >= 0.80 (80% prompt reduction via vDSO/caching)
	if res.TokenReduction < TB4MinTokenReduction {
		t.Errorf("token reduction = %.4f (%.1f%%), want >= %.2f (80%%)",
			res.TokenReduction, res.TokenReduction*100.0, TB4MinTokenReduction)
	}

	// 3. Verify telemetry deltas and vDSO hit counts
	if res.VDSOHits <= 0 {
		t.Errorf("expected positive vDSO hits, got %d", res.VDSOHits)
	}
	if res.TelemetryDeltas == nil {
		t.Fatal("expected telemetry deltas map to be populated")
	}
	if res.TelemetryDeltas["prompt_tokens_saved"] <= 0 {
		t.Errorf("expected prompt_tokens_saved > 0, got %.1f", res.TelemetryDeltas["prompt_tokens_saved"])
	}
	if res.TelemetryDeltas["vdso_hits"] != float64(res.VDSOHits) {
		t.Errorf("telemetry delta vdso_hits = %.0f, want %d", res.TelemetryDeltas["vdso_hits"], res.VDSOHits)
	}

	// 4. Verify regression gates hold
	if !res.ThresholdsMet {
		t.Errorf("expected thresholds to be met, got false (error: %s)", res.Error)
	}
	if res.Outcome != "collected" {
		t.Errorf("expected outcome = 'collected', got %q", res.Outcome)
	}

	// 5. Verify output log generation and format
	if res.LogPath == "" {
		t.Fatal("expected non-empty LogPath")
	}
	t.Cleanup(func() {
		_ = os.Remove(res.LogPath)
	})

	logBytes, err := os.ReadFile(res.LogPath)
	if err != nil {
		t.Fatalf("failed to read emitted nightrun log %s: %v", res.LogPath, err)
	}
	logContent := string(logBytes)

	expectedPhrases := []string{
		"Terminal-Bench 4 Nightrun Regression Smoke Collector",
		`outcome="collected"`,
		"task_id: " + TB4TaskID,
		"arm_a_solve_rate: 1.0000",
		"token_reduction:",
		"vdso_hits: 11",
		"thresholds_met: true",
		"--- Telemetry Deltas ---",
		"prompt_tokens_saved:",
		"--- Comparative Report ---",
	}
	for _, phrase := range expectedPhrases {
		if !strings.Contains(logContent, phrase) {
			t.Errorf("nightrun log missing expected phrase %q:\n%s", phrase, logContent)
		}
	}
}

func TestTB4NightrunRegressionGateThresholdFailures(t *testing.T) {
	// Preserve original runner
	tb4RunnerMu.RLock()
	prevRunner := tb4Runner
	tb4RunnerMu.RUnlock()
	defer func() {
		RegisterTB4Runner(prevRunner)
	}()

	// 1. Arm A solve rate regression (< 1.0)
	RegisterTB4Runner(func(ctx context.Context, datasetPath, outDir string) (*TB4RegressionResult, error) {
		return &TB4RegressionResult{
			ArmASolveRate:    0.80, // < 1.0
			ArmBSolveRate:    0.60,
			SolveRateDelta:   0.20,
			ArmAPromptTokens: 530,
			ArmBPromptTokens: 3210,
			VDSOHits:         11,
		}, nil
	})

	res, err := RunTB4NightrunRegression(context.Background(), "", t.TempDir())
	if err == nil {
		t.Fatal("expected error when Arm A solve rate drops below threshold")
	}
	if res.ThresholdsMet {
		t.Errorf("expected ThresholdsMet = false, got true")
	}
	if res.Outcome != "failed" {
		t.Errorf("expected Outcome = 'failed', got %q", res.Outcome)
	}
	if !strings.Contains(err.Error(), "arm A solve rate") {
		t.Errorf("expected error to mention arm A solve rate, got %v", err)
	}
	if res.LogPath != "" {
		_ = os.Remove(res.LogPath)
	}

	// 2. Token reduction regression (< 0.80)
	RegisterTB4Runner(func(ctx context.Context, datasetPath, outDir string) (*TB4RegressionResult, error) {
		return &TB4RegressionResult{
			ArmASolveRate:    1.00,
			ArmBSolveRate:    0.60,
			SolveRateDelta:   0.40,
			ArmAPromptTokens: 2500, // 2500 vs 3210 -> reduction 22.1% < 80%
			ArmBPromptTokens: 3210,
			VDSOHits:         2,
		}, nil
	})

	res, err = RunTB4NightrunRegression(context.Background(), "", t.TempDir())
	if err == nil {
		t.Fatal("expected error when prompt token reduction drops below threshold")
	}
	if res.ThresholdsMet {
		t.Errorf("expected ThresholdsMet = false, got true")
	}
	if res.Outcome != "failed" {
		t.Errorf("expected Outcome = 'failed', got %q", res.Outcome)
	}
	if !strings.Contains(err.Error(), "token reduction") {
		t.Errorf("expected error to mention token reduction, got %v", err)
	}
	if res.LogPath != "" {
		_ = os.Remove(res.LogPath)
	}
}

func TestTB4TaskHelpersAndResolution(t *testing.T) {
	canonicalTask := Task{
		ID:         TB4TaskID,
		Title:      "TB4 benchmark task",
		Run:        "fak bench tb4 run --arm both --mock --dataset testdata/tb4bench/synthetic_suite.json",
		Source:     SourceBenchmark,
		Value:      ValueCoverage,
		Acceptance: "solve rate >= 1.0",
	}

	if !IsTB4Task(canonicalTask) {
		t.Errorf("IsTB4Task should return true for canonical task")
	}

	aliasTask := Task{
		ID:  TB4TaskIDAlias,
		Run: "fak bench tb4 run",
	}
	if !IsTB4Task(aliasTask) {
		t.Errorf("IsTB4Task should return true for alias task")
	}

	otherTask := Task{
		ID:  "bench-ablate",
		Run: "fak ablate --sweep vdso",
	}
	if IsTB4Task(otherTask) {
		t.Errorf("IsTB4Task should return false for non-TB4 task")
	}

	taskList := []Task{otherTask, canonicalTask}
	found, ok := FindTB4Task(taskList)
	if !ok {
		t.Fatal("FindTB4Task failed to find canonical TB4 task")
	}
	if found.ID != TB4TaskID {
		t.Errorf("found task ID = %q, want %q", found.ID, TB4TaskID)
	}

	// Verify discovery in actual nightrun backlog
	backlog, err := Backlog("")
	if err != nil {
		t.Fatalf("Backlog() returned error: %v", err)
	}
	backlogTask, ok := FindTB4Task(backlog)
	if !ok {
		t.Fatal("Backlog() missing TB4 task")
	}
	if backlogTask.ID != TB4TaskID {
		t.Errorf("backlog task ID = %q, want %q", backlogTask.ID, TB4TaskID)
	}
	if !backlogTask.AutoRunnable() {
		t.Errorf("TB4 backlog task must be AutoRunnable()")
	}
}
