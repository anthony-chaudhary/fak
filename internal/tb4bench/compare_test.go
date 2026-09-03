package tb4bench

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCompareArtifactGeneration(t *testing.T) {
	contract := DefaultRunContract("qwen3.8-coder.gguf", "sha256:1234abcd", "Q4_K_M", []string{"task-1", "task-2"})

	tasks := []TaskManifest{
		{TaskID: "task-1", Category: CategoryRefactor},
		{TaskID: "task-2", Category: CategoryGit},
	}

	receiptsA := map[string]*GradingReceipt{
		"task-1": {TaskID: "task-1", Arm: "fak_inkernel", Verdict: "SOLVED", FailureReason: ReasonSolved, DurationMs: 1200},
		"task-2": {TaskID: "task-2", Arm: "fak_inkernel", Verdict: "SOLVED", FailureReason: ReasonSolved, DurationMs: 1500},
	}
	telemetryA := TelemetryTierMetrics{
		TotalPromptTokens:     2000,
		TotalCompletionTokens: 800,
		VDSOHits:              15,
		PolicyBlocks:          1,
	}

	receiptsB := map[string]*GradingReceipt{
		"task-1": {TaskID: "task-1", Arm: "opencode_llamacpp", Verdict: "SOLVED", FailureReason: ReasonSolved, DurationMs: 1600},
		"task-2": {TaskID: "task-2", Arm: "opencode_llamacpp", Verdict: "FAILED", FailureReason: ReasonTestFailed, DurationMs: 2500},
	}
	telemetryB := TelemetryTierMetrics{
		TotalPromptTokens:     2800,
		TotalCompletionTokens: 1100,
	}

	report, err := BuildCompareReport(contract, receiptsA, receiptsB, telemetryA, telemetryB, tasks)
	if err != nil {
		t.Fatalf("failed to build compare report: %v", err)
	}

	// 1. Check delta metrics
	if report.ArmAMetrics.Official.SolveRate != 1.0 {
		t.Errorf("expected Arm A solve rate 1.0, got %f", report.ArmAMetrics.Official.SolveRate)
	}
	if report.ArmBMetrics.Official.SolveRate != 0.5 {
		t.Errorf("expected Arm B solve rate 0.5, got %f", report.ArmBMetrics.Official.SolveRate)
	}
	if report.SolveRateDelta != 0.5 {
		t.Errorf("expected solve rate delta +0.5, got %f", report.SolveRateDelta)
	}

	// 2. Check head-to-head matrix
	if report.WinTieLoss.BothSolved != 1 {
		t.Errorf("expected 1 both solved, got %d", report.WinTieLoss.BothSolved)
	}
	if report.WinTieLoss.ArmAWins != 1 {
		t.Errorf("expected 1 Arm A win, got %d", report.WinTieLoss.ArmAWins)
	}

	// 3. Test Markdown generation
	md := report.GenerateMarkdown()
	if !strings.Contains(md, "# Terminal-Bench 4 Comparative Analysis") {
		t.Errorf("markdown missing title")
	}
	if !strings.Contains(md, "+50.0%") {
		t.Errorf("markdown missing delta solve rate +50.0%%: %s", md)
	}

	// 4. Test file export
	tempDir, err := os.MkdirTemp("", "tb4-compare-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	jsonPath := filepath.Join(tempDir, "compare.json")
	mdPath := filepath.Join(tempDir, "compare.md")
	if err := report.Save(jsonPath, mdPath); err != nil {
		t.Fatalf("failed to save compare report: %v", err)
	}

	if _, err := os.Stat(jsonPath); err != nil {
		t.Errorf("compare.json not created")
	}
	if _, err := os.Stat(mdPath); err != nil {
		t.Errorf("compare.md not created")
	}
}
