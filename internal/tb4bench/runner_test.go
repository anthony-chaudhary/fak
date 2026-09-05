package tb4bench

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestRunCampaignSyntheticMock(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "tb4-campaign-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	manifestPath := filepath.Join("..", "..", "testdata", "tb4bench", "synthetic_suite.json")
	suite, err := LoadManifestFile(manifestPath)
	if err != nil {
		t.Fatalf("failed to load manifest: %v", err)
	}

	ctx := context.Background()

	// 1. Run Campaign in MockMode
	runCfg := RunCampaignConfig{
		Tasks:       suite.Tasks,
		Arm:         "both",
		OutDir:      tempDir,
		MockMode:    true,
		Determinism: DefaultDeterminismEnvelope(),
	}

	runRes, err := RunCampaign(ctx, runCfg)
	if err != nil {
		t.Fatalf("RunCampaign failed: %v", err)
	}

	if len(runRes.ArmExecuted) != 2 {
		t.Fatalf("expected 2 arms executed, got %d", len(runRes.ArmExecuted))
	}

	// Verify required campaign output files
	for _, expectedFile := range []string{
		"contract.json",
		"manifest.json",
		filepath.Join("fak", "telemetry.json"),
		filepath.Join("opencode", "telemetry.json"),
	} {
		p := filepath.Join(tempDir, expectedFile)
		if _, err := os.Stat(p); err != nil {
			t.Errorf("missing expected output file %s: %v", expectedFile, err)
		}
	}

	for _, task := range suite.Tasks {
		for _, arm := range []string{"fak", "opencode"} {
			taskDir := filepath.Join(tempDir, arm, "tasks", task.TaskID)
			transcriptPath := filepath.Join(taskDir, "transcript.jsonl")
			resultPath := filepath.Join(taskDir, "result.json")
			wsPath := filepath.Join(taskDir, "workspace")

			if _, err := os.Stat(transcriptPath); err != nil {
				t.Errorf("missing transcript for %s %s: %v", arm, task.TaskID, err)
			}
			if _, err := os.Stat(resultPath); err != nil {
				t.Errorf("missing result.json for %s %s: %v", arm, task.TaskID, err)
			}
			if info, err := os.Stat(wsPath); err != nil || !info.IsDir() {
				t.Errorf("missing workspace directory for %s %s", arm, task.TaskID)
			}
		}
	}

	// 2. Evaluate Campaign
	evalCfg := EvaluateCampaignConfig{
		RunDir:  tempDir,
		Dataset: manifestPath,
	}

	evalRes, err := EvaluateCampaign(ctx, evalCfg)
	if err != nil {
		t.Fatalf("EvaluateCampaign failed: %v", err)
	}

	if evalRes.SolvedCount["fak"] != 5 {
		t.Errorf("expected Arm A (fak) to solve 5 tasks, got %d", evalRes.SolvedCount["fak"])
	}
	if evalRes.SolvedCount["opencode"] != 3 {
		t.Errorf("expected Arm B (opencode) to solve 3 tasks, got %d", evalRes.SolvedCount["opencode"])
	}

	// Verify grading receipts on disk
	for _, task := range suite.Tasks {
		for _, arm := range []string{"fak", "opencode"} {
			receiptPath := filepath.Join(tempDir, arm, "tasks", task.TaskID, "receipt.json")
			receipt, err := LoadReceipt(receiptPath)
			if err != nil {
				t.Errorf("failed to load receipt %s: %v", receiptPath, err)
			}
			if receipt.TaskID != task.TaskID {
				t.Errorf("receipt task mismatch: got %s, want %s", receipt.TaskID, task.TaskID)
			}
		}
	}

	// 3. Compare Campaign
	jsonOut := filepath.Join(tempDir, "compare_report.json")
	mdOut := filepath.Join(tempDir, "compare_report.md")

	compCfg := CompareCampaignConfig{
		FakDir:      filepath.Join(tempDir, "fak"),
		OpenCodeDir: filepath.Join(tempDir, "opencode"),
		Dataset:     manifestPath,
		OutJSON:     jsonOut,
		OutMD:       mdOut,
	}

	report, err := CompareCampaign(ctx, compCfg)
	if err != nil {
		t.Fatalf("CompareCampaign failed: %v", err)
	}

	if report.ArmAMetrics.Official.SolvedTasks != 5 {
		t.Errorf("expected Arm A solved 5, got %d", report.ArmAMetrics.Official.SolvedTasks)
	}
	if report.ArmBMetrics.Official.SolvedTasks != 3 {
		t.Errorf("expected Arm B solved 3, got %d", report.ArmBMetrics.Official.SolvedTasks)
	}
	if report.SolveRateDelta != 0.4 {
		t.Errorf("expected solve rate delta 0.4, got %f", report.SolveRateDelta)
	}
	if report.WinTieLoss.ArmAWins != 2 {
		t.Errorf("expected Arm A wins = 2, got %d", report.WinTieLoss.ArmAWins)
	}
	if report.WinTieLoss.BothSolved != 3 {
		t.Errorf("expected Both solved = 3, got %d", report.WinTieLoss.BothSolved)
	}

	if _, err := os.Stat(jsonOut); err != nil {
		t.Errorf("missing out JSON report: %v", err)
	}
	if _, err := os.Stat(mdOut); err != nil {
		t.Errorf("missing out MD report: %v", err)
	}
}

func TestRunCampaignSingleArm(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "tb4-single-arm-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	manifestPath := filepath.Join("..", "..", "testdata", "tb4bench", "synthetic_suite.json")
	suite, err := LoadManifestFile(manifestPath)
	if err != nil {
		t.Fatalf("failed to load manifest: %v", err)
	}

	ctx := context.Background()

	// Run only fak arm
	runCfg := RunCampaignConfig{
		Tasks:       suite.Tasks[:1],
		Arm:         "fak",
		OutDir:      filepath.Join(tempDir, "fak_only"),
		MockMode:    true,
		Determinism: DefaultDeterminismEnvelope(),
	}

	res, err := RunCampaign(ctx, runCfg)
	if err != nil {
		t.Fatalf("single arm run failed: %v", err)
	}
	if len(res.ArmExecuted) != 1 || res.ArmExecuted[0] != "fak" {
		t.Errorf("expected only 'fak' executed, got %v", res.ArmExecuted)
	}
	if _, err := os.Stat(filepath.Join(tempDir, "fak_only", "fak")); err != nil {
		t.Errorf("missing fak directory: %v", err)
	}
	if _, err := os.Stat(filepath.Join(tempDir, "fak_only", "opencode")); !os.IsNotExist(err) {
		t.Errorf("opencode directory should not exist in fak-only run")
	}
}

func TestRunCampaignValidationErrors(t *testing.T) {
	ctx := context.Background()

	// Empty tasks
	_, err := RunCampaign(ctx, RunCampaignConfig{
		OutDir: "some_dir",
	})
	if err == nil {
		t.Errorf("expected error for empty tasks")
	}

	// Empty outDir
	_, err = RunCampaign(ctx, RunCampaignConfig{
		Tasks: []TaskManifest{{TaskID: "t1"}},
	})
	if err == nil {
		t.Errorf("expected error for empty outDir")
	}

	// Invalid arm
	_, err = RunCampaign(ctx, RunCampaignConfig{
		Tasks:  []TaskManifest{{TaskID: "t1"}},
		OutDir: "some_dir",
		Arm:    "invalid_arm",
	})
	if err == nil {
		t.Errorf("expected error for invalid arm")
	}
}
