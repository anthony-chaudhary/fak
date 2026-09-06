package tb4bench

import (
	"context"
	"errors"
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

type mockCountingPseudoAdapter struct {
	invocations int
}

func (a *mockCountingPseudoAdapter) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	a.invocations++
	return &CompletionResponse{
		Text:         "TASK_COMPLETED",
		FinishReason: "stop",
	}, nil
}

func (a *mockCountingPseudoAdapter) Reset() {}

func (a *mockCountingPseudoAdapter) Telemetry() EngineTelemetry {
	return EngineTelemetry{}
}

func (a *mockCountingPseudoAdapter) IsPseudo() bool {
	return true
}

func TestRejectStrictRealPseudoAdapter(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "tb4-reject-pseudo-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	ctx := context.Background()
	tasks := []TaskManifest{
		{
			TaskID:                 "tb4-synth-01-syntax-fix",
			Prompt:                 "Fix syntax in main.py",
			EnvironmentImageDigest: "ghcr.io/fak/tb4-sandbox@sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
		},
	}

	// 1. Explicit pseudo adapter with StrictRealParity = true
	adapter := &mockCountingPseudoAdapter{}
	contract := DefaultRunContract("qwen3.8-coder.gguf", "sha256:pinned", "Q4_K_M", []string{"tb4-synth-01-syntax-fix"})
	contract.TaskSelection.Parity.StrictRealParityRequired = true

	runCfg := RunCampaignConfig{
		Tasks:            tasks,
		Arm:              "fak",
		ModelPath:        "qwen3.8-coder.gguf",
		OutDir:           tempDir,
		StrictRealParity: true,
		Contract:         contract,
		Adapter:          adapter,
	}

	res, err := RunCampaign(ctx, runCfg)
	if err == nil {
		t.Fatalf("expected typed refusal error under strict real parity with pseudo adapter, got nil")
	}

	// Assert typed refusal
	var refusalErr *RefusalError
	if !errors.As(err, &refusalErr) {
		t.Fatalf("expected typed refusal error (*RefusalError), got %T: %v", err, err)
	}
	if refusalErr.Reason != ReasonStrictRealParityViolation {
		t.Errorf("expected refusal reason %q, got %q", ReasonStrictRealParityViolation, refusalErr.Reason)
	}

	// Assert zero adapter invocations
	if adapter.invocations != 0 {
		t.Errorf("expected 0 adapter invocations, got %d", adapter.invocations)
	}

	// Assert no successful strict-real receipt
	if res != nil {
		t.Errorf("expected nil result on refusal, got %+v", res)
	}
	for _, cand := range []string{
		filepath.Join(tempDir, "fak", "tasks", "tb4-synth-01-syntax-fix", "receipt.json"),
		filepath.Join(tempDir, "fak", "tasks", "tb4-synth-01-syntax-fix", "result.json"),
	} {
		if _, err := os.Stat(cand); !os.IsNotExist(err) {
			t.Errorf("expected file %s to not exist, found it on disk", cand)
		}
	}

	// 2. Default InKernelModelAdapter (pseudo) under strict real parity contract
	t.Run("DefaultInKernelAdapterRejected", func(t *testing.T) {
		tempDir2, err := os.MkdirTemp("", "tb4-reject-inkernel-*")
		if err != nil {
			t.Fatalf("failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(tempDir2)

		runCfg2 := RunCampaignConfig{
			Tasks:            tasks,
			Arm:              "fak",
			ModelPath:        "qwen3.8-coder.gguf",
			OutDir:           tempDir2,
			StrictRealParity: true,
			Contract:         contract,
		}

		res2, err2 := RunCampaign(ctx, runCfg2)
		if err2 == nil {
			t.Fatalf("expected typed refusal error for default in-kernel pseudo adapter, got nil")
		}
		var refErr2 *RefusalError
		if !errors.As(err2, &refErr2) {
			t.Fatalf("expected typed refusal error (*RefusalError), got %T: %v", err2, err2)
		}
		if refErr2.Reason != ReasonStrictRealParityViolation {
			t.Errorf("expected reason %q, got %q", ReasonStrictRealParityViolation, refErr2.Reason)
		}
		if res2 != nil {
			t.Errorf("expected nil result on refusal, got %+v", res2)
		}
	})

	// 3. Mock mode execution under strict real parity contract
	t.Run("MockModeRejected", func(t *testing.T) {
		tempDir3, err := os.MkdirTemp("", "tb4-reject-mock-*")
		if err != nil {
			t.Fatalf("failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(tempDir3)

		runCfg3 := RunCampaignConfig{
			Tasks:            tasks,
			Arm:              "fak",
			OutDir:           tempDir3,
			MockMode:         true,
			StrictRealParity: true,
		}

		res3, err3 := RunCampaign(ctx, runCfg3)
		if err3 == nil {
			t.Fatalf("expected typed refusal error for mock mode under strict real parity, got nil")
		}
		var refErr3 *RefusalError
		if !errors.As(err3, &refErr3) {
			t.Fatalf("expected typed refusal error (*RefusalError), got %T: %v", err3, err3)
		}
		if refErr3.Reason != ReasonStrictRealParityViolation {
			t.Errorf("expected reason %q, got %q", ReasonStrictRealParityViolation, refErr3.Reason)
		}
		if res3 != nil {
			t.Errorf("expected nil result on refusal, got %+v", res3)
		}
	})
}
