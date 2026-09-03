package tb4bench

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestEndToEndSyntheticSuite(t *testing.T) {
	manifestPath := filepath.Join("..", "..", "testdata", "tb4bench", "synthetic_suite.json")
	suite, err := LoadManifestFile(manifestPath)
	if err != nil {
		t.Fatalf("failed to load synthetic manifest: %v", err)
	}

	if len(suite.Tasks) != 5 {
		t.Fatalf("expected 5 synthetic tasks, got %d", len(suite.Tasks))
	}

	// 1. Generate and validate contract
	var taskIDs []string
	for _, task := range suite.Tasks {
		taskIDs = append(taskIDs, task.TaskID)
	}
	contract := DefaultRunContract("qwen3.8-coder-7b-q4_k_m.gguf", "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "Q4_K_M", taskIDs)
	if err := contract.Validate(true); err != nil {
		t.Fatalf("contract validation failed: %v", err)
	}

	mockEngine := NewMockContainerEngine()
	defer mockEngine.Close()
	ctx := context.Background()

	armAReceipts := make(map[string]*GradingReceipt)
	armBReceipts := make(map[string]*GradingReceipt)
	grader := NewGrader()

	// Setup scripted responses for in-kernel adapter
	adapter, err := NewInKernelModelAdapter("", "")
	if err != nil {
		t.Fatalf("failed to init adapter: %v", err)
	}

	adapter.RegisterScriptedResponse("Fix the syntax error in main.py", &CompletionResponse{
		Text: "Fixed syntax.",
		ToolCalls: []ToolCallProposal{
			{ID: "c1", Name: "write_file", Arguments: `{"path":"main.py","content":"print('fixed')\n"}`},
		},
		PromptTokens:     100,
		CompletionTokens: 20,
	})
	adapter.RegisterScriptedResponse("Wrote 15 bytes to main.py", &CompletionResponse{
		Text:         "TASK_COMPLETED",
		PromptTokens: 120, CompletionTokens: 10,
	})

	adapter.RegisterScriptedResponse("Rebase feature branch", &CompletionResponse{
		Text:         "TASK_COMPLETED",
		PromptTokens: 100, CompletionTokens: 10,
	})
	adapter.RegisterScriptedResponse("Process log files into output.log", &CompletionResponse{
		Text: "Created log.",
		ToolCalls: []ToolCallProposal{
			{ID: "c3", Name: "write_file", Arguments: `{"path":"output.log","content":"processed logs\n"}`},
		},
		PromptTokens:     100,
		CompletionTokens: 20,
	})
	adapter.RegisterScriptedResponse("Wrote 15 bytes to output.log", &CompletionResponse{
		Text:         "TASK_COMPLETED",
		PromptTokens: 120, CompletionTokens: 10,
	})
	adapter.RegisterScriptedResponse("Configure missing PORT", &CompletionResponse{
		Text: "Configured PORT.",
		ToolCalls: []ToolCallProposal{
			{ID: "c4", Name: "write_file", Arguments: `{"path":".env","content":"PORT=8080\n"}`},
		},
		PromptTokens:     100,
		CompletionTokens: 20,
	})
	adapter.RegisterScriptedResponse("Wrote 10 bytes to .env", &CompletionResponse{
		Text:         "TASK_COMPLETED",
		PromptTokens: 120, CompletionTokens: 10,
	})
	adapter.RegisterScriptedResponse("Refactor Go package", &CompletionResponse{
		Text: "Refactored pkg.",
		ToolCalls: []ToolCallProposal{
			{ID: "c5", Name: "write_file", Arguments: `{"path":"pkg/refactored.go","content":"package pkg\n"}`},
		},
		PromptTokens:     100,
		CompletionTokens: 20,
	})
	adapter.RegisterScriptedResponse("Wrote 12 bytes to pkg/refactored.go", &CompletionResponse{
		Text:         "TASK_COMPLETED",
		PromptTokens: 120, CompletionTokens: 10,
	})

	// Execute Arm A for each task
	for _, task := range suite.Tasks {
		cConfig := ContainerConfig{
			ImageDigest: task.EnvironmentImageDigest,
			Name:        "e2e-" + task.TaskID,
			NetworkMode: NetworkModeNone,
			WorkingDir:  "/workspace",
		}
		inst, err := mockEngine.CreateContainer(ctx, cConfig)
		if err != nil {
			t.Fatalf("failed to create container: %v", err)
		}

		wsDir := mockEngine.workspaces[inst.ID]
		wsMgr := NewWorkspaceManager(mockEngine, inst.ID, task.TaskID, wsDir)
		_, _ = wsMgr.SeedWorkspace(ctx, task.Prompt, nil)

		harness := NewFakHarness(adapter, nil, wsMgr)
		armExec, err := harness.ExecuteTask(ctx, task, DefaultDeterminismEnvelope())
		if err != nil {
			t.Fatalf("task execution failed: %v", err)
		}

		receipt, err := grader.Grade(ctx, "fak_inkernel", task, wsMgr, armExec)
		if err != nil {
			t.Fatalf("grading failed: %v", err)
		}
		armAReceipts[task.TaskID] = receipt
	}

	// Arm B mock: 3 solved, 2 failed
	for i, task := range suite.Tasks {
		verdict := "SOLVED"
		reason := ReasonSolved
		exitCode := 0
		if i >= 3 {
			verdict = "FAILED"
			reason = ReasonTestFailed
			exitCode = 1
		}
		armBReceipts[task.TaskID] = &GradingReceipt{
			TaskID:        task.TaskID,
			Arm:           "opencode_llamacpp",
			Verdict:       verdict,
			FailureReason: reason,
			ExitCode:      exitCode,
			DurationMs:    1500,
			Timestamp:     time.Now().UTC().Format(time.RFC3339),
		}
	}

	telemetryA := TelemetryTierMetrics{
		TotalPromptTokens:     3000,
		TotalCompletionTokens: 800,
		VDSOHits:              24,
	}
	telemetryB := TelemetryTierMetrics{
		TotalPromptTokens:     4200,
		TotalCompletionTokens: 1100,
	}

	report, err := BuildCompareReport(contract, armAReceipts, armBReceipts, telemetryA, telemetryB, suite.Tasks)
	if err != nil {
		t.Fatalf("failed to build compare report: %v", err)
	}

	// Verify solve rates
	if report.ArmAMetrics.Official.SolvedTasks != 5 {
		t.Errorf("expected Arm A to solve all 5 tasks, got %d", report.ArmAMetrics.Official.SolvedTasks)
	}
	if report.ArmBMetrics.Official.SolvedTasks != 3 {
		t.Errorf("expected Arm B to solve 3 tasks, got %d", report.ArmBMetrics.Official.SolvedTasks)
	}
	if report.SolveRateDelta != 0.4 {
		t.Errorf("expected delta 0.4 (1.0 - 0.6), got %f", report.SolveRateDelta)
	}

	// Verify markdown output contains summary
	md := report.GenerateMarkdown()
	if len(md) == 0 {
		t.Errorf("markdown report should not be empty")
	}

	// Save golden output if missing
	goldenPath := filepath.Join("..", "..", "testdata", "tb4bench", "golden_compare.json")
	if _, err := os.Stat(goldenPath); os.IsNotExist(err) {
		data, _ := json.MarshalIndent(report, "", "  ")
		_ = os.WriteFile(goldenPath, data, 0644)
	}
}
