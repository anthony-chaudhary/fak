package tb4bench

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

func TestFakHarnessSyntheticRun(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "tb4-fakharness-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	mockEngine := NewMockContainerEngine()
	defer mockEngine.Close()

	ctx := context.Background()
	config := ContainerConfig{
		ImageDigest: "ghcr.io/fak/tb4@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		Name:        "tb4-harness-c",
		NetworkMode: NetworkModeNone,
		WorkingDir:  "/workspace",
	}
	inst, err := mockEngine.CreateContainer(ctx, config)
	if err != nil {
		t.Fatalf("failed to create container: %v", err)
	}

	wsDir := mockEngine.workspaces[inst.ID]
	wsMgr := NewWorkspaceManager(mockEngine, inst.ID, "synth-task-01", wsDir)

	// Seed workspace with a broken script
	initialFiles := map[string][]byte{
		"main.py": []byte("print('broken')\n"),
	}
	_, err = wsMgr.SeedWorkspace(ctx, "Fix the script to print 'fixed'", initialFiles)
	if err != nil {
		t.Fatalf("failed to seed workspace: %v", err)
	}

	// Create in-kernel model adapter with 3 scripted responses:
	// Turn 1: Propose a forbidden tool call (e.g. "sudo_rm" or unauthorized tool denied by policy)
	// Turn 2: Propose allowed edit_file tool call to fix main.py
	// Turn 3: Emit TASK_COMPLETED
	adapter, err := NewInKernelModelAdapter("", "")
	if err != nil {
		t.Fatalf("failed to initialize adapter: %v", err)
	}

	adapter.RegisterScriptedResponse("Fix the script", &CompletionResponse{
		Text: "Attempting privileged cleanup first.",
		ToolCalls: []ToolCallProposal{
			{
				ID:        "call_1_rm",
				Name:      "rm_root",
				Arguments: `{"path": "/"}`,
			},
		},
		PromptTokens:     50,
		CompletionTokens: 25,
		FinishReason:     "tool_calls",
	})

	adapter.RegisterScriptedResponse("kernel security policy", &CompletionResponse{
		Text: "Policy blocked privileged action. Now fixing main.py properly.",
		ToolCalls: []ToolCallProposal{
			{
				ID:        "call_2_edit",
				Name:      "edit_file",
				Arguments: `{"path": "main.py", "find": "broken", "replace": "fixed"}`,
			},
		},
		PromptTokens:     80,
		CompletionTokens: 30,
		FinishReason:     "tool_calls",
	})

	adapter.RegisterScriptedResponse("Successfully edited main.py", &CompletionResponse{
		Text:             "Verification finished. TASK_COMPLETED",
		PromptTokens:     110,
		CompletionTokens: 10,
		FinishReason:     "stop",
	})

	// Define kernel security policy that denies rm_root and allows read_file, edit_file, bash
	pol := adjudicator.Policy{
		Allow: map[string]bool{
			"edit_file":  true,
			"read_file":  true,
			"write_file": true,
			"bash":       true,
		},
		Deny: map[string]abi.ReasonCode{
			"rm_root": abi.ReasonPolicyBlock,
		},
	}

	harness := NewFakHarnessWithPolicy(adapter, pol, wsMgr)

	task := TaskManifest{
		TaskID:         "synth-task-01",
		Category:       CategoryRefactor,
		Prompt:         "Fix the script to print 'fixed'",
		BudgetTurns:    5,
		TimeoutSeconds: 60,
	}

	res, err := harness.ExecuteTask(ctx, task, DefaultDeterminismEnvelope())
	if err != nil {
		t.Fatalf("agent loop execution failed: %v", err)
	}

	if res.Status != "COMPLETED" {
		t.Errorf("expected status COMPLETED, got %s", res.Status)
	}
	if res.TotalTurns != 3 {
		t.Fatalf("expected exactly 3 turns, got %d", res.TotalTurns)
	}
	if res.PolicyBlocks != 1 {
		t.Errorf("expected 1 policy block, got %d", res.PolicyBlocks)
	}

	// Check Turn 1 adjudication verdict
	if res.Turns[0].AdjudicationVerdict != "DENIED (POLICY_BLOCK)" {
		t.Errorf("expected turn 1 verdict DENIED (POLICY_BLOCK), got %s", res.Turns[0].AdjudicationVerdict)
	}

	// Check Turn 2 adjudication verdict
	if res.Turns[1].AdjudicationVerdict != "ALLOWED" {
		t.Errorf("expected turn 2 verdict ALLOWED, got %s", res.Turns[1].AdjudicationVerdict)
	}

	// Check file was actually edited
	content, err := os.ReadFile(filepath.Join(wsDir, "main.py"))
	if err != nil {
		t.Fatalf("failed to read main.py: %v", err)
	}
	if !strings.Contains(string(content), "print('fixed')") {
		t.Errorf("main.py was not updated with fix: %s", string(content))
	}

	// Check journal hash is non-empty
	if len(res.JournalHash) != 64 {
		t.Errorf("expected 64-char journal hash, got %q", res.JournalHash)
	}
}
