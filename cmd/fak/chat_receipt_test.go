package main

import (
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestChatReceiptAsJSON(t *testing.T) {
	root := t.TempDir()
	filePath := filepath.Join(root, "note.txt")
	if err := os.WriteFile(filePath, []byte("headless receipt content\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := agent.ArmFocusedCodeTools(root)
	if err != nil {
		t.Fatal(err)
	}
	defer agent.DisarmCodeTools()

	relPath := filepath.ToSlash(filePath)
	planner := &chatScript{turns: []*agent.Completion{
		toolTurn("Read", `{"file_path":"`+relPath+`"}`),
		finalTurn("Headless task completed successfully."),
	}}

	var out strings.Builder
	err = runChatHeadless(&out, planner, "read note.txt", 10, true, "", root, agent.WithToolCatalog(catalog))
	if err != nil {
		t.Fatalf("runChatHeadless failed: %v", err)
	}

	var receipt nativeAgentReceipt
	if err := json.Unmarshal([]byte(out.String()), &receipt); err != nil {
		t.Fatalf("failed to unmarshal receipt JSON: %v\nOutput was:\n%s", err, out.String())
	}

	if receipt.Schema != nativeAgentReceiptSchema {
		t.Fatalf("receipt.Schema = %q, want %q", receipt.Schema, nativeAgentReceiptSchema)
	}
	if receipt.Status != "completed" {
		t.Fatalf("receipt.Status = %q, want completed", receipt.Status)
	}
	if receipt.Task != "read note.txt" {
		t.Fatalf("receipt.Task = %q, want read note.txt", receipt.Task)
	}
	if receipt.Model != "chat-script" {
		t.Fatalf("receipt.Model = %q, want chat-script", receipt.Model)
	}
	if receipt.FinalAnswer != "Headless task completed successfully." {
		t.Fatalf("receipt.FinalAnswer = %q", receipt.FinalAnswer)
	}
	if receipt.Metrics.Arm != "fak" || receipt.Metrics.Turns <= 0 {
		t.Fatalf("unexpected metrics: %#v", receipt.Metrics)
	}
	if len(receipt.TouchedPaths) == 0 || receipt.TouchedPaths[0] != relPath {
		t.Fatalf("receipt.TouchedPaths = %v, want [%s]", receipt.TouchedPaths, relPath)
	}
}

func TestChatReceiptReceiptOut(t *testing.T) {
	root := t.TempDir()
	filePath := filepath.Join(root, "target.txt")
	if err := os.WriteFile(filePath, []byte("data\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := agent.ArmFocusedCodeTools(root)
	if err != nil {
		t.Fatal(err)
	}
	defer agent.DisarmCodeTools()

	relPath := filepath.ToSlash(filePath)
	planner := &chatScript{turns: []*agent.Completion{
		toolTurn("Read", `{"file_path":"`+relPath+`"}`),
		finalTurn("File read done."),
	}}

	receiptFile := filepath.Join(t.TempDir(), "subdir", "receipt.json")
	var out strings.Builder
	err = runChatHeadless(&out, planner, "process target", 10, false, receiptFile, root, agent.WithToolCatalog(catalog))
	if err != nil {
		t.Fatalf("runChatHeadless failed: %v", err)
	}

	// In non-JSON mode, stdout should contain human-readable output
	stdout := out.String()
	if !strings.Contains(stdout, "[tool] Read") || !strings.Contains(stdout, "File read done.") {
		t.Fatalf("expected tool execution and final text in stdout, got:\n%s", stdout)
	}

	// Verify receipt file was written to disk
	data, err := os.ReadFile(receiptFile)
	if err != nil {
		t.Fatalf("failed to read receipt file: %v", err)
	}
	var receipt nativeAgentReceipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		t.Fatalf("failed to unmarshal receipt from file: %v", err)
	}
	if receipt.Schema != nativeAgentReceiptSchema {
		t.Fatalf("receipt.Schema = %q, want %q", receipt.Schema, nativeAgentReceiptSchema)
	}
	if receipt.Status != "completed" {
		t.Fatalf("receipt.Status = %q, want completed", receipt.Status)
	}
	if receipt.FinalAnswer != "File read done." {
		t.Fatalf("receipt.FinalAnswer = %q", receipt.FinalAnswer)
	}
}

func TestChatReceiptTouchedPaths(t *testing.T) {
	calls := []agent.CallTrace{
		{Tool: "Read", Args: `{"file_path":"src/b.go"}`},
		{Tool: "Write", Args: `{"path":"src/a.go","content":"package a"}`},
		{Tool: "Edit", Args: `{"file_path":"src/b.go","old":"1","new":"2"}`}, // duplicate
		{Tool: "sandbox_read", Args: `{"path":"src/c.go"}`},
		{Tool: "mcp_read", Args: `{"file_paths":["src/d.go","src/e.go"]}`},
	}
	metrics := agent.ArmMetrics{FinalAnswer: "all touched"}
	receipt := newHeadlessAgentReceipt("multi touch", "test-model", metrics, calls, "", nil)

	want := []string{"src/a.go", "src/b.go", "src/c.go", "src/d.go", "src/e.go"}
	if len(receipt.TouchedPaths) != len(want) {
		t.Fatalf("receipt.TouchedPaths len = %d, want %d: %v", len(receipt.TouchedPaths), len(want), receipt.TouchedPaths)
	}
	for i, p := range want {
		if receipt.TouchedPaths[i] != p {
			t.Errorf("receipt.TouchedPaths[%d] = %q, want %q", i, receipt.TouchedPaths[i], p)
		}
	}
}

func TestChatReceiptStatuses(t *testing.T) {
	calls := []agent.CallTrace{}

	// 1. Error status
	recErr := newHeadlessAgentReceipt("task", "m", agent.ArmMetrics{}, calls, "", errors.New("boom"))
	if recErr.Status != "failed" {
		t.Fatalf("status = %q, want failed", recErr.Status)
	}

	// 2. Circuit breaker tripped
	recCB := newHeadlessAgentReceipt("task", "m", agent.ArmMetrics{CircuitBreakerTripped: true}, calls, "", nil)
	if recCB.Status != "stopped_circuit_breaker" {
		t.Fatalf("status = %q, want stopped_circuit_breaker", recCB.Status)
	}

	// 3. Turn cap exceeded
	recCap := newHeadlessAgentReceipt("task", "m", agent.ArmMetrics{HitTurnCap: true}, calls, "", nil)
	if recCap.Status != "turn_cap_exceeded" {
		t.Fatalf("status = %q, want turn_cap_exceeded", recCap.Status)
	}

	// 4. Completed
	recDone := newHeadlessAgentReceipt("task", "m", agent.ArmMetrics{TaskCompleted: true}, calls, "", nil)
	if recDone.Status != "completed" {
		t.Fatalf("status = %q, want completed", recDone.Status)
	}
}

func TestChatReceiptGitDiffHash(t *testing.T) {
	// Non-git directory should produce empty hash
	nonGitDir := t.TempDir()
	recNonGit := newHeadlessAgentReceipt("task", "m", agent.ArmMetrics{}, nil, nonGitDir, nil)
	if recNonGit.GitDiffHash != "" {
		t.Fatalf("expected empty GitDiffHash in non-git directory, got %q", recNonGit.GitDiffHash)
	}

	// Setup a temporary git repo with a commit
	gitDir := t.TempDir()
	initCmd := exec.Command("git", "init")
	initCmd.Dir = gitDir
	if err := initCmd.Run(); err != nil {
		t.Skipf("git not available or git init failed: %v", err)
	}
	exec.Command("git", "-C", gitDir, "config", "user.email", "test@fak.local").Run()
	exec.Command("git", "-C", gitDir, "config", "user.name", "Test").Run()

	testFile := filepath.Join(gitDir, "hello.txt")
	if err := os.WriteFile(testFile, []byte("initial\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	exec.Command("git", "-C", gitDir, "add", "hello.txt").Run()
	exec.Command("git", "-C", gitDir, "commit", "-m", "initial commit").Run()

	// Clean git repo should have empty diff hash
	recClean := newHeadlessAgentReceipt("task", "m", agent.ArmMetrics{}, nil, gitDir, nil)
	if recClean.GitDiffHash != "" {
		t.Fatalf("expected empty GitDiffHash in clean git repo, got %q", recClean.GitDiffHash)
	}

	// Modify file to create diff
	if err := os.WriteFile(testFile, []byte("modified\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	recDirty := newHeadlessAgentReceipt("task", "m", agent.ArmMetrics{}, nil, gitDir, nil)
	if !strings.HasPrefix(recDirty.GitDiffHash, "sha256:") {
		t.Fatalf("expected sha256: prefix, got %q", recDirty.GitDiffHash)
	}
	if len(recDirty.GitDiffHash) != 7+64 {
		t.Fatalf("expected sha256:<64-hex> (len 71), got len %d: %q", len(recDirty.GitDiffHash), recDirty.GitDiffHash)
	}
}

func TestNativeAgentEnforcementReceipt(t *testing.T) {
	const helperEnv = "FAK_CHAT_ENFORCEMENT_RECEIPT_HELPER"
	if os.Getenv(helperEnv) == "1" {
		for i, arg := range os.Args {
			if arg == "--" {
				cmdChat(os.Args[i+1:])
				return
			}
		}
		t.Fatal("chat enforcement receipt helper missing -- separator")
	}

	type toolState struct {
		System bool `json:"system"`
		MCP    bool `json:"mcp"`
		Skills bool `json:"skills"`
		Memory bool `json:"memory"`
	}
	type enforcementEvidence struct {
		Schema          string    `json:"schema"`
		GuardPosture    string    `json:"guard_posture"`
		PolicyDigest    string    `json:"policy_digest"`
		WorkspaceDigest string    `json:"workspace_digest"`
		Tools           toolState `json:"tools"`
	}
	type receiptEnvelope struct {
		Schema      string               `json:"schema"`
		Enforcement *enforcementEvidence `json:"enforcement"`
	}

	for _, tc := range []struct {
		name  string
		extra []string
		want  toolState
	}{
		{name: "effective_enabled", want: toolState{System: true, MCP: true, Skills: true, Memory: true}},
		{
			name: "explicit_disabled",
			extra: []string{
				"--tools=none", "--code-tools=false", "--sys-tools=false", "--mcp-tools=false", "--skills=false", "--memory=false",
			},
			want: toolState{},
		},
		{
			name: "skills_requested_without_code_tools",
			extra: []string{
				"--tools=none", "--code-tools=false", "--sys-tools=false", "--mcp-tools=false", "--skills=true", "--memory=false",
			},
			want: toolState{},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			workspace := t.TempDir()
			buildTestMemoryStore(t, filepath.Join(workspace, ".fak", "memory"))
			policyBytes := append([]byte(nil), guardDefaultPolicyJSON...)
			policyPath := filepath.Join(workspace, "policy.json")
			if err := os.WriteFile(policyPath, policyBytes, 0o600); err != nil {
				t.Fatal(err)
			}
			receiptPath := filepath.Join(t.TempDir(), "receipt.json")
			args := []string{
				os.Args[0], "-test.run=^TestNativeAgentEnforcementReceipt$", "--",
				"--offline", "--task", "emit enforcement receipt", "--max-turns", "1",
				"--receipt", receiptPath, "--code-workspace", workspace, "--policy", policyPath,
			}
			args = append(args, tc.extra...)
			cmd := exec.Command(args[0], args[1:]...)
			cmd.Env = append(os.Environ(), helperEnv+"=1", "FAK_AGENT_POSTURE=default_open", "FAK_GUARD_POSTURE=admit_and_log")
			if output, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("real fak chat receipt producer failed: %v\n%s", err, output)
			}

			data, err := os.ReadFile(receiptPath)
			if err != nil {
				t.Fatal(err)
			}
			var receipt receiptEnvelope
			if err := json.Unmarshal(data, &receipt); err != nil {
				t.Fatalf("decode native receipt: %v\n%s", err, data)
			}
			if receipt.Schema != nativeAgentReceiptSchema {
				t.Fatalf("receipt schema = %q, want %q", receipt.Schema, nativeAgentReceiptSchema)
			}
			if receipt.Enforcement == nil {
				t.Fatalf("receipt lacks child-originated enforcement evidence: %s", data)
			}
			got := receipt.Enforcement
			if got.Schema != "fak.agent.native.enforcement.v1" {
				t.Errorf("enforcement schema = %q", got.Schema)
			}
			if got.GuardPosture != "fail_closed" {
				t.Errorf("effective guard posture = %q, want fail_closed", got.GuardPosture)
			}
			if want := guardPolicyDigest(policyBytes); got.PolicyDigest != want {
				t.Errorf("policy digest = %q, want %q", got.PolicyDigest, want)
			}
			canonicalWorkspace, err := resolveOpsRunWorkspace(workspace)
			if err != nil {
				t.Fatal(err)
			}
			if want := opsRunDigest(canonicalWorkspace); got.WorkspaceDigest != want {
				t.Errorf("workspace digest = %q, want %q", got.WorkspaceDigest, want)
			}
			if got.Tools != tc.want {
				t.Errorf("effective tools = %+v, want %+v", got.Tools, tc.want)
			}
		})
	}
}
