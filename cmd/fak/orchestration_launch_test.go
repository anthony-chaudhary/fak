package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
)

func TestOrchestrationLaunchWritesJoinedWorkerReceipt(t *testing.T) {
	home := t.TempDir()
	t.Setenv("CODEX_THREAD_ID", "session-launch")
	old := orchestrationWorkerLauncher
	orchestrationWorkerLauncher = func(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: 100 + len(req.Role.ID), Status: "started", LogPath: filepath.Join(req.RunDir, req.Role.ID+".jsonl")}, nil
	}
	t.Cleanup(func() { orchestrationWorkerLauncher = old })
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--task-text", "continue the multi-step implementation, add observability, dogfood it, and ship it", "--codex-home", home, "--launch", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	receipt, ok := readCodexOrchestrationLaunchReceipt(home, "session-launch")
	if !ok {
		t.Fatal("missing valid launch receipt")
	}
	if receipt.Status != "launched" || receipt.ResolvedProfile != "ultracode" || len(receipt.Workers) != 3 {
		t.Fatalf("receipt=%+v", receipt)
	}
	if receipt.RunID == "" || receipt.TaskID == "" {
		t.Fatalf("receipt lacks join identity: %+v", receipt)
	}
}

func TestOrchestrationLaunchRecordsDirectDecline(t *testing.T) {
	home := t.TempDir()
	t.Setenv("CODEX_THREAD_ID", "session-direct")
	old := orchestrationWorkerLauncher
	orchestrationWorkerLauncher = func(orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		t.Fatal("direct plan launched worker")
		return codexOrchestrationWorkerLaunch{}, nil
	}
	t.Cleanup(func() { orchestrationWorkerLauncher = old })
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--task-text", "fix typo", "--codex-home", home, "--launch", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	receipt, ok := readCodexOrchestrationLaunchReceipt(home, "session-direct")
	if !ok || receipt.Status != "declined" || receipt.DeclineReason != "resolved-direct" || len(receipt.Workers) != 0 {
		t.Fatalf("receipt=%+v ok=%v", receipt, ok)
	}
}

func TestOrchestrationWorkerPromptIsBoundedReadOnly(t *testing.T) {
	got := orchestrationWorkerPrompt(orchestrationWorkerLaunchRequest{Role: orchestration.Role{ID: "worker-1", Purpose: "inspect"}, TaskText: "find evidence"})
	for _, want := range []string{"read-only", "Do not edit files", "find evidence"} {
		if !bytes.Contains([]byte(got), []byte(want)) {
			t.Fatalf("prompt missing %q: %s", want, got)
		}
	}
}

func TestReadCodexOrchestrationLaunchReceiptRejectsMalformed(t *testing.T) {
	home := t.TempDir()
	dir := filepath.Join(home, "fak-orchestration-launches")
	if err := os.MkdirAll(dir, 0700); err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(codexOrchestrationLaunchReceipt{Schema: "wrong", SessionID: "s", RunID: "r"})
	if err := os.WriteFile(filepath.Join(dir, "s.json"), raw, 0600); err != nil {
		t.Fatal(err)
	}
	if _, ok := readCodexOrchestrationLaunchReceipt(home, "s"); ok {
		t.Fatal("accepted malformed receipt")
	}
}
