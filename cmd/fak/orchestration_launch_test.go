package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
)

func externalOrchestrationTestHome(t *testing.T) string {
	t.Helper()
	base, err := os.UserCacheDir()
	if err != nil {
		t.Fatal(err)
	}
	home, err := os.MkdirTemp(base, "fak-orchestration-test-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(home) })
	return home
}
func TestOrchestrationLaunchWritesJoinedWorkerReceipt(t *testing.T) {
	home := externalOrchestrationTestHome(t)
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
	for _, worker := range receipt.Workers {
		if worker.Model != "gpt-5.6-sol" || worker.Mode != "ultra" || worker.Effort != "high" {
			t.Fatalf("grind worker route = %s/%s/%s, want gpt-5.6-sol/ultra/high: %+v", worker.Model, worker.Mode, worker.Effort, worker)
		}
	}
}

func TestOrchestrationLaunchRecordsDirectDecline(t *testing.T) {
	home := externalOrchestrationTestHome(t)
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

func TestOrchestrationLaunchRefusesUnsupportedProRoute(t *testing.T) {
	home := externalOrchestrationTestHome(t)
	t.Setenv("CODEX_THREAD_ID", "session-pro")
	old := orchestrationWorkerLauncher
	orchestrationWorkerLauncher = func(orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		t.Fatal("unsupported Pro route launched a standard Codex worker")
		return codexOrchestrationWorkerLaunch{}, nil
	}
	t.Cleanup(func() { orchestrationWorkerLauncher = old })
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--task-text", "consult pro for an adversarial review and then verify the result", "--codex-home", home, "--launch", "--json"})
	if code != 1 || !strings.Contains(stderr.String(), "SOL_ROUTE_PRO_CONSULT_ONLY") {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
}

func TestOrchestrationWorkerArgsPinResolvedSOLRoute(t *testing.T) {
	args := orchestrationWorkerArgs(orchestrationWorkerLaunchRequest{Model: "gpt-5.6-sol", Effort: "high"}, "audit.jsonl")
	joined := strings.Join(args, " ")
	for _, want := range []string{"--codex-loop-gate off", `model="gpt-5.6-sol"`, `model_reasoning_effort="high"`} {
		if !strings.Contains(joined, want) {
			t.Fatalf("worker args missing %q: %q", want, joined)
		}
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

func TestOrchestrationLaunchRejectsGitWorktreeArtifactHomeBeforeWriting(t *testing.T) {
	home := t.TempDir()
	git := exec.Command("git", "init", "--quiet", home)
	if output, err := git.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, output)
	}
	t.Setenv("CODEX_THREAD_ID", "session-worktree-home")
	old := orchestrationWorkerLauncher
	orchestrationWorkerLauncher = func(orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		t.Fatal("unsafe worktree home launched a worker")
		return codexOrchestrationWorkerLaunch{}, nil
	}
	t.Cleanup(func() { orchestrationWorkerLauncher = old })

	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--task-text", "continue the multi-step implementation, add observability, dogfood it, and ship it", "--codex-home", filepath.Join(home, "nested", "state"), "--launch", "--json"})
	if code != 1 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	for _, want := range []string{"unsafe Codex home", "inside Git worktree", "omit --codex-home", "$CODEX_HOME", "external path"} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("stderr=%q, want %q", stderr.String(), want)
		}
	}
	for _, name := range []string{"fak-orchestration-invocations", "fak-orchestration-launches", "fak-orchestration-runs"} {
		if _, err := os.Stat(filepath.Join(home, "nested", "state", name)); !os.IsNotExist(err) {
			t.Fatalf("%s exists after rejected launch: %v", name, err)
		}
	}
}

func TestOrchestrationWorkerEnvMarksChildAndStripsParentIdentity(t *testing.T) {
	got := orchestrationWorkerEnv([]string{"PATH=keep", "CODEX_THREAD_ID=parent", "FAK_GUARD_PARENT=strip", orchestrationChildEnv + "=stale"})
	want := []string{"PATH=keep", orchestrationChildEnv + "=1"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("env = %q, want %q", got, want)
	}
}

func TestOrchestrationLaunchRefusesChildBeforeWritingReceipts(t *testing.T) {
	home := t.TempDir()
	t.Setenv("CODEX_THREAD_ID", "child-session")
	t.Setenv(orchestrationChildEnv, "1")
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--task-text", "split independent checks and reconcile them", "--codex-home", home, "--launch", "--json"})
	if code != 2 || !strings.Contains(stderr.String(), "nested --launch is refused") {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	for _, name := range []string{"fak-orchestration-invocations", "fak-orchestration-launches", "fak-orchestration-runs"} {
		if _, err := os.Stat(filepath.Join(home, name)); !os.IsNotExist(err) {
			t.Fatalf("%s exists after nested refusal: %v", name, err)
		}
	}
}
