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
	"time"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
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
	var launchedRequests []orchestrationWorkerLaunchRequest
	orchestrationWorkerLauncher = func(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		launchedRequests = append(launchedRequests, req)
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
		if worker.Model != "gemini-3.8-flash" || worker.Mode != "ultra" || worker.Effort != "medium" {
			t.Fatalf("grind worker route = %s/%s/%s, want gemini-3.8-flash/ultra/medium: %+v", worker.Model, worker.Mode, worker.Effort, worker)
		}
	}
	if len(launchedRequests) != 3 {
		t.Fatalf("launched requests=%d, want 3", len(launchedRequests))
	}
	for _, req := range launchedRequests {
		if req.Model != "gemini-3.8-flash" || req.Effort != "medium" {
			t.Fatalf("grind worker request = %s/%s, want gemini-3.8-flash/medium: %+v", req.Model, req.Effort, req)
		}
		if req.Access.Mode != orchestration.ChildAccessObserve || !req.Access.Admission.ReadOnly ||
			len(req.Access.Admission.Tree) != 0 {
			t.Fatalf("--task-text inferred write authority for %+v", req)
		}
		prompt := orchestrationWorkerPrompt(req)
		if !strings.Contains(prompt, "Work read-only") || !strings.Contains(prompt, "Do not edit files") {
			t.Fatalf("observe-only prompt = %q", prompt)
		}
	}
}

func TestOrchestrationLaunchAstraManagerDelegatesToGeminiWorker(t *testing.T) {
	home := externalOrchestrationTestHome(t)
	t.Setenv("CODEX_THREAD_ID", "session-astra-gemini")
	old := orchestrationWorkerLauncher
	var launchedRequests []orchestrationWorkerLaunchRequest
	orchestrationWorkerLauncher = func(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		launchedRequests = append(launchedRequests, req)
		return codexOrchestrationWorkerLaunch{
			RoleID:  req.Role.ID,
			PID:     200 + len(req.Role.ID),
			Status:  "started",
			LogPath: filepath.Join(req.RunDir, req.Role.ID+".jsonl"),
		}, nil
	}
	t.Cleanup(func() { orchestrationWorkerLauncher = old })

	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{
		"plan",
		"--task-text", "parallel grind implementation and testing",
		"--codex-home", home,
		"--launch",
		"--json",
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	receipt, ok := readCodexOrchestrationLaunchReceipt(home, "session-astra-gemini")
	if !ok {
		t.Fatal("missing valid launch receipt")
	}
	if len(receipt.Workers) == 0 {
		t.Fatal("expected launched workers, got 0")
	}
	for _, worker := range receipt.Workers {
		if worker.Model != "gemini-3.8-flash" || worker.Effort != "medium" {
			t.Errorf("worker %s: got model=%q effort=%q, want gemini-3.8-flash/medium", worker.RoleID, worker.Model, worker.Effort)
		}
	}
	for _, req := range launchedRequests {
		if req.Model != "gemini-3.8-flash" || req.Effort != "medium" {
			t.Errorf("worker req %s: got model=%q effort=%q, want gemini-3.8-flash/medium", req.Role.ID, req.Model, req.Effort)
		}
	}

	// Explicit override preserved
	launchedRequests = nil
	t.Setenv("CODEX_THREAD_ID", "session-astra-override")
	stdout.Reset()
	stderr.Reset()
	code = runOrchestration(&stdout, &stderr, []string{
		"plan",
		"--task-text", "parallel grind implementation and testing",
		"--codex-home", home,
		"--worker-model", "custom-override-model",
		"--worker-effort", "high",
		"--launch",
		"--json",
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	receiptOverride, ok := readCodexOrchestrationLaunchReceipt(home, "session-astra-override")
	if !ok {
		t.Fatal("missing valid override launch receipt")
	}
	for _, worker := range receiptOverride.Workers {
		if worker.Model != "custom-override-model" || worker.Effort != "high" {
			t.Errorf("override worker %s: got model=%q effort=%q, want custom-override-model/high", worker.RoleID, worker.Model, worker.Effort)
		}
	}
	for _, req := range launchedRequests {
		if req.Model != "custom-override-model" || req.Effort != "high" {
			t.Errorf("override req %s: got model=%q effort=%q, want custom-override-model/high", req.Role.ID, req.Model, req.Effort)
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

func TestOrchestrationWorkerArgsUsePersistedHookTrust(t *testing.T) {
	args := orchestrationWorkerArgs(orchestrationWorkerLaunchRequest{Model: "gpt-5.6-sol", Effort: "high"}, "audit.jsonl")
	joined := strings.Join(args, " ")
	if strings.Contains(joined, "--dangerously-bypass-hook-trust") {
		t.Fatalf("orchestration worker bypasses persisted project-hook trust: %q", joined)
	}
	for _, want := range []string{"--", "codex exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", "--json"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("worker args missing %q after narrowing hook trust: %q", want, joined)
		}
	}
}

func TestCodexOrchestrationHookTrustWitnessBlocksBothArms(t *testing.T) {
	path := filepath.Join("..", "..", "experiments", "agent-live", "codex-orchestration-hook-trust-witness-2026-08-25.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var witness struct {
		Status string `json:"status"`
		Arms   []struct {
			BypassFlag           bool   `json:"bypass_flag"`
			ExitCode             int    `json:"exit_code"`
			HookDispatchCount    int    `json:"hook_dispatch_count"`
			HookDecision         string `json:"hook_decision"`
			ProviderTrafficCount int    `json:"provider_traffic_count"`
			ModelOutputCount     int    `json:"model_output_count"`
			ToolCallCount        int    `json:"tool_call_count"`
		} `json:"arms"`
	}
	if err := json.Unmarshal(data, &witness); err != nil {
		t.Fatal(err)
	}
	if witness.Status != "passed" || len(witness.Arms) != 2 || witness.Arms[0].BypassFlag || !witness.Arms[1].BypassFlag {
		t.Fatalf("witness does not contain matched persisted-trust and bypass arms: %+v", witness)
	}
	for _, arm := range witness.Arms {
		if arm.ExitCode != 0 || arm.HookDispatchCount != 1 || arm.HookDecision != "block" || arm.ProviderTrafficCount != 0 || arm.ModelOutputCount != 0 || arm.ToolCallCount != 0 {
			t.Fatalf("hook block did not stop arm before provider/model/tool traffic: %+v", arm)
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

func TestOrchestrationLaunchBoundsConfiguredQwenEmptyUsage(t *testing.T) {
	home := externalOrchestrationTestHome(t)
	configureQwenUsageTest(t, "1")
	oldLauncher := orchestrationWorkerLauncher
	oldMonitor := orchestrationWorkerUsageMonitor
	oldStopper := orchestrationWorkerStopper
	var launches, stops int
	orchestrationWorkerLauncher = func(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		launches++
		return codexOrchestrationWorkerLaunch{
			RoleID: req.Role.ID, PID: 100 + launches, Status: "started",
			LogPath: orchestrationWorkerLogPath(req), StartedAt: time.Date(2026, 8, 25, 12, 0, launches, 0, time.UTC),
		}, nil
	}
	orchestrationWorkerUsageMonitor = func(req orchestrationWorkerLaunchRequest, launched codexOrchestrationWorkerLaunch, window time.Duration, workload *orchestrationWorkloadReceipt) (trajectory.QwenEmptyUsageAssessment, error) {
		return trajectory.AssessQwenEmptyUsage(trajectory.QwenEmptyUsageInput{
			WorkloadKind: workload.Kind, TargetModelFamily: workload.TargetModelFamily,
			WorkerKind: workload.WorkerKind, UsageExpectation: workload.UsageExpectation,
			WorkerModel: req.Model, LaunchStatus: launched.Status, PID: launched.PID,
			StartedAt: launched.StartedAt, ObservedAt: launched.StartedAt.Add(window),
			Window: window, ProcessAlive: true,
			Usage: trajectory.CodexExecUsage{LogReadable: true, TurnsStarted: 1},
		}), nil
	}
	orchestrationWorkerStopper = func(int) error {
		stops++
		return nil
	}
	t.Cleanup(func() {
		orchestrationWorkerLauncher = oldLauncher
		orchestrationWorkerUsageMonitor = oldMonitor
		orchestrationWorkerStopper = oldStopper
	})

	receipt, err := launchCodexOrchestrationWorkers(home, "session-qwen-empty", "ultracode", "native", "run the configured performance workload", qwenEmptyUsageResolution())
	if err == nil || !strings.Contains(err.Error(), qwenEmptyUsageTerminalReason) {
		t.Fatalf("err=%v, want %s", err, qwenEmptyUsageTerminalReason)
	}
	if launches != 2 || stops != 2 || receipt.Status != "terminal" || len(receipt.Workers) != 1 {
		t.Fatalf("launches=%d stops=%d receipt=%+v", launches, stops, receipt)
	}
	if receipt.EmptyUsagePolicy == nil || receipt.EmptyUsagePolicy.MaxRecoveryAttempts != 1 ||
		len(receipt.EmptyUsagePolicy.ValidExclusions) != 3 {
		t.Fatalf("empty-usage policy = %+v", receipt.EmptyUsagePolicy)
	}
	worker := receipt.Workers[0]
	if worker.Terminal == nil || worker.Terminal.Schema != qwenEmptyUsageTerminalSchema ||
		worker.Terminal.Reason != qwenEmptyUsageTerminalReason || worker.Terminal.Attempts != 2 ||
		worker.Terminal.RecoveryAttempts != 1 || worker.RecoveryAttempts != 1 ||
		len(worker.AttemptLogs) != 2 || worker.AttemptLogs[0] == worker.AttemptLogs[1] {
		t.Fatalf("terminal worker = %+v", worker)
	}
	raw, err := os.ReadFile(filepath.Join(home, "fak-orchestration-launches", "session-qwen-empty.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(raw, []byte(`"empty_usage_policy"`)) {
		t.Fatalf("persisted launch receipt lacks empty_usage_policy: %s", raw)
	}
	persisted, ok := readCodexOrchestrationLaunchReceipt(home, "session-qwen-empty")
	if !ok || persisted.Workers[0].Terminal == nil || persisted.Workers[0].Terminal.Reason != qwenEmptyUsageTerminalReason ||
		!reflect.DeepEqual(persisted.EmptyUsagePolicy, receipt.EmptyUsagePolicy) {
		t.Fatalf("persisted=%+v ok=%v", persisted, ok)
	}
}

func TestOrchestrationLaunchLeavesHealthyQwenUsageAlone(t *testing.T) {
	home := externalOrchestrationTestHome(t)
	configureQwenUsageTest(t, "1")
	oldLauncher := orchestrationWorkerLauncher
	oldMonitor := orchestrationWorkerUsageMonitor
	oldStopper := orchestrationWorkerStopper
	var launches, stops int
	orchestrationWorkerLauncher = func(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		launches++
		return codexOrchestrationWorkerLaunch{
			RoleID: req.Role.ID, PID: 200, Status: "started",
			LogPath: orchestrationWorkerLogPath(req), StartedAt: time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC),
		}, nil
	}
	orchestrationWorkerUsageMonitor = func(req orchestrationWorkerLaunchRequest, launched codexOrchestrationWorkerLaunch, window time.Duration, workload *orchestrationWorkloadReceipt) (trajectory.QwenEmptyUsageAssessment, error) {
		return trajectory.AssessQwenEmptyUsage(trajectory.QwenEmptyUsageInput{
			WorkloadKind: workload.Kind, TargetModelFamily: workload.TargetModelFamily,
			WorkerKind: workload.WorkerKind, UsageExpectation: workload.UsageExpectation,
			WorkerModel: req.Model, LaunchStatus: launched.Status, PID: launched.PID,
			StartedAt: launched.StartedAt, ObservedAt: launched.StartedAt.Add(time.Second),
			Window: window, ProcessAlive: true,
			Usage: trajectory.CodexExecUsage{
				LogReadable: true, TurnsStarted: 1, TurnsCompleted: 1,
				InputTokens: 12, OutputTokens: 3, ProviderTokens: 15, UsageCovered: true,
			},
		}), nil
	}
	orchestrationWorkerStopper = func(int) error {
		stops++
		return nil
	}
	t.Cleanup(func() {
		orchestrationWorkerLauncher = oldLauncher
		orchestrationWorkerUsageMonitor = oldMonitor
		orchestrationWorkerStopper = oldStopper
	})

	receipt, err := launchCodexOrchestrationWorkers(home, "session-qwen-healthy", "ultracode", "native", "run the configured performance workload", qwenEmptyUsageResolution())
	if err != nil {
		t.Fatal(err)
	}
	if launches != 1 || stops != 0 || receipt.Status != "launched" || len(receipt.Workers) != 1 {
		t.Fatalf("launches=%d stops=%d receipt=%+v", launches, stops, receipt)
	}
	worker := receipt.Workers[0]
	if worker.Terminal != nil || worker.RecoveryAttempts != 0 || worker.Usage == nil ||
		worker.Usage.State != trajectory.QwenUsageStateHealthy {
		t.Fatalf("healthy worker = %+v", worker)
	}
}

func TestOrchestrationLaunchDoesNotInferQwenFromTaskProse(t *testing.T) {
	home := externalOrchestrationTestHome(t)
	oldLauncher := orchestrationWorkerLauncher
	oldMonitor := orchestrationWorkerUsageMonitor
	orchestrationWorkerLauncher = func(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: 300, Status: "started", StartedAt: time.Now(), LogPath: orchestrationWorkerLogPath(req)}, nil
	}
	orchestrationWorkerUsageMonitor = func(orchestrationWorkerLaunchRequest, codexOrchestrationWorkerLaunch, time.Duration, *orchestrationWorkloadReceipt) (trajectory.QwenEmptyUsageAssessment, error) {
		t.Fatal("task prose activated Qwen usage monitoring")
		return trajectory.QwenEmptyUsageAssessment{}, nil
	}
	t.Cleanup(func() {
		orchestrationWorkerLauncher = oldLauncher
		orchestrationWorkerUsageMonitor = oldMonitor
	})
	receipt, err := launchCodexOrchestrationWorkers(home, "session-prose-only", "ultracode", "native", "Qwen Qwen Qwen", qwenEmptyUsageResolution())
	if err != nil || receipt.EmptyUsagePolicy != nil || receipt.Workload != nil {
		t.Fatalf("receipt=%+v err=%v", receipt, err)
	}
}

func TestOrchestrationLaunchRejectsMoreThanOneQwenRecovery(t *testing.T) {
	home := externalOrchestrationTestHome(t)
	configureQwenUsageTest(t, "2")
	oldLauncher := orchestrationWorkerLauncher
	orchestrationWorkerLauncher = func(orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		t.Fatal("invalid recovery policy launched a worker")
		return codexOrchestrationWorkerLaunch{}, nil
	}
	t.Cleanup(func() { orchestrationWorkerLauncher = oldLauncher })
	receipt, err := launchCodexOrchestrationWorkers(home, "session-qwen-invalid-recovery", "ultracode", "native", "run configured workload", qwenEmptyUsageResolution())
	if err == nil || !strings.Contains(err.Error(), qwenEmptyUsageRecoveryAttemptsEnv) || receipt.Status != "declined" {
		t.Fatalf("receipt=%+v err=%v", receipt, err)
	}
}

func configureQwenUsageTest(t *testing.T, recoveryAttempts string) {
	t.Helper()
	t.Setenv(orchestrationWorkloadKindEnv, trajectory.QwenWorkloadKindModelPerformance)
	t.Setenv(orchestrationTargetModelFamilyEnv, trajectory.QwenTargetModelFamily)
	t.Setenv(orchestrationUsageExpectationEnv, trajectory.QwenUsageExpectationProvider)
	t.Setenv(qwenEmptyUsageWindowEnv, "1m")
	t.Setenv(qwenEmptyUsageRecoveryAttemptsEnv, recoveryAttempts)
}

func qwenEmptyUsageResolution() orchestration.Resolution {
	return orchestration.Resolution{Resolved: orchestration.WorkflowPlan{
		Profile: orchestration.ProfileUltracode, TaskID: "task-qwen-empty-usage",
		WorkClass: orchestration.WorkGrind,
		Roles: []orchestration.Role{
			{ID: "lead", TaskID: "task-qwen-empty-usage", Access: orchestration.ChildAccess{Mode: orchestration.ChildAccessObserve}},
			{ID: "worker-1", TaskID: "task-qwen-empty-usage", Access: orchestration.ChildAccess{Mode: orchestration.ChildAccessObserve}},
		},
		Budget:   orchestration.Budget{MaxWorkers: 2, MaxTokens: 4096},
		SOLRoute: orchestration.SOLRoute{Model: "gpt-5.6-sol", Mode: orchestration.SOLUltra, ReasoningEffort: "high"},
	}}
}
