package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func TestOrchestrationLaunchLowersFormalPacketAndBindsAssignmentDigest(t *testing.T) {
	home := externalOrchestrationTestHome(t)
	t.Setenv("CODEX_THREAD_ID", "session-formal-packet")
	fixture := filepath.Join(t.TempDir(), "formal-task.json")
	body := `{
		"schema":"fak-orchestration-task/1",
		"id":"formal-launch",
		"work_class":"rigor",
		"formal_packet":{
			"schema":"fak-formal-packet/1",
			"task_kinds":["state_machine_audit"],
			"definitions_and_assumptions":"For finite states S, transition relation R is total.",
			"exact_proposition":"Prove every reachable state has exactly one canonical successor.",
			"required_output_form":"DEFINITIONS, PROPOSITION, PROOF, COUNTEREXAMPLES, WITNESS",
			"deterministic_witness":"go test ./internal/orchestration -run TestFormalPacket",
			"surfaces":["internal/orchestration/**"]
		}
	}`
	if err := os.WriteFile(fixture, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}

	old := orchestrationWorkerLauncher
	var requests []orchestrationWorkerLaunchRequest
	orchestrationWorkerLauncher = func(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		requests = append(requests, req)
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: 700 + len(requests), Status: "started", LogPath: filepath.Join(req.RunDir, req.Role.ID+".jsonl")}, nil
	}
	t.Cleanup(func() { orchestrationWorkerLauncher = old })

	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--profile", "auto", "--task", fixture, "--codex-home", home, "--launch", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var result struct {
		Plan   orchestration.Resolution        `json:"plan"`
		Launch codexOrchestrationLaunchReceipt `json:"launch"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("decode launch result: %v\n%s", err, stdout.String())
	}
	if result.Plan.Resolved.SOLRoute.WorkerModel != orchestration.AstraWorkerModel || result.Plan.Resolved.SOLRoute.WorkerReasoningEffort != orchestration.AstraWorkerEffort {
		t.Fatalf("formal worker route = %+v", result.Plan.Resolved.SOLRoute)
	}
	if len(requests) == 0 || len(result.Launch.Workers) != len(requests) {
		t.Fatalf("requests=%d workers=%d", len(requests), len(result.Launch.Workers))
	}

	wantFragments := []string{
		"schema: fak-formal-packet/1",
		"task_kinds:\n- state_machine_audit",
		"definitions_and_assumptions:\nFor finite states S, transition relation R is total.",
		"exact_proposition:\nProve every reachable state has exactly one canonical successor.",
		"required_output_form:\nDEFINITIONS, PROPOSITION, PROOF, COUNTEREXAMPLES, WITNESS",
		"deterministic_witness:\ngo test ./internal/orchestration -run TestFormalPacket",
		"surfaces:\n- internal/orchestration/**",
	}
	assignment := requests[0].TaskText
	digest := sha256.Sum256([]byte(assignment))
	wantDigest := "sha256:" + hex.EncodeToString(digest[:])
	if wantDigest == "sha256:" || result.Launch.AssignmentDigest != wantDigest {
		t.Fatalf("launch assignment digest = %q, want %q", result.Launch.AssignmentDigest, wantDigest)
	}
	for _, req := range requests {
		if req.TaskText != assignment || req.AssignmentDigest != wantDigest || req.Model != orchestration.AstraWorkerModel || req.Effort != orchestration.AstraWorkerEffort {
			t.Fatalf("formal request mismatch: %+v", req)
		}
		if req.Access.Mode != orchestration.ChildAccessObserve || !req.Access.Admission.ReadOnly {
			t.Fatalf("formal request gained write access: %+v", req.Access)
		}
		prompt := orchestrationWorkerPrompt(req)
		if !strings.Contains(prompt, "Work read-only") || !strings.Contains(prompt, assignment) {
			t.Fatalf("formal prompt lost read-only assignment: %q", prompt)
		}
		for _, fragment := range wantFragments {
			if !strings.Contains(prompt, fragment) {
				t.Errorf("formal prompt missing %q: %q", fragment, prompt)
			}
		}
	}
	for _, worker := range result.Launch.Workers {
		if worker.AssignmentDigest != wantDigest || worker.Model != orchestration.AstraWorkerModel || worker.Effort != orchestration.AstraWorkerEffort || !worker.ReadOnly {
			t.Fatalf("formal worker receipt mismatch: %+v", worker)
		}
	}
	persisted, ok := readCodexOrchestrationLaunchReceipt(home, "session-formal-packet")
	if !ok || persisted.AssignmentDigest != wantDigest || !reflect.DeepEqual(persisted.Workers, result.Launch.Workers) {
		t.Fatalf("persisted launch receipt mismatch: ok=%v receipt=%+v", ok, persisted)
	}
}

func TestOrchestrationCLIWorkerOverridesReplaceRouteProvenance(t *testing.T) {
	fixture := filepath.Join(t.TempDir(), "formal-task.json")
	body := `{
		"schema":"fak-orchestration-task/1",
		"id":"formal-overrides",
		"work_class":"rigor",
		"formal_packet":{
			"schema":"fak-formal-packet/1",
			"task_kinds":["formal_proof"],
			"definitions_and_assumptions":"Natural numbers use ordinary addition.",
			"exact_proposition":"Prove zero is the additive identity.",
			"required_output_form":"Definitions, proposition, proof, witness.",
			"deterministic_witness":"go test ./internal/orchestration -run TestFormalPacket",
			"surfaces":["internal/orchestration/**"]
		}
	}`
	if err := os.WriteFile(fixture, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{
		"plan", "--profile", "auto", "--task", fixture, "--json", "--selfcheck",
		"--worker-model", "gpt-5.6-sol", "--worker-effort", "high",
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got orchestration.Resolution
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"sol_route.worker_model":            "gpt-5.6-sol",
		"sol_route.worker_reasoning_effort": "high",
	}
	counts := map[string]int{}
	for _, override := range got.Overrides {
		value, tracked := want[override.Field]
		if !tracked {
			continue
		}
		counts[override.Field]++
		if override.Source != orchestration.AstraRouteSourceOperatorPin || override.Value != value {
			t.Fatalf("override = %+v, want source operator-pin value %q", override, value)
		}
	}
	for field := range want {
		if counts[field] != 1 {
			t.Fatalf("override field %q count=%d, want exactly one; overrides=%+v", field, counts[field], got.Overrides)
		}
	}

	stdout.Reset()
	stderr.Reset()
	code = runOrchestration(&stdout, &stderr, []string{
		"plan", "--profile", "off", "--task", fixture, "--json", "--selfcheck",
		"--worker-model", "astra",
	})
	if code != 0 {
		t.Fatalf("profile-off code=%d stderr=%s", code, stderr.String())
	}
	got = orchestration.Resolution{}
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	route := got.Resolved.AstraRoute
	if route == nil || !route.Eligible || route.Selected || route.Model != "astra" || route.Source != orchestration.AstraRouteSourceOperatorPin {
		t.Fatalf("profile-off Astra receipt = %+v", route)
	}
	if got.Resolved.Profile != orchestration.ProfileOff || got.Resolved.Budget.MaxWorkers > 1 || got.Resolved.SOLRoute.WorkerModel != "astra" {
		t.Fatalf("profile-off effective route = %+v", got.Resolved)
	}
	modelOverrides := 0
	for _, override := range got.Overrides {
		if override.Field != "sol_route.worker_model" {
			continue
		}
		modelOverrides++
		if override.Source != orchestration.AstraRouteSourceOperatorPin || override.Value != "astra" {
			t.Fatalf("profile-off override = %+v", override)
		}
	}
	if modelOverrides != 1 {
		t.Fatalf("profile-off model override count=%d, want one; overrides=%+v", modelOverrides, got.Overrides)
	}
}

func TestOrchestrationWorkerControlPrecedenceIsResolvedBeforeLaunch(t *testing.T) {
	fixture := writeFormalWorkerControlFixture(t, "")
	pinnedFixture := writeFormalWorkerControlFixture(t, `"pins":{"model":"gpt-5.6-sol","effort":"high"},`)
	home := externalOrchestrationTestHome(t)
	t.Setenv("CODEX_THREAD_ID", "session-worker-control-precedence")

	old := orchestrationWorkerLauncher
	var requests []orchestrationWorkerLaunchRequest
	orchestrationWorkerLauncher = func(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		requests = append(requests, req)
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: 900 + len(requests), Status: "started", LogPath: filepath.Join(req.RunDir, req.Role.ID+".jsonl")}, nil
	}
	t.Cleanup(func() { orchestrationWorkerLauncher = old })

	t.Setenv("FAK_ORCHESTRATION_WORKER_MODEL", "astra")
	t.Setenv("FAK_ORCHESTRATION_WORKER_EFFORT", "low")
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--profile", "auto", "--task", fixture, "--codex-home", home, "--launch", "--json"})
	if code != 0 {
		t.Fatalf("environment launch code=%d stderr=%s", code, stderr.String())
	}
	var launched struct {
		Plan   orchestration.Resolution        `json:"plan"`
		Launch codexOrchestrationLaunchReceipt `json:"launch"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &launched); err != nil {
		t.Fatal(err)
	}
	assertWorkerControlRoute(t, launched.Plan, "astra", "low", orchestrationRouteSourceEnvironment)
	if len(requests) == 0 || len(launched.Launch.Workers) != len(requests) {
		t.Fatalf("environment launch requests=%d workers=%d", len(requests), len(launched.Launch.Workers))
	}
	for i, req := range requests {
		if req.Model != "astra" || req.Effort != "low" || launched.Launch.Workers[i].Model != req.Model || launched.Launch.Workers[i].Effort != req.Effort {
			t.Fatalf("plan/launch environment route diverged: request=%+v worker=%+v", req, launched.Launch.Workers[i])
		}
	}

	stdout.Reset()
	stderr.Reset()
	t.Setenv("FAK_ORCHESTRATION_WORKER_EFFORT", "invalid-task-shadowed")
	code = runOrchestration(&stdout, &stderr, []string{"plan", "--profile", "auto", "--task", pinnedFixture, "--json", "--selfcheck"})
	if code != 0 {
		t.Fatalf("task-pin code=%d stderr=%s", code, stderr.String())
	}
	var taskPinned orchestration.Resolution
	if err := json.Unmarshal(stdout.Bytes(), &taskPinned); err != nil {
		t.Fatal(err)
	}
	assertWorkerControlRoute(t, taskPinned, "gpt-5.6-sol", "high", orchestration.AstraRouteSourceTaskPin)

	t.Setenv("FAK_ORCHESTRATION_WORKER_MODEL", "gpt-5.6-sol")
	t.Setenv("FAK_ORCHESTRATION_WORKER_EFFORT", "invalid-cli-shadowed")
	stdout.Reset()
	stderr.Reset()
	code = runOrchestration(&stdout, &stderr, []string{
		"plan", "--profile", "auto", "--task", fixture, "--json", "--selfcheck",
		"--worker-model", "astra", "--worker-effort", "xhigh",
	})
	if code != 0 {
		t.Fatalf("CLI-pin code=%d stderr=%s", code, stderr.String())
	}
	var cliPinned orchestration.Resolution
	if err := json.Unmarshal(stdout.Bytes(), &cliPinned); err != nil {
		t.Fatal(err)
	}
	assertWorkerControlRoute(t, cliPinned, "astra", "xhigh", orchestration.AstraRouteSourceOperatorPin)

	t.Setenv("FAK_ORCHESTRATION_WORKER_EFFORT", "ultra")
	beforeInvalid := len(requests)
	stdout.Reset()
	stderr.Reset()
	code = runOrchestration(&stdout, &stderr, []string{"plan", "--profile", "auto", "--task", fixture, "--codex-home", home, "--launch", "--json"})
	if code != 2 || !strings.Contains(stderr.String(), "invalid FAK_ORCHESTRATION_WORKER_EFFORT") {
		t.Fatalf("invalid environment effort code=%d stderr=%s", code, stderr.String())
	}
	if len(requests) != beforeInvalid {
		t.Fatalf("invalid environment effort launched %d worker(s)", len(requests)-beforeInvalid)
	}

	t.Setenv("FAK_ORCHESTRATION_WORKER_MODEL", "")
	t.Setenv("FAK_ORCHESTRATION_WORKER_EFFORT", "   ")
	beforeBlankCLI := len(requests)
	stdout.Reset()
	stderr.Reset()
	code = runOrchestration(&stdout, &stderr, []string{
		"plan", "--profile", "auto", "--task", fixture, "--codex-home", home, "--launch", "--json",
		"--worker-effort", "   ",
	})
	if code != 2 || !strings.Contains(stderr.String(), "invalid --worker-effort") {
		t.Fatalf("blank CLI effort code=%d stderr=%s", code, stderr.String())
	}
	if len(requests) != beforeBlankCLI {
		t.Fatalf("blank CLI effort launched %d worker(s)", len(requests)-beforeBlankCLI)
	}
}

func writeFormalWorkerControlFixture(t *testing.T, pins string) string {
	t.Helper()
	fixture := filepath.Join(t.TempDir(), "formal-worker-control.json")
	body := `{
		"schema":"fak-orchestration-task/1",
		"id":"formal-worker-control",
		"work_class":"rigor",
		` + pins + `
		"formal_packet":{
			"schema":"fak-formal-packet/1",
			"task_kinds":["formal_proof"],
			"definitions_and_assumptions":"Natural numbers use ordinary addition.",
			"exact_proposition":"Prove zero is the additive identity.",
			"required_output_form":"Definitions, proposition, proof, witness.",
			"deterministic_witness":"go test ./internal/orchestration -run TestFormalPacket",
			"surfaces":["internal/orchestration/**"]
		}
	}`
	if err := os.WriteFile(fixture, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return fixture
}

func assertWorkerControlRoute(t *testing.T, resolution orchestration.Resolution, model, effort, source string) {
	t.Helper()
	if resolution.Resolved.SOLRoute.WorkerModel != model || resolution.Resolved.SOLRoute.WorkerReasoningEffort != effort {
		t.Fatalf("effective worker route=%+v, want %s/%s", resolution.Resolved.SOLRoute, model, effort)
	}
	route := resolution.Resolved.AstraRoute
	wantSelected := resolution.Resolved.Profile == orchestration.ProfileUltracode &&
		resolution.Resolved.Budget.MaxWorkers > 1 && orchestration.IsAstraModel(model)
	if route == nil || !route.Eligible || route.Model != model || route.ReasoningEffort != effort || route.Source != source || route.ReasoningEffortSource != source || route.Selected != wantSelected {
		t.Fatalf("Astra route=%+v, want model=%s effort=%s source=%s selected=%v", route, model, effort, source, wantSelected)
	}
	counts := map[string]int{}
	for _, override := range resolution.Overrides {
		if override.Field != "sol_route.worker_model" && override.Field != "sol_route.worker_reasoning_effort" {
			continue
		}
		counts[override.Field]++
		if override.Source != source {
			t.Fatalf("worker override=%+v, want source=%s", override, source)
		}
	}
	for _, field := range []string{"sol_route.worker_model", "sol_route.worker_reasoning_effort"} {
		if counts[field] != 1 {
			t.Fatalf("worker override %q count=%d, want one; overrides=%+v", field, counts[field], resolution.Overrides)
		}
	}
}

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

func TestCodexOrchestrationWorkerUsesManagedWorktree(t *testing.T) {
	root := t.TempDir()
	runDir := t.TempDir()
	worktreePath := filepath.Join(t.TempDir(), "fak-worker-wt-cmd-managed")
	baseSHA := strings.Repeat("a", 40)
	home := externalOrchestrationTestHome(t)

	oldPrepare := orchestrationWorkerWorktreePreparer
	oldBuildDirs := orchestrationWorkerWorktreeBuildDirs
	oldHandoff := orchestrationWorkerWorktreeHandoff
	oldCleanup := orchestrationWorkerWorktreePreOwnerCleanup
	oldStart := orchestrationWorkerProcessStarter
	oldProbe := orchestrationWorkerLaunchProbe
	t.Cleanup(func() {
		orchestrationWorkerWorktreePreparer = oldPrepare
		orchestrationWorkerWorktreeBuildDirs = oldBuildDirs
		orchestrationWorkerWorktreeHandoff = oldHandoff
		orchestrationWorkerWorktreePreOwnerCleanup = oldCleanup
		orchestrationWorkerProcessStarter = oldStart
		orchestrationWorkerLaunchProbe = oldProbe
	})

	var prepareCalls int
	var handoffCalls int
	var handoffPath string
	var handoffPID int
	var commandDirs []string
	var commandEnvs []map[string]string
	var events []string
	orchestrationWorkerWorktreePreparer = func(req orchestrationWorkerLaunchRequest) workerworktree.Result {
		prepareCalls++
		events = append(events, "prepare")
		if req.Access.Mode != orchestration.ChildAccessEffect || req.Access.Admission.Lane != "cmd" {
			t.Fatalf("prepare request = %+v", req)
		}
		return workerworktree.Result{OK: true, Path: worktreePath, BaseSHA: baseSHA}
	}
	orchestrationWorkerWorktreeBuildDirs = func(path string) (map[string]string, error) {
		if path != worktreePath {
			t.Fatalf("build dirs path=%q, want %q", path, worktreePath)
		}
		events = append(events, "ensure")
		return workerworktree.WorktreeEnv(nil, path), nil
	}
	orchestrationWorkerWorktreeHandoff = func(path string, pid int) error {
		handoffCalls++
		events = append(events, "handoff")
		handoffPath, handoffPID = path, pid
		return nil
	}
	orchestrationWorkerProcessStarter = func(cmd *exec.Cmd) (orchestrationStartedWorkerProcess, error) {
		events = append(events, "start")
		commandDirs = append(commandDirs, cmd.Dir)
		commandEnvs = append(commandEnvs, envMap(cmd.Env))
		return orchestrationStartedWorkerProcess{PID: 4242 + len(commandDirs), Release: func() error { return nil }}, nil
	}
	orchestrationWorkerLaunchProbe = func(int) bool { return true }

	req := orchestrationWorkerLaunchRequest{
		Role: orchestration.Role{
			ID: "effect-worker", Purpose: "implement",
			Access: orchestration.ChildAccess{Mode: orchestration.ChildAccessEffect, Lane: "cmd", WriteTree: "cmd/fak/**"},
		},
		Access: orchestrationCompiledChildAccess{
			Mode:      orchestration.ChildAccessEffect,
			Admission: laneadmit.Request{Lane: "cmd", Tree: []string{"cmd/fak/**"}},
		},
		Root: root, RunDir: runDir, RunID: "managed-worktree-test", Attempt: 1,
		Model: "test-model", Effort: "low", RemainingWall: time.Minute,
	}
	req.RecordStarted = func(started codexOrchestrationWorkerLaunch) error {
		events = append(events, "record")
		return persistCodexOrchestrationLaunchReceipt(home, codexOrchestrationLaunchReceipt{
			Schema: codexOrchestrationLaunchSchema, SessionID: "managed-worktree-test", RunID: req.RunID,
			Workers: []codexOrchestrationWorkerLaunch{started},
		})
	}

	launched, err := launchGuardedCodexOrchestrationWorker(req)
	if err != nil {
		t.Fatal(err)
	}
	if prepareCalls != 1 || launched.WorktreePath != worktreePath || launched.WorktreeBaseSHA != baseSHA {
		t.Fatalf("effect launch = %+v, prepare calls=%d", launched, prepareCalls)
	}
	if handoffCalls != 1 || handoffPath != worktreePath || handoffPID != launched.PID {
		t.Fatalf("handoff path=%q pid=%d, launch=%+v", handoffPath, handoffPID, launched)
	}
	if got := strings.Join(events, ","); got != "prepare,ensure,start,handoff,record" {
		t.Fatalf("launch event order = %q", got)
	}
	if len(commandDirs) != 1 || commandDirs[0] != worktreePath {
		t.Fatalf("effect command dirs = %v", commandDirs)
	}
	for key, want := range workerworktree.WorktreeEnv(nil, worktreePath) {
		if got := commandEnvs[0][key]; got != want {
			t.Errorf("effect env %s=%q, want %q", key, got, want)
		}
	}
	for key, want := range map[string]string{
		orchestrationWorktreeRootEnv:    root,
		orchestrationWorktreePathEnv:    worktreePath,
		orchestrationWorktreeBaseSHAEnv: baseSHA,
		orchestrationWorktreeTreeEnv:    `["cmd/fak/**"]`,
	} {
		if got := commandEnvs[0][key]; got != want {
			t.Errorf("lifecycle env %s=%q, want %q", key, got, want)
		}
	}
	if launched.WorktreeReceipt == "" || commandEnvs[0][orchestrationWorktreeLifecycleReceiptEnv] != launched.WorktreeReceipt {
		t.Fatalf("lifecycle receipt env=%q launch=%q", commandEnvs[0][orchestrationWorktreeLifecycleReceiptEnv], launched.WorktreeReceipt)
	}
	stripped := guardStripOrchestrationWorktreeLifecycleEnv(envSliceFromMap(commandEnvs[0]))
	for key := range envMap(stripped) {
		if strings.HasPrefix(strings.ToUpper(key), orchestrationWorktreeLifecycleEnvPrefix) {
			t.Fatalf("inner child inherited lifecycle metadata %q", key)
		}
	}
	persisted, ok := readCodexOrchestrationLaunchReceipt(home, "managed-worktree-test")
	if !ok || len(persisted.Workers) != 1 || persisted.Workers[0].WorktreePath != worktreePath || persisted.Workers[0].WorktreeBaseSHA != baseSHA || persisted.Workers[0].WorktreeReceipt != launched.WorktreeReceipt {
		t.Fatalf("persisted managed worker receipt = %+v, ok=%v", persisted, ok)
	}

	observe := req
	observe.Role.ID = "observe-worker"
	observe.Role.Access = orchestration.ChildAccess{Mode: orchestration.ChildAccessObserve}
	observe.Access = orchestrationCompiledChildAccess{Mode: orchestration.ChildAccessObserve, Admission: laneadmit.Request{ReadOnly: true}}
	observe.RecordStarted = nil
	if _, err := launchGuardedCodexOrchestrationWorker(observe); err != nil {
		t.Fatal(err)
	}
	if prepareCalls != 1 {
		t.Fatalf("observe worker prepared a managed worktree; calls=%d", prepareCalls)
	}
	if len(commandDirs) != 2 || commandDirs[1] != root {
		t.Fatalf("observe command dirs = %v", commandDirs)
	}
	if handoffCalls != 1 || handoffPath != worktreePath || handoffPID != launched.PID {
		t.Fatalf("observe worker invoked owner handoff: path=%q pid=%d", handoffPath, handoffPID)
	}

	orchestrationWorkerWorktreePreparer = func(orchestrationWorkerLaunchRequest) workerworktree.Result {
		return workerworktree.Result{Code: "PREPARE_TIMEOUT", Reason: "bounded preparation timed out"}
	}
	failed := req
	failed.Role.ID = "prepare-failure"
	if _, err := launchGuardedCodexOrchestrationWorker(failed); err == nil || !strings.Contains(err.Error(), "WORKTREE_PREPARE_FAILED") {
		t.Fatalf("effect prepare failure error = %v", err)
	}
	if len(commandDirs) != 2 {
		t.Fatalf("effect prepare failure started a process; command dirs=%v", commandDirs)
	}

	orchestrationWorkerWorktreePreparer = oldPrepare
	orchestrationWorkerWorktreeBuildDirs = func(string) (map[string]string, error) {
		return nil, errors.New("cache directory unavailable")
	}
	var cleanupCalls int
	orchestrationWorkerWorktreePreOwnerCleanup = func(gotRoot, gotPath string) workerworktree.Result {
		cleanupCalls++
		if gotRoot != root || gotPath != worktreePath {
			t.Fatalf("cleanup root=%q path=%q", gotRoot, gotPath)
		}
		return workerworktree.Result{Code: workerworktree.ReapCodeDirtyWorktreeRefused, Preserved: true}
	}
	buildFailed := req
	buildFailed.Role.ID = "build-dirs-failure"
	buildFailed.WorktreePath = worktreePath
	buildFailed.WorktreeBaseSHA = baseSHA
	failedLaunch, err := launchGuardedCodexOrchestrationWorker(buildFailed)
	if err == nil || !strings.Contains(err.Error(), "WORKTREE_BUILD_DIRS_FAILED") {
		t.Fatalf("effect build-dir failure error = %v", err)
	}
	if cleanupCalls != 1 || failedLaunch.WorktreePath != worktreePath || failedLaunch.WorktreeBaseSHA != baseSHA {
		t.Fatalf("dirty pre-owner cleanup calls=%d launch=%+v", cleanupCalls, failedLaunch)
	}
	if len(commandDirs) != 2 {
		t.Fatalf("effect build-dir failure started a process; command dirs=%v", commandDirs)
	}
	orchestrationWorkerWorktreePreOwnerCleanup = func(string, string) workerworktree.Result {
		cleanupCalls++
		return workerworktree.Result{OK: true, Removed: true}
	}
	buildFailed.Role.ID = "build-dirs-clean-failure"
	cleanedLaunch, err := launchGuardedCodexOrchestrationWorker(buildFailed)
	if err == nil || cleanedLaunch.WorktreePath != "" || cleanedLaunch.WorktreeBaseSHA != "" || cleanupCalls != 2 {
		t.Fatalf("clean pre-owner cleanup calls=%d launch=%+v err=%v", cleanupCalls, cleanedLaunch, err)
	}
}

func TestGuardOrchestrationWorktreeLifecycle(t *testing.T) {
	root := t.TempDir()
	worktree := filepath.Join(t.TempDir(), "fak-worker-wt-cmd-lifecycle")
	receiptPath := filepath.Join(t.TempDir(), "lifecycle.json")
	baseSHA := strings.Repeat("b", 40)
	t.Setenv(orchestrationWorktreeRootEnv, root)
	t.Setenv(orchestrationWorktreePathEnv, worktree)
	t.Setenv(orchestrationWorktreeBaseSHAEnv, baseSHA)
	t.Setenv(orchestrationWorktreeTreeEnv, `["cmd/fak/**"]`)
	t.Setenv(orchestrationWorktreeLifecycleReceiptEnv, receiptPath)

	oldLand, oldOwner, oldReap := guardOrchestrationWorktreeLand, guardOrchestrationWorktreeOwnerProcessLive, guardOrchestrationWorktreeReap
	t.Cleanup(func() {
		guardOrchestrationWorktreeLand = oldLand
		guardOrchestrationWorktreeOwnerProcessLive = oldOwner
		guardOrchestrationWorktreeReap = oldReap
	})
	var events []string
	guardOrchestrationWorktreeOwnerProcessLive = func(_ string, alive workerworktree.ProcessLiveFn) (bool, bool) {
		return alive(os.Getpid()), true
	}
	guardOrchestrationWorktreeLand = func(meta guardOrchestrationWorktreeMetadata) workerworktree.Result {
		events = append(events, "land")
		if meta.Root != root || meta.Path != worktree || meta.BaseSHA != baseSHA || !reflect.DeepEqual(meta.Trees, []string{"cmd/fak/**"}) {
			t.Fatalf("land metadata=%+v", meta)
		}
		return workerworktree.Result{OK: true}
	}
	guardOrchestrationWorktreeReap = func(gotRoot, gotPath string) workerworktree.Result {
		events = append(events, "reap")
		if gotRoot != root || gotPath != worktree {
			t.Fatalf("reap root=%q path=%q", gotRoot, gotPath)
		}
		return workerworktree.Result{OK: true}
	}
	readReceipt := func() guardOrchestrationWorktreeLifecycleReceipt {
		raw, err := os.ReadFile(receiptPath)
		if err != nil {
			t.Fatal(err)
		}
		var receipt guardOrchestrationWorktreeLifecycleReceipt
		if err := json.Unmarshal(raw, &receipt); err != nil {
			t.Fatal(err)
		}
		return receipt
	}

	if err := guardFinalizeOrchestrationWorktree(nil, true, nil); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(events, ","); got != "land,reap" {
		t.Fatalf("success lifecycle order=%q", got)
	}
	if receipt := readReceipt(); receipt.Status != "landed_reaped" || receipt.Worktree != worktree {
		t.Fatalf("success receipt=%+v", receipt)
	}

	events = nil
	guardOrchestrationWorktreeLand = func(guardOrchestrationWorktreeMetadata) workerworktree.Result {
		events = append(events, "land")
		return workerworktree.Result{Reason: "merge conflict"}
	}
	if err := guardFinalizeOrchestrationWorktree(nil, true, nil); err == nil || !strings.Contains(err.Error(), "ORCHESTRATION_WORKTREE_LAND_FAILED") {
		t.Fatalf("land failure=%v", err)
	}
	if got := strings.Join(events, ","); got != "land" {
		t.Fatalf("land failure lifecycle order=%q", got)
	}
	if receipt := readReceipt(); receipt.Status != "preserved" || !strings.Contains(receipt.Reason, "ORCHESTRATION_WORKTREE_LAND_FAILED") {
		t.Fatalf("land failure receipt=%+v", receipt)
	}

	events = nil
	guardOrchestrationWorktreeLand = func(guardOrchestrationWorktreeMetadata) workerworktree.Result {
		events = append(events, "land")
		return workerworktree.Result{OK: true, DroppedOutOfLane: 2}
	}
	if err := guardFinalizeOrchestrationWorktree(nil, true, nil); err == nil || !strings.Contains(err.Error(), "ORCHESTRATION_WORKTREE_POLICY_VIOLATION") {
		t.Fatalf("out-of-lane land failure=%v", err)
	}
	if got := strings.Join(events, ","); got != "land" {
		t.Fatalf("out-of-lane lifecycle order=%q", got)
	}
	if receipt := readReceipt(); receipt.Status != "landed_preserved" || receipt.Code != "ORCHESTRATION_WORKTREE_POLICY_VIOLATION" || receipt.Land.DroppedOutOfLane != 2 || receipt.Worktree != worktree {
		t.Fatalf("out-of-lane receipt=%+v", receipt)
	}

	events = nil
	guardOrchestrationWorktreeLand = func(guardOrchestrationWorktreeMetadata) workerworktree.Result {
		events = append(events, "land")
		return workerworktree.Result{OK: true}
	}
	guardOrchestrationWorktreeReap = func(string, string) workerworktree.Result {
		events = append(events, "reap")
		return workerworktree.Result{Code: workerworktree.ReapCodeDirtyWorktreeRefused, Preserved: true, Path: worktree}
	}
	if err := guardFinalizeOrchestrationWorktree(nil, true, nil); err == nil || !strings.Contains(err.Error(), "ORCHESTRATION_WORKTREE_REAP_FAILED") {
		t.Fatalf("checked reap refusal=%v", err)
	}
	if got := strings.Join(events, ","); got != "land,reap" {
		t.Fatalf("checked reap lifecycle order=%q", got)
	}
	if receipt := readReceipt(); receipt.Status != "landed_preserved" || receipt.Reap.Code != workerworktree.ReapCodeDirtyWorktreeRefused || receipt.Worktree != worktree {
		t.Fatalf("checked reap refusal receipt=%+v", receipt)
	}

	events = nil
	ownerRefusalReceipt := filepath.Join(t.TempDir(), "owner-refusal.json")
	t.Setenv(orchestrationWorktreeLifecycleReceiptEnv, ownerRefusalReceipt)
	guardOrchestrationWorktreeOwnerProcessLive = func(string, workerworktree.ProcessLiveFn) (bool, bool) {
		return false, false
	}
	if err := guardFinalizeOrchestrationWorktree(nil, true, nil); err == nil || !strings.Contains(err.Error(), "ORCHESTRATION_WORKTREE_OWNER_REFUSED") {
		t.Fatalf("owner refusal=%v", err)
	}
	if len(events) != 0 {
		t.Fatalf("owner refusal mutated worktree: %v", events)
	}
	if _, err := os.Stat(ownerRefusalReceipt); !os.IsNotExist(err) {
		t.Fatalf("owner refusal wrote untrusted receipt path: %v", err)
	}

	events = nil
	t.Setenv(orchestrationWorktreeLifecycleReceiptEnv, receiptPath)
	guardOrchestrationWorktreeOwnerProcessLive = func(_ string, alive workerworktree.ProcessLiveFn) (bool, bool) {
		return alive(os.Getpid()), true
	}
	if err := guardFinalizeOrchestrationWorktree(errors.New("codex crashed"), false, nil); err != nil {
		t.Fatal(err)
	}
	if len(events) != 0 {
		t.Fatalf("child failure mutated worktree: %v", events)
	}
	if receipt := readReceipt(); receipt.Status != "preserved" || !strings.Contains(receipt.Reason, "ORCHESTRATION_WORKTREE_CHILD_FAILED") {
		t.Fatalf("child failure receipt=%+v", receipt)
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
