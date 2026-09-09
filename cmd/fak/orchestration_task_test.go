package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func TestOrchestrationPlanAcceptsCurrentTaskText(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--task-text", "fan out independent checks in parallel", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	var got orchestration.Resolution
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Requested.Name != orchestration.ProfileAuto || got.Resolved.Profile != orchestration.ProfileUltracode || got.Resolved.TaskID == "" || got.Resolved.SOLRoute.Mode != orchestration.SOLUltra {
		t.Fatalf("resolved=%+v", got)
	}
	for _, role := range got.Resolved.Roles {
		if role.Access.Mode != orchestration.ChildAccessObserve || len(role.Access.WriteSet) != 0 {
			t.Fatalf("--task-text inferred worker authority: %+v", role)
		}
	}
	if strings.Contains(stdout.String(), "fan out independent checks") {
		t.Fatal("raw task text leaked into receipt")
	}
}

func TestOrchestrationPlanAcceptsTypedWorkerAccessFixture(t *testing.T) {
	fixture := t.TempDir() + "/task.json"
	body := `{
		"schema":"fak-orchestration-task/1",
		"id":"typed-access",
		"work_class":"grind",
		"max_workers":2,
		"worker_access":[{
			"role_id":"worker-1",
			"access":{
				"mode":"effect",
				"read_set":["internal/orchestration"],
				"write_set":["internal/orchestration"],
				"tools":["Read","Write"]
			}
		}]
	}`
	if err := os.WriteFile(fixture, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--profile", "ultracode", "--task", fixture, "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	var got orchestration.Resolution
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Resolved.Roles) != 2 || got.Resolved.Roles[0].Access.Mode != orchestration.ChildAccessObserve {
		t.Fatalf("roles=%+v", got.Resolved.Roles)
	}
	worker := got.Resolved.Roles[1]
	if worker.ID != "worker-1" || worker.Access.Mode != orchestration.ChildAccessEffect ||
		len(worker.Access.WriteSet) != 1 || worker.Access.WriteSet[0] != "internal/orchestration" {
		t.Fatalf("worker access=%+v", worker.Access)
	}
}

func TestOrchestrationPlanCombinesTypedWorkerAccessWithEphemeralTaskText(t *testing.T) {
	home := externalOrchestrationTaskTestHome(t)
	t.Setenv("CODEX_HOME", "  "+home+"  ")
	t.Setenv("CODEX_THREAD_ID", "session-typed-effect-assignment")
	fixture := t.TempDir() + "/task.json"
	body := `{
		"schema":"fak-orchestration-task/1",
		"id":"typed-effect-assignment",
		"work_class":"grind",
		"max_workers":2,
		"worker_access":[{
			"role_id":"worker-1",
			"access":{
				"mode":"effect",
				"read_set":["cmd/fak"],
				"write_set":["cmd/fak/orchestration_task_contract/**"],
				"tools":["Read","Write"]
			}
		}]
	}`
	if err := os.WriteFile(fixture, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	const assignment = "implement ephemeral assignment sentinel-9e6c without persisting this text"
	parentPolicy, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		t.Fatal(err)
	}
	oldSnapshot := orchestrationChildAccessSnapshotLoader
	orchestrationChildAccessSnapshotLoader = func(string) (orchestrationChildAccessSnapshot, error) {
		return orchestrationChildAccessSnapshot{
			Parent: parentPolicy,
			Taxonomy: laneadmit.Taxonomy{
				Loaded: true, Trees: map[string][]string{"cmd": {"cmd/**"}}, Exclusive: map[string]bool{},
			},
		}, nil
	}
	t.Cleanup(func() { orchestrationChildAccessSnapshotLoader = oldSnapshot })
	oldLauncher := orchestrationWorkerLauncher
	var requests []orchestrationWorkerLaunchRequest
	orchestrationWorkerLauncher = func(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		requests = append(requests, req)
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: 800 + len(requests), Status: "started"}, nil
	}
	t.Cleanup(func() { orchestrationWorkerLauncher = oldLauncher })

	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{
		"plan", "--profile", "ultracode", "--task", fixture,
		"--task-text", "  " + assignment + " \r\n", "--launch", "--json",
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	var got struct {
		Plan   orchestration.Resolution        `json:"plan"`
		Launch codexOrchestrationLaunchReceipt `json:"launch"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Plan.Resolved.TaskID != "typed-effect-assignment" {
		t.Fatalf("task id=%q, want fixture identity", got.Plan.Resolved.TaskID)
	}
	if len(got.Plan.Resolved.Roles) != 2 {
		t.Fatalf("roles=%+v", got.Plan.Resolved.Roles)
	}
	worker := got.Plan.Resolved.Roles[1]
	if worker.ID != "worker-1" || worker.Access.Mode != orchestration.ChildAccessEffect ||
		len(worker.Access.WriteSet) != 1 || worker.Access.WriteSet[0] != "cmd/fak/orchestration_task_contract/**" {
		t.Fatalf("worker access=%+v", worker.Access)
	}
	if len(requests) != 1 || requests[0].TaskText != assignment {
		t.Fatalf("launch requests=%+v, want one normalized ephemeral assignment", requests)
	}
	if !filepath.IsAbs(requests[0].RunDir) || !filepath.IsAbs(requests[0].Access.PolicyPath) ||
		!orchestrationTaskTestPathWithin(home, requests[0].RunDir) || !orchestrationTaskTestPathWithin(home, requests[0].Access.PolicyPath) {
		t.Fatalf("launch artifacts escaped effective CODEX_HOME: run_dir=%q policy=%q home=%q", requests[0].RunDir, requests[0].Access.PolicyPath, home)
	}
	if strings.Contains(stdout.String(), assignment) || strings.Contains(stdout.String(), "sentinel-9e6c") {
		t.Fatal("raw task text leaked into stable plan output")
	}
	assertOrchestrationReceiptOmitsText(t, home, "session-typed-effect-assignment", assignment)
}

func TestOrchestrationPlanComposesFormalPacketWithEphemeralTaskText(t *testing.T) {
	home := externalOrchestrationTaskTestHome(t)
	t.Setenv("CODEX_THREAD_ID", "session-formal-ephemeral-assignment")
	fixture := filepath.Join(t.TempDir(), "task.json")
	body := `{
		"schema":"fak-orchestration-task/1",
		"id":"formal-ephemeral-assignment",
		"work_class":"rigor",
		"max_workers":2,
		"formal_packet":{
			"schema":"fak-formal-packet/1",
			"task_kinds":["state_machine_audit"],
			"definitions_and_assumptions":"State set S is finite.",
			"exact_proposition":"Prove the transition is total.",
			"required_output_form":"DEFINITIONS, PROOF, WITNESS",
			"deterministic_witness":"go test ./internal/orchestration -run TestFormalPacket",
			"surfaces":["internal/orchestration/**"]
		}
	}`
	if err := os.WriteFile(fixture, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	const assignment = "check the extra edge case sentinel-formal-42"
	oldLauncher := orchestrationWorkerLauncher
	var requests []orchestrationWorkerLaunchRequest
	orchestrationWorkerLauncher = func(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
		requests = append(requests, req)
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: 900 + len(requests), Status: "started"}, nil
	}
	t.Cleanup(func() { orchestrationWorkerLauncher = oldLauncher })

	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{
		"plan", "--profile", "ultracode", "--task", fixture, "--task-text", assignment,
		"--codex-home", home, "--launch", "--json",
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	if len(requests) == 0 {
		t.Fatal("formal launch produced no worker requests")
	}
	for _, req := range requests {
		for _, want := range []string{
			"definitions_and_assumptions:\nState set S is finite.",
			"exact_proposition:\nProve the transition is total.",
			"required_output_form:\nDEFINITIONS, PROOF, WITNESS",
			"deterministic_witness:\ngo test ./internal/orchestration -run TestFormalPacket",
			"surfaces:\n- internal/orchestration/**",
			"ephemeral_assignment:\n" + assignment,
		} {
			if !strings.Contains(req.TaskText, want) {
				t.Errorf("launch assignment missing %q: %q", want, req.TaskText)
			}
		}
		if strings.Index(req.TaskText, "exact_proposition:") > strings.Index(req.TaskText, "ephemeral_assignment:") {
			t.Errorf("ephemeral assignment was not appended after formal packet: %q", req.TaskText)
		}
	}
	if strings.Contains(stdout.String(), assignment) || strings.Contains(stdout.String(), "sentinel-formal-42") {
		t.Fatal("raw ephemeral assignment leaked into stable launch result")
	}
	assertOrchestrationReceiptOmitsText(t, home, "session-formal-ephemeral-assignment", assignment)
}

func TestCodexOrchestrationWorkerProbeFailureCleansOrPreservesManagedWorktree(t *testing.T) {
	for _, tc := range []struct {
		name         string
		cleanup      workerworktree.Result
		wantPreserve bool
	}{
		{name: "clean worktree reaped", cleanup: workerworktree.Result{OK: true, Removed: true}},
		{name: "dirty worktree preserved", cleanup: workerworktree.Result{Code: workerworktree.ReapCodeDirtyWorktreeRefused, Preserved: true}, wantPreserve: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			runDir := t.TempDir()
			worktreePath := filepath.Join(t.TempDir(), "fak-worker-wt-cmd-probe-failure")
			baseSHA := strings.Repeat("c", 40)

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

			orchestrationWorkerWorktreePreparer = func(orchestrationWorkerLaunchRequest) workerworktree.Result {
				return workerworktree.Result{OK: true, Path: worktreePath, BaseSHA: baseSHA}
			}
			orchestrationWorkerWorktreeBuildDirs = func(string) (map[string]string, error) {
				return workerworktree.WorktreeEnv(nil, worktreePath), nil
			}
			var handoffPID, cleanupCalls, killAndWaitCalls int
			orchestrationWorkerWorktreeHandoff = func(path string, pid int) error {
				if path != worktreePath {
					t.Fatalf("handoff path=%q, want %q", path, worktreePath)
				}
				handoffPID = pid
				return nil
			}
			orchestrationWorkerProcessStarter = func(*exec.Cmd) (orchestrationStartedWorkerProcess, error) {
				return orchestrationStartedWorkerProcess{PID: 4242, KillAndWait: func() { killAndWaitCalls++ }}, nil
			}
			orchestrationWorkerLaunchProbe = func(pid int) bool {
				if pid != 4242 {
					t.Fatalf("probe pid=%d", pid)
				}
				return false
			}
			orchestrationWorkerWorktreePreOwnerCleanup = func(gotRoot, gotPath string) workerworktree.Result {
				cleanupCalls++
				if gotRoot != root || gotPath != worktreePath {
					t.Fatalf("cleanup root=%q path=%q", gotRoot, gotPath)
				}
				return tc.cleanup
			}

			req := orchestrationWorkerLaunchRequest{
				Role: orchestration.Role{
					ID: "effect-worker", Purpose: "implement",
					Access: orchestration.ChildAccess{Mode: orchestration.ChildAccessEffect, Lane: "cmd", WriteTree: "cmd/fak/**"},
				},
				Access: orchestrationCompiledChildAccess{
					Mode: orchestration.ChildAccessEffect, Admission: laneadmit.Request{Lane: "cmd", Tree: []string{"cmd/fak/**"}},
				},
				Root: root, RunDir: runDir, RunID: "probe-failure-test", Attempt: 1,
				Model: "test-model", Effort: "low", RemainingWall: time.Minute,
			}
			launch, err := launchGuardedCodexOrchestrationWorker(req)
			if err == nil || !strings.Contains(err.Error(), "worker exited during launch probe") {
				t.Fatalf("probe failure err=%v", err)
			}
			if handoffPID != 4242 || cleanupCalls != 1 || killAndWaitCalls != 1 {
				t.Fatalf("handoff_pid=%d cleanup_calls=%d kill_wait_calls=%d", handoffPID, cleanupCalls, killAndWaitCalls)
			}
			if tc.wantPreserve {
				if launch.WorktreePath != worktreePath || launch.WorktreeBaseSHA != baseSHA || launch.WorktreeReceipt == "" {
					t.Fatalf("dirty failed launch lost worktree identity: %+v", launch)
				}
			} else if launch.WorktreePath != "" || launch.WorktreeBaseSHA != "" || launch.WorktreeReceipt != "" {
				t.Fatalf("clean failed launch retained reaped worktree identity: %+v", launch)
			}
		})
	}
}

func externalOrchestrationTaskTestHome(t *testing.T) string {
	t.Helper()
	base, err := os.UserCacheDir()
	if err != nil {
		t.Fatal(err)
	}
	home, err := os.MkdirTemp(base, "fak-orchestration-task-test-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(home) })
	return home
}

func orchestrationTaskTestPathWithin(root, candidate string) bool {
	rel, err := filepath.Rel(filepath.Clean(root), filepath.Clean(candidate))
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(os.PathSeparator)) && !filepath.IsAbs(rel)
}

func assertOrchestrationReceiptOmitsText(t *testing.T, home, sessionID, text string) {
	t.Helper()
	paths := []string{
		filepath.Join(home, "fak-orchestration-launches", sessionID+".json"),
		filepath.Join(home, "fak-orchestration-invocations", sessionID+".json"),
	}
	for _, path := range paths {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read receipt %q: %v", path, err)
		}
		if strings.Contains(string(raw), text) {
			t.Fatalf("raw task text leaked into receipt %q", path)
		}
	}
}

func TestOrchestrationPlanTaskTextKeepsTinyWorkDirect(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runOrchestration(&stdout, &stderr, []string{"plan", "--task-text", "fix typo", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
	var got orchestration.Resolution
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Resolved.Profile != orchestration.ProfileOff || got.Resolved.SOLRoute.Mode != orchestration.SOLStandard {
		t.Fatalf("resolved=%+v, want off/standard", got.Resolved)
	}
}

func TestOrchestrationPlanRequiresTaskSource(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runOrchestration(&stdout, &stderr, []string{"plan"}); code != 2 {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
}

func TestOrchestrationPlanRoutesCompleteFormalPacketAndPreservesOperatorPins(t *testing.T) {
	fixture := t.TempDir() + "/formal-task.json"
	body := `{
		"schema":"fak-orchestration-task/1",
		"id":"formal-cli-spine",
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

	run := func(extra ...string) (orchestration.Resolution, string) {
		t.Helper()
		args := []string{"plan", "--profile", "auto", "--task", fixture, "--json", "--selfcheck"}
		args = append(args, extra...)
		var stdout, stderr bytes.Buffer
		if code := runOrchestration(&stdout, &stderr, args); code != 0 {
			t.Fatalf("args=%v code=%d stderr=%s", extra, code, stderr.String())
		}
		var got orchestration.Resolution
		if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
			t.Fatalf("decode: %v\n%s", err, stdout.String())
		}
		if !strings.Contains(stderr.String(), "SELFCHECK PASS") || !strings.Contains(stderr.String(), "launched=0") {
			t.Fatalf("selfcheck receipt missing: %s", stderr.String())
		}
		return got, stderr.String()
	}

	automatic, _ := run()
	route := automatic.Resolved.AstraRoute
	if route == nil || !route.Eligible || !route.Selected || route.Source != orchestration.AstraRouteSourceFormalPacket || route.Model != orchestration.AstraWorkerModel || route.ReasoningEffort != orchestration.AstraWorkerEffort || route.ReasoningEffortSource != orchestration.AstraRouteSourceFormalPacket {
		t.Fatalf("automatic Astra receipt = %+v", route)
	}
	if automatic.Resolved.SOLRoute.WorkerModel != orchestration.AstraWorkerModel || automatic.Resolved.SOLRoute.WorkerReasoningEffort != orchestration.AstraWorkerEffort {
		t.Fatalf("automatic worker route = %+v", automatic.Resolved.SOLRoute)
	}

	pinned, _ := run("--worker-model", "gpt-5.6-sol", "--worker-effort", "high")
	route = pinned.Resolved.AstraRoute
	if route == nil || !route.Eligible || route.Selected || route.Source != orchestration.AstraRouteSourceOperatorPin || route.Model != "gpt-5.6-sol" || route.ReasoningEffort != "high" || route.ReasoningEffortSource != orchestration.AstraRouteSourceOperatorPin {
		t.Fatalf("operator-pinned Astra receipt = %+v", route)
	}
	if pinned.Resolved.SOLRoute.WorkerModel != "gpt-5.6-sol" || pinned.Resolved.SOLRoute.WorkerReasoningEffort != "high" {
		t.Fatalf("operator pins lost from worker route: %+v", pinned.Resolved.SOLRoute)
	}
}
