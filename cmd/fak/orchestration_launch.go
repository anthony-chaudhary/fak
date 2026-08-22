package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
	"github.com/anthony-chaudhary/fak/internal/orchestration"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

const codexOrchestrationLaunchSchema = "fak.codex_orchestration_launch.v1"

func validateCodexOrchestrationArtifactHome(codexHome string) error {
	home, err := resolvedCodexLoopHome(codexHome)
	if err != nil {
		return err
	}
	absHome, err := filepath.Abs(home)
	if err != nil {
		return fmt.Errorf("resolve Codex home: %w", err)
	}
	probe := absHome
	for {
		if _, statErr := os.Stat(probe); statErr == nil {
			break
		} else if !os.IsNotExist(statErr) {
			return fmt.Errorf("inspect Codex home %q: %w", absHome, statErr)
		}
		parent := filepath.Dir(probe)
		if parent == probe {
			return nil
		}
		probe = parent
	}
	cmd := exec.Command("git", "-C", probe, "rev-parse", "--show-toplevel")
	configureDispatchHelperCommand(cmd)
	root, err := cmd.Output()
	if err != nil {
		return nil
	}
	worktree := strings.TrimSpace(string(root))
	if worktree == "" {
		return nil
	}
	return fmt.Errorf("unsafe Codex home %q is inside Git worktree %q; omit --codex-home to use $CODEX_HOME, or choose an external path allocated for scratch/runtime state", absHome, worktree)
}

const orchestrationChildEnv = "FAK_ORCHESTRATION_CHILD"

type codexOrchestrationWorkerLaunch struct {
	RoleID     string `json:"role_id"`
	PID        int    `json:"pid,omitempty"`
	Status     string `json:"status"`
	LogPath    string `json:"log_path,omitempty"`
	Model      string `json:"model"`
	Mode       string `json:"mode"`
	Effort     string `json:"reasoning_effort"`
	AccessMode string `json:"access_mode,omitempty"`
	ReadOnly   bool   `json:"read_only"`
	WriteTree  string `json:"write_tree,omitempty"`
	PolicyPath string `json:"policy_path,omitempty"`
	Refusal    string `json:"refusal,omitempty"`
}

type codexOrchestrationLaunchReceipt struct {
	Schema            string                           `json:"schema"`
	SessionID         string                           `json:"session_id"`
	RunID             string                           `json:"run_id"`
	LaunchedAt        time.Time                        `json:"launched_at"`
	TaskID            string                           `json:"task_id"`
	RequestedProfile  string                           `json:"requested_profile"`
	ResolvedProfile   string                           `json:"resolved_profile"`
	WorkClass         string                           `json:"work_class"`
	CapabilityProfile string                           `json:"capability_profile"`
	Degradations      []string                         `json:"degradations"`
	Status            string                           `json:"status"`
	DeclineReason     string                           `json:"decline_reason,omitempty"`
	Workers           []codexOrchestrationWorkerLaunch `json:"workers"`
}

func orchestrationDegradationNames(items []orchestration.Degradation) []string {
	out := make([]string, 0, len(items))
	for _, item := range items {
		out = append(out, item.Capability+":"+item.Reason)
	}
	return out
}

type orchestrationWorkerLaunchRequest struct {
	Role      orchestration.Role
	Access    orchestrationCompiledChildAccess
	WorkClass orchestration.WorkClass
	TaskText  string
	Root      string
	RunDir    string
	Model     string
	Mode      orchestration.SOLMode
	Effort    string
}

var orchestrationWorkerLauncher = launchGuardedCodexOrchestrationWorker

func launchCodexOrchestrationWorkers(home, sessionID, requestedProfile, capabilityProfile, taskText string, resolution orchestration.Resolution) (codexOrchestrationLaunchReceipt, error) {
	runID, err := newCodexOrchestrationRunID()
	if err != nil {
		return codexOrchestrationLaunchReceipt{}, err
	}
	receipt := codexOrchestrationLaunchReceipt{
		Schema: codexOrchestrationLaunchSchema, SessionID: sessionID, RunID: runID,
		LaunchedAt: time.Now().UTC(), TaskID: resolution.Resolved.TaskID,
		RequestedProfile: requestedProfile, ResolvedProfile: string(resolution.Resolved.Profile),
		WorkClass: string(resolution.Resolved.WorkClass), CapabilityProfile: capabilityProfile,
		Degradations: orchestrationDegradationNames(resolution.Degradations), Workers: []codexOrchestrationWorkerLaunch{},
	}
	if resolution.Resolved.Profile != orchestration.ProfileUltracode || resolution.Resolved.Budget.MaxWorkers <= 1 {
		receipt.Status = "declined"
		receipt.DeclineReason = "resolved-direct"
		return receipt, persistCodexOrchestrationLaunchReceipt(home, receipt)
	}
	runDir := filepath.Join(home, "fak-orchestration-runs", runID)
	if err := os.MkdirAll(runDir, 0o700); err != nil {
		return receipt, err
	}
	root, err := os.Getwd()
	if err != nil {
		return receipt, err
	}
	snapshot, err := orchestrationLaunchAccessSnapshot(root, resolution.Resolved.Roles)
	if err != nil {
		return receipt, fmt.Errorf("child access snapshot: %w", err)
	}
	live := append([]laneadmit.Lease(nil), snapshot.Live...)
	route := resolution.Resolved.SOLRoute
	if route.Model == "" {
		route = orchestration.SelectSOLRoute(taskText, resolution.Resolved.Profile, resolution.Resolved.WorkClass, guardCodexDefaultModelID)
	}
	if route.ConsultOnly {
		return receipt, fmt.Errorf("SOL_ROUTE_PRO_CONSULT_ONLY: Codex cannot transmit reasoning.mode=pro; launch a separately metered Pro consultation instead")
	}
	for _, role := range resolution.Resolved.Roles {
		if role.ID == "lead" {
			continue
		}
		access, compileErr := compileOrchestrationChildAccess(role, snapshot.Parent, laneadmit.Request{})
		if compileErr != nil {
			receipt.Workers = append(receipt.Workers, refusedOrchestrationWorker(role, access, compileErr))
			receipt.Status = "partial"
			_ = persistCodexOrchestrationLaunchReceipt(home, receipt)
			return receipt, compileErr
		}
		admission := laneadmit.Decide(access.Admission, live, snapshot.Taxonomy)
		if !admission.Admit {
			admitErr := fmt.Errorf("%s: child %q: %s", admission.Reason, role.ID, admission.Detail)
			receipt.Workers = append(receipt.Workers, refusedOrchestrationWorker(role, access, admitErr))
			receipt.Status = "partial"
			_ = persistCodexOrchestrationLaunchReceipt(home, receipt)
			return receipt, admitErr
		}
		access.PolicyPath, err = persistOrchestrationChildEnvelope(runDir, role.ID, access.ManifestJSON)
		if err != nil {
			return receipt, fmt.Errorf("persist %s child policy: %w", role.ID, err)
		}
		request := orchestrationWorkerLaunchRequest{Role: role, Access: access, WorkClass: resolution.Resolved.WorkClass, TaskText: taskText, Root: root, RunDir: runDir, Model: route.Model, Mode: route.Mode, Effort: route.ReasoningEffort}
		launched, launchErr := orchestrationWorkerLauncher(request)
		launched.Model = route.Model
		launched.Mode = string(route.Mode)
		launched.Effort = route.ReasoningEffort
		launched.AccessMode = string(access.Mode)
		launched.ReadOnly = access.Admission.ReadOnly
		launched.PolicyPath = access.PolicyPath
		if len(access.Admission.Tree) > 0 {
			launched.WriteTree = access.Admission.Tree[0]
		}
		if launchErr != nil {
			launched.RoleID = role.ID
			launched.Status = "failed"
			receipt.Workers = append(receipt.Workers, launched)
			receipt.Status = "partial"
			_ = persistCodexOrchestrationLaunchReceipt(home, receipt)
			return receipt, fmt.Errorf("launch %s: %w", role.ID, launchErr)
		}
		receipt.Workers = append(receipt.Workers, launched)
		if !access.Admission.ReadOnly {
			live = append(live, laneadmit.Lease{
				ID: "orchestration-child-" + role.ID, Lane: access.Admission.Lane,
				Tree: append([]string(nil), access.Admission.Tree...), Holder: role.ID,
			})
		}
	}
	receipt.Status = "launched"
	if err := persistCodexOrchestrationLaunchReceipt(home, receipt); err != nil {
		return receipt, err
	}
	return receipt, nil
}

func orchestrationWorkerArgs(req orchestrationWorkerLaunchRequest, auditPath string) []string {
	args := []string{
		"guard", "--codex-loop-gate", "off", "--provider", "openai-responses", "--audit", auditPath, "--expose-profile", "headless",
	}
	if req.Access.PolicyPath != "" {
		args = append(args, "--policy", req.Access.PolicyPath)
	}
	if !req.Access.Admission.ReadOnly && len(req.Access.Admission.Tree) > 0 {
		lease := "mode=enforce"
		if req.Access.Admission.Lane != "" {
			lease += ",lane=" + req.Access.Admission.Lane
		}
		for _, tree := range req.Access.Admission.Tree {
			lease += ",tree=" + tree
		}
		args = append(args, "--lease", lease)
	}
	return append(args,
		"--", "codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", "--dangerously-bypass-hook-trust", "--json",
		"-c", "model="+strconv.Quote(req.Model), "-c", "model_reasoning_effort="+strconv.Quote(req.Effort), "-",
	)
}

func launchGuardedCodexOrchestrationWorker(req orchestrationWorkerLaunchRequest) (codexOrchestrationWorkerLaunch, error) {
	fakBin, err := os.Executable()
	if err != nil {
		return codexOrchestrationWorkerLaunch{}, err
	}
	logPath := filepath.Join(req.RunDir, req.Role.ID+".jsonl")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_EXCL, 0o600)
	if err != nil {
		return codexOrchestrationWorkerLaunch{}, err
	}
	auditPath := filepath.Join(req.RunDir, req.Role.ID+"-guard.audit.jsonl")
	cmd := exec.Command(fakBin, orchestrationWorkerArgs(req, auditPath)...)
	cmd.Dir = req.Root
	cmd.Stdin = strings.NewReader(orchestrationWorkerPrompt(req))
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.Env = orchestrationWorkerEnv(os.Environ())
	configureDispatchSpawn(cmd)
	if err := cmd.Start(); err != nil {
		_ = logFile.Close()
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, LogPath: logPath}, err
	}
	_ = logFile.Close()
	if cmd.Process == nil {
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, LogPath: logPath}, fmt.Errorf("worker started without process")
	}
	pid := cmd.Process.Pid
	time.Sleep(3 * time.Second)
	if !dispatchPIDAlive(pid) {
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: pid, Status: "failed", LogPath: logPath}, fmt.Errorf("worker exited during launch probe; inspect %s", logPath)
	}
	if err := cmd.Process.Release(); err != nil {
		return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: pid, Status: "failed", LogPath: logPath}, fmt.Errorf("release worker process handle: %w", err)
	}
	return codexOrchestrationWorkerLaunch{RoleID: req.Role.ID, PID: pid, Status: "started", LogPath: logPath}, nil
}

func orchestrationWorkerPrompt(req orchestrationWorkerLaunchRequest) string {
	if req.Access.Mode == orchestration.ChildAccessEffect {
		return fmt.Sprintf("You are %s in a fak ultracode workflow. Work only through the compiled effect envelope: lane %s, write tree %s, tools %s. Do not widen the region, change policy, commit, push, or launch more workers. Return a concise evidence-linked report to the lead.\n\nTask:\n%s\n",
			req.Role.Purpose, req.Access.Admission.Lane, strings.Join(req.Access.Admission.Tree, ","), strings.Join(req.Role.Access.Tools, ","), strings.TrimSpace(req.TaskText))
	}
	return fmt.Sprintf("You are %s in a fak ultracode workflow. Work read-only: inspect evidence for the task, identify concrete implementation or verification findings, and return a concise evidence-linked report to the lead. Do not edit files, commit, push, or launch more workers.\n\nTask:\n%s\n", req.Role.Purpose, strings.TrimSpace(req.TaskText))
}

func orchestrationLaunchAccessSnapshot(root string, roles []orchestration.Role) (orchestrationChildAccessSnapshot, error) {
	for _, role := range roles {
		if strings.EqualFold(strings.TrimSpace(string(role.Access.Mode)), string(orchestration.ChildAccessEffect)) {
			return orchestrationChildAccessSnapshotLoader(root)
		}
	}
	parent, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	if err != nil {
		return orchestrationChildAccessSnapshot{}, err
	}
	return orchestrationChildAccessSnapshot{Parent: parent, Taxonomy: laneadmit.Taxonomy{Loaded: true, Exclusive: map[string]bool{}, Trees: map[string][]string{}}}, nil
}

func persistOrchestrationChildEnvelope(runDir, roleID string, raw []byte) (string, error) {
	path := filepath.Join(runDir, toolcallFileStem(roleID)+"-policy.json")
	if err := writeFileAtomic(path, raw, 0o600); err != nil {
		return "", err
	}
	return path, nil
}

func refusedOrchestrationWorker(role orchestration.Role, access orchestrationCompiledChildAccess, err error) codexOrchestrationWorkerLaunch {
	row := codexOrchestrationWorkerLaunch{RoleID: role.ID, Status: "refused", AccessMode: string(access.Mode), ReadOnly: access.Admission.ReadOnly}
	if len(access.Admission.Tree) > 0 {
		row.WriteTree = access.Admission.Tree[0]
	}
	if err != nil {
		row.Refusal = err.Error()
	}
	return row
}

func orchestrationWorkerEnv(env []string) []string {
	out := make([]string, 0, len(env)+1)
	for _, item := range env {
		key, _, _ := strings.Cut(item, "=")
		if strings.EqualFold(key, "CODEX_THREAD_ID") || strings.EqualFold(key, orchestrationChildEnv) || strings.HasPrefix(strings.ToUpper(key), "FAK_GUARD_") {
			continue
		}
		out = append(out, item)
	}
	return append(out, orchestrationChildEnv+"=1")
}

func newCodexOrchestrationRunID() (string, error) {
	var raw [12]byte
	if _, err := io.ReadFull(rand.Reader, raw[:]); err != nil {
		return "", err
	}
	return "orch-" + hex.EncodeToString(raw[:]), nil
}

func persistCodexOrchestrationLaunchReceipt(home string, receipt codexOrchestrationLaunchReceipt) error {
	dir := filepath.Join(home, "fak-orchestration-launches")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	raw, err := json.Marshal(receipt)
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	return writeFileAtomic(filepath.Join(dir, receipt.SessionID+".json"), raw, 0o600)
}

func readCodexOrchestrationLaunchReceipt(home, sessionID string) (codexOrchestrationLaunchReceipt, bool) {
	raw, err := os.ReadFile(filepath.Join(home, "fak-orchestration-launches", sessionID+".json"))
	if err != nil {
		return codexOrchestrationLaunchReceipt{}, false
	}
	var receipt codexOrchestrationLaunchReceipt
	if json.Unmarshal(raw, &receipt) != nil || receipt.Schema != codexOrchestrationLaunchSchema || receipt.SessionID != sessionID || receipt.RunID == "" {
		return codexOrchestrationLaunchReceipt{}, false
	}
	return receipt, true
}
