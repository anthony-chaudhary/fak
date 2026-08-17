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
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
)

const codexOrchestrationLaunchSchema = "fak.codex_orchestration_launch.v1"

type codexOrchestrationWorkerLaunch struct {
	RoleID  string `json:"role_id"`
	PID     int    `json:"pid,omitempty"`
	Status  string `json:"status"`
	LogPath string `json:"log_path,omitempty"`
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
	Role     orchestration.Role
	TaskText string
	Root     string
	RunDir   string
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
	for _, role := range resolution.Resolved.Roles {
		if role.ID == "lead" {
			continue
		}
		launched, launchErr := orchestrationWorkerLauncher(orchestrationWorkerLaunchRequest{Role: role, TaskText: taskText, Root: root, RunDir: runDir})
		if launchErr != nil {
			launched.RoleID = role.ID
			launched.Status = "failed"
			receipt.Workers = append(receipt.Workers, launched)
			receipt.Status = "partial"
			_ = persistCodexOrchestrationLaunchReceipt(home, receipt)
			return receipt, fmt.Errorf("launch %s: %w", role.ID, launchErr)
		}
		receipt.Workers = append(receipt.Workers, launched)
	}
	receipt.Status = "launched"
	if err := persistCodexOrchestrationLaunchReceipt(home, receipt); err != nil {
		return receipt, err
	}
	return receipt, nil
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
	args := []string{"guard", "--provider", "openai-responses", "--audit", auditPath, "--expose-profile", "headless", "--", "codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", "--dangerously-bypass-hook-trust", "--json", "-"}
	cmd := exec.Command(fakBin, args...)
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
	return fmt.Sprintf("You are %s in a fak ultracode workflow. Work read-only: inspect evidence for the task, identify concrete implementation or verification findings, and return a concise evidence-linked report to the lead. Do not edit files, commit, push, or launch more workers.\n\nTask:\n%s\n", req.Role.Purpose, strings.TrimSpace(req.TaskText))
}

func orchestrationWorkerEnv(env []string) []string {
	out := make([]string, 0, len(env))
	for _, item := range env {
		key, _, _ := strings.Cut(item, "=")
		if strings.EqualFold(key, "CODEX_THREAD_ID") || strings.HasPrefix(strings.ToUpper(key), "FAK_GUARD_") {
			continue
		}
		out = append(out, item)
	}
	return out
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
