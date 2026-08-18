package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestOrchestrationStatusJoinsReceiptProcessAndTurnLog(t *testing.T) {
	home := t.TempDir()
	runID := "orch-test"
	runDir := filepath.Join(home, "fak-orchestration-runs", runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	logRel := filepath.Join("fak-orchestration-runs", runID, "worker-1.jsonl")
	log := "{\"type\":\"thread.started\"}\n{\"type\":\"turn.started\"}\n{\"type\":\"turn.completed\"}\n"
	if err := os.WriteFile(filepath.Join(home, logRel), []byte(log), 0o600); err != nil {
		t.Fatal(err)
	}
	receipt := codexOrchestrationLaunchReceipt{Schema: codexOrchestrationLaunchSchema, SessionID: "sess-1", RunID: runID, LaunchedAt: time.Date(2026, 8, 18, 17, 0, 0, 0, time.UTC), RequestedProfile: "auto", ResolvedProfile: "ultracode", WorkClass: "grind", Status: "launched", Workers: []codexOrchestrationWorkerLaunch{{RoleID: "worker-1", PID: 99999999, Status: "started", LogPath: logRel}}}
	if err := persistCodexOrchestrationLaunchReceipt(home, receipt); err != nil {
		t.Fatal(err)
	}

	var out, stderr bytes.Buffer
	if code := runOrchestration(&out, &stderr, []string{"status", "--home", home, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got orchestrationRunStatus
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != orchestrationStatusSchema || got.State != "complete" || got.Completed != 1 || len(got.Workers) != 1 {
		t.Fatalf("unexpected status: %+v", got)
	}
	if got.Workers[0].TurnsStarted != 1 || got.Workers[0].TurnsDone != 1 || got.Workers[0].LastEvent != "turn.completed" {
		t.Fatalf("unexpected worker: %+v", got.Workers[0])
	}
}

func TestOrchestrationStatusHumanOutputNamesCurrentActivity(t *testing.T) {
	home := t.TempDir()
	runID := "orch-live"
	runDir := filepath.Join(home, "fak-orchestration-runs", runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	logRel := filepath.Join("fak-orchestration-runs", runID, "worker-1.jsonl")
	if err := os.WriteFile(filepath.Join(home, logRel), []byte("{\"type\":\"turn.started\"}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	receipt := codexOrchestrationLaunchReceipt{Schema: codexOrchestrationLaunchSchema, SessionID: "sess-live", RunID: runID, LaunchedAt: time.Now(), Workers: []codexOrchestrationWorkerLaunch{{RoleID: "worker-1", PID: os.Getpid(), Status: "started", LogPath: logRel}}}
	if err := persistCodexOrchestrationLaunchReceipt(home, receipt); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	if code := runOrchestration(&out, &stderr, []string{"status", "--home", home}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	for _, want := range []string{"Orchestration orch-live - running", "1 running", "worker-1", "turn.started", "turns 0/1", "machine: fak orchestration status --session sess-live --json"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("missing %q in:\n%s", want, out.String())
		}
	}
}

func TestInspectOrchestrationRunDoesNotHideANewerActiveTurn(t *testing.T) {
	home := t.TempDir()
	runID := "orch-multi-turn"
	runDir := filepath.Join(home, "fak-orchestration-runs", runID)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		t.Fatal(err)
	}
	logRel := filepath.Join("fak-orchestration-runs", runID, "worker-1.jsonl")
	log := "{\"type\":\"turn.started\"}\n{\"type\":\"turn.completed\"}\n{\"type\":\"turn.started\"}\n"
	if err := os.WriteFile(filepath.Join(home, logRel), []byte(log), 0o600); err != nil {
		t.Fatal(err)
	}
	receipt := codexOrchestrationLaunchReceipt{Schema: codexOrchestrationLaunchSchema, SessionID: "sess-multi", RunID: runID, LaunchedAt: time.Now(), Workers: []codexOrchestrationWorkerLaunch{{RoleID: "worker-1", PID: os.Getpid(), Status: "started", LogPath: logRel}}}
	got := inspectOrchestrationRun(home, receipt)
	if got.State != "running" || got.Running != 1 || got.Completed != 0 {
		t.Fatalf("newer active turn hidden by older completion: %+v", got)
	}
}
