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
	log := "{\"type\":\"thread.started\"}\n{\"type\":\"turn.started\"}\n{\"type\":\"turn.completed\"}\n"
	workers := make([]codexOrchestrationWorkerLaunch, 0, 2)
	for _, roleID := range []string{"worker-1", "worker-2"} {
		logRel := filepath.Join("fak-orchestration-runs", runID, roleID+".jsonl")
		if err := os.WriteFile(filepath.Join(home, logRel), []byte(log), 0o600); err != nil {
			t.Fatal(err)
		}
		workers = append(workers, codexOrchestrationWorkerLaunch{RoleID: roleID, PID: 99999999, Status: "started", LogPath: logRel})
	}
	receipt := codexOrchestrationLaunchReceipt{Schema: codexOrchestrationLaunchSchema, SessionID: "sess-1", RunID: runID, LaunchedAt: time.Date(2026, 8, 18, 17, 0, 0, 0, time.UTC), RequestedProfile: "auto", ResolvedProfile: "ultracode", WorkClass: "grind", Status: "launched", Workers: workers}
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
	if got.Schema != orchestrationStatusSchema || got.State != "complete" || got.Completed != 2 || len(got.Workers) != 2 {
		t.Fatalf("unexpected status: %+v", got)
	}
	if got.Outcome.Verdict != "unverified" || got.Outcome.EffectReadback != orchestrationEvidenceNotObserved || got.Outcome.IndependentWitness != orchestrationEvidenceNotObserved || got.Outcome.Reconciliation != orchestrationEvidenceNotObserved {
		t.Fatalf("turn completion was mistaken for verified outcome: %+v", got.Outcome)
	}
	for _, worker := range got.Workers {
		if worker.TurnsStarted != 1 || worker.TurnsDone != 1 || worker.LastEvent != "turn.completed" {
			t.Fatalf("unexpected worker: %+v", worker)
		}
	}

	out.Reset()
	stderr.Reset()
	if code := runOrchestration(&out, &stderr, []string{"status", "--home", home}); code != 0 {
		t.Fatalf("human code=%d stderr=%s", code, stderr.String())
	}
	for _, want := range []string{
		"Orchestration orch-test - complete (worker execution)",
		"Outcome unverified | effects=not_observed | witness=not_observed | reconciliation=not_observed",
		"worker and turn events do not prove effects, independent witness acceptance, or reconciliation",
	} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("missing %q in:\n%s", want, out.String())
		}
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
	for _, want := range []string{"Orchestration orch-live - running (worker execution)", "Outcome unverified", "1 running", "worker-1", "turn.started", "turns 0/1", "machine: fak orchestration status --session sess-live --json"} {
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

func TestOrchestrationStatusRejectsConflictingSelection(t *testing.T) {
	var out, stderr bytes.Buffer
	code := runOrchestrationStatus(&out, &stderr, []string{"--session", "x", "unexpected"})
	if code != 2 || !strings.Contains(stderr.String(), "usage: fak orchestration status") {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
}

func TestOrchestrationStatusSkipsInvalidNewestReceipt(t *testing.T) {
	home := t.TempDir()
	valid := codexOrchestrationLaunchReceipt{Schema: codexOrchestrationLaunchSchema, SessionID: "valid", RunID: "orch-valid", LaunchedAt: time.Now()}
	if err := persistCodexOrchestrationLaunchReceipt(home, valid); err != nil {
		t.Fatal(err)
	}
	invalidPath := filepath.Join(home, "fak-orchestration-launches", "newest-invalid.json")
	if err := os.WriteFile(invalidPath, []byte("not-json\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	future := time.Now().Add(time.Hour)
	if err := os.Chtimes(invalidPath, future, future); err != nil {
		t.Fatal(err)
	}
	got, err := newestOrchestrationReceipt(home, "")
	if err != nil || got.SessionID != "valid" {
		t.Fatalf("got=%+v err=%v", got, err)
	}
}

func TestOrchestrationStatusClassifiesMissingLogByProcessLiveness(t *testing.T) {
	home := t.TempDir()
	receipt := codexOrchestrationLaunchReceipt{Schema: codexOrchestrationLaunchSchema, SessionID: "missing-log", RunID: "orch-missing", Workers: []codexOrchestrationWorkerLaunch{
		{RoleID: "live", PID: os.Getpid(), LogPath: "missing-live.jsonl"},
		{RoleID: "dead", PID: 99999999, LogPath: "missing-dead.jsonl"},
	}}
	got := inspectOrchestrationRun(home, receipt)
	if got.State != "running" || got.Running != 1 || got.Exited != 1 {
		t.Fatalf("unexpected aggregate: %+v", got)
	}
	if got.Workers[0].State != "running" || got.Workers[1].State != "exited" {
		t.Fatalf("unexpected workers: %+v", got.Workers)
	}
}

func TestOrchestrationStatusUsesExplicitSessionOverNewerReceipt(t *testing.T) {
	home := t.TempDir()
	older := codexOrchestrationLaunchReceipt{Schema: codexOrchestrationLaunchSchema, SessionID: "chosen", RunID: "orch-chosen", LaunchedAt: time.Now().Add(-time.Hour)}
	newer := codexOrchestrationLaunchReceipt{Schema: codexOrchestrationLaunchSchema, SessionID: "newest", RunID: "orch-newest", LaunchedAt: time.Now()}
	if err := persistCodexOrchestrationLaunchReceipt(home, older); err != nil {
		t.Fatal(err)
	}
	if err := persistCodexOrchestrationLaunchReceipt(home, newer); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	if code := runOrchestrationStatus(&out, &stderr, []string{"--home", home, "--session", "chosen", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got orchestrationRunStatus
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.SessionID != "chosen" || got.RunID != "orch-chosen" {
		t.Fatalf("explicit selection ignored: %+v", got)
	}
}
