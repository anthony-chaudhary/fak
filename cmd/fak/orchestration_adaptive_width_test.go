package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
)

func TestAdaptiveWidthDryRunReportsSelectionWithoutLaunch(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "internal", "orchestration", "testdata", "fast-width-scout.json"))
	if err != nil {
		t.Fatal(err)
	}
	var evidence orchestration.WidthEvidence
	if err := json.Unmarshal(raw, &evidence); err != nil {
		t.Fatal(err)
	}
	task := orchestration.TaskSpec{Schema: "fak-orchestration-task/1", ID: "fast-dry-run", WorkClass: orchestration.WorkGrind, Width: &orchestration.WidthRequest{ObjectiveMillis: 2000, AsOf: "2026-08-25T13:00:00Z", Key: evidence.Key, Evidence: &evidence}}
	taskRaw, _ := json.Marshal(task)
	path := filepath.Join(t.TempDir(), "task.json")
	if err := os.WriteFile(path, taskRaw, 0600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runOrchestration(&stdout, &stderr, []string{"plan", "--profile", "fast", "--task", path, "--json", "--selfcheck"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var got orchestration.Resolution
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Resolved.Width == nil || got.Resolved.Width.Selected != 4 || got.Resolved.Width.EvidenceDigest != evidence.Digest || got.Resolved.Width.Realized != 0 {
		t.Fatalf("width=%+v", got.Resolved.Width)
	}
	if got.Resolved.Budget.MaxWorkers != 4 || len(got.Resolved.Roles) != 4 {
		t.Fatalf("plan=%+v", got.Resolved)
	}
}

func TestAdaptiveWidthCLIExactPin(t *testing.T) {
	task := orchestration.TaskSpec{Schema: "fak-orchestration-task/1", ID: "fast-pin", WorkClass: orchestration.WorkGrind}
	raw, _ := json.Marshal(task)
	path := filepath.Join(t.TempDir(), "task.json")
	_ = os.WriteFile(path, raw, 0600)
	var stdout, stderr bytes.Buffer
	if code := runOrchestration(&stdout, &stderr, []string{"plan", "--profile", "fast", "--task", path, "--max-workers", "4", "--exact-workers", "3", "--json"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var got orchestration.Resolution
	_ = json.Unmarshal(stdout.Bytes(), &got)
	if got.Resolved.Budget.MaxWorkers != 3 || got.Resolved.Width.Reason != "explicit operator exact-width pin" {
		t.Fatalf("width=%+v", got.Resolved.Width)
	}
}
