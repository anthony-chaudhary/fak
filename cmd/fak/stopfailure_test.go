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

func TestStopFailurePlanAndResetStale(t *testing.T) {
	root := t.TempDir()
	stopDir := filepath.Join(root, ".dos", "stop-failures")
	if err := os.MkdirAll(stopDir, 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 6, 26, 12, 0, 0, 0, time.UTC)
	writeStopFailureFixture(t, stopDir, "recent", `{"total":1,"consecutive":1}`, now.Add(-time.Hour))
	writeStopFailureFixture(t, stopDir, "stale", `{"total":3,"consecutive":2}`, now.Add(-8*time.Hour))
	writeStopFailureFixture(t, stopDir, "healed", `{"total":2,"consecutive":0}`, now.Add(-9*time.Hour))
	writeStopFailureFixture(t, stopDir, "zero", `{"total":0,"consecutive":0}`, now.Add(-10*time.Hour))
	writeStopFailureFixture(t, stopDir, "old", `{"total":5,"consecutive":5}`, now.Add(-26*time.Hour))

	var stdout, stderr bytes.Buffer
	code := runStopFailure(&stdout, &stderr, []string{
		"plan",
		"--root", root,
		"--now", now.Format(time.RFC3339),
		"--json",
	})
	if code != 0 {
		t.Fatalf("plan code=%d stderr=%s", code, stderr.String())
	}
	var plan struct {
		Counts     map[string]int `json:"counts"`
		IgnoredOld int            `json:"ignored_old_markers"`
		Candidates map[string][]struct {
			MarkerPath       string `json:"marker_path"`
			Consecutive      int    `json:"consecutive"`
			SettlementAction string `json:"settlement_action"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("plan JSON: %v\n%s", err, stdout.String())
	}
	if plan.Counts["RECENT_REVIEW"] != 1 || plan.Counts["STALE_RESET_CANDIDATE"] != 1 || plan.Counts["HEALED_NONZERO"] != 1 || plan.Counts["ZERO_TOTAL"] != 1 {
		t.Fatalf("counts = %#v", plan.Counts)
	}
	if plan.IgnoredOld != 1 {
		t.Fatalf("ignored_old_markers = %d, want 1", plan.IgnoredOld)
	}
	staleRows := plan.Candidates["STALE_RESET_CANDIDATE"]
	if len(staleRows) != 1 || staleRows[0].MarkerPath != ".dos/stop-failures/stale.json" || staleRows[0].Consecutive != 2 {
		t.Fatalf("stale rows = %#v", staleRows)
	}

	stdout.Reset()
	stderr.Reset()
	code = runStopFailure(&stdout, &stderr, []string{
		"reset-stale",
		"--root", root,
		"--now", now.Format(time.RFC3339),
	})
	if code != 0 {
		t.Fatalf("dry-run code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "DRY-RUN candidates=1 updated=0") {
		t.Fatalf("dry-run output:\n%s", stdout.String())
	}
	assertStopFailureConsecutive(t, stopDir, "stale", 2)
	assertStopFailureConsecutive(t, stopDir, "recent", 1)

	stdout.Reset()
	stderr.Reset()
	code = runStopFailure(&stdout, &stderr, []string{
		"reset-stale",
		"--root", root,
		"--now", now.Format(time.RFC3339),
		"--apply",
		"--json",
	})
	if code != 0 {
		t.Fatalf("apply code=%d stderr=%s", code, stderr.String())
	}
	assertStopFailureConsecutive(t, stopDir, "stale", 0)
	assertStopFailureConsecutive(t, stopDir, "recent", 1)
}

func writeStopFailureFixture(t *testing.T, dir, session, body string, mtime time.Time) {
	t.Helper()
	path := filepath.Join(dir, session+".json")
	if err := os.WriteFile(path, []byte(body+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, mtime, mtime); err != nil {
		t.Fatal(err)
	}
}

func assertStopFailureConsecutive(t *testing.T, dir, session string, want int) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, session+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]int
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	if got := doc["consecutive"]; got != want {
		t.Fatalf("%s consecutive=%d, want %d", session, got, want)
	}
}
