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
	streamDir := filepath.Join(root, ".dos", "streams")
	if err := os.MkdirAll(streamDir, 0o755); err != nil {
		t.Fatal(err)
	}
	claudeHome := filepath.Join(root, "home")
	claudeProject := filepath.Join(claudeHome, ".claude", "projects", "C--work-fak")
	if err := os.MkdirAll(claudeProject, 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 6, 26, 12, 0, 0, 0, time.UTC)
	writeStopFailureFixture(t, stopDir, "recent", `{"total":1,"consecutive":1}`, now.Add(-time.Hour))
	writeStopFailureFixture(t, stopDir, "stale", `{"total":3,"consecutive":2}`, now.Add(-8*time.Hour))
	writeStopFailureFixture(t, stopDir, "claudeonly", `{"total":2,"consecutive":1}`, now.Add(-7*time.Hour))
	writeStopFailureFixture(t, stopDir, "markeronly", `{"total":1,"consecutive":1}`, now.Add(-9*time.Hour))
	writeStopFailureFixture(t, stopDir, "healed", `{"total":2,"consecutive":0}`, now.Add(-9*time.Hour))
	writeStopFailureFixture(t, stopDir, "zero", `{"total":0,"consecutive":0}`, now.Add(-10*time.Hour))
	writeStopFailureFixture(t, stopDir, "old", `{"total":5,"consecutive":5}`, now.Add(-26*time.Hour))
	if err := os.WriteFile(filepath.Join(streamDir, "stale.jsonl"), []byte("{}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(claudeProject, "claudeonly.jsonl"), []byte("{}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runStopFailure(&stdout, &stderr, []string{
		"plan",
		"--root", root,
		"--claude-home", claudeHome,
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
			MarkerPath        string `json:"marker_path"`
			Consecutive       int    `json:"consecutive"`
			Origin            string `json:"origin"`
			TranscriptProject string `json:"transcript_project"`
			SettlementAction  string `json:"settlement_action"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
		t.Fatalf("plan JSON: %v\n%s", err, stdout.String())
	}
	if plan.Counts["RECENT_REVIEW"] != 1 || plan.Counts["STALE_RESET_CANDIDATE"] != 2 || plan.Counts["STALE_MARKER_ONLY_ARCHIVE_CANDIDATE"] != 1 || plan.Counts["HEALED_NONZERO"] != 1 || plan.Counts["ZERO_TOTAL"] != 1 {
		t.Fatalf("counts = %#v", plan.Counts)
	}
	if plan.IgnoredOld != 1 {
		t.Fatalf("ignored_old_markers = %d, want 1", plan.IgnoredOld)
	}
	staleRows := plan.Candidates["STALE_RESET_CANDIDATE"]
	if len(staleRows) != 2 || staleRows[0].MarkerPath != ".dos/stop-failures/stale.json" || staleRows[0].Consecutive != 2 || staleRows[0].Origin != "dos_stream" {
		t.Fatalf("stale rows = %#v", staleRows)
	}
	if staleRows[1].MarkerPath != ".dos/stop-failures/claudeonly.json" || staleRows[1].Origin != "claude_transcript" || staleRows[1].TranscriptProject != "C--work-fak" {
		t.Fatalf("claude stale row = %#v", staleRows[1])
	}
	archiveRows := plan.Candidates["STALE_MARKER_ONLY_ARCHIVE_CANDIDATE"]
	if len(archiveRows) != 1 || archiveRows[0].MarkerPath != ".dos/stop-failures/markeronly.json" {
		t.Fatalf("archive rows = %#v", archiveRows)
	}

	stdout.Reset()
	stderr.Reset()
	code = runStopFailure(&stdout, &stderr, []string{
		"reset-stale",
		"--root", root,
		"--claude-home", claudeHome,
		"--now", now.Format(time.RFC3339),
	})
	if code != 0 {
		t.Fatalf("dry-run code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "DRY-RUN candidates=2 updated=0") {
		t.Fatalf("dry-run output:\n%s", stdout.String())
	}
	assertStopFailureConsecutive(t, stopDir, "stale", 2)
	assertStopFailureConsecutive(t, stopDir, "claudeonly", 1)
	assertStopFailureConsecutive(t, stopDir, "markeronly", 1)
	assertStopFailureConsecutive(t, stopDir, "recent", 1)

	stdout.Reset()
	stderr.Reset()
	code = runStopFailure(&stdout, &stderr, []string{
		"reset-stale",
		"--root", root,
		"--claude-home", claudeHome,
		"--now", now.Format(time.RFC3339),
		"--apply",
		"--json",
	})
	if code != 0 {
		t.Fatalf("apply code=%d stderr=%s", code, stderr.String())
	}
	assertStopFailureConsecutive(t, stopDir, "stale", 0)
	assertStopFailureConsecutive(t, stopDir, "claudeonly", 0)
	assertStopFailureConsecutive(t, stopDir, "markeronly", 1)
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
