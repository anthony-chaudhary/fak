package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/goalsync"
)

func setupTestWorkspace(t *testing.T) (string, string, string) {
	t.Helper()
	ws := t.TempDir()
	t.Setenv("FAK_WORKSPACE_ROOT", ws)

	if err := os.WriteFile(filepath.Join(ws, "go.mod"), []byte("module testws\n"), 0644); err != nil {
		t.Fatal(err)
	}

	goalsDir := filepath.Join(ws, "goals")
	subDir := filepath.Join(goalsDir, "subagents")
	fakDir := filepath.Join(ws, ".fak")
	parkDir := filepath.Join(fakDir, "goal-park")

	if err := os.MkdirAll(subDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(parkDir, 0755); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(goalsDir, "GOAL-root.md"), []byte("# Root Goal\nSpec\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(subDir, "GOAL-child.md"), []byte("# Child Goal\nSubagent\n"), 0644); err != nil {
		t.Fatal(err)
	}
	regPath := filepath.Join(fakDir, "goals.json")
	if err := os.WriteFile(regPath, []byte(`{"schema":"fak-goal-registry/1","goals":[]}`), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(parkDir, "park-a.json"), []byte(`{"schema":"fak.goal-park.v1","goal":"gA"}`), 0644); err != nil {
		t.Fatal(err)
	}

	return ws, regPath, parkDir
}

func TestGoalSyncStatus(t *testing.T) {
	_, regPath, parkDir := setupTestWorkspace(t)
	targetDir := t.TempDir()

	var stdout, stderr bytes.Buffer
	code := runGoal(&stdout, &stderr, []string{"sync", "status", "--target", targetDir, "--registry", regPath, "--goal-park", parkDir})
	if code != 0 {
		t.Fatalf("runGoal sync status code=%d, stderr=%s", code, stderr.String())
	}
	outStr := stdout.String()
	if !strings.Contains(outStr, "Goal Sync Status") || !strings.Contains(outStr, "Total: 4") {
		t.Fatalf("unexpected stdout: %s", outStr)
	}

	// Test --json flag
	stdout.Reset()
	stderr.Reset()
	code = runGoal(&stdout, &stderr, []string{"sync", "status", "--target", targetDir, "--registry", regPath, "--goal-park", parkDir, "--json"})
	if code != 0 {
		t.Fatalf("runGoal sync status --json code=%d, stderr=%s", code, stderr.String())
	}
	var st goalsync.SyncStatus
	if err := json.Unmarshal(stdout.Bytes(), &st); err != nil {
		t.Fatalf("json decode status: %v (raw: %s)", err, stdout.String())
	}
	if st.TotalCount != 4 || st.PushCount != 4 || st.InSyncCount != 0 {
		t.Fatalf("unexpected sync status JSON: %+v", st)
	}
}

func TestGoalSyncPush(t *testing.T) {
	_, regPath, parkDir := setupTestWorkspace(t)
	targetDir := t.TempDir()

	// 1. Explicit push command
	var stdout, stderr bytes.Buffer
	code := runGoal(&stdout, &stderr, []string{"sync", "push", "--target", targetDir, "--registry", regPath, "--goal-park", parkDir})
	if code != 0 {
		t.Fatalf("runGoal sync push code=%d, stderr=%s", code, stderr.String())
	}
	outStr := stdout.String()
	if !strings.Contains(outStr, "Goal Sync Push") || !strings.Contains(outStr, "Transferred: 4") {
		t.Fatalf("unexpected push stdout: %s", outStr)
	}

	// 2. Default action (omitting push sub-arg) with --json flag
	stdout.Reset()
	stderr.Reset()
	code = runGoal(&stdout, &stderr, []string{"sync", "--target", targetDir, "--registry", regPath, "--goal-park", parkDir, "--json"})
	if code != 0 {
		t.Fatalf("runGoal sync default push code=%d, stderr=%s", code, stderr.String())
	}
	var report goalsync.SyncReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("json decode push report: %v", err)
	}
	if report.Schema != goalsync.Schema {
		t.Fatalf("schema = %q, want %q", report.Schema, goalsync.Schema)
	}
	if len(report.Skipped) != 4 || len(report.Transferred) != 0 {
		t.Fatalf("expected 4 skipped on re-push: %+v", report)
	}
}

func TestGoalSyncPull(t *testing.T) {
	_, regPath, parkDir := setupTestWorkspace(t)
	targetDir := t.TempDir()

	// Push first so targetDir has all artifacts
	if code := runGoal(&bytes.Buffer{}, &bytes.Buffer{}, []string{"sync", "push", "--target", targetDir, "--registry", regPath, "--goal-park", parkDir}); code != 0 {
		t.Fatalf("initial push failed with code %d", code)
	}

	// Setup clean workspace for pull
	destWs := t.TempDir()
	t.Setenv("FAK_WORKSPACE_ROOT", destWs)
	if err := os.WriteFile(filepath.Join(destWs, "go.mod"), []byte("module destws\n"), 0644); err != nil {
		t.Fatal(err)
	}
	destReg := filepath.Join(destWs, ".fak", "goals.json")
	destPark := filepath.Join(destWs, ".fak", "goal-park")

	var stdout, stderr bytes.Buffer
	code := runGoal(&stdout, &stderr, []string{"sync", "pull", "--target", targetDir, "--registry", destReg, "--goal-park", destPark, "--json"})
	if code != 0 {
		t.Fatalf("runGoal sync pull code=%d, stderr=%s", code, stderr.String())
	}
	var report goalsync.SyncReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("json decode pull report: %v", err)
	}
	if len(report.Transferred) != 4 {
		t.Fatalf("expected 4 transferred on pull, got %d: %+v", len(report.Transferred), report)
	}

	// Verify file on disk in destWs
	restoredGoal := filepath.Join(destWs, "goals", "GOAL-root.md")
	data, err := os.ReadFile(restoredGoal)
	if err != nil {
		t.Fatalf("read restored goal: %v", err)
	}
	if string(data) != "# Root Goal\nSpec\n" {
		t.Fatalf("unexpected content: %q", string(data))
	}

	// Modify local file to be newer -> pull without force should fail
	futureTime := time.Now().Add(10 * time.Minute)
	if err := os.WriteFile(restoredGoal, []byte("newer local content\n"), 0644); err != nil {
		t.Fatal(err)
	}
	_ = os.Chtimes(restoredGoal, futureTime, futureTime)

	stdout.Reset()
	stderr.Reset()
	code = runGoal(&stdout, &stderr, []string{"sync", "pull", "--target", targetDir, "--registry", destReg, "--goal-park", destPark})
	if code != 1 {
		t.Fatalf("expected code 1 on conflict pull without force, got %d (stdout: %s, stderr: %s)", code, stdout.String(), stderr.String())
	}

	// Pull with --force should succeed
	stdout.Reset()
	stderr.Reset()
	code = runGoal(&stdout, &stderr, []string{"sync", "pull", "--target", targetDir, "--registry", destReg, "--goal-park", destPark, "--force"})
	if code != 0 {
		t.Fatalf("forced pull failed with code %d: stderr=%s", code, stderr.String())
	}
	overwritten, err := os.ReadFile(restoredGoal)
	if err != nil {
		t.Fatal(err)
	}
	if string(overwritten) != "# Root Goal\nSpec\n" {
		t.Fatalf("forced pull failed to overwrite: %q", string(overwritten))
	}
}

func TestGoalSyncDryRun(t *testing.T) {
	_, regPath, parkDir := setupTestWorkspace(t)
	targetDir := t.TempDir()

	var stdout, stderr bytes.Buffer
	code := runGoal(&stdout, &stderr, []string{"sync", "push", "--target", targetDir, "--registry", regPath, "--goal-park", parkDir, "--dry-run", "--json"})
	if code != 0 {
		t.Fatalf("dry run failed: %s", stderr.String())
	}
	var report goalsync.SyncReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if len(report.Transferred) != 4 {
		t.Fatalf("expected 4 reported transferred in dry run, got %d", len(report.Transferred))
	}

	// Verify no files written to targetDir
	entries, _ := os.ReadDir(targetDir)
	if len(entries) != 0 {
		t.Fatalf("dry-run wrote entries to targetDir: %d", len(entries))
	}
}

func TestGoalSyncInvalidAction(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runGoal(&stdout, &stderr, []string{"sync", "invalid-action"})
	if code != 2 {
		t.Fatalf("expected code 2 for invalid action, got %d", code)
	}
	if !strings.Contains(stderr.String(), "unknown action") {
		t.Fatalf("unexpected stderr: %s", stderr.String())
	}
}
