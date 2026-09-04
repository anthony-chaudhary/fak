package main

import (
	"bytes"
	"encoding/json"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/wipinventory"
)

func TestWIPInventoryReconcileCLI(t *testing.T) {
	dir := t.TempDir()

	runGit := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s failed: %v\nOutput: %s", strings.Join(args, " "), err, string(out))
		}
		return strings.TrimSpace(string(out))
	}

	runGit("init")
	runGit("config", "user.name", "Test Runner")
	runGit("config", "user.email", "test@example.com")
	runGit("commit", "--allow-empty", "-m", "init")
	expectedHEAD := runGit("rev-parse", "HEAD")

	// Snapshot git state before execution to verify read-only behavior
	refsBefore := runGit("for-each-ref")
	statusBefore := runGit("status", "--porcelain")

	// 1. Test human output via --reconcile flag
	{
		var stdout, stderr bytes.Buffer
		code := runWIPInventory([]string{"--reconcile", "--repo", dir}, &stdout, &stderr)
		if code != 0 {
			t.Fatalf("runWIPInventory --reconcile failed with code %d; stderr: %s", code, stderr.String())
		}
		outStr := stdout.String()
		lines := strings.Split(strings.TrimSpace(outStr), "\n")
		if len(lines) == 0 || !strings.HasPrefix(lines[0], "Active logical WIP:") {
			t.Errorf("expected output to lead with active logical WIP count, got:\n%s", outStr)
		}
		if !strings.Contains(outStr, "Raw surfaces:") {
			t.Errorf("expected output to contain 'Raw surfaces:', got:\n%s", outStr)
		}
		if !strings.Contains(outStr, "Transitions:") {
			t.Errorf("expected output to contain 'Transitions:', got:\n%s", outStr)
		}
		if !strings.Contains(outStr, "Unresolved join debt") {
			t.Errorf("expected output to contain 'Unresolved join debt', got:\n%s", outStr)
		}
	}

	// 2. Test JSON output via --reconcile --json
	{
		var stdout, stderr bytes.Buffer
		code := runWIPInventory([]string{"--reconcile", "--json", "--repo", dir}, &stdout, &stderr)
		if code != 0 {
			t.Fatalf("runWIPInventory --reconcile --json failed with code %d; stderr: %s", code, stderr.String())
		}
		var report wipinventory.ReconciliationReport
		if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
			t.Fatalf("failed to unmarshal JSON output: %v\nOutput was:\n%s", err, stdout.String())
		}
		if report.Schema != wipinventory.ReconcileSchema {
			t.Errorf("schema = %q, want %q", report.Schema, wipinventory.ReconcileSchema)
		}
		if report.HEAD != expectedHEAD {
			t.Errorf("HEAD = %q, want %q", report.HEAD, expectedHEAD)
		}
		cleanDir := filepath.ToSlash(filepath.Clean(dir))
		if report.Repo != cleanDir {
			t.Errorf("Repo = %q, want %q", report.Repo, cleanDir)
		}
	}

	// 3. Test subcommand reconcile
	{
		var stdout, stderr bytes.Buffer
		code := runWIPInventory([]string{"reconcile", "--json", "--root", dir}, &stdout, &stderr)
		if code != 0 {
			t.Fatalf("runWIPInventory reconcile --json failed with code %d; stderr: %s", code, stderr.String())
		}
		var report wipinventory.ReconciliationReport
		if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
			t.Fatalf("failed to unmarshal JSON output: %v", err)
		}
		if report.Schema != wipinventory.ReconcileSchema {
			t.Errorf("schema = %q, want %q", report.Schema, wipinventory.ReconcileSchema)
		}
	}

	// 4. Verify read-only execution: git state must not have changed
	refsAfter := runGit("for-each-ref")
	statusAfter := runGit("status", "--porcelain")
	headAfter := runGit("rev-parse", "HEAD")

	if headAfter != expectedHEAD {
		t.Errorf("HEAD changed: before %q, after %q", expectedHEAD, headAfter)
	}
	if refsAfter != refsBefore {
		t.Errorf("refs changed: before %q, after %q", refsBefore, refsAfter)
	}
	if statusAfter != statusBefore {
		t.Errorf("status changed: before %q, after %q", statusBefore, statusAfter)
	}
}
