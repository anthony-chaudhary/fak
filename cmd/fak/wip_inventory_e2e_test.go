package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/wipinventory"
)

func TestWIPInventoryReconcileE2E(t *testing.T) {
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

	// 1. Initialize git repository with an initial commit
	runGit("init")
	runGit("config", "user.name", "Test Runner")
	runGit("config", "user.email", "test@example.com")
	runGit("commit", "--allow-empty", "-m", "initial repository commit")
	expectedHEAD := runGit("rev-parse", "HEAD")

	// 2. Add an explicit checkpoint reference
	runGit("update-ref", "refs/fak/wip/sess-10441-e2e", expectedHEAD)

	// 3. Create an untracked scratch file
	scratchFile := filepath.Join(dir, "scratch_sample.go")
	if err := os.WriteFile(scratchFile, []byte("// untracked scratch\npackage main\n"), 0o644); err != nil {
		t.Fatalf("failed to write scratch file: %v", err)
	}

	// Capture repository state before running CLI to prove read-only guarantee
	headBefore := runGit("rev-parse", "HEAD")
	refsBefore := runGit("for-each-ref")
	statusBefore := runGit("status", "--porcelain")

	// 4. Test runWIPInventory with --reconcile --json
	{
		var stdout, stderr bytes.Buffer
		exitCode := runWIPInventory([]string{"--reconcile", "--json", "--repo", dir}, &stdout, &stderr)
		if exitCode != 0 {
			t.Fatalf("runWIPInventory --reconcile --json failed with exit code %d; stderr:\n%s", exitCode, stderr.String())
		}

		var report wipinventory.ReconciliationReport
		if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
			t.Fatalf("failed to unmarshal JSON output: %v\nOutput:\n%s", err, stdout.String())
		}

		// Verify schema contract matches fak-wip-reconcile/1
		if report.Schema != wipinventory.ReconcileSchema {
			t.Errorf("Schema = %q, want %q", report.Schema, wipinventory.ReconcileSchema)
		}
		if report.HEAD != expectedHEAD {
			t.Errorf("HEAD = %q, want %q", report.HEAD, expectedHEAD)
		}
		cleanDir := filepath.ToSlash(filepath.Clean(dir))
		if report.Repo != cleanDir {
			t.Errorf("Repo = %q, want %q", report.Repo, cleanDir)
		}
		if report.ObservedAt.IsZero() {
			t.Errorf("ObservedAt should not be zero time")
		}

		// Verify raw surfaces
		if report.RawSurfaces.Checkpoints < 1 {
			t.Errorf("RawSurfaces.Checkpoints = %d, want at least 1", report.RawSurfaces.Checkpoints)
		}
		if report.RawSurfaces.UnlinkedFiles < 1 {
			t.Errorf("RawSurfaces.UnlinkedFiles = %d, want at least 1", report.RawSurfaces.UnlinkedFiles)
		}

		// Verify logical units conservation
		if report.LogicalUnits.Total != report.LogicalUnits.Active+report.LogicalUnits.Terminal {
			t.Errorf("Total != Active + Terminal: %d != %d + %d",
				report.LogicalUnits.Total, report.LogicalUnits.Active, report.LogicalUnits.Terminal)
		}
		if report.LogicalUnits.Total != report.LogicalUnits.SingleRepresentation+report.LogicalUnits.SplitRepresentations {
			t.Errorf("Total != Single + Split: %d != %d + %d",
				report.LogicalUnits.Total, report.LogicalUnits.SingleRepresentation, report.LogicalUnits.SplitRepresentations)
		}

		// Verify unresolved join debt has typed items with all fields populated
		if len(report.UnresolvedJoinDebt) == 0 {
			t.Errorf("expected unresolved join debt for orphan checkpoint/unlinked file, got none")
		}
		for _, debt := range report.UnresolvedJoinDebt {
			if debt.Reason == "" || debt.Surface == "" || debt.Details == "" {
				t.Errorf("incomplete debt item: %+v", debt)
			}
		}
	}

	// 5. Test top-level runWip dispatcher with "inventory", "--reconcile", "--json"
	{
		var stdout, stderr bytes.Buffer
		exitCode := runWip(&stdout, &stderr, []string{"inventory", "--reconcile", "--json", "--repo", dir})
		if exitCode != 0 {
			t.Fatalf("runWip inventory --reconcile --json failed with exit code %d; stderr:\n%s", exitCode, stderr.String())
		}

		var report wipinventory.ReconciliationReport
		if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
			t.Fatalf("failed to unmarshal JSON output from runWip: %v", err)
		}
		if report.Schema != wipinventory.ReconcileSchema {
			t.Errorf("runWip Schema = %q, want %q", report.Schema, wipinventory.ReconcileSchema)
		}
		if report.HEAD != expectedHEAD {
			t.Errorf("runWip HEAD = %q, want %q", report.HEAD, expectedHEAD)
		}
	}

	// 6. Test human-readable summary output via runWip
	{
		var stdout, stderr bytes.Buffer
		exitCode := runWip(&stdout, &stderr, []string{"inventory", "--reconcile", "--repo", dir})
		if exitCode != 0 {
			t.Fatalf("runWip inventory --reconcile failed with exit code %d; stderr:\n%s", exitCode, stderr.String())
		}

		outStr := stdout.String()
		if !strings.HasPrefix(strings.TrimSpace(outStr), "Active logical WIP:") {
			t.Errorf("human output must lead with 'Active logical WIP:', got:\n%s", outStr)
		}
		if !strings.Contains(outStr, "Raw surfaces:") {
			t.Errorf("human output missing 'Raw surfaces:'")
		}
		if !strings.Contains(outStr, "Transitions:") {
			t.Errorf("human output missing 'Transitions:'")
		}
		if !strings.Contains(outStr, "Unresolved join debt") {
			t.Errorf("human output missing 'Unresolved join debt'")
		}
	}

	// 7. Verify read-only execution: git HEAD, refs, and working status are unchanged
	headAfter := runGit("rev-parse", "HEAD")
	refsAfter := runGit("for-each-ref")
	statusAfter := runGit("status", "--porcelain")

	if headAfter != headBefore {
		t.Errorf("HEAD mutated: before %q, after %q", headBefore, headAfter)
	}
	if refsAfter != refsBefore {
		t.Errorf("refs mutated: before %q, after %q", refsBefore, refsAfter)
	}
	if statusAfter != statusBefore {
		t.Errorf("status mutated: before %q, after %q", statusBefore, statusAfter)
	}
}
