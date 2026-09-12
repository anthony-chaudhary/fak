package debtlane

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestShouldSkipDirHeavyRoots pins the heavy non-source root skip list: every
// operational root that must never be walked, plus a timestamped dgxbridge
// tail-readback dir, is skipped; real source roots are not.
func TestShouldSkipDirHeavyRoots(t *testing.T) {
	heavy := []string{
		".worktrees",
		"worktrees",
		"fleet-runs",
		"factory-cache",
		"session-checkpoints",
		"artifacts",
		"_wip-preservation",
		"campaigns",
		"outreach",
		"coverage",
		".dispatch-runs",
		".fak",
		".dos",
		".hypothesis",
		"agent-memory",
		"fleet",
		"dgxbridge-tail-readback-2026-06-23",
	}
	for _, name := range heavy {
		if !shouldSkipDir(name) {
			t.Errorf("shouldSkipDir(%q) = false, want true (heavy non-source root)", name)
		}
	}

	keep := []string{"internal", "pkg", "platform", "cmd", "tools"}
	for _, name := range keep {
		if shouldSkipDir(name) {
			t.Errorf("shouldSkipDir(%q) = true, want false (source root)", name)
		}
	}
}

// TestScanTerminatesWithHeavyWorktreeDirs is the hang regression: a .worktrees
// tree with hundreds of nested files must not stall expanded-breadth scanning,
// and package-root discovery must still find the platform package.
func TestScanTerminatesWithHeavyWorktreeDirs(t *testing.T) {
	tmp := t.TempDir()

	pkgDir := filepath.Join(tmp, "platform", "fixturepkg")
	if err := os.MkdirAll(pkgDir, 0o755); err != nil {
		t.Fatalf("mkdir fixture package: %v", err)
	}
	pkgSrc := "package fixturepkg\n\n// Fixture is a minimal valid package for discovery.\ntype Fixture struct{}\n"
	if err := os.WriteFile(filepath.Join(pkgDir, "fixture.go"), []byte(pkgSrc), 0o644); err != nil {
		t.Fatalf("write fixture package: %v", err)
	}

	heavyRoot := filepath.Join(tmp, ".worktrees", "fake-wt", "internal", "x")
	if err := os.MkdirAll(heavyRoot, 0o755); err != nil {
		t.Fatalf("mkdir heavy worktree root: %v", err)
	}
	for i := 0; i < 200; i++ {
		dir := filepath.Join(heavyRoot, "n"+strconv.Itoa(i/10))
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("mkdir heavy nested dir: %v", err)
		}
		content := "package x\n\n// Filler" + strconv.Itoa(i) + " is a nested heavy file.\ntype Filler" + strconv.Itoa(i) + " struct{}\n"
		if err := os.WriteFile(filepath.Join(dir, "f"+strconv.Itoa(i)+".go"), []byte(content), 0o644); err != nil {
			t.Fatalf("write heavy nested file: %v", err)
		}
	}

	type scanResult struct {
		report Report
		err    error
	}
	done := make(chan scanResult, 1)
	go func() {
		report, err := Scan(Options{WorkspaceRoot: tmp, ExpandedBreadth: true})
		done <- scanResult{report: report, err: err}
	}()

	select {
	case res := <-done:
		if res.err != nil {
			t.Fatalf("Scan returned error: %v", res.err)
		}
		found := false
		for _, lane := range res.report.Lanes {
			if strings.HasPrefix(filepath.ToSlash(lane.UnitOfWork), "platform/") {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected a platform lane (UnitOfWork prefixed \"platform/\") in report; got %d lanes", len(res.report.Lanes))
		}
	case <-time.After(30 * time.Second):
		t.Fatal("Scan did not terminate within 30s with heavy .worktrees present (hang regression)")
	}
}
