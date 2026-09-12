package debtlane

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestIsBuildOrToolingRootExcludesPhantomDebt pins the build/tooling root
// classifier: machine-generated cache/temp/agent-state dirs are non-source;
// real source roots and explicitly-handled standard surfaces are not.
func TestIsBuildOrToolingRootExcludesPhantomDebt(t *testing.T) {
	build := []string{
		".gocache",
		".gocache-9123",
		".gotmp-mtp-baseline",
		".worker-cache",
		".goal-runs",
		".claude-plugin",
		".codex",
		".continue",
	}
	for _, name := range build {
		if !isBuildOrToolingRoot(name) {
			t.Errorf("isBuildOrToolingRoot(%q) = false, want true (build/tooling root)", name)
		}
	}

	keep := []string{".claude", ".agents", ".github", "internal", "platform", "cmd"}
	for _, name := range keep {
		if isBuildOrToolingRoot(name) {
			t.Errorf("isBuildOrToolingRoot(%q) = true, want false (source surface)", name)
		}
	}
}

// TestScanExcludesBuildAndToolingRoots is the phantom-debt regression: a
// workspace with a real source package plus build/temp/data-only roots must
// yield no debt lane for the machine-generated roots, while a genuinely
// source-bearing unrecognized root still yields a lane.
func TestScanExcludesBuildAndToolingRoots(t *testing.T) {
	tmp := t.TempDir()

	pkgDir := filepath.Join(tmp, "platform", "fixturepkg")
	if err := os.MkdirAll(pkgDir, 0o755); err != nil {
		t.Fatalf("mkdir fixture package: %v", err)
	}
	pkgSrc := "package fixturepkg\n\n// Fixture is a minimal valid package for discovery.\ntype Fixture struct{}\n"
	if err := os.WriteFile(filepath.Join(pkgDir, "fixture.go"), []byte(pkgSrc), 0o644); err != nil {
		t.Fatalf("write fixture package: %v", err)
	}

	// Even a Go-bearing build cache is skipped by name: .gocache* is never a
	// source unit-of-work.
	gocacheDir := filepath.Join(tmp, ".gocache")
	if err := os.MkdirAll(gocacheDir, 0o755); err != nil {
		t.Fatalf("mkdir .gocache: %v", err)
	}
	if err := os.WriteFile(filepath.Join(gocacheDir, "x.go"), []byte(pkgSrc), 0o644); err != nil {
		t.Fatalf("write .gocache/x.go: %v", err)
	}

	tmpDir := filepath.Join(tmp, ".tmp")
	if err := os.MkdirAll(tmpDir, 0o755); err != nil {
		t.Fatalf("mkdir .tmp: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "scratch.txt"), []byte("scratch\n"), 0o644); err != nil {
		t.Fatalf("write .tmp/scratch.txt: %v", err)
	}

	dataDir := filepath.Join(tmp, "spamdata")
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		t.Fatalf("mkdir spamdata: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dataDir, "notes.txt"), []byte("data only\n"), 0o644); err != nil {
		t.Fatalf("write spamdata/notes.txt: %v", err)
	}

	// A source-bearing unrecognized root must still produce a debt lane.
	customDir := filepath.Join(tmp, "customsrc")
	if err := os.MkdirAll(customDir, 0o755); err != nil {
		t.Fatalf("mkdir customsrc: %v", err)
	}
	if err := os.WriteFile(filepath.Join(customDir, "main.go"), []byte(pkgSrc), 0o644); err != nil {
		t.Fatalf("write customsrc/main.go: %v", err)
	}

	report, err := Scan(Options{WorkspaceRoot: tmp, ExpandedBreadth: true})
	if err != nil {
		t.Fatalf("Scan returned error: %v", err)
	}

	forbidden := []string{".gocache", ".tmp", "spamdata"}
	for _, lane := range report.Lanes {
		unit := filepath.ToSlash(lane.UnitOfWork)
		for _, prefix := range forbidden {
			if unit == prefix || strings.HasPrefix(unit, prefix+"/") {
				t.Errorf("phantom debt lane for machine-generated root: %q (lane %q)", unit, lane.Lane)
			}
		}
	}

	found := false
	for _, lane := range report.Lanes {
		unit := filepath.ToSlash(lane.UnitOfWork)
		if unit == "customsrc" || strings.HasPrefix(unit, "customsrc/") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected a customsrc lane (source-bearing unrecognized root); got %d lanes", len(report.Lanes))
	}
}
