package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeSizedFile drops a file of n bytes and fails the test if it cannot.
func writeSizedFile(t *testing.T, path string, n int) {
	t.Helper()
	if err := os.WriteFile(path, make([]byte, n), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// TestCleanBinsSelectsIgnoredArtifactsOnly is the load-bearing gate: the core removes
// stray gitignored build binaries, protects the live fak binary, and NEVER touches a
// tracked file — even one whose name looks like a build target.
func TestCleanBinsSelectsIgnoredArtifactsOnly(t *testing.T) {
	root := t.TempDir()
	live := liveBinaryName() // fak.exe on Windows, fak elsewhere

	// A cmd/<name> layout so bare-name (Unix) artifacts are recognizable.
	if err := os.MkdirAll(filepath.Join(root, "cmd", "batchbench"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "cmd", "fak"), 0o755); err != nil {
		t.Fatal(err)
	}

	// Stray build binaries (ignored, removable).
	writeSizedFile(t, filepath.Join(root, "conceptbench.exe"), 100)
	writeSizedFile(t, filepath.Join(root, "fak-new.exe"), 200)
	writeSizedFile(t, filepath.Join(root, "fak.exe~"), 50)    // atomic-replace scratch
	writeSizedFile(t, filepath.Join(root, "batchbench"), 300) // bare Unix artifact matching cmd/batchbench
	writeSizedFile(t, filepath.Join(root, live), 1000)        // the live binary — protected by default

	// Non-artifacts that must be left alone regardless of gitignore state.
	writeSizedFile(t, filepath.Join(root, "README.md"), 10)
	writeSizedFile(t, filepath.Join(root, "notes"), 10) // bare name with no matching cmd/<name> dir

	// A file that looks like an artifact but is TRACKED (not ignored) — must be skipped.
	writeSizedFile(t, filepath.Join(root, "tracked.exe"), 999)

	ignored := map[string]bool{
		"conceptbench.exe": true,
		"fak-new.exe":      true,
		"fak.exe~":         true,
		"batchbench":       true,
		live:               true,
		// "tracked.exe" is deliberately absent -> treated as tracked.
	}
	opts := cleanBinsOptions{
		Root:      root,
		Apply:     true,
		CmdDirs:   map[string]bool{"batchbench": true, "fak": true},
		IsIgnored: func(name string) (bool, error) { return ignored[name], nil },
	}

	res := runCleanBins(opts)

	if len(res.Errors) != 0 {
		t.Fatalf("unexpected errors: %v", res.Errors)
	}
	wantRemoved := map[string]int64{
		"batchbench":       300,
		"conceptbench.exe": 100,
		"fak-new.exe":      200,
		"fak.exe~":         50,
	}
	if len(res.Removed) != len(wantRemoved) {
		t.Fatalf("removed %d files, want %d: %+v", len(res.Removed), len(wantRemoved), res.Removed)
	}
	var wantBytes int64
	for _, r := range res.Removed {
		w, ok := wantRemoved[r.Name]
		if !ok {
			t.Errorf("removed unexpected file %q", r.Name)
		}
		if r.Bytes != w {
			t.Errorf("removed %q reported %d bytes, want %d", r.Name, r.Bytes, w)
		}
		wantBytes += w
	}
	if res.TotalBytes != wantBytes {
		t.Errorf("TotalBytes = %d, want %d", res.TotalBytes, wantBytes)
	}

	// The live binary and the tracked look-alike must survive on disk.
	for _, keep := range []string{live, "tracked.exe", "README.md", "notes"} {
		if _, err := os.Stat(filepath.Join(root, keep)); err != nil {
			t.Errorf("%s should have survived: %v", keep, err)
		}
	}
	// The removed ones must be gone.
	for name := range wantRemoved {
		if _, err := os.Stat(filepath.Join(root, name)); !os.IsNotExist(err) {
			t.Errorf("%s should have been removed (stat err=%v)", name, err)
		}
	}

	// Idempotent: a second apply over the swept tree removes nothing.
	res2 := runCleanBins(opts)
	if len(res2.Removed) != 0 {
		t.Errorf("second run should remove nothing, got %+v", res2.Removed)
	}
}

// TestCleanBinsDryRunMutatesNothing pins that --dry-run reports candidates but leaves
// every file on disk.
func TestCleanBinsDryRunMutatesNothing(t *testing.T) {
	root := t.TempDir()
	writeSizedFile(t, filepath.Join(root, "conceptbench.exe"), 100)

	res := runCleanBins(cleanBinsOptions{
		Root:      root,
		Apply:     false,
		CmdDirs:   map[string]bool{},
		IsIgnored: func(string) (bool, error) { return true, nil },
	})

	if res.Apply {
		t.Error("Apply should be false in dry-run")
	}
	if len(res.Removed) != 1 || res.Removed[0].Name != "conceptbench.exe" {
		t.Fatalf("dry-run should list the candidate, got %+v", res.Removed)
	}
	if _, err := os.Stat(filepath.Join(root, "conceptbench.exe")); err != nil {
		t.Errorf("dry-run deleted the file: %v", err)
	}
}

// TestCleanBinsClassifiesRemoveErrors pins the housekeeping-loop contract: a locked /
// in-use artifact (a permission error — the memory-mapped .exe~ a live fak holds open on
// Windows) is a SKIP that keeps the loop green, while a genuine I/O error is a hard error
// that a loop/CI must notice.
func TestCleanBinsClassifiesRemoveErrors(t *testing.T) {
	root := t.TempDir()
	writeSizedFile(t, filepath.Join(root, "locked.exe"), 10)
	writeSizedFile(t, filepath.Join(root, "broken.exe"), 10)

	res := runCleanBins(cleanBinsOptions{
		Root:      root,
		Apply:     true,
		CmdDirs:   map[string]bool{},
		IsIgnored: func(string) (bool, error) { return true, nil },
		Remove: func(path string) error {
			switch filepath.Base(path) {
			case "locked.exe":
				return os.ErrPermission // in-use / locked -> skip, not error
			case "broken.exe":
				return fmt.Errorf("disk on fire") // genuine fault -> error
			}
			return nil
		},
	})

	if len(res.Removed) != 0 {
		t.Errorf("no file should count as removed when both removes fail, got %+v", res.Removed)
	}
	if len(res.Errors) != 1 {
		t.Fatalf("want exactly 1 hard error (broken.exe), got %v", res.Errors)
	}
	if !strings.Contains(res.Errors[0], "broken.exe") {
		t.Errorf("hard error should name broken.exe, got %q", res.Errors[0])
	}
	var sawLocked bool
	for _, s := range res.Skipped {
		if strings.Contains(s, "locked.exe") && strings.Contains(s, "in use") {
			sawLocked = true
		}
	}
	if !sawLocked {
		t.Errorf("locked.exe should be a skip (in use), got skips %v", res.Skipped)
	}
}

// TestCleanBinsAllIncludesLive pins that --all removes the live binary too.
func TestCleanBinsAllIncludesLive(t *testing.T) {
	root := t.TempDir()
	live := liveBinaryName()
	writeSizedFile(t, filepath.Join(root, live), 1000)

	res := runCleanBins(cleanBinsOptions{
		Root:        root,
		Apply:       true,
		IncludeLive: true,
		CmdDirs:     map[string]bool{"fak": true},
		IsIgnored:   func(string) (bool, error) { return true, nil },
	})

	if len(res.Removed) != 1 || res.Removed[0].Name != live {
		t.Fatalf("--all should remove the live binary, got %+v (skipped %v)", res.Removed, res.Skipped)
	}
	if _, err := os.Stat(filepath.Join(root, live)); !os.IsNotExist(err) {
		t.Errorf("live binary should have been removed with --all")
	}
}
