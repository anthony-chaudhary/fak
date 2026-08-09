package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPrintDoesNotTouchDiskAndCheckDetectsDrift(t *testing.T) {
	root := repoRoot(t)
	target := filepath.Join(root, filepath.FromSlash(outputPath))
	before, err := os.Stat(target)
	if err != nil {
		t.Fatal(err)
	}
	if code := run([]string{"-root", root, "-print"}); code != 0 {
		t.Fatalf("print code=%d", code)
	}
	after, err := os.Stat(target)
	if err != nil {
		t.Fatal(err)
	}
	if !before.ModTime().Equal(after.ModTime()) || before.Size() != after.Size() {
		t.Fatal("-print touched generated page")
	}

	scratch := t.TempDir()
	if err := os.MkdirAll(filepath.Join(scratch, "docs", "generated"), 0o755); err != nil {
		t.Fatal(err)
	}
	// A non-repository root must fail rather than blessing an empty source tree.
	if code := run([]string{"-root", scratch}); code == 0 {
		t.Fatal("check accepted an empty source tree")
	}
}

func TestCheckedInPageIsFresh(t *testing.T) {
	if code := run([]string{"-root", repoRoot(t)}); code != 0 {
		t.Fatalf("check code=%d", code)
	}
}

func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}
