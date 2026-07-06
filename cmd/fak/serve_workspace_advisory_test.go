package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestWarnIfNotFakWorkspace_WarnsOutsideFakTree asserts the #3094 mis-binding guard:
// a serve launched from a cwd with no dos.toml upward gets a loud stderr advisory, so
// an operator who drops fak serve into a sibling repo (the failure mode that left the
// substrate pointed at the wrong tree) is told rather than silently mis-binding.
func TestWarnIfNotFakWorkspace_WarnsOutsideFakTree(t *testing.T) {
	// A bare temp dir has no dos.toml anywhere up to the filesystem root.
	t.Chdir(t.TempDir())

	var errb bytes.Buffer
	warnIfNotFakWorkspace(&errb)

	got := errb.String()
	if !strings.Contains(got, "not a fak workspace") {
		t.Fatalf("expected advisory outside a fak workspace, got: %q", got)
	}
}

// TestWarnIfNotFakWorkspace_SilentInsideFakTree asserts the guard is quiet inside a real
// fak workspace (a cwd with dos.toml on the path) — no false-positive noise on the
// happy path.
func TestWarnIfNotFakWorkspace_SilentInsideFakTree(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte("[branch_roles]\n"), 0o644); err != nil {
		t.Fatalf("seed dos.toml: %v", err)
	}
	t.Chdir(root)

	var errb bytes.Buffer
	warnIfNotFakWorkspace(&errb)

	if got := errb.String(); got != "" {
		t.Fatalf("expected no advisory inside a fak workspace, got: %q", got)
	}
}
