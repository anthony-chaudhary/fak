package main

// backendscaffold_test.go — CLI-shell tests for `fak backend scaffold` (#1685). The generation
// logic itself is witnessed in internal/compute/scaffold_test.go (including the real go-build/
// go-test golden proof); this file only checks the flag parsing and file-writing shell around it.

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunBackendScaffoldWritesFiles(t *testing.T) {
	dir := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := runBackend(&stdout, &stderr, []string{"scaffold", "cliztest", "--lane", "custom", "--dir", dir})
	if code != 0 {
		t.Fatalf("runBackend scaffold exit = %d, stderr=%s", code, stderr.String())
	}
	for _, want := range []string{"cliztest_arch.go", "cliztest_arch_test.go", "cliztest_backend.go", "CLIZTEST-NOTES.md"} {
		if _, err := os.Stat(filepath.Join(dir, want)); err != nil {
			t.Errorf("expected file %s not written: %v", want, err)
		}
	}
	if !strings.Contains(stdout.String(), "wrote 4 files") {
		t.Errorf("stdout missing file-count summary: %s", stdout.String())
	}
}

func TestRunBackendScaffoldRequiresLane(t *testing.T) {
	dir := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := runBackend(&stdout, &stderr, []string{"scaffold", "cliztest2", "--dir", dir})
	if code == 0 {
		t.Fatal("runBackend scaffold with no --lane succeeded, want a nonzero exit")
	}
	if !strings.Contains(stderr.String(), "--lane") {
		t.Errorf("stderr should mention --lane: %s", stderr.String())
	}
}

func TestRunBackendScaffoldRejectsUnknownLane(t *testing.T) {
	dir := t.TempDir()
	var stdout, stderr bytes.Buffer
	code := runBackend(&stdout, &stderr, []string{"scaffold", "cliztest3", "--lane", "notareallane", "--dir", dir})
	if code == 0 {
		t.Fatal("runBackend scaffold with an unknown --lane succeeded, want a nonzero exit")
	}
}

func TestRunBackendUnknownSubcommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBackend(&stdout, &stderr, []string{"not-a-subcommand"})
	if code != 2 {
		t.Errorf("runBackend unknown subcommand exit = %d, want 2", code)
	}
}
