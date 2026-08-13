package treedoctor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPrepareScratchDirCreatesIgnoredNamespaceChild(t *testing.T) {
	repo := t.TempDir()
	got, err := PrepareScratchDir(repo, "fleet-loop/run-17")
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(repo, "_scratch", "fleet-loop", "run-17")
	if got != want {
		t.Fatalf("path = %q, want %q", got, want)
	}
	if info, err := os.Stat(got); err != nil || !info.IsDir() {
		t.Fatalf("prepared directory missing: info=%v err=%v", info, err)
	}
}

func TestPrepareScratchPathCreatesParentAndKeepsFilename(t *testing.T) {
	repo := t.TempDir()
	got, err := PrepareScratchPath(repo, "fleet-loop/tick.json")
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(repo, "_scratch", "fleet-loop", "tick.json")
	if got != want {
		t.Fatalf("path = %q, want %q", got, want)
	}
	if _, err := os.Stat(filepath.Dir(got)); err != nil {
		t.Fatalf("parent was not created: %v", err)
	}
}

func TestPrepareScratchLocationRefusesEscapeAndFlatFiles(t *testing.T) {
	repo := t.TempDir()
	for _, tc := range []struct {
		name string
		call func() (string, error)
		want string
	}{
		{"empty", func() (string, error) { return PrepareScratchDir(repo, "") }, "empty"},
		{"parent", func() (string, error) { return PrepareScratchDir(repo, "../outside") }, "escapes"},
		{"absolute", func() (string, error) { return PrepareScratchDir(repo, filepath.Join(repo, "outside")) }, "repository-relative"},
		{"double namespace", func() (string, error) { return PrepareScratchDir(repo, "_scratch/run") }, "not include it"},
		{"flat file", func() (string, error) { return PrepareScratchPath(repo, "tick.json") }, "producer directory"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := tc.call()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want substring %q", err, tc.want)
			}
		})
	}
}
