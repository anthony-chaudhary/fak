package main

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// availableSuites is the core of the actionable unknown-suite error: it lists the
// suite NAMES (the *.json basenames) a --suite flag could name, so a cold-start user
// who guesses wrong sees the real choices instead of a raw file-not-found. These pin
// that it returns sorted names, strips .json, ignores non-json + dirs, and is empty
// (never a panic) on a missing dir.
func TestAvailableSuites(t *testing.T) {
	dir := t.TempDir()
	for _, f := range []string{"turntax-airline.json", "guard-redteam.json", "turntax-happy.json", "README.md", "notes.txt"} {
		if err := os.WriteFile(filepath.Join(dir, f), []byte("{}"), 0o644); err != nil {
			t.Fatalf("write %s: %v", f, err)
		}
	}
	if err := os.Mkdir(filepath.Join(dir, "subdir.json"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	got := availableSuites(dir)
	want := []string{"guard-redteam", "turntax-airline", "turntax-happy"} // sorted, .json stripped, non-json + dir excluded
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("availableSuites(%s) = %v, want %v", dir, got, want)
	}
}

func TestAvailableSuitesMissingDirIsEmpty(t *testing.T) {
	got := availableSuites(filepath.Join(t.TempDir(), "does-not-exist"))
	if len(got) != 0 {
		t.Fatalf("a missing dir must yield no suites (so the caller reports it), got %v", got)
	}
}
