package selfinstall

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestFailedVetRemovesCandidateAndPreservesTarget(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "fak")
	candidate := target + ".new"
	original := []byte("known-good")
	if err := os.WriteFile(target, original, 0o700); err != nil {
		t.Fatal(err)
	}
	run := func(_ context.Context, _ string, name string, args ...string) (string, bool) {
		if name == "git" {
			return "", true
		}
		if name == "go" && len(args) > 0 && args[0] == "build" {
			if err := os.WriteFile(candidate, []byte("candidate"), 0o700); err != nil {
				t.Fatal(err)
			}
			return "", true
		}
		if name == "go" && len(args) > 0 && args[0] == "vet" {
			return "forced vet failure", false
		}
		return "", false
	}
	swapped := false
	result := Install(context.Background(), run, func(string, string) error { swapped = true; return nil }, Options{RepoRoot: dir, Target: target})
	if result.Stage != StageVet || result.Installed || swapped {
		t.Fatalf("result=%+v swapped=%v", result, swapped)
	}
	if _, err := os.Stat(candidate); !os.IsNotExist(err) {
		t.Fatalf("candidate remains after vet failure: %v", err)
	}
	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(original) {
		t.Fatalf("target changed: got=%q want=%q", got, original)
	}
}
