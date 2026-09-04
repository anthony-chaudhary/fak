package gitgate

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestGitGateShadowCheckpoint(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "gitgate-checkpoint-test-*")
	if err != nil {
		t.Fatalf("create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	runGit := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = tempDir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v failed: %v: %s", args, err, string(out))
		}
	}

	runGit("init")
	runGit("config", "user.name", "Test User")
	runGit("config", "user.email", "test@example.com")

	trackedPath := filepath.Join(tempDir, "tracked.txt")
	if err := os.WriteFile(trackedPath, []byte("v1"), 0o644); err != nil {
		t.Fatalf("write tracked: %v", err)
	}
	runGit("add", "tracked.txt")
	runGit("commit", "-m", "initial commit")

	// Make modifications and add untracked file
	if err := os.WriteFile(trackedPath, []byte("v2"), 0o644); err != nil {
		t.Fatalf("update tracked: %v", err)
	}
	untrackedPath := filepath.Join(tempDir, "untracked.txt")
	if err := os.WriteFile(untrackedPath, []byte("untracked-v1"), 0o644); err != nil {
		t.Fatalf("write untracked: %v", err)
	}

	ref, err := CreateShadowCheckpoint(tempDir)
	if err != nil {
		t.Fatalf("CreateShadowCheckpoint: %v", err)
	}
	if ref == "" {
		t.Fatal("expected non-empty shadow checkpoint ref")
	}

	// Change working tree further
	if err := os.WriteFile(trackedPath, []byte("v3-dirty"), 0o644); err != nil {
		t.Fatalf("write tracked dirty: %v", err)
	}
	if err := os.Remove(untrackedPath); err != nil {
		t.Fatalf("delete untracked: %v", err)
	}

	// Restore checkpoint
	if err := RestoreShadowCheckpoint(tempDir, ref); err != nil {
		t.Fatalf("RestoreShadowCheckpoint: %v", err)
	}

	trackedBytes, err := os.ReadFile(trackedPath)
	if err != nil || string(trackedBytes) != "v2" {
		t.Fatalf("expected tracked content 'v2', got %q (err=%v)", string(trackedBytes), err)
	}

	untrackedBytes, err := os.ReadFile(untrackedPath)
	if err != nil || string(untrackedBytes) != "untracked-v1" {
		t.Fatalf("expected untracked content 'untracked-v1', got %q (err=%v)", string(untrackedBytes), err)
	}
}
