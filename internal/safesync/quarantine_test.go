package safesync

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPrepareQuarantineAndCommit(t *testing.T) {
	tmp := t.TempDir()
	fileA := filepath.Join(tmp, "a.txt")
	fileB := filepath.Join(tmp, "b.txt")
	if err := os.WriteFile(fileA, []byte("content-a"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fileB, []byte("content-b"), 0o644); err != nil {
		t.Fatal(err)
	}

	tx, err := PrepareQuarantine(tmp, []string{"a.txt", "b.txt"}, map[string]bool{"a.txt": true, "b.txt": false})
	if err != nil {
		t.Fatalf("PrepareQuarantine: %v", err)
	}
	if len(tx.Files) != 2 {
		t.Fatalf("quarantined files = %d, want 2", len(tx.Files))
	}

	// Working tree files should have been moved
	if _, err := os.Stat(fileA); !os.IsNotExist(err) {
		t.Fatalf("expected a.txt to be quarantined from working tree")
	}
	if _, err := os.Stat(fileB); !os.IsNotExist(err) {
		t.Fatalf("expected b.txt to be quarantined from working tree")
	}

	// Simulate incoming git fast-forward creating a.txt (identical)
	if err := os.WriteFile(fileA, []byte("content-a"), 0o644); err != nil {
		t.Fatal(err)
	}

	receipt, err := tx.Commit()
	if err != nil {
		t.Fatalf("tx.Commit: %v", err)
	}

	if receipt.QuarantinedCount != 2 {
		t.Fatalf("QuarantinedCount = %d, want 2", receipt.QuarantinedCount)
	}
	if receipt.IdenticalCount != 1 {
		t.Fatalf("IdenticalCount = %d, want 1", receipt.IdenticalCount)
	}
	if receipt.RestoredCount != 1 {
		t.Fatalf("RestoredCount = %d, want 1", receipt.RestoredCount)
	}

	// b.txt should have been restored byte-identical
	bBytes, err := os.ReadFile(fileB)
	if err != nil || string(bBytes) != "content-b" {
		t.Fatalf("b.txt restored content = %q, err = %v", string(bBytes), err)
	}
}

func TestQuarantineRollback(t *testing.T) {
	tmp := t.TempDir()
	fileA := filepath.Join(tmp, "scratch.txt")
	if err := os.WriteFile(fileA, []byte("scratch-data"), 0o644); err != nil {
		t.Fatal(err)
	}

	tx, err := PrepareQuarantine(tmp, []string{"scratch.txt"}, nil)
	if err != nil {
		t.Fatal(err)
	}

	if err := tx.Rollback(); err != nil {
		t.Fatalf("Rollback: %v", err)
	}

	got, err := os.ReadFile(fileA)
	if err != nil || string(got) != "scratch-data" {
		t.Fatalf("scratch.txt after rollback = %q, err = %v", string(got), err)
	}
}

// TestApplyFastForwardWithQuarantineWitness proves the exact #10913 witness:
// A deterministic temp-repo fixture where an untracked scratch file in an unedited
// directory does not wedge safesync.Apply, is safely quarantined across the
// fast-forward merge, and is restored byte-identical with a passing verification receipt.
func TestApplyFastForwardWithQuarantineWitness(t *testing.T) {
	clone := behindClone(t)

	// In behindClone, remote commit added new.txt.
	// Create an untracked scratch file matching new.txt with identical content.
	writeFile(t, filepath.Join(clone, "new.txt"), "n1\n")
	headBefore := revString(t, clone, "HEAD")

	// Apply with QuarantineScratch enabled
	applied, err := Apply(context.Background(), Options{
		Repo:              clone,
		Remote:            "origin",
		Branch:            "work",
		QuarantineScratch: true,
	})
	if err != nil {
		t.Fatalf("Apply with QuarantineScratch: %v", err)
	}

	if !applied.Applied || !applied.OK {
		t.Fatalf("Apply refused despite QuarantineScratch: %+v", applied)
	}

	if applied.Quarantine == nil {
		t.Fatal("expected Quarantine receipt on Assessment")
	}
	if applied.Quarantine.QuarantinedCount != 1 || applied.Quarantine.IdenticalCount != 1 {
		t.Fatalf("unexpected quarantine receipt: %+v", applied.Quarantine)
	}

	headAfter := revString(t, clone, "HEAD")
	if headAfter == headBefore {
		t.Fatalf("HEAD was not advanced: %s", headAfter)
	}

	if got := readFile(t, filepath.Join(clone, "new.txt")); got != "n1\n" {
		t.Fatalf("new.txt content = %q, want n1\n", got)
	}
}

// TestSafesyncAutoQuarantineCollidingUntracked proves Issue #11233:
// An untracked local file internal/foo/new.go with conflicting content is safely
// quarantined across fast-forward, sync completes cleanly without "untracked working tree
// files would be overwritten" aborts, and receipt contains the SHA256 of the preserved local file.
func TestSafesyncAutoQuarantineCollidingUntracked(t *testing.T) {
	clone := behindClone(t)
	originDir := filepath.Join(filepath.Dir(clone), "origin")

	// Upstream commit in origin introduces internal/foo/new.go
	originFile := filepath.Join(originDir, "internal", "foo", "new.go")
	if err := os.MkdirAll(filepath.Dir(originFile), 0o755); err != nil {
		t.Fatal(err)
	}
	remoteContent := "package foo\n\n// remote incoming addition\nfunc New() string { return \"incoming\" }\n"
	writeFile(t, originFile, remoteContent)
	git(t, originDir, "add", "internal/foo/new.go")
	git(t, originDir, "commit", "-m", "add internal/foo/new.go")
	git(t, clone, "fetch", "origin")

	// Local untracked file internal/foo/new.go with conflicting content
	localFile := filepath.Join(clone, "internal", "foo", "new.go")
	if err := os.MkdirAll(filepath.Dir(localFile), 0o755); err != nil {
		t.Fatal(err)
	}
	localContent := "package foo\n\n// local conflicting untracked content\nfunc New() string { return \"conflicting-local\" }\n"
	writeFile(t, localFile, localContent)

	expectedHash, _, err := FileSHA256(localFile)
	if err != nil {
		t.Fatalf("FileSHA256: %v", err)
	}

	headBefore := revString(t, clone, "HEAD")

	// Apply with AutoQuarantine enabled
	applied, err := Apply(context.Background(), Options{
		Repo:           clone,
		Remote:         "origin",
		Branch:         "work",
		AutoQuarantine: true,
	})
	if err != nil {
		t.Fatalf("Apply with AutoQuarantine: %v", err)
	}
	if !applied.OK || !applied.Applied {
		t.Fatalf("expected apply to succeed, got: %+v", applied)
	}

	headAfter := revString(t, clone, "HEAD")
	if headAfter == headBefore {
		t.Fatalf("HEAD did not advance: %s", headAfter)
	}

	// Fast-forward should have populated internal/foo/new.go with remoteContent
	if got := readFile(t, localFile); got != remoteContent {
		t.Fatalf("working tree internal/foo/new.go = %q, want remote content %q", got, remoteContent)
	}

	// Quarantine receipt must be attached and contain the preserved file's SHA256
	if applied.Quarantine == nil {
		t.Fatal("expected Quarantine receipt on Assessment")
	}
	if applied.Quarantine.SHA256 != expectedHash && applied.Quarantine.Preserved["internal/foo/new.go"] != expectedHash {
		t.Fatalf("quarantine receipt does not contain expected SHA256 %s: %+v", expectedHash, applied.Quarantine)
	}

	// Verify receipt on disk
	receiptPath := applied.Quarantine.ReceiptPath
	if receiptPath == "" {
		gitDir, _ := worktreeGitDir(clone)
		receiptPath = filepath.Join(gitDir, "fak-quarantine", "receipt.json")
	}
	receiptBytes, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatalf("read receipt from disk (%s): %v", receiptPath, err)
	}
	if !strings.Contains(string(receiptBytes), expectedHash) {
		t.Fatalf("receipt on disk does not contain expected SHA256: %s", string(receiptBytes))
	}
}
