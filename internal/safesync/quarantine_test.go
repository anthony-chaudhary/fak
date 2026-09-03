package safesync

import (
	"context"
	"os"
	"path/filepath"
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
