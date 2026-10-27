package leaseref

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// TestStrictLeaseSnapshotRejectsEnumerationFailure distinguishes an empty valid
// store from a failed Git observation, while retaining legacy List compatibility.
// fak-test:runtime fast est=100ms lane=default
func TestStrictLeaseSnapshotRejectsEnumerationFailure(t *testing.T) {
	t.Parallel()
	dir := initTestRealGitRepo(t)
	ctx := context.Background()
	store := NewInDir(dir)

	// An absent lease namespace in a valid repository is a successful empty read.
	if out, code, err := gitRunner(ctx, dir, "for-each-ref", "--format=%(refname)", refPrefix); err != nil || code != 0 || out != "" {
		t.Fatalf("empty enumeration precondition: out=%q code=%d err=%v", out, code, err)
	}
	if got, err := store.StrictSnapshot(ctx); err != nil || got == nil || len(got) != 0 {
		t.Fatalf("empty strict snapshot = %#v, %v; want non-nil empty slice, nil", got, err)
	}
	if _, err := store.Acquire(ctx, Record{
		ID:         "enumeration-peer",
		TreeGlobs:  []string{"internal/leaseref/**"},
		Holder:     "synthetic-peer",
		AcquiredAt: 1700000000,
		TTLSeconds: 600,
	}); err != nil {
		t.Fatalf("seed temporary lease: %v", err)
	}
	if got, err := store.StrictSnapshot(ctx); err != nil || len(got) != 1 || got[0].ID != "enumeration-peer" {
		t.Fatalf("healthy strict snapshot = %+v, %v; want seeded peer", got, err)
	}

	// Corrupt only this disposable repository's packed-ref storage. Unlike a
	// malformed lease blob, this makes enumeration itself fail before blob reads.
	if err := os.WriteFile(filepath.Join(dir, ".git", "packed-refs"), []byte("not a packed ref\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if out, code, err := gitRunner(ctx, dir, "for-each-ref", "--format=%(refname)", refPrefix); err != nil || code == 0 {
		t.Fatalf("genuine Git read failure precondition: out=%q code=%d err=%v", out, code, err)
	}
	if got, err := store.List(ctx); err != nil || len(got) != 0 {
		t.Fatalf("legacy tolerant List = %+v, %v; want empty, nil", got, err)
	}
	if got, err := store.StrictSnapshot(ctx); err == nil || len(got) != 0 {
		t.Fatalf("StrictSnapshot after failed enumeration = %+v, %v; want no records and an error", got, err)
	}
}
