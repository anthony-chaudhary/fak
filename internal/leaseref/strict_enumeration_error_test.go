package leaseref

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=100ms
func TestStrictSnapshotRejectsEnumerationFailure(t *testing.T) {
	t.Parallel()
	dir := initTestRealGitRepo(t)
	s := NewInDir(dir)
	ctx := context.Background()
	if records, err := s.StrictSnapshot(ctx); err != nil || len(records) != 0 {
		t.Fatalf("empty namespace: records=%v, err=%v", records, err)
	}

	// A Git failure is not evidence of an empty namespace. Break only this
	// fixture's config to exercise the real runner's nonzero-exit contract.
	if err := os.WriteFile(filepath.Join(dir, ".git", "config"), []byte("[unterminated\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if records, err := s.StrictSnapshot(ctx); err == nil || records != nil ||
		!strings.Contains(err.Error(), "for-each-ref") || IsCorruptRef(err) {
		t.Fatalf("failed enumeration: records=%v, err=%v; want enumeration error, not empty success or per-ref corruption", records, err)
	}
	if live, expired, err := s.StrictLive(ctx, time.Now()); err == nil || live != nil || expired != nil {
		t.Fatalf("failed live enumeration: live=%v, expired=%v, err=%v", live, expired, err)
	}
	if records, err := s.List(ctx); err != nil || len(records) != 0 {
		t.Fatalf("legacy tolerant enumeration: records=%v, err=%v", records, err)
	}
}
