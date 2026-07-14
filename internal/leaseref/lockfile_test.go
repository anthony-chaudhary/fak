package leaseref

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// mkLock creates a .lock file (or any file) under dir with the given age before now,
// returning its absolute path. Age is applied via Chtimes so the sweep's mtime clock
// sees exactly the intended staleness.
func mkLock(t *testing.T, dir, name string, now time.Time, age time.Duration) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", path, err)
	}
	if err := os.WriteFile(path, []byte("0000000000000000000000000000000000000000\n"), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	stamp := now.Add(-age)
	if err := os.Chtimes(path, stamp, stamp); err != nil {
		t.Fatalf("chtimes %s: %v", path, err)
	}
	return path
}

// TestReapLockFilesInDirRemovesOnlyStale: a 4.9-day-old orphan (the #4605 evidence
// shape) is removed; a milliseconds-fresh .lock (a possibly-live CAS) is kept and
// reported, never raced.
func TestReapLockFilesInDirRemovesOnlyStale(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	stale := mkLock(t, dir, "session-guard.lock", now, 422099*time.Second)
	fresh := mkLock(t, dir, "session-win-live.lock", now, 5*time.Millisecond)

	reaped, kept, err := ReapLockFilesInDir(dir, now, 0)
	if err != nil {
		t.Fatalf("ReapLockFilesInDir: %v", err)
	}
	if len(reaped) != 1 || reaped[0] != "session-guard.lock" {
		t.Fatalf("reaped = %v, want [session-guard.lock]", reaped)
	}
	if len(kept) != 1 || kept[0] != "session-win-live.lock" {
		t.Fatalf("kept = %v, want [session-win-live.lock]", kept)
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Fatalf("stale lock still present after sweep")
	}
	if _, err := os.Stat(fresh); err != nil {
		t.Fatalf("fresh lock wrongly removed: %v", err)
	}
}

// TestReapLockFilesInDirBoundIsInclusiveOfTTL: exactly at maxAge the lock has
// outlived its lease and is reaped; one second inside the bound it is kept — the
// "lock outlived its lease" rule, measured against an explicit bound.
func TestReapLockFilesInDirBoundIsInclusiveOfTTL(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	mkLock(t, dir, "at-bound.lock", now, 2400*time.Second)
	mkLock(t, dir, "inside-bound.lock", now, 2399*time.Second)

	reaped, kept, err := ReapLockFilesInDir(dir, now, 2400*time.Second)
	if err != nil {
		t.Fatalf("ReapLockFilesInDir: %v", err)
	}
	if len(reaped) != 1 || reaped[0] != "at-bound.lock" {
		t.Fatalf("reaped = %v, want [at-bound.lock]", reaped)
	}
	if len(kept) != 1 || kept[0] != "inside-bound.lock" {
		t.Fatalf("kept = %v, want [inside-bound.lock]", kept)
	}
}

// TestReapLockFilesInDirIgnoresNonLockFiles: an ancient loose ref (no .lock suffix)
// is never touched, however old — only git's transient lockfiles are in scope.
func TestReapLockFilesInDirIgnoresNonLockFiles(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	ref := mkLock(t, dir, "session-win-old", now, 30*24*time.Hour) // a loose ref, not a lock

	reaped, kept, err := ReapLockFilesInDir(dir, now, 0)
	if err != nil {
		t.Fatalf("ReapLockFilesInDir: %v", err)
	}
	if len(reaped) != 0 || len(kept) != 0 {
		t.Fatalf("reaped=%v kept=%v, want none of either (not a .lock)", reaped, kept)
	}
	if _, err := os.Stat(ref); err != nil {
		t.Fatalf("non-lock file wrongly removed: %v", err)
	}
}

// TestReapLockFilesInDirFutureMtimeKept: a .lock dated in the future (clock skew)
// has an unknowable age — it fails closed to not-orphaned and is kept.
func TestReapLockFilesInDirFutureMtimeKept(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	future := mkLock(t, dir, "skewed.lock", now, -48*time.Hour)

	reaped, kept, err := ReapLockFilesInDir(dir, now, 0)
	if err != nil {
		t.Fatalf("ReapLockFilesInDir: %v", err)
	}
	if len(reaped) != 0 {
		t.Fatalf("reaped = %v, want none (future mtime must fail closed)", reaped)
	}
	if len(kept) != 1 || kept[0] != "skewed.lock" {
		t.Fatalf("kept = %v, want [skewed.lock]", kept)
	}
	if _, err := os.Stat(future); err != nil {
		t.Fatalf("future-dated lock wrongly removed: %v", err)
	}
}

// TestReapLockFilesInDirAbsentDirCleanNoOp: a repo that never wrote a lease has no
// refs/fak/locks dir at all — the sweep reports the valid, empty view, not an error.
func TestReapLockFilesInDirAbsentDirCleanNoOp(t *testing.T) {
	reaped, kept, err := ReapLockFilesInDir(filepath.Join(t.TempDir(), "no-such-dir"), time.Now(), 0)
	if err != nil {
		t.Fatalf("absent dir must be a clean no-op, got: %v", err)
	}
	if len(reaped) != 0 || len(kept) != 0 {
		t.Fatalf("reaped=%v kept=%v, want empty", reaped, kept)
	}
}

// TestReapLockFilesInDirIdempotent: a second sweep after the first removed the orphan
// finds nothing to do — two peers racing the same sweep both converge, like Reap.
func TestReapLockFilesInDirIdempotent(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	mkLock(t, dir, "session-x.lock", now, 5*time.Hour)

	if _, _, err := ReapLockFilesInDir(dir, now, 0); err != nil {
		t.Fatalf("first sweep: %v", err)
	}
	reaped, kept, err := ReapLockFilesInDir(dir, now, 0)
	if err != nil {
		t.Fatalf("second sweep must be a clean no-op: %v", err)
	}
	if len(reaped) != 0 || len(kept) != 0 {
		t.Fatalf("second sweep reaped=%v kept=%v, want empty", reaped, kept)
	}
}

// TestStoreReapLockFilesResolvesCommonDir: the Store-level sweep resolves the git
// COMMON dir through the one Runner seam (linked worktrees share their refs — and
// their locks — with the main clone) and sweeps <common>/refs/fak/locks.
func TestStoreReapLockFilesResolvesCommonDir(t *testing.T) {
	common := t.TempDir()
	now := time.Now()
	locks := filepath.Join(common, "refs", "fak", "locks")
	mkLock(t, locks, "session-dead.lock", now, 6*time.Hour)

	var gotArgs []string
	s := NewWithRunner(func(_ context.Context, _ string, args ...string) (string, int, error) {
		gotArgs = args
		return common + "\n", 0, nil
	}, "")
	reaped, kept, err := s.ReapLockFiles(ctx(), now, 0)
	if err != nil {
		t.Fatalf("ReapLockFiles: %v", err)
	}
	want := []string{"rev-parse", "--path-format=absolute", "--git-common-dir"}
	if len(gotArgs) != len(want) {
		t.Fatalf("git argv = %v, want %v", gotArgs, want)
	}
	for i := range want {
		if gotArgs[i] != want[i] {
			t.Fatalf("git argv = %v, want %v", gotArgs, want)
		}
	}
	if len(reaped) != 1 || reaped[0] != "session-dead.lock" {
		t.Fatalf("reaped = %v, want [session-dead.lock]", reaped)
	}
	if len(kept) != 0 {
		t.Fatalf("kept = %v, want empty", kept)
	}
}

// TestStoreReapLockFilesGitFailureIsTyped: a non-zero rev-parse (not a repo) is the
// package's typed error, and a non-executable git is the runner's hard error — the
// same split every Store method holds.
func TestStoreReapLockFilesGitFailureIsTyped(t *testing.T) {
	s := NewWithRunner(func(_ context.Context, _ string, _ ...string) (string, int, error) {
		return "", 128, nil
	}, "")
	if _, _, err := s.ReapLockFiles(ctx(), time.Now(), 0); err == nil {
		t.Fatalf("want error on rev-parse exit 128, got nil")
	}
}
