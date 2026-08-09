package leaseref

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestScanLockFilesInDirPreviewsWithoutRemoving pins the property the dry-run depends
// on: Scan reports the SAME partition Reap would act on, and every file survives. A
// preview that removed anything (or that classified differently from the reaper) would
// make `fak git-daily --dry-run` a lie about what the real run does.
func TestScanLockFilesInDirPreviewsWithoutRemoving(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	ghost := writeLockAged(t, dir, "session-dead.lock", now.Add(-3*time.Hour))
	fresh := writeLockAged(t, dir, "session-live.lock", now.Add(-5*time.Second))
	// A non-.lock sibling: the loose lease REF itself, which no sweep may ever touch.
	ref := filepath.Join(dir, "session-live")
	if err := os.WriteFile(ref, []byte("deadbeef\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	orphans, kept, err := ScanLockFilesInDir(dir, now, DefaultLockFileMaxAge)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if len(orphans) != 1 || orphans[0] != "session-dead.lock" {
		t.Fatalf("orphans = %v, want [session-dead.lock]", orphans)
	}
	if len(kept) != 1 || kept[0] != "session-live.lock" {
		t.Fatalf("kept = %v, want [session-live.lock]", kept)
	}
	for _, p := range []string{ghost, fresh, ref} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("scan removed %s: %v", filepath.Base(p), err)
		}
	}

	// The reaper, on the same tree, must agree exactly — and only IT removes.
	reaped, rkept, err := ReapLockFilesInDir(dir, now, DefaultLockFileMaxAge)
	if err != nil {
		t.Fatalf("reap: %v", err)
	}
	if len(reaped) != 1 || reaped[0] != orphans[0] || len(rkept) != 1 || rkept[0] != kept[0] {
		t.Fatalf("reap partition %v/%v disagrees with scan %v/%v", reaped, rkept, orphans, kept)
	}
	if _, err := os.Stat(ghost); !os.IsNotExist(err) {
		t.Fatalf("reap left the ghost behind: %v", err)
	}
	if _, err := os.Stat(fresh); err != nil {
		t.Fatalf("reap removed the fresh lock: %v", err)
	}
}

// TestScanLockFilesInDirAbsentDir keeps the first-run contract on the read-only twin:
// a repo that never took a lease has an empty view, not an error.
func TestScanLockFilesInDirAbsentDir(t *testing.T) {
	orphans, kept, err := ScanLockFilesInDir(filepath.Join(t.TempDir(), "nope"), time.Now(), 0)
	if err != nil || len(orphans) != 0 || len(kept) != 0 {
		t.Fatalf("absent dir: orphans=%v kept=%v err=%v, want empty/nil", orphans, kept, err)
	}
}

func writeLockAged(t *testing.T, dir, name string, mtime time.Time) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(p, mtime, mtime); err != nil {
		t.Fatal(err)
	}
	return p
}
