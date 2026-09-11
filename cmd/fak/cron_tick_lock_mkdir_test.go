package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestCronTickLockCreatesMissingParentDir proves the regression from production:
// cronTickLock on "<ledger>.tick.lock" used to fail with ENOENT when the ledger's
// parent directory did not yet exist (O_CREATE|O_EXCL cannot create parents). The
// fix MkdirAll's the parent before each CreatePIDTime attempt, so acquiring a tick
// lock in a not-yet-created .fak/ledgers/ dir must succeed, remove cleanly on
// release, and succeed again immediately after (idempotent re-acquire).
func TestCronTickLockCreatesMissingParentDir(t *testing.T) {
	ledgerPath := filepath.Join(t.TempDir(), "does-not-exist-yet", "ledgers", "job.jsonl")
	lockPath := ledgerPath + ".tick.lock"

	// Precondition: the lock's parent directory must NOT exist yet.
	if _, err := os.Stat(filepath.Dir(lockPath)); !os.IsNotExist(err) {
		t.Fatalf("parent dir %s should not exist before the test; stat err=%v", filepath.Dir(lockPath), err)
	}

	release, err := cronTickLock(lockPath, 0, time.Minute)
	if err != nil {
		t.Fatalf("cronTickLock with missing parent dir %s: err=%v, want nil (MkdirAll should create it)", filepath.Dir(lockPath), err)
	}
	if release == nil {
		t.Fatalf("cronTickLock returned nil release func with nil error")
	}
	if _, err := os.Stat(lockPath); err != nil {
		t.Fatalf("lock file %s should exist on disk after acquire: %v", lockPath, err)
	}

	if err := release(); err != nil {
		t.Fatalf("release: err=%v, want nil", err)
	}
	if _, err := os.Stat(lockPath); !os.IsNotExist(err) {
		t.Fatalf("lock file %s should be removed after release; stat err=%v", lockPath, err)
	}

	// Idempotency: a second immediate acquire on the same path must succeed again
	// (the dir now exists, and release removed the lockfile).
	release2, err := cronTickLock(lockPath, 0, time.Minute)
	if err != nil {
		t.Fatalf("second cronTickLock on same path: err=%v, want nil", err)
	}
	if release2 == nil {
		t.Fatalf("second cronTickLock returned nil release func with nil error")
	}
	if _, err := os.Stat(lockPath); err != nil {
		t.Fatalf("lock file %s should exist after second acquire: %v", lockPath, err)
	}
	if err := release2(); err != nil {
		t.Fatalf("second release: err=%v, want nil", err)
	}
	if _, err := os.Stat(lockPath); !os.IsNotExist(err) {
		t.Fatalf("lock file %s should be removed after second release; stat err=%v", lockPath, err)
	}
}
