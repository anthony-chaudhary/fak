package resume

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestTickLockSerializesOverlappingTicks simulates two overlapping resume-watchdog
// ticks contending for the same registry dir's lock (#3110): the first acquires, a
// concurrent second must back off (acquired=false, no error) so it never admits the
// same still-starting session the first tick is mid-launch on. Releasing frees the
// lock for a later tick.
func TestTickLockSerializesOverlappingTicks(t *testing.T) {
	regDir := t.TempDir()

	release1, acquired1, err1 := TryTickLock(regDir)
	if err1 != nil {
		t.Fatalf("first TryTickLock: unexpected error: %v", err1)
	}
	if !acquired1 {
		t.Fatalf("first TryTickLock: want acquired=true (uncontended), got false")
	}

	// A concurrent second tick (e.g. a cron/--live tick overlapping a slow tick)
	// reads the same regDir while the first tick's lock is still live.
	release2, acquired2, err2 := TryTickLock(regDir)
	if err2 != nil {
		t.Fatalf("second TryTickLock: unexpected error: %v", err2)
	}
	if acquired2 {
		t.Fatalf("second TryTickLock: want acquired=false while the first tick holds the lock, got true — two ticks could both admit the same session")
	}
	// A refused acquire's release must be a harmless no-op — it never held anything.
	if err := release2(); err != nil {
		t.Fatalf("second (unacquired) release: want nil, got %v", err)
	}

	lockPath := filepath.Join(regDir, TickLockName)
	if _, err := os.Stat(lockPath); err != nil {
		t.Fatalf("lockfile %s missing while first tick holds it: %v", lockPath, err)
	}

	// The first tick finishes and releases; the lock must now be free for a later tick.
	if err := release1(); err != nil {
		t.Fatalf("first release: unexpected error: %v", err)
	}
	if _, err := os.Stat(lockPath); !os.IsNotExist(err) {
		t.Fatalf("lockfile %s still present after release: err=%v", lockPath, err)
	}

	release3, acquired3, err3 := TryTickLock(regDir)
	if err3 != nil {
		t.Fatalf("third TryTickLock (after release): unexpected error: %v", err3)
	}
	if !acquired3 {
		t.Fatalf("third TryTickLock: want acquired=true once the lock is free, got false")
	}
	if err := release3(); err != nil {
		t.Fatalf("third release: unexpected error: %v", err)
	}
}

// TestTickLockReclaimsStaleLock covers the crash-recovery path: a lockfile whose mtime
// is older than TickLockTTL is a crashed tick's leftover (ticks run seconds, not
// minutes), and must be reclaimed rather than permanently wedging every future tick.
func TestTickLockReclaimsStaleLock(t *testing.T) {
	regDir := t.TempDir()
	lockPath := filepath.Join(regDir, TickLockName)

	if err := os.WriteFile(lockPath, []byte("99999 1\n"), 0o644); err != nil {
		t.Fatalf("seeding stale lockfile: %v", err)
	}
	stale := time.Now().Add(-2 * TickLockTTL)
	if err := os.Chtimes(lockPath, stale, stale); err != nil {
		t.Fatalf("backdating lockfile mtime: %v", err)
	}

	release, acquired, err := TryTickLock(regDir)
	if err != nil {
		t.Fatalf("TryTickLock over a stale lock: unexpected error: %v", err)
	}
	if !acquired {
		t.Fatalf("TryTickLock over a stale (past-TTL) lock: want acquired=true (reclaim), got false")
	}
	if err := release(); err != nil {
		t.Fatalf("release after reclaim: unexpected error: %v", err)
	}
	if _, err := os.Stat(lockPath); !os.IsNotExist(err) {
		t.Fatalf("lockfile %s still present after reclaim+release: err=%v", lockPath, err)
	}
}

// TestTickLockFreshLockNotReclaimed is the negative control for the staleness check:
// a lock well inside the TTL must NOT be reclaimed, even though the reclaim path is
// reachable — this is what actually prevents a live overlapping tick from being
// clobbered mid-run.
func TestTickLockFreshLockNotReclaimed(t *testing.T) {
	regDir := t.TempDir()
	lockPath := filepath.Join(regDir, TickLockName)

	if err := os.WriteFile(lockPath, []byte("12345 1\n"), 0o644); err != nil {
		t.Fatalf("seeding fresh lockfile: %v", err)
	}
	// mtime defaults to now — well inside TickLockTTL.

	_, acquired, err := TryTickLock(regDir)
	if err != nil {
		t.Fatalf("TryTickLock over a fresh lock: unexpected error: %v", err)
	}
	if acquired {
		t.Fatalf("TryTickLock over a fresh (live) lock: want acquired=false, got true — a live tick would be clobbered")
	}
	if _, err := os.Stat(lockPath); err != nil {
		t.Fatalf("fresh lockfile %s must survive a refused acquire: %v", lockPath, err)
	}
}
