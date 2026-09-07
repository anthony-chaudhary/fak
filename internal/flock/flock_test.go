package flock

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// TestTryLockBusyThenReleased exercises the cross-platform exclusive-lock
// primitive end to end: a second, independent handle to the same file must
// observe the lock as ErrLockBusy while the first holder owns it, and must be
// able to take the lock once the first holder releases. This is the contract
// gpulease and loopmgr depend on, and it can fail concretely if TryLock stops
// mapping the OS "would block" / "lock violation" error onto ErrLockBusy.
func TestTryLockBusyThenReleased(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lease.lock")

	h1, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		t.Fatalf("open first handle: %v", err)
	}
	defer h1.Close()

	if err := TryLock(h1); err != nil {
		t.Fatalf("first TryLock should succeed, got %v", err)
	}

	h2, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatalf("open second handle: %v", err)
	}
	defer h2.Close()

	// Held by h1 → the second handle must see it busy (not a generic error, not nil).
	if err := TryLock(h2); !errors.Is(err, ErrLockBusy) {
		t.Fatalf("contended TryLock: want ErrLockBusy, got %v", err)
	}

	if err := Unlock(h1); err != nil {
		t.Fatalf("Unlock first handle: %v", err)
	}

	// Released → the second handle can now take it.
	if err := TryLock(h2); err != nil {
		t.Fatalf("TryLock after release: want nil, got %v", err)
	}
	if err := Unlock(h2); err != nil {
		t.Fatalf("Unlock second handle: %v", err)
	}
}

// TestErrLockBusySentinel pins the sentinel's identity and message so callers
// that distinguish a contended lock from a real I/O failure via errors.Is keep
// working.
func TestErrLockBusySentinel(t *testing.T) {
	if ErrLockBusy == nil {
		t.Fatal("ErrLockBusy must be a non-nil sentinel")
	}
	if !errors.Is(ErrLockBusy, ErrLockBusy) {
		t.Fatal("ErrLockBusy must satisfy errors.Is against itself")
	}
	if got, want := ErrLockBusy.Error(), "flock: lock busy"; got != want {
		t.Fatalf("ErrLockBusy.Error() = %q, want %q", got, want)
	}
	// A plainly different error must NOT match the sentinel.
	if errors.Is(errors.New("flock: lock busy"), ErrLockBusy) {
		t.Fatal("a distinct error value with the same text must not match ErrLockBusy")
	}
}

// TestSharedLockConcurrencyAndExclusion verifies that:
// (a) multiple shared readers can acquire the lock simultaneously,
// (b) exclusive acquisition is refused while shared readers hold the lock,
// (c) exclusive acquisition succeeds once all readers have released, and
// (d) shared acquisition is refused while an exclusive writer holds the lock.
func TestSharedLockConcurrencyAndExclusion(t *testing.T) {
	path := filepath.Join(t.TempDir(), "shared.lock")

	r1, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		t.Fatalf("open r1: %v", err)
	}
	defer r1.Close()

	r2, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatalf("open r2: %v", err)
	}
	defer r2.Close()

	w1, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatalf("open w1: %v", err)
	}
	defer w1.Close()

	// (a) First shared reader succeeds.
	if err := TryLockShared(r1); err != nil {
		t.Fatalf("r1 TryLockShared: %v", err)
	}

	// Second shared reader also succeeds concurrently (using LockShared alias).
	if err := LockShared(r2); err != nil {
		t.Fatalf("r2 LockShared concurrent with r1: %v", err)
	}

	// (b) Exclusive writer must be refused while shared locks are held.
	if err := TryLock(w1); !errors.Is(err, ErrLockBusy) {
		t.Fatalf("w1 TryLock while readers active: want ErrLockBusy, got %v", err)
	}

	// Releasing one reader still leaves r2 holding shared lock.
	if err := Unlock(r1); err != nil {
		t.Fatalf("r1 Unlock: %v", err)
	}
	if err := TryLock(w1); !errors.Is(err, ErrLockBusy) {
		t.Fatalf("w1 TryLock while r2 still active: want ErrLockBusy, got %v", err)
	}

	// (c) Releasing the final reader permits the exclusive writer.
	if err := Unlock(r2); err != nil {
		t.Fatalf("r2 Unlock: %v", err)
	}
	if err := Lock(w1); err != nil {
		t.Fatalf("w1 Lock after all readers unlocked: %v", err)
	}

	// (d) Shared reader must be refused while exclusive writer holds the lock.
	if err := RLock(r1); !errors.Is(err, ErrLockBusy) {
		t.Fatalf("r1 RLock while writer active: want ErrLockBusy, got %v", err)
	}

	if err := Unlock(w1); err != nil {
		t.Fatalf("w1 Unlock: %v", err)
	}

	// Shared reader succeeds once writer releases.
	if err := RLock(r1); err != nil {
		t.Fatalf("r1 RLock after writer unlocked: %v", err)
	}
	if err := Unlock(r1); err != nil {
		t.Fatalf("r1 Unlock final: %v", err)
	}
}
