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
