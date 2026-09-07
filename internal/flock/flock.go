// Package flock is a cross-platform, non-blocking advisory file lock on an open
// *os.File. It is the single home for the LockFileEx (windows) / flock(LOCK_EX)
// (unix) primitive that gpulease (GPU lease) and loopmgr (loop-ledger append
// critical section) each used to copy verbatim.
//
// The lock is non-blocking: TryLock returns ErrLockBusy when another holder owns
// the file, so the caller polls. The OS drops the lock when the fd is closed or the
// holder exits cleanly. NOTE: this primitive does NOT by itself guarantee release on
// an abnormal Windows termination — a killed/crashed holder's LockFileEx region can
// stay orphaned on the path. Callers that must survive that (e.g. safecommit's
// fak-commit.lock, which guards the shared trunk) layer a stale-lock reap on top, keyed
// on a pid recorded in the file: if that pid is dead, the lockfile is removed before the
// next acquire. flock stays a pure primitive.
package flock

import (
	"errors"
	"os"
)

// ErrLockBusy is the sentinel TryLock returns when the file is already locked by
// another holder. Callers test for it with errors.Is to distinguish a contended
// lock (retry/poll) from a real I/O failure.
var ErrLockBusy = errors.New("flock: lock busy")

// Lock is an alias for TryLock, providing exclusive write-locking semantics.
func Lock(f *os.File) error {
	return TryLock(f)
}

// LockShared is an alias for TryLockShared, acquiring a non-blocking shared read lock.
func LockShared(f *os.File) error {
	return TryLockShared(f)
}

// RLock is an alias for TryLockShared for callers preferring read-lock terminology.
func RLock(f *os.File) error {
	return TryLockShared(f)
}
