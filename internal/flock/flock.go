// Package flock is a cross-platform, non-blocking advisory file lock on an open
// *os.File. It is the single home for the LockFileEx (windows) / flock(LOCK_EX)
// (unix) primitive that gpulease (GPU lease) and loopmgr (loop-ledger append
// critical section) each used to copy verbatim.
//
// The lock is non-blocking: TryLock returns ErrLockBusy when another holder owns
// the file, so the caller polls. The OS drops the lock when the fd is closed or
// the holder process dies, so a crashed peer never wedges the lane.
package flock

import "errors"

// ErrLockBusy is the sentinel TryLock returns when the file is already locked by
// another holder. Callers test for it with errors.Is to distinguish a contended
// lock (retry/poll) from a real I/O failure.
var ErrLockBusy = errors.New("flock: lock busy")
