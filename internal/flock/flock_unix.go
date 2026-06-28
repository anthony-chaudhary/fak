//go:build !windows

package flock

import (
	"errors"
	"os"
	"syscall"
)

// TryLock takes a non-blocking exclusive advisory lock on f. It returns
// ErrLockBusy when another holder owns the lock, nil on success.
func TryLock(f *os.File) error {
	err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
	if errors.Is(err, syscall.EWOULDBLOCK) {
		return ErrLockBusy
	}
	return err
}

// Unlock releases the advisory lock held on f.
func Unlock(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
}
