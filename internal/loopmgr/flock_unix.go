//go:build !windows

package loopmgr

// Cross-process advisory lock for the loop ledger's append critical section. See
// flock_windows.go for the rationale; this mirrors internal/gpulease/lock_unix.go:
// a non-blocking flock(LOCK_EX) on a sidecar fd, so the caller polls.

import (
	"errors"
	"os"
	"syscall"
)

func tryFlock(f *os.File) error {
	err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
	if errors.Is(err, syscall.EWOULDBLOCK) {
		return errLockBusy
	}
	return err
}

func unflock(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
}
