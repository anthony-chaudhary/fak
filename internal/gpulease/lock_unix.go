//go:build !windows

package gpulease

import (
	"errors"
	"os"
	"syscall"
)

func tryLock(f *os.File, _ string) error {
	err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
	if errors.Is(err, syscall.EWOULDBLOCK) {
		return errLockBusy
	}
	return err
}

func unlock(f *os.File, _ string) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
}
