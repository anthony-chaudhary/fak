//go:build windows

package gpulease

import (
	"errors"
	"os"
	"syscall"
	"unsafe"
)

const (
	lockfileFailImmediately = 0x00000001
	lockfileExclusiveLock   = 0x00000002
	errSharingViolation     = syscall.Errno(32)
	errLockViolation        = syscall.Errno(33)
)

var (
	kernel32         = syscall.NewLazyDLL("kernel32.dll")
	procLockFileEx   = kernel32.NewProc("LockFileEx")
	procUnlockFileEx = kernel32.NewProc("UnlockFileEx")
)

func tryLock(f *os.File, _ string) error {
	var ol syscall.Overlapped
	r, _, err := procLockFileEx.Call(
		f.Fd(),
		lockfileExclusiveLock|lockfileFailImmediately,
		0,
		1,
		0,
		uintptr(unsafe.Pointer(&ol)),
	)
	if r != 0 {
		return nil
	}
	if errors.Is(err, errLockViolation) || errors.Is(err, errSharingViolation) {
		return errLockBusy
	}
	return err
}

func unlock(f *os.File, _ string) error {
	var ol syscall.Overlapped
	r, _, err := procUnlockFileEx.Call(
		f.Fd(),
		0,
		1,
		0,
		uintptr(unsafe.Pointer(&ol)),
	)
	if r != 0 {
		return nil
	}
	return err
}
