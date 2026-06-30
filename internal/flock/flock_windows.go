//go:build windows

package flock

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
	// Keep the advisory byte out of the file's metadata prefix. Windows byte-range
	// locks are mandatory for reads/writes in that range, and gpulease records the
	// holder pid at offset 0 while the lease is held.
	lockOffsetLow = 1 << 30
)

var (
	kernel32         = syscall.NewLazyDLL("kernel32.dll")
	procLockFileEx   = kernel32.NewProc("LockFileEx")
	procUnlockFileEx = kernel32.NewProc("UnlockFileEx")
)

// TryLock takes a non-blocking exclusive advisory lock on f. It returns
// ErrLockBusy when another holder owns the lock, nil on success.
func TryLock(f *os.File) error {
	ol := syscall.Overlapped{Offset: lockOffsetLow}
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
		return ErrLockBusy
	}
	return err
}

// Unlock releases the advisory lock held on f.
func Unlock(f *os.File) error {
	ol := syscall.Overlapped{Offset: lockOffsetLow}
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
