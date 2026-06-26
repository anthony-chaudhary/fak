//go:build windows

package loopmgr

// Cross-process advisory lock for the loop ledger's append critical section, so
// that two unrelated `fak loop ...` processes cannot both read the same tail seq
// and append a forked chain (two events sharing one seq + prev_hash). It mirrors
// internal/gpulease/lock_windows.go: a non-blocking LockFileEx on a sidecar fd, so
// the caller polls. The OS drops the lock when the fd closes (or the holder dies).

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

func tryFlock(f *os.File) error {
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

func unflock(f *os.File) error {
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
