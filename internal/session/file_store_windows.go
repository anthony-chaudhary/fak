//go:build windows

package session

import (
	"errors"
	"fmt"
	"os"
	"syscall"
	"time"
	"unsafe"
)

const (
	lockfileExclusiveLock   = 0x00000002
	lockfileFailImmediately = 0x00000001
	movefileReplaceExisting = 0x00000001
	movefileWriteThrough    = 0x00000008
)

var (
	kernel32         = syscall.NewLazyDLL("kernel32.dll")
	procLockFileEx   = kernel32.NewProc("LockFileEx")
	procUnlockFileEx = kernel32.NewProc("UnlockFileEx")
	procMoveFileExW  = kernel32.NewProc("MoveFileExW")
)

func readDescriptorFile(path string) ([]byte, error) {
	deadline := time.Now().Add(descriptorLockTimeout)
	for {
		b, err := os.ReadFile(path)
		if err == nil || errors.Is(err, os.ErrNotExist) {
			return b, err
		}
		var pathErr *os.PathError
		if !errors.As(err, &pathErr) || (!errors.Is(pathErr.Err, syscall.Errno(5)) && !errors.Is(pathErr.Err, syscall.Errno(32))) {
			return nil, err
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("read session descriptor file: timed out after %s: %w", descriptorLockTimeout, err)
		}
		time.Sleep(time.Millisecond)
	}
}

func lockFile(f *os.File, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var ov syscall.Overlapped
	for {
		r, _, callErr := procLockFileEx.Call(f.Fd(), lockfileExclusiveLock|lockfileFailImmediately, 0, 1, 0, uintptr(unsafe.Pointer(&ov)))
		if r != 0 {
			return nil
		}
		if !errors.Is(callErr, syscall.Errno(33)) {
			return callErr
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("timed out after %s", timeout)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func unlockFile(f *os.File) error {
	var ov syscall.Overlapped
	r, _, err := procUnlockFileEx.Call(f.Fd(), 0, 1, 0, uintptr(unsafe.Pointer(&ov)))
	if r == 0 {
		return err
	}
	return nil
}

func replaceFile(tmpName, path string) error {
	src, err := syscall.UTF16PtrFromString(tmpName)
	if err != nil {
		return fmt.Errorf("replace session descriptor file: %w", err)
	}
	dst, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return fmt.Errorf("replace session descriptor file: %w", err)
	}
	deadline := time.Now().Add(descriptorLockTimeout)
	for {
		r, _, callErr := procMoveFileExW.Call(uintptr(unsafe.Pointer(src)), uintptr(unsafe.Pointer(dst)), movefileReplaceExisting|movefileWriteThrough)
		if r != 0 {
			break
		}
		// Go readers do not request delete sharing on Windows. Wait for the
		// bounded read to close rather than removing the destination and
		// exposing a transiently absent registry.
		if !errors.Is(callErr, syscall.Errno(5)) && !errors.Is(callErr, syscall.Errno(32)) {
			return fmt.Errorf("replace session descriptor file: %w", callErr)
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("replace session descriptor file: timed out after %s: %w", descriptorLockTimeout, callErr)
		}
		time.Sleep(time.Millisecond)
	}
	fileStoreBoundary("replace")
	// MOVEFILE_WRITE_THROUGH does not return until publication has reached disk.
	fileStoreBoundary("directory-sync")
	return nil
}
