//go:build windows

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"path/filepath"
	"runtime"
	"strings"

	"golang.org/x/sys/windows"
)

type cronExecutionMutexAcquire struct {
	release chan struct{}
	done    chan error
	busy    bool
	err     error
}

// cronRunExecutionLock uses a named Windows mutex rather than LockFileEx. The
// kernel abandons a mutex when its owner dies, so a crashed child runner cannot
// leave an orphaned byte-range lock that requires deleting a live lockfile.
func cronRunExecutionLock(path string) (release func() error, busy bool, err error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, false, fmt.Errorf("execution lock %s: %w", path, err)
	}
	sum := sha256.Sum256([]byte(strings.ToLower(filepath.Clean(abs))))
	name := `Global\FAK.CronExecution.` + hex.EncodeToString(sum[:])
	nameUTF16, err := windows.UTF16PtrFromString(name)
	if err != nil {
		return nil, false, fmt.Errorf("execution lock %s: %w", path, err)
	}
	acquired := make(chan cronExecutionMutexAcquire, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		handle, createErr := windows.CreateMutex(nil, false, nameUTF16)
		if createErr != nil && !errors.Is(createErr, windows.ERROR_ALREADY_EXISTS) {
			if handle != 0 {
				_ = windows.CloseHandle(handle)
			}
			acquired <- cronExecutionMutexAcquire{err: fmt.Errorf("create execution mutex: %w", createErr)}
			return
		}
		wait, waitErr := windows.WaitForSingleObject(handle, 0)
		if waitErr != nil || (wait != windows.WAIT_OBJECT_0 && wait != windows.WAIT_ABANDONED) {
			_ = windows.CloseHandle(handle)
			if wait == uint32(windows.WAIT_TIMEOUT) {
				acquired <- cronExecutionMutexAcquire{busy: true}
			} else if waitErr != nil {
				acquired <- cronExecutionMutexAcquire{err: fmt.Errorf("wait for execution mutex: %w", waitErr)}
			} else {
				acquired <- cronExecutionMutexAcquire{err: fmt.Errorf("unexpected execution mutex wait result %d", wait)}
			}
			return
		}
		release := make(chan struct{})
		done := make(chan error, 1)
		acquired <- cronExecutionMutexAcquire{release: release, done: done}
		<-release
		releaseErr := windows.ReleaseMutex(handle)
		closeErr := windows.CloseHandle(handle)
		if releaseErr != nil {
			done <- releaseErr
		} else {
			done <- closeErr
		}
	}()
	result := <-acquired
	if result.err != nil || result.busy {
		return nil, result.busy, result.err
	}
	return func() error {
		close(result.release)
		return <-result.done
	}, false, nil
}
