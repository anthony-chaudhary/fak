//go:build windows

package gateway

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strings"

	"golang.org/x/sys/windows"
)

type controlMutexAcquire struct {
	release chan struct{}
	done    chan error
	err     error
}

func acquireControlJournalOwnership(path, resolvedPath string) (*controlJournalOwnership, error) {
	journal, err := os.OpenFile(resolvedPath, os.O_RDWR|os.O_APPEND, 0)
	if err != nil {
		return nil, fmt.Errorf("control ingress: open journal for ownership: %w", err)
	}
	info, err := journal.Stat()
	if err != nil {
		_ = journal.Close()
		return nil, fmt.Errorf("control ingress: stat owned journal: %w", err)
	}

	var fileInfo windows.ByHandleFileInformation
	if err := windows.GetFileInformationByHandle(windows.Handle(journal.Fd()), &fileInfo); err != nil {
		_ = journal.Close()
		return nil, fmt.Errorf("control ingress: identify journal: %w", err)
	}
	identity := fmt.Sprintf("%08x:%08x%08x", fileInfo.VolumeSerialNumber, fileInfo.FileIndexHigh, fileInfo.FileIndexLow)
	keys := []string{
		controlIngressMutexName("path:" + strings.ToLower(path)),
		controlIngressMutexName("identity:" + identity),
	}
	sort.Strings(keys)
	mutex := acquireControlIngressMutexes(keys)
	if mutex.err != nil {
		_ = journal.Close()
		return nil, mutex.err
	}

	return &controlJournalOwnership{
		file: journal,
		info: info,
		release: func() error {
			close(mutex.release)
			return <-mutex.done
		},
	}, nil
}

func controlIngressMutexName(key string) string {
	sum := sha256.Sum256([]byte(key))
	return `Global\FAK.ControlIngress.` + hex.EncodeToString(sum[:])
}

func acquireControlIngressMutexes(names []string) controlMutexAcquire {
	acquired := make(chan controlMutexAcquire, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()

		handles := make([]windows.Handle, 0, len(names))
		for _, name := range names {
			nameUTF16, err := windows.UTF16PtrFromString(name)
			if err != nil {
				releaseControlIngressMutexes(handles)
				acquired <- controlMutexAcquire{err: fmt.Errorf("control ingress: encode writer mutex name: %w", err)}
				return
			}
			handle, err := windows.CreateMutex(nil, false, nameUTF16)
			// CreateMutex returns a usable handle together with
			// ERROR_ALREADY_EXISTS when another process created the named
			// object first. Contention is decided by the zero-time wait.
			if err != nil && !errors.Is(err, windows.ERROR_ALREADY_EXISTS) {
				releaseControlIngressMutexes(handles)
				acquired <- controlMutexAcquire{err: fmt.Errorf("control ingress: create writer mutex: %w", err)}
				return
			}
			wait, waitErr := windows.WaitForSingleObject(handle, 0)
			if waitErr != nil || (wait != windows.WAIT_OBJECT_0 && wait != windows.WAIT_ABANDONED) {
				_ = windows.CloseHandle(handle)
				releaseControlIngressMutexes(handles)
				if wait == uint32(windows.WAIT_TIMEOUT) {
					acquired <- controlMutexAcquire{err: errControlJournalOwned}
				} else if waitErr != nil {
					acquired <- controlMutexAcquire{err: fmt.Errorf("control ingress: wait for writer mutex: %w", waitErr)}
				} else {
					acquired <- controlMutexAcquire{err: fmt.Errorf("control ingress: unexpected writer mutex wait result %d", wait)}
				}
				return
			}
			handles = append(handles, handle)
		}

		release := make(chan struct{})
		done := make(chan error, 1)
		acquired <- controlMutexAcquire{release: release, done: done}
		<-release
		done <- releaseControlIngressMutexes(handles)
	}()
	return <-acquired
}

func releaseControlIngressMutexes(handles []windows.Handle) error {
	var firstErr error
	for i := len(handles) - 1; i >= 0; i-- {
		if err := windows.ReleaseMutex(handles[i]); firstErr == nil && err != nil {
			firstErr = err
		}
		if err := windows.CloseHandle(handles[i]); firstErr == nil && err != nil {
			firstErr = err
		}
	}
	return firstErr
}
