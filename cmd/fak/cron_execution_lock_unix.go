//go:build !windows

package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

// cronRunExecutionLock holds a kernel file lock for the entire child lifetime.
// Unix releases flock when the process exits, including abnormal termination.
func cronRunExecutionLock(path string) (release func() error, busy bool, err error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, false, fmt.Errorf("execution lock %s: %w", path, err)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, false, fmt.Errorf("execution lock %s: %w", path, err)
	}
	if err := flock.TryLock(f); err != nil {
		_ = f.Close()
		if errors.Is(err, flock.ErrLockBusy) {
			return nil, true, nil
		}
		return nil, false, fmt.Errorf("execution lock %s: %w", path, err)
	}
	if err := f.Truncate(0); err == nil {
		_, err = f.Seek(0, io.SeekStart)
	}
	if err == nil {
		_, err = fmt.Fprintf(f, "%d\n", os.Getpid())
	}
	if err == nil {
		err = f.Sync()
	}
	if err != nil {
		_ = flock.Unlock(f)
		_ = f.Close()
		return nil, false, fmt.Errorf("record execution lock owner %s: %w", path, err)
	}
	return func() error {
		unlockErr := flock.Unlock(f)
		closeErr := f.Close()
		if unlockErr != nil {
			return unlockErr
		}
		return closeErr
	}, false, nil
}
