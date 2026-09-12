//go:build windows

package codetools

import (
	"errors"
	"os"
	"syscall"
	"time"
)

func renameReplacing(oldPath, newPath string) error {
	return renameReplacingWith(oldPath, newPath, os.Rename, time.Sleep)
}

// renameReplacingWith retries only the two Windows errors that can represent a
// transient foreign handle opened without delete sharing. The dependency seam
// keeps the retry schedule deterministic in tests without mutable package state.
func renameReplacingWith(oldPath, newPath string, rename func(string, string) error, sleep func(time.Duration)) error {
	backoffs := [...]time.Duration{0, time.Millisecond, 2 * time.Millisecond, 4 * time.Millisecond, 8 * time.Millisecond, 16 * time.Millisecond}
	var err error
	for _, backoff := range backoffs {
		if backoff > 0 {
			sleep(backoff)
		}
		err = rename(oldPath, newPath)
		if err == nil || !transientWindowsRenameError(err) {
			return err
		}
	}
	return err
}

func transientWindowsRenameError(err error) bool {
	return errors.Is(err, syscall.Errno(5)) || // ERROR_ACCESS_DENIED
		errors.Is(err, syscall.Errno(32)) // ERROR_SHARING_VIOLATION
}
