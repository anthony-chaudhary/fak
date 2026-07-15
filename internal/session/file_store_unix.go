//go:build !windows

package session

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"syscall"
	"time"
)

func readDescriptorFile(path string) ([]byte, error) { return os.ReadFile(path) }

func lockFile(f *os.File, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if err == nil {
			return nil
		}
		if !errors.Is(err, syscall.EWOULDBLOCK) && !errors.Is(err, syscall.EAGAIN) {
			return err
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("timed out after %s", timeout)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func unlockFile(f *os.File) error { return syscall.Flock(int(f.Fd()), syscall.LOCK_UN) }

func replaceFile(tmpName, path string) error {
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("replace session descriptor file: %w", err)
	}
	fileStoreBoundary("replace")
	dir, err := os.Open(filepath.Dir(path))
	if err != nil {
		return fmt.Errorf("open session descriptor dir for sync: %w", err)
	}
	defer dir.Close()
	if err := dir.Sync(); err != nil {
		return fmt.Errorf("sync session descriptor dir: %w", err)
	}
	fileStoreBoundary("directory-sync")
	return nil
}
