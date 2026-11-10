// Package fsatomic provides the single shared atomic file writer for the
// private factory: a sibling temporary file that is written, synced, chmod'd
// and renamed over the destination, so a crash or a failed write can never
// leave a truncated or partially written file in place.
//
// It replaces the divergent copies of this pattern that had drifted across
// platform/ and cmd/ (fak-private#2785): unique temp names, the caller's
// permission honored, parent creation, an fsync before the rename, and one
// Windows replace strategy.
package fsatomic

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

// WriteFileAtomic writes data to path through a sibling temporary file that is
// fsynced and then renamed over the destination. Readers see either the
// previous contents or the complete new contents, never a mix.
//
// The temporary file is created in the destination directory (same filesystem,
// so the rename is atomic) with a unique name, written, synced, chmod'd to perm
// and closed before the rename. Any failure removes the temporary file and
// leaves the destination untouched.
func WriteFileAtomic(path string, data []byte, perm os.FileMode) error {
	return WriteFileAtomicWithRename(path, data, perm, defaultRename)
}

// WriteFileAtomicWithRename is WriteFileAtomic with an injectable rename, so a
// caller (or a test) can own the final replacement step. A nil rename uses the
// default strategy.
func WriteFileAtomicWithRename(path string, data []byte, perm os.FileMode, rename func(string, string) error) error {
	if rename == nil {
		rename = defaultRename
	}

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("fsatomic: create directory %s: %w", dir, err)
	}

	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return fmt.Errorf("fsatomic: create temporary file for %s: %w", path, err)
	}
	tmpName := tmp.Name()

	cleanup := true
	defer func() {
		if cleanup {
			_ = tmp.Close()
			_ = os.Remove(tmpName)
		}
	}()

	if _, err := tmp.Write(data); err != nil {
		return fmt.Errorf("fsatomic: write temporary file for %s: %w", path, err)
	}
	if err := tmp.Sync(); err != nil {
		return fmt.Errorf("fsatomic: sync temporary file for %s: %w", path, err)
	}
	if err := tmp.Chmod(perm); err != nil {
		return fmt.Errorf("fsatomic: chmod temporary file for %s: %w", path, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("fsatomic: close temporary file for %s: %w", path, err)
	}

	if err := rename(tmpName, path); err != nil {
		return fmt.Errorf("fsatomic: replace %s: %w", path, err)
	}

	cleanup = false
	return nil
}

// defaultRename replaces to with from. On Windows a reader that holds the
// destination open without FILE_SHARE_DELETE (another Go process, an AV
// scanner, the search indexer, a sync client) refuses the replace with "Access
// is denied" until it closes, so a transient refusal is retried within a short
// bounded budget. If the destination still refuses (read-only), drop it and
// retry once. POSIX rename is already atomic and never needs this path.
func defaultRename(from, to string) error {
	if runtime.GOOS != "windows" {
		return os.Rename(from, to)
	}
	err := renameRetrying(from, to, os.Rename, isTransientRenameError, time.Sleep)
	if err == nil {
		return nil
	}
	if rmErr := os.Remove(to); rmErr != nil && !os.IsNotExist(rmErr) {
		return err
	}
	return os.Rename(from, to)
}

// renameAttempts bounds renameRetrying; with the capped exponential backoff the
// worst case waits about 200ms before giving up.
const renameAttempts = 8

// renameRetrying calls rename until it succeeds, fails with an error
// isTransient rejects, or renameAttempts calls are spent.
func renameRetrying(from, to string, rename func(string, string) error, isTransient func(error) bool, sleep func(time.Duration)) error {
	backoff := 2 * time.Millisecond
	var err error
	for attempt := 1; ; attempt++ {
		if err = rename(from, to); err == nil || !isTransient(err) || attempt == renameAttempts {
			return err
		}
		sleep(backoff)
		backoff = min(2*backoff, 50*time.Millisecond)
	}
}
