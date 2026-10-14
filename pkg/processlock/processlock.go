// Package processlock provides a non-blocking process lock on an exact file name.
package processlock

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// ErrBusy reports that another process owns the requested lock.
var ErrBusy = errors.New("processlock: lock busy")

// Try acquires an exclusive lock on name within dir without waiting.
// Only positive OS-level lock contention is reported as ErrBusy; path, safety,
// permission, and I/O failures remain distinct hard errors.
func Try(dir, name string) (func(), error) {
	if filepath.Base(name) != name || name == "." || name == ".." {
		return nil, fmt.Errorf("processlock: invalid file name %q", name)
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, fmt.Errorf("processlock: create directory: %w", err)
	}
	directory, err := os.Lstat(dir)
	if err != nil {
		return nil, fmt.Errorf("processlock: inspect directory: %w", err)
	}
	if !directory.IsDir() || directory.Mode()&os.ModeSymlink != 0 {
		return nil, fmt.Errorf("processlock: unsafe directory %q", dir)
	}
	root, err := os.OpenRoot(dir)
	if err != nil {
		return nil, fmt.Errorf("processlock: open directory: %w", err)
	}
	defer root.Close()
	boundDir, err := root.Stat(".")
	if err != nil {
		return nil, fmt.Errorf("processlock: inspect opened directory: %w", err)
	}
	if !os.SameFile(directory, boundDir) {
		return nil, errors.New("processlock: directory changed during acquisition")
	}
	before, err := root.Lstat(name)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("processlock: inspect file: %w", err)
	}
	if err == nil && !before.Mode().IsRegular() {
		return nil, fmt.Errorf("processlock: unsafe lock file %q", name)
	}
	f, err := openFile(root, name)
	if err != nil {
		return nil, fmt.Errorf("processlock: open file: %w", err)
	}
	info, statErr := f.Stat()
	bound, boundErr := root.Lstat(name)
	if statErr != nil || boundErr != nil || !bound.Mode().IsRegular() || !info.Mode().IsRegular() || !os.SameFile(info, bound) {
		_ = f.Close()
		return nil, fmt.Errorf("processlock: file changed during acquisition: stat=%v bound=%v", statErr, boundErr)
	}
	if err := lockFile(f); err != nil {
		_ = f.Close()
		return nil, err
	}
	var once sync.Once
	return func() {
		once.Do(func() {
			_ = unlockFile(f)
			_ = f.Close()
		})
	}, nil
}
