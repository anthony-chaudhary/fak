// Package parentdir prepares parent directories for file writes.
package parentdir

import (
	"os"
	"path/filepath"
)

// Dir returns path's parent directory.
func Dir(path string) string { return filepath.Dir(path) }

// Ensure creates path's non-trivial parent directory.
func Ensure(path string, mode os.FileMode) error {
	dir := filepath.Dir(path)
	if dir == "." || dir == "" {
		return nil
	}
	return os.MkdirAll(dir, mode)
}
