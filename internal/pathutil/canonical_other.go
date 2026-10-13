//go:build !windows

package pathutil

import "path/filepath"

// CanonicalExisting resolves an existing filesystem path to its physical location.
func CanonicalExisting(path string) (string, error) {
	return filepath.EvalSymlinks(filepath.Clean(path))
}
