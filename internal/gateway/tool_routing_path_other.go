//go:build !windows

package gateway

import "path/filepath"

func canonicalExistingWorkspacePath(path string) (string, error) {
	return filepath.EvalSymlinks(path)
}
