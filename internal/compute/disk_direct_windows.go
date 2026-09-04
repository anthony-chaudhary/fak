//go:build windows

package compute

import "os"

// dropFilePagesOS on Windows provides a safe fallback (no-op) because Windows does not have
// posix_fadvise.
func dropFilePagesOS(fd uintptr, offset, length int64) error {
	return nil
}

func openDirectIOFile(name string) (*os.File, error) {
	return os.OpenFile(name, os.O_RDONLY, 0)
}
