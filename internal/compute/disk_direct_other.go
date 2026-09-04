//go:build !linux && !darwin && !windows

package compute

import "os"

// dropFilePagesOS is the safe no-op fallback on other platforms.
func dropFilePagesOS(fd uintptr, offset, length int64) error {
	return nil
}

func openDirectIOFile(name string) (*os.File, error) {
	return os.OpenFile(name, os.O_RDONLY, 0)
}
