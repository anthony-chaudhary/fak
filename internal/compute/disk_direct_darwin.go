//go:build darwin

package compute

import (
	"os"
	"syscall"
)

const fcntlFNOCACHE = 48

// dropFilePagesOS disables the unified buffer cache (UBC) caching for the descriptor on Darwin (macOS)
// using fcntl F_NOCACHE.
func dropFilePagesOS(fd uintptr, offset, length int64) error {
	_, _, errno := syscall.Syscall(syscall.SYS_FCNTL, fd, fcntlFNOCACHE, 1)
	if errno != 0 {
		return errno
	}
	return nil
}

func openDirectIOFile(name string) (*os.File, error) {
	return os.OpenFile(name, os.O_RDONLY, 0)
}
