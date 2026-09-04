//go:build linux

package compute

import (
	"os"
	"syscall"

	"golang.org/x/sys/unix"
)

// dropFilePagesOS drops the specified byte range from the Linux kernel page cache
// using the POSIX_FADV_DONTNEED advisory.
func dropFilePagesOS(fd uintptr, offset, length int64) error {
	return unix.Fadvise(int(fd), offset, length, unix.FADV_DONTNEED)
}

// openDirectIOFile attempts to open a file with O_DIRECT on Linux, falling back to standard
// open if the filesystem does not support direct I/O (e.g. tmpfs returning EINVAL).
func openDirectIOFile(name string) (*os.File, error) {
	f, err := os.OpenFile(name, os.O_RDONLY|syscall.O_DIRECT, 0)
	if err == nil {
		return f, nil
	}
	return os.OpenFile(name, os.O_RDONLY, 0)
}
