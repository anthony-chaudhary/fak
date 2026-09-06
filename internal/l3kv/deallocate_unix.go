//go:build linux

package l3kv

import (
	"errors"
	"os"
	"syscall"

	"golang.org/x/sys/unix"
)

func deallocateFileRangeOS(file *os.File, offset, length int64) error {
	mode := uint32(unix.FALLOC_FL_PUNCH_HOLE | unix.FALLOC_FL_KEEP_SIZE)
	err := unix.Fallocate(int(file.Fd()), mode, offset, length)
	if err == nil {
		return nil
	}

	// Graceful fallback if hole punching is unsupported (e.g. EOPNOTSUPP, ENOSYS, EINVAL on tmpfs/NFS).
	if errors.Is(err, unix.EOPNOTSUPP) || errors.Is(err, unix.ENOTSUP) ||
		errors.Is(err, unix.ENOSYS) || errors.Is(err, unix.EINVAL) ||
		errors.Is(err, syscall.ENOSYS) || errors.Is(err, syscall.EINVAL) {
		return fallbackDeallocate(file, offset, length)
	}

	return fallbackDeallocate(file, offset, length)
}
