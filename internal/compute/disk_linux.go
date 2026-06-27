//go:build linux

package compute

import (
	"syscall"
)

// diskInfo reports disk total/free bytes for path on Linux using statfs.
func diskInfo(path string) (total, free int64, known bool) {
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		return 0, FreeUnknown, false
	}
	if stat.Blocks == 0 || stat.Bsize == 0 {
		return 0, FreeUnknown, false
	}
	total = uint64ToCapInt64(uint64(stat.Blocks) * uint64(stat.Bsize))
	free = uint64ToCapInt64(uint64(stat.Bfree) * uint64(stat.Bsize))
	return total, free, total > 0
}