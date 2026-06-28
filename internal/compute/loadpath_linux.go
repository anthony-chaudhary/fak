//go:build linux

package compute

import "syscall"

// loadPathFSMagic reports the Linux statfs f_type magic for path so ProbeLoadPath can tell a
// network weights mount (NFS/CIFS — the ~50–100x load-time tax) from a local disk (#1062).
// syscall.Statfs_t.Type carries the filesystem magic on Linux; a statfs error fails open
// (known=false) so the caller proceeds without a warning rather than blocking a valid load.
func loadPathFSMagic(path string) (magic int64, known bool) {
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		return 0, false
	}
	return int64(stat.Type), true
}
