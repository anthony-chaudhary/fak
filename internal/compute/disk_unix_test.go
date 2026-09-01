//go:build darwin || linux

package compute

import (
	"syscall"
	"testing"
)

func TestDiskInfoReportsCallerAvailableBytes(t *testing.T) {
	path := t.TempDir()
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		t.Fatalf("statfs: %v", err)
	}
	if stat.Blocks == 0 || stat.Bsize == 0 {
		t.Skip("filesystem does not report block capacity")
	}

	total, available, known := DiskInfo(path)
	if !known {
		t.Fatal("DiskInfo returned unknown for an accessible temporary directory")
	}
	wantTotal := uint64ToCapInt64(uint64(stat.Blocks) * uint64(stat.Bsize))
	wantAvailable := uint64ToCapInt64(uint64(stat.Bavail) * uint64(stat.Bsize))
	if total != wantTotal || available != wantAvailable {
		t.Fatalf("DiskInfo = total %d available %d, want total %d available %d", total, available, wantTotal, wantAvailable)
	}
	if stat.Bfree != stat.Bavail {
		reservedInclusive := uint64ToCapInt64(uint64(stat.Bfree) * uint64(stat.Bsize))
		if available == reservedInclusive {
			t.Fatalf("DiskInfo exposed Bfree %d instead of caller-available Bavail %d", reservedInclusive, wantAvailable)
		}
	}
}
