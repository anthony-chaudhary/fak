//go:build linux && amd64

package compute

import (
	"syscall"
	"unsafe"
)

// Linux mempolicy constants (uapi/linux/mempolicy.h). Kept local — stdlib syscall does not
// export them, and the design constraint (numa-decode-replication-design.md) is stdlib-only.
const (
	mpolInterleave = 3      // MPOL_INTERLEAVE: stripe faults round-robin across the node mask
	mpolMFMove     = 1 << 1 // MPOL_MF_MOVE (2): migrate already-faulted pages onto the policy
)

// mbindInterleave binds one weight region to MPOL_INTERLEAVE across nodes via the raw
// SYS_MBIND syscall — the in-process equivalent of `numactl --interleave=all` for that
// buffer, applied by address so it does not depend on which OS thread faulted the pages.
// MPOL_MF_MOVE migrates pages the loader already first-touched onto node 0, so this restripes
// weights that are resident before it runs. addr/len are page-aligned (mbind requires it):
// the start rounds down and the end rounds up so the region is fully covered.
func mbindInterleave(region []byte, nodes []int) error {
	if len(region) == 0 || len(nodes) == 0 {
		return nil
	}
	page := uintptr(syscall.Getpagesize())
	start := uintptr(unsafe.Pointer(&region[0]))
	end := start + uintptr(len(region))
	aStart := start &^ (page - 1)
	aEnd := (end + page - 1) &^ (page - 1)
	length := aEnd - aStart

	maxNode := 0
	for _, n := range nodes {
		if n > maxNode {
			maxNode = n
		}
	}
	mask := make([]uint64, maxNode/64+1)
	for _, n := range nodes {
		if n >= 0 {
			mask[n/64] |= 1 << (uint(n) % 64)
		}
	}
	// maxnode is the number of bits the kernel scans in the mask; a full word count covers
	// every set bit without the off-by-one traps of passing the highest node id.
	maxnode := uintptr(len(mask) * 64)

	_, _, errno := syscall.Syscall6(
		syscall.SYS_MBIND,
		aStart, length, mpolInterleave,
		uintptr(unsafe.Pointer(&mask[0])), maxnode, mpolMFMove,
	)
	if errno != 0 {
		return errno
	}
	return nil
}
