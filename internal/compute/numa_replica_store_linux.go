//go:build linux && amd64

package compute

import (
	"syscall"
	"unsafe"
)

// mpolBind is MPOL_BIND (uapi/linux/mempolicy.h): allocations in the region come strictly from
// the node mask. Sibling of mpolInterleave in decode_interleave_mbind_linux.go.
const mpolBind = 2

// allocNodeRegion reserves n bytes of anonymous memory OFF the Go heap and binds it strictly to
// node, returning the region, its release func, and whether the bind took effect. The order is
// load-bearing: mmap reserves address space WITHOUT faulting pages, then mbind(MPOL_BIND) sets
// the region's policy, so the caller's subsequent copy first-touches every page onto the target
// node. Binding after the copy would instead demand a migration of the whole slab.
//
// A failed mbind is NOT fatal: the region is still valid, byte-correct storage, so the caller
// keeps it and learns bound=false — decode stays correct and merely forfeits the locality claim
// (the same fail-visible discipline ApplyDecodeInterleave uses).
func allocNodeRegion(n int, node int) ([]byte, func() error, bool, error) {
	if n <= 0 {
		return nil, nil, false, errReplicaUnsupported
	}
	region, err := syscall.Mmap(-1, 0, n,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		return nil, nil, false, err
	}
	free := func() error { return syscall.Munmap(region) }
	bound := mbindBindNode(region, node) == nil
	return region, free, bound, nil
}

// mbindBindNode binds one region strictly to a single NUMA node via the raw SYS_MBIND syscall —
// the in-process equivalent of `numactl --membind=<node>` for that buffer. No MPOL_MF_MOVE: this
// runs before the region is faulted, so the policy governs first touch and there is nothing to
// migrate. addr/len are page-aligned as mbind requires (start rounds down, end rounds up).
func mbindBindNode(region []byte, node int) error {
	if len(region) == 0 || node < 0 {
		return errReplicaUnsupported
	}
	page := uintptr(syscall.Getpagesize())
	start := uintptr(unsafe.Pointer(&region[0]))
	end := start + uintptr(len(region))
	aStart := start &^ (page - 1)
	aEnd := (end + page - 1) &^ (page - 1)
	length := aEnd - aStart

	mask := make([]uint64, node/64+1)
	mask[node/64] |= 1 << (uint(node) % 64)
	maxnode := uintptr(len(mask) * 64)

	_, _, errno := syscall.Syscall6(
		syscall.SYS_MBIND,
		aStart, length, mpolBind,
		uintptr(unsafe.Pointer(&mask[0])), maxnode, 0,
	)
	if errno != 0 {
		return errno
	}
	return nil
}
