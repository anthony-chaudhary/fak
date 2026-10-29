//go:build cuda && cgo

package compute

/*
#include <stdlib.h>
#include "cuda_backend.h"
*/
import "C"

import (
	"sync"
	"unsafe"
)

// cudaAllocSiteIDs caches interned site ids so the C string crosses cgo once
// per distinct site, not once per allocation.
var cudaAllocSiteIDs sync.Map // string -> C.int

func cudaAllocSite(site string) C.int {
	if id, ok := cudaAllocSiteIDs.Load(site); ok {
		return id.(C.int)
	}
	cs := C.CString(site)
	id := C.fcuda_alloc_site_intern(cs)
	C.free(unsafe.Pointer(cs))
	cudaAllocSiteIDs.Store(site, id)
	return id
}

// cudaAllocTag returns the ledger class ordinal and site id for an allocation.
func cudaAllocTag(class MemoryClass, site string, promptPanel bool) (C.int, C.int) {
	return C.int(DeviceAllocClassFor(class, site, promptPanel)), cudaAllocSite(site)
}

// DeviceAllocLedger reads the tagged device-allocation ledger and folds in
// cudaMemGetInfo. The ledger read itself takes no lock.
func (c *cudaBackend) DeviceAllocLedger() DeviceAllocSnapshot {
	raw := make([]int64, DeviceAllocCells*5)
	var untracked C.int64_t
	n := int(C.fcuda_alloc_ledger_read((*C.int64_t)(unsafe.Pointer(&raw[0])), C.int(DeviceAllocCells), &untracked))
	if n != DeviceAllocCells {
		raw = raw[:0]
	}
	var sites []DeviceAllocSite
	var name [64]C.char
	var vals [5]C.int64_t
	for i := 0; ; i++ {
		count := int(C.fcuda_alloc_site_read(C.int(i), &name[0], C.int(len(name)), &vals[0]))
		if count < 0 || i >= count {
			break
		}
		sites = append(sites, DeviceAllocSite{
			Site:      C.GoString(&name[0]),
			Class:     DeviceAllocClass(vals[0]).String(),
			LiveBytes: int64(vals[1]),
			PeakBytes: int64(vals[2]),
			LiveCount: int64(vals[3]),
			Allocs:    int64(vals[4]),
		})
	}
	total, free, known := c.DeviceMemory()
	return BuildDeviceAllocSnapshot(raw, int64(untracked), sites, total, free, known && free != FreeUnknown)
}

// ResetDeviceAllocPeaks restarts every ledger peak at its current live bytes.
func (c *cudaBackend) ResetDeviceAllocPeaks() { C.fcuda_alloc_ledger_reset_peak() }
