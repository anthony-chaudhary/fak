//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import "unsafe"

// uploadRawKQuantClass keeps immutable weights on the existing shared cache,
// but gives an offload caller its own allocation. Independent expert rings may
// Free their residents without invalidating another ring's handle.
// Caller holds cudaMu, including the synchronous copy and ownership transfer.
func (c *cudaBackend) uploadRawKQuantClass(t Tensor, hb HostBuffer, class MemoryClass, site string) Tensor {
	return c.uploadRawKQuantClassWithOperations(t, hb, class, c.uploadRawKQuant,
		func(nbytes int) *cudaBuf { return c.dallocClass(nbytes, MemoryOffload, site) },
		func(dst unsafe.Pointer, raw []int8) {
			C.fcuda_h2d(dst, unsafe.Pointer(&raw[0]), C.size_t(len(raw)))
		},
		func(ptr unsafe.Pointer) { C.fcuda_free(ptr) },
	)
}

// The operations seam exercises the production ownership path without a CUDA
// device. Offload storage never enters uploadCache, retains no host-cache key,
// and carries no immutable-weight placement or transfer charges. Other classes
// keep their existing behavior through the unchanged weight upload method.
func (c *cudaBackend) uploadRawKQuantClassWithOperations(t Tensor, hb HostBuffer, class MemoryClass,
	uploadWeight func(Tensor, HostBuffer) Tensor,
	allocate func(int) *cudaBuf,
	copyToDevice func(unsafe.Pointer, []int8),
	release func(unsafe.Pointer),
) Tensor {
	if class != MemoryOffload {
		return uploadWeight(t, hb)
	}
	var owner *cudaBuf
	published := false
	defer func() {
		if !published && owner != nil && owner.ptr != nil {
			ptr := owner.ptr
			owner.ptr = nil
			// Preserve the original failure even if cleanup itself panics.
			func() {
				defer func() { _ = recover() }()
				release(ptr)
			}()
		}
	}()
	raw := hb.I8()
	out, _ := c.devRawKQuantWithAllocator(t.Dtype, t.Shape, len(raw), func(nbytes int) *cudaBuf {
		owner = allocate(nbytes)
		return owner
	})
	if len(raw) > 0 {
		copyToDevice(owner.ptr, raw)
	}
	published = true
	return out
}
