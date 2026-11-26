//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

var _ MappedHostWeightUploader = (*cudaBackend)(nil)

// UploadMappedHostWeight publishes an immutable packed weight only after its
// complete payload is copied into one mapped pinned-host allocation.
func (c *cudaBackend) UploadMappedHostWeight(t Tensor, as Dtype, maxBytes int64) (Tensor, bool, error) {
	if as != t.Dtype || (!cudaRawUploadIsVerbatim(t.Dtype) && t.Dtype != NVFP4) {
		return Tensor{}, false, fmt.Errorf("cuda mapped-host weight: %s is not a supported immutable packed dtype", t.Dtype)
	}
	hb, ok := t.buf.(HostBuffer)
	if !ok {
		return Tensor{}, false, fmt.Errorf("cuda mapped-host weight: source is not host-backed")
	}
	var first, second []byte
	if t.Dtype == NVFP4 {
		packed, scales, _ := nvfp4WeightPayload(t)
		first, second = packed, scales
	} else {
		raw := hb.I8()
		if len(raw) > 0 {
			first = unsafe.Slice((*byte)(unsafe.Pointer(&raw[0])), len(raw))
		}
	}
	nbytes := len(first) + len(second)
	if nbytes == 0 || maxBytes <= 0 || int64(nbytes) > maxBytes {
		return Tensor{}, false, nil
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var host, device unsafe.Pointer
	if status := int(C.fcuda_mapped_host_alloc(C.size_t(nbytes), &host, &device)); status != 0 {
		if host != nil {
			if cleanup := int(C.fcuda_mapped_host_free(host)); cleanup != 0 {
				panic(fmt.Sprintf("cuda mapped-host weight allocation cleanup: cuda status %d (allocation status %d)", cleanup, status))
			}
		}
		return Tensor{}, false, nil
	}
	published := false
	defer func() {
		if !published && host != nil {
			if status := int(C.fcuda_mapped_host_free(host)); status != 0 {
				panic(fmt.Sprintf("cuda mapped-host weight rollback: cuda status %d", status))
			}
		}
	}()
	dst := unsafe.Slice((*byte)(host), nbytes)
	copy(dst, first)
	copy(dst[len(first):], second)
	runtime.KeepAlive(hb)
	buf := &cudaBuf{ptr: device, n: nbytes, class: MemoryWeights, mappedHost: host}
	var out Tensor
	if as == NVFP4 {
		out = makeTensor(c, NVFP4, RowMajor, append([]int(nil), t.Shape...), cloneQuantSpec(t.Quant), buf)
	} else {
		out, _ = c.devRawKQuantWithAllocator(as, t.Shape, nbytes, func(int) *cudaBuf { return buf })
	}
	published = true
	return out, true, nil
}
