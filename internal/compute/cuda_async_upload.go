//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import (
	"strconv"
	"unsafe"
)

// cuda_async_upload.go — compute.AsyncUploader on CUDA (spec-prefetch TICKET-02). The copy runs on
// a dedicated non-blocking copy stream through a bounded pinned staging pool
// (cuda/async_upload.cuh), so ops enqueued on g_stream after the issue run underneath it.
//
// Two shapes go async, both byte-verbatim copies: F32 host data uploaded as F32, and raw
// Q4_K/Q5_K/Q6_K/Q2_K bytes uploaded as their own dtype. Each lands in its OWN MemoryOffload
// allocation, never the host-pointer uploadCache, so an expert ring can Free it independently
// (the same ownership UploadClass(..., MemoryOffload, ...) gives the raw k-quant path). Every
// other request, and any request the slot pool cannot take, is the synchronous Upload with an
// already-landed fence, so narrowing (F16, Q8_0, NVFP4, Q2_0) is identical to Upload by
// construction.

var _ ClassedAsyncUploader = (*cudaBackend)(nil)

// cudaUploadFence is one in-flight copy. slot/gen name the pinned slot's event; a slot is reused
// only after its event completed, so a stale generation reads as landed.
type cudaUploadFence struct {
	slot   C.int
	gen    C.uint64_t
	landed bool
}

// cudaLandedFence is the fence of a transfer that already completed synchronously.
var cudaLandedFence Fence = &cudaUploadFence{slot: -1, landed: true}

// cudaAsyncPending maps a destination pointer to its in-flight fence so Free cannot return a
// buffer to the device pool while the copy stream still writes it. Guarded by cudaMu.
var cudaAsyncPending = map[unsafe.Pointer]*cudaUploadFence{}

func (f *cudaUploadFence) Done() bool {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return f.pollLocked(false)
}

func (f *cudaUploadFence) Wait() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	f.pollLocked(true)
}

// pollLocked queries (or, with block, waits for) the copy. Caller holds cudaMu.
func (f *cudaUploadFence) pollLocked(block bool) bool {
	if f.landed {
		return true
	}
	b := C.int(0)
	if block {
		b = 1
	}
	switch st := int(C.fcuda_upload_async_fence(f.slot, f.gen, b)); st {
	case 1:
		f.landed = true
		for ptr, pending := range cudaAsyncPending {
			if pending == f {
				delete(cudaAsyncPending, ptr)
			}
		}
		return true
	case 0:
		return false
	default:
		panic(&CUDAOpError{
			Op: "UploadAsync", Site: "upload-async-fence", Code: st - 1000,
			Msg:   "cuda async upload fence failed (status " + strconv.Itoa(st) + ")",
			Class: MemoryOffload,
		})
	}
}

// awaitAsyncUploadLocked blocks until any in-flight async copy into ptr has landed. Free calls it
// before releasing a buffer. Caller holds cudaMu.
func awaitAsyncUploadLocked(ptr unsafe.Pointer) {
	if f, ok := cudaAsyncPending[ptr]; ok {
		f.pollLocked(true)
		delete(cudaAsyncPending, ptr)
	}
}

// UploadAsync issues t's host->device copy without fencing the caller. The returned Tensor must not
// be read or multiplied before the Fence is satisfied (compute.AsyncUploader contract).
func (c *cudaBackend) UploadAsync(t Tensor, as Dtype) (Tensor, Fence) {
	hb, isHost := t.buf.(HostBuffer)
	if !isHost {
		return c.Upload(t, as), cudaLandedFence
	}
	if out, fence, ok := c.uploadAsyncShape(t, hb, as, MemoryOffload, "upload-async"); ok {
		return out, fence
	}
	return c.Upload(t, as), cudaLandedFence
}

// UploadAsyncClass is UploadAsync with every allocation booked under class at site
// (compute.ClassedAsyncUploader). The async shapes go async only for MemoryOffload, the class
// their independently Freed allocation already models; any other class, every shape the copy
// stream does not take, and a full slot pool are synchronous and return a nil Fence, so a caller's
// overlap meter never books a copy that never left the critical path.
func (c *cudaBackend) UploadAsyncClass(t Tensor, as Dtype, class MemoryClass, site string) (Tensor, Fence) {
	class, site = normalizeUploadClass(class, site, "upload-")
	if hb, isHost := t.buf.(HostBuffer); isHost && class == MemoryOffload {
		if out, fence, ok := c.uploadAsyncShape(t, hb, as, class, site); ok {
			if fence == cudaLandedFence {
				fence = nil
			}
			return out, fence
		}
	}
	return c.UploadClass(t, as, class, site), nil
}

// uploadAsyncShape issues the copy-stream upload for the two byte-verbatim shapes and reports
// ok=false for every other request.
func (c *cudaBackend) uploadAsyncShape(t Tensor, hb HostBuffer, as Dtype, class MemoryClass, site string) (Tensor, Fence, bool) {
	switch {
	case t.Dtype == F32 && as == F32 && len(t.Shape) > 0:
		out, fence := c.uploadF32Async(t, hb, class, site)
		return out, fence, true
	case (t.Dtype == Q3_K || t.Dtype == Q4_K || t.Dtype == Q5_K || t.Dtype == Q6_K || t.Dtype == Q2_K) && as == t.Dtype:
		out, fence := c.uploadRawKQuantAsync(t, hb, class, site)
		return out, fence, true
	}
	return Tensor{}, nil, false
}

func (c *cudaBackend) uploadF32Async(t Tensor, hb HostBuffer, class MemoryClass, site string) (Tensor, Fence) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	f := hb.F32()
	buf := c.dallocClass(t.Numel()*F32.Bytes(), class, site)
	out := makeF32TensorLike(c, t, buf)
	if len(f) == 0 {
		return out, cudaLandedFence
	}
	published := false
	defer func() {
		if !published && buf.ptr != nil {
			ptr := buf.ptr
			buf.ptr = nil
			func() {
				defer func() { _ = recover() }() // preserve the issue failure
				C.fcuda_free(ptr)
			}()
		}
	}()
	fence := c.issueAsyncLocked(buf.ptr, unsafe.Pointer(&f[0]), len(f)*F32.Bytes())
	published = true
	return out, fence
}

func (c *cudaBackend) uploadRawKQuantAsync(t Tensor, hb HostBuffer, class MemoryClass, site string) (Tensor, Fence) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	fence := cudaLandedFence
	out := c.uploadRawKQuantClassWithOperations(t, hb, class, c.uploadRawKQuant,
		func(nbytes int) *cudaBuf { return c.dallocClass(nbytes, class, site) },
		func(dst unsafe.Pointer, raw []int8) {
			fence = c.issueAsyncLocked(dst, unsafe.Pointer(&raw[0]), len(raw))
		},
		func(ptr unsafe.Pointer) { C.fcuda_free(ptr) },
	)
	return out, fence
}

// issueAsyncLocked starts one staged copy, or copies synchronously when the pinned pool cannot
// take it. The host bytes are consumed (memcpy'd into pinned staging) before it returns, so the
// caller's slice need not outlive the fence. Caller holds cudaMu.
func (c *cudaBackend) issueAsyncLocked(dst, src unsafe.Pointer, n int) Fence {
	var slot C.int
	var gen C.uint64_t
	switch st := int(C.fcuda_upload_async(dst, src, C.size_t(n), &slot, &gen)); st {
	case 0:
		f := &cudaUploadFence{slot: slot, gen: gen}
		cudaAsyncPending[dst] = f
		return f
	case 1:
		C.fcuda_h2d(dst, src, C.size_t(n))
		return cudaLandedFence
	default:
		panic(&CUDAOpError{
			Op: "UploadAsync", Site: "upload-async", Code: st - 1000,
			Msg:   "cuda async upload issue failed (status " + strconv.Itoa(st) + ")",
			Class: MemoryOffload,
		})
	}
}

// cudaAsyncUploadStats reports issued copies, synchronous fallbacks, and pinned staging bytes.
func cudaAsyncUploadStats() (issued, fallback uint64, pinned int) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var i, f C.uint64_t
	var p C.size_t
	C.fcuda_upload_async_stats(&i, &f, &p)
	return uint64(i), uint64(f), int(p)
}
