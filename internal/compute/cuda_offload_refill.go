//go:build cuda && cgo

package compute

/*
#include <stddef.h>
extern int fcuda_offload_refill_probe(void *d, const void *h, size_t n);
extern int fcuda_offload_refill_copy(void *d, const void *h, size_t n);
extern void fcuda_offload_refill_stop(void *d);
*/
import "C"

import (
	"fmt"
	"os"
	"runtime"
	"slices"
	"sync/atomic"
	"unsafe"
)

// Geometry-slot/completion adaptation of vLLM fa3b9d1377219247060d3c8b49eae199c9e7e154.
// See third_party/vllm-offloader-refill. Eager only, default off, no graph admission.
func (c *cudaBackend) OffloadRefillEnabled() bool {
	return c != nil && c == cudaDev && os.Getenv("FAK_CUDA_EXPERT_RING_REFILL") == "1"
}

func (c *cudaBackend) CanRefillOffload(destination, source Tensor, bytes int64) bool {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if !c.OffloadRefillEnabled() {
		return false
	}
	b, p, ok := c.offloadRefillOperands(destination, source, bytes)
	if !ok {
		return false
	}
	accepted := C.fcuda_offload_refill_probe(b.ptr, p, C.size_t(bytes)) == 1
	runtime.KeepAlive(source)
	return accepted
}

func (c *cudaBackend) RefillOffload(destination, source Tensor, bytes int64) (Tensor, OwnedTransferResult) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return c.offloadRefillWithOperations(destination, source, bytes, c.OffloadRefillEnabled(),
		func(d, h unsafe.Pointer, n int) bool { return C.fcuda_offload_refill_probe(d, h, C.size_t(n)) == 1 },
		func(d, h unsafe.Pointer, n int) int { return int(C.fcuda_offload_refill_copy(d, h, C.size_t(n))) },
		func(d unsafe.Pointer) { C.fcuda_offload_refill_stop(d) })
}

// BEGIN offload refill owner core.
// One in-flight or permanently quarantined attempt under cudaMu. Native pinned
// source/device retention is independent of these Go roots and the model ring.
type cudaOffloadRefillAttempt struct {
	source, destination, replacement Tensor
	pointer                          unsafe.Pointer
	failure                          error
}

var cudaOffloadRefillActive *cudaOffloadRefillAttempt

func cudaOffloadRefillBytes(t Tensor) (int, bool) {
	if len(t.Shape) != 2 || t.Shape[0] <= 0 || t.Shape[1] <= 0 || t.Layout != RowMajor {
		return 0, false
	}
	const capBytes = 64 << 20
	rows, cols := t.Shape[0], t.Shape[1]
	if t.Dtype == F32 {
		if t.Quant != nil || cols > capBytes/4 || rows > capBytes/(cols*4) {
			return 0, false
		}
		return rows * cols * 4, true
	}
	if !cudaRawUploadIsVerbatim(t.Dtype) || cols%256 != 0 || t.Quant == nil {
		return 0, false
	}
	q := t.Quant
	if q.Block != 256 || q.Axis != 2 || q.Symmetric || q.StaticAct ||
		len(q.Scale) != 0 || len(q.ZeroPoint) != 0 {
		return 0, false
	}
	block, bits := 0, uint8(4)
	switch t.Dtype {
	case Q2_K:
		block, bits = 84, 2
	case Q3_K:
		block, bits = 110, 3
	case Q4_K:
		block = 144
	case Q5_K:
		block, bits = 176, 5
	case Q6_K:
		block, bits = 210, 6
	case IQ4_NL, IQ4_XS:
		var ok bool
		block, ok = RawIQBlockBytes(t.Dtype)
		if !ok {
			return 0, false
		}
	default:
		// Initial IQ arm is IQ4 only. IQ2/IQ1 constructor metadata has multiple
		// existing conventions; do not infer a new source/device descriptor law.
		return 0, false
	}
	if q.Bits != bits || block <= 0 || cols/256 > capBytes/block {
		return 0, false
	}
	rowBytes := cols / 256 * block
	if rows > capBytes/rowBytes {
		return 0, false
	}
	return rows * rowBytes, true
}

// Pure shape/owner admission, no callbacks on arbitrary HostBuffer values.
// Native probe must independently establish exact live allocation/ledger and
// shared-stream health. These checks do not establish an outer execution lease.
func (c *cudaBackend) offloadRefillOperands(dst, src Tensor, bytes int64) (*cudaBuf, unsafe.Pointer, bool) {
	if c == nil || c != cudaDev || cudaOffloadRefillActive != nil || graphEnabled ||
		c.graphOnce != nil || c.capturing || c.faultLatch.Health().Refusing() ||
		dst.Backend() != c || dst.Dtype != src.Dtype || !slices.Equal(dst.Shape, src.Shape) {
		return nil, nil, false
	}
	dn, ok := cudaOffloadRefillBytes(dst)
	sn, sok := cudaOffloadRefillBytes(src)
	if !ok || !sok || dn != sn || int64(dn) != bytes {
		return nil, nil, false
	}
	b, ok := dst.buf.(*cudaBuf)
	h, host := src.buf.(*hostBuf)
	if !ok || b == nil || !host || h == nil || h.nvfp4 ||
		b.ptr == nil || b.n != dn || b.class != MemoryOffload || b.device != 0 || b.be != nil ||
		b.invalidForSubmit() || b.graphOncePinned || b.managed || b.host != 0 || b.hostKeep != nil ||
		b.budgetedWeightBytes != 0 || b.managedWeight || b.scales != nil || b.scalesN != 0 ||
		cudaAsyncPending[b.ptr] != nil {
		return nil, nil, false
	}
	for _, t := range uploadCache {
		if x, ok := t.buf.(*cudaBuf); ok && x != nil && (x == b || x.ptr == b.ptr) {
			return nil, nil, false
		}
	}
	for _, x := range c.transient {
		if x != nil && (x == b || x.ptr == b.ptr) {
			return nil, nil, false
		}
	}
	var p unsafe.Pointer
	if src.Dtype == F32 {
		if len(h.f32) != sn/4 || len(h.i8) != 0 {
			return nil, nil, false
		}
		p = unsafe.Pointer(&h.f32[0])
	} else {
		if len(h.i8) != sn || len(h.f32) != 0 {
			return nil, nil, false
		}
		p = unsafe.Pointer(&h.i8[0])
	}
	d, s, n := uintptr(b.ptr), uintptr(p), uintptr(dn)
	if d > ^uintptr(0)-n || s > ^uintptr(0)-n || d < s+n && s < d+n {
		return nil, nil, false
	}
	return b, p, true
}

// Called with cudaMu and the caller's complete ForwardExecutionOwner already
// held. Operations are a software-witness seam, not replaceable public hooks.
func (c *cudaBackend) offloadRefillWithOperations(dst, src Tensor, bytes int64, enabled bool,
	probe func(unsafe.Pointer, unsafe.Pointer, int) bool,
	copyOwned func(unsafe.Pointer, unsafe.Pointer, int) int,
	stop func(unsafe.Pointer)) (out Tensor, result OwnedTransferResult) {
	if cudaOffloadRefillActive != nil {
		a := cudaOffloadRefillActive
		if a.failure == nil {
			a.failure = ErrOwnedTransferUnknown
		}
		if old, ok := a.destination.buf.(*cudaBuf); ok && old != nil {
			atomic.StoreUint32(&old.invalid, 1)
		}
		if fresh, ok := a.replacement.buf.(*cudaBuf); ok && fresh != nil {
			atomic.StoreUint32(&fresh.invalid, 1)
		}
		if stop != nil {
			stop(a.pointer)
		}
		return Tensor{}, OwnedTransferResult{State: OwnedTransferUnknown, Err: ErrOwnedTransferUnknown}
	}
	b, p, ok := c.offloadRefillOperands(dst, src, bytes)
	a := &cudaOffloadRefillAttempt{source: src, destination: dst}
	if b != nil {
		a.pointer = b.ptr
	}
	cudaOffloadRefillActive = a
	completed := false
	defer func() {
		if !completed {
			if a.failure == nil {
				a.failure = ErrOwnedTransferUnknown
			}
			if b != nil {
				atomic.StoreUint32(&b.invalid, 1)
			}
			if n, ok := a.replacement.buf.(*cudaBuf); ok && n != nil {
				atomic.StoreUint32(&n.invalid, 1)
			}
			if stop != nil {
				stop(a.pointer)
			}
		}
		runtime.KeepAlive(a)
	}()
	fail := func(code int) (Tensor, OwnedTransferResult) {
		if a.failure == nil {
			a.failure = fmt.Errorf("compute: CUDA offload refill failed (status %d)", code)
		}
		if c != nil {
			c.faultLatch.Observe(DeviceFaultExecution, "offload-refill", code)
		}
		return Tensor{}, OwnedTransferResult{State: OwnedTransferUnknown, Err: a.failure}
	}
	if !enabled || !ok || probe == nil || copyOwned == nil || stop == nil {
		return fail(46011)
	}
	if !probe(b.ptr, p, b.n) {
		return fail(46012)
	}
	// Allocate Go-only descriptor owners before detaching the old name. A new
	// Quant pointer is not a new backend handle; the cudaBuf itself must be fresh.
	fresh := &cudaBuf{ptr: b.ptr, n: b.n, class: MemoryOffload, device: 0}
	var q *QuantSpec
	if src.Quant != nil {
		v := *src.Quant
		q = &v
	}
	out = makeTensor(c, src.Dtype, src.Layout, slices.Clone(src.Shape), q, fresh)
	a.replacement = out
	atomic.StoreUint32(&b.invalid, 1)
	b.ptr = nil // stale Free can no longer return the reused allocation to a pool
	status := copyOwned(fresh.ptr, p, fresh.n)
	if status != 0 {
		return fail(status)
	}
	if a.failure != nil || cudaOffloadRefillActive != a {
		return fail(46013)
	}
	completed = true
	cudaOffloadRefillActive = nil
	return out, OwnedTransferResult{State: OwnedTransferComplete}
}

// END offload refill owner core.
