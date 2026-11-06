//go:build cuda && cgo && fak_cuda_fp8_kv_qualification

package compute

/*
#include "cuda_kv_fp8.h"
#include "cuda_kv_fp8_rotary.h"
#include "cuda_kv_fp8_lifecycle.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// This qualification-only adapter has no NewKV, public tensor, model or graph
// call site. Caller holds cudaMu. Do not expose it before the full ingress and
// native-to-Go shared-fault bridge are integrated and reviewed.
type cudaKVFP8Adapter struct {
	be                    *cudaBackend
	identity              kvFP8BackendIdentity
	lastPhase, lastStatus int
	lastNumericalMask     uint32
}

// There is exactly one persistent identity for the registered device-0 backend.
// A different backend instance or terminal identity cannot replace it to revive
// existing owners. Process teardown is the only reset supported by this leaf.
var cudaKVFP8AdapterOwner *cudaKVFP8Adapter

func cudaKVFP8AdapterLocked(c *cudaBackend) (*cudaKVFP8Adapter, error) {
	if c == nil || c != cudaDev || c.capturing {
		return nil, errKVFP8Owner
	}
	if err := AdmitDevice(c, "kv-fp8-adapter"); err != nil {
		return nil, err
	}
	if cudaKVFP8AdapterOwner != nil {
		if cudaKVFP8AdapterOwner.be != c || cudaKVFP8AdapterOwner.identity.stopped.Load() {
			return nil, errKVFP8Owner
		}
		if err := cudaKVFP8AdapterOwner.admitLocked(); err != nil {
			return nil, err
		}
		return cudaKVFP8AdapterOwner, nil
	}
	a := &cudaKVFP8Adapter{be: c}
	if err := a.admitLocked(); err != nil {
		return nil, err
	}
	cudaKVFP8AdapterOwner = a
	return a, nil
}

func (a *cudaKVFP8Adapter) observe(r C.fcuda_kv_fp8_lifecycle_result) {
	if r.status != 0 {
		a.lastPhase, a.lastStatus = int(r.phase), int(r.status)
	}
	if r.completed == 1 && r.numerical_checked == 1 && r.numerical != 0 {
		a.lastNumericalMask = uint32(r.numerical)
	}
	if r.terminal != 0 {
		a.identity.stopped.Store(true)
	}
}

func (a *cudaKVFP8Adapter) admitLocked() error {
	if a == nil || a.be == nil || a.be != cudaDev || a.be.capturing || a.identity.stopped.Load() {
		return errKVFP8Owner
	}
	if err := AdmitDevice(a.be, "kv-fp8-operation"); err != nil {
		a.identity.stopped.Store(true)
		return err
	}
	r := C.fcuda_kv_fp8_lifecycle_admit()
	a.observe(r)
	if r.status != 0 {
		return &kvFP8ReceiptError{cause: errKVFP8Owner, status: int(r.status)}
	}
	return nil
}

func (a *cudaKVFP8Adapter) newOwnerLocked(layers, heads, headDim int) (*kvFP8Owner, error) {
	if err := a.admitLocked(); err != nil {
		return nil, err
	}
	return newKVFP8Owner(&a.identity, 0, layers, heads, headDim)
}

func (a *cudaKVFP8Adapter) owns(o *kvFP8Owner) bool {
	return a != nil && o != nil && o.backend == &a.identity && o.device == 0
}

// selectedStatus is used only after an owner transaction has selected native
// work. Any failure then closes both native and Go admission before unwind.
func (a *cudaKVFP8Adapter) selectedStatus(code, phase int) int {
	if code == 0 {
		return 0
	}
	r := C.fcuda_kv_fp8_lifecycle_failed(C.int(code), C.int(phase))
	a.observe(r)
	return int(r.status)
}

func (a *cudaKVFP8Adapter) operations() kvFP8EagerOps {
	owner := kvFP8OwnerOps{
		allocate: func(dst *kvFP8Allocation, bytes, layer, plane int) error {
			role := C.int(C.FCUDA_KV_FP8_OWNER_PLANE)
			if layer == -1 {
				role = C.int(C.FCUDA_KV_FP8_OWNER_WORKSPACE)
			}
			var ptr unsafe.Pointer
			r := C.fcuda_kv_fp8_allocate_owned(&ptr, C.size_t(bytes), role)
			dst.ptr = ptr // record returned ownership BEFORE observing/panicking
			a.observe(r)
			if r.status != 0 {
				err := &kvFP8ReceiptError{cause: errKVFP8Owner, status: int(r.status)}
				if r.terminal != 0 {
					panic(err)
				} // owner abnormal-unwind quarantine
				return err
			}
			return nil
		},
		release: func(allocation kvFP8Allocation) kvFP8Retirement {
			r := C.fcuda_kv_fp8_release_owned(allocation.ptr, C.size_t(allocation.bytes))
			a.observe(r) // never panic before the caller consumes retired disposition
			return kvFP8Retirement{retired: r.retired == 1, status: int(r.status)}
		},
	}
	return kvFP8EagerOps{
		owner: owner,
		begin: func(status kvFP8Allocation) int {
			r := C.fcuda_kv_fp8_begin_owned((*C.int)(status.ptr))
			a.observe(r)
			return int(r.status)
		},
		copy: func(dst kvFP8Allocation, dstOffset int, src kvFP8Allocation, srcOffset, bytes int) int {
			r := C.fcuda_kv_fp8_copy_owned(dst.ptr, C.size_t(dst.bytes), C.size_t(dstOffset),
				src.ptr, C.size_t(src.bytes), C.size_t(srcOffset), C.size_t(bytes))
			a.observe(r)
			return int(r.status)
		},
		rotary: func(dst, raw kvFP8Allocation, before, from, end, heads int, contract kvFP8RotaryContract, status kvFP8Allocation) int {
			// Native exact geometry and private source lifetime were already
			// admitted by the bound plan; no full-head/theta fallback exists.
			code := int(C.fcuda_kv_fp8_partial_rope_stage_async((*C.float)(dst.ptr), (*C.float)(raw.ptr),
				C.int(before), C.int(from), C.int(end), C.int(heads), C.int(contract.headDim),
				C.int(contract.rotaryDim), C.double(math.Float64frombits(contract.thetaBits)),
				C.int(contract.producer), (*C.int)(status.ptr)))
			return a.selectedStatus(code, int(C.FCUDA_KV_FP8_PHASE_ROTARY))
		},
		quantize: func(codes kvFP8Allocation, codeOffset int, scales kvFP8Allocation, scaleOffset int,
			src kvFP8Allocation, rows, hd int, status kvFP8Allocation) int {
			code := int(C.fcuda_kv_fp8_quantize_async((*C.uint8_t)(unsafe.Add(codes.ptr, codeOffset)),
				(*C.float)(unsafe.Add(scales.ptr, scaleOffset)), (*C.float)(src.ptr),
				C.int(rows), C.int(hd), (*C.int)(status.ptr)))
			return a.selectedStatus(code, int(C.FCUDA_KV_FP8_PHASE_QUANTIZE))
		},
		finish: func(status *kvFP8Allocation) kvFP8Completion {
			var ptr *C.int
			readNumerical := C.int(0)
			if status != nil {
				ptr = (*C.int)(status.ptr)
				readNumerical = 1
			}
			r := C.fcuda_kv_fp8_finish_owned(ptr, readNumerical)
			a.observe(r)
			return kvFP8Completion{completed: r.completed == 1, numericalChecked: r.numerical_checked == 1,
				numerical: uint32(r.numerical), status: int(r.status)}
		},
	}
}

func (a *cudaKVFP8Adapter) cloneLocked(o *kvFP8Owner, peakBudget int64) (*kvFP8Owner, error) {
	if !a.owns(o) {
		return nil, errKVFP8Owner
	}
	if err := o.usable(); err != nil {
		return nil, err
	}
	if err := a.admitLocked(); err != nil {
		return nil, err
	}
	return kvFP8CloneEager(o, peakBudget, a.operations())
}

func (a *cudaKVFP8Adapter) evictLocked(o *kvFP8Owner, from, count int, peakBudget int64) (int, error) {
	if !a.owns(o) {
		return 0, errKVFP8Owner
	}
	// Bindings and stale/frontier geometry are pure preflight before native
	// admission. No-op eviction does not probe/fence unrelated device work.
	plan, err := o.planEvict(from, count)
	if err != nil || plan == nil {
		return 0, err
	}
	if err := a.admitLocked(); err != nil {
		return 0, err
	}
	return kvFP8ApplyEvictionEager(plan, peakBudget, a.operations())
}

func (a *cudaKVFP8Adapter) freeLocked(o *kvFP8Owner) error {
	if !a.owns(o) {
		return errKVFP8Owner
	}
	// A native terminal discovered here roots this owner instead of losing it
	// after its enclosing cudaKV/session handle has already been detached.
	if err := a.admitLocked(); err != nil {
		o.retainUncertain()
		return err
	}
	return o.free(a.operations().owner)
}

// All native outcomes observed through this adapter update stopped immediately.
// Other native producers are NOT yet connected to this Go atomic. Their bridge
// must land before borrowed Ready() or a production FP8 constructor is exposed.
func (a *cudaKVFP8Adapter) diagnosticLocked() string {
	return fmt.Sprintf("FP8 lifecycle failure_phase=%d failure_status=%d numerical_mask=%#x", a.lastPhase, a.lastStatus, a.lastNumericalMask)
}
