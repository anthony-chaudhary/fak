//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import (
	"math"
	"runtime"
	"unsafe"
)

var _ V41F32FiniteBackend = (*vulkanBackend)(nil)

func (v *vulkanBackend) SupportsV41F32Finite(count int) bool {
	if v == nil || count <= 0 || count > math.MaxInt32 || count > math.MaxInt/4 {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_f32_finite() != 0 && C.fvk_v41_f32_finite_count_supported(C.int(count)) != 0
}

// V41F32FiniteDeferred borrows input through the execution owner's completion.
// The returned fresh record has one owner: pass it once to V41ReadDeferred, or
// Free it after abandoning the consumer (native Free parks pending batch work).
// Availability is advisory; any selected error must propagate without replay.
func (v *vulkanBackend) V41F32FiniteDeferred(input Tensor, site uint32) (status V41DeferredStatus, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	defer runtime.KeepAlive(input)
	var record *vulkanBuf
	armed, bodyComplete := false, false
	fail := func(code int, reason string) error {
		e := vulkanV41AttentionBF16StatusError(code, reason)
		e.Site = "V41F32Finite"
		return e
	}
	defer func() {
		recovered := recover()
		incomplete := armed && !bodyComplete
		if incomplete || recovered != nil {
			if armed {
				C.fvk_v41_f32_finite_abandon()
			}
			if primary, ok := recovered.(error); ok && primary != nil {
				err = primary
			} else {
				if recovered == nil {
					recovered = "incomplete finite-status acquisition or dispatch"
				}
				err = ClassifyVulkanPanic(recovered, "V41F32Finite")
			}
		}
		// Retirement has its own completion witness. A panic(nil) or Goexit
		// during cleanup cannot turn a detached handle into successful retirement.
		retired := false
		defer func() {
			if !retired {
				C.fvk_v41_f32_finite_abandon()
				status = V41DeferredStatus{}
				if err == nil {
					err = fail(4, "incomplete finite-status retirement")
				}
			}
		}()
		if err != nil || incomplete {
			status = V41DeferredStatus{}
			v.freeDeferredLocked(record)
		}
		retired = true
	}()
	if v == nil {
		return V41DeferredStatus{}, fail(3, "nil backend")
	}
	if code := int(C.fvk_v41_attention_bf16_state()); code != 0 {
		return V41DeferredStatus{}, fail(code, "execution domain refuses finite status")
	}
	if C.fvk_have_v41_f32_finite() == 0 {
		return V41DeferredStatus{}, fail(3, "optional finite-status module unavailable")
	}
	bytes, invalid := validateV41F32Finite(input, site)
	if invalid != nil {
		return V41DeferredStatus{}, fail(2, invalid.Error())
	}
	source, ok := input.buf.(*vulkanBuf)
	if input.be != v || !ok || source == nil || source.ptr == nil || source.n < bytes || source.scalePtr != nil || len(source.q8Chunks) != 0 {
		return V41DeferredStatus{}, fail(2, "input must be live F32 storage owned by this backend")
	}
	count := input.Shape[0]
	if C.fvk_v41_f32_finite_count_supported(C.int(count)) == 0 {
		return V41DeferredStatus{}, fail(2, "selected device count limit")
	}
	// Arm before entering any resource-bearing callback. An interrupted
	// allocation may not have returned its handle, so poison the whole domain.
	armed = true
	record = v.dallocForClass(v41F32FiniteStatusBytes, MemoryActivation, "V4.1 raw finite status")
	if code := int(C.fvk_v41_f32_finite(source.ptr, record.ptr, C.int(count), C.uint32_t(site))); code != 0 {
		bodyComplete = true
		return V41DeferredStatus{}, fail(code, "initialization or checked dispatch failed")
	}
	status = V41DeferredStatus{
		Kind: V41DeferredF32Finite, Count: count, Site: site,
		Record: makeTensor(v, F32, RowMajor, []int{v41F32FiniteStatusBytes / 4}, nil, record),
	}
	bodyComplete = true
	return status, nil
}

func v41F32FiniteModuleABI(code []byte) bool {
	if len(code) == 0 {
		return false
	}
	return C.fvk_v41_f32_finite_module_abi(unsafe.Pointer(&code[0]), C.size_t(len(code))) != 0
}
