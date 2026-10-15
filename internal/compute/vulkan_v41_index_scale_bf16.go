//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "unsafe"

var _ V41IndexScaleBF16Backend = (*vulkanBackend)(nil)

func (v *vulkanBackend) SupportsV41IndexScaleBF16(input Tensor, scale float32) bool {
	if v == nil {
		return false
	}
	bytes, err := validateV41ScaleExact(input, scale)
	if err != nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	b, ok := input.buf.(*vulkanBuf)
	return input.be == v && ok && b != nil && b.ptr != nil && b.n >= bytes && b.scalePtr == nil && len(b.q8Chunks) == 0 && C.fvk_have_v41_index_scale_bf16() != 0 && C.fvk_v41_index_scale_bf16_count(C.int(input.Shape[0])) != 0
}

// Preserve ordinary publication's submission-state precedence, fresh ownership
// and native deferred-free quarantine. Only fixed IndexerScaledWeight site25 is exposed.
func (v *vulkanBackend) V41IndexScaleBF16Deferred(input Tensor, scale float32) (out Tensor, status V41DeferredStatus, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output, statusBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			if primary, ok := recovered.(error); ok {
				err = primary
			} else {
				err = ClassifyVulkanPanic(recovered, "V41IndexScaleBF16")
			}
		}
		if err != nil {
			v.freeDeferredLocked(statusBuf, output)
			out, status = Tensor{}, V41DeferredStatus{}
		}
	}()
	fail := func(code int, reason string) error {
		e := vulkanV41AttentionBF16StatusError(code, reason)
		e.Site = "V41IndexScaleBF16"
		return e
	}
	if v == nil {
		return Tensor{}, V41DeferredStatus{}, fail(3, "nil backend")
	}
	if code := int(C.fvk_v41_attention_bf16_state()); code != 0 {
		return Tensor{}, V41DeferredStatus{}, fail(code, "submission state refuses publication")
	}
	if C.fvk_have_v41_index_scale_bf16() == 0 {
		return Tensor{}, V41DeferredStatus{}, fail(3, "ordinary checked module unavailable")
	}
	bytes, invalid := validateV41ScaleExact(input, scale)
	if invalid != nil {
		return Tensor{}, V41DeferredStatus{}, fail(2, invalid.Error())
	}
	count := input.Shape[0]
	source, ok := input.buf.(*vulkanBuf)
	if input.be != v || !ok || source == nil || source.ptr == nil || source.n < bytes || source.scalePtr != nil || len(source.q8Chunks) != 0 {
		return Tensor{}, V41DeferredStatus{}, fail(2, "input must be live F32 storage owned by this backend")
	}
	if C.fvk_v41_index_scale_bf16_count(C.int(input.Shape[0])) == 0 {
		return Tensor{}, V41DeferredStatus{}, fail(2, "selected device count limit")
	}
	output = v.dallocForClass(bytes, MemoryActivation, "V4.1 fused index scale publication output")
	statusBuf = v.dallocForClass(v41BF16CheckedStatusBytes, MemoryActivation, "V4.1 fused index scale publication status")
	if code := int(C.fvk_v41_index_scale_bf16_f32(source.ptr, output.ptr, statusBuf.ptr, C.int(count), C.float(scale))); code != 0 {
		return Tensor{}, V41DeferredStatus{}, fail(code, "initialization or checked dispatch failed")
	}
	output.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{count}, nil, output), V41DeferredStatus{
		Kind: V41DeferredBF16Publish, Count: count, Site: V41BF16PublishIndexerScaledWeight,
		Record: makeTensor(v, F32, RowMajor, []int{v41BF16CheckedStatusBytes / 4}, nil, statusBuf),
	}, nil
}

// Software ABI qualification only; hardware numerical execution remains separate.
func v41IndexScaleBF16ModuleABI(code []byte) bool {
	if len(code) == 0 {
		return false
	}
	return C.fvk_v41_index_scale_bf16_module_abi(unsafe.Pointer(&code[0]), C.size_t(len(code))) != 0
}
