//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

var _ V41AttentionBF16Backend = (*vulkanBackend)(nil)

func v41AttentionBF16ModuleABI(code []byte) bool {
	return len(code) != 0 && C.fvk_v41_attention_bf16_module_abi(unsafe.Pointer(&code[0]), C.size_t(len(code))) != 0
}

func (v *vulkanBackend) SupportsV41AttentionBF16(count int) bool {
	if v == nil || count <= 0 || count > math.MaxInt32 || count > math.MaxInt/4 {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_attention_bf16() != 0 && C.fvk_v41_attention_bf16_count_supported(C.int(count)) != 0
}

// V41AttentionBF16 keeps initialization, dispatch, status transport and cleanup
// under the native owner mutex. No public Read/Free is called while locked.
// fvk_free retains uncertain submissions in the existing native quarantine.
func (v *vulkanBackend) V41AttentionBF16(input Tensor, site uint32) (out Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output, statusBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			if primary, ok := recovered.(error); ok {
				err = primary
			} else {
				err = ClassifyVulkanPanic(recovered, "V41AttentionBF16")
			}
		}
		for _, b := range []*vulkanBuf{statusBuf, output} {
			if b != nil && b.ptr != nil && (b == statusBuf || err != nil) {
				C.fvk_free(b.ptr)
				b.ptr, b.n = nil, 0
			}
		}
		if err != nil {
			out = Tensor{}
		}
	}()
	if v == nil {
		return Tensor{}, vulkanV41AttentionBF16StatusError(3, "nil backend")
	}
	// Fatal sticky/pending state precedes healthy batch/restore refusal. No
	// allocation can hide an independent pending fault behind a geometry error.
	if status := int(C.fvk_v41_attention_bf16_state()); status != 0 {
		return Tensor{}, vulkanV41AttentionBF16StatusError(status, "submission state refuses publication")
	}
	if C.fvk_have_v41_attention_bf16() == 0 {
		return Tensor{}, vulkanV41AttentionBF16StatusError(3, "ordinary checked module unavailable")
	}
	bytes, invalid := validateV41AttentionBF16(input, site)
	if invalid != nil {
		return Tensor{}, vulkanV41AttentionBF16StatusError(2, invalid.Error())
	}
	source, ok := input.buf.(*vulkanBuf)
	if input.be != v || !ok || source == nil || source.ptr == nil || source.n < bytes || source.scalePtr != nil || len(source.q8Chunks) != 0 {
		return Tensor{}, vulkanV41AttentionBF16StatusError(2, "input must be live F32 storage owned by this backend")
	}
	if C.fvk_v41_attention_bf16_count_supported(C.int(input.Shape[0])) == 0 {
		return Tensor{}, vulkanV41AttentionBF16StatusError(2, "selected device count limit")
	}
	output = v.dallocForClass(bytes, MemoryActivation, "V4.1 attention BF16 output")
	statusBuf = v.dallocForClass(v41BF16CheckedStatusBytes, MemoryActivation, "V4.1 attention BF16 status")
	if status := int(C.fvk_v41_attention_bf16_f32(source.ptr, output.ptr, statusBuf.ptr, C.int(input.Shape[0]), C.uint32_t(site))); status != 0 {
		return Tensor{}, vulkanV41AttentionBF16StatusError(status, "initialization or checked dispatch failed")
	}
	var raw [v41BF16CheckedStatusBytes]byte
	if status := int(C.fvk_v41_d2h(unsafe.Pointer(&raw[0]), statusBuf.ptr, C.size_t(len(raw)))); status != 0 {
		readErr := vulkanReadError(status)
		readErr.Site = "V41AttentionBF16 status readback"
		return Tensor{}, readErr
	}
	if statusErr := decodeV41AttentionBF16Status(raw[:], input.Shape[0], site); statusErr != nil {
		return Tensor{}, &BackendError{Backend: "vulkan", Site: "V41AttentionBF16", Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: statusErr.Error(), Recovered: statusErr}
	}
	output.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{input.Shape[0]}, nil, output), nil
}

func vulkanV41AttentionBF16StatusError(status int, reason string) *BackendError {
	class, cause := VulkanClassExecutionFailed, ErrVulkanExecutionFailed
	switch {
	case status == -4:
		class, cause = VulkanClassDeviceLost, ErrVulkanDeviceLost
	case status == -1 || status == -2:
		class, cause = VulkanClassAllocationFailed, ErrVulkanAllocationFailed
	case status < 0:
		class, cause = VulkanClassSubmissionFailed, ErrVulkanSubmissionFailed
	case status == 2:
		class, cause = VulkanClassInvalidGeometry, ErrVulkanInvalidGeometry
	}
	return &BackendError{Backend: "vulkan", Site: "V41AttentionBF16", Class: class, Err: cause, Message: fmt.Sprintf("attention BF16 failed closed (code %d): %s", status, reason)}
}
