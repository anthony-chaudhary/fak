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

var _ V41BF16PublishBackend = (*vulkanBackend)(nil)

func (v *vulkanBackend) SupportsV41BF16Publish(count int) bool {
	if v == nil || count <= 0 || count > math.MaxInt32 || count > math.MaxInt/4 {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_attention_bf16() != 0 && C.fvk_v41_attention_bf16_count_supported(C.int(count)) != 0
}

// V41BF16Publish shares the attention publication's ownership protocol: the
// native owner mutex covers initialization, dispatch, status transport and
// cleanup, and fvk_free retains uncertain submissions in native quarantine.
func (v *vulkanBackend) V41BF16Publish(input Tensor, site uint32) (out Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output, statusBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			if primary, ok := recovered.(error); ok {
				err = primary
			} else {
				err = ClassifyVulkanPanic(recovered, "V41BF16Publish")
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
	fail := func(status int, reason string) error {
		e := vulkanV41AttentionBF16StatusError(status, reason)
		e.Site = "V41BF16Publish"
		e.Message = fmt.Sprintf("BF16 publication failed closed (code %d): %s", status, reason)
		return e
	}
	if v == nil {
		return Tensor{}, fail(3, "nil backend")
	}
	if status := int(C.fvk_v41_attention_bf16_state()); status != 0 {
		return Tensor{}, fail(status, "submission state refuses publication")
	}
	if C.fvk_have_v41_attention_bf16() == 0 {
		return Tensor{}, fail(3, "ordinary checked module unavailable")
	}
	bytes, invalid := validateV41BF16Publish(input, site)
	if invalid != nil {
		return Tensor{}, fail(2, invalid.Error())
	}
	source, ok := input.buf.(*vulkanBuf)
	if input.be != v || !ok || source == nil || source.ptr == nil || source.n < bytes || source.scalePtr != nil || len(source.q8Chunks) != 0 {
		return Tensor{}, fail(2, "input must be live F32 storage owned by this backend")
	}
	if C.fvk_v41_attention_bf16_count_supported(C.int(input.Shape[0])) == 0 {
		return Tensor{}, fail(2, "selected device count limit")
	}
	output = v.dallocForClass(bytes, MemoryActivation, "V4.1 BF16 publication output")
	statusBuf = v.dallocForClass(v41BF16CheckedStatusBytes, MemoryActivation, "V4.1 BF16 publication status")
	if status := int(C.fvk_v41_bf16_publish_f32(source.ptr, output.ptr, statusBuf.ptr, C.int(input.Shape[0]), C.uint32_t(site))); status != 0 {
		return Tensor{}, fail(status, "initialization or checked dispatch failed")
	}
	var raw [v41BF16CheckedStatusBytes]byte
	if status := int(C.fvk_v41_d2h(unsafe.Pointer(&raw[0]), statusBuf.ptr, C.size_t(len(raw)))); status != 0 {
		readErr := vulkanReadError(status)
		readErr.Site = "V41BF16Publish status readback"
		return Tensor{}, readErr
	}
	if statusErr := decodeV41BF16PublishStatus(raw[:], input.Shape[0], site); statusErr != nil {
		return Tensor{}, &BackendError{Backend: "vulkan", Site: "V41BF16Publish", Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: statusErr.Error(), Recovered: statusErr}
	}
	output.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{input.Shape[0]}, nil, output), nil
}
