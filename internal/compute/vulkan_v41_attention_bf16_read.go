//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

// The mutex covers dispatch, one grouped read, verdict, and transient cleanup.
// Input remains borrowed; no native tensor escapes this owned-host operation.
func (v *vulkanBackend) v41AttentionBF16ReadOwned(original Backend, input Tensor, site uint32) (row []float32, selected bool, err error) {
	canonical, ok := original.(*vulkanBackend)
	if !ok || canonical != v || v == nil {
		return nil, false, nil
	}
	// Keep grouped status+row within the existing64KiB staging minimum.
	// Larger rows retain the ordinary path before any allocation or submission.
	if len(input.Shape) == 1 && input.Shape[0] > (64*1024-v41BF16CheckedStatusBytes)/4 {
		return nil, false, nil
	}
	selected = true
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output, statusBuf *vulkanBuf
	completed := false
	defer func() {
		recovered := recover()
		if !completed {
			if primary, ok := recovered.(error); ok {
				err = primary
			} else {
				err = ClassifyVulkanPanic(recovered, "V41AttentionBF16")
				if err == nil {
					err = vulkanV41AttentionBF16StatusError(4, "interrupted owned read")
				}
			}
		}
		for _, b := range []*vulkanBuf{statusBuf, output} {
			if b != nil && b.ptr != nil {
				C.fvk_free(b.ptr)
				b.ptr, b.n = nil, 0
			}
		}
		if !completed || err != nil {
			row = nil
		}
	}()
	row, err = func() ([]float32, error) {
		if v == nil {
			return nil, vulkanV41AttentionBF16StatusError(3, "nil backend")
		}
		// Fatal sticky/pending state precedes healthy batch/restore refusal. No
		// allocation can hide an independent pending fault behind a geometry error.
		if status := int(C.fvk_v41_attention_bf16_state()); status != 0 {
			return nil, vulkanV41AttentionBF16StatusError(status, "submission state refuses publication")
		}
		if C.fvk_have_v41_attention_bf16() == 0 {
			return nil, vulkanV41AttentionBF16StatusError(3, "ordinary checked module unavailable")
		}
		bytes, invalid := validateV41AttentionBF16(input, site)
		if invalid != nil {
			return nil, vulkanV41AttentionBF16StatusError(2, invalid.Error())
		}
		source, ok := input.buf.(*vulkanBuf)
		if input.be != v || !ok || source == nil || source.ptr == nil || source.n < bytes || source.scalePtr != nil || len(source.q8Chunks) != 0 {
			return nil, vulkanV41AttentionBF16StatusError(2, "input must be live F32 storage owned by this backend")
		}
		if C.fvk_v41_attention_bf16_count_supported(C.int(input.Shape[0])) == 0 {
			return nil, vulkanV41AttentionBF16StatusError(2, "selected device count limit")
		}
		output = v.dallocForClass(bytes, MemoryActivation, "V4.1 attention BF16 output")
		statusBuf = v.dallocForClass(v41BF16CheckedStatusBytes, MemoryActivation, "V4.1 attention BF16 status")
		if status := int(C.fvk_v41_attention_bf16_f32(source.ptr, output.ptr, statusBuf.ptr, C.int(input.Shape[0]), C.uint32_t(site))); status != 0 {
			return nil, vulkanV41AttentionBF16StatusError(status, "initialization or checked dispatch failed")
		}
		sizes := []int{v41BF16CheckedStatusBytes, bytes}
		image, readErr := vulkanV41ReadImage("V41AttentionBF16 status", []*vulkanBuf{statusBuf, output}, sizes, 1)
		if readErr != nil {
			return nil, readErr
		}
		// Grouped transfer may resume a step batch. Never publish after failed resume.
		if status := int(C.fvk_v41_attention_bf16_state()); status != 0 {
			return nil, vulkanV41AttentionBF16StatusError(status, "submission state after grouped readback")
		}
		// Materialize only after verdict; a one-row arena retains16 status bytes.
		return v41AttentionBF16OwnedRow(image, input.Shape[0], site)
	}()
	completed = true
	return row, selected, err
}
