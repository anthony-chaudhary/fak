//go:build v41_bf16_boundary_witness && vulkan && (windows || linux) && cgo

package compute

/*
#cgo v41_indexer_witness CFLAGS: -DFAK_V41_BF16_CHECKED_CONFLICT
#cgo v41_logit_scale_witness CFLAGS: -DFAK_V41_BF16_CHECKED_CONFLICT
#ifdef FAK_V41_BF16_CHECKED_CONFLICT
#error "checked BF16 uses only the isolated BF16 witness archive"
#endif
#define FAK_V41_BF16_BOUNDARY_WITNESS
#include "vulkan_backend.h"
*/
import "C"

import "unsafe"

var _ V41BF16CheckedBackend = (*vulkanBackend)(nil)

func v41BF16CheckedModuleABI(code []byte) bool {
	return len(code) != 0 && C.fvk_v41_bf16_checked_module_abi(unsafe.Pointer(&code[0]), C.size_t(len(code))) != 0
}

func (v *vulkanBackend) V41BF16CheckedAvailable() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_bf16_checked() != 0
}

// V41BF16Checked is synchronous: native status initialization, visibility and
// dispatch use the checked one-shot path, followed by checked status readback.
// The raw all-bit operation and ordinary archive remain unchanged. Pending
// submissions keep native quarantine; no public Read/Free is called under mu.
func (v *vulkanBackend) V41BF16Checked(input Tensor, site uint32) (out Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output, statusBuf *vulkanBuf
	defer func() {
		if r := recover(); r != nil {
			err = ClassifyVulkanPanic(r, "V41BF16Checked")
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
	if v == nil || C.fvk_have_v41_bf16_checked() == 0 {
		return Tensor{}, vulkanV41BF16CheckedStatusError(3, "checked module unavailable")
	}
	bytes, invalid := validateV41BF16Checked(input, site)
	if invalid != nil {
		return Tensor{}, vulkanV41BF16CheckedStatusError(2, invalid.Error())
	}
	source, ok := input.buf.(*vulkanBuf)
	if input.be != v || !ok || source == nil || source.ptr == nil || source.n < bytes || source.scalePtr != nil || len(source.q8Chunks) != 0 {
		return Tensor{}, vulkanV41BF16CheckedStatusError(2, "input must be a live F32 buffer owned by this backend")
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, vulkanV41BF16CheckedStatusError(status, "sticky submission fault")
	}
	// This prerequisite does not acquire or complete another operation's batch.
	// A model selected-error Close retains the existing teardown contract.
	if C.fvk_batch_active() || C.fvk_v41_bf16_checked_count_supported(C.int(input.Shape[0])) == 0 {
		return Tensor{}, vulkanV41BF16CheckedStatusError(2, "active batch or unsupported checked geometry")
	}
	output = v.dallocForClass(bytes, MemoryActivation, "V4.1 checked BF16 output")
	statusBuf = v.dallocForClass(v41BF16CheckedStatusBytes, MemoryActivation, "V4.1 checked BF16 status")
	if status := int(C.fvk_v41_bf16_checked_f32(source.ptr, output.ptr, statusBuf.ptr, C.int(input.Shape[0]), C.uint32_t(site))); status != 0 {
		return Tensor{}, vulkanV41BF16CheckedStatusError(status, "initialization or checked dispatch failed")
	}
	var raw [v41BF16CheckedStatusBytes]byte
	if status := int(C.fvk_v41_d2h(unsafe.Pointer(&raw[0]), statusBuf.ptr, C.size_t(len(raw)))); status != 0 {
		readErr := vulkanReadError(status)
		readErr.Site = "V41BF16Checked status readback"
		return Tensor{}, readErr
	}
	if statusErr := decodeV41BF16CheckedStatus(raw[:], input.Shape[0], site); statusErr != nil {
		return Tensor{}, &BackendError{Backend: "vulkan", Site: "V41BF16Checked", Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: statusErr.Error(), Recovered: statusErr}
	}
	output.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{input.Shape[0]}, nil, output), nil
}

func vulkanV41BF16CheckedStatusError(status int, reason string) *BackendError {
	err := vulkanV41BF16BoundaryStatusError(status)
	err.Site = "V41BF16Checked"
	err.Message += ": " + reason
	return err
}
