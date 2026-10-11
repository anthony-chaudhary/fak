//go:build v41_bf16_boundary_witness && vulkan && (windows || linux) && cgo

package compute

/*
#cgo v41_indexer_witness CFLAGS: -DFAK_V41_BF16_BOUNDARY_CONFLICT
#ifdef FAK_V41_BF16_BOUNDARY_CONFLICT
#error "v41_bf16_boundary_witness and v41_indexer_witness must not be combined"
#endif
#define FAK_V41_BF16_BOUNDARY_WITNESS
#include "vulkan_backend.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

var _ V41BF16BoundaryBackend = (*vulkanBackend)(nil)

func v41BF16BoundaryModuleABI(code []byte) bool {
	if len(code) == 0 {
		return false
	}
	return C.fvk_v41_bf16_boundary_module_abi(unsafe.Pointer(&code[0]), C.size_t(len(code))) != 0
}

// This adapter is absent from ordinary production builds. The matching native
// archive must carry FAK_V41_BF16_BOUNDARY_WITNESS; otherwise it cannot link.
// Module presence/ABI checks are not physical qualification or trusted registry
// provenance. The candidate needs its own reviewed source/archive/module receipt.
func (v *vulkanBackend) V41BF16BoundaryAvailable() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_bf16_boundary() != 0
}

func (v *vulkanBackend) V41BF16Boundary(input Tensor) (out Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output *vulkanBuf
	defer func() {
		if r := recover(); r != nil {
			err = ClassifyVulkanPanic(r, "V41BF16Boundary")
		}
		if err != nil {
			if output != nil && output.ptr != nil {
				C.fvk_free(output.ptr)
				output.ptr, output.n = nil, 0
			}
			out = Tensor{}
		}
	}()
	if v == nil || C.fvk_have_v41_bf16_boundary() == 0 {
		return Tensor{}, vulkanV41BF16BoundaryStatusError(3)
	}
	bytes, invalid := validateV41BF16Boundary(input)
	if invalid != nil {
		return Tensor{}, &BackendError{Backend: "vulkan", Site: "V41BF16Boundary", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: invalid.Error()}
	}
	source, ok := input.buf.(*vulkanBuf)
	if input.be != v || !ok || source == nil || source.ptr == nil || source.n < bytes || source.scalePtr != nil || len(source.q8Chunks) != 0 {
		return Tensor{}, &BackendError{Backend: "vulkan", Site: "V41BF16Boundary", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: "input must be a live full-sized F32 allocation owned by this backend"}
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, vulkanV41BF16BoundaryStatusError(status)
	}
	output = v.dallocForClass(bytes, MemoryActivation, "V4.1 BF16 boundary output")
	if status := int(C.fvk_v41_bf16_boundary_f32(source.ptr, output.ptr, C.int(input.Shape[0]))); status != 0 {
		return Tensor{}, vulkanV41BF16BoundaryStatusError(status)
	}
	// Fence the selected operation before publication. Checked Read preserves
	// sticky submission/device-loss errors instead of exposing incomplete data.
	if status := int(C.fvk_batch_flush_status()); status != 0 {
		return Tensor{}, vulkanV41BF16BoundaryStatusError(status)
	}
	output.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{input.Shape[0]}, nil, output), nil
}

func vulkanV41BF16BoundaryStatusError(status int) *BackendError {
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
	return &BackendError{Backend: "vulkan", Site: "V41BF16Boundary", Class: class, Err: cause, Message: fmt.Sprintf("BF16 boundary failed closed (code %d)", status)}
}
