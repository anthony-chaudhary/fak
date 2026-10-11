//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

var _ V41IndexSelectBackend = (*vulkanBackend)(nil)

func (v *vulkanBackend) V41IndexSelectAvailable() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_index_select() != 0
}

// V41IndexSelect fences the selection and checks its NaN status before
// publishing the fresh output. The caller reads and frees that output through
// the checked V4.1 D2H route. Scratch and status never enter Go's transient
// pool, and no public Upload, Read or Free method is called under vulkanMu.
func (v *vulkanBackend) V41IndexSelect(scores Tensor, rows, topK, topKBlocks, blockSize int) (out Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var outBuf, scratchBuf, statusBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41IndexSelect")
		}
		for _, b := range []*vulkanBuf{scratchBuf, statusBuf, outBuf} {
			if b != nil && b.ptr != nil && (b != outBuf || err != nil) {
				C.fvk_free(b.ptr)
				b.ptr = nil
				b.n = 0
			}
		}
		if err != nil {
			out = Tensor{}
		}
	}()
	if v == nil || C.fvk_have_v41_index_select() == 0 {
		return Tensor{}, vulkanV41IndexSelectStatusError(3)
	}
	selected, validationErr := validateV41IndexSelect(scores, rows, topK, topKBlocks, blockSize)
	if validationErr != nil {
		return Tensor{}, &BackendError{Backend: "vulkan", Site: "V41IndexSelect", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: validationErr.Error()}
	}
	b, ok := scores.buf.(*vulkanBuf)
	if scores.be != v || !ok || b == nil || b.ptr == nil || b.n < rows*4 || b.scalePtr != nil || len(b.q8Chunks) != 0 {
		return Tensor{}, &BackendError{Backend: "vulkan", Site: "V41IndexSelect", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: "scores must be a live, full-sized F32 buffer owned by this backend"}
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, vulkanV41IndexSelectStatusError(status)
	}
	outBuf = v.dallocForClass(selected*4, MemoryActivation, "V4.1 index select output")
	scratchBuf = v.dallocForClass(v41IndexSelectScratchWords(rows, topKBlocks, blockSize)*4, MemoryActivation, "V4.1 index select scratch")
	statusBuf = v.dallocForClass(4, MemoryActivation, "V4.1 index select status")
	if status := v41IndexSelectStatusLocked(b, scratchBuf, outBuf, statusBuf, rows, topK, topKBlocks, blockSize); status != 0 {
		return Tensor{}, vulkanV41IndexSelectStatusError(status)
	}
	var word uint32
	if status := int(C.fvk_v41_d2h(unsafe.Pointer(&word), statusBuf.ptr, C.size_t(4))); status != 0 {
		readErr := vulkanReadError(status)
		readErr.Site = "V41IndexSelect status readback"
		return Tensor{}, readErr
	}
	if word != 0 {
		return Tensor{}, &BackendError{Backend: "vulkan", Site: "V41IndexSelect", Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: fmt.Sprintf("index select refused a NaN score (status %#x)", word)}
	}
	outBuf.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{selected}, nil, outBuf), nil
}

// Caller holds vulkanMu and passes live handles; native code repeats extent and
// alias checks before recording commands.
func v41IndexSelectStatusLocked(scores, scratch, out, status *vulkanBuf, rows, topK, topKBlocks, blockSize int) int {
	return int(C.fvk_v41_index_select_f32(scores.ptr, scratch.ptr, out.ptr, status.ptr, C.int(rows), C.int(topK), C.int(topKBlocks), C.int(blockSize)))
}

func vulkanV41IndexSelectStatusError(status int) *BackendError {
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
	return &BackendError{Backend: "vulkan", Site: "V41IndexSelect", Class: class, Err: cause, Message: fmt.Sprintf("dedicated index select failed closed (code %d)", status)}
}
