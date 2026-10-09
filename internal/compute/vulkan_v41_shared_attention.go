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

var _ V41SharedAttentionBackend = (*vulkanBackend)(nil)

// VulkanPhysicalDeviceType reports Vulkan's selected device-type enum for
// metadata-only qualification: 0 unknown, 1 integrated, 2 discrete, 3 virtual,
// 4 CPU. The existing helper owns locking and unavailable/nil handling.
func (v *vulkanBackend) VulkanPhysicalDeviceType() uint32 {
	return uint32(v.v41PhysicalDeviceType())
}

func (v *vulkanBackend) SupportsV41SharedAttention() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_shared_attention() != 0
}

// V41SharedAttention fences and validates every head's status before publishing
// its fresh device output. The caller reads and frees that output; its marked
// buffer uses the checked V4.1 D2H route. Neither output nor status enters Go's
// transient pool. Native pooling still applies after a known-complete operation.
// No public Upload, Read or Free method is called while vulkanMu is held.
func (v *vulkanBackend) V41SharedAttention(q, sharedKV, sink Tensor, selectedRows, heads, headDim int, scale float32, mode V41SharedAttentionMode, hasSink bool) (out Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var outBuf, statusBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41SharedAttention")
		}
		// fvk_free parks resources on an uncertain submit/wait, rather than
		// recycling in-flight memory. That process-terminal quarantine is not
		// recoverable here, and no Go transient-pool release bypasses it.
		for _, b := range []*vulkanBuf{statusBuf, outBuf} {
			if b != nil && b.ptr != nil && (b == statusBuf || err != nil) {
				C.fvk_free(b.ptr)
				b.ptr = nil
				b.n = 0
			}
		}
		if err != nil {
			out = Tensor{}
		}
	}()
	if v == nil || C.fvk_have_v41_shared_attention() == 0 {
		return Tensor{}, vulkanV41SharedAttentionStatusError(3)
	}
	qBytes, kvBytes, sinkBytes, statusBytes, validationErr := validateV41SharedAttention(q, sharedKV, sink, selectedRows, heads, headDim, scale, mode, hasSink)
	if validationErr != nil {
		return Tensor{}, &BackendError{Backend: "vulkan", Site: "V41SharedAttention", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: validationErr.Error(), Recovered: validationErr}
	}
	for _, operand := range []struct {
		name  string
		t     Tensor
		bytes int
	}{
		{"q", q, qBytes}, {"sharedKV", sharedKV, kvBytes}, {"sink", sink, sinkBytes},
	} {
		b, ok := operand.t.buf.(*vulkanBuf)
		if operand.t.be != v || !ok || b == nil || b.ptr == nil || b.n != operand.bytes || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return Tensor{}, &BackendError{Backend: "vulkan", Site: "V41SharedAttention", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: operand.name + " must be a live, exact-sized F32 device buffer owned by this backend"}
		}
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, vulkanV41SharedAttentionStatusError(status)
	}
	// The inherited allocator fails closed and uses the same memory accounting
	// as other activations. It may collapse allocation-time VkResult details.
	outBuf = v.dallocForClass(qBytes, MemoryActivation, "V4.1 shared attention output")
	statusBuf = v.dallocForClass(statusBytes, MemoryActivation, "V4.1 shared attention status")
	status := v41SharedAttentionStatusLocked(q.buf.(*vulkanBuf), sharedKV.buf.(*vulkanBuf), sink.buf.(*vulkanBuf), outBuf, statusBuf, selectedRows, heads, headDim, scale, mode, hasSink)
	if status != 0 {
		return Tensor{}, vulkanV41SharedAttentionStatusError(status)
	}
	words := make([]uint32, statusBytes/4)
	status = int(C.fvk_v41_d2h(unsafe.Pointer(&words[0]), statusBuf.ptr, C.size_t(statusBytes)))
	if status != 0 {
		readErr := vulkanReadError(status)
		readErr.Site = "V41SharedAttention status readback"
		return Tensor{}, readErr
	}
	if statusErr := decodeV41SharedAttentionStatus(words, heads, selectedRows, headDim); statusErr != nil {
		return Tensor{}, &BackendError{Backend: "vulkan", Site: "V41SharedAttention", Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: statusErr.Error(), Recovered: statusErr}
	}
	outBuf.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{heads, headDim}, nil, outBuf), nil
}

// Caller holds vulkanMu and owns live native handles. The native entrypoint
// independently checks byte extents and output/status aliases before dispatch.
func v41SharedAttentionStatusLocked(q, sharedKV, sink, out, status *vulkanBuf, selectedRows, heads, headDim int, scale float32, mode V41SharedAttentionMode, hasSink bool) int {
	var sinkFlag C.int
	if hasSink {
		sinkFlag = 1
	}
	return int(C.fvk_v41_shared_attention_f32(q.ptr, sharedKV.ptr, sink.ptr, out.ptr, status.ptr, C.int(heads), C.int(headDim), C.int(selectedRows), C.int(mode), sinkFlag, C.float(scale)))
}

func vulkanV41SharedAttentionStatusError(status int) *BackendError {
	class, cause := VulkanClassExecutionFailed, ErrVulkanExecutionFailed
	switch {
	case status == -4: // VK_ERROR_DEVICE_LOST
		class, cause = VulkanClassDeviceLost, ErrVulkanDeviceLost
	case status == -1 || status == -2:
		class, cause = VulkanClassAllocationFailed, ErrVulkanAllocationFailed
	case status < 0:
		class, cause = VulkanClassSubmissionFailed, ErrVulkanSubmissionFailed
	case status == 2:
		class, cause = VulkanClassInvalidGeometry, ErrVulkanInvalidGeometry
	}
	return &BackendError{Backend: "vulkan", Site: "V41SharedAttention", Class: class, Err: cause, Message: fmt.Sprintf("dedicated shared attention failed closed (code %d)", status)}
}
