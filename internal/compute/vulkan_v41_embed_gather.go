//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "fmt"

var _ V41EmbeddingGatherBackend = (*vulkanBackend)(nil)

// The gather is a sequence of device-to-device row copies through the core
// transfer path, so it needs no SPIR-V module and no registry change.
func (v *vulkanBackend) V41EmbeddingGatherAvailable() bool { return v != nil }

// V41EmbeddingGather records one buffer-to-buffer copy per id. Inside an open
// batch the copies are ordered after prior work by the batch barrier; the
// caller's checked readback fences them. No public Upload, Read or Free method
// is called while vulkanMu is held.
func (v *vulkanBackend) V41EmbeddingGather(table Tensor, ids []int, hidden int) (out Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var outBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41EmbeddingGather")
		}
		if err != nil {
			if outBuf != nil && outBuf.ptr != nil {
				C.fvk_free(outBuf.ptr)
				outBuf.ptr = nil
				outBuf.n = 0
			}
			out = Tensor{}
		}
	}()
	if v == nil {
		return Tensor{}, vulkanV41EmbeddingGatherStatusError(3)
	}
	rowBytes, validationErr := validateV41EmbeddingGather(table, ids, hidden)
	if validationErr != nil {
		return Tensor{}, &BackendError{Backend: "vulkan", Site: "V41EmbeddingGather", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: validationErr.Error()}
	}
	tb, ok := table.buf.(*vulkanBuf)
	if table.be != v || !ok || tb == nil || tb.ptr == nil || tb.n < table.Shape[0]*rowBytes || tb.scalePtr != nil || len(tb.q8Chunks) != 0 {
		return Tensor{}, &BackendError{Backend: "vulkan", Site: "V41EmbeddingGather", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: "table must be a live, full-sized F32 buffer owned by this backend"}
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, vulkanV41EmbeddingGatherStatusError(status)
	}
	outBuf = v.dallocForClass(len(ids)*rowBytes, MemoryActivation, "V4.1 embedding gather panel")
	for row, id := range ids {
		C.fvk_d2d_range(outBuf.ptr, C.size_t(row*rowBytes), tb.ptr, C.size_t(id*rowBytes), C.size_t(rowBytes))
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, vulkanV41EmbeddingGatherStatusError(status)
	}
	return makeTensor(v, F32, RowMajor, []int{len(ids), hidden}, nil, outBuf), nil
}

func vulkanV41EmbeddingGatherStatusError(status int) *BackendError {
	class, cause := VulkanClassExecutionFailed, ErrVulkanExecutionFailed
	switch {
	case status == -4:
		class, cause = VulkanClassDeviceLost, ErrVulkanDeviceLost
	case status < 0:
		class, cause = VulkanClassSubmissionFailed, ErrVulkanSubmissionFailed
	}
	return &BackendError{Backend: "vulkan", Site: "V41EmbeddingGather", Class: class, Err: cause, Message: fmt.Sprintf("embedding gather failed closed (code %d)", status)}
}
