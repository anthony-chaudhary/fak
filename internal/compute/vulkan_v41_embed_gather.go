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

var (
	_ V41EmbeddingGatherBackend    = (*vulkanBackend)(nil)
	_ V41Q2KEmbeddingGatherBackend = (*vulkanBackend)(nil)
)

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

// The Q2_K gather is an optional module outside the trusted shader registry;
// its absence leaves the capability false.
func (v *vulkanBackend) V41Q2KEmbeddingGatherAvailable() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_q2k_embed_gather() != 0
}

// V41Q2KEmbeddingGather uploads the ids, dequantizes only those rows of the
// resident Q2_K table into a fresh F32 panel, and fences the dispatch before
// publishing it. The ids scratch never enters Go's transient pool, and no public
// Upload, Read or Free method is called while vulkanMu is held.
func (v *vulkanBackend) V41Q2KEmbeddingGather(table Tensor, ids []int, hidden int) (out Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var outBuf, idsBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41Q2KEmbeddingGather")
		}
		for _, b := range []*vulkanBuf{idsBuf, outBuf} {
			if b != nil && b.ptr != nil && (b == idsBuf || err != nil) {
				C.fvk_free(b.ptr)
				b.ptr = nil
				b.n = 0
			}
		}
		if err != nil {
			out = Tensor{}
		}
	}()
	if v == nil || C.fvk_have_v41_q2k_embed_gather() == 0 {
		return Tensor{}, vulkanV41Q2KEmbeddingGatherStatusError(3)
	}
	rowBytes, validationErr := validateV41Q2KEmbeddingGather(table, ids, hidden)
	if validationErr != nil {
		return Tensor{}, &BackendError{Backend: "vulkan", Site: "V41Q2KEmbeddingGather", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: validationErr.Error()}
	}
	tb, ok := table.buf.(*vulkanBuf)
	if table.be != v || !ok || tb == nil || tb.ptr == nil || tb.n < table.Shape[0]*rowBytes || tb.scalePtr != nil || len(tb.q8Chunks) != 0 {
		return Tensor{}, &BackendError{Backend: "vulkan", Site: "V41Q2KEmbeddingGather", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: "table must be a live, full-sized Q2_K buffer owned by this backend"}
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, vulkanV41Q2KEmbeddingGatherStatusError(status)
	}
	words := make([]uint32, len(ids))
	for i, id := range ids {
		words[i] = uint32(id)
	}
	idsBuf = v.dallocForClass(len(words)*4, MemoryActivation, "V4.1 Q2_K embedding gather ids")
	C.fvk_h2d(idsBuf.ptr, unsafe.Pointer(&words[0]), C.size_t(len(words)*4))
	outBuf = v.dallocForClass(len(ids)*hidden*4, MemoryActivation, "V4.1 Q2_K embedding gather panel")
	status := int(C.fvk_v41_q2k_embed_gather_f32(tb.ptr, idsBuf.ptr, outBuf.ptr, C.int(len(ids)), C.int(hidden), C.int(table.Shape[0])))
	if status != 0 {
		return Tensor{}, vulkanV41Q2KEmbeddingGatherStatusError(status)
	}
	// Fence before the ids scratch is released and before publishing the panel.
	if status = int(C.fvk_batch_flush_status()); status != 0 {
		return Tensor{}, vulkanV41Q2KEmbeddingGatherStatusError(status)
	}
	outBuf.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{len(ids), hidden}, nil, outBuf), nil
}

func vulkanV41Q2KEmbeddingGatherStatusError(status int) *BackendError {
	err := vulkanV41EmbeddingGatherStatusError(status)
	err.Site = "V41Q2KEmbeddingGather"
	if status == 2 {
		err.Class, err.Err = VulkanClassInvalidGeometry, ErrVulkanInvalidGeometry
	}
	return err
}
