//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

// The mutex covers dispatch, one grouped read, verdict, and transient cleanup.
// Input remains borrowed; no native tensor escapes this owned-host operation.
func (v *vulkanBackend) v41IndexSelectReadOwned(original Backend, scores Tensor, rows, topK, topKBlocks, blockSize int) (row []float32, selected bool, err error) {
	canonical, ok := original.(*vulkanBackend)
	if !ok || canonical != v || v == nil {
		return nil, false, nil
	}
	// Keep grouped status+row within the existing64KiB staging minimum.
	// The 4-byte status occupies a 16-byte aligned prefix. Tiny rows cannot
	// retain that prefix within the one-payload ownership bound; decline them.
	// Larger rows also retain the ordinary path before allocation/submission.
	if count := min(rows, topK); count < 4 || count > (64*1024-16)/4 {
		return nil, false, nil
	}
	selected = true
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output, scratchBuf, statusBuf *vulkanBuf
	completed := false
	defer func() {
		recovered := recover()
		if !completed {
			if primary, ok := recovered.(error); ok {
				err = primary
			} else {
				err = ClassifyVulkanPanic(recovered, "V41IndexSelect")
				if err == nil {
					err = vulkanV41IndexSelectStatusError(4)
				}
			}
		}
		for _, b := range []*vulkanBuf{scratchBuf, statusBuf, output} {
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
		if v == nil || C.fvk_have_v41_index_select() == 0 {
			return nil, vulkanV41IndexSelectStatusError(3)
		}
		count, validationErr := validateV41IndexSelect(scores, rows, topK, topKBlocks, blockSize)
		if validationErr != nil {
			return nil, &BackendError{Backend: "vulkan", Site: "V41IndexSelect", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: validationErr.Error()}
		}
		b, ok := scores.buf.(*vulkanBuf)
		if scores.be != v || !ok || b == nil || b.ptr == nil || b.n < rows*4 || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return nil, &BackendError{Backend: "vulkan", Site: "V41IndexSelect", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: "scores must be a live, full-sized F32 buffer owned by this backend"}
		}
		if status := int(C.fvk_submission_status()); status != 0 {
			return nil, vulkanV41IndexSelectStatusError(status)
		}
		output = v.dallocForClass(count*4, MemoryActivation, "V4.1 index select output")
		scratchBuf = v.dallocForClass(v41IndexSelectScratchWords(rows, topKBlocks, blockSize)*4, MemoryActivation, "V4.1 index select scratch")
		statusBuf = v.dallocForClass(4, MemoryActivation, "V4.1 index select status")
		if status := v41IndexSelectStatusLocked(b, scratchBuf, output, statusBuf, rows, topK, topKBlocks, blockSize); status != 0 {
			return nil, vulkanV41IndexSelectStatusError(status)
		}
		sizes := []int{4, 4 * count}
		image, readErr := vulkanV41ReadImage("V41IndexSelect status", []*vulkanBuf{statusBuf, output}, sizes, 1)
		if readErr != nil {
			return nil, readErr
		}
		if status := int(C.fvk_submission_status()); status != 0 {
			return nil, vulkanV41IndexSelectStatusError(status)
		}
		return v41IndexSelectOwnedRows(image, count)
	}()
	completed = true
	return row, selected, err
}
