//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

var _ V41CompressorPoolBackend = (*vulkanBackend)(nil)

func (v *vulkanBackend) SupportsV41CompressorPool() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_compressor_pool() != 0
}

// V41CompressorPool fences the dispatch through the checked V4.1 completion
// path before publishing a fresh pooled row. The output never enters the Go
// transient pool, and no public Read/Free/Upload is called under vulkanMu.
func (v *vulkanBackend) V41CompressorPool(kv, scores Tensor, ratio, width int) (out Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var outBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41CompressorPool")
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
		return Tensor{}, vulkanV41CompressorPoolStatusError(3, "Vulkan backend is nil")
	}
	if validationErr := validateV41CompressorPool(kv, scores, ratio, width); validationErr != nil {
		return Tensor{}, vulkanV41CompressorPoolStatusError(2, validationErr.Error())
	}
	operands := [2]*vulkanBuf{}
	for i, operand := range []Tensor{kv, scores} {
		b, ok := operand.buf.(*vulkanBuf)
		if operand.be != v || !ok || b == nil || b.ptr == nil || b.n < ratio*width*4 || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return Tensor{}, vulkanV41CompressorPoolStatusError(2, "kv and scores must be live, full-sized F32 device buffers owned by this backend")
		}
		operands[i] = b
	}
	if C.fvk_have_v41_compressor_pool() == 0 {
		return Tensor{}, vulkanV41CompressorPoolStatusError(3, "compressor pool module or binary32 float controls unavailable")
	}
	if code := int(C.fvk_submission_status()); code != 0 {
		return Tensor{}, vulkanV41CompressorPoolStatusError(code, "sticky submission fault")
	}
	outBuf = v.dallocForClass(width*4, MemoryActivation, "V4.1 compressor pool output")
	code := int(C.fvk_v41_compressor_pool_f32(operands[0].ptr, operands[1].ptr, outBuf.ptr, C.int(ratio), C.int(width)))
	if code != 0 {
		return Tensor{}, vulkanV41CompressorPoolStatusError(code, "native dispatch refused")
	}
	if code = int(C.fvk_batch_flush_status()); code != 0 {
		return Tensor{}, vulkanV41CompressorPoolStatusError(code, "checked completion failed")
	}
	outBuf.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{width}, nil, outBuf), nil
}
