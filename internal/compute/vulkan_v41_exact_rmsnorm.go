//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

var _ V41ExactRMSNormBackend = (*vulkanBackend)(nil)

func (v *vulkanBackend) SupportsV41ExactRMSNorm(width int, eps float32) bool {
	if v == nil || !V41ExactRMSNormDomain(width, eps) {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_exact_rmsnorm() != 0
}

// V41ExactRMSNorm fences the dispatch through the checked V4.1 completion path
// before publishing a fresh output. The output never enters the Go transient
// pool, and no public Read/Free/Upload is called while vulkanMu is held.
func (v *vulkanBackend) V41ExactRMSNorm(x, weight Tensor, eps float32) (out Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var outBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41ExactRMSNorm")
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
		return Tensor{}, vulkanV41ExactRMSNormStatusError(3, "Vulkan backend is nil")
	}
	rows, width, validationErr := validateV41ExactRMSNorm(x, weight, eps)
	if validationErr != nil {
		return Tensor{}, vulkanV41ExactRMSNormStatusError(2, validationErr.Error())
	}
	for _, operand := range []struct {
		name  string
		t     Tensor
		bytes int
	}{{"x", x, rows * width * 4}, {"weight", weight, width * 4}} {
		b, ok := operand.t.buf.(*vulkanBuf)
		if operand.t.be != v || !ok || b == nil || b.ptr == nil || b.n < operand.bytes || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return Tensor{}, vulkanV41ExactRMSNormStatusError(2, operand.name+" must be a live, full-sized F32 device buffer owned by this backend")
		}
	}
	if C.fvk_have_v41_exact_rmsnorm() == 0 {
		return Tensor{}, vulkanV41ExactRMSNormStatusError(3, "exact RMSNorm module or binary32 float controls unavailable")
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, vulkanV41ExactRMSNormStatusError(status, "sticky submission fault")
	}
	outBuf = v.dallocForClass(rows*width*4, MemoryActivation, "V4.1 exact RMSNorm output")
	status := int(C.fvk_v41_exact_rmsnorm_f32(x.buf.(*vulkanBuf).ptr, weight.buf.(*vulkanBuf).ptr, outBuf.ptr,
		C.int(rows), C.int(width), C.float(eps)))
	if status != 0 {
		return Tensor{}, vulkanV41ExactRMSNormStatusError(status, "native dispatch refused")
	}
	if status = int(C.fvk_batch_flush_status()); status != 0 {
		return Tensor{}, vulkanV41ExactRMSNormStatusError(status, "checked completion failed")
	}
	outBuf.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, append([]int(nil), x.Shape...), nil, outBuf), nil
}
