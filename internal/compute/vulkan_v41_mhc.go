//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "fmt"

var _ V41MHCBackend = (*vulkanBackend)(nil)

func (v *vulkanBackend) SupportsV41MHC() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_mhc() != 0
}

func (v *vulkanBackend) V41MHCInitialStreams(x Tensor, rows, hidden int) (Tensor, error) {
	return v.v41MHC(V41MHCOpInitial, rows, hidden, 0, 0, x)
}

func (v *vulkanBackend) V41MHCSplit(mixes, scale, base Tensor, rows, iters int, eps float32) (Tensor, error) {
	return v.v41MHC(V41MHCOpSplit, rows, 0, iters, eps, mixes, scale, base)
}

func (v *vulkanBackend) V41MHCPre(streams, coeffs Tensor, rows, hidden int) (Tensor, error) {
	return v.v41MHC(V41MHCOpPre, rows, hidden, 0, 0, streams, coeffs)
}

func (v *vulkanBackend) V41MHCPost(x, streams, coeffs Tensor, rows, hidden int) (Tensor, error) {
	return v.v41MHC(V41MHCOpPost, rows, hidden, 0, 0, x, streams, coeffs)
}

// v41MHC records one mode of the shared mHC module. The output is caller
// owned, never registered as transient scratch. No Upload, Read, or Free method
// is called under vulkanMu. A pending batch is fenced by the caller's checked
// readback; an already-observed submission fault fails here immediately.
func (v *vulkanBackend) v41MHC(op V41MHCOp, rows, hidden, iters int, eps float32, operands ...Tensor) (out Tensor, err error) {
	site := "V41MHC/" + op.String()
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var outBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, site)
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
	if v == nil || C.fvk_have_v41_mhc() == 0 {
		return Tensor{}, vulkanV41MHCStatusError(site, 3)
	}
	inputs, outShape, validationErr := validateV41MHC(op, rows, hidden, iters, eps, operands...)
	if validationErr != nil {
		return Tensor{}, &BackendError{Backend: "vulkan", Site: site, Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: validationErr.Error(), Recovered: validationErr}
	}
	var bufs [3]*vulkanBuf
	for i, operand := range inputs {
		b, ok := operand.t.buf.(*vulkanBuf)
		if operand.t.be != v || !ok || b == nil || b.ptr == nil || b.n < 4*v41MHCElements(operand.shape) || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return Tensor{}, &BackendError{Backend: "vulkan", Site: site, Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: operand.name + " must be a live, full-sized F32 buffer owned by this backend"}
		}
		bufs[i] = b
	}
	// Unused bindings repeat the first input; every mode reads them read-only.
	for i := len(inputs); i < len(bufs); i++ {
		bufs[i] = bufs[0]
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, vulkanV41MHCStatusError(site, status)
	}
	outBytes := 4 * v41MHCElements(outShape)
	outBuf = v.dallocForClass(outBytes, MemoryActivation, "V4.1 mHC "+op.String()+" output")
	status := v41MHCStatusLocked(op, bufs[0], bufs[1], bufs[2], outBuf, rows, hidden, iters, eps)
	if status != 0 {
		return Tensor{}, vulkanV41MHCStatusError(site, status)
	}
	return makeTensor(v, F32, RowMajor, outShape, nil, outBuf), nil
}

// Caller holds vulkanMu and owns the live handles. Keeping this adapter separate
// lets the native rejection witness exercise the actual alias/extent checks.
func v41MHCStatusLocked(op V41MHCOp, a, b, c, out *vulkanBuf, rows, hidden, iters int, eps float32) int {
	return int(C.fvk_v41_mhc_f32(C.int(op), a.ptr, b.ptr, c.ptr, out.ptr, C.int(rows), C.int(hidden), C.int(iters), C.float(eps)))
}

func vulkanV41MHCStatusError(site string, status int) *BackendError {
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
	return &BackendError{Backend: "vulkan", Site: site, Class: class, Err: cause, Message: fmt.Sprintf("dedicated mHC operation failed closed (code %d)", status)}
}
