//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

var _ V41EngramGateBackend = (*vulkanBackend)(nil)

func (v *vulkanBackend) SupportsV41EngramGate() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_engram_gate() != 0
}

// V41EngramGate fences the dispatch through the checked V4.1 completion path
// before publishing the fresh panel and status. Neither output enters the Go
// transient pool, and no public Read/Free/Upload is called under vulkanMu.
func (v *vulkanBackend) V41EngramGate(streams, projected, norms Tensor, rows, hidden int, eps float32) (next, status Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var nextBuf, statusBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41EngramGate")
		}
		if err != nil {
			for _, b := range []*vulkanBuf{nextBuf, statusBuf} {
				if b != nil && b.ptr != nil {
					C.fvk_free(b.ptr)
					b.ptr = nil
					b.n = 0
				}
			}
			next, status = Tensor{}, Tensor{}
		}
	}()
	if v == nil {
		return Tensor{}, Tensor{}, vulkanV41EngramGateStatusError(3, "Vulkan backend is nil")
	}
	if validationErr := validateV41EngramGate(streams, projected, norms, rows, hidden, eps); validationErr != nil {
		return Tensor{}, Tensor{}, vulkanV41EngramGateStatusError(2, validationErr.Error())
	}
	operands := [3]*vulkanBuf{}
	for i, operand := range []Tensor{streams, projected, norms} {
		b, ok := operand.buf.(*vulkanBuf)
		if operand.be != v || !ok || b == nil || b.ptr == nil || b.n < operand.Numel()*4 || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return Tensor{}, Tensor{}, vulkanV41EngramGateStatusError(2, "every operand must be a live, full-sized F32 device buffer owned by this backend")
		}
		operands[i] = b
	}
	if C.fvk_have_v41_engram_gate() == 0 {
		return Tensor{}, Tensor{}, vulkanV41EngramGateStatusError(3, "Engram gate module or binary32 float controls unavailable")
	}
	if code := int(C.fvk_submission_status()); code != 0 {
		return Tensor{}, Tensor{}, vulkanV41EngramGateStatusError(code, "sticky submission fault")
	}
	panel := rows * V41EngramGateStreams * hidden
	nextBuf = v.dallocForClass(panel*4, MemoryActivation, "V4.1 Engram gate streams")
	statusBuf = v.dallocForClass(rows*V41EngramGateStreams*4, MemoryActivation, "V4.1 Engram gate status")
	code := int(C.fvk_v41_engram_gate_f32(operands[0].ptr, operands[1].ptr, operands[2].ptr, nextBuf.ptr, statusBuf.ptr,
		C.int(rows), C.int(hidden), C.float(eps), C.float(V41EngramInvSqrt(float32(hidden)))))
	if code != 0 {
		return Tensor{}, Tensor{}, vulkanV41EngramGateStatusError(code, "native dispatch refused")
	}
	if code = int(C.fvk_batch_flush_status()); code != 0 {
		return Tensor{}, Tensor{}, vulkanV41EngramGateStatusError(code, "checked completion failed")
	}
	nextBuf.v41CheckedRead = true
	statusBuf.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{rows, V41EngramGateStreams, hidden}, nil, nextBuf),
		makeTensor(v, F32, RowMajor, []int{rows, V41EngramGateStreams}, nil, statusBuf), nil
}
