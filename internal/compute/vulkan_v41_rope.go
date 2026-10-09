//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "fmt"

var _ V41TailRoPEBackend = (*vulkanBackend)(nil)

// Native quarantine alone cannot fence Go's freeTransient pool, which bypasses
// fvk_malloc/fvk_free. Keep live owners and both pools unchanged on a sticky fault.
// Allocation/recycling callers already hold vulkanMu and expose typed panics at
// the existing request boundary; this is not a reset or context recovery path.
func requireVulkanSubmissionHealthyLocked(site string) {
	if status := int(C.fvk_submission_status()); status != 0 {
		err := vulkanV41RoPEStatusError(status)
		err.Site = site
		err.Message = fmt.Sprintf("sticky native submission fault %d; allocation/recycling refused", status)
		panic(err)
	}
}

func (v *vulkanBackend) SupportsV41TailRoPE() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_tail_rope_qk() != 0
}

func (v *vulkanBackend) v41PhysicalDeviceType() int {
	if v == nil {
		return 0
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return int(C.fvk_device_type())
}

// V41TailRoPEQK records the dedicated table-driven kernel. Outputs are caller
// owned, rather than registered as transient scratch. No Upload, Read, or Free
// method is called under vulkanMu. A pending batch is fenced by the caller's
// checked readback; an already-observed submission fault fails here immediately.
func (v *vulkanBackend) V41TailRoPEQK(q, kv, sinCos Tensor, heads, headDim, rotaryDim int) (qOut, kvOut Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var qBuf, kvBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41TailRoPEQK")
		}
		if err != nil {
			for _, b := range []*vulkanBuf{qBuf, kvBuf} {
				if b != nil && b.ptr != nil {
					C.fvk_free(b.ptr)
					b.ptr = nil
					b.n = 0
				}
			}
			qOut, kvOut = Tensor{}, Tensor{}
		}
	}()
	if v == nil || C.fvk_have_v41_tail_rope_qk() == 0 {
		return Tensor{}, Tensor{}, vulkanV41RoPEStatusError(3)
	}
	qBytes, kvBytes, tableBytes, validationErr := validateV41TailRoPE(q, kv, sinCos, heads, headDim, rotaryDim)
	if validationErr != nil {
		return Tensor{}, Tensor{}, &BackendError{Backend: "vulkan", Site: "V41TailRoPEQK", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: validationErr.Error()}
	}
	for _, operand := range []struct {
		name  string
		t     Tensor
		bytes int
	}{
		{"q", q, qBytes}, {"kv", kv, kvBytes}, {"sinCos", sinCos, tableBytes},
	} {
		b, ok := operand.t.buf.(*vulkanBuf)
		if operand.t.be != v || !ok || b == nil || b.ptr == nil || b.n < operand.bytes || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return Tensor{}, Tensor{}, &BackendError{Backend: "vulkan", Site: "V41TailRoPEQK", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: operand.name + " must be a live, full-sized F32 buffer owned by this backend"}
		}
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, Tensor{}, vulkanV41RoPEStatusError(status)
	}
	qBuf = v.dallocForClass(qBytes, MemoryActivation, "V4.1 RoPE Q output")
	kvBuf = v.dallocForClass(kvBytes, MemoryActivation, "V4.1 RoPE KV output")
	status := v41TailRoPEQKStatusLocked(q.buf.(*vulkanBuf), kv.buf.(*vulkanBuf), qBuf, kvBuf, sinCos.buf.(*vulkanBuf), heads, headDim, rotaryDim)
	if status != 0 {
		return Tensor{}, Tensor{}, vulkanV41RoPEStatusError(status)
	}
	return makeTensor(v, F32, RowMajor, []int{heads, headDim}, nil, qBuf), makeTensor(v, F32, RowMajor, []int{headDim}, nil, kvBuf), nil
}

// Caller holds vulkanMu and owns the live handles. Keeping this adapter separate
// lets the native rejection witness exercise the actual alias/extent checks.
func v41TailRoPEQKStatusLocked(q, kv, qOut, kvOut, sinCos *vulkanBuf, heads, headDim, rotaryDim int) int {
	return int(C.fvk_v41_tail_rope_qk_f32(q.ptr, kv.ptr, qOut.ptr, kvOut.ptr, sinCos.ptr, C.int(heads), C.int(headDim), C.int(rotaryDim)))
}

func vulkanV41RoPEStatusError(status int) *BackendError {
	class, cause := VulkanClassExecutionFailed, ErrVulkanExecutionFailed
	switch {
	case status == -4: // VK_ERROR_DEVICE_LOST
		class, cause = VulkanClassDeviceLost, ErrVulkanDeviceLost
	case status < 0:
		class, cause = VulkanClassSubmissionFailed, ErrVulkanSubmissionFailed
	case status == 2:
		class, cause = VulkanClassInvalidGeometry, ErrVulkanInvalidGeometry
	}
	return &BackendError{Backend: "vulkan", Site: "V41TailRoPEQK", Class: class, Err: cause, Message: fmt.Sprintf("dedicated tail RoPE failed closed (code %d)", status)}
}
