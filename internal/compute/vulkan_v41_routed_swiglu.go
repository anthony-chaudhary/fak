//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

var _ V41RoutedSwiGLUBackend = (*vulkanBackend)(nil)

// V41RoutedSwiGLUAvailable reports the registry V10 module, its binary32 float
// controls and a built pipeline. Absence leaves the model seam unbound, so
// device-only V4.1 refuses instead of running the activation on the host.
func (v *vulkanBackend) V41RoutedSwiGLUAvailable() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_routed_swiglu() != 0
}

// V41RoutedSwiGLU runs one synchronous checked dispatch and reads back only the
// 16-byte status record. The output stays resident for the down projection.
// No public Upload, Read or Free method runs while vulkanMu is held.
func (v *vulkanBackend) V41RoutedSwiGLU(gate, up Tensor, limit, weight float32) (out Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output, statusBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41RoutedSwiGLU")
		}
		for _, b := range []*vulkanBuf{statusBuf, output} {
			if b != nil && b.ptr != nil && (b == statusBuf || err != nil) {
				C.fvk_free(b.ptr)
				b.ptr, b.n = nil, 0
			}
		}
		if err != nil {
			out = Tensor{}
		}
	}()
	if v == nil || C.fvk_have_v41_routed_swiglu() == 0 {
		return Tensor{}, vulkanV41RoutedSwiGLUStatusError(3, "routed SwiGLU module or binary32 float controls unavailable")
	}
	count, invalid := validateV41RoutedSwiGLU(gate, up, limit, weight)
	if invalid != nil {
		return Tensor{}, vulkanV41RoutedSwiGLUStatusError(2, invalid.Error())
	}
	bytes := count * 4
	for _, operand := range []struct {
		name string
		t    Tensor
	}{{"gate", gate}, {"up", up}} {
		b, ok := operand.t.buf.(*vulkanBuf)
		if operand.t.be != v || !ok || b == nil || b.ptr == nil || b.n < bytes || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return Tensor{}, vulkanV41RoutedSwiGLUStatusError(2, operand.name+" must be a live, full-sized F32 device buffer owned by this backend")
		}
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, vulkanV41RoutedSwiGLUStatusError(status, "sticky submission fault")
	}
	if C.fvk_batch_foreign_active() {
		return Tensor{}, vulkanV41RoutedSwiGLUStatusError(2, "the synchronous routed activation does not join an active batch")
	}
	output = v.dallocForClass(bytes, MemoryActivation, "V4.1 routed SwiGLU output")
	statusBuf = v.dallocForClass(v41RoutedSwiGLUStatusBytes, MemoryActivation, "V4.1 routed SwiGLU status")
	if status := int(C.fvk_v41_routed_swiglu_f32(gate.buf.(*vulkanBuf).ptr, up.buf.(*vulkanBuf).ptr, output.ptr, statusBuf.ptr,
		C.int(count), C.float(limit), C.float(weight))); status != 0 {
		return Tensor{}, vulkanV41RoutedSwiGLUStatusError(status, "initialization or checked dispatch failed")
	}
	var words [4]uint32
	if status := int(C.fvk_v41_d2h(unsafe.Pointer(&words[0]), statusBuf.ptr, C.size_t(v41RoutedSwiGLUStatusBytes))); status != 0 {
		readErr := vulkanReadError(status)
		readErr.Site = "V41RoutedSwiGLU status readback"
		return Tensor{}, readErr
	}
	decodeErr := decodeV41RoutedSwiGLUStatus(words, count)
	var fault *V41RoutedSwiGLUFault
	if errors.As(decodeErr, &fault) {
		return Tensor{}, fault
	}
	if decodeErr != nil {
		return Tensor{}, &BackendError{Backend: "vulkan", Site: "V41RoutedSwiGLU", Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: decodeErr.Error(), Recovered: decodeErr}
	}
	output.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{count}, nil, output), nil
}

func vulkanV41RoutedSwiGLUStatusError(status int, reason string) *BackendError {
	err := vulkanV41RouterStatusError(status)
	err.Site = "V41RoutedSwiGLU"
	err.Message = fmt.Sprintf("routed SwiGLU failed closed (code %d): %s", status, reason)
	return err
}
