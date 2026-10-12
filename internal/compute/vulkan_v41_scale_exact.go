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

var _ V41ScaleExactBackend = (*vulkanBackend)(nil)

// v41ScaleExactModuleABI runs the shim's exact scalar proof on caller bytes.
// It needs no device and is no numerical or physical qualification.
func v41ScaleExactModuleABI(code []byte) bool {
	if len(code) == 0 {
		return false
	}
	return C.fvk_v41_scale_exact_module_abi(unsafe.Pointer(&code[0]), C.size_t(len(code))) != 0
}

// V41ScaleExactAvailable reports the registry V13 strict scalar module: loaded
// only when the selected device reports binary32 DenormPreserve,
// SignedZeroInfNanPreserve and RoundingModeRTE and the exact scalar proof
// admitted the loaded bytes.
func (v *vulkanBackend) V41ScaleExactAvailable() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_scale_exact() != 0
}

func (v *vulkanBackend) V41ScaleExact(input Tensor, scale float32) (out Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output *vulkanBuf
	defer func() {
		if r := recover(); r != nil {
			err = ClassifyVulkanPanic(r, "V41ScaleExact")
		}
		if err != nil {
			if output != nil && output.ptr != nil {
				C.fvk_free(output.ptr)
				output.ptr, output.n = nil, 0
			}
			out = Tensor{}
		}
	}()
	if v == nil || C.fvk_have_v41_scale_exact() == 0 {
		return Tensor{}, vulkanV41ScaleExactStatusError(3, "strict module or selected-device binary32 modes unavailable")
	}
	bytes, invalid := validateV41ScaleExact(input, scale)
	if invalid != nil {
		return Tensor{}, vulkanV41ScaleExactStatusError(2, invalid.Error())
	}
	source, ok := input.buf.(*vulkanBuf)
	if input.be != v || !ok || source == nil || source.ptr == nil || source.n < bytes || source.scalePtr != nil || len(source.q8Chunks) != 0 {
		return Tensor{}, vulkanV41ScaleExactStatusError(2, "input must be a live full-sized F32 allocation owned by this backend")
	}
	// One invocation writes one element: never clamp ceil(N/64) and publish
	// an unwritten suffix.
	if C.fvk_v41_scale_exact_count_supported(C.int(input.Shape[0])) == 0 {
		return Tensor{}, vulkanV41ScaleExactStatusError(2, "row exceeds the selected device's workgroup limit")
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, vulkanV41ScaleExactStatusError(status, "sticky submission failure")
	}
	output = v.dallocForClass(bytes, MemoryActivation, "V4.1 exact scale output")
	if status := int(C.fvk_v41_scale_exact_f32(source.ptr, output.ptr, C.int(input.Shape[0]), C.float(scale))); status != 0 {
		return Tensor{}, vulkanV41ScaleExactStatusError(status, "checked dispatch refused")
	}
	if status := int(C.fvk_batch_flush_status()); status != 0 {
		return Tensor{}, vulkanV41ScaleExactStatusError(status, "checked completion failed")
	}
	output.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{input.Shape[0]}, nil, output), nil
}

func vulkanV41ScaleExactStatusError(status int, reason string) *BackendError {
	err := vulkanV41IndexerScoreStatusError(status, reason)
	err.Site = "V41ScaleExact"
	err.Message = fmt.Sprintf("exact scalar multiply failed closed (code %d): %s", status, reason)
	return err
}
