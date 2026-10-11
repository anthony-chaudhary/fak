//go:build v41_logit_scale_witness && vulkan && (windows || linux) && cgo

package compute

/*
#cgo v41_indexer_witness CFLAGS: -DFAK_V41_LOGIT_SCALE_CONFLICT
#cgo v41_bf16_boundary_witness CFLAGS: -DFAK_V41_LOGIT_SCALE_CONFLICT
#ifdef FAK_V41_LOGIT_SCALE_CONFLICT
#error "v41_logit_scale_witness cannot be combined with other V4.1 witness archives"
#endif
#define FAK_V41_LOGIT_SCALE_WITNESS
#include "vulkan_backend.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

var _ V41LogitScaleBackend = (*vulkanBackend)(nil)

func v41LogitScaleModuleABI(code []byte) bool {
	if len(code) == 0 {
		return false
	}
	return C.fvk_v41_logit_scale_module_abi(unsafe.Pointer(&code[0]), C.size_t(len(code))) != 0
}

// Only the separately linked witness archive supplies these symbols. The
// native predicate requires the exact module and actual selected-device
// DenormPreserve, SignedZeroInfNanPreserve and RoundingModeRTE F32 properties.
// Availability is not physical qualification or production registry provenance.
func (v *vulkanBackend) V41LogitScaleAvailable() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_logit_scale() != 0
}

func (v *vulkanBackend) V41LogitScale(input Tensor, scale float32) (out Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output *vulkanBuf
	defer func() {
		if r := recover(); r != nil {
			err = ClassifyVulkanPanic(r, "V41LogitScale")
		}
		if err != nil {
			if output != nil && output.ptr != nil {
				C.fvk_free(output.ptr)
				output.ptr, output.n = nil, 0
			}
			out = Tensor{}
		}
	}()
	if v == nil || C.fvk_have_v41_logit_scale() == 0 {
		return Tensor{}, vulkanV41LogitScaleStatusError(3, "exact module or selected-device binary32 modes unavailable")
	}
	bytes, invalid := validateV41LogitScale(input, scale)
	if invalid != nil {
		return Tensor{}, vulkanV41LogitScaleStatusError(2, invalid.Error())
	}
	source, ok := input.buf.(*vulkanBuf)
	if input.be != v || !ok || source == nil || source.ptr == nil || source.n < bytes || source.scalePtr != nil || len(source.q8Chunks) != 0 {
		return Tensor{}, vulkanV41LogitScaleStatusError(2, "input must be a live full-sized F32 allocation owned by this backend")
	}
	// One invocation writes one element: never clamp ceil(N/64) and publish
	// an unwritten suffix. The native check is read-only and precedes output
	// allocation; the model's input upload has already occurred.
	if C.fvk_v41_logit_scale_count_supported(C.int(input.Shape[0])) == 0 {
		return Tensor{}, vulkanV41LogitScaleStatusError(2, "row exceeds the selected device's workgroup limit")
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, vulkanV41LogitScaleStatusError(status, "sticky submission failure")
	}
	output = v.dallocForClass(bytes, MemoryActivation, "V4.1 logit scale output")
	if status := int(C.fvk_v41_logit_scale_f32(source.ptr, output.ptr, C.int(input.Shape[0]), C.float(scale))); status != 0 {
		return Tensor{}, vulkanV41LogitScaleStatusError(status, "checked dispatch refused")
	}
	if status := int(C.fvk_batch_flush_status()); status != 0 {
		return Tensor{}, vulkanV41LogitScaleStatusError(status, "checked completion failed")
	}
	output.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{input.Shape[0]}, nil, output), nil
}

func vulkanV41LogitScaleStatusError(status int, reason string) *BackendError {
	err := vulkanV41IndexerScoreStatusError(status, reason)
	err.Site = "V41LogitScale"
	err.Message = fmt.Sprintf("dedicated logit scale failed closed (code %d): %s", status, reason)
	return err
}
