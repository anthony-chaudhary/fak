//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

var _ V41Q3KTokenPanelBackend = (*vulkanBackend)(nil)

// The candidate is default-on unless FAK_V41_Q3K_TOKEN_PANEL=0 and needs a live optional V19
// pipeline. A missing module leaves the historical scalar/batched path alone.
func (v *vulkanBackend) SupportsV41Q3KTokenPanel(weight Tensor, rows int) bool {
	if v == nil || !v41Q3KTokenPanelEnabled() || weight.Dtype != Q3_K || weight.Layout != RowMajor || len(weight.Shape) != 2 {
		return false
	}
	plan, err := PlanV41Q3KTokenPanel(weight.Shape[0], weight.Shape[1], rows)
	if err != nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	w, ok := v.ownedVulkanBuf(weight)
	return ok && w.n >= plan.WeightBytes && !bool(C.fvk_batch_foreign_active()) && C.fvk_have_v41_q3k_token_panel() != 0 &&
		C.fvk_v41_q3k_token_panel_shape(C.int(plan.Out), C.int(plan.In), C.int(plan.Rows)) != 0
}

func (v *vulkanBackend) V41Q3KTokenPanel(weight, input Tensor, rows int) (out Tensor, err error) {
	const site = "V41Q3KTokenPanel"
	plan, invalid := validateV41Q3KTokenPanel(weight, input, rows)
	if invalid != nil {
		return Tensor{}, vulkanV41RoutedIDError(site, 2, invalid.Error())
	}
	if v == nil || !v41Q3KTokenPanelEnabled() {
		return Tensor{}, vulkanV41RoutedIDError(site, 3, "token panel is disabled by FAK_V41_Q3K_TOKEN_PANEL=0")
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, site)
		}
		if err != nil {
			v.freeDeferredLocked(output)
			out = Tensor{}
		}
	}()
	if C.fvk_have_v41_q3k_token_panel() == 0 {
		return Tensor{}, vulkanV41RoutedIDError(site, 3, "optional token-panel module is unavailable")
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, vulkanV41RoutedIDError(site, status, "sticky submission fault")
	}
	if C.fvk_batch_foreign_active() || C.fvk_v41_q3k_token_panel_shape(C.int(plan.Out), C.int(plan.In), C.int(plan.Rows)) == 0 {
		return Tensor{}, vulkanV41RoutedIDError(site, 2, "foreign recording window or unsupported launch geometry")
	}
	w, wok := v.ownedVulkanBuf(weight)
	x, xok := v.ownedVulkanBuf(input)
	if !wok || !xok || w.n < plan.WeightBytes || x.n < plan.InputBytes {
		return Tensor{}, vulkanV41RoutedIDError(site, 2, "weight and input must have sufficient live storage owned by this backend")
	}
	output = v.dallocForClass(plan.OutputBytes, MemoryActivation, "V4.1 Q3_K token panel")
	out = makeTensor(v, F32, RowMajor, []int{plan.OutputBytes / 4}, nil, output)
	v.recordBatchDispatch(
		VulkanBufferAccess{BufferID: v.bufferIdentity(weight), Size: 0, Mode: VulkanAccessRead},
		VulkanBufferAccess{BufferID: v.bufferIdentity(input), Size: 0, Mode: VulkanAccessRead},
		VulkanBufferAccess{BufferID: v.bufferIdentity(out), Size: 0, Mode: VulkanAccessWrite},
	)
	if status := int(C.fvk_v41_q3k_token_panel_f32(w.ptr, x.ptr, output.ptr, C.int(plan.Out), C.int(plan.In), C.int(plan.Rows))); status != 0 {
		return Tensor{}, vulkanV41RoutedIDError(site, status, "selected token-panel dispatch failed")
	}
	output.v41CheckedRead = true
	return out, nil
}
