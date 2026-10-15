//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

var _ V41RoutedDownPanelRowBackend = (*vulkanBackend)(nil)

func (v *vulkanBackend) SupportsV41RoutedDownPanelRow(panel Tensor, row int) bool {
	if v == nil {
		return false
	}
	rows, count, err := validateV41RoutedDownPanelRow(panel, row)
	if err != nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	b, ok := panel.buf.(*vulkanBuf)
	return panel.be == v && ok && b != nil && b.ptr != nil && b.n >= rows*count*4 && b.scalePtr == nil && len(b.q8Chunks) == 0 && C.fvk_have_v41_routed_down_panel_row() != 0 && C.fvk_v41_routed_down_panel_row_shape(C.int(rows), C.int(count), C.int(row)) != 0
}

// Preserve ordinary publication's submission-state precedence, fresh ownership
// and native deferred-free quarantine. Only fixed RoutedDown site21 is exposed.
func (v *vulkanBackend) V41RoutedDownPanelRowDeferred(input Tensor, row int) (out Tensor, status V41DeferredStatus, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output, statusBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			if primary, ok := recovered.(error); ok {
				err = primary
			} else {
				err = ClassifyVulkanPanic(recovered, "V41RoutedDownPanelRow")
			}
		}
		if err != nil {
			v.freeDeferredLocked(statusBuf, output)
			out, status = Tensor{}, V41DeferredStatus{}
		}
	}()
	fail := func(code int, reason string) error {
		e := vulkanV41AttentionBF16StatusError(code, reason)
		e.Site = "V41RoutedDownPanelRow"
		return e
	}
	if v == nil {
		return Tensor{}, V41DeferredStatus{}, fail(3, "nil backend")
	}
	if code := int(C.fvk_v41_attention_bf16_state()); code != 0 {
		return Tensor{}, V41DeferredStatus{}, fail(code, "submission state refuses publication")
	}
	if C.fvk_have_v41_routed_down_panel_row() == 0 {
		return Tensor{}, V41DeferredStatus{}, fail(3, "ordinary checked module unavailable")
	}
	rows, count, invalid := validateV41RoutedDownPanelRow(input, row)
	if invalid != nil {
		return Tensor{}, V41DeferredStatus{}, fail(2, invalid.Error())
	}
	bytes := count * 4
	source, ok := input.buf.(*vulkanBuf)
	if input.be != v || !ok || source == nil || source.ptr == nil || source.n < rows*bytes || source.scalePtr != nil || len(source.q8Chunks) != 0 {
		return Tensor{}, V41DeferredStatus{}, fail(2, "input must be live F32 storage owned by this backend")
	}
	if C.fvk_v41_routed_down_panel_row_shape(C.int(rows), C.int(count), C.int(row)) == 0 {
		return Tensor{}, V41DeferredStatus{}, fail(2, "selected device count limit")
	}
	output = v.dallocForClass(bytes, MemoryActivation, "V4.1 routed down row publication output")
	statusBuf = v.dallocForClass(v41BF16CheckedStatusBytes, MemoryActivation, "V4.1 routed down row publication status")
	if code := int(C.fvk_v41_routed_down_panel_row_f32(source.ptr, output.ptr, statusBuf.ptr, C.int(rows), C.int(count), C.int(row))); code != 0 {
		return Tensor{}, V41DeferredStatus{}, fail(code, "initialization or checked dispatch failed")
	}
	output.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{count}, nil, output), V41DeferredStatus{
		Kind: V41DeferredBF16Publish, Count: count, Site: V41BF16PublishRoutedDown,
		Record: makeTensor(v, F32, RowMajor, []int{v41BF16CheckedStatusBytes / 4}, nil, statusBuf),
	}, nil
}
