//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

var _ V41RoutedDownPanelChunkBackend = (*vulkanBackend)(nil)

func (v *vulkanBackend) SupportsV41RoutedDownPanelChunk(panel Tensor, first, chunk int) bool {
	if v == nil {
		return false
	}
	plan, err := validateV41RoutedDownPanelChunk(panel, first, chunk)
	if err != nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	b, ok := panel.buf.(*vulkanBuf)
	return panel.be == v && ok && b != nil && b.ptr != nil && b.n >= plan.PanelBytes && b.scalePtr == nil && len(b.q8Chunks) == 0 && C.fvk_have_v41_routed_down_panel_chunk() != 0 && C.fvk_v41_routed_down_panel_chunk_shape(C.int(plan.PanelRows), C.int(plan.Width), C.int(plan.First), C.int(plan.Rows)) != 0
}

// Preserve ordinary publication's submission-state precedence, fresh ownership
// and native deferred-free quarantine. Only fixed RoutedDown site21 is exposed.
func (v *vulkanBackend) V41RoutedDownPanelChunkDeferred(input Tensor, first, chunk int) (out Tensor, status V41DeferredStatus, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output, statusBuf *vulkanBuf
	completed := false
	defer func() {
		if recovered := recover(); recovered != nil {
			if primary, ok := recovered.(error); ok {
				err = primary
			} else {
				err = ClassifyVulkanPanic(recovered, "V41RoutedDownPanelChunk")
			}
		}
		if !completed {
			// A nil recovery also covers Goexit and legacy panic(nil). No
			// allocation escapes an interrupted result construction. Goexit
			// still propagates after these defers; it cannot publish a result.
			if err == nil {
				err = ClassifyVulkanPanic("incomplete deferred publication", "V41RoutedDownPanelChunk")
			}
			out, status = Tensor{}, V41DeferredStatus{}
			// Schedule both releases before either runs. Native Free retains
			// its ordinary in-flight quarantine contract; a cleanup panic
			// cannot replace the selected primary error or strand its peer.
			release := func(b *vulkanBuf) {
				defer func() { _ = recover() }()
				v.freeDeferredLocked(b)
			}
			defer release(output)
			defer release(statusBuf)
		}
	}()
	fail := func(code int, reason string) error {
		e := vulkanV41AttentionBF16StatusError(code, reason)
		e.Site = "V41RoutedDownPanelChunk"
		return e
	}
	if v == nil {
		return Tensor{}, V41DeferredStatus{}, fail(3, "nil backend")
	}
	if code := int(C.fvk_v41_attention_bf16_state()); code != 0 {
		return Tensor{}, V41DeferredStatus{}, fail(code, "submission state refuses publication")
	}
	if C.fvk_have_v41_routed_down_panel_chunk() == 0 {
		return Tensor{}, V41DeferredStatus{}, fail(3, "ordinary checked module unavailable")
	}
	plan, invalid := validateV41RoutedDownPanelChunk(input, first, chunk)
	if invalid != nil {
		return Tensor{}, V41DeferredStatus{}, fail(2, invalid.Error())
	}
	bytes := plan.OutputBytes
	source, ok := input.buf.(*vulkanBuf)
	if input.be != v || !ok || source == nil || source.ptr == nil || source.n < plan.PanelBytes || source.scalePtr != nil || len(source.q8Chunks) != 0 {
		return Tensor{}, V41DeferredStatus{}, fail(2, "input must be live F32 storage owned by this backend")
	}
	if C.fvk_v41_routed_down_panel_chunk_shape(C.int(plan.PanelRows), C.int(plan.Width), C.int(plan.First), C.int(plan.Rows)) == 0 {
		return Tensor{}, V41DeferredStatus{}, fail(2, "selected device count limit")
	}
	output = v.dallocForClass(bytes, MemoryActivation, "V4.1 routed down chunk publication output")
	statusBuf = v.dallocForClass(plan.StatusBytes, MemoryActivation, "V4.1 routed down chunk publication status")
	if code := int(C.fvk_v41_routed_down_panel_chunk_f32(source.ptr, output.ptr, statusBuf.ptr, C.int(plan.PanelRows), C.int(plan.Width), C.int(plan.First), C.int(plan.Rows))); code != 0 {
		return Tensor{}, V41DeferredStatus{}, fail(code, "initialization or checked dispatch failed")
	}
	output.v41CheckedRead = true
	out = makeTensor(v, F32, RowMajor, []int{plan.Rows * plan.Width}, nil, output)
	status = V41DeferredStatus{
		Kind: V41DeferredBF16Publish, Count: plan.Width, Records: plan.Rows, Site: V41BF16PublishRoutedDown,
		Record: makeTensor(v, F32, RowMajor, []int{plan.StatusBytes / 4}, nil, statusBuf),
	}
	completed = true
	return out, status, nil
}
