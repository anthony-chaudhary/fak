//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

var _ V41RoutedSwiGLUPanelChunkBackend = (*vulkanBackend)(nil)

// SupportsV41RoutedSwiGLUPanelChunk requires the separately marked optional
// module and full live panels. The module remains explicitly opt-in; this
// adapter does not select a model execution path.
func (v *vulkanBackend) SupportsV41RoutedSwiGLUPanelChunk(gate, up Tensor, first, rows int) bool {
	if v == nil || (rows != 2 && rows != 4 && rows != 8) {
		return false
	}
	plan, err := validateV41RoutedSwiGLUPanelChunk(gate, up, first, rows, 0, make([]float32, rows))
	if err != nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if C.fvk_have_v41_routed_swiglu_panel_chunk() == 0 || C.fvk_v41_routed_swiglu_panel_chunk_shape(C.int(plan.PanelRows), C.int(plan.Width), C.int(plan.First), C.int(plan.Rows)) == 0 {
		return false
	}
	for _, t := range []Tensor{gate, up} {
		b, ok := t.buf.(*vulkanBuf)
		if t.be != v || !ok || b == nil || b.ptr == nil || b.n < plan.PanelBytes || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return false
		}
	}
	return gate.buf.(*vulkanBuf).ptr != up.buf.(*vulkanBuf).ptr
}

// V41RoutedSwiGLUPanelChunkDeferred borrows whole panels and returns one packed
// chunk plus exact row-local status records. Selected errors are never replayed; native
// checked-dispatch and deferred-free quarantine semantics remain authoritative.
func (v *vulkanBackend) V41RoutedSwiGLUPanelChunkDeferred(gate, up Tensor, first int, limit float32, weights []float32) (out Tensor, status V41DeferredStatus, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output, statusBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41RoutedSwiGLUPanelChunk")
		}
		if err != nil {
			v.freeDeferredLocked(statusBuf, output)
			out, status = Tensor{}, V41DeferredStatus{}
		}
	}()
	if v == nil || C.fvk_have_v41_routed_swiglu_panel_chunk() == 0 {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(3, "routed SwiGLU module or binary32 float controls unavailable")
	}
	plan, invalid := validateV41RoutedSwiGLUPanelChunk(gate, up, first, len(weights), limit, weights)
	if invalid != nil {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, invalid.Error())
	}
	count, bytes := plan.Width, plan.OutputBytes
	for _, operand := range []struct {
		name string
		t    Tensor
	}{{"gate", gate}, {"up", up}} {
		b, ok := operand.t.buf.(*vulkanBuf)
		if operand.t.be != v || !ok || b == nil || b.ptr == nil || b.n < plan.PanelBytes || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, operand.name+" must be a live, full-sized F32 device buffer owned by this backend")
		}
	}
	if gate.buf.(*vulkanBuf).ptr == up.buf.(*vulkanBuf).ptr || C.fvk_v41_routed_swiglu_panel_chunk_shape(C.int(plan.PanelRows), C.int(plan.Width), C.int(plan.First), C.int(plan.Rows)) == 0 {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, "panel alias or unsupported native row geometry")
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(status, "sticky submission fault")
	}
	if C.fvk_batch_foreign_active() {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, "the routed activation does not join a foreign batch")
	}
	output = v.dallocForClass(bytes, MemoryActivation, "V4.1 routed panel chunk SwiGLU output")
	statusBuf = v.dallocForClass(plan.StatusBytes, MemoryActivation, "V4.1 routed panel chunk SwiGLU status")
	var route [8]C.float
	for i, weight := range weights {
		route[i] = C.float(weight)
	}
	if status := int(C.fvk_v41_routed_swiglu_panel_chunk_f32(gate.buf.(*vulkanBuf).ptr, up.buf.(*vulkanBuf).ptr, output.ptr, statusBuf.ptr,
		C.int(plan.PanelRows), C.int(count), C.int(first), C.int(plan.Rows), C.float(limit), &route[0])); status != 0 {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(status, "initialization or checked dispatch failed")
	}
	output.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{plan.Rows * count}, nil, output), V41DeferredStatus{
		Kind: V41DeferredRoutedSwiGLU, Count: count, Records: plan.Rows,
		Record: makeTensor(v, F32, RowMajor, []int{plan.StatusBytes / 4}, nil, statusBuf),
	}, nil
}
