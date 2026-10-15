//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

var _ V41RoutedSwiGLUPanelRowBackend = (*vulkanBackend)(nil)

// SupportsV41RoutedSwiGLUPanelRow requires the separately marked optional
// module and full live panels. The module remains explicitly opt-in; this
// adapter does not select a model execution path.
func (v *vulkanBackend) SupportsV41RoutedSwiGLUPanelRow(gate, up Tensor, row int) bool {
	if v == nil {
		return false
	}
	plan, err := validateV41RoutedSwiGLUPanelRow(gate, up, row, 0, 0)
	if err != nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if C.fvk_have_v41_routed_swiglu_panel_row() == 0 || C.fvk_v41_routed_swiglu_panel_row_shape(C.int(plan.Rows), C.int(plan.Width), C.int(plan.Row)) == 0 {
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

// V41RoutedSwiGLUPanelRowDeferred borrows whole panels and returns exactly one
// row plus a row-local status. Selected errors are never replayed; native
// checked-dispatch and deferred-free quarantine semantics remain authoritative.
func (v *vulkanBackend) V41RoutedSwiGLUPanelRowDeferred(gate, up Tensor, row int, limit, weight float32) (out Tensor, status V41DeferredStatus, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output, statusBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41RoutedSwiGLUPanelRow")
		}
		if err != nil {
			v.freeDeferredLocked(statusBuf, output)
			out, status = Tensor{}, V41DeferredStatus{}
		}
	}()
	if v == nil || C.fvk_have_v41_routed_swiglu_panel_row() == 0 {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(3, "routed SwiGLU module or binary32 float controls unavailable")
	}
	plan, invalid := validateV41RoutedSwiGLUPanelRow(gate, up, row, limit, weight)
	if invalid != nil {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, invalid.Error())
	}
	count, bytes := plan.Width, plan.RowBytes
	for _, operand := range []struct {
		name string
		t    Tensor
	}{{"gate", gate}, {"up", up}} {
		b, ok := operand.t.buf.(*vulkanBuf)
		if operand.t.be != v || !ok || b == nil || b.ptr == nil || b.n < plan.PanelBytes || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, operand.name+" must be a live, full-sized F32 device buffer owned by this backend")
		}
	}
	if gate.buf.(*vulkanBuf).ptr == up.buf.(*vulkanBuf).ptr || C.fvk_v41_routed_swiglu_panel_row_shape(C.int(plan.Rows), C.int(plan.Width), C.int(plan.Row)) == 0 {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, "panel alias or unsupported native row geometry")
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(status, "sticky submission fault")
	}
	if C.fvk_batch_foreign_active() {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, "the routed activation does not join a foreign batch")
	}
	output = v.dallocForClass(bytes, MemoryActivation, "V4.1 routed panel row SwiGLU output")
	statusBuf = v.dallocForClass(v41RoutedSwiGLUStatusBytes, MemoryActivation, "V4.1 routed panel row SwiGLU status")
	if status := int(C.fvk_v41_routed_swiglu_panel_row_f32(gate.buf.(*vulkanBuf).ptr, up.buf.(*vulkanBuf).ptr, output.ptr, statusBuf.ptr,
		C.int(plan.Rows), C.int(count), C.int(row), C.float(limit), C.float(weight))); status != 0 {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(status, "initialization or checked dispatch failed")
	}
	output.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{count}, nil, output), V41DeferredStatus{
		Kind: V41DeferredRoutedSwiGLU, Count: count,
		Record: makeTensor(v, F32, RowMajor, []int{v41RoutedSwiGLUStatusBytes / 4}, nil, statusBuf),
	}, nil
}
