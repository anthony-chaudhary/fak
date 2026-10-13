//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "unsafe"

var _ V41RoutedFusedBackend = (*vulkanBackend)(nil)

// V41RoutedFusedAvailable reports the registry V16 fused routed modules on this
// device and the binary32 float controls the fused SwiGLU lane needs.
func (v *vulkanBackend) V41RoutedFusedAvailable() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_routed_fused() != 0
}

// SupportsV41RoutedFusedGateUp serves the pinned V4.1 Flash expert pair: a Q2_K
// gate and a Q2_K up share the fused Q2_K GEMV.
func (v *vulkanBackend) SupportsV41RoutedFusedGateUp(gate, up Dtype) bool {
	return gate == Q2_K && up == Q2_K
}

// SupportsV41RoutedFusedDown serves the pinned V4.1 Flash down projection, a
// Q3_K expert matrix.
func (v *vulkanBackend) SupportsV41RoutedFusedDown(down Dtype) bool { return down == Q3_K }

// fusedVulkanSlotPtrs collects one live weight pointer per slot, refusing any
// slot this backend does not own. Caller holds vulkanMu.
func (v *vulkanBackend) fusedVulkanSlotPtrs(weights []Tensor) ([]unsafe.Pointer, bool) {
	ptrs := make([]unsafe.Pointer, len(weights))
	for i, w := range weights {
		b, ok := v.ownedVulkanBuf(w)
		if !ok {
			return nil, false
		}
		ptrs[i] = b.ptr
	}
	return ptrs, true
}

func (v *vulkanBackend) V41RoutedGateUpSwiGLUDeferred(gates, ups []Tensor, ids Tensor, x Tensor, picks int, limit float32) (out Tensor, status V41DeferredStatus, err error) {
	const site = "V41RoutedFusedGateUp"
	supported := func(dt Dtype) bool { return v.SupportsV41RoutedFusedGateUp(dt, dt) }
	rows, in, invalid := validateV41RoutedFusedWeights("gate", gates, picks, supported)
	var xStride int
	if invalid == nil {
		if _, upIn, upInvalid := validateV41RoutedFusedWeights("up", ups, picks, supported); upInvalid != nil {
			invalid = upInvalid
		} else if upIn != in {
			invalid = vulkanV41RoutedIDError(site, 2, "gate and up input widths differ")
		}
	}
	if invalid == nil {
		invalid = validateV41RoutedIDsTensor(ids, picks)
	}
	if invalid == nil {
		xStride, invalid = validateV41RoutedFusedActivation(x, picks, in)
	}
	if invalid == nil && !finite32(limit) {
		invalid = vulkanV41RoutedIDError(site, 2, "limit is not finite")
	}
	if invalid != nil {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, invalid.Error())
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output, statusBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, site)
		}
		if err != nil {
			v.freeDeferredLocked(statusBuf, output)
			out, status = Tensor{}, V41DeferredStatus{}
		}
	}()
	if C.fvk_have_v41_routed_fused() == 0 {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(3, "fused routed modules or binary32 float controls unavailable")
	}
	gatePtrs, ok := v.fusedVulkanSlotPtrs(gates)
	if !ok {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, "every gate slot must be live weight storage owned by this backend")
	}
	upPtrs, ok := v.fusedVulkanSlotPtrs(ups)
	if !ok {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, "every up slot must be live weight storage owned by this backend")
	}
	xb, xOK := v.ownedVulkanBuf(x)
	ib, iOK := v.ownedVulkanBuf(ids)
	if !xOK || !iOK {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, "activation and ids must be live F32 storage owned by this backend")
	}
	if code := int(C.fvk_submission_status()); code != 0 {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(code, "sticky submission fault")
	}
	if C.fvk_batch_foreign_active() {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, "the routed activation does not join a foreign batch")
	}
	output = v.dallocForClass(picks*rows*4, MemoryActivation, "V4.1 routed fused mid")
	statusBuf = v.dallocForClass(picks*v41RoutedSwiGLUStatusBytes, MemoryActivation, "V4.1 routed fused gate/up status")
	if code := int(C.fvk_v41_routed_gate_up_swiglu_id_f32((*unsafe.Pointer)(unsafe.Pointer(&gatePtrs[0])), (*unsafe.Pointer)(unsafe.Pointer(&upPtrs[0])),
		C.int(len(gatePtrs)), xb.ptr, ib.ptr, output.ptr, statusBuf.ptr,
		C.int(rows), C.int(in), C.int(picks), C.int(xStride), C.float(limit))); code != 0 {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(code, "initialization or checked fused dispatch failed")
	}
	output.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{picks * rows}, nil, output), V41DeferredStatus{
		Kind: V41DeferredRoutedSwiGLU, Count: rows, Records: picks,
		Record: makeTensor(v, F32, RowMajor, []int{picks * v41RoutedSwiGLUStatusBytes / 4}, nil, statusBuf),
	}, nil
}

func (v *vulkanBackend) V41RoutedDownSumDeferred(downs []Tensor, mid, ids Tensor, picks int, site uint32) (out Tensor, status V41DeferredStatus, err error) {
	fail := func(code int, reason string) error {
		e := vulkanV41AttentionBF16StatusError(code, reason)
		e.Site = "V41BF16Publish"
		return e
	}
	rows, in, invalid := validateV41RoutedFusedWeights("down", downs, picks, v.SupportsV41RoutedFusedDown)
	if invalid == nil {
		invalid = validateV41RoutedIDsTensor(ids, picks)
	}
	if invalid == nil && !V41BF16PublishSite(site) {
		invalid = vulkanV41RoutedIDError("V41BF16Publish", 2, "invalid ordinary publication site")
	}
	// The mid operand is the fused gate/up output: an unquantized F32
	// [picks*in] row set, one row per pick.
	if invalid == nil && (mid.Dtype != F32 || mid.Layout != RowMajor || mid.Quant != nil || len(mid.Shape) != 1 ||
		mid.Shape[0] <= 0 || mid.Shape[0]%picks != 0 || mid.Shape[0]/picks != in) {
		invalid = vulkanV41RoutedIDError("V41BF16Publish", 2, "mid must be the unquantized row-major F32 [picks*inter] gate/up output")
	}
	if invalid != nil {
		return Tensor{}, V41DeferredStatus{}, fail(2, invalid.Error())
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var output, statusBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			if primary, ok := recovered.(error); ok {
				err = primary
			} else {
				err = ClassifyVulkanPanic(recovered, "V41BF16Publish")
			}
		}
		if err != nil {
			v.freeDeferredLocked(statusBuf, output)
			out, status = Tensor{}, V41DeferredStatus{}
		}
	}()
	if code := int(C.fvk_v41_attention_bf16_state()); code != 0 {
		return Tensor{}, V41DeferredStatus{}, fail(code, "submission state refuses publication")
	}
	if C.fvk_have_v41_routed_fused() == 0 {
		return Tensor{}, V41DeferredStatus{}, fail(3, "fused routed modules unavailable")
	}
	downPtrs, ok := v.fusedVulkanSlotPtrs(downs)
	if !ok {
		return Tensor{}, V41DeferredStatus{}, fail(2, "every down slot must be live weight storage owned by this backend")
	}
	mb, mOK := v.ownedVulkanBuf(mid)
	ib, iOK := v.ownedVulkanBuf(ids)
	if !mOK || !iOK {
		return Tensor{}, V41DeferredStatus{}, fail(2, "mid and ids must be live F32 storage owned by this backend")
	}
	if C.fvk_v41_attention_bf16_count_supported(C.int(rows)) == 0 {
		return Tensor{}, V41DeferredStatus{}, fail(2, "selected device count limit")
	}
	output = v.dallocForClass(rows*4, MemoryActivation, "V4.1 routed fused sum")
	statusBuf = v.dallocForClass(picks*v41BF16CheckedStatusBytes, MemoryActivation, "V4.1 routed fused down status")
	if code := int(C.fvk_v41_routed_down_sum_f32((*unsafe.Pointer)(unsafe.Pointer(&downPtrs[0])), C.int(len(downPtrs)),
		mb.ptr, output.ptr, ib.ptr, statusBuf.ptr,
		C.int(rows), C.int(in), C.int(picks), C.uint32_t(site))); code != 0 {
		return Tensor{}, V41DeferredStatus{}, fail(code, "initialization or checked fused dispatch failed")
	}
	output.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{rows}, nil, output), V41DeferredStatus{
		Kind: V41DeferredBF16Publish, Count: rows, Site: site, Records: picks,
		Record: makeTensor(v, F32, RowMajor, []int{picks * v41BF16CheckedStatusBytes / 4}, nil, statusBuf),
	}, nil
}
