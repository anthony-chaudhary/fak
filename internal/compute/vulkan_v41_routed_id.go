//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import (
	"unsafe"
)

var _ V41RoutedIDBackend = (*vulkanBackend)(nil)

// V41RoutedIDAvailable reports the registry V15 id modules on this device.
func (v *vulkanBackend) V41RoutedIDAvailable() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_routed_id() != 0 && int(C.fvk_v41_routed_id_max_picks()) == V41RoutedIDMaxPicks
}

func (v *vulkanBackend) SupportsV41RoutedIDWeight(dt Dtype) bool { return dt == Q2_K || dt == Q3_K }

func vulkanV41RoutedIDError(site string, status int, reason string) *BackendError {
	if status == 2 {
		return &BackendError{Backend: "vulkan", Site: site, Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry, Message: reason}
	}
	return &BackendError{Backend: "vulkan", Site: site, Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: reason}
}

// ownedVulkanBuf returns t's live storage when this backend owns it.
func (v *vulkanBackend) ownedVulkanBuf(t Tensor) (*vulkanBuf, bool) {
	b, ok := t.buf.(*vulkanBuf)
	return b, t.be == v && ok && b != nil && b.ptr != nil && b.scalePtr == nil && len(b.q8Chunks) == 0
}

func (v *vulkanBackend) V41RoutedIDs(slots []int, weights []float32, order []int) (out Tensor, err error) {
	words, err := V41RoutedIDWords(slots, weights, order)
	if err != nil {
		return Tensor{}, vulkanV41RoutedIDError("V41RoutedIDs", 2, err.Error())
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var buf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41RoutedIDs")
		}
		if err != nil {
			v.freeDeferredLocked(buf)
			out = Tensor{}
		}
	}()
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, vulkanV41RoutedIDError("V41RoutedIDs", status, "sticky submission fault")
	}
	buf = v.dallocForClass(len(words)*4, MemoryActivation, "V4.1 routed ids")
	C.fvk_h2d(buf.ptr, unsafe.Pointer(&words[0]), C.size_t(len(words)*4))
	return makeTensor(v, F32, RowMajor, []int{len(words)}, nil, buf), nil
}

func (v *vulkanBackend) V41MatVecID(weights []Tensor, ids Tensor, picks int, x Tensor) (out Tensor, err error) {
	const site = "V41MatVecID"
	rows, in, xStride, invalid := validateV41MatVecID(weights, ids, picks, x)
	if invalid != nil {
		return Tensor{}, vulkanV41RoutedIDError(site, 2, invalid.Error())
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
	ptrs := make([]unsafe.Pointer, len(weights))
	for i, w := range weights {
		b, ok := w.buf.(*vulkanBuf)
		if w.be != v || !ok || b == nil || b.ptr == nil {
			return Tensor{}, vulkanV41RoutedIDError(site, 2, "every slot weight must be live storage owned by this backend")
		}
		ptrs[i] = b.ptr
	}
	xb, ok := v.ownedVulkanBuf(x)
	ib, idsOK := v.ownedVulkanBuf(ids)
	if !ok || !idsOK {
		return Tensor{}, vulkanV41RoutedIDError(site, 2, "activation and ids must be live F32 storage owned by this backend")
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, vulkanV41RoutedIDError(site, status, "sticky submission fault")
	}
	dtype := 0
	if weights[0].Dtype == Q3_K {
		dtype = 1
	}
	output = v.dallocForClass(picks*rows*4, MemoryActivation, "V4.1 routed id GEMV rows")
	if status := int(C.fvk_v41_matvec_id_f32(C.int(dtype), (*unsafe.Pointer)(unsafe.Pointer(&ptrs[0])), C.int(len(ptrs)),
		xb.ptr, output.ptr, ib.ptr, C.int(rows), C.int(in), C.int(picks), C.int(xStride))); status != 0 {
		return Tensor{}, vulkanV41RoutedIDError(site, status, "id GEMV dispatch refused or failed")
	}
	return makeTensor(v, F32, RowMajor, []int{picks * rows}, nil, output), nil
}

func (v *vulkanBackend) V41RoutedSwiGLUIDDeferred(gate, up, ids Tensor, picks int, limit float32) (out Tensor, status V41DeferredStatus, err error) {
	const site = "V41RoutedSwiGLU"
	count, invalid := validateV41RoutedRows("gate", gate, picks)
	if invalid == nil {
		if upCount, upInvalid := validateV41RoutedRows("up", up, picks); upInvalid != nil {
			invalid = upInvalid
		} else if upCount != count {
			invalid = vulkanV41RoutedIDError(site, 2, "gate and up widths differ")
		}
	}
	if invalid == nil {
		invalid = validateV41RoutedIDsTensor(ids, picks)
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
	if C.fvk_have_v41_routed_id() == 0 {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(3, "routed id modules or binary32 float controls unavailable")
	}
	gb, gOK := v.ownedVulkanBuf(gate)
	ub, uOK := v.ownedVulkanBuf(up)
	ib, iOK := v.ownedVulkanBuf(ids)
	if !gOK || !uOK || !iOK {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, "gate, up and ids must be live F32 storage owned by this backend")
	}
	if code := int(C.fvk_submission_status()); code != 0 {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(code, "sticky submission fault")
	}
	if C.fvk_batch_foreign_active() {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(2, "the routed activation does not join a foreign batch")
	}
	output = v.dallocForClass(picks*count*4, MemoryActivation, "V4.1 routed SwiGLU id output")
	statusBuf = v.dallocForClass(picks*v41RoutedSwiGLUStatusBytes, MemoryActivation, "V4.1 routed SwiGLU id status")
	if code := int(C.fvk_v41_routed_swiglu_id_f32(gb.ptr, ub.ptr, output.ptr, ib.ptr, statusBuf.ptr,
		C.int(count), C.int(picks), C.float(limit))); code != 0 {
		return Tensor{}, V41DeferredStatus{}, vulkanV41RoutedSwiGLUStatusError(code, "initialization or checked id dispatch failed")
	}
	output.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{picks * count}, nil, output), V41DeferredStatus{
		Kind: V41DeferredRoutedSwiGLU, Count: count, Records: picks,
		Record: makeTensor(v, F32, RowMajor, []int{picks * v41RoutedSwiGLUStatusBytes / 4}, nil, statusBuf),
	}, nil
}

func (v *vulkanBackend) V41RoutedPublishSumDeferred(down, ids Tensor, picks int, site uint32) (out Tensor, status V41DeferredStatus, err error) {
	fail := func(code int, reason string) error {
		e := vulkanV41AttentionBF16StatusError(code, reason)
		e.Site = "V41BF16Publish"
		return e
	}
	count, invalid := validateV41RoutedRows("down", down, picks)
	if invalid == nil {
		invalid = validateV41RoutedIDsTensor(ids, picks)
	}
	if invalid == nil && !V41BF16PublishSite(site) {
		invalid = vulkanV41RoutedIDError("V41BF16Publish", 2, "invalid ordinary publication site")
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
	if C.fvk_have_v41_routed_id() == 0 || C.fvk_v41_attention_bf16_count_supported(C.int(count)) == 0 {
		return Tensor{}, V41DeferredStatus{}, fail(3, "routed id modules unavailable or count unsupported")
	}
	db, dOK := v.ownedVulkanBuf(down)
	ib, iOK := v.ownedVulkanBuf(ids)
	if !dOK || !iOK {
		return Tensor{}, V41DeferredStatus{}, fail(2, "down and ids must be live F32 storage owned by this backend")
	}
	output = v.dallocForClass(count*4, MemoryActivation, "V4.1 routed sum")
	statusBuf = v.dallocForClass(picks*v41BF16CheckedStatusBytes, MemoryActivation, "V4.1 routed publication id status")
	if code := int(C.fvk_v41_routed_publish_sum_f32(db.ptr, output.ptr, ib.ptr, statusBuf.ptr, C.int(count), C.int(picks), C.uint32_t(site))); code != 0 {
		return Tensor{}, V41DeferredStatus{}, fail(code, "initialization or checked id dispatch failed")
	}
	output.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{count}, nil, output), V41DeferredStatus{
		Kind: V41DeferredBF16Publish, Count: count, Site: site, Records: picks,
		Record: makeTensor(v, F32, RowMajor, []int{picks * v41BF16CheckedStatusBytes / 4}, nil, statusBuf),
	}, nil
}
