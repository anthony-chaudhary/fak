//go:build v41_bf16_boundary_witness && vulkan && (windows || linux) && cgo

package compute

/*
#cgo v41_indexer_witness CFLAGS: -DFAK_V41_QUERY_PREFIX_CONFLICT
#cgo v41_logit_scale_witness CFLAGS: -DFAK_V41_QUERY_PREFIX_CONFLICT
#cgo v41_tail_rope_exact_witness CFLAGS: -DFAK_V41_QUERY_PREFIX_CONFLICT
#ifdef FAK_V41_QUERY_PREFIX_CONFLICT
#error "query prefix uses only the isolated BF16 witness archive"
#endif
#define FAK_V41_BF16_BOUNDARY_WITNESS
#include "vulkan_backend.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

var _ V41QueryPrefixBackend = (*vulkanBackend)(nil)

func (v *vulkanBackend) V41QueryPrefixAvailable() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_query_prefix() != 0
}

func (v *vulkanBackend) V41QueryPrefix(in V41QueryPrefixInputs) (out V41QueryPrefixResult, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var owned []*vulkanBuf
	stage := "payload"
	defer func() {
		if recovered := recover(); recovered != nil {
			primary, isError := recovered.(error)
			if !isError {
				primary = ClassifyVulkanPanic(recovered, "V41QueryPrefix")
			}
			// The generic classifier can unwrap a BackendError. Keep the exact
			// recovered error chain so wrappers and primary identity survive.
			err = &V41QueryPrefixStageError{Stage: stage, Cause: primary}
		}
		for i := len(owned) - 1; i >= 0; i-- {
			b := owned[i]
			if b != nil && b.ptr != nil {
				// fvk_free parks uncertain submissions. Do not re-enter public
				// Free/Read while holding the owner mutex.
				C.fvk_free(b.ptr)
				b.ptr, b.n = nil, 0
			}
		}
		if err != nil {
			out = V41QueryPrefixResult{}
		}
	}()
	fail := func(status int, why string) (V41QueryPrefixResult, error) {
		cause := vulkanV41BF16BoundaryStatusError(status)
		cause.Site = "V41QueryPrefix"
		cause.Message += ": " + why
		return V41QueryPrefixResult{}, &V41QueryPrefixStageError{Stage: stage, Cause: cause}
	}
	if v == nil {
		return fail(3, "nil backend")
	}
	// Native state checks sticky/pending BEFORE healthy busy. No allocation,
	// host copy, descriptor or command precedes this pre-selection refusal.
	switch status := int(C.fvk_v41_query_prefix_state()); status {
	case 0:
	case int(C.FVK_V41_QUERY_PREFIX_BUSY_BATCH):
		return V41QueryPrefixResult{}, &V41QueryPrefixBusyError{State: "batch"}
	case int(C.FVK_V41_QUERY_PREFIX_BUSY_RESTORE):
		return V41QueryPrefixResult{}, &V41QueryPrefixBusyError{State: "restore"}
	default:
		return fail(status, "sticky or pending submission fault")
	}
	if C.fvk_have_v41_query_prefix() == 0 {
		return fail(3, "selected modules unavailable")
	}
	h, r, invalid := validateV41QueryPrefix(in)
	if invalid != nil {
		return V41QueryPrefixResult{}, invalid
	}
	if C.fvk_v41_query_prefix_geometry_supported(C.int(h), C.int(r)) == 0 {
		return fail(2, "selected device geometry limit")
	}
	var args C.fvk_v41_query_prefix_buffers
	for i, operand := range []struct {
		t     Tensor
		bytes int
	}{{in.AttentionGain, 4 * h}, {in.WQA, 4 * h * r}, {in.QueryGain, 4 * r}} {
		b, ok := operand.t.buf.(*vulkanBuf)
		if operand.t.be != v || !ok || b == nil || b.ptr == nil || b.n < operand.bytes || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return fail(2, "weight must be a live full-sized resident F32 tensor owned by this backend")
		}
		args.weights[i] = b.ptr
	}
	stage = "allocation"
	allocate := func(bytes int, what string) *vulkanBuf {
		b := v.dallocForClass(bytes, MemoryActivation, "V4.1 query prefix "+what)
		owned = append(owned, b)
		return b
	}
	input := allocate(4*h, "input")
	args.input = input.ptr
	var values [7]*vulkanBuf
	for i := range values {
		n := r
		if i < 3 {
			n = h
		}
		values[i] = allocate(4*n, fmt.Sprintf("value %d", i))
		args.values[i] = values[i].ptr
	}
	var statusBuffers [4]*vulkanBuf
	for i := range statusBuffers {
		statusBuffers[i] = allocate(v41BF16CheckedStatusBytes, fmt.Sprintf("status %d", i+3))
		args.status[i] = statusBuffers[i].ptr
	}
	stage = "native command"
	var failedStage C.uint32_t
	status := int(C.fvk_v41_query_prefix_f32((*C.float)(unsafe.Pointer(&in.Collapse[0])), &args,
		C.int(h), C.int(r), C.float(in.Epsilon), &failedStage))
	if status != 0 {
		names := [...]string{"payload", "input copy", "attention-norm input", "attention norm", "attention-norm output", "wq_a", "query-norm input", "query norm", "query-norm output", "completion"}
		if uint64(failedStage) < uint64(len(names)) {
			stage = names[int(failedStage)]
		}
		return fail(status, "checked native prefix failed")
	}
	stage = "status transport"
	var raw [4][v41BF16CheckedStatusBytes]byte
	var records [4][]byte
	for i := range raw {
		if status := int(C.fvk_v41_d2h(unsafe.Pointer(&raw[i][0]), statusBuffers[i].ptr, C.size_t(len(raw[i])))); status != 0 {
			return V41QueryPrefixResult{}, &V41QueryPrefixStageError{Stage: stage, Cause: vulkanReadError(status)}
		}
		records[i] = raw[i][:]
	}
	stage = "status validation"
	if statusErr := decodeV41QueryPrefixStatuses(records, h, r); statusErr != nil {
		return V41QueryPrefixResult{}, &V41QueryPrefixStageError{Stage: stage, Cause: statusErr}
	}
	stage = "output transport"
	out.Collapsed, out.QLat = make([]float32, h), make([]float32, r)
	for _, result := range []struct {
		row []float32
		buf *vulkanBuf
	}{{out.Collapsed, values[2]}, {out.QLat, values[6]}} {
		if status := int(C.fvk_v41_d2h(unsafe.Pointer(&result.row[0]), result.buf.ptr, C.size_t(4*len(result.row)))); status != 0 {
			return V41QueryPrefixResult{}, &V41QueryPrefixStageError{Stage: stage, Cause: vulkanReadError(status)}
		}
	}
	if invalid := validateV41QueryPrefixResult(out, h, r); invalid != nil {
		return V41QueryPrefixResult{}, invalid
	}
	return out, nil
}
