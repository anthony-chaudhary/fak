//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "fmt"

var _ V41IndexerScoreBackend = (*vulkanBackend)(nil)

func (v *vulkanBackend) SupportsV41IndexerScore() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_indexer_score() != 0
}

func (v *vulkanBackend) V41IndexerScoreUnavailableReason() string {
	if v == nil {
		return "Vulkan backend is nil"
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.GoString(C.fvk_v41_indexer_score_unavailable_reason())
}

// V41IndexerScore uses the existing checked V4.1 submission and readback path.
// Native uncertain completion quarantines memory until process exit. The output
// never enters the Go transient pool, and no public Read/Free/Upload is called
// while vulkanMu is held.
func (v *vulkanBackend) V41IndexerScore(q, keys, weights Tensor, rows, heads, headDim int) (out Tensor, err error) {
	return v.v41IndexerScoreWithNative(q, keys, weights, rows, heads, headDim,
		func() bool { return C.fvk_have_v41_indexer_score() != 0 },
		func() string { return C.GoString(C.fvk_v41_indexer_score_unavailable_reason()) },
		v41IndexerScoreStatusLocked)
}

// Internal plumbing shared only with the separately build-tagged physical
// witness. No callback or qualification override is exposed on Backend.
func (v *vulkanBackend) v41IndexerScoreWithNative(q, keys, weights Tensor, rows, heads, headDim int,
	ready func() bool, unavailable func() string,
	call func(q, keys, weights, out *vulkanBuf, rows, heads, headDim int) int) (out Tensor, err error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	var outBuf *vulkanBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err = ClassifyVulkanPanic(recovered, "V41IndexerScore")
		}
		if err != nil {
			if outBuf != nil && outBuf.ptr != nil {
				C.fvk_free(outBuf.ptr)
				outBuf.ptr = nil
				outBuf.n = 0
			}
			out = Tensor{}
		}
	}()
	if v == nil {
		return Tensor{}, vulkanV41IndexerScoreStatusError(3, "Vulkan backend is nil")
	}
	qb, kb, wb, ob, validationErr := validateV41IndexerScore(q, keys, weights, rows, heads, headDim)
	if validationErr != nil {
		return Tensor{}, vulkanV41IndexerScoreStatusError(2, validationErr.Error())
	}
	for _, operand := range []struct {
		name  string
		t     Tensor
		bytes int
	}{
		{"q", q, qb}, {"keys", keys, kb}, {"weights", weights, wb},
	} {
		// Empty key storage is never bound or read. All other operands must be
		// exact-sized live device allocations owned by this backend.
		if operand.bytes == 0 && operand.t.buf == nil && operand.t.be == nil {
			continue
		}
		b, ok := operand.t.buf.(*vulkanBuf)
		if operand.t.be != v || !ok || b == nil || (operand.bytes != 0 && b.ptr == nil) || b.n != operand.bytes || b.scalePtr != nil || len(b.q8Chunks) != 0 {
			return Tensor{}, vulkanV41IndexerScoreStatusError(2, operand.name+" must be an exact-sized F32 device buffer owned by this backend")
		}
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return Tensor{}, vulkanV41IndexerScoreStatusError(status, "sticky submission fault")
	}
	if rows == 0 {
		return makeTensor(v, F32, RowMajor, []int{0}, nil, &vulkanBuf{v41CheckedRead: true}), nil
	}
	if !ready() {
		return Tensor{}, vulkanV41IndexerScoreStatusError(3, unavailable())
	}
	outBuf = v.dallocForClass(ob, MemoryActivation, "V4.1 indexer score output")
	status := call(q.buf.(*vulkanBuf), keys.buf.(*vulkanBuf), weights.buf.(*vulkanBuf), outBuf, rows, heads, headDim)
	if status != 0 {
		return Tensor{}, vulkanV41IndexerScoreStatusError(status, "native dispatch refused")
	}
	// A recorded batch has not completed yet. This flush uses g_batchHasV41's
	// checked path and cannot silently turn an uncertain submission into output.
	if status = int(C.fvk_batch_flush_status()); status != 0 {
		return Tensor{}, vulkanV41IndexerScoreStatusError(status, "checked completion failed")
	}
	outBuf.v41CheckedRead = true
	return makeTensor(v, F32, RowMajor, []int{rows}, nil, outBuf), nil
}

// Caller holds vulkanMu and passes live handles; native code repeats extent and
// output-alias checks before recording commands.
func v41IndexerScoreStatusLocked(q, keys, weights, out *vulkanBuf, rows, heads, headDim int) int {
	return int(C.fvk_v41_indexer_score_f32(q.ptr, keys.ptr, weights.ptr, out.ptr, C.int(rows), C.int(heads), C.int(headDim)))
}

func vulkanV41IndexerScoreStatusError(status int, reason string) *BackendError {
	class, cause := VulkanClassExecutionFailed, ErrVulkanExecutionFailed
	switch {
	case status == -4:
		class, cause = VulkanClassDeviceLost, ErrVulkanDeviceLost
	case status == -1 || status == -2:
		class, cause = VulkanClassAllocationFailed, ErrVulkanAllocationFailed
	case status < 0:
		class, cause = VulkanClassSubmissionFailed, ErrVulkanSubmissionFailed
	case status == 2:
		class, cause = VulkanClassInvalidGeometry, ErrVulkanInvalidGeometry
	}
	return &BackendError{Backend: "vulkan", Site: "V41IndexerScore", Class: class, Err: cause, Message: fmt.Sprintf("dedicated indexer score failed closed (code %d): %s", status, reason)}
}
