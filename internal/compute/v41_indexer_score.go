package compute

import (
	"fmt"
	"slices"
)

// V41IndexerScoreBackend is the optional one-position DeepSeek-V4.1 scoring
// primitive. It does not widen Backend or perform candidate/row selection.
// Inputs are immutable unquantized row-major F32 tensors on this backend:
// q[heads,headDim], keys[rows,headDim], weights[heads]. The caller validates all
// original values as finite before upload, including for an empty selection.
// Projection, normalization, RoPE and all weight scaling precede this boundary.
//
// Each row accumulates dimensions in ascending order, rectifies each head's dot
// with `if dot < 0 { dot = 0 }`, then accumulates signed weighted heads in ascending
// order. Every product and sum is separately rounded to binary32, preserving
// denormals, signed zero, infinity and NaN under round-to-nearest-even. No FMA,
// reassociation, F64, BF16, extra scaling or alternative reduction is permitted.
// Arithmetic-produced infinities and NaN classification are preserved; NaN
// payload bits are unspecified. The existing downstream selection policy,
// including its NaN rejection, remains the caller's.
//
// Output is a fresh caller-owned scores[rows] tensor, disjoint from every input.
// rows==0 returns a fresh empty logical tensor without device allocation or
// dispatch. For that case keys may be a metadata-only empty tensor. A selected
// operation, completion or checked-read error must propagate without CPU replay.
// Capability absence permits the caller to choose its unchanged host reference.
// Capability admission additionally requires trusted source/compiler provenance
// and a numerical witness bound to the exact module/archive/device identities.
// This source-only candidate keeps production capability false until that
// separately reviewed qualification admission is integrated.
type V41IndexerScoreBackend interface {
	Backend
	// V41IndexerScoreAdmission atomically distinguishes healthy capability
	// absence (false, nil) from a backend fault (false, typed error). It only
	// observes state: no upload, allocation, dispatch, flush, wait or recovery.
	// Call before binding and at the actual score seam even when initial absence
	// retained the host scorer. Every selected invocation, including rows==0,
	// also rechecks. Any error must close the request without host replay.
	// Only an initial false, nil result permits the unchanged host path;
	// a later false, nil does not authorize a selected caller to switch paths.
	// The snapshot is not a completion fence or a future-health guarantee.
	V41IndexerScoreAdmission() (supported bool, err error)
	// SupportsV41IndexerScore is a capability-only diagnostic. False cannot
	// distinguish absent qualification from a sticky backend failure; callers
	// deciding whether to use the host path must use admission instead.
	SupportsV41IndexerScore() bool
	V41IndexerScoreUnavailableReason() string
	V41IndexerScore(q, keys, weights Tensor, rows, heads, headDim int) (Tensor, error)
}

// Keep admission and operation error classification identical. This pure
// adapter is portable so fault precedence can be witnessed without a device.
func vulkanV41IndexerScoreAdmissionResult(status int, supported bool) (bool, error) {
	if status == 0 {
		return supported, nil
	}
	err := vulkanV41IndexerScoreStatusError(status, "backend admission refused")
	err.Site = "V41IndexerScoreAdmission"
	return false, err
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

// validateV41IndexerScore checks all products before narrowing to the int32
// push ABI or host byte counts. It neither reads device data nor allocates.
func validateV41IndexerScore(q, keys, weights Tensor, rows, heads, headDim int) (qBytes, keyBytes, weightBytes, outBytes int, err error) {
	const maxABI = uint64(1<<31 - 1)
	if rows < 0 || heads <= 0 || headDim <= 0 || uint64(rows) > maxABI || uint64(heads) > maxABI || uint64(headDim) > maxABI {
		return 0, 0, 0, 0, fmt.Errorf("compute: V4.1 indexer score invalid geometry rows=%d heads=%d headDim=%d", rows, heads, headDim)
	}
	qCount, keyCount := uint64(heads)*uint64(headDim), uint64(rows)*uint64(headDim)
	maxElements := uint64(int(^uint(0)>>1)) / 4
	if qCount > maxABI || keyCount > maxABI || qCount > maxElements || keyCount > maxElements || uint64(heads) > maxElements || uint64(rows) > maxElements {
		return 0, 0, 0, 0, fmt.Errorf("compute: V4.1 indexer score geometry exceeds shader index or host byte range")
	}
	for _, operand := range []struct {
		name  string
		t     Tensor
		shape []int
	}{
		{"q", q, []int{heads, headDim}},
		{"keys", keys, []int{rows, headDim}},
		{"weights", weights, []int{heads}},
	} {
		if operand.t.Dtype != F32 || operand.t.Layout != RowMajor || operand.t.Quant != nil || !slices.Equal(operand.t.Shape, operand.shape) {
			return 0, 0, 0, 0, fmt.Errorf("compute: V4.1 indexer score %s must be unquantized row-major F32 with shape %v", operand.name, operand.shape)
		}
	}
	return int(qCount) * 4, int(keyCount) * 4, heads * 4, rows * 4, nil
}
