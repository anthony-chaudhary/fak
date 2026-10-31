//go:build cuda && cgo && fak_cuda_v41_routed_mmid

package compute

/*
int fak_v41_mmid_ordinary_compiled(int);
*/
import "C"

import "math"

// cudaV41DeferredQualification is a private hardware-witness adapter, not a
// Backend. Keep owner named: embedding *cudaBackend would accidentally expose
// this interface to model consumers which select it by method presence alone.
// The candidate reader admits bounded row-major vectors and panels, not the
// full Vulkan reader domain. This private adapter is not a production backend
// capability, even after individual consumer shapes become executable.
// Callers retain output ownership and must cancel issued statuses on early abort.
type cudaV41DeferredQualification struct {
	owner    *cudaBackend
	readOnly bool
}

var _ V41DeferredStatusBackend = (*cudaV41DeferredQualification)(nil)

func (q *cudaV41DeferredQualification) valid() bool { return q != nil && q.owner != nil }

func (q *cudaV41DeferredQualification) SupportsV41DeferredStatus(count int) bool {
	if !q.valid() || q.readOnly || count <= 0 || count > math.MaxInt32 || count > math.MaxInt/4 {
		return false
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return q.owner.cudaV41MMIDState("qualification-deferred-status") == nil &&
		C.fak_v41_mmid_ordinary_compiled(1) == 0 && C.fak_v41_mmid_ordinary_compiled(2) == 0
}

func (q *cudaV41DeferredQualification) V41RoutedSwiGLUDeferred(gate, up Tensor, limit, weight float32) (Tensor, V41DeferredStatus, error) {
	if !q.valid() || q.readOnly {
		return Tensor{}, V41DeferredStatus{}, cudaV41MMIDError("V41RoutedSwiGLUDeferred", "qualification producer unavailable", -3, nil)
	}
	return q.owner.cudaV41MMIDRoutedSwiGLUDeferred(gate, up, limit, weight, true)
}

func (q *cudaV41DeferredQualification) V41BF16PublishDeferred(input Tensor, site uint32) (Tensor, V41DeferredStatus, error) {
	if !q.valid() || q.readOnly {
		return Tensor{}, V41DeferredStatus{}, cudaV41MMIDError("V41BF16PublishDeferred", "qualification producer unavailable", -3, nil)
	}
	return q.owner.cudaV41MMIDBF16PublishDeferred(input, site, true)
}

func (q *cudaV41DeferredQualification) V41ReadDeferred(statuses []V41DeferredStatus, rows []Tensor) ([]error, [][]float32, error) {
	if !q.valid() {
		return nil, nil, cudaV41MMIDError("V41ReadDeferred", "nil qualification owner", -3, nil)
	}
	// Do not preflight here: the ownership core must collect and settle issued
	// records even when later metadata, admission or readback fails.
	return q.owner.cudaV41MMIDReadBatch(statuses, rows)
}

func (q *cudaV41DeferredQualification) cancel(statuses []V41DeferredStatus, primary error) error {
	if !q.valid() {
		if primary != nil {
			return primary
		}
		return cudaV41MMIDError("V41ReadDeferred", "nil qualification owner", -3, nil)
	}
	// Cleanup must not consult Supports or retry sticky producer admission.
	return q.owner.cudaV41MMIDCancel(statuses, primary)
}

// V41CancelDeferred delegates the optional model abort seam to the same owner
// as the qualification reader. It never changes production provider admission.
func (q *cudaV41DeferredQualification) V41CancelDeferred(statuses []V41DeferredStatus, primary error) (bool, error) {
	if !q.valid() {
		return false, q.cancel(statuses, primary)
	}
	return q.owner.cudaV41MMIDCancelCompletion(statuses, primary)
}
