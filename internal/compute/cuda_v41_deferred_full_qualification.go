//go:build cuda && cgo && fak_cuda_v41_deferred_full_qualification && fak_cuda_v41_routed_mmid

package compute

/*
int fak_v41_mmid_ordinary_compiled(int);
*/
import "C"

// This build is a qualification executable, not a production default. The
// routed ABI is linked by fak_cuda_v41_routed_mmid; both ordinary kernels must
// also be present in the recovered native archive. These probes check compiled
// features only: runtime state/device admission remains in Supports and each op.
func cudaV41FullDeferredArchivePresent() bool {
	return C.fak_v41_mmid_ordinary_compiled(1) == 0 && C.fak_v41_mmid_ordinary_compiled(2) == 0
}

func (c *cudaBackend) V41RoutedLayerQualified() bool {
	return c != nil && cudaV41FullDeferredArchivePresent()
}

// Supply full producers only to the two cancellation-owned routed seams.
// Generic consumers retain the ordinary refusing or separately selected
// read-only provider. No Backend embedding or tensor owner rewriting occurs.
// Compiled-feature probes are stable; a later sticky fault cannot revoke an
// already selected cancellation/readback owner.
func (c *cudaBackend) V41RoutedDeferredStatusForModel() (V41DeferredStatusBackend, bool) {
	if !c.V41RoutedLayerQualified() {
		return nil, false
	}
	return &cudaV41DeferredQualification{owner: c}, true
}
