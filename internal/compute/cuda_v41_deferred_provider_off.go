//go:build cuda && cgo && (!fak_cuda_v41_deferred_read_qualification || !fak_cuda_v41_routed_mmid)

package compute

// Candidate kernels alone never expose a model deferred operator.
func (*cudaBackend) V41DeferredStatusForModel() (V41DeferredStatusBackend, bool) {
	return nil, false
}
