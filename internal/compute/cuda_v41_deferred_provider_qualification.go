//go:build cuda && cgo && fak_cuda_v41_deferred_read_qualification && fak_cuda_v41_routed_mmid

package compute

// Qualification builds only: preserve c as every tensor's Backend and expose
// the separate reader to actual model row-only consumers. Ordinary deferred
// producers remain refused until model abort paths use their cancellation owner.
// Do not probe native state here: issued work must retain its cleanup reader
// after a sticky fault. Each submitted read still performs native admission.
func (c *cudaBackend) V41DeferredStatusForModel() (V41DeferredStatusBackend, bool) {
	if c == nil {
		return nil, false
	}
	return &cudaV41DeferredQualification{owner: c, readOnly: true}, true
}
