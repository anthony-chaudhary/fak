package compute

// ReadV41RMSNormOutput reads once and returns independent storage only after the
// expected element count matches. It performs no finite validation: scalar and
// panel callers preserve their different validation/publication ordering.
// Host and arbitrary backend reads remain defensively copied; only a sealed
// canonical device read can transfer its independently allocated host slice.
func ReadV41RMSNormOutput(backend Backend, tensor Tensor, expectedCount int) ([]float32, bool) {
	return readV41OwnedF32Row(backend, tensor, expectedCount)
}
