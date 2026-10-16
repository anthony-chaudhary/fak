package compute

// The original backend prevents promoted methods on wrappers from granting a loan.
type v41PoolNormPanelBackend interface {
	v41TryPoolNormPanel(Backend, Tensor, []float32, int, int, float32) (bool, error)
}

// TryV41PoolNormPanel borrows exclusive host storage synchronously. On selection
// the input is consumed, including on error; success replaces it with raw norms.
// A miss preserves it and does no transport. Ordinary output publication is separate.
func TryV41PoolNormPanel(be Backend, gain Tensor, owned []float32, rows, width int, eps float32) (bool, error) {
	if b, ok := be.(v41PoolNormPanelBackend); ok {
		return b.v41TryPoolNormPanel(be, gain, owned, rows, width, eps)
	}
	return false, nil
}
