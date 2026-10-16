package compute

// A sealed backend capability avoids granting authority through promoted methods.
type v41FFNNormPublicationBackend interface {
	v41TryFFNNormPublication(Backend, Tensor, []float32, int, int, float32) (*V41FFNNormPublication, bool, error)
}

// V41FFNNormPublication holds one independently allocated published slab. The
// caller must inspect the corresponding raw norm row for finiteness before Row:
// a later nonfinite element in that row precedes its checked rounding error.
// Rows preceding an attributed fault are available; the fault and later rows
// are withheld. This is a synchronous result, not a reusable ownership grant.
type V41FFNNormPublication struct {
	values      []float32
	rows, width int
	firstBad    int
}

func (p *V41FFNNormPublication) Row(row int) ([]float32, error) {
	if p == nil || row < 0 || row >= p.rows || p.width <= 0 || len(p.values) != p.rows*p.width {
		return nil, &V41BF16CheckedProtocolError{Message: "invalid FFN publication row"}
	}
	if p.firstBad >= 0 && row >= p.firstBad/p.width {
		if row != p.firstBad/p.width {
			return nil, &V41BF16CheckedProtocolError{Message: "FFN publication requested after failed row"}
		}
		fault := &V41BF16CheckedArithmeticError{Site: V41BF16PublishFFNNormOutput, Index: p.firstBad % p.width}
		return nil, &BackendError{Backend: "vulkan", Site: "V41BF16Publish", Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: fault.Error(), Recovered: fault}
	}
	lo, hi := row*p.width, (row+1)*p.width
	return p.values[lo:hi:hi], nil
}

// TryV41FFNNormPublication consumes owned raw input only when selected. A
// capacity miss has no device effects and allocates no published host slab.
// Successful transport replaces owned with raw norms; Row exposes independent
// checked BF16 output. Numeric verdicts must be consumed in logical row order.
func TryV41FFNNormPublication(be Backend, gain Tensor, owned []float32, rows, width int, eps float32) (*V41FFNNormPublication, bool, error) {
	if b, ok := be.(v41FFNNormPublicationBackend); ok {
		return b.v41TryFFNNormPublication(be, gain, owned, rows, width, eps)
	}
	return nil, false, nil
}
