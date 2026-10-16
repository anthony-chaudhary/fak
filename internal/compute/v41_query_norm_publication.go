package compute

// A sealed backend capability avoids granting authority through promoted methods.
type v41QueryNormPublicationBackend interface {
	v41TryQueryNormPublication(Backend, Tensor, []float32, int, int, int, float32) (*V41QueryNormPublication, bool, error)
}

// V41QueryNormPublication holds one independently allocated published slab. The
// caller must inspect the corresponding raw norm row for finiteness before Row:
// a later nonfinite element in that row precedes its checked rounding error.
// Rows preceding an attributed fault are available; the fault and later rows
// are withheld. This is a synchronous result, not a reusable ownership grant.
type V41QueryNormPublication struct {
	values      []float32
	rows, width int
	firstBad    int
	inputBad    int
}

func (p *V41QueryNormPublication) Row(row int) ([]float32, error) {
	if p == nil || row < 0 || row >= p.rows || p.width <= 0 || len(p.values) != p.rows*p.width {
		return nil, &V41BF16CheckedProtocolError{Message: "invalid Query publication row"}
	}
	if p.firstBad >= 0 && row >= p.firstBad/p.width {
		if row != p.firstBad/p.width {
			return nil, &V41BF16CheckedProtocolError{Message: "Query publication requested after failed row"}
		}
		fault := &V41BF16CheckedArithmeticError{Site: V41BF16PublishQueryNormOutput, Index: p.firstBad % p.width}
		return nil, &BackendError{Backend: "vulkan", Site: "V41BF16Publish", Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: fault.Error(), Recovered: fault}
	}
	lo, hi := row*p.width, (row+1)*p.width
	return p.values[lo:hi:hi], nil
}

// TryV41QueryNormPublication consumes owned raw input only when selected. A
// capacity miss has no device effects and allocates no published host slab.
// Successful transport replaces owned with raw norms; Row exposes independent
// checked BF16 output. Numeric verdicts must be consumed in logical row order.
func TryV41QueryNormPublication(be Backend, gain Tensor, owned []float32, rows, width, capacityRows int, eps float32) (*V41QueryNormPublication, bool, error) {
	if b, ok := be.(v41QueryNormPublicationBackend); ok {
		return b.v41TryQueryNormPublication(be, gain, owned, rows, width, capacityRows, eps)
	}
	return nil, false, nil
}

// InputRowError precedes raw norm and output verdicts for this logical row.
func (p *V41QueryNormPublication) InputRowError(row int) error {
	if p == nil || row < 0 || row >= p.rows || p.width <= 0 {
		return &V41BF16CheckedProtocolError{Message: "invalid query publication input row"}
	}
	if p.inputBad < 0 || row < p.inputBad/p.width {
		return nil
	}
	if row != p.inputBad/p.width {
		return &V41BF16CheckedProtocolError{Message: "query input requested after failed row"}
	}
	fault := &V41BF16CheckedArithmeticError{Site: V41BF16PublishQueryNormInput, Index: p.inputBad % p.width}
	return &BackendError{Backend: "vulkan", Site: "V41BF16Publish", Class: VulkanClassExecutionFailed, Err: ErrVulkanExecutionFailed, Message: fault.Error(), Recovered: fault}
}
