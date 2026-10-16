package compute

import "fmt"

// V41Q2KTokenPairBackend optionally records two same-shape Q2_K contractions
// sharing one input panel in one dispatch. It does not normalize or publish
// BF16 values. Selected failures are fatal and must not replay scalar calls.
type V41Q2KTokenPairBackend interface {
	SupportsV41Q2KTokenPair(gate, up Tensor, rows int) bool
	// Outputs are independent fresh flat F32 panels, each rows*out elements.
	// Caller owns every nonempty partial result even when err is nonnil.
	V41Q2KTokenPair(gate, up, input Tensor, rows int) (Tensor, Tensor, error)
}

// V41Q2KTokenPairPlan reuses the existing three-int32 panel geometry and
// concatenates the two output-bank workgroup ranges in X. No payload grows.
type V41Q2KTokenPairPlan struct {
	V41Q2KTokenPanelPlan
	WorkgroupsPairX uint32
}

func PlanV41Q2KTokenPair(out, in, rows int) (V41Q2KTokenPairPlan, error) {
	p, err := PlanV41Q2KTokenPanel(out, in, rows)
	if err != nil {
		return V41Q2KTokenPairPlan{}, err
	}
	return V41Q2KTokenPairPlan{V41Q2KTokenPanelPlan: p, WorkgroupsPairX: 2 * p.WorkgroupsX}, nil
}

func validateV41Q2KTokenPair(gate, up, input Tensor, rows int) (V41Q2KTokenPairPlan, error) {
	p, err := validateV41Q2KTokenPanel(gate, input, rows)
	if err != nil {
		return V41Q2KTokenPairPlan{}, err
	}
	q, err := validateV41Q2KTokenPanel(up, input, rows)
	if err != nil {
		return V41Q2KTokenPairPlan{}, err
	}
	if p.Out != q.Out || p.In != q.In {
		return V41Q2KTokenPairPlan{}, fmt.Errorf("compute: Q2_K token pair requires identical gate/up matrix shapes")
	}
	return PlanV41Q2KTokenPair(p.Out, p.In, rows)
}
