package compute

import "fmt"

// V41Q2KTokenPanelBackend contracts several rows through ONE expert's packed
// Q2_K matrix. It is separate from the routed-ID ABI: tokens share a weight,
// rather than selecting independent weights from an eight-slot table.
// Availability is experimental and default-off. A selected error is fatal;
// callers may choose their scalar arm only before calling the contraction.
type V41Q2KTokenPanelBackend interface {
	SupportsV41Q2KTokenPanel(weight Tensor, rows int) bool
	// V41Q2KTokenPanel consumes Q2_K [out,in] and contiguous F32 [rows*in],
	// producing caller-owned F32 [rows*out]. It performs neither BF16
	// publication nor host readback; the caller retains its ordered deferred
	// activation/publication checks before publishing any result or state.
	V41Q2KTokenPanel(weight, input Tensor, rows int) (Tensor, error)
}

// V41Q2KTokenPanelPlan is the exact three-int32 push ABI and bounded storage
// geometry of v41_q2k_token_panel.comp. One 64-lane workgroup covers two output
// columns for every token; its quantized loads do not have a token dimension.
type V41Q2KTokenPanelPlan struct {
	Out, In, Rows                        int
	WeightBytes, InputBytes, OutputBytes int
	WorkgroupsX                          uint32
}

// PlanV41Q2KTokenPanel admits only 2/4/8 rows. A single-token decode cannot
// accidentally select this candidate. Signed 32-bit byte extents also keep
// host int arithmetic safe on supported 32-bit software-test hosts.
func PlanV41Q2KTokenPanel(out, in, rows int) (V41Q2KTokenPanelPlan, error) {
	bad := func() (V41Q2KTokenPanelPlan, error) {
		return V41Q2KTokenPanelPlan{}, fmt.Errorf("compute: Q2_K token panel requires bounded [out,in] with in divisible by 256 and exactly 2, 4 or 8 rows")
	}
	if (rows != 2 && rows != 4 && rows != 8) || out <= 0 || in <= 0 || in%256 != 0 || uint64(out) > 0x7fffffff || uint64(in) > 0x7fffffff {
		return bad()
	}
	wb := uint64(out) * uint64(in/256) * 84
	xb, yb := uint64(rows)*uint64(in)*4, uint64(rows)*uint64(out)*4
	if wb > 0x7fffffff || xb > 0x7fffffff || yb > 0x7fffffff {
		return bad()
	}
	return V41Q2KTokenPanelPlan{Out: out, In: in, Rows: rows,
		WeightBytes: int(wb), InputBytes: int(xb), OutputBytes: int(yb),
		WorkgroupsX: uint32((uint64(out) + 1) / 2)}, nil
}

func validateV41Q2KTokenPanel(weight, input Tensor, rows int) (V41Q2KTokenPanelPlan, error) {
	if weight.Dtype != Q2_K || weight.Layout != RowMajor || len(weight.Shape) != 2 ||
		input.Dtype != F32 || input.Layout != RowMajor || input.Quant != nil || len(input.Shape) != 1 {
		return V41Q2KTokenPanelPlan{}, fmt.Errorf("compute: Q2_K token panel needs a row-major Q2_K matrix and flat unquantized F32 panel")
	}
	plan, err := PlanV41Q2KTokenPanel(weight.Shape[0], weight.Shape[1], rows)
	if err != nil {
		return plan, err
	}
	if input.Shape[0] != plan.InputBytes/4 {
		return V41Q2KTokenPanelPlan{}, fmt.Errorf("compute: Q2_K token panel activation length does not match rows*in")
	}
	return plan, nil
}
