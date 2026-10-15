package compute

import (
	"fmt"
	"os"
)

// V41Q3KTokenPanelBackend contracts several rows through ONE expert's packed
// Q3_K matrix. It is separate from the routed-ID ABI: tokens share a weight,
// rather than selecting independent weights from an eight-slot table.
// Availability is experimental and default-off. FAK_V41_Q3K_TOKEN_PANEL=1
// opts into Q3 panels independently; the legacy Q2 flag still enables both.
// This applies to every eligible Q3 panel caller, not only expert down.
// A selected error is fatal;
// callers may choose their scalar arm only before calling the contraction.
type V41Q3KTokenPanelBackend interface {
	SupportsV41Q3KTokenPanel(weight Tensor, rows int) bool
	// V41Q3KTokenPanel consumes Q3_K [out,in] and contiguous F32 [rows*in],
	// producing caller-owned F32 [rows*out]. It performs neither BF16
	// publication nor host readback; the caller retains its ordered deferred
	// activation/publication checks before publishing any result or state.
	V41Q3KTokenPanel(weight, input Tensor, rows int) (Tensor, error)
}

func v41Q3KTokenPanelOptedIn() bool {
	// Keep exact legacy "1" semantics. Q3=0 must not veto Q2=1, and this
	// separate Q3 request never enables the Q2 panel family.
	return os.Getenv("FAK_V41_Q3K_TOKEN_PANEL") == "1" || os.Getenv("FAK_V41_Q2K_TOKEN_PANEL") == "1"
}

// V41Q3KTokenPanelPlan is the exact three-int32 push ABI and bounded storage
// geometry of v41_q3k_token_panel.comp. Each of 64 lanes owns one output
// column for up to eight tokens; workgroup Y tiles the full panel while
// retaining the scalar Q3_K accumulation order.
type V41Q3KTokenPanelPlan struct {
	Out, In, Rows                        int
	PayloadBytes                         int
	WeightBytes, InputBytes, OutputBytes int
	WorkgroupsX, WorkgroupsY             uint32
}

// PlanV41Q3KTokenPanel admits 2..64 rows in at-most-eight-token workgroups.
// A single-token decode cannot
// accidentally select this candidate. Signed 32-bit byte extents also keep
// host int arithmetic safe on supported 32-bit software-test hosts.
func PlanV41Q3KTokenPanel(out, in, rows int) (V41Q3KTokenPanelPlan, error) {
	bad := func() (V41Q3KTokenPanelPlan, error) {
		return V41Q3KTokenPanelPlan{}, fmt.Errorf("compute: Q3_K token panel requires bounded [out,in] with in divisible by 256 and 2 through 64 rows")
	}
	if (rows < 2 || rows > 64) || out <= 0 || in <= 0 || in%256 != 0 || uint64(out) > 0x7fffffff || uint64(in) > 0x7fffffff {
		return bad()
	}
	payload := uint64(out) * uint64(in/256) * 110
	// byteAt reads uint words, including the last two bytes of an odd number
	// of 110-byte blocks. Match Vulkan's existing Q3_K upload tail padding.
	wb := (payload + 3) &^ uint64(3)
	xb, yb := uint64(rows)*uint64(in)*4, uint64(rows)*uint64(out)*4
	if wb > 0x7fffffff || xb > 0x7fffffff || yb > 0x7fffffff {
		return bad()
	}
	return V41Q3KTokenPanelPlan{Out: out, In: in, Rows: rows,
		PayloadBytes: int(payload), WeightBytes: int(wb), InputBytes: int(xb), OutputBytes: int(yb),
		WorkgroupsX: uint32((uint64(out) + 63) / 64), WorkgroupsY: uint32((rows + 7) / 8)}, nil
}

func validateV41Q3KTokenPanel(weight, input Tensor, rows int) (V41Q3KTokenPanelPlan, error) {
	if weight.Dtype != Q3_K || weight.Layout != RowMajor || len(weight.Shape) != 2 ||
		input.Dtype != F32 || input.Layout != RowMajor || input.Quant != nil || len(input.Shape) != 1 {
		return V41Q3KTokenPanelPlan{}, fmt.Errorf("compute: Q3_K token panel needs a row-major Q3_K matrix and flat unquantized F32 panel")
	}
	plan, err := PlanV41Q3KTokenPanel(weight.Shape[0], weight.Shape[1], rows)
	if err != nil {
		return plan, err
	}
	if input.Shape[0] != plan.InputBytes/4 {
		return V41Q3KTokenPanelPlan{}, fmt.Errorf("compute: Q3_K token panel activation length does not match rows*in")
	}
	return plan, nil
}
