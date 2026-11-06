package compute

import "fmt"

// V41PrefillExpertPanelMaxRows bounds the activation rows one batched
// routed-expert prefill panel dispatch covers (DRC-K17). The bound matches
// PlanV41Q2KTokenPanel's row ceiling so a panel extent the Q2_K kernel admits
// is admissible here too; a single-row decode cannot accidentally select the
// batched dispatch.
const V41PrefillExpertPanelMaxRows = 64

// V41PrefillExpertPanelPlan is the bounded storage geometry of one batched
// routed-expert panel GEMM: [out,in] expert matrix times a contiguous F32
// [rows*in] activation panel, producing F32 [rows*out]. Signed 32-bit byte
// extents keep host int arithmetic safe on supported 32-bit software-test
// hosts, as in PlanV41Q2KTokenPanel.
type V41PrefillExpertPanelPlan struct {
	Out, In, Rows           int
	InputBytes, OutputBytes int
}

// PlanV41PrefillExpertPanel admits 2 through V41PrefillExpertPanelMaxRows rows
// against bounded [out,in] geometry. The weight dtype is deliberately NOT part
// of the plan: the pinned V4.1 slate serves gate/up as ring-staged Q2_K and
// down as checkpoint-backed Q3_K, while an F32-resident expert keeps its
// ordinary encoding, so dtype admission belongs to the backend's Supports
// probe, not the geometry planner.
func PlanV41PrefillExpertPanel(out, in, rows int) (V41PrefillExpertPanelPlan, error) {
	bad := func() (V41PrefillExpertPanelPlan, error) {
		return V41PrefillExpertPanelPlan{}, fmt.Errorf("compute: V4.1 prefill expert panel requires bounded [out,in] with 2 through %d rows", V41PrefillExpertPanelMaxRows)
	}
	if rows < 2 || rows > V41PrefillExpertPanelMaxRows || out <= 0 || in <= 0 ||
		uint64(out) > 0x7fffffff || uint64(in) > 0x7fffffff ||
		uint64(rows)*uint64(in)*4 > 0x7fffffff || uint64(rows)*uint64(out)*4 > 0x7fffffff {
		return bad()
	}
	return V41PrefillExpertPanelPlan{Out: out, In: in, Rows: rows,
		InputBytes: rows * in * 4, OutputBytes: rows * out * 4}, nil
}

// validateV41PrefillExpertPanel checks geometry only; native admission
// additionally checks module ABI, live ownership, storage extents and disjoint
// output bindings. The weight may carry any row-major two-dimensional dtype the
// backend serves; the activation panel must be flat unquantized F32.
func validateV41PrefillExpertPanel(weight, input Tensor, rows int) (V41PrefillExpertPanelPlan, error) {
	if weight.Layout != RowMajor || len(weight.Shape) != 2 {
		return V41PrefillExpertPanelPlan{}, fmt.Errorf("compute: V4.1 prefill expert panel needs a row-major two-dimensional expert matrix")
	}
	if input.Dtype != F32 || input.Layout != RowMajor || input.Quant != nil || len(input.Shape) != 1 {
		return V41PrefillExpertPanelPlan{}, fmt.Errorf("compute: V4.1 prefill expert panel activation must be a flat unquantized F32 panel")
	}
	plan, err := PlanV41PrefillExpertPanel(weight.Shape[0], weight.Shape[1], rows)
	if err != nil {
		return plan, err
	}
	if input.Shape[0] != plan.InputBytes/4 {
		return V41PrefillExpertPanelPlan{}, fmt.Errorf("compute: V4.1 prefill expert panel activation length does not match rows*in")
	}
	return plan, nil
}

// V41PrefillExpertPanelBackend is the optional batched DeepSeek-V4.1
// routed-expert prefill panel GEMM (DRC-K17 leaf 2). It contracts several
// activation rows through ONE expert's ring-staged projection matrix in one
// dispatch: weight [out,in] (a ring-staged k-quant or F32 matrix) and
// contiguous F32 [rows*in] activations produce caller-owned F32 [rows*out].
// It performs neither BF16 publication nor host readback; the caller retains
// its ordered activation/publication checks before publishing any result or
// state. A selected error is fatal and must not be replayed on the host.
// Availability is gated by the backend's own serving qualification: a CUDA
// build without the panel archive declines, exactly as V41RoutedLayerQualified
// gates the decode layer.
type V41PrefillExpertPanelBackend interface {
	SupportsV41PrefillExpertPanel(weight Tensor, rows int) bool
	V41PrefillExpertPanel(weight, input Tensor, rows int) (Tensor, error)
}
