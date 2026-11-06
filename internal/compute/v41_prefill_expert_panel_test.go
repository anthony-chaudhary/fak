package compute

import "testing"

// DRC-K17 leaf 1: the bounded storage geometry of the batched routed-expert
// prefill panel. The planner admits 2 through V41PrefillExpertPanelMaxRows rows
// against bounded [out,in] dimensions, keeps signed 32-bit byte extents safe on
// supported 32-bit software-test hosts, and stays dtype-neutral: the pinned
// slate serves gate/up as Q2_K and down as Q3_K while an F32-resident expert
// keeps its ordinary encoding, so dtype admission belongs to the backend's
// Supports probe, not the geometry planner.

// fak-test:runtime fast est=10ms lane=default
func TestV41PrefillExpertPanelPlanBounds(t *testing.T) {
	// Every legal panel width over the pinned V4.1 expert geometry (gate/up
	// out=2304 in=5120, down out=5120 in=2304) must plan exact byte extents.
	for _, dims := range [][3]int{{2304, 5120, 2}, {2304, 5120, 8}, {2304, 5120, 64}, {5120, 2304, 3}, {5120, 2304, 64}} {
		plan, err := PlanV41PrefillExpertPanel(dims[0], dims[1], dims[2])
		if err != nil {
			t.Fatalf("expert geometry %v refused: %v", dims, err)
		}
		if plan.Out != dims[0] || plan.In != dims[1] || plan.Rows != dims[2] ||
			plan.InputBytes != dims[2]*dims[1]*4 || plan.OutputBytes != dims[2]*dims[0]*4 {
			t.Fatalf("geometry %v planned %+v, want exact byte extents", dims, plan)
		}
	}
	for _, rows := range []int{2, 3, 7, 8, 16, 31, 32, 63, 64} {
		if _, err := PlanV41PrefillExpertPanel(5, 512, rows); err != nil {
			t.Fatalf("rows=%d refused: %v", rows, err)
		}
	}
	// A single-row decode must never select the batched dispatch, and neither
	// may an unbounded extent: the 32-bit byte caps refuse before allocation.
	for _, dims := range [][3]int{
		{5, 512, 0}, {5, 512, 1}, {5, 512, 65},
		{0, 512, 2}, {5, 0, 2}, {-1, 512, 2}, {5, -1, 2},
		{1 << 30, 512, 8}, {5, 1 << 30, 8},
		{1 << 29, 2, 2}, // output byte extent rows*out*4 overflows int32
		{2, 1 << 29, 2}, // input byte extent rows*in*4 overflows int32
	} {
		if _, err := PlanV41PrefillExpertPanel(dims[0], dims[1], dims[2]); err == nil {
			t.Fatalf("invalid geometry %v admitted", dims)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41PrefillExpertPanelValidate(t *testing.T) {
	const out, in, rows = 4, 512, 3
	data := make([]float32, out*in)
	inputData := make([]float32, rows*in)
	weight := NewF32(Default(), []int{out, in}, data)
	input := NewF32(Default(), []int{rows * in}, inputData)

	plan, err := validateV41PrefillExpertPanel(weight, input, rows)
	if err != nil {
		t.Fatalf("legal panel refused: %v", err)
	}
	want := V41PrefillExpertPanelPlan{Out: out, In: in, Rows: rows, InputBytes: rows * in * 4, OutputBytes: rows * out * 4}
	if plan != want {
		t.Fatalf("plan=%+v, want %+v", plan, want)
	}

	// The weight may carry any row-major two-dimensional dtype the backend
	// serves: a quantized matrix passes the geometry check exactly like F32.
	qWeight := makeTensor(Default(), Q2_K, RowMajor, []int{out, in}, &QuantSpec{}, nil)
	if _, err := validateV41PrefillExpertPanel(qWeight, input, rows); err != nil {
		t.Fatalf("quantized expert matrix refused by the geometry check: %v", err)
	}

	cases := []struct {
		name   string
		weight Tensor
		input  Tensor
		rows   int
	}{
		{"column-major weight", makeTensor(Default(), F32, ColMajor, []int{out, in}, nil, nil), input, rows},
		{"one-dimensional weight", NewF32(Default(), []int{out * in}, data), input, rows},
		{"three-dimensional weight", NewF32(Default(), []int{out, in, 1}, data), input, rows},
		{"quantized activation", makeTensor(Default(), F32, RowMajor, []int{rows * in}, &QuantSpec{}, &hostBuf{f32: inputData}), weight, rows},
		{"column-major activation", makeTensor(Default(), F32, ColMajor, []int{rows * in}, nil, &hostBuf{f32: inputData}), weight, rows},
		{"short activation", NewF32(Default(), []int{rows*in - 1}, inputData[:rows*in-1]), weight, rows},
		{"long activation", NewF32(Default(), []int{rows*in + 1}, append([]float32(nil), inputData...)), weight, rows},
		{"activation narrower than rows*in", NewF32(Default(), []int{in}, inputData[:in]), weight, rows},
		{"single row", weight, NewF32(Default(), []int{in}, inputData[:in]), 1},
		{"row ceiling exceeded", weight, NewF32(Default(), []int{65 * in}, make([]float32, 65*in)), 65},
	}
	for _, tc := range cases {
		if _, err := validateV41PrefillExpertPanel(tc.weight, tc.input, tc.rows); err == nil {
			t.Fatalf("%s admitted", tc.name)
		}
	}
}
