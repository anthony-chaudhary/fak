package compute

import (
	"errors"
	"math"
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41RouterTopKContract(t *testing.T) {
	t.Parallel()
	logits := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{2, 384}}
	bias := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{384}}
	dummy := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{1}}
	lb, bb, pb, sb, err := validateV41RouterTopK(logits, bias, 2, 384, 6, 1.5, true)
	if err != nil || lb != 2*384*4 || bb != 384*4 || pb != 2*6*4 || sb != 2*16 {
		t.Fatalf("valid V4.1 geometry: bytes=%d/%d/%d/%d err=%v", lb, bb, pb, sb, err)
	}
	if _, bb, _, _, err := validateV41RouterTopK(logits, dummy, 2, 384, 6, 1.5, false); err != nil || bb != 4 {
		t.Fatalf("bias-free route must take a [1] dummy: bytes=%d err=%v", bb, err)
	}
	for _, dims := range [][3]int{{0, 384, 6}, {2, 0, 6}, {2, 384, 0}, {2, 4, 6}, {2, 384, V41RouterTopKMaxK + 1}} {
		if _, _, _, _, err := validateV41RouterTopK(logits, bias, dims[0], dims[1], dims[2], 1.5, true); err == nil {
			t.Errorf("accepted invalid geometry %v", dims)
		}
	}
	for _, scale := range []float32{0, -1, float32(math.Inf(1)), float32(math.NaN())} {
		if _, _, _, _, err := validateV41RouterTopK(logits, bias, 2, 384, 6, scale, true); err == nil {
			t.Errorf("accepted route scale %g", scale)
		}
	}
	for _, mutate := range []func(*Tensor){
		func(x *Tensor) { x.Shape = []int{768} },
		func(x *Tensor) { x.Dtype = BF16 },
		func(x *Tensor) { x.Layout = ColMajor },
		func(x *Tensor) { x.Quant = &QuantSpec{} },
	} {
		bad := logits
		mutate(&bad)
		if _, _, _, _, err := validateV41RouterTopK(bad, bias, 2, 384, 6, 1.5, true); err == nil {
			t.Errorf("accepted malformed logits: %+v", bad)
		}
	}
	if _, _, _, _, err := validateV41RouterTopK(logits, dummy, 2, 384, 6, 1.5, true); err == nil {
		t.Fatal("accepted a [1] bias with hasBias")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestV41RouterTopKDecode(t *testing.T) {
	t.Parallel()
	ok := []uint32{0, 0, 0, 0, 0, 0, 0, 0}
	picks, err := decodeV41RouterTopK(ok, []uint32{5, 1, 2, 5}, []float32{0.9, 0.6, 0.8, 0.7}, 2, 8, 2)
	if err != nil || len(picks.Experts) != 4 || picks.Experts[0] != 5 || picks.Experts[3] != 5 || picks.Weights[2] != 0.8 {
		t.Fatalf("valid decode: %+v err=%v", picks, err)
	}
	fault := []uint32{0, 0, 0, 0, uint32(V41RouterTopKStageBias), 4, 0x7fc00000, 0}
	_, err = decodeV41RouterTopK(fault, []uint32{5, 1, 0, 0}, make([]float32, 4), 2, 8, 2)
	var f *V41RouterTopKFault
	if !errors.As(err, &f) || f.Row != 1 || f.Stage != V41RouterTopKStageBias || f.Expert != 3 || f.Bits != 0x7fc00000 {
		t.Fatalf("fault record decode: %v", err)
	}
	norm := []uint32{uint32(V41RouterTopKStageNormalization), 0, 0, 0}
	if _, err = decodeV41RouterTopK(norm, []uint32{0}, []float32{0}, 1, 8, 1); !errors.As(err, &f) || f.Expert != -1 {
		t.Fatalf("row-level normalization fault must carry expert -1: %v", err)
	}
	for name, tc := range map[string]struct {
		status  []uint32
		experts []uint32
	}{
		"dirty-success":   {[]uint32{0, 1, 0, 0}, []uint32{1, 2}},
		"unknown-stage":   {[]uint32{7, 1, 0, 0}, []uint32{1, 2}},
		"expert-overflow": {[]uint32{2, 9, 0, 0}, []uint32{1, 2}},
		"reserved-word":   {[]uint32{2, 1, 0, 1}, []uint32{1, 2}},
		"duplicate-pick":  {[]uint32{0, 0, 0, 0}, []uint32{3, 3}},
		"out-of-range":    {[]uint32{0, 0, 0, 0}, []uint32{3, 8}},
		"short-status":    {[]uint32{0, 0, 0}, []uint32{1, 2}},
	} {
		_, err := decodeV41RouterTopK(tc.status, tc.experts, make([]float32, 2), 1, 8, 2)
		if err == nil || errors.As(err, &f) {
			t.Errorf("%s: want a malformed-readback error, got %v", name, err)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
// Source witness only: no compiler invocation or GPU proof.
func TestV41RouterTopKSourceContract(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile("shaders/v41_router_topk.comp")
	if err != nil {
		t.Fatal(err)
	}
	src := string(raw)
	for _, want := range []string{
		"layout(local_size_x = 1) in;",
		"if (c > selectedValues[slot])", // strict > keeps the lower expert on ties
		"int filled = min(e, pc.topK);",
		"selectedScores[insert] = s;",
		"fault(row, 1u, e, z)", "fault(row, 2u, e, s)", "fault(row, 3u, e, b)", "fault(row, 4u, e, c)",
		"fault(row, 5u, -1, sum)", "fault(row, 6u, picked[slot], w)",
		"(pc.full == 0 && sum == 0.0)",
		"precise float d = sum + 1e-20;",
		"precise float q = s / denominator;", "precise float w = q * pc.routeScale;",
		"const int MAX_K = 8;",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("router shader lost contract fragment %q", want)
		}
	}
	if strings.Count(src, "routerScore(") != 2 || strings.Count(src, "precise float s = selectedScores[slot];") != 2 {
		t.Fatal("router must evaluate each expert score once and reuse selected raw scores for sum and weights")
	}
	if V41RouterTopKStageLogit != 1 || V41RouterTopKStageWeight != 6 || V41RouterTopKMaxK != 8 {
		t.Fatal("Go stage or bound constants drifted from the shader ABI")
	}
}
