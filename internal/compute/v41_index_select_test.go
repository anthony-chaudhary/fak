package compute

import (
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41IndexSelectContract(t *testing.T) {
	t.Parallel()
	scores := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{5}}
	if n, err := validateV41IndexSelect(scores, 5, 8, 0, 0); err != nil || n != 5 {
		t.Fatalf("valid geometry: selected=%d err=%v", n, err)
	}
	if n, err := validateV41IndexSelect(scores, 5, 2, 1, 3); err != nil || n != 2 {
		t.Fatalf("valid masked geometry: selected=%d err=%v", n, err)
	}
	for _, g := range [][4]int{{0, 1, 0, 0}, {5, 0, 0, 0}, {5, 1, -1, 0}, {5, 1, 1, 0}, {V41IndexSelectMaxRows + 1, 1, 0, 0}} {
		if _, err := validateV41IndexSelect(scores, g[0], g[1], g[2], g[3]); err == nil {
			t.Errorf("accepted invalid geometry %v", g)
		}
	}
	for _, mutate := range []func(*Tensor){
		func(v *Tensor) { v.Shape = []int{1, 5} },
		func(v *Tensor) { v.Dtype = BF16 },
		func(v *Tensor) { v.Layout = ColMajor },
		func(v *Tensor) { v.Quant = &QuantSpec{} },
	} {
		bad := scores
		mutate(&bad)
		if _, err := validateV41IndexSelect(bad, 5, 1, 0, 0); err == nil {
			t.Errorf("accepted malformed scores: %+v", bad)
		}
	}
	if got := v41IndexSelectScratchWords(9, 2, 4); got != 2*3+9 {
		t.Fatalf("scratch words=%d", got)
	}

	negZero := float32(math.Copysign(0, -1))
	ninf := float32(math.Inf(-1))
	for _, tc := range []struct {
		scores              []float32
		topK, blocks, bsize int
		want                []int32
	}{
		// Ties rank the lower position first; -0 ties +0.
		{[]float32{1, 2, 2, 1}, 2, 0, 0, []int32{1, 2}},
		{[]float32{negZero, 0, negZero}, 2, 0, 0, []int32{0, 1}},
		// A -Inf selection publishes -1 in its position-sorted slot.
		{[]float32{ninf, 5, ninf}, 2, 0, 0, []int32{-1, 1}},
		// The newest block is pinned even when its scores are lowest.
		{[]float32{9, 9, 9, 9, -5}, 2, 1, 2, []int32{-1, 4}},
		{[]float32{9, 8, 7}, 5, 0, 0, []int32{0, 1, 2}},
	} {
		got, err := V41IndexSelectReference(tc.scores, tc.topK, tc.blocks, tc.bsize)
		if err != nil || !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("scores=%v topK=%d blocks=%d/%d: got %v err %v want %v", tc.scores, tc.topK, tc.blocks, tc.bsize, got, err, tc.want)
		}
	}
}
