package compute

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestV41TailRoPERowsContract(t *testing.T) {
	t.Parallel()
	x := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{3, 8}}
	table := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{2, 2}}
	if xb, tb, err := validateV41TailRoPERows(x, table, 3, 8, 4); err != nil || xb != 96 || tb != 16 {
		t.Fatalf("valid rows geometry: bytes=%d/%d err=%v", xb, tb, err)
	}
	for _, dims := range [][3]int{{0, 8, 4}, {3, 8, 3}, {3, 8, 10}, {3, 8, 0}} {
		if _, _, err := validateV41TailRoPERows(x, table, dims[0], dims[1], dims[2]); err == nil {
			t.Errorf("accepted invalid geometry %v", dims)
		}
	}
	for _, mutate := range []func(*Tensor){
		func(v *Tensor) { v.Shape = []int{24} },
		func(v *Tensor) { v.Shape = []int{2, 8} },
		func(v *Tensor) { v.Dtype = BF16 },
		func(v *Tensor) { v.Quant = &QuantSpec{} },
	} {
		bad := x
		mutate(&bad)
		if _, _, err := validateV41TailRoPERows(bad, table, 3, 8, 4); err == nil {
			t.Errorf("accepted malformed rows panel: %+v", bad)
		}
	}
	// The conjugate table is the inverse: rotating by [sin,cos] then by
	// [-sin,cos] returns a unit-amplitude input within one rounding per lane.
	input := []float32{7, -3, 0.75, -1.25}
	forward := []float32{0.6, 0.8}
	conj := []float32{-0.6, 0.8}
	back := v41TailRoPEReference(v41TailRoPEReference(input, forward, 4, 2), conj, 4, 2)
	for i, want := range input {
		if d := back[i] - want; d > 1e-6 || d < -1e-6 {
			t.Fatalf("conjugate round trip lane %d got %g want %g", i, back[i], want)
		}
	}
}
