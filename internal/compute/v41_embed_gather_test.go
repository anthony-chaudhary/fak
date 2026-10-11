package compute

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestV41EmbeddingGatherContract(t *testing.T) {
	t.Parallel()
	table := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{6, 4}}
	if rowBytes, err := validateV41EmbeddingGather(table, []int{0, 5, 5, 2}, 4); err != nil || rowBytes != 16 {
		t.Fatalf("valid gather: rowBytes=%d err=%v", rowBytes, err)
	}
	for _, ids := range [][]int{nil, {6}, {-1}, {0, 7}} {
		if _, err := validateV41EmbeddingGather(table, ids, 4); err == nil {
			t.Errorf("accepted invalid ids %v", ids)
		}
	}
	for _, mutate := range []func(*Tensor){
		func(v *Tensor) { v.Shape = []int{24} },
		func(v *Tensor) { v.Shape = []int{6, 5} },
		func(v *Tensor) { v.Dtype = BF16 },
		func(v *Tensor) { v.Layout = ColMajor },
		func(v *Tensor) { v.Quant = &QuantSpec{} },
	} {
		bad := table
		mutate(&bad)
		if _, err := validateV41EmbeddingGather(bad, []int{1}, 4); err == nil {
			t.Errorf("accepted malformed table: %+v", bad)
		}
	}
	if _, err := validateV41EmbeddingGather(table, []int{1}, 0); err == nil {
		t.Fatal("accepted zero hidden width")
	}
}
