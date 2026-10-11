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

// fak-test:runtime fast est=1ms lane=default
func TestV41Q2KEmbeddingGatherContract(t *testing.T) {
	t.Parallel()
	table := Tensor{Dtype: Q2_K, Layout: RowMajor, Shape: []int{6, 512}}
	if rowBytes, err := validateV41Q2KEmbeddingGather(table, []int{0, 5, 5}, 512); err != nil || rowBytes != 2*q2kSuperBlock {
		t.Fatalf("valid gather: rowBytes=%d err=%v", rowBytes, err)
	}
	for _, ids := range [][]int{nil, {6}, {-1}} {
		if _, err := validateV41Q2KEmbeddingGather(table, ids, 512); err == nil {
			t.Errorf("accepted invalid ids %v", ids)
		}
	}
	for _, hidden := range []int{0, 300, 256} {
		if _, err := validateV41Q2KEmbeddingGather(table, []int{1}, hidden); err == nil {
			t.Errorf("accepted hidden %d", hidden)
		}
	}
	for _, mutate := range []func(*Tensor){
		func(v *Tensor) { v.Dtype = F32 },
		func(v *Tensor) { v.Dtype = Q4_K },
		func(v *Tensor) { v.Shape = []int{6 * 512} },
	} {
		bad := table
		mutate(&bad)
		if _, err := validateV41Q2KEmbeddingGather(bad, []int{1}, 512); err == nil {
			t.Errorf("accepted malformed table: %+v", bad)
		}
	}
}
