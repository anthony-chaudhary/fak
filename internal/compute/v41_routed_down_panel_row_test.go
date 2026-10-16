package compute

import (
	"math"
	"testing"
)

func TestV41RoutedDownPanelRowGeometry(t *testing.T) {
	for rows := 1; rows <= 64; rows++ {
		for row := 0; row < rows; row++ {
			p := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{rows, 65}}
			r, w, e := validateV41RoutedDownPanelRow(p, row)
			if e != nil || r != rows || w != 65 {
				t.Fatalf("%d/%d: %v", rows, row, e)
			}
		}
	}
	for _, shape := range [][]int{{0, 1}, {65, 1}, {1, 0}, {1, -1}, {1, math.MaxInt32/4 + 1}, {64, math.MaxInt32/4/64 + 1}, {3}, {1, 2, 3}} {
		if _, _, e := validateV41RoutedDownPanelRow(Tensor{Dtype: F32, Layout: RowMajor, Shape: shape}, 0); e == nil {
			t.Fatalf("invalid %v", shape)
		}
	}
	p := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{3, 65}}
	for _, row := range []int{-1, 3} {
		if _, _, e := validateV41RoutedDownPanelRow(p, row); e == nil {
			t.Fatal("invalid row")
		}
	}
	for _, change := range []func(*Tensor){func(x *Tensor) { x.Dtype = Q2_K }, func(x *Tensor) { x.Layout = Layout(99) }, func(x *Tensor) { x.Quant = &QuantSpec{} }} {
		bad := p
		change(&bad)
		if _, _, e := validateV41RoutedDownPanelRow(bad, 0); e == nil {
			t.Fatal("invalid storage geometry")
		}
	}
}
