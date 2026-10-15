package compute

import (
	"math"
	"testing"
)

func TestV41RoutedSwiGLUPanelRowPlan(t *testing.T) {
	for _, rows := range []int{1, 2, 3, 8, 31, 32, 64} {
		for row := 0; row < rows; row++ {
			p, err := PlanV41RoutedSwiGLUPanelRow(rows, 2304, row)
			if err != nil || p.Rows != rows || p.Width != 2304 || p.Row != row || p.RowBytes != 9216 || p.OffsetBytes != row*9216 || p.PanelBytes != rows*9216 || p.OffsetBytes+p.RowBytes > p.PanelBytes {
				t.Fatalf("rows=%d row=%d: %+v %v", rows, row, p, err)
			}
		}
	}
	for _, g := range [][3]int{{0, 1, 0}, {65, 1, 0}, {2, 0, 0}, {2, -1, 0}, {2, 1, -1}, {2, 1, 2}, {64, math.MaxInt32, 0}, {1, math.MaxInt32/4 + 1, 0}} {
		if p, err := PlanV41RoutedSwiGLUPanelRow(g[0], g[1], g[2]); err == nil || p != (V41RoutedSwiGLUPanelRowPlan{}) {
			t.Fatalf("accepted invalid geometry %v: %+v %v", g, p, err)
		}
	}
	if _, err := PlanV41RoutedSwiGLUPanelRow(1, math.MaxInt32/4, 0); err != nil {
		t.Fatal(err)
	}
}

func TestV41RoutedSwiGLUPanelRowValidation(t *testing.T) {
	good := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{32, 2304}}
	if p, err := validateV41RoutedSwiGLUPanelRow(good, good, 31, 0, -1); err != nil || p.OffsetBytes != 31*9216 {
		t.Fatalf("%+v %v", p, err)
	}
	bad := []Tensor{{Dtype: F32, Layout: RowMajor, Shape: []int{32 * 2304}}, {Dtype: Q2_K, Layout: RowMajor, Shape: []int{32, 2304}}, {Dtype: F32, Layout: RowMajor, Shape: []int{31, 2304}}, {Dtype: F32, Layout: RowMajor, Shape: []int{32, 2303}}}
	for _, other := range bad {
		for _, pair := range [][2]Tensor{{good, other}, {other, good}} {
			if _, err := validateV41RoutedSwiGLUPanelRow(pair[0], pair[1], 0, 1, 1); err == nil {
				t.Fatalf("accepted mismatched operand %+v", other)
			}
		}
	}
	for _, v := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if _, err := validateV41RoutedSwiGLUPanelRow(good, good, 0, v, 1); err == nil {
			t.Fatal("accepted nonfinite limit")
		}
		if _, err := validateV41RoutedSwiGLUPanelRow(good, good, 0, 1, v); err == nil {
			t.Fatal("accepted nonfinite weight")
		}
	}
}
