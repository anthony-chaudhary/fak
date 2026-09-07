package compute

import (
	"math"
	"testing"
)

func TestCPURMSNormPackedRows(t *testing.T) {
	c := cpu()
	x := []float32{1, -2, 3, -4, 100, 20, -30, 40, 0.001, -0.004, 0.002, 0.003}
	w := []float32{0.5, 2, -1, 1.25}
	eps := float32(0.125)
	weight := NewF32(c, []int{len(w)}, w)
	want := make([]float32, 0, len(x))
	for start := 0; start < len(x); start += len(w) {
		row := NewF32(c, []int{len(w)}, x[start:start+len(w)])
		want = append(want, c.Read(c.RMSNorm(row, weight, eps))...)
	}
	packed := NewF32(c, []int{3, 4}, x)
	result := c.RMSNorm(packed, weight, eps)
	got := c.Read(result)
	if len(result.Shape) != 2 || result.Shape[0] != 3 || result.Shape[1] != 4 || len(got) != len(want) {
		t.Fatalf("packed result shape=%v values=%d", result.Shape, len(got))
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Float32bits(v) != math.Float32bits(want[i]) {
			t.Fatalf("row %d column %d: %g want exact %g", i/len(w), i%len(w), v, want[i])
		}
	}
	for _, width := range []int{0, 5} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("weight width %d admitted for %d inputs", width, len(x))
				}
			}()
			c.RMSNorm(packed, NewF32(c, []int{width}, make([]float32, width)), eps)
		}()
	}
}
