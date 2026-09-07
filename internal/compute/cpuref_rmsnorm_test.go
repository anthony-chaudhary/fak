package compute

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

func TestCPURMSNorm(t *testing.T) {
	t.Run("MultiRowRegression", func(t *testing.T) {
		testCPURMSNormMultiRow(t)
	})
	t.Run("InvalidGeometry", func(t *testing.T) {
		testCPURMSNormInvalidGeometry(t)
	})
	t.Run("SingleRowParity", func(t *testing.T) {
		testCPURMSNormSingleRow(t)
	})
}

// TestCPURMSNorm_MultiRow compares multi-row output against single-row reference calls,
// and verifies panic on invalid geometry (non-divisible or zero).
func TestCPURMSNorm_MultiRow(t *testing.T) {
	t.Run("MultiRowRegression", func(t *testing.T) {
		testCPURMSNormMultiRow(t)
	})
	t.Run("InvalidGeometry", func(t *testing.T) {
		testCPURMSNormInvalidGeometry(t)
	})
}

func TestRMSNormCPURowSemantics(t *testing.T) {
	TestCPURMSNorm(t)
}

func testCPURMSNormMultiRow(t *testing.T) {
	c := cpu()
	var s lcg = 42
	eps := float32(1e-5)

	// Test case 1: 4 rows of width 16 (2D shape [4, 16])
	rows, width := 4, 16
	x := randVec(&s, rows*width)
	w := randVec(&s, width)

	xTensor := NewF32(c, []int{rows, width}, x)
	wTensor := NewF32(c, []int{width}, w)
	outTensor := c.RMSNorm(xTensor, wTensor, eps)

	if len(outTensor.Shape) != 2 || outTensor.Shape[0] != rows || outTensor.Shape[1] != width {
		t.Fatalf("unexpected shape: got %v, want [%d, %d]", outTensor.Shape, rows, width)
	}

	got := c.Read(outTensor)
	for r := 0; r < rows; r++ {
		rowSlice := x[r*width : (r+1)*width]
		rowTensor := NewF32(c, []int{width}, rowSlice)
		refTensor := c.RMSNorm(rowTensor, wTensor, eps)
		ref := c.Read(refTensor)

		for i := 0; i < width; i++ {
			idx := r*width + i
			if math.Float32bits(got[idx]) != math.Float32bits(ref[i]) {
				t.Fatalf("row %d elem %d mismatch: multi-row got %g (bits %x), single-row ref %g (bits %x)",
					r, i, got[idx], math.Float32bits(got[idx]), ref[i], math.Float32bits(ref[i]))
			}
		}
	}

	// Test case 2: 3 rows of width 32 (3D shape [1, 3, 32] - multihead layout)
	rows2, width2 := 3, 32
	x2 := randVec(&s, rows2*width2)
	w2 := randVec(&s, width2)

	xTensor2 := NewF32(c, []int{1, rows2, width2}, x2)
	wTensor2 := NewF32(c, []int{width2}, w2)
	outTensor2 := c.RMSNorm(xTensor2, wTensor2, eps)

	if len(outTensor2.Shape) != 3 || outTensor2.Shape[0] != 1 || outTensor2.Shape[1] != rows2 || outTensor2.Shape[2] != width2 {
		t.Fatalf("unexpected 3D shape: got %v, want [1, %d, %d]", outTensor2.Shape, rows2, width2)
	}

	got2 := c.Read(outTensor2)
	for r := 0; r < rows2; r++ {
		rowSlice := x2[r*width2 : (r+1)*width2]
		rowTensor := NewF32(c, []int{width2}, rowSlice)
		refTensor := c.RMSNorm(rowTensor, wTensor2, eps)
		ref := c.Read(refTensor)

		for i := 0; i < width2; i++ {
			idx := r*width2 + i
			if math.Float32bits(got2[idx]) != math.Float32bits(ref[i]) {
				t.Fatalf("3D row %d elem %d mismatch: multi-row got %g (bits %x), single-row ref %g (bits %x)",
					r, i, got2[idx], math.Float32bits(got2[idx]), ref[i], math.Float32bits(ref[i]))
			}
		}
	}
}

func testCPURMSNormInvalidGeometry(t *testing.T) {
	c := cpu()
	eps := float32(1e-5)

	assertPanics := func(name string, fn func()) {
		t.Helper()
		defer func() {
			r := recover()
			if r == nil {
				t.Fatalf("%s: expected panic, did not panic", name)
			}
			msg := fmt.Sprintf("%v", r)
			if !strings.Contains(msg, "RMSNorm invalid geometry") {
				t.Fatalf("%s: expected panic containing 'RMSNorm invalid geometry', got: %v", name, r)
			}
		}()
		fn()
	}

	// 1. len(xf) not divisible by len(wf): len(xf) = 35, len(wf) = 16
	assertPanics("not divisible (35 by 16)", func() {
		x := NewF32(c, []int{35}, make([]float32, 35))
		w := NewF32(c, []int{16}, make([]float32, 16))
		_ = c.RMSNorm(x, w, eps)
	})

	// 2. empty weight: len(wf) = 0
	assertPanics("empty weight", func() {
		x := NewF32(c, []int{16}, make([]float32, 16))
		w := NewF32(c, []int{0}, make([]float32, 0))
		_ = c.RMSNorm(x, w, eps)
	})

	// 3. empty input: len(xf) = 0
	assertPanics("empty input", func() {
		x := NewF32(c, []int{0}, make([]float32, 0))
		w := NewF32(c, []int{16}, make([]float32, 16))
		_ = c.RMSNorm(x, w, eps)
	})

	// 4. both empty: len(xf) = 0, len(wf) = 0
	assertPanics("both empty", func() {
		x := NewF32(c, []int{0}, make([]float32, 0))
		w := NewF32(c, []int{0}, make([]float32, 0))
		_ = c.RMSNorm(x, w, eps)
	})

	// 5. input smaller than weight: len(xf) = 10, len(wf) = 16
	assertPanics("input smaller than weight", func() {
		x := NewF32(c, []int{10}, make([]float32, 10))
		w := NewF32(c, []int{16}, make([]float32, 16))
		_ = c.RMSNorm(x, w, eps)
	})
}

func testCPURMSNormSingleRow(t *testing.T) {
	c := cpu()
	var s lcg = 101
	n := 64
	x := randVec(&s, n)
	w := randVec(&s, n)
	eps := float32(1e-5)

	xTensor := NewF32(c, []int{n}, x)
	wTensor := NewF32(c, []int{n}, w)
	got := c.Read(c.RMSNorm(xTensor, wTensor, eps))

	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(n)+eps)))

	for i := 0; i < n; i++ {
		want := x[i] * inv * w[i]
		if math.Float32bits(got[i]) != math.Float32bits(want) {
			t.Fatalf("SingleRow index %d drift: got %g (bits %x), want %g (bits %x)",
				i, got[i], math.Float32bits(got[i]), want, math.Float32bits(want))
		}
	}
}
