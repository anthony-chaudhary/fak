package model

import (
	"errors"
	"math"
	"testing"
)

// approxEqual reports whether a and b agree within a float32-friendly tolerance.
func approxEqual(a, b float32, tol float32) bool {
	return float32(math.Abs(float64(a-b))) <= tol
}

func rowsApproxEqual(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !approxEqual(a[i], b[i], tol) {
			return false
		}
	}
	return true
}

// TestHadamardRoundTripIdentity checks the involution: rotate then un-rotate restores
// the original row for every supported power-of-two length.
func TestHadamardRoundTripIdentity(t *testing.T) {
	const tol = 1e-4
	rows := [][]float32{
		{3.5},
		{1.0, -2.0},
		{0.25, -1.5, 4.0, 7.0},
		{8, -1, 0.5, 2, -3, 6, -0.75, 1.25},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
	}
	for _, orig := range rows {
		work := append([]float32(nil), orig...)
		if err := HadamardRotate(work); err != nil {
			t.Fatalf("HadamardRotate(len=%d) unexpected error: %v", len(orig), err)
		}
		if err := HadamardInverse(work); err != nil {
			t.Fatalf("HadamardInverse(len=%d) unexpected error: %v", len(orig), err)
		}
		if !rowsApproxEqual(work, orig, tol) {
			t.Fatalf("round trip len=%d: got %v want %v", len(orig), work, orig)
		}
	}
}

// TestHadamardKnownH2 pins the normalized H_2 result against a hand-computed value.
func TestHadamardKnownH2(t *testing.T) {
	const tol = 1e-6
	inv := float32(1.0 / math.Sqrt(2))
	cases := []struct {
		in   []float32
		want []float32
	}{
		{[]float32{1, 0}, []float32{inv, inv}},
		{[]float32{1, 1}, []float32{2 * inv, 0}},
		{[]float32{3, 5}, []float32{8 * inv, -2 * inv}},
	}
	for _, c := range cases {
		work := append([]float32(nil), c.in...)
		if err := HadamardTransform(work); err != nil {
			t.Fatalf("HadamardTransform(%v) error: %v", c.in, err)
		}
		if !rowsApproxEqual(work, c.want, tol) {
			t.Fatalf("H_2 %v: got %v want %v", c.in, work, c.want)
		}
	}
}

// TestHadamardKnownH4 pins the normalized H_4 result: a unit spike at index 0 becomes an
// even 0.5 across all four dimensions, and the constant row collapses to the first bin.
func TestHadamardKnownH4(t *testing.T) {
	const tol = 1e-6
	cases := []struct {
		in   []float32
		want []float32
	}{
		{[]float32{1, 0, 0, 0}, []float32{0.5, 0.5, 0.5, 0.5}},
		{[]float32{1, 1, 1, 1}, []float32{2, 0, 0, 0}},
		// Row [1,2,3,4]: unnormalized WHT = [10,-2,-4,0], /sqrt(4)=2 -> [5,-1,-2,0].
		{[]float32{1, 2, 3, 4}, []float32{5, -1, -2, 0}},
	}
	for _, c := range cases {
		work := append([]float32(nil), c.in...)
		if err := HadamardTransform(work); err != nil {
			t.Fatalf("HadamardTransform(%v) error: %v", c.in, err)
		}
		if !rowsApproxEqual(work, c.want, tol) {
			t.Fatalf("H_4 %v: got %v want %v", c.in, work, c.want)
		}
	}
}

// TestHadamardLinearity checks H(a*x + b*y) == a*H(x) + b*H(y).
func TestHadamardLinearity(t *testing.T) {
	const tol = 1e-4
	x := []float32{2, -1, 0.5, 3, -4, 1, 0, 6}
	y := []float32{1, 1, -2, 0, 5, -3, 2, -1}
	const a, b float32 = 1.5, -0.75

	combined := make([]float32, len(x))
	for i := range x {
		combined[i] = a*x[i] + b*y[i]
	}
	if err := HadamardTransform(combined); err != nil {
		t.Fatalf("transform combined: %v", err)
	}

	hx := append([]float32(nil), x...)
	hy := append([]float32(nil), y...)
	if err := HadamardTransform(hx); err != nil {
		t.Fatalf("transform x: %v", err)
	}
	if err := HadamardTransform(hy); err != nil {
		t.Fatalf("transform y: %v", err)
	}
	want := make([]float32, len(x))
	for i := range x {
		want[i] = a*hx[i] + b*hy[i]
	}
	if !rowsApproxEqual(combined, want, tol) {
		t.Fatalf("linearity: got %v want %v", combined, want)
	}
}

// TestHadamardSpreadsOutlier checks the core purpose: a single large outlier is spread
// across dimensions after the forward rotation, so the peak magnitude and the
// concentration ratio both drop sharply.
func TestHadamardSpreadsOutlier(t *testing.T) {
	row := []float32{8, 0, 0, 0, 0, 0, 0, 0}
	beforePeak := MaxAbs(row)
	beforeRatio := OutlierRatio(row)

	if err := HadamardRotate(row); err != nil {
		t.Fatalf("rotate: %v", err)
	}
	afterPeak := MaxAbs(row)
	afterRatio := OutlierRatio(row)

	if !(afterPeak < beforePeak) {
		t.Fatalf("expected peak to drop: before=%v after=%v", beforePeak, afterPeak)
	}
	if !(afterRatio < beforeRatio) {
		t.Fatalf("expected outlier ratio to drop: before=%v after=%v", beforeRatio, afterRatio)
	}
	// The spike 8 at one of 8 bins becomes 8/sqrt(8)=2.828... in every bin; ratio -> 1.
	wantPeak := float32(8.0 / math.Sqrt(8))
	if !approxEqual(afterPeak, wantPeak, 1e-4) {
		t.Fatalf("spread peak: got %v want %v", afterPeak, wantPeak)
	}
	if math.Abs(afterRatio-1.0) > 1e-4 {
		t.Fatalf("spread ratio: got %v want ~1", afterRatio)
	}
}

// TestHadamardRejectsNonPowerOfTwo checks that non-power-of-two lengths return the typed
// ErrNotPowerOfTwo and leave the row untouched.
func TestHadamardRejectsNonPowerOfTwo(t *testing.T) {
	for _, n := range []int{0, 3, 5, 6, 7, 9, 12} {
		row := make([]float32, n)
		for i := range row {
			row[i] = float32(i + 1)
		}
		snapshot := append([]float32(nil), row...)
		err := HadamardTransform(row)
		if err == nil {
			t.Fatalf("len=%d: expected ErrNotPowerOfTwo, got nil", n)
		}
		var typed ErrNotPowerOfTwo
		if !errors.As(err, &typed) {
			t.Fatalf("len=%d: expected ErrNotPowerOfTwo, got %T (%v)", n, err, err)
		}
		if typed.N != n {
			t.Fatalf("len=%d: ErrNotPowerOfTwo.N = %d, want %d", n, typed.N, n)
		}
		if !rowsApproxEqual(row, snapshot, 0) {
			t.Fatalf("len=%d: row mutated on rejection: %v", n, row)
		}
	}
}
