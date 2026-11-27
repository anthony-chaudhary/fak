package compute

import (
	"math"
	"testing"
)

func TestInk2FP8QuantizeDequantRows(t *testing.T) {
	rows, cols := 3, 8
	src := []float32{
		0.5, -1.25, 3.5, -0.125, 2.0, -4.0, 0.25, 1.0,
		0, 0, 0, 0, 0, 0, 0, 0, // zero row -> scale 0
		100.0, -250.0, 50.0, -12.5, 0.01, -0.02, 448.0, -448.0,
	}
	codes, scales := Ink2FP8QuantizeRows(src, rows, cols)
	if len(codes) != rows*cols || len(scales) != rows {
		t.Fatalf("lengths = (%d, %d), want (%d, %d)", len(codes), len(scales), rows*cols, rows)
	}
	if scales[1] != 0 {
		t.Fatalf("zero row scale = %v, want 0", scales[1])
	}
	for _, b := range codes {
		if b == 0x7F || b == 0xFF {
			t.Fatalf("emitted NaN E4M3 code %#02x", b)
		}
	}
	got := Ink2FP8DequantRows(codes, scales, rows, cols)
	for i, want := range src {
		err := math.Abs(float64(got[i] - want))
		tol := math.Max(1e-5, math.Abs(float64(want))*0.07) // E4M3FN has 3 mantissa bits (<= 6.25% rel err)
		if err > tol {
			t.Fatalf("elem %d: got %v, want %v (err %v > tol %v)", i, got[i], want, err, tol)
		}
	}
}

func TestInk2FP8ArenaBytesAlignment(t *testing.T) {
	if got := Ink2FP8ArenaBytes(100, 3); got != 128+64 {
		t.Fatalf("Ink2FP8ArenaBytes(100, 3) = %d, want 192", got)
	}
}
