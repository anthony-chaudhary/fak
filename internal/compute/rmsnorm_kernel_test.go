package compute

import (
	"math"
	"testing"
)

func TestRMSNormKernel(t *testing.T) {
	t.Parallel()
	x := []float32{1, -2, 3, -4, 0.5, 1.5, -2.5, 3.5}
	w := []float32{1.0, 0.5, 2.0, -1.0}
	const eps = float32(1e-6)

	got, err := FusedRMSNormF32(x, w, 2, 4, eps)
	if err != nil {
		t.Fatal(err)
	}
	// Two-pass reference check per row.
	for r := 0; r < 2; r++ {
		var meanSq float64
		for c := 0; c < 4; c++ {
			v := float64(x[r*4+c])
			meanSq += v * v
		}
		meanSq /= 4.0
		inv := 1.0 / math.Sqrt(meanSq+float64(eps))
		for c := 0; c < 4; c++ {
			want := float32(float64(x[r*4+c]) * inv * float64(w[c]))
			if math.Abs(float64(got[r*4+c]-want)) > 1e-6 {
				t.Fatalf("row %d col %d: got %v, want %v", r, c, got[r*4+c], want)
			}
		}
	}
	if _, err := FusedRMSNormF32(x, w, 2, 4, 0); err == nil {
		t.Fatal("expected error for eps=0")
	}
}
