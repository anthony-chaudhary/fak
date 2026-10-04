package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model/norm"
)

// preExtractionRMSNorm is the verbatim serial RMSNorm arithmetic that lived in
// forward.go before the extraction to internal/model/norm. It is retained here
// as an independent oracle so the migrated adapter can be compared against the
// original operation, not merely against the new helper.
func preExtractionRMSNorm(x, w []float32, eps float32) []float32 {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * inv * w[i]
	}
	return out
}

// TestRMSNormLeafAdapterParity proves the model-package rmsnorm adapter (the
// real consumer called by rmsnormCfg/normCfg through normCfg) delegates to
// norm.SerialRMSNorm and stays bit-identical to the pre-extraction arithmetic.
// No tolerance substitution or arithmetic reassociation is allowed.
func TestRMSNormLeafAdapterParity(t *testing.T) {
	nan := float32(math.NaN())
	pinf := float32(math.Inf(1))
	ninf := float32(math.Inf(-1))

	cases := []struct {
		name string
		x, w []float32
		eps  float32
	}{
		{"ramp", []float32{1, 2, 3, 4, 5}, []float32{0.1, -0.2, 0.3, 0.5, -0.7}, 1e-5},
		{"sign-mixed", []float32{-3.5, 0, 2.25, -0.125, 8}, []float32{1, 1, 1, 1, 1}, 1e-6},
		{"signed-zero-x", []float32{0, float32(math.Copysign(0, -1)), 0}, []float32{1, -1, 0}, 0},
		{"signed-zero-w", []float32{2, -2, 2}, []float32{0, float32(math.Copysign(0, -1)), 1}, 1e-5},
		{"single", []float32{7}, []float32{0.5}, 1e-5},
		{"large", []float32{1e19, -1e19, 3e18}, []float32{1, 1, 1}, 1e-5},
		{"nan-x", []float32{nan, 1, 2}, []float32{1, 1, 1}, 1e-5},
		{"posinf-x", []float32{pinf, 1, -2}, []float32{1, 1, 1}, 1e-5},
		{"neginf-x", []float32{ninf, 0, 3}, []float32{1, 1, 1}, 1e-5},
		{"nan-w", []float32{1, 2, 3}, []float32{1, nan, 1}, 1e-5},
		{"inf-w", []float32{1, 2, 3}, []float32{1, ninf, 1}, 1e-5},
	}

	for _, tc := range cases {
		x := append([]float32(nil), tc.x...)
		w := append([]float32(nil), tc.w...)
		xBefore := append([]float32(nil), x...)
		wBefore := append([]float32(nil), w...)

		want := preExtractionRMSNorm(tc.x, tc.w, tc.eps)
		gotAdapter := rmsnorm(x, w, tc.eps)
		gotLeaf := norm.SerialRMSNorm(tc.x, tc.w, tc.eps)

		assertFloat32BitsEqual(t, tc.name+"/adapter==oracle", want, gotAdapter)
		assertFloat32BitsEqual(t, tc.name+"/leaf==oracle", want, gotLeaf)
		assertFloat32BitsEqual(t, tc.name+"/x-unmutated", xBefore, x)
		assertFloat32BitsEqual(t, tc.name+"/w-unmutated", wBefore, w)

		if len(x) > 0 && len(gotAdapter) > 0 && &gotAdapter[0] == &x[0] {
			t.Fatalf("%s: adapter output aliases input x (must be a fresh allocation)", tc.name)
		}
	}

	if out := rmsnorm(nil, nil, 1e-5); len(out) != 0 {
		t.Fatalf("empty input: adapter len = %d, want 0", len(out))
	}
}
