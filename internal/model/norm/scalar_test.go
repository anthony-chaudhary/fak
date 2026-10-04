package norm

import (
	"math"
	"testing"
)

// legacySerialRMSNorm is the verbatim pre-extraction implementation, retained
// here as an independent oracle. It is intentionally NOT refactored: its serial
// in-order reduction, epsilon placement, sqrt conversion and multiplication
// order ARE the contract the extracted SerialRMSNorm must preserve bit-for-bit.
func legacySerialRMSNorm(x, w []float32, eps float32) []float32 {
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

func assertBits(t *testing.T, name string, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: len = %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
			t.Fatalf("%s[%d] = %08x (%g), want %08x (%g)",
				name, i, math.Float32bits(got[i]), got[i],
				math.Float32bits(want[i]), want[i])
		}
	}
}

// TestSerialRMSNormBitExact pins the extracted serial reference bit-for-bit
// against the verbatim original arithmetic across finite, signed-zero and
// non-finite payloads, and proves the inputs are not mutated.
func TestSerialRMSNormBitExact(t *testing.T) {
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
		{"zero-eps", []float32{3, 4}, []float32{1, 1}, 0},
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

		want := legacySerialRMSNorm(tc.x, tc.w, tc.eps)
		got := SerialRMSNorm(x, w, tc.eps)

		assertBits(t, tc.name, want, got)
		assertBits(t, tc.name+"/x-unmutated", xBefore, x)
		assertBits(t, tc.name+"/w-unmutated", wBefore, w)

		if len(x) > 0 && len(got) > 0 && &got[0] == &x[0] {
			t.Fatalf("%s: output aliases input x (must be a fresh allocation)", tc.name)
		}
	}

	// Empty input is a zero-length result, matching the original loop shape.
	if out := SerialRMSNorm(nil, nil, 1e-5); len(out) != 0 {
		t.Fatalf("empty input: len = %d, want 0", len(out))
	}
}
