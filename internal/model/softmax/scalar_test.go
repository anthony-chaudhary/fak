package softmax

import (
	"math"
	"testing"
)

// legacyInPlace is the verbatim pre-extraction in-place helper retained as an
// independent oracle. It is intentionally NOT refactored: its max scan,
// float32 exp/sum accumulation and normalize order ARE the contract the
// extracted InPlace must preserve bit-for-bit.
func legacyInPlace(s []float32) {
	mx := s[0]
	for _, v := range s {
		if v > mx {
			mx = v
		}
	}
	var sum float32
	for i, v := range s {
		e := float32(math.Exp(float64(v - mx)))
		s[i] = e
		sum += e
	}
	for i := range s {
		s[i] /= sum
	}
}

// legacyCopy is the verbatim pre-extraction allocating helper retained as the
// oracle for Copy.
func legacyCopy(z []float32) []float32 {
	out := make([]float32, len(z))
	mx := z[0]
	for _, v := range z {
		if v > mx {
			mx = v
		}
	}
	var sum float32
	for i, v := range z {
		e := float32(math.Exp(float64(v - mx)))
		out[i] = e
		sum += e
	}
	for i := range out {
		out[i] /= sum
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

func softmaxCases() [][]float32 {
	nan := float32(math.NaN())
	pinf := float32(math.Inf(1))
	ninf := float32(math.Inf(-1))
	return [][]float32{
		{1, 2, 3, 4, 5},
		{-3.5, 0, 2.25, -0.125, 8},
		{0, float32(math.Copysign(0, -1)), 0},
		{-1, -1, -1},
		{7},
		{1e19, -1e19, 3e18},
		{1e-30, 1e-30, 1e-30},
		{nan, 1, 2},
		{pinf, 1, -2},
		{ninf, 0, 3},
	}
}

// TestScalarSoftmaxBitExact pins both extracted helpers bit-for-bit against the
// verbatim originals across finite, signed-zero, uniform and non-finite
// payloads, and proves InPlace mutates only its own slice while Copy leaves its
// input unmutated.
func TestScalarSoftmaxBitExact(t *testing.T) {
	for ci, tc := range softmaxCases() {
		// In-place contract.
		want := append([]float32(nil), tc...)
		legacyInPlace(want)
		got := append([]float32(nil), tc...)
		InPlace(got)
		assertBits(t, "inplace", want, got)

		// Allocating contract: input unmutated, fresh allocation.
		inBefore := append([]float32(nil), tc...)
		wantCopy := legacyCopy(tc)
		gotCopy := Copy(tc)
		assertBits(t, "copy", wantCopy, gotCopy)
		assertBits(t, "copy/z-unmutated", inBefore, tc)
		if len(gotCopy) > 0 && &gotCopy[0] == &tc[0] {
			t.Fatalf("case %d: Copy output aliases input z (must be a fresh allocation)", ci)
		}
	}
}

// TestScalarSoftmaxEmptyPanics locks the empty-input contract: both originals
// index element 0 on the max scan, so an empty slice must panic rather than
// return an undefined result.
func TestScalarSoftmaxEmptyPanics(t *testing.T) {
	assertPanics := func(name string, fn func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("%s: expected panic on empty input", name)
			}
		}()
		fn()
	}
	assertPanics("InPlace", func() { InPlace(nil) })
	assertPanics("Copy", func() { Copy(nil) })
}
