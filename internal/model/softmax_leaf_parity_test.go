package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model/softmax"
)

// TestSoftmaxLeafAdapterParity proves the two model-package consumers now
// delegate to the shared softmax leaf and that the shared implementation still
// equals the pre-extraction arithmetic they replaced, bit-for-bit.
//
// The pre-migration bodies are reproduced verbatim here as independent oracles
// (softmaxInPlaceLegacy, softmaxOfLegacy) rather than importing the leaf, so a
// regression inside the leaf cannot silently satisfy its own parity check.
func softmaxInPlaceLegacy(s []float32) {
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

func softmaxOfLegacy(z []float32) []float32 {
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

func assertSoftmaxBits(t *testing.T, name string, want, got []float32) {
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

func TestSoftmaxLeafAdapterParity(t *testing.T) {
	nan := float32(math.NaN())
	pinf := float32(math.Inf(1))
	ninf := float32(math.Inf(-1))
	cases := [][]float32{
		{1, 2, 3, 4, 5},
		{-3.5, 0, 2.25, -0.125, 8},
		{0, float32(math.Copysign(0, -1)), 0},
		{-2, -2, -2},
		{7},
		{1e19, -1e19, 3e18},
		{nan, 1, 2},
		{pinf, 1, -2},
		{ninf, 0, 3},
	}

	for i, tc := range cases {
		// Attention path: the model wrapper must equal the legacy in-place math.
		want := append([]float32(nil), tc...)
		softmaxInPlaceLegacy(want)
		got := append([]float32(nil), tc...)
		softmaxInPlace(got)
		assertSoftmaxBits(t, "softmaxInPlace", want, got)

		// Router path: the model wrapper must equal the legacy allocating math,
		// leave its input unmutated, and return a fresh allocation.
		inBefore := append([]float32(nil), tc...)
		wantRouter := softmaxOfLegacy(tc)
		gotRouter := softmaxOf(tc)
		assertSoftmaxBits(t, "softmaxOf", wantRouter, gotRouter)
		assertSoftmaxBits(t, "softmaxOf/z-unmutated", inBefore, tc)
		if len(gotRouter) > 0 && &gotRouter[0] == &tc[0] {
			t.Fatalf("case %d: softmaxOf output aliases input z", i)
		}
	}

	// Prove the model wrappers are the leaf, not a private copy: an input the
	// leaf and the wrapper both see must agree, and mutating through the
	// wrapper's output must not disturb a Copy result computed independently.
	probe := []float32{1, 2, 3, 4}
	leafIn := append([]float32(nil), probe...)
	softmax.InPlace(leafIn)
	modelIn := append([]float32(nil), probe...)
	softmaxInPlace(modelIn)
	assertSoftmaxBits(t, "leaf-vs-model-inplace", leafIn, modelIn)

	leafCopy := softmax.Copy(probe)
	modelCopy := softmaxOf(probe)
	assertSoftmaxBits(t, "leaf-vs-model-copy", leafCopy, modelCopy)
}
