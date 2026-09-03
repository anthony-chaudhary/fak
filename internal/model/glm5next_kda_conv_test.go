package model

import (
	"math"
	"math/rand"
	"testing"
)

func TestGLM5NextKDAConvPrefillMatchesDecode(t *testing.T) {
	const dim = 16
	const kSize = 4
	const T = 8

	filter := NewGLM5NextKDAConvFilter(dim, kSize)
	rng := rand.New(rand.NewSource(42))
	for i := range filter.Weight {
		filter.Weight[i] = rng.Float32()*2.0 - 1.0
	}

	// Generate input sequence
	seq := make([]float32, T*dim)
	for i := range seq {
		seq[i] = rng.Float32()*2.0 - 1.0
	}

	// 1. Run via Prefill
	bufPrefill := make([]float32, (kSize-1)*dim)
	prefillOut := filter.Prefill(seq, T, bufPrefill)

	// 2. Run via Stepwise decode token-by-token
	bufDecode := make([]float32, (kSize-1)*dim)
	decodeOut := make([]float32, T*dim)
	for step := 0; step < T; step++ {
		tokenX := seq[step*dim : (step+1)*dim]
		tokenOut := filter.Step(tokenX, bufDecode)
		copy(decodeOut[step*dim:(step+1)*dim], tokenOut)
	}

	// Verify prefill matches decode bit-for-bit
	for i := range prefillOut {
		if math.Float32bits(prefillOut[i]) != math.Float32bits(decodeOut[i]) {
			t.Fatalf("mismatch at output index %d: prefill=%g (%08x) decode=%g (%08x)",
				i, prefillOut[i], math.Float32bits(prefillOut[i]), decodeOut[i], math.Float32bits(decodeOut[i]))
		}
	}

	// Verify history buffers match bit-for-bit
	for i := range bufPrefill {
		if math.Float32bits(bufPrefill[i]) != math.Float32bits(bufDecode[i]) {
			t.Fatalf("buffer mismatch at index %d: prefill=%g decode=%g", i, bufPrefill[i], bufDecode[i])
		}
	}
}

func TestGLM5NextKDAConvCausality(t *testing.T) {
	const dim = 8
	const kSize = 4
	const T = 5

	filter := NewGLM5NextKDAConvFilter(dim, kSize)
	for i := range filter.Weight {
		filter.Weight[i] = 0.5
	}

	seq1 := make([]float32, T*dim)
	seq2 := make([]float32, T*dim)
	for i := 0; i < 3*dim; i++ {
		seq1[i] = 1.0
		seq2[i] = 1.0
	}
	// At t=3 and t=4, seq2 differs
	for i := 3 * dim; i < T*dim; i++ {
		seq1[i] = 1.0
		seq2[i] = 99.0
	}

	buf1 := make([]float32, (kSize-1)*dim)
	buf2 := make([]float32, (kSize-1)*dim)
	out1 := filter.Prefill(seq1, T, buf1)
	out2 := filter.Prefill(seq2, T, buf2)

	// Outputs for steps 0, 1, 2 must be bit-identical despite future tokens differing
	for i := 0; i < 3*dim; i++ {
		if math.Float32bits(out1[i]) != math.Float32bits(out2[i]) {
			t.Fatalf("causality violation at index %d (t < 3): out1=%g out2=%g", i, out1[i], out2[i])
		}
	}
}
