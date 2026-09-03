package model

import (
	"math"
	"math/rand"
	"testing"
)

func TestForwardGLM5NextKDAPrefillEqualsStepwiseDecode(t *testing.T) {
	const numHeads = 4
	const headDim = 16
	const featureDim = numHeads * headDim
	const hiddenSize = 32
	const convWindow = 4
	const T = 6

	rng := rand.New(rand.NewSource(12345))

	initConv := func() *GLM5NextKDAConvFilter {
		f := NewGLM5NextKDAConvFilter(featureDim, convWindow)
		for i := range f.Weight {
			f.Weight[i] = rng.Float32()*2.0 - 1.0
		}
		return f
	}

	params := GLM5NextKDAParams{
		ConvQ:      initConv(),
		ConvK:      initConv(),
		ConvV:      initConv(),
		BaseDecay:  make([]float32, numHeads),
		Wout:       make([]float32, hiddenSize*featureDim),
		HiddenSize: hiddenSize,
	}
	for h := 0; h < numHeads; h++ {
		params.BaseDecay[h] = 0.8
	}
	for i := range params.Wout {
		params.Wout[i] = rng.Float32()*0.2 - 0.1
	}

	// Generate input sequence
	randSeq := func(size int) []float32 {
		s := make([]float32, size)
		for i := range s {
			s[i] = rng.Float32()*2.0 - 1.0
		}
		return s
	}

	qSeq := randSeq(T * featureDim)
	kSeq := randSeq(T * featureDim)
	vSeq := randSeq(T * featureDim)
	decaySeq := randSeq(T * numHeads)
	modSeq := randSeq(T * featureDim)

	// 1. Prefill pass
	stPrefill := NewGLM5NextKDALayerState(numHeads, headDim, convWindow)
	outPrefill := ForwardGLM5NextKDAPrefill(stPrefill, params, qSeq, kSeq, vSeq, decaySeq, modSeq, T, 1e-6)

	// 2. Decode pass token-by-token
	stDecode := NewGLM5NextKDALayerState(numHeads, headDim, convWindow)
	outDecode := make([]float32, T*hiddenSize)
	for step := 0; step < T; step++ {
		qTok := qSeq[step*featureDim : (step+1)*featureDim]
		kTok := kSeq[step*featureDim : (step+1)*featureDim]
		vTok := vSeq[step*featureDim : (step+1)*featureDim]
		decTok := decaySeq[step*numHeads : (step+1)*numHeads]
		modTok := modSeq[step*featureDim : (step+1)*featureDim]

		tokOut := ForwardGLM5NextKDADecode(stDecode, params, qTok, kTok, vTok, decTok, modTok, 1e-6)
		copy(outDecode[step*hiddenSize:(step+1)*hiddenSize], tokOut)
	}

	// 3. Verify bit-for-bit equality of output hidden states across all positions
	if len(outPrefill) != len(outDecode) {
		t.Fatalf("lengths mismatch: prefill=%d decode=%d", len(outPrefill), len(outDecode))
	}
	for i := range outPrefill {
		if math.Float32bits(outPrefill[i]) != math.Float32bits(outDecode[i]) {
			t.Fatalf("hidden state mismatch at pos %d/%d (step %d): prefill=%g (%08x) decode=%g (%08x)",
				i, len(outPrefill), i/hiddenSize,
				outPrefill[i], math.Float32bits(outPrefill[i]),
				outDecode[i], math.Float32bits(outDecode[i]))
		}
	}

	// 4. Verify bit-for-bit equality of final recurrent state S
	for i := range stPrefill.S {
		if math.Float32bits(stPrefill.S[i]) != math.Float32bits(stDecode.S[i]) {
			t.Fatalf("recurrent matrix S mismatch at %d: prefill=%g decode=%g", i, stPrefill.S[i], stDecode.S[i])
		}
	}

	// 5. Verify bit-for-bit equality of conv buffers
	for i := range stPrefill.ConvQ {
		if math.Float32bits(stPrefill.ConvQ[i]) != math.Float32bits(stDecode.ConvQ[i]) {
			t.Fatalf("ConvQ mismatch at %d: prefill=%g decode=%g", i, stPrefill.ConvQ[i], stDecode.ConvQ[i])
		}
	}
}
