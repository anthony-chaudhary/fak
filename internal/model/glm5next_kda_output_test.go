package model

import (
	"math"
	"testing"
)

func TestApplyGLM5NextKDAOutputModulationAndProj(t *testing.T) {
	const numHeads = 4
	const headDim = 8
	const featureDim = numHeads * headDim
	const hiddenSize = 16

	attnOut := make([]float32, featureDim)
	for i := range attnOut {
		attnOut[i] = float32(i + 1)
	}

	modLogits := make([]float32, featureDim)
	for i := range modLogits {
		modLogits[i] = 1.0 // SiLU(1.0) approx 0.731
	}

	// Identity-like projection for first hiddenSize features
	Wout := make([]float32, hiddenSize*featureDim)
	for i := 0; i < hiddenSize; i++ {
		Wout[i*featureDim+i] = 1.0
	}

	out := ApplyGLM5NextKDAOutputModulationAndProj(attnOut, modLogits, Wout, numHeads, headDim, hiddenSize, 1e-6)
	if len(out) != hiddenSize {
		t.Fatalf("len(out) = %d, want %d", len(out), hiddenSize)
	}

	// 1. Scale invariance test: scale attnOut by 10x
	attnOutScaled := make([]float32, featureDim)
	for i := range attnOutScaled {
		attnOutScaled[i] = attnOut[i] * 10.0
	}
	outScaled := ApplyGLM5NextKDAOutputModulationAndProj(attnOutScaled, modLogits, Wout, numHeads, headDim, hiddenSize, 1e-6)
	for i := 0; i < hiddenSize; i++ {
		if math.Abs(float64(out[i]-outScaled[i])) > 1e-5 {
			t.Fatalf("RMSNorm scale invariance failed at %d: out=%g, outScaled=%g", i, out[i], outScaled[i])
		}
	}

	// 2. Output modulation zeroing test: large negative logits
	modNegative := make([]float32, featureDim)
	for i := range modNegative {
		modNegative[i] = -100.0 // SiLU(-100) = 0
	}
	outZero := ApplyGLM5NextKDAOutputModulationAndProj(attnOut, modNegative, Wout, numHeads, headDim, hiddenSize, 1e-6)
	for i := 0; i < hiddenSize; i++ {
		if outZero[i] != 0 {
			t.Fatalf("expected 0 output for negative logits, got %g at %d", outZero[i], i)
		}
	}
}
