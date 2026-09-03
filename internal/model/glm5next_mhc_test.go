package model

import (
	"math"
	"testing"
)

func TestApplyGLM5NextMHC(t *testing.T) {
	const hiddenSize = 8
	const hcMult = 4
	const expandedDim = hcMult * hiddenSize

	x := make([]float32, hiddenSize)
	for i := range x {
		x[i] = 1.0
	}

	// 1. Zero weights passthrough test
	emptyParams := GLM5NextMHCParams{HiddenSize: hiddenSize, HCMult: hcMult}
	outPassthrough := ApplyGLM5NextMHC(x, emptyParams, 1e-6)
	for i := range x {
		if outPassthrough[i] != x[i] {
			t.Fatalf("passthrough mismatch at %d: got %g, want %g", i, outPassthrough[i], x[i])
		}
	}

	// 2. Huge expansion manifold constraint test
	WUp := make([]float32, expandedDim*hiddenSize)
	for i := range WUp {
		WUp[i] = 100.0 // produces huge activations in z
	}
	WDown := make([]float32, hiddenSize*expandedDim)
	for i := 0; i < hiddenSize; i++ {
		WDown[i*expandedDim+i] = 1.0
	}

	params := GLM5NextMHCParams{
		HiddenSize:    hiddenSize,
		HCMult:        hcMult,
		WUp:           WUp,
		WDown:         WDown,
		ResidualScale: 1.0,
	}

	out := ApplyGLM5NextMHC(x, params, 1e-6)
	if len(out) != hiddenSize {
		t.Fatalf("len(out) = %d, want %d", len(out), hiddenSize)
	}

	// Since manifold sphere bounds each group norm to <= 1.0,
	// the contribution added to x[i] must be strictly bounded and not explode to 1000s
	for i := 0; i < hiddenSize; i++ {
		added := out[i] - x[i]
		if math.Abs(float64(added)) > 10.0 {
			t.Fatalf("mHC manifold constraint failed to bound contribution at %d: added=%g (out=%g)", i, added, out[i])
		}
	}
}
