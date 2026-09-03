package model

import (
	"math"
	"testing"
)

func TestExecuteGLM5NextDenseMLP(t *testing.T) {
	const inDim = 4
	const interDim = 8

	x := []float32{1.0, 0.5, -0.5, 2.0}

	WAct := make([]float32, interDim*inDim)
	WUp := make([]float32, interDim*inDim)
	WDown := make([]float32, inDim*interDim)

	for i := range WAct {
		WAct[i] = 0.1
		WUp[i] = 0.2
	}
	for i := range WDown {
		WDown[i] = 0.05
	}

	params := GLM5NextDenseMLPParams{
		InDim:    inDim,
		InterDim: interDim,
		WAct:     WAct,
		WUp:      WUp,
		WDown:    WDown,
	}

	out := ExecuteGLM5NextDenseMLP(x, params)
	if len(out) != inDim {
		t.Fatalf("len(out) = %d, want %d", len(out), inDim)
	}

	// Verify non-zero output
	for i, v := range out {
		if v == 0 {
			t.Fatalf("dense MLP output[%d] = 0", i)
		}
	}
}

func TestExecuteGLM5NextSparseMoE(t *testing.T) {
	const inDim = 4
	const interDim = 8
	const numExperts = 16

	x := []float32{1.0, 1.0, 1.0, 1.0}

	initWeight := func(scale float32) GLM5NextExpertWeight {
		g := make([]float32, interDim*inDim)
		u := make([]float32, interDim*inDim)
		d := make([]float32, inDim*interDim)
		for i := range g {
			g[i] = scale
			u[i] = scale
		}
		for i := range d {
			d[i] = scale
		}
		return GLM5NextExpertWeight{WAct: g, WUp: u, WDown: d}
	}

	routed := make([]GLM5NextExpertWeight, numExperts)
	for e := 0; e < numExperts; e++ {
		routed[e] = initWeight(float32(e+1) * 0.01)
	}

	params := GLM5NextMoEMLPParams{
		InDim:         inDim,
		MoEInterDim:   interDim,
		SharedExpert:  initWeight(0.05),
		RoutedExperts: routed,
	}

	// Route to expert 0 and expert 1 with weights 0.6 and 0.4
	route := GLM5NextMoERouteResult{
		ExpertIndices: []int{0, 1},
		Weights:       []float32{0.6, 0.4},
	}

	out := ExecuteGLM5NextSparseMoE(x, route, params)
	if len(out) != inDim {
		t.Fatalf("len(out) = %d, want %d", len(out), inDim)
	}

	for i, v := range out {
		if v <= 0 || math.IsNaN(float64(v)) {
			t.Fatalf("sparse MoE output[%d] = %g, expected positive finite", i, v)
		}
	}
}
