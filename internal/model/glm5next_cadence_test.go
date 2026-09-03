package model

import (
	"math"
	"testing"
)

func TestRunGLM5Next4LayerCadenceBlock(t *testing.T) {
	const hiddenSize = 16
	const numHeads = 2
	const headDim = 8
	const convWindow = 4

	state := NewGLM5Next4LayerOracleState(numHeads, headDim, convWindow)
	if len(state.KDAStates) != 3 {
		t.Fatalf("expected 3 KDA states for layers 0..2, got %d", len(state.KDAStates))
	}

	x := make([]float32, hiddenSize)
	for i := range x {
		x[i] = 1.0
	}

	// First token
	out1 := RunGLM5Next4LayerCadenceBlock(x, state, hiddenSize)
	if len(out1) != hiddenSize {
		t.Fatalf("len(out1) = %d, want %d", len(out1), hiddenSize)
	}
	if state.TotalTokens != 1 {
		t.Fatalf("TotalTokens = %d, want 1", state.TotalTokens)
	}

	for i, v := range out1 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v == 0 {
			t.Fatalf("out1[%d] = %g, expected finite non-zero", i, v)
		}
	}

	// Second token
	out2 := RunGLM5Next4LayerCadenceBlock(x, state, hiddenSize)
	if state.TotalTokens != 2 {
		t.Fatalf("TotalTokens = %d, want 2", state.TotalTokens)
	}
	for i, v := range out2 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v == 0 {
			t.Fatalf("out2[%d] = %g, expected finite non-zero", i, v)
		}
	}
}
