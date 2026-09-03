package model

import (
	"math"
	"testing"
)

func TestExpandDSABlocksToTokens(t *testing.T) {
	blocks := []int{0, 2}
	stride := 4
	totalTokens := 11 // block 0: 0..3, block 2: 8..10 (since 11 is total)

	tokens := ExpandDSABlocksToTokens(blocks, stride, totalTokens)
	expected := []int{0, 1, 2, 3, 8, 9, 10}

	if len(tokens) != len(expected) {
		t.Fatalf("len(tokens) = %d, want %d (%v)", len(tokens), len(expected), tokens)
	}
	for i := range expected {
		if tokens[i] != expected[i] {
			t.Fatalf("tokens[%d] = %d, want %d", i, tokens[i], expected[i])
		}
	}
}

func TestComputeGLM5NextDSASparseMixer(t *testing.T) {
	const numHeads = 2
	const qkNopeHeadDim = 4
	const vHeadDim = 4
	const totalTokens = 4

	// Q for current token
	q := make([]float32, numHeads*qkNopeHeadDim)
	for i := range q {
		q[i] = 1.0
	}

	allK := make([]float32, totalTokens*numHeads*qkNopeHeadDim)
	allV := make([]float32, totalTokens*numHeads*vHeadDim)

	// Set tokens: token 0..2 have specific values
	for tok := 0; tok < totalTokens; tok++ {
		for h := 0; h < numHeads; h++ {
			kOff := tok*numHeads*qkNopeHeadDim + h*qkNopeHeadDim
			vOff := tok*numHeads*vHeadDim + h*vHeadDim
			for d := 0; d < qkNopeHeadDim; d++ {
				allK[kOff+d] = float32(tok + 1)
			}
			for d := 0; d < vHeadDim; d++ {
				allV[vOff+d] = float32((tok + 1) * 10)
			}
		}
	}

	// Attend to token 0 and 1 at currentToken = 2
	selected := []int{0, 1, 3} // token 3 is future (> 2), must be causally ignored

	out := ComputeGLM5NextDSASparseMixer(q, allK, allV, selected, 2, numHeads, qkNopeHeadDim, vHeadDim)

	if len(out) != numHeads*vHeadDim {
		t.Fatalf("len(out) = %d, want %d", len(out), numHeads*vHeadDim)
	}

	// Output must be a convex combination of token 0 values (10) and token 1 values (20)
	// Because token 3 is excluded causally
	for h := 0; h < numHeads; h++ {
		val := out[h*vHeadDim]
		if val < 10.0 || val > 20.0 {
			t.Fatalf("head %d output %g not in expected range [10, 20]", h, val)
		}
		// Since token 1 has higher dot product (2 vs 1), weight for token 1 is higher -> val > 15
		if val <= 15.0 {
			t.Fatalf("head %d output %g should be skewed towards token 1 (>15)", h, val)
		}
	}

	// Empty selection produces zeroes
	outEmpty := ComputeGLM5NextDSASparseMixer(q, allK, allV, nil, 2, numHeads, qkNopeHeadDim, vHeadDim)
	for _, v := range outEmpty {
		if v != 0 {
			t.Fatalf("empty selection produced non-zero output: %g", v)
		}
	}
	_ = math.Abs
}
