package model

import (
	"math"
	"testing"
)

func TestRouteGLM5NextMoE(t *testing.T) {
	const numExperts = 288
	const topK = 8
	const hiddenSize = 16

	x := make([]float32, hiddenSize)
	for i := range x {
		x[i] = 1.0
	}

	WRouter := make([]float32, numExperts*hiddenSize)
	// Assign experts different values so expert 42, 100, 200 have highest logits
	for e := 0; e < numExperts; e++ {
		val := float32(e) * 0.01 // monotonic
		for j := 0; j < hiddenSize; j++ {
			WRouter[e*hiddenSize+j] = val
		}
	}

	res := RouteGLM5NextMoE(x, WRouter, numExperts, topK)

	if len(res.ExpertIndices) != topK || len(res.Weights) != topK {
		t.Fatalf("expected length %d, got indices=%d weights=%d", topK, len(res.ExpertIndices), len(res.Weights))
	}

	// 1. Check that the top-8 highest indexed experts (287, 286, ...) are selected
	for i := 0; i < topK; i++ {
		expectedIdx := numExperts - 1 - i
		if res.ExpertIndices[i] != expectedIdx {
			t.Fatalf("res.ExpertIndices[%d] = %d, want %d", i, res.ExpertIndices[i], expectedIdx)
		}
	}

	// 2. Weights must sum to 1.0
	var sum float32
	for _, w := range res.Weights {
		if w <= 0 || w > 1.0 {
			t.Fatalf("expert weight %g out of bounds (0, 1]", w)
		}
		sum += w
	}
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Fatalf("weights sum = %g, want 1.0", sum)
	}

	// 3. Descending order of weights
	for i := 1; i < topK; i++ {
		if res.Weights[i] > res.Weights[i-1] {
			t.Fatalf("weights not descending: w[%d]=%g > w[%d]=%g", i, res.Weights[i], i-1, res.Weights[i-1])
		}
	}
}
