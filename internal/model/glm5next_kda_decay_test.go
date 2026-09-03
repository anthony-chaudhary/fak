package model

import (
	"testing"
)

func TestComputeGLM5NextKDADecay(t *testing.T) {
	const numHeads = 64
	logits := make([]float32, numHeads)
	baseDecay := make([]float32, numHeads)

	// Vary logits from large negative to large positive
	for h := 0; h < numHeads; h++ {
		logits[h] = float32(h - 32) // range [-32, 31]
		baseDecay[h] = 0.5
	}

	alpha := ComputeGLM5NextKDADecay(logits, baseDecay)

	if len(alpha) != numHeads {
		t.Fatalf("len(alpha) = %d, want %d", len(alpha), numHeads)
	}

	// 1. All values must be in (0, 1]
	for h, a := range alpha {
		if a <= 0 || a > 1.0 {
			t.Fatalf("alpha[%d] = %g out of bounds (0, 1]", h, a)
		}
	}

	// 2. Monotonicity: larger logits (higher forget intensity) -> smaller retention alpha
	for h := 1; h < numHeads; h++ {
		if alpha[h] > alpha[h-1] {
			t.Fatalf("alpha is not monotonically decreasing: alpha[%d]=%g > alpha[%d]=%g",
				h, alpha[h], h-1, alpha[h-1])
		}
	}

	// 3. Extreme values must not NaN or Inf
	extremes := []float32{-1000.0, -100.0, 0.0, 100.0, 1000.0}
	alphaExt := ComputeGLM5NextKDADecay(extremes, nil)
	for i, a := range alphaExt {
		if a <= 0 || a > 1.0 {
			t.Fatalf("extreme input %g produced invalid alpha %g", extremes[i], a)
		}
	}
}

func TestComputeGLM5NextKDABeta(t *testing.T) {
	const numHeads = 64
	logits := make([]float32, numHeads)
	for h := 0; h < numHeads; h++ {
		logits[h] = float32(h - 32)
	}

	beta := ComputeGLM5NextKDABeta(logits)
	if len(beta) != numHeads {
		t.Fatalf("len(beta) = %d, want %d", len(beta), numHeads)
	}

	for h, b := range beta {
		if b < 0 || b > 1.0 {
			t.Fatalf("beta[%d] = %g out of bounds [0, 1]", h, b)
		}
	}

	// Monotonicity: larger logits -> larger beta
	for h := 1; h < numHeads; h++ {
		if beta[h] < beta[h-1] {
			t.Fatalf("beta is not monotonically increasing: beta[%d]=%g < beta[%d]=%g",
				h, beta[h], h-1, beta[h-1])
		}
	}
}
