package model

import (
	"math"
	"testing"
)

func TestStepGLM5NextKDAHeadValueRecall(t *testing.T) {
	const Dk = 4
	const Dv = 4
	S := make([]float32, Dk*Dv)

	// Normalized key
	k := []float32{1.0, 0.0, 0.0, 0.0}
	v := []float32{0.5, -0.2, 0.8, 1.2}
	q := []float32{1.0, 0.0, 0.0, 0.0}

	// Insert into state with alpha=1.0, beta=1.0
	out := StepGLM5NextKDAHead(S, Dk, Dv, q, k, v, 1.0, 1.0)

	// Since q == k and k has norm 1, out should equal v
	for j := 0; j < Dv; j++ {
		if math.Abs(float64(out[j]-v[j])) > 1e-6 {
			t.Fatalf("out[%d] = %g, want %g", j, out[j], v[j])
		}
	}
}

func TestStepGLM5NextKDAHeadDecay(t *testing.T) {
	const Dk = 2
	const Dv = 2
	S := []float32{1.0, 2.0, 3.0, 4.0}

	k := []float32{0.0, 0.0}
	v := []float32{0.0, 0.0}
	q := []float32{1.0, 0.0}

	// Decay with alpha = 0.5, beta = 0.0 (no new content)
	out := StepGLM5NextKDAHead(S, Dk, Dv, q, k, v, 0.5, 0.0)

	// Initial row 0 was [1.0, 2.0], after 0.5 decay should be [0.5, 1.0]
	if math.Abs(float64(out[0]-0.5)) > 1e-6 || math.Abs(float64(out[1]-1.0)) > 1e-6 {
		t.Fatalf("decay output mismatch: got %v, want [0.5, 1.0]", out)
	}

	// Decay with alpha = 0.0 wipes state
	StepGLM5NextKDAHead(S, Dk, Dv, q, k, v, 0.0, 0.0)
	for idx, val := range S {
		if val != 0 {
			t.Fatalf("S[%d] = %g after zero alpha, want 0", idx, val)
		}
	}
}

func TestStepGLM5NextKDALayerMultiHead(t *testing.T) {
	const numHeads = 64
	const Dk = 128
	const Dv = 128

	st := NewGLM5NextKDALayerState(numHeads, Dk, 4)

	Q := make([]float32, numHeads*Dk)
	K := make([]float32, numHeads*Dk)
	V := make([]float32, numHeads*Dv)
	alpha := make([]float32, numHeads)
	beta := make([]float32, numHeads)

	for h := 0; h < numHeads; h++ {
		alpha[h] = 0.95
		beta[h] = 1.0
		// Set unit vector for head h at position 0
		Q[h*Dk] = 1.0
		K[h*Dk] = 1.0
		V[h*Dv] = float32(h + 1)
	}

	out := StepGLM5NextKDALayer(st, Q, K, V, alpha, beta)
	if len(out) != numHeads*Dv {
		t.Fatalf("len(out) = %d, want %d", len(out), numHeads*Dv)
	}

	// Check each head output at position 0
	for h := 0; h < numHeads; h++ {
		expected := float32(h + 1)
		got := out[h*Dv]
		if math.Abs(float64(got-expected)) > 1e-5 {
			t.Fatalf("head %d output mismatch: got %g, want %g", h, got, expected)
		}
	}
}
