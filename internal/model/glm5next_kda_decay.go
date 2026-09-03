package model

import (
	"math"
)

func kdaSoftplus(x float32) float32 {
	if x > 20.0 {
		return x
	}
	if x < -20.0 {
		return float32(math.Exp(float64(x)))
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}

func kdaSigmoid(x float32) float32 {
	if x > 20.0 {
		return 1.0
	}
	if x < -20.0 {
		return 0.0
	}
	return 1.0 / (1.0 + float32(math.Exp(-float64(x))))
}

// ComputeGLM5NextKDADecay computes the per-head retention factor alpha in (0, 1].
// alpha[h] = exp(-softplus(logits[h]) * baseDecay[h]).
// If baseDecay is nil or empty, default baseDecay[h] = 1.0 is used.
func ComputeGLM5NextKDADecay(logits []float32, baseDecay []float32) []float32 {
	n := len(logits)
	alpha := make([]float32, n)
	for i := 0; i < n; i++ {
		b := float32(1.0)
		if i < len(baseDecay) && baseDecay[i] > 0 {
			b = baseDecay[i]
		}
		sp := kdaSoftplus(logits[i])
		val := float32(math.Exp(-float64(sp * b)))
		if val > 1.0 {
			val = 1.0
		}
		if val < 1e-8 {
			val = 1e-8
		}
		alpha[i] = val
	}
	return alpha
}

// ComputeGLM5NextKDABeta computes per-head update scale beta = sigmoid(logits).
func ComputeGLM5NextKDABeta(logits []float32) []float32 {
	n := len(logits)
	beta := make([]float32, n)
	for i := 0; i < n; i++ {
		beta[i] = kdaSigmoid(logits[i])
	}
	return beta
}
