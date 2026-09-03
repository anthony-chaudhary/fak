package model

import (
	"math"
)

// kdaRMSNorm normalizes slice x in place using RMSNorm: x_i = x_i / sqrt(mean(x^2) + eps).
func kdaRMSNorm(x []float32, eps float32) {
	if len(x) == 0 {
		return
	}
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	mean := sum / float64(len(x))
	scale := float32(1.0 / math.Sqrt(mean+float64(eps)))
	for i := range x {
		x[i] *= scale
	}
}

// ApplyGLM5NextKDAOutputModulationAndProj processes the concatenated linear attention outputs:
// 1. Applies per-head RMSNorm across each head's Dv elements.
// 2. Multiplies by SiLU(mod) element-wise.
// 3. Projects down through Wout (shape hiddenSize x (numHeads*Dv)) to produce residual vector.
func ApplyGLM5NextKDAOutputModulationAndProj(
	attnOut []float32, // [numHeads * Dv]
	modLogits []float32, // [numHeads * Dv]
	Wout []float32, // [hiddenSize * (numHeads * Dv)]
	numHeads, headDim, hiddenSize int,
	eps float32,
) []float32 {
	featureDim := numHeads * headDim
	gated := make([]float32, featureDim)

	// 1. Per-head RMSNorm and SiLU modulation
	for h := 0; h < numHeads; h++ {
		headSlice := make([]float32, headDim)
		copy(headSlice, attnOut[h*headDim:(h+1)*headDim])
		kdaRMSNorm(headSlice, eps)

		for j := 0; j < headDim; j++ {
			idx := h*headDim + j
			gVal := float32(1.0)
			if idx < len(modLogits) {
				gVal = silu(modLogits[idx])
			}
			gated[idx] = headSlice[j] * gVal
		}
	}

	// 2. Projection: y = Wout * gated
	out := make([]float32, hiddenSize)
	if len(Wout) != hiddenSize*featureDim {
		limit := min(hiddenSize, featureDim)
		copy(out[:limit], gated[:limit])
		return out
	}

	for i := 0; i < hiddenSize; i++ {
		var sum float32
		rowOff := i * featureDim
		for j := 0; j < featureDim; j++ {
			sum += Wout[rowOff+j] * gated[j]
		}
		out[i] = sum
	}

	return out
}
