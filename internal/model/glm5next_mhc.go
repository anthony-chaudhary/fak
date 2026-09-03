package model

import (
	"math"
)

// GLM5NextMHCParams holds the projection weights for the Manifold-constrained Hidden Channel (mHC) sublayer.
type GLM5NextMHCParams struct {
	HiddenSize    int
	HCMult        int
	WUp           []float32
	WDown         []float32
	ResidualScale float32
}

// ApplyGLM5NextMHC executes the manifold-constrained hidden channel transformation:
//  1. Up-projection: z = WUp * x (shape HCMult * HiddenSize)
//  2. Manifold projection: partitions z into HCMult groups of size HiddenSize;
//     for each group g, bounds L2 norm <= 1.0 (manifold sphere constraint).
//  3. Down-projection: y = WDown * z_manifold (shape HiddenSize)
//  4. Residual fusion: out = x + ResidualScale * y
func ApplyGLM5NextMHC(x []float32, params GLM5NextMHCParams, eps float32) []float32 {
	hiddenSize := params.HiddenSize
	hcMult := params.HCMult
	if hcMult <= 0 {
		hcMult = 4
	}
	expandedDim := hcMult * hiddenSize

	out := make([]float32, hiddenSize)
	copy(out, x)

	if len(params.WUp) != expandedDim*hiddenSize || len(params.WDown) != hiddenSize*expandedDim {
		return out
	}

	// 1. Up-projection: z = WUp * x
	z := make([]float32, expandedDim)
	for i := 0; i < expandedDim; i++ {
		var sum float32
		rowOff := i * hiddenSize
		for j := 0; j < hiddenSize; j++ {
			sum += params.WUp[rowOff+j] * x[j]
		}
		z[i] = sum
	}

	// 2. Manifold constraint per channel group of size hiddenSize
	for g := 0; g < hcMult; g++ {
		groupSlice := z[g*hiddenSize : (g+1)*hiddenSize]
		var normSq float64
		for _, v := range groupSlice {
			normSq += float64(v) * float64(v)
		}
		radius := math.Sqrt(normSq + float64(eps))
		if radius > 1.0 {
			factor := float32(1.0 / radius)
			for j := range groupSlice {
				groupSlice[j] *= factor
			}
		}
	}

	// 3. Down-projection: y = WDown * z
	scale := params.ResidualScale
	if scale == 0 {
		scale = 1.0
	}

	for i := 0; i < hiddenSize; i++ {
		var sum float32
		rowOff := i * expandedDim
		for j := 0; j < expandedDim; j++ {
			sum += params.WDown[rowOff+j] * z[j]
		}
		out[i] += scale * sum
	}

	return out
}
