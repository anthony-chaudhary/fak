package model

func executeSwiGLU(x, WAct, WUp, WDown []float32, inDim, interDim int) []float32 {
	inter := make([]float32, interDim)
	for i := 0; i < interDim; i++ {
		var gSum, uSum float32
		rowOff := i * inDim
		for j := 0; j < inDim; j++ {
			xj := x[j]
			gSum += WAct[rowOff+j] * xj
			uSum += WUp[rowOff+j] * xj
		}
		inter[i] = silu(gSum) * uSum
	}

	out := make([]float32, inDim)
	for i := 0; i < inDim; i++ {
		var sum float32
		rowOff := i * interDim
		for j := 0; j < interDim; j++ {
			sum += WDown[rowOff+j] * inter[j]
		}
		out[i] = sum
	}
	return out
}

// GLM5NextDenseMLPParams holds weights for dense SwiGLU (layers 0..2).
type GLM5NextDenseMLPParams struct {
	InDim    int
	InterDim int
	WAct     []float32
	WUp      []float32
	WDown    []float32
}

// ExecuteGLM5NextDenseMLP executes the dense SwiGLU MLP for layers 0..2.
func ExecuteGLM5NextDenseMLP(x []float32, params GLM5NextDenseMLPParams) []float32 {
	if len(params.WAct) == 0 || len(params.WUp) == 0 || len(params.WDown) == 0 {
		out := make([]float32, len(x))
		copy(out, x)
		return out
	}
	return executeSwiGLU(x, params.WAct, params.WUp, params.WDown, params.InDim, params.InterDim)
}

// GLM5NextExpertWeight holds weights for a single expert (intermediate = 2048).
type GLM5NextExpertWeight struct {
	WAct  []float32
	WUp   []float32
	WDown []float32
}

// GLM5NextMoEMLPParams holds shared expert and routed experts for layers 3..44.
type GLM5NextMoEMLPParams struct {
	InDim         int
	MoEInterDim   int
	SharedExpert  GLM5NextExpertWeight
	RoutedExperts []GLM5NextExpertWeight
}

// ExecuteGLM5NextSparseMoE evaluates the shared expert and top-8 routed experts:
// out = SharedExpert(x) + sum_{k in top8} w_k * RoutedExpert_k(x).
func ExecuteGLM5NextSparseMoE(
	x []float32,
	route GLM5NextMoERouteResult,
	params GLM5NextMoEMLPParams,
) []float32 {
	inDim := params.InDim
	interDim := params.MoEInterDim

	out := make([]float32, inDim)

	// 1. Shared expert
	if len(params.SharedExpert.WAct) > 0 {
		sharedOut := executeSwiGLU(x, params.SharedExpert.WAct, params.SharedExpert.WUp, params.SharedExpert.WDown, inDim, interDim)
		for i := 0; i < inDim; i++ {
			out[i] += sharedOut[i]
		}
	}

	// 2. Routed experts
	for i, expIdx := range route.ExpertIndices {
		if expIdx < 0 || expIdx >= len(params.RoutedExperts) {
			continue
		}
		weight := route.Weights[i]
		if weight == 0 {
			continue
		}
		exp := params.RoutedExperts[expIdx]
		if len(exp.WAct) == 0 {
			continue
		}
		expOut := executeSwiGLU(x, exp.WAct, exp.WUp, exp.WDown, inDim, interDim)
		for j := 0; j < inDim; j++ {
			out[j] += weight * expOut[j]
		}
	}

	return out
}
