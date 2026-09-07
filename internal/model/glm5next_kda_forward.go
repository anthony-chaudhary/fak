package model

// GLM5NextKDAParams holds the projection and filter weights for one KDA layer.
// Adapted from HuggingFace Transformers (transformers/models/glm5_next) under Apache-2.0 license.
type GLM5NextKDAParams struct {
	ConvQ      *GLM5NextKDAConvFilter
	ConvK      *GLM5NextKDAConvFilter
	ConvV      *GLM5NextKDAConvFilter
	BaseDecay  []float32
	Wout       []float32
	HiddenSize int
	Wq         []float32 // [featureDim * hiddenSize]
	Wk         []float32 // [featureDim * hiddenSize]
	Wv         []float32 // [featureDim * hiddenSize]
	Wdecay     []float32 // [numHeads * hiddenSize]
	Wmod       []float32 // [featureDim * hiddenSize]
	RMSNormEps float32
}

// ForwardGLM5NextKDADecode runs a single-token step through KDA layer:
// 1. Convolves Q, K, V
// 2. Computes decay alpha and learning rate beta
// 3. Updates recurrent state and computes attention output
// 4. Modulates and projects output to hidden dimension
func ForwardGLM5NextKDADecode(
	st *GLM5NextKDALayerState,
	params GLM5NextKDAParams,
	qRaw, kRaw, vRaw []float32,
	decayLogits, modLogits []float32,
	eps float32,
) []float32 {
	var qConv, kConv, vConv []float32
	if params.ConvQ != nil {
		qConv = params.ConvQ.Step(qRaw, st.ConvQ)
	} else {
		qConv = qRaw
	}
	if params.ConvK != nil {
		kConv = params.ConvK.Step(kRaw, st.ConvK)
	} else {
		kConv = kRaw
	}
	if params.ConvV != nil {
		vConv = params.ConvV.Step(vRaw, st.ConvV)
	} else {
		vConv = vRaw
	}

	alpha := ComputeGLM5NextKDADecay(decayLogits, params.BaseDecay)
	beta := ComputeGLM5NextKDABeta(decayLogits)

	attnOut := StepGLM5NextKDALayer(st, qConv, kConv, vConv, alpha, beta)

	return ApplyGLM5NextKDAOutputModulationAndProj(
		attnOut, modLogits, params.Wout,
		st.NumHeads, st.HeadDim, params.HiddenSize, eps,
	)
}

func kdaMatVecMul(W, x []float32, outDim, inDim int) []float32 {
	out := make([]float32, outDim)
	for i := 0; i < outDim; i++ {
		var sum float32
		rowOff := i * inDim
		for j := 0; j < inDim; j++ {
			sum += W[rowOff+j] * x[j]
		}
		out[i] = sum
	}
	return out
}

// ForwardGLM5NextKDALayerStep executes a complete single-token step for a KDA layer:
// 1. If in-projections (Wq, Wk, Wv) are present, applies RMSNorm to x and projects to Q, K, V, decayLogits, modLogits.
// 2. If in-projections are empty, handles input appropriately (using x directly or sliced).
// 3. Runs ForwardGLM5NextKDADecode and returns projected output [hiddenSize].
func ForwardGLM5NextKDALayerStep(
	st *GLM5NextKDALayerState,
	params GLM5NextKDAParams,
	x []float32,
	eps float32,
) []float32 {
	featureDim := st.NumHeads * st.HeadDim
	hiddenSize := params.HiddenSize
	if hiddenSize <= 0 {
		hiddenSize = len(x)
		if hiddenSize <= 0 {
			hiddenSize = featureDim
		}
		params.HiddenSize = hiddenSize
	}

	hasInProj := len(params.Wq) == featureDim*hiddenSize &&
		len(params.Wk) == featureDim*hiddenSize &&
		len(params.Wv) == featureDim*hiddenSize

	var qRaw, kRaw, vRaw, decayLogits, modLogits []float32

	if hasInProj {
		normEps := params.RMSNormEps
		if normEps <= 0 {
			normEps = eps
		}
		if normEps <= 0 {
			normEps = 1e-6
		}
		normX := append([]float32(nil), x...)
		kdaRMSNorm(normX, normEps)

		qRaw = kdaMatVecMul(params.Wq, normX, featureDim, hiddenSize)
		kRaw = kdaMatVecMul(params.Wk, normX, featureDim, hiddenSize)
		vRaw = kdaMatVecMul(params.Wv, normX, featureDim, hiddenSize)

		if len(params.Wdecay) == st.NumHeads*hiddenSize {
			decayLogits = kdaMatVecMul(params.Wdecay, normX, st.NumHeads, hiddenSize)
		}
		if len(params.Wmod) == featureDim*hiddenSize {
			modLogits = kdaMatVecMul(params.Wmod, normX, featureDim, hiddenSize)
		}
	} else {
		qRaw = make([]float32, featureDim)
		kRaw = make([]float32, featureDim)
		vRaw = make([]float32, featureDim)
		copy(qRaw, x)
		copy(kRaw, x)
		copy(vRaw, x)

		if len(x) >= st.NumHeads {
			decayLogits = append([]float32(nil), x[:st.NumHeads]...)
		}
		if len(x) >= featureDim {
			modLogits = append([]float32(nil), x[:featureDim]...)
		}
	}

	return ForwardGLM5NextKDADecode(st, params, qRaw, kRaw, vRaw, decayLogits, modLogits, eps)
}

// ForwardGLM5NextKDALayerPrefillSeq processes a sequence of T tokens through ForwardGLM5NextKDALayerStep,
// updating st and returning concatenated outputs [T * hiddenSize].
func ForwardGLM5NextKDALayerPrefillSeq(
	st *GLM5NextKDALayerState,
	params GLM5NextKDAParams,
	xSeq []float32,
	T int,
	eps float32,
) []float32 {
	hDim := params.HiddenSize
	if hDim <= 0 && T > 0 {
		hDim = len(xSeq) / T
	}
	out := make([]float32, T*hDim)
	for t := 0; t < T; t++ {
		xt := xSeq[t*hDim : (t+1)*hDim]
		tokOut := ForwardGLM5NextKDALayerStep(st, params, xt, eps)
		copy(out[t*hDim:(t+1)*hDim], tokOut)
	}
	return out
}

// ForwardGLM5NextKDAPrefill processes a sequence of T tokens through the KDA layer,
// updating st and returning output sequence [T * hiddenSize].
func ForwardGLM5NextKDAPrefill(
	st *GLM5NextKDALayerState,
	params GLM5NextKDAParams,
	qSeq, kSeq, vSeq []float32, // each [T * (NumHeads*HeadDim)]
	decayLogitsSeq, modLogitsSeq []float32,
	T int,
	eps float32,
) []float32 {
	dim := st.NumHeads * st.HeadDim
	hDim := params.HiddenSize
	out := make([]float32, T*hDim)

	for t := 0; t < T; t++ {
		qt := qSeq[t*dim : (t+1)*dim]
		kt := kSeq[t*dim : (t+1)*dim]
		vt := vSeq[t*dim : (t+1)*dim]

		var decayLogits, modLogits []float32
		if len(decayLogitsSeq) >= (t+1)*st.NumHeads {
			decayLogits = decayLogitsSeq[t*st.NumHeads : (t+1)*st.NumHeads]
		}
		if len(modLogitsSeq) >= (t+1)*dim {
			modLogits = modLogitsSeq[t*dim : (t+1)*dim]
		}

		tokOut := ForwardGLM5NextKDADecode(st, params, qt, kt, vt, decayLogits, modLogits, eps)
		copy(out[t*hDim:(t+1)*hDim], tokOut)
	}

	return out
}
