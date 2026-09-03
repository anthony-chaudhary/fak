package model

// GLM5NextKDAParams holds the projection and filter weights for one KDA layer.
type GLM5NextKDAParams struct {
	ConvQ      *GLM5NextKDAConvFilter
	ConvK      *GLM5NextKDAConvFilter
	ConvV      *GLM5NextKDAConvFilter
	BaseDecay  []float32
	Wout       []float32
	HiddenSize int
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
	qConv := params.ConvQ.Step(qRaw, st.ConvQ)
	kConv := params.ConvK.Step(kRaw, st.ConvK)
	vConv := params.ConvV.Step(vRaw, st.ConvV)

	alpha := ComputeGLM5NextKDADecay(decayLogits, params.BaseDecay)
	beta := ComputeGLM5NextKDABeta(decayLogits)

	attnOut := StepGLM5NextKDALayer(st, qConv, kConv, vConv, alpha, beta)

	return ApplyGLM5NextKDAOutputModulationAndProj(
		attnOut, modLogits, params.Wout,
		st.NumHeads, st.HeadDim, params.HiddenSize, eps,
	)
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
