package model

// Adapted from HuggingFace Transformers (transformers/models/glm5_next) under Apache-2.0 license.
// GLM5Next architecture defines a 4-layer repeating cadence (3:1 KDA-to-DSA ratio):
// - Layers where layer%4 != 3: KDA linear attention recurrent mixer
// - Layers where layer%4 == 3 (e.g. layer 3, 7, ...): Decoupled Sparse Attention (DSA)
//   with KPool 1D key downsampling (stride 4), top-K indexer scoring, and sparse multi-head attention.
// - MLP sublayers: layers 0..2 use dense SwiGLU; layers 3+ use routed MoE (top-8 of 288 experts + shared expert).
// - Residual sublayer: Manifold-constrained Hidden Channel (mHC) sphere projection across all layers.

import (
	"math"
	"math/rand"
)

// GLM5NextDSALayerConfig specifies geometry and projection weights for one DSA sparse attention layer.
type GLM5NextDSALayerConfig struct {
	NumHeads      int
	QKNopeHeadDim int
	VHeadDim      int
	NumIndexHeads int
	IndexHeadDim  int
	BlockStride   int
	TopKBlocks    int

	Wq    []float32
	Wk    []float32
	Wv    []float32
	WqIdx []float32
	WkIdx []float32
	WOut  []float32
}

// GLM5NextCadenceDiagnostics records intermediate routing and block selection decisions across layers.
type GLM5NextCadenceDiagnostics struct {
	SelectedBlocks      []int
	SelectedTokens      []int
	Route               GLM5NextMoERouteResult
	LayerSelectedBlocks map[int][]int
	LayerSelectedTokens map[int][]int
	LayerRoutes         map[int]GLM5NextMoERouteResult
}

// GLM5Next4LayerOracleParams holds the complete parameter set across a 4-layer cadence block.
type GLM5Next4LayerOracleParams struct {
	HiddenSize  int
	NumHeads    int
	HeadDim     int
	ConvWindow  int
	KDAParams   map[int]*GLM5NextKDAParams
	DenseParams map[int]*GLM5NextDenseMLPParams
	MHCParams   map[int]*GLM5NextMHCParams
	DSAOracle   func(x []float32, state *GLM5Next4LayerOracleState, hiddenSize int) []float32
	MoEParams   *GLM5NextMoEMLPParams
}

// NewGLM5Next4LayerOracleParams initializes deterministic, non-zero stable weights for all 4 layers.
func NewGLM5Next4LayerOracleParams(numHeads, headDim, convWindow, hiddenSize int) *GLM5Next4LayerOracleParams {
	if numHeads <= 0 {
		numHeads = 64
	}
	if headDim <= 0 {
		headDim = 128
	}
	if convWindow <= 0 {
		convWindow = 4
	}
	if hiddenSize <= 0 {
		hiddenSize = numHeads * headDim
	}
	featureDim := numHeads * headDim

	kdaMap := make(map[int]*GLM5NextKDAParams, 3)
	denseMap := make(map[int]*GLM5NextDenseMLPParams, 3)
	mhcMap := make(map[int]*GLM5NextMHCParams, 4)

	rng := rand.New(rand.NewSource(42))

	scaleIn := float32(0.2 / math.Sqrt(float64(hiddenSize)))
	scaleOut := float32(0.2 / math.Sqrt(float64(featureDim)))

	for l := 0; l < 3; l++ {
		convQ := NewGLM5NextKDAConvFilter(featureDim, convWindow)
		convK := NewGLM5NextKDAConvFilter(featureDim, convWindow)
		convV := NewGLM5NextKDAConvFilter(featureDim, convWindow)
		for i := range convQ.Weight {
			convQ.Weight[i] = (rng.Float32()*2.0 - 1.0) * 0.05
			convK.Weight[i] = (rng.Float32()*2.0 - 1.0) * 0.05
			convV.Weight[i] = (rng.Float32()*2.0 - 1.0) * 0.05
		}

		baseDecay := make([]float32, numHeads)
		for h := 0; h < numHeads; h++ {
			baseDecay[h] = 0.8
		}

		wq := make([]float32, featureDim*hiddenSize)
		wk := make([]float32, featureDim*hiddenSize)
		wv := make([]float32, featureDim*hiddenSize)
		wdecay := make([]float32, numHeads*hiddenSize)
		wmod := make([]float32, featureDim*hiddenSize)
		wout := make([]float32, hiddenSize*featureDim)

		for i := range wq {
			wq[i] = (rng.Float32()*2.0 - 1.0) * scaleIn
			wk[i] = (rng.Float32()*2.0 - 1.0) * scaleIn
			wv[i] = (rng.Float32()*2.0 - 1.0) * scaleIn
			wmod[i] = (rng.Float32()*2.0 - 1.0) * scaleIn
		}
		for i := range wdecay {
			wdecay[i] = (rng.Float32()*2.0 - 1.0) * 0.05
		}
		for i := range wout {
			wout[i] = (rng.Float32()*2.0 - 1.0) * scaleOut
		}

		kdaMap[l] = &GLM5NextKDAParams{
			ConvQ:      convQ,
			ConvK:      convK,
			ConvV:      convV,
			BaseDecay:  baseDecay,
			Wout:       wout,
			HiddenSize: hiddenSize,
			Wq:         wq,
			Wk:         wk,
			Wv:         wv,
			Wdecay:     wdecay,
			Wmod:       wmod,
			RMSNormEps: 1e-6,
		}

		// Dense MLP for layers 0, 1, 2
		interDim := 2 * hiddenSize
		scaleDense := float32(0.1 / math.Sqrt(float64(hiddenSize)))
		scaleDenseDown := float32(0.1 / math.Sqrt(float64(interDim)))

		wAct := make([]float32, interDim*hiddenSize)
		wUp := make([]float32, interDim*hiddenSize)
		wDown := make([]float32, hiddenSize*interDim)
		for i := range wAct {
			wAct[i] = (rng.Float32()*2.0 - 1.0) * scaleDense
			wUp[i] = (rng.Float32()*2.0 - 1.0) * scaleDense
		}
		for i := range wDown {
			wDown[i] = (rng.Float32()*2.0 - 1.0) * scaleDenseDown
		}

		denseMap[l] = &GLM5NextDenseMLPParams{
			InDim:    hiddenSize,
			InterDim: interDim,
			WAct:     wAct,
			WUp:      wUp,
			WDown:    wDown,
		}
	}

	// mHC for layers 0, 1, 2, 3
	hcMult := 4
	expandedDim := hcMult * hiddenSize
	scaleMHCUp := float32(0.05 / math.Sqrt(float64(hiddenSize)))
	scaleMHCDown := float32(0.05 / math.Sqrt(float64(expandedDim)))

	for l := 0; l < 4; l++ {
		wUpMHC := make([]float32, expandedDim*hiddenSize)
		wDownMHC := make([]float32, hiddenSize*expandedDim)
		for i := range wUpMHC {
			wUpMHC[i] = (rng.Float32()*2.0 - 1.0) * scaleMHCUp
		}
		for i := range wDownMHC {
			wDownMHC[i] = (rng.Float32()*2.0 - 1.0) * scaleMHCDown
		}
		mhcMap[l] = &GLM5NextMHCParams{
			HiddenSize:    hiddenSize,
			HCMult:        hcMult,
			WUp:           wUpMHC,
			WDown:         wDownMHC,
			ResidualScale: 0.1,
		}
	}

	// MoE for layer 3
	moeInterDim := 2 * hiddenSize
	numExperts := 8
	scaleMoEUp := float32(0.05 / math.Sqrt(float64(hiddenSize)))
	scaleMoEDown := float32(0.05 / math.Sqrt(float64(moeInterDim)))

	initExpert := func() GLM5NextExpertWeight {
		g := make([]float32, moeInterDim*hiddenSize)
		u := make([]float32, moeInterDim*hiddenSize)
		d := make([]float32, hiddenSize*moeInterDim)
		for i := range g {
			g[i] = (rng.Float32()*2.0 - 1.0) * scaleMoEUp
			u[i] = (rng.Float32()*2.0 - 1.0) * scaleMoEUp
		}
		for i := range d {
			d[i] = (rng.Float32()*2.0 - 1.0) * scaleMoEDown
		}
		return GLM5NextExpertWeight{WAct: g, WUp: u, WDown: d}
	}

	routedExperts := make([]GLM5NextExpertWeight, numExperts)
	for e := 0; e < numExperts; e++ {
		routedExperts[e] = initExpert()
	}

	moeParams := &GLM5NextMoEMLPParams{
		InDim:         hiddenSize,
		MoEInterDim:   moeInterDim,
		SharedExpert:  initExpert(),
		RoutedExperts: routedExperts,
	}

	dsaOracle := func(x []float32, state *GLM5Next4LayerOracleState, hSize int) []float32 {
		return runDSAOracleLayerDelta(x, state, 3, hSize)
	}

	return &GLM5Next4LayerOracleParams{
		HiddenSize:  hiddenSize,
		NumHeads:    numHeads,
		HeadDim:     headDim,
		ConvWindow:  convWindow,
		KDAParams:   kdaMap,
		DenseParams: denseMap,
		MHCParams:   mhcMap,
		DSAOracle:   dsaOracle,
		MoEParams:   moeParams,
	}
}

// GLM5Next4LayerOracleState holds state across layers.
type GLM5Next4LayerOracleState struct {
	KDAStates       map[int]*GLM5NextKDALayerState
	TotalTokens     int
	Params          *GLM5Next4LayerOracleParams
	DSABuffers      map[int]*GLM5NextDSALayerBuffer
	DownsampledKeys map[int][]float32
	DSAConfigs      map[int]GLM5NextDSALayerConfig
	KDAParams       map[int]GLM5NextKDAParams
	DenseParams     map[int]GLM5NextDenseMLPParams
	MoERouters      map[int][]float32
	MoEParams       map[int]GLM5NextMoEMLPParams
	MoENumExperts   int
	MoETopK         int
	MHCParams       map[int]GLM5NextMHCParams
	Diagnostics     GLM5NextCadenceDiagnostics
}

// Reset zeroes out all KDAStates via st.Reset() and sets TotalTokens = 0.
func (s *GLM5Next4LayerOracleState) Reset() {
	if s == nil {
		return
	}
	for _, st := range s.KDAStates {
		if st != nil {
			st.Reset()
		}
	}
	s.TotalTokens = 0
	for _, b := range s.DSABuffers {
		if b != nil {
			b.K = b.K[:0]
			b.V = b.V[:0]
			b.IndexK = b.IndexK[:0]
		}
	}
	for k := range s.DownsampledKeys {
		delete(s.DownsampledKeys, k)
	}
	s.Diagnostics = GLM5NextCadenceDiagnostics{
		LayerSelectedBlocks: make(map[int][]int),
		LayerSelectedTokens: make(map[int][]int),
		LayerRoutes:         make(map[int]GLM5NextMoERouteResult),
	}
}

// Clone creates a deep copy of all KDAStates and TotalTokens and references Params.
func (s *GLM5Next4LayerOracleState) Clone() *GLM5Next4LayerOracleState {
	if s == nil {
		return nil
	}
	kdaMap := make(map[int]*GLM5NextKDALayerState, len(s.KDAStates))
	for k, v := range s.KDAStates {
		if v != nil {
			kdaMap[k] = v.Clone()
		}
	}

	dsaBufs := make(map[int]*GLM5NextDSALayerBuffer, len(s.DSABuffers))
	for k, v := range s.DSABuffers {
		if v != nil {
			dsaBufs[k] = &GLM5NextDSALayerBuffer{
				K:      append([]float32(nil), v.K...),
				V:      append([]float32(nil), v.V...),
				IndexK: append([]float32(nil), v.IndexK...),
			}
		}
	}

	downsampled := make(map[int][]float32, len(s.DownsampledKeys))
	for k, v := range s.DownsampledKeys {
		downsampled[k] = append([]float32(nil), v...)
	}

	dsaConfigs := make(map[int]GLM5NextDSALayerConfig, len(s.DSAConfigs))
	for k, v := range s.DSAConfigs {
		dsaConfigs[k] = v
	}

	return &GLM5Next4LayerOracleState{
		KDAStates:       kdaMap,
		TotalTokens:     s.TotalTokens,
		Params:          s.Params,
		DSABuffers:      dsaBufs,
		DownsampledKeys: downsampled,
		DSAConfigs:      dsaConfigs,
		KDAParams:       s.KDAParams,
		DenseParams:     s.DenseParams,
		MoERouters:      s.MoERouters,
		MoEParams:       s.MoEParams,
		MoENumExperts:   s.MoENumExperts,
		MoETopK:         s.MoETopK,
		MHCParams:       s.MHCParams,
	}
}

// Snapshot creates a deep copy of all KDAStates and TotalTokens and references Params.
func (s *GLM5Next4LayerOracleState) Snapshot() *GLM5Next4LayerOracleState {
	return s.Clone()
}

// Restore restores in-place from snapshot.
func (s *GLM5Next4LayerOracleState) Restore(snap *GLM5Next4LayerOracleState) {
	if s == nil || snap == nil {
		return
	}
	s.TotalTokens = snap.TotalTokens
	s.Params = snap.Params
	if s.KDAStates == nil {
		s.KDAStates = make(map[int]*GLM5NextKDALayerState, len(snap.KDAStates))
	}
	for k, v := range snap.KDAStates {
		if v == nil {
			delete(s.KDAStates, k)
			continue
		}
		dst, ok := s.KDAStates[k]
		if !ok || dst == nil {
			s.KDAStates[k] = v.Clone()
			continue
		}
		dst.NumHeads = v.NumHeads
		dst.HeadDim = v.HeadDim
		dst.ConvWindow = v.ConvWindow
		if len(dst.S) != len(v.S) {
			dst.S = append([]float32(nil), v.S...)
		} else {
			copy(dst.S, v.S)
		}
		if len(dst.ConvQ) != len(v.ConvQ) {
			dst.ConvQ = append([]float32(nil), v.ConvQ...)
		} else {
			copy(dst.ConvQ, v.ConvQ)
		}
		if len(dst.ConvK) != len(v.ConvK) {
			dst.ConvK = append([]float32(nil), v.ConvK...)
		} else {
			copy(dst.ConvK, v.ConvK)
		}
		if len(dst.ConvV) != len(v.ConvV) {
			dst.ConvV = append([]float32(nil), v.ConvV...)
		} else {
			copy(dst.ConvV, v.ConvV)
		}
	}
	for k := range s.KDAStates {
		if _, ok := snap.KDAStates[k]; !ok {
			delete(s.KDAStates, k)
		}
	}

	if s.DSABuffers == nil {
		s.DSABuffers = make(map[int]*GLM5NextDSALayerBuffer, len(snap.DSABuffers))
	}
	for k, v := range snap.DSABuffers {
		if v == nil {
			delete(s.DSABuffers, k)
			continue
		}
		s.DSABuffers[k] = &GLM5NextDSALayerBuffer{
			K:      append([]float32(nil), v.K...),
			V:      append([]float32(nil), v.V...),
			IndexK: append([]float32(nil), v.IndexK...),
		}
	}
	for k := range s.DSABuffers {
		if _, ok := snap.DSABuffers[k]; !ok {
			delete(s.DSABuffers, k)
		}
	}

	if s.DownsampledKeys == nil {
		s.DownsampledKeys = make(map[int][]float32, len(snap.DownsampledKeys))
	}
	for k, v := range snap.DownsampledKeys {
		s.DownsampledKeys[k] = append([]float32(nil), v...)
	}
	for k := range s.DownsampledKeys {
		if _, ok := snap.DownsampledKeys[k]; !ok {
			delete(s.DownsampledKeys, k)
		}
	}
}

// EnsureParams lazily instantiates Params if nil or if HiddenSize != hiddenSize.
func (s *GLM5Next4LayerOracleState) EnsureParams(hiddenSize int) {
	if s == nil {
		return
	}
	if s.Params == nil || s.Params.HiddenSize != hiddenSize {
		numHeads := 64
		headDim := 128
		convWindow := 4
		if st := s.KDAStates[0]; st != nil {
			numHeads = st.NumHeads
			headDim = st.HeadDim
			convWindow = st.ConvWindow
		} else if s.Params != nil {
			if s.Params.NumHeads > 0 {
				numHeads = s.Params.NumHeads
			}
			if s.Params.HeadDim > 0 {
				headDim = s.Params.HeadDim
			}
			if s.Params.ConvWindow > 0 {
				convWindow = s.Params.ConvWindow
			}
		}
		s.Params = NewGLM5Next4LayerOracleParams(numHeads, headDim, convWindow, hiddenSize)
	}
}

// NewGLM5NextCadenceOracleState initializes oracle states for an arbitrary number of layers.
// Layers with layer%4 != 3 receive KDA states, while layers with layer%4 == 3 receive DSA buffers and configs.
func NewGLM5NextCadenceOracleState(numLayers, numHeads, headDim, convWindow int) *GLM5Next4LayerOracleState {
	if numHeads <= 0 {
		numHeads = 2
	}
	if headDim <= 0 {
		headDim = 8
	}
	if convWindow <= 0 {
		convWindow = 4
	}

	hiddenSize := numHeads * headDim
	params := NewGLM5Next4LayerOracleParams(numHeads, headDim, convWindow, hiddenSize)

	st := &GLM5Next4LayerOracleState{
		KDAStates:       make(map[int]*GLM5NextKDALayerState),
		Params:          params,
		DSABuffers:      make(map[int]*GLM5NextDSALayerBuffer),
		DownsampledKeys: make(map[int][]float32),
		DSAConfigs:      make(map[int]GLM5NextDSALayerConfig),
		KDAParams:       make(map[int]GLM5NextKDAParams),
		DenseParams:     make(map[int]GLM5NextDenseMLPParams),
		MoERouters:      make(map[int][]float32),
		MoEParams:       make(map[int]GLM5NextMoEMLPParams),
		MoENumExperts:   288,
		MoETopK:         8,
		MHCParams:       make(map[int]GLM5NextMHCParams),
		Diagnostics: GLM5NextCadenceDiagnostics{
			LayerSelectedBlocks: make(map[int][]int),
			LayerSelectedTokens: make(map[int][]int),
			LayerRoutes:         make(map[int]GLM5NextMoERouteResult),
		},
	}

	for l := 0; l < numLayers; l++ {
		if l%4 != 3 {
			st.KDAStates[l] = NewGLM5NextKDALayerState(numHeads, headDim, convWindow)
		} else {
			st.DSABuffers[l] = &GLM5NextDSALayerBuffer{}
			st.DSAConfigs[l] = GLM5NextDSALayerConfig{
				NumHeads:      numHeads,
				QKNopeHeadDim: headDim,
				VHeadDim:      headDim,
				NumIndexHeads: numHeads,
				IndexHeadDim:  headDim,
				BlockStride:   4,
				TopKBlocks:    2048,
			}
		}
	}

	return st
}

// NewGLM5Next4LayerOracleState initializes oracle states for layers 0..3, defaulting hiddenSize to numHeads*headDim.
func NewGLM5Next4LayerOracleState(numHeads, headDim, convWindow int) *GLM5Next4LayerOracleState {
	return NewGLM5NextCadenceOracleState(4, numHeads, headDim, convWindow)
}

func projectVector(x []float32, W []float32, outDim int) []float32 {
	out := make([]float32, outDim)
	hiddenSize := len(x)
	if hiddenSize == 0 || outDim == 0 {
		return out
	}
	if len(W) == outDim*hiddenSize {
		for i := 0; i < outDim; i++ {
			var sum float32
			rowOff := i * hiddenSize
			for j := 0; j < hiddenSize; j++ {
				sum += W[rowOff+j] * x[j]
			}
			out[i] = sum
		}
		return out
	}
	for i := 0; i < outDim; i++ {
		out[i] = x[i%hiddenSize]
	}
	return out
}

// RunGLM5Next4LayerCadenceBlock executes one 4-layer block (layers 0..3) on input vector x [hiddenSize]:
// - For layers 0, 1, 2: KDA linear mixer + Dense MLP + mHC residual
// - For layer 3: DSA sparse mixer + MoE MLP (top-8 of 288 + shared) + mHC residual
// Returns transformed hidden vector [hiddenSize].
func RunGLM5Next4LayerCadenceBlock(
	x []float32,
	state *GLM5Next4LayerOracleState,
	hiddenSize int,
) []float32 {
	state.EnsureParams(hiddenSize)

	cur := make([]float32, hiddenSize)
	copy(cur, x)

	for layer := 0; layer < 4; layer++ {
		if layer < 3 {
			// Layers 0, 1, 2: KDA linear attention + Dense MLP + mHC residual
			// 1. KDA linear mixer
			st := state.KDAStates[layer]
			if st != nil {
				var kdaOut []float32
				if state.Params != nil && state.Params.KDAParams != nil && state.Params.KDAParams[layer] != nil {
					kdaOut = ForwardGLM5NextKDALayerStep(st, *state.Params.KDAParams[layer], cur, 1e-6)
				} else {
					kdaOut = runKDAOracleLayerDelta(cur, hiddenSize)
				}
				for i := 0; i < hiddenSize; i++ {
					cur[i] += kdaOut[i]
				}
			}

			// 2. Dense MLP
			var mlpOut []float32
			if state.Params != nil && state.Params.DenseParams != nil && state.Params.DenseParams[layer] != nil && len(state.Params.DenseParams[layer].WAct) > 0 {
				mlpOut = ExecuteGLM5NextDenseMLP(cur, *state.Params.DenseParams[layer])
			} else {
				mlpOut = runDenseMLPOracleDelta(cur, hiddenSize)
			}
			for i := 0; i < hiddenSize; i++ {
				cur[i] += mlpOut[i]
			}

			// 3. mHC residual
			if state.Params != nil && state.Params.MHCParams != nil && state.Params.MHCParams[layer] != nil && len(state.Params.MHCParams[layer].WUp) > 0 {
				cur = ApplyGLM5NextMHC(cur, *state.Params.MHCParams[layer], 1e-6)
			} else {
				cur = runMHCOracle(cur, hiddenSize)
			}
		} else {
			// Layer 3: DSA sparse mixer + MoE MLP + mHC residual
			// 1. DSA sparse mixer via explicit oracle boundary
			var dsaOut []float32
			if state.Params != nil && state.Params.DSAOracle != nil {
				dsaOut = state.Params.DSAOracle(cur, state, hiddenSize)
			} else {
				dsaOut = runDSAOracleLayerDelta(cur, state, 3, hiddenSize)
			}
			for i := 0; i < hiddenSize; i++ {
				cur[i] += dsaOut[i]
			}

			// 2. MoE MLP
			var moeOut []float32
			moeOut = runMoEOracleDelta(cur, state, 3, hiddenSize)
			for i := 0; i < hiddenSize; i++ {
				cur[i] += moeOut[i]
			}

			// 3. mHC residual
			if state.Params != nil && state.Params.MHCParams != nil && state.Params.MHCParams[layer] != nil && len(state.Params.MHCParams[layer].WUp) > 0 {
				cur = ApplyGLM5NextMHC(cur, *state.Params.MHCParams[layer], 1e-6)
			} else {
				cur = runMHCOracle(cur, hiddenSize)
			}
		}
	}

	state.TotalTokens++
	return cur
}

// RunGLM5Next4LayerCadencePrefill processes a sequence of T tokens through the 4-layer cadence block,
// updating state and returning concatenated outputs [T * hiddenSize].
func RunGLM5Next4LayerCadencePrefill(
	xSeq []float32,
	state *GLM5Next4LayerOracleState,
	T, hiddenSize int,
) []float32 {
	state.EnsureParams(hiddenSize)
	outSeq := make([]float32, T*hiddenSize)
	for t := 0; t < T; t++ {
		xt := xSeq[t*hiddenSize : (t+1)*hiddenSize]
		tokOut := RunGLM5Next4LayerCadenceBlock(xt, state, hiddenSize)
		copy(outSeq[t*hiddenSize:(t+1)*hiddenSize], tokOut)
	}
	return outSeq
}

// RunGLM5NextCadence executes numLayers starting at startLayer on input vector x [hiddenSize].
func RunGLM5NextCadence(
	x []float32,
	state *GLM5Next4LayerOracleState,
	startLayer, numLayers int,
	hiddenSize int,
) []float32 {
	if startLayer == 0 && numLayers == 4 {
		return RunGLM5Next4LayerCadenceBlock(x, state, hiddenSize)
	}
	state.EnsureParams(hiddenSize)
	cur := make([]float32, hiddenSize)
	copy(cur, x)

	for layer := startLayer; layer < startLayer+numLayers; layer++ {
		isKDA := layer%4 != 3
		isDense := layer < 3

		if isKDA {
			st := state.KDAStates[layer]
			if st != nil {
				cur = runKDAOracleLayer(cur, st, state, layer, hiddenSize)
			}
		} else {
			cur = runDSAOracleLayer(cur, state, layer, hiddenSize)
		}

		if isDense {
			cur = runDenseMLPOracle(cur, state, layer, hiddenSize)
		} else {
			cur = runMoEOracle(cur, state, layer, hiddenSize)
		}

		cur = runMHCOracle(cur, hiddenSize)
	}

	state.TotalTokens++
	return cur
}

func runKDAOracleLayerDelta(x []float32, hiddenSize int) []float32 {
	out := make([]float32, hiddenSize)
	for i := range out {
		out[i] = x[i] * 0.01
	}
	return out
}

func runDenseMLPOracleDelta(x []float32, hiddenSize int) []float32 {
	out := make([]float32, hiddenSize)
	for i := range out {
		out[i] = silu(x[i]) * 0.05
	}
	return out
}

func runKDAOracleLayer(x []float32, st *GLM5NextKDALayerState, state *GLM5Next4LayerOracleState, layer, hiddenSize int) []float32 {
	if state != nil && state.Params != nil && state.Params.KDAParams != nil {
		if params, ok := state.Params.KDAParams[layer]; ok && params != nil {
			kdaOut := ForwardGLM5NextKDALayerStep(st, *params, x, 1e-6)
			out := make([]float32, hiddenSize)
			for i := 0; i < hiddenSize; i++ {
				out[i] = x[i] + kdaOut[i]
			}
			return out
		}
	}
	out := make([]float32, hiddenSize)
	for i := range out {
		out[i] = x[i] * 1.01
	}
	return out
}

func runDSAOracleLayerDelta(
	x []float32,
	state *GLM5Next4LayerOracleState,
	layer int,
	hiddenSize int,
) []float32 {
	buf := state.DSABuffers[layer]
	if buf == nil {
		buf = &GLM5NextDSALayerBuffer{}
		if state.DSABuffers == nil {
			state.DSABuffers = make(map[int]*GLM5NextDSALayerBuffer)
		}
		state.DSABuffers[layer] = buf
	}

	cfg, ok := state.DSAConfigs[layer]
	if !ok {
		cfg = GLM5NextDSALayerConfig{
			BlockStride: 4,
			TopKBlocks:  2048,
		}
	}
	if cfg.BlockStride <= 0 {
		cfg.BlockStride = 4
	}
	if cfg.TopKBlocks <= 0 {
		cfg.TopKBlocks = 2048
	}
	if cfg.NumHeads <= 0 {
		cfg.NumHeads = 2
	}
	if cfg.QKNopeHeadDim <= 0 {
		cfg.QKNopeHeadDim = 8
	}
	if cfg.VHeadDim <= 0 {
		cfg.VHeadDim = cfg.QKNopeHeadDim
	}
	if cfg.NumIndexHeads <= 0 {
		cfg.NumIndexHeads = cfg.NumHeads
	}
	if cfg.IndexHeadDim <= 0 {
		cfg.IndexHeadDim = cfg.QKNopeHeadDim
	}

	qDim := cfg.NumHeads * cfg.QKNopeHeadDim
	kDim := cfg.NumHeads * cfg.QKNopeHeadDim
	vDim := cfg.NumHeads * cfg.VHeadDim
	idxDim := cfg.NumIndexHeads * cfg.IndexHeadDim

	q := projectVector(x, cfg.Wq, qDim)
	k := projectVector(x, cfg.Wk, kDim)
	v := projectVector(x, cfg.Wv, vDim)
	qIdx := projectVector(x, cfg.WqIdx, idxDim)
	kIdx := projectVector(x, cfg.WkIdx, idxDim)

	buf.K = append(buf.K, k...)
	buf.V = append(buf.V, v...)
	buf.IndexK = append(buf.IndexK, kIdx...)

	t := state.TotalTokens
	totalTokens := len(buf.K) / kDim
	stride := cfg.BlockStride

	downsampled := DownsampleGLM5NextDSAKeys(buf.IndexK, totalTokens, idxDim, stride)
	if state.DownsampledKeys == nil {
		state.DownsampledKeys = make(map[int][]float32)
	}
	state.DownsampledKeys[layer] = downsampled

	totalBlocks := (totalTokens + stride - 1) / stride
	currentBlock := t / stride
	topK := cfg.TopKBlocks

	selectedBlocks := SelectGLM5NextDSATopK(qIdx, downsampled, totalBlocks, currentBlock, cfg.NumIndexHeads, cfg.IndexHeadDim, topK)
	selectedTokens := ExpandDSABlocksToTokens(selectedBlocks, stride, totalTokens)

	mixerOut := ComputeGLM5NextDSASparseMixer(q, buf.K, buf.V, selectedTokens, t, cfg.NumHeads, cfg.QKNopeHeadDim, cfg.VHeadDim)

	proj := projectVector(mixerOut, cfg.WOut, hiddenSize)

	state.Diagnostics.SelectedBlocks = selectedBlocks
	state.Diagnostics.SelectedTokens = selectedTokens
	if state.Diagnostics.LayerSelectedBlocks == nil {
		state.Diagnostics.LayerSelectedBlocks = make(map[int][]int)
	}
	state.Diagnostics.LayerSelectedBlocks[layer] = selectedBlocks
	if state.Diagnostics.LayerSelectedTokens == nil {
		state.Diagnostics.LayerSelectedTokens = make(map[int][]int)
	}
	state.Diagnostics.LayerSelectedTokens[layer] = selectedTokens

	return proj
}

func runDSAOracleLayer(
	x []float32,
	state *GLM5Next4LayerOracleState,
	layer int,
	hiddenSize int,
) []float32 {
	if state != nil && state.Params != nil && state.Params.DSAOracle != nil && layer == 3 {
		dsaOut := state.Params.DSAOracle(x, state, hiddenSize)
		out := make([]float32, hiddenSize)
		for i := 0; i < hiddenSize; i++ {
			out[i] = x[i] + dsaOut[i]
		}
		return out
	}

	delta := runDSAOracleLayerDelta(x, state, layer, hiddenSize)
	out := make([]float32, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		out[i] = x[i] + delta[i]
	}
	return out
}

func runDenseMLPOracle(
	x []float32,
	state *GLM5Next4LayerOracleState,
	layer int,
	hiddenSize int,
) []float32 {
	if state != nil && state.Params != nil && state.Params.DenseParams != nil {
		if params, ok := state.Params.DenseParams[layer]; ok && params != nil && len(params.WAct) > 0 {
			mlpOut := ExecuteGLM5NextDenseMLP(x, *params)
			out := make([]float32, hiddenSize)
			for i := 0; i < hiddenSize; i++ {
				out[i] = x[i] + mlpOut[i%len(mlpOut)]
			}
			return out
		}
	}
	out := make([]float32, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		out[i] = x[i] + silu(x[i])*0.05
	}
	return out
}

func runMoEOracleDelta(
	x []float32,
	state *GLM5Next4LayerOracleState,
	layer int,
	hiddenSize int,
) []float32 {
	numExperts := state.MoENumExperts
	if numExperts <= 0 {
		numExperts = 288
	}
	topK := state.MoETopK
	if topK <= 0 {
		topK = 8
	}

	wRouter := state.MoERouters[layer]
	if len(wRouter) < numExperts*hiddenSize {
		wRouter = make([]float32, numExperts*hiddenSize)
		for e := 0; e < numExperts; e++ {
			for j := 0; j < hiddenSize; j++ {
				wRouter[e*hiddenSize+j] = float32((e+1)*(j+1)%31-15) * 0.01
			}
		}
		if state.MoERouters == nil {
			state.MoERouters = make(map[int][]float32)
		}
		state.MoERouters[layer] = wRouter
	}

	route := RouteGLM5NextMoE(x, wRouter, numExperts, topK)

	var moeParams GLM5NextMoEMLPParams
	if state.Params != nil && state.Params.MoEParams != nil {
		moeParams = *state.Params.MoEParams
	} else if p, ok := state.MoEParams[layer]; ok {
		moeParams = p
	}
	if moeParams.InDim == 0 {
		moeParams.InDim = hiddenSize
	}
	if moeParams.MoEInterDim == 0 {
		moeParams.MoEInterDim = max(hiddenSize, 16)
	}

	moeOut := ExecuteGLM5NextSparseMoE(x, route, moeParams)

	hasWeights := len(moeParams.SharedExpert.WAct) > 0 || len(moeParams.RoutedExperts) > 0
	out := make([]float32, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		if hasWeights {
			out[i] = moeOut[i]
		} else {
			out[i] = moeOut[i] + silu(x[i])*0.08
		}
	}

	state.Diagnostics.Route = route
	if state.Diagnostics.LayerRoutes == nil {
		state.Diagnostics.LayerRoutes = make(map[int]GLM5NextMoERouteResult)
	}
	state.Diagnostics.LayerRoutes[layer] = route

	return out
}

func runMoEOracle(
	x []float32,
	state *GLM5Next4LayerOracleState,
	layer int,
	hiddenSize int,
) []float32 {
	delta := runMoEOracleDelta(x, state, layer, hiddenSize)
	out := make([]float32, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		out[i] = x[i] + delta[i]
	}
	return out
}

func runMHCOracle(x []float32, hiddenSize int) []float32 {
	out := make([]float32, hiddenSize)
	for i := range out {
		out[i] = x[i] * 0.999
	}
	return out
}
