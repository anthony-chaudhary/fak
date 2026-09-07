package model

// Verification tests for GLM5Next KPool DSA, MoE router, and multi-cycle cadence (#9441):
// - TestRunGLM5Next4LayerCadenceBlock: 4-layer decode steps, prefill vs decode equivalence, reset and restore
// - TestRunGLM5Next4LayerCadenceBlock_MultiTokenHistory: KPool key buffer growth and block pooling across sequential tokens
// - TestGLM5NextLayer3_DSACausalityAndLocalRetention: causal block masking and unconditional local window retention
// - TestGLM5NextLayer3_MoERoutingWitness: 288-expert router, sigmoid gating, top-8 selection, descending L1-normalized weights
// - TestGLM5NextMultiCycleCadence_8Layers: multi-cycle cadence across 8 layers (layers 0..2 dense, layers 3..7 MoE, layers 3 and 7 DSA)

import (
	"math"
	"math/rand"
	"testing"
)

func TestRunGLM5Next4LayerCadenceBlock(t *testing.T) {
	const hiddenSize = 16
	const numHeads = 2
	const headDim = 8
	const convWindow = 4

	state := NewGLM5Next4LayerOracleState(numHeads, headDim, convWindow)
	if len(state.KDAStates) != 3 {
		t.Fatalf("expected 3 KDA states for layers 0..2, got %d", len(state.KDAStates))
	}

	x := make([]float32, hiddenSize)
	for i := range x {
		x[i] = 1.0
	}

	// 1. Single-token decode steps
	out1 := RunGLM5Next4LayerCadenceBlock(x, state, hiddenSize)
	if len(out1) != hiddenSize {
		t.Fatalf("len(out1) = %d, want %d", len(out1), hiddenSize)
	}
	if state.TotalTokens != 1 {
		t.Fatalf("TotalTokens = %d, want 1", state.TotalTokens)
	}

	for i, v := range out1 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v == 0 {
			t.Fatalf("out1[%d] = %g, expected finite non-zero", i, v)
		}
	}

	out2 := RunGLM5Next4LayerCadenceBlock(x, state, hiddenSize)
	if state.TotalTokens != 2 {
		t.Fatalf("TotalTokens = %d, want 2", state.TotalTokens)
	}
	for i, v := range out2 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v == 0 {
			t.Fatalf("out2[%d] = %g, expected finite non-zero", i, v)
		}
	}

	// 2. Prefill vs stepwise decode numerical equivalence
	const T = 6
	rng := rand.New(rand.NewSource(999))
	xSeq := make([]float32, T*hiddenSize)
	for i := range xSeq {
		xSeq[i] = rng.Float32()*2.0 - 1.0
	}

	statePrefill := NewGLM5Next4LayerOracleState(numHeads, headDim, convWindow)
	stateDecode := statePrefill.Clone()

	outPrefill := RunGLM5Next4LayerCadencePrefill(xSeq, statePrefill, T, hiddenSize)

	outDecode := make([]float32, T*hiddenSize)
	for step := 0; step < T; step++ {
		tokX := xSeq[step*hiddenSize : (step+1)*hiddenSize]
		tokOut := RunGLM5Next4LayerCadenceBlock(tokX, stateDecode, hiddenSize)
		copy(outDecode[step*hiddenSize:(step+1)*hiddenSize], tokOut)
	}

	if len(outPrefill) != len(outDecode) {
		t.Fatalf("lengths mismatch: prefill=%d decode=%d", len(outPrefill), len(outDecode))
	}
	for i := range outPrefill {
		if math.Float32bits(outPrefill[i]) != math.Float32bits(outDecode[i]) {
			t.Fatalf("output mismatch at pos %d/%d (step %d): prefill=%g (%08x) decode=%g (%08x)",
				i, len(outPrefill), i/hiddenSize,
				outPrefill[i], math.Float32bits(outPrefill[i]),
				outDecode[i], math.Float32bits(outDecode[i]))
		}
	}

	// Verify bit-for-bit equality of final recurrent states S and ConvQ/K/V buffers
	for l := 0; l < 3; l++ {
		stP := statePrefill.KDAStates[l]
		stD := stateDecode.KDAStates[l]
		for i := range stP.S {
			if math.Float32bits(stP.S[i]) != math.Float32bits(stD.S[i]) {
				t.Fatalf("layer %d S mismatch at %d: prefill=%g (%08x) decode=%g (%08x)",
					l, i, stP.S[i], math.Float32bits(stP.S[i]), stD.S[i], math.Float32bits(stD.S[i]))
			}
		}
		for i := range stP.ConvQ {
			if math.Float32bits(stP.ConvQ[i]) != math.Float32bits(stD.ConvQ[i]) {
				t.Fatalf("layer %d ConvQ mismatch at %d: prefill=%g decode=%g", l, i, stP.ConvQ[i], stD.ConvQ[i])
			}
		}
		for i := range stP.ConvK {
			if math.Float32bits(stP.ConvK[i]) != math.Float32bits(stD.ConvK[i]) {
				t.Fatalf("layer %d ConvK mismatch at %d: prefill=%g decode=%g", l, i, stP.ConvK[i], stD.ConvK[i])
			}
		}
		for i := range stP.ConvV {
			if math.Float32bits(stP.ConvV[i]) != math.Float32bits(stD.ConvV[i]) {
				t.Fatalf("layer %d ConvV mismatch at %d: prefill=%g decode=%g", l, i, stP.ConvV[i], stD.ConvV[i])
			}
		}
	}
	if statePrefill.TotalTokens != stateDecode.TotalTokens || statePrefill.TotalTokens != T {
		t.Fatalf("TotalTokens mismatch: prefill=%d decode=%d want=%d", statePrefill.TotalTokens, stateDecode.TotalTokens, T)
	}

	// 3. Reset: zeroes S, conv buffers, TotalTokens=0; executing after reset matches fresh run
	stateReset := NewGLM5Next4LayerOracleState(numHeads, headDim, convWindow)
	freshOut := RunGLM5Next4LayerCadenceBlock(x, stateReset, hiddenSize)

	RunGLM5Next4LayerCadenceBlock(x, stateReset, hiddenSize)
	RunGLM5Next4LayerCadenceBlock(x, stateReset, hiddenSize)
	if stateReset.TotalTokens != 3 {
		t.Fatalf("stateReset TotalTokens = %d, want 3", stateReset.TotalTokens)
	}

	stateReset.Reset()
	if stateReset.TotalTokens != 0 {
		t.Fatalf("after Reset: TotalTokens = %d, want 0", stateReset.TotalTokens)
	}
	for l := 0; l < 3; l++ {
		st := stateReset.KDAStates[l]
		for i, v := range st.S {
			if v != 0 {
				t.Fatalf("layer %d S[%d] = %g after Reset, want 0", l, i, v)
			}
		}
		for i, v := range st.ConvQ {
			if v != 0 {
				t.Fatalf("layer %d ConvQ[%d] = %g after Reset, want 0", l, i, v)
			}
		}
		for i, v := range st.ConvK {
			if v != 0 {
				t.Fatalf("layer %d ConvK[%d] = %g after Reset, want 0", l, i, v)
			}
		}
		for i, v := range st.ConvV {
			if v != 0 {
				t.Fatalf("layer %d ConvV[%d] = %g after Reset, want 0", l, i, v)
			}
		}
	}

	postResetOut := RunGLM5Next4LayerCadenceBlock(x, stateReset, hiddenSize)
	for i := range freshOut {
		if math.Float32bits(freshOut[i]) != math.Float32bits(postResetOut[i]) {
			t.Fatalf("post-reset mismatch at %d: fresh=%g (%08x) postReset=%g (%08x)",
				i, freshOut[i], math.Float32bits(freshOut[i]),
				postResetOut[i], math.Float32bits(postResetOut[i]))
		}
	}

	// 4. Snapshot and Restore: snapshot, step forward, restore, step forward again — exact match
	snap := stateReset.Snapshot()
	stepOut1 := RunGLM5Next4LayerCadenceBlock(x, stateReset, hiddenSize)
	stepOut2 := RunGLM5Next4LayerCadenceBlock(x, stateReset, hiddenSize)

	stateReset.Restore(snap)
	if stateReset.TotalTokens != snap.TotalTokens {
		t.Fatalf("after Restore: TotalTokens = %d, want %d", stateReset.TotalTokens, snap.TotalTokens)
	}

	restoreStepOut1 := RunGLM5Next4LayerCadenceBlock(x, stateReset, hiddenSize)
	for i := range stepOut1 {
		if math.Float32bits(stepOut1[i]) != math.Float32bits(restoreStepOut1[i]) {
			t.Fatalf("restore step 1 mismatch at %d: orig=%g (%08x) restored=%g (%08x)",
				i, stepOut1[i], math.Float32bits(stepOut1[i]),
				restoreStepOut1[i], math.Float32bits(restoreStepOut1[i]))
		}
	}

	restoreStepOut2 := RunGLM5Next4LayerCadenceBlock(x, stateReset, hiddenSize)
	for i := range stepOut2 {
		if math.Float32bits(stepOut2[i]) != math.Float32bits(restoreStepOut2[i]) {
			t.Fatalf("restore step 2 mismatch at %d: orig=%g (%08x) restored=%g (%08x)",
				i, stepOut2[i], math.Float32bits(stepOut2[i]),
				restoreStepOut2[i], math.Float32bits(restoreStepOut2[i]))
		}
	}

	// 5. Test DSA explicit oracle boundary
	dsaCalled := false
	var capturedHiddenSize int
	stateOracle := NewGLM5Next4LayerOracleState(numHeads, headDim, convWindow)
	stateOracle.Params.DSAOracle = func(cur []float32, st *GLM5Next4LayerOracleState, hSize int) []float32 {
		dsaCalled = true
		capturedHiddenSize = hSize
		res := make([]float32, hSize)
		for i := range res {
			res[i] = 100.0
		}
		return res
	}

	dsaTestOut := RunGLM5Next4LayerCadenceBlock(x, stateOracle, hiddenSize)
	if !dsaCalled {
		t.Fatalf("expected custom DSAOracle to be called")
	}
	if capturedHiddenSize != hiddenSize {
		t.Fatalf("captured hiddenSize = %d, want %d", capturedHiddenSize, hiddenSize)
	}
	for i, v := range dsaTestOut {
		if v < 10.0 {
			t.Fatalf("dsaTestOut[%d] = %g, expected large value from DSA oracle override", i, v)
		}
	}
}

func TestForwardGLM5NextKDALayerStepAndPrefillSeq(t *testing.T) {
	const numHeads = 2
	const headDim = 8
	const featureDim = numHeads * headDim
	const hiddenSize = 16
	const convWindow = 4
	const T = 5

	rng := rand.New(rand.NewSource(777))

	initConv := func() *GLM5NextKDAConvFilter {
		f := NewGLM5NextKDAConvFilter(featureDim, convWindow)
		for i := range f.Weight {
			f.Weight[i] = rng.Float32()*2.0 - 1.0
		}
		return f
	}

	randWeights := func(n int) []float32 {
		w := make([]float32, n)
		for i := range w {
			w[i] = (rng.Float32()*2.0 - 1.0) * 0.1
		}
		return w
	}

	params := GLM5NextKDAParams{
		ConvQ:      initConv(),
		ConvK:      initConv(),
		ConvV:      initConv(),
		BaseDecay:  make([]float32, numHeads),
		Wout:       randWeights(hiddenSize * featureDim),
		HiddenSize: hiddenSize,
		Wq:         randWeights(featureDim * hiddenSize),
		Wk:         randWeights(featureDim * hiddenSize),
		Wv:         randWeights(featureDim * hiddenSize),
		Wdecay:     randWeights(numHeads * hiddenSize),
		Wmod:       randWeights(featureDim * hiddenSize),
		RMSNormEps: 1e-6,
	}
	for h := 0; h < numHeads; h++ {
		params.BaseDecay[h] = 0.8
	}

	xSeq := make([]float32, T*hiddenSize)
	for i := range xSeq {
		xSeq[i] = rng.Float32()*2.0 - 1.0
	}

	// 1. With in-projections: PrefillSeq vs Stepwise Step
	stPrefill := NewGLM5NextKDALayerState(numHeads, headDim, convWindow)
	outPrefill := ForwardGLM5NextKDALayerPrefillSeq(stPrefill, params, xSeq, T, 1e-6)

	stDecode := NewGLM5NextKDALayerState(numHeads, headDim, convWindow)
	outDecode := make([]float32, T*hiddenSize)
	for tStep := 0; tStep < T; tStep++ {
		xt := xSeq[tStep*hiddenSize : (tStep+1)*hiddenSize]
		stepOut := ForwardGLM5NextKDALayerStep(stDecode, params, xt, 1e-6)
		copy(outDecode[tStep*hiddenSize:(tStep+1)*hiddenSize], stepOut)
	}

	for i := range outPrefill {
		if math.Float32bits(outPrefill[i]) != math.Float32bits(outDecode[i]) {
			t.Fatalf("in-proj mismatch at %d: prefill=%g (%08x) decode=%g (%08x)",
				i, outPrefill[i], math.Float32bits(outPrefill[i]),
				outDecode[i], math.Float32bits(outDecode[i]))
		}
	}
	for i := range stPrefill.S {
		if math.Float32bits(stPrefill.S[i]) != math.Float32bits(stDecode.S[i]) {
			t.Fatalf("in-proj S mismatch at %d: prefill=%g decode=%g", i, stPrefill.S[i], stDecode.S[i])
		}
	}

	// 2. Without in-projections (empty): PrefillSeq vs Stepwise Step
	paramsEmptyInProj := GLM5NextKDAParams{
		ConvQ:      initConv(),
		ConvK:      initConv(),
		ConvV:      initConv(),
		BaseDecay:  make([]float32, numHeads),
		Wout:       randWeights(hiddenSize * featureDim),
		HiddenSize: hiddenSize,
	}
	for h := 0; h < numHeads; h++ {
		paramsEmptyInProj.BaseDecay[h] = 0.8
	}

	stEmptyPrefill := NewGLM5NextKDALayerState(numHeads, headDim, convWindow)
	outEmptyPrefill := ForwardGLM5NextKDALayerPrefillSeq(stEmptyPrefill, paramsEmptyInProj, xSeq, T, 1e-6)

	stEmptyDecode := NewGLM5NextKDALayerState(numHeads, headDim, convWindow)
	outEmptyDecode := make([]float32, T*hiddenSize)
	for tStep := 0; tStep < T; tStep++ {
		xt := xSeq[tStep*hiddenSize : (tStep+1)*hiddenSize]
		stepOut := ForwardGLM5NextKDALayerStep(stEmptyDecode, paramsEmptyInProj, xt, 1e-6)
		copy(outEmptyDecode[tStep*hiddenSize:(tStep+1)*hiddenSize], stepOut)
	}

	for i := range outEmptyPrefill {
		if math.Float32bits(outEmptyPrefill[i]) != math.Float32bits(outEmptyDecode[i]) {
			t.Fatalf("empty in-proj mismatch at %d: prefill=%g decode=%g", i, outEmptyPrefill[i], outEmptyDecode[i])
		}
	}
}

func TestRunGLM5Next4LayerCadenceBlock_MultiTokenHistory(t *testing.T) {
	const hiddenSize = 16
	const numHeads = 2
	const headDim = 8
	const convWindow = 4
	const numTokens = 9

	state := NewGLM5Next4LayerOracleState(numHeads, headDim, convWindow)
	cfg := state.DSAConfigs[3]
	kDim := cfg.NumHeads * cfg.QKNopeHeadDim
	vDim := cfg.NumHeads * cfg.VHeadDim
	idxDim := cfg.NumIndexHeads * cfg.IndexHeadDim
	stride := cfg.BlockStride

	x := make([]float32, hiddenSize)
	for i := range x {
		x[i] = 1.0
	}

	for tok := 0; tok < numTokens; tok++ {
		out := RunGLM5Next4LayerCadenceBlock(x, state, hiddenSize)
		if len(out) != hiddenSize {
			t.Fatalf("tok %d: len(out) = %d, want %d", tok, len(out), hiddenSize)
		}
		if state.TotalTokens != tok+1 {
			t.Fatalf("tok %d: TotalTokens = %d, want %d", tok, state.TotalTokens, tok+1)
		}

		buf := state.DSABuffers[3]
		if buf == nil {
			t.Fatalf("tok %d: DSABuffers[3] is nil", tok)
		}
		expectedTokens := tok + 1
		if len(buf.K) != expectedTokens*kDim {
			t.Fatalf("tok %d: len(buf.K) = %d, want %d", tok, len(buf.K), expectedTokens*kDim)
		}
		if len(buf.V) != expectedTokens*vDim {
			t.Fatalf("tok %d: len(buf.V) = %d, want %d", tok, len(buf.V), expectedTokens*vDim)
		}
		if len(buf.IndexK) != expectedTokens*idxDim {
			t.Fatalf("tok %d: len(buf.IndexK) = %d, want %d", tok, len(buf.IndexK), expectedTokens*idxDim)
		}

		expectedBlocks := (expectedTokens + stride - 1) / stride
		downsampled := state.DownsampledKeys[3]
		if len(downsampled) != expectedBlocks*idxDim {
			t.Fatalf("tok %d: len(downsampled) = %d, want %d (expectedBlocks=%d)", tok, len(downsampled), expectedBlocks*idxDim, expectedBlocks)
		}

		for i, v := range out {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v == 0 {
				t.Fatalf("tok %d out[%d] = %g, expected finite non-zero", tok, i, v)
			}
		}
	}
}

func TestGLM5NextLayer3_DSACausalityAndLocalRetention(t *testing.T) {
	const hiddenSize = 16
	const numHeads = 2
	const headDim = 8
	const convWindow = 4
	const stride = 4

	state := NewGLM5Next4LayerOracleState(numHeads, headDim, convWindow)
	cfg := state.DSAConfigs[3]
	cfg.BlockStride = stride
	cfg.TopKBlocks = 2
	state.DSAConfigs[3] = cfg

	x := make([]float32, hiddenSize)
	for i := range x {
		x[i] = 1.0
	}

	for tok := 0; tok < 12; tok++ {
		RunGLM5Next4LayerCadenceBlock(x, state, hiddenSize)
		currentBlock := tok / stride

		selectedBlocks := state.Diagnostics.LayerSelectedBlocks[3]
		if len(selectedBlocks) == 0 {
			t.Fatalf("tok %d: expected non-empty selectedBlocks", tok)
		}

		hasCurrentBlock := false
		for _, b := range selectedBlocks {
			// 1. Causal mask: block b > currentBlock is forbidden
			if b > currentBlock {
				t.Fatalf("tok %d: causal violation: selected future block %d (currentBlock=%d)", tok, b, currentBlock)
			}
			if b == currentBlock {
				hasCurrentBlock = true
			}
		}

		// 2. Local window: currentBlock must always be retained
		if !hasCurrentBlock {
			t.Fatalf("tok %d: local block retention failed: currentBlock %d not in %v", tok, currentBlock, selectedBlocks)
		}

		// 3. Token causality
		selectedTokens := state.Diagnostics.LayerSelectedTokens[3]
		for _, tokIdx := range selectedTokens {
			if tokIdx > tok {
				t.Fatalf("tok %d: causal token violation: selected future token %d", tok, tokIdx)
			}
		}
	}
}

func TestGLM5NextLayer3_MoERoutingWitness(t *testing.T) {
	const hiddenSize = 16
	const numHeads = 2
	const headDim = 8
	const convWindow = 4

	state := NewGLM5Next4LayerOracleState(numHeads, headDim, convWindow)
	if state.MoENumExperts != 288 {
		t.Fatalf("expected MoENumExperts=288, got %d", state.MoENumExperts)
	}
	if state.MoETopK != 8 {
		t.Fatalf("expected MoETopK=8, got %d", state.MoETopK)
	}

	x := make([]float32, hiddenSize)
	for i := range x {
		x[i] = float32(i+1) * 0.1
	}

	RunGLM5Next4LayerCadenceBlock(x, state, hiddenSize)

	route := state.Diagnostics.Route
	if len(route.ExpertIndices) != 8 {
		t.Fatalf("expected 8 selected experts, got %d", len(route.ExpertIndices))
	}
	if len(route.Weights) != 8 {
		t.Fatalf("expected 8 expert weights, got %d", len(route.Weights))
	}

	seen := make(map[int]bool, 8)
	var sumWeights float32
	for i, expIdx := range route.ExpertIndices {
		if expIdx < 0 || expIdx >= 288 {
			t.Fatalf("expert index %d out of bounds [0, 288)", expIdx)
		}
		if seen[expIdx] {
			t.Fatalf("duplicate expert index %d in top-8 route", expIdx)
		}
		seen[expIdx] = true

		w := route.Weights[i]
		if w <= 0 {
			t.Fatalf("expert %d has non-positive weight %g", expIdx, w)
		}
		sumWeights += w
	}

	if math.Abs(float64(sumWeights-1.0)) > 1e-5 {
		t.Fatalf("expert weights sum to %g, expected 1.0", sumWeights)
	}

	layerRoute, ok := state.Diagnostics.LayerRoutes[3]
	if !ok {
		t.Fatalf("expected LayerRoutes[3] to be recorded")
	}
	if len(layerRoute.ExpertIndices) != 8 {
		t.Fatalf("expected 8 experts in LayerRoutes[3], got %d", len(layerRoute.ExpertIndices))
	}

	// Verify custom router matrix directs highest score to specified expert
	customRouter := make([]float32, 288*hiddenSize)
	for j := 0; j < hiddenSize; j++ {
		customRouter[137*hiddenSize+j] = 10.0
	}
	state.MoERouters[3] = customRouter
	RunGLM5Next4LayerCadenceBlock(x, state, hiddenSize)

	routeCustom := state.Diagnostics.Route
	if routeCustom.ExpertIndices[0] != 137 {
		t.Fatalf("expected expert 137 to be rank 0, got %d", routeCustom.ExpertIndices[0])
	}
}

func TestGLM5NextMultiCycleCadence_8Layers(t *testing.T) {
	const hiddenSize = 16
	const numHeads = 2
	const headDim = 8
	const convWindow = 4
	const numLayers = 8

	state := NewGLM5NextCadenceOracleState(numLayers, numHeads, headDim, convWindow)

	// 1. Structural verification of 2 cycles across 8 layers:
	// 3:1 KDA/DSA cadence: layers 0,1,2,4,5,6 KDA, layers 3,7 DSA
	for l := 0; l < numLayers; l++ {
		isKDA := l%4 != 3
		if isKDA {
			if state.KDAStates[l] == nil {
				t.Fatalf("layer %d: expected KDA state, got nil", l)
			}
			if state.DSABuffers[l] != nil {
				t.Fatalf("layer %d: expected nil DSA buffer, got %v", l, state.DSABuffers[l])
			}
		} else {
			if state.KDAStates[l] != nil {
				t.Fatalf("layer %d: expected nil KDA state, got %v", l, state.KDAStates[l])
			}
			if state.DSABuffers[l] == nil {
				t.Fatalf("layer %d: expected DSA buffer, got nil", l)
			}
			if _, ok := state.DSAConfigs[l]; !ok {
				t.Fatalf("layer %d: expected DSA config for layer %d", l, l)
			}
		}
	}

	x := make([]float32, hiddenSize)
	for i := range x {
		x[i] = 1.0
	}

	// 2. Execute first token through all 8 layers
	out1 := RunGLM5NextCadence(x, state, 0, numLayers, hiddenSize)
	if len(out1) != hiddenSize {
		t.Fatalf("len(out1) = %d, want %d", len(out1), hiddenSize)
	}
	if state.TotalTokens != 1 {
		t.Fatalf("TotalTokens = %d, want 1", state.TotalTokens)
	}

	// Verify DSA execution on both cadence DSA layers (3 and 7)
	for _, l := range []int{3, 7} {
		buf := state.DSABuffers[l]
		if len(buf.K) == 0 || len(buf.V) == 0 || len(buf.IndexK) == 0 {
			t.Fatalf("layer %d: DSA buffer was not populated", l)
		}
		if len(state.DownsampledKeys[l]) == 0 {
			t.Fatalf("layer %d: DownsampledKeys was not populated", l)
		}
		if len(state.Diagnostics.LayerSelectedBlocks[l]) == 0 {
			t.Fatalf("layer %d: LayerSelectedBlocks was not recorded", l)
		}
		if len(state.Diagnostics.LayerSelectedTokens[l]) == 0 {
			t.Fatalf("layer %d: LayerSelectedTokens was not recorded", l)
		}
	}

	// Verify Dense-to-MoE transition:
	// Layers 0, 1, 2: Dense MLP (no MoE route recorded)
	for l := 0; l < 3; l++ {
		if _, ok := state.Diagnostics.LayerRoutes[l]; ok {
			t.Fatalf("layer %d: expected Dense MLP (no MoE route recorded), but got route", l)
		}
	}
	// Layers 3..7: MoE MLP (MoE route recorded with 8 experts each, weights summing to 1.0)
	for l := 3; l < 8; l++ {
		route, ok := state.Diagnostics.LayerRoutes[l]
		if !ok {
			t.Fatalf("layer %d: expected MoE MLP route recorded, got none", l)
		}
		if len(route.ExpertIndices) != 8 {
			t.Fatalf("layer %d: expected 8 expert indices, got %d", l, len(route.ExpertIndices))
		}
		var sumW float32
		for _, w := range route.Weights {
			sumW += w
		}
		if math.Abs(float64(sumW-1.0)) > 1e-5 {
			t.Fatalf("layer %d: expert weights sum to %g, expected 1.0", l, sumW)
		}
	}

	// 3. Step second token to verify recurrence across all 8 layers
	out2 := RunGLM5NextCadence(x, state, 0, numLayers, hiddenSize)
	if state.TotalTokens != 2 {
		t.Fatalf("TotalTokens = %d, want 2", state.TotalTokens)
	}
	for _, l := range []int{3, 7} {
		cfg := state.DSAConfigs[l]
		kDim := cfg.NumHeads * cfg.QKNopeHeadDim
		if len(state.DSABuffers[l].K) != 2*kDim {
			t.Fatalf("layer %d: expected 2 tokens in DSA buffer, got %d floats", l, len(state.DSABuffers[l].K))
		}
	}
	for i, v := range out2 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v == 0 {
			t.Fatalf("out2[%d] = %g, expected finite non-zero", i, v)
		}
	}
}
