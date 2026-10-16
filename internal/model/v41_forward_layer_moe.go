package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/model/ffn"
)

// v41LayerMoEInput carries the attention-half results of one v41Layer into its
// MoE half.
type v41LayerMoEInput struct {
	l         int
	x         [][]float32
	streams   [][][]float32
	full      bool
	H         int
	eps       float32
	hcIters   int
	hcEps     float32
	routeCfg  v41RouterConfig
	st        *v41ForwardState
	scratch   *v41ProjScratch
	trace     *v41ActivationTraceState
	mhcDevice *v41MHCDevice
	resident  *v41MHCResident
	gateBias  []float32
	hcByPos   []v41MHCMix
	preByPos  [][]float32
	attnOut   [][]float32
}

// v41LayerFFNPanel is the normalized FFN input panel shared by the router,
// every routed expert, and the shared expert.
type v41LayerFFNPanel struct {
	inputs   [][]float32
	flat     []float32
	residual [][][]float32
	mixes    []v41MHCMix
}

// v41LayerMoE runs the router, the routed and shared experts, and the mHC
// post-mix for one layer, then publishes the next streams.
func (m *Model) v41LayerMoE(in v41LayerMoEInput) error {
	panel, err := m.v41LayerFFNInputs(in)
	if err != nil {
		return err
	}
	perTokenPicks, err := m.v41LayerRoute(in, panel)
	if err != nil {
		return err
	}
	routedByToken, err := m.v41LayerRoutedContraction(in, panel.inputs, perTokenPicks)
	if err != nil {
		return err
	}
	return m.v41LayerSharedExpertAndPost(in, panel, routedByToken)
}

// v41LayerFFNInputs resets the layer-scoped expert cache, so its working set
// is exactly this layer's routed experts (#13296), and builds the normalized
// FFN panel. Device normalization uploads and reads back only once per
// layer/token; grouped contraction consumes this same panel.
func (m *Model) v41LayerFFNInputs(in v41LayerMoEInput) (v41LayerFFNPanel, error) {
	l, H, scratch := in.l, in.H, in.scratch
	seq := len(in.x)
	var panel v41LayerFFNPanel
	scratch.v41LayerCacheReset()
	if scratch.expertLayerCacheBytes == 0 {
		if v41TestLayerCacheBudgetOverride > 0 {
			scratch.expertLayerCacheBytes = v41TestLayerCacheBudgetOverride
		} else {
			scratch.expertLayerCacheBytes = m.v41LayerExpertCacheBudget()
		}
	}
	panel.inputs = make([][]float32, seq)
	var projectFFN func([]float32) ([]float32, error)
	var err error
	if in.full {
		// Every attention projection has completed. Reusing scratch.mhc is now safe.
		projectFFN, err = m.v41FFNMHCProjector(l, scratch)
		if err != nil {
			return panel, err
		}
		if in.resident == nil {
			panel.residual = make([][][]float32, seq)
			panel.mixes = make([]v41MHCMix, seq)
		}
	}
	var ffnCollapsed [][]float32
	var ffnOwned *v41OwnedFFNCollapse
	if in.resident != nil {
		if seq > 1 {
			ffnOwned, err = in.resident.ffnInputOwned(l, in.attnOut, in.hcIters, in.hcEps, in.eps)
			if err == nil {
				ffnCollapsed = ffnOwned.rows
			}
		} else {
			ffnCollapsed, err = in.resident.ffnInput(l, in.attnOut, in.hcIters, in.hcEps, in.eps)
		}
		if err != nil {
			return panel, err
		}
	}
	panelDone := false
	if ffnOwned != nil {
		var normalized [][]float32
		normalized, panelDone, err = scratch.ffnNorm.panel(m, in.st, scratch, in.resident, l, ffnOwned, in.eps)
		if err != nil {
			return panel, err
		}
		if panelDone {
			copy(panel.inputs, normalized)
		}
	}
	for t := 0; !panelDone && t < seq; t++ {
		var xn []float32
		if in.resident != nil {
			xn, err = m.v41FFNNormWithPublish(l, ffnCollapsed[t], in.eps, scratch.ffnNorm.normFunc(), scratch.gatherOps.bf16PublishFunc())
		} else if in.full {
			panel.residual[t], panel.mixes[t], xn, err = m.v41FullFFNInput(l, in.streams[t], in.hcByPos[t], in.attnOut[t], projectFFN, scratch)
		} else {
			xn, err = m.v41FFNNormWithPublish(l, in.x[t], in.eps, scratch.ffnNorm.normFunc(), scratch.gatherOps.bf16PublishFunc())
		}
		if err != nil {
			return panel, err
		}
		if len(xn) != H {
			return panel, v41StageErr(v41StageMoE, l, fmt.Errorf("%w: invalid normalized FFN row width", ErrV41ForwardStage))
		}
		panel.inputs[t] = xn
	}
	ffnElements, ok := checkedMulInt(seq, H)
	if !ok {
		return panel, v41StageErr(v41StageMoE, l, fmt.Errorf("%w: FFN panel dimensions overflow", ErrV41ForwardStage))
	}
	if seq == 1 {
		panel.flat = panel.inputs[0]
	} else {
		panel.flat = make([]float32, ffnElements)
		for t, xn := range panel.inputs {
			copy(panel.flat[t*H:(t+1)*H], xn)
		}
	}
	return panel, nil
}

// v41LayerRoute projects the normalized panel through the router and routes
// each position in order.
func (m *Model) v41LayerRoute(in v41LayerMoEInput, panel v41LayerFFNPanel) ([][]routePick, error) {
	cfg := m.Cfg
	l, H, scratch := in.l, in.H, in.scratch
	seq := len(in.x)
	perTokenPicks := make([][]routePick, seq)
	routerPanel, err := m.v41ProjPanelWithProjection(l, "ffn.gate.weight", panel.flat, cfg.NumExperts, H, seq, scratch.denseProjection())
	if err != nil {
		return nil, err
	}
	var panelPicks [][]routePick
	panelRouted := false
	for t := 0; t < seq; t++ {
		// Keep the first successful route/emission at its original boundary,
		// including lazy process trace initialization and observer side effects.
		// Only the remaining rows may share selected router completions.
		if t == 1 {
			panelPicks, panelRouted, err = m.v41ColdRouterPanel(l, routerPanel[cfg.NumExperts:], in.gateBias, seq-1, in.routeCfg, in.full, scratch)
			if err != nil {
				return nil, v41StageErr(v41StageMoE, l, err)
			}
		}
		var picks []routePick
		if panelRouted {
			picks = panelPicks[t-1]
		} else {
			routerLogits := routerPanel[t*cfg.NumExperts : (t+1)*cfg.NumExperts]
			picks, err = v41RouteForGeometryWith(l, routerLogits, in.gateBias, in.routeCfg, in.full, scratch.routerTopK())
			if err != nil {
				return nil, v41StageErr(v41StageMoE, l, err)
			}
		}
		m.emitExpertRouteAt(l, t, picks)
		perTokenPicks[t] = picks
	}
	return perTokenPicks, nil
}

// v41LayerRoutedContraction contracts every routed pick into a per-token sum.
//
// #13304: a multi-token prefill panel is contracted EXPERT-MAJOR when its
// routed union exceeds the layer cache. The grouped path materializes one
// expert triple, contracts every (token, slot) row assigned to it, releases
// it, then replays each token's weighted sum in original slot order -- so a
// panel whose alternating routes re-read the same experts faults each distinct
// projection once, and peak retained expert materialization stays one triple.
// When the union fits the layer cache the two paths are equivalent; the
// single-token path keeps the historical token-major stream byte-for-byte.
func (m *Model) v41LayerRoutedContraction(in v41LayerMoEInput, ffnInputs [][]float32, perTokenPicks [][]routePick) ([][]float32, error) {
	cfg := m.Cfg
	l, H, st, scratch := in.l, in.H, in.st, in.scratch
	seq := len(in.x)
	if m.v41EPRequested() {
		if !in.full || st == nil || st.routedPrefill == nil {
			return nil, fmt.Errorf("%w: missing Session routed-prefill binding", ErrV41ExpertParallel)
		}
		routedByToken, err := st.routedPrefill(l, perTokenPicks, ffnInputs, scratch)
		if err != nil {
			return nil, err
		}
		if len(routedByToken) != seq {
			return nil, fmt.Errorf("%w: invalid distributed prefill panel", ErrV41ExpertParallel)
		}
		for _, values := range routedByToken {
			if len(values) != H {
				return nil, fmt.Errorf("%w: invalid distributed prefill row", ErrV41ExpertParallel)
			}
		}
		return routedByToken, nil
	}
	if seq > 1 && !v41ForceTokenMajor {
		// #13511: thread the session's optional device gate/up callback into the
		// grouped (expert-major) contraction too, so a multi-token prefill offers
		// each routed row to the same device seam the token-major arm uses
		// (#13358). A nil callback (Model.Forward, or no device backend) keeps the
		// grouped path byte-for-byte.
		return m.v41ContractRoutedGroupedFresh(l, ffnInputs, perTokenPicks, scratch, cfg, st)
	}
	routedByToken := make([][]float32, seq)
	// The multi-token token-major arm (test-forced, or a panel that the grouped
	// path did not take) contracts each pick through the row-parallel twin; the
	// seq == 1 decode fallback keeps the package-level serial contraction,
	// because three tiny expert GEMVs per pick would pay the parFor dispatch
	// barrier with no bandwidth to reclaim.
	contract := v41SwiGLU
	if seq > 1 {
		contract = v41SwiGLUParallel
	}
	for t := 0; t < seq; t++ {
		xn := ffnInputs[t]
		if in.full {
			sum, err := m.v41LayerRoutedTokenFull(in, xn, perTokenPicks[t], seq)
			if err != nil {
				return nil, err
			}
			routedByToken[t] = sum
			continue
		}
		routed := make([]float32, H)
		for _, pick := range perTokenPicks[t] {
			if err := m.v41LayerRoutedPickHost(in, xn, pick, routed, contract); err != nil {
				return nil, err
			}
		}
		routedByToken[t] = routed
	}
	return routedByToken, nil
}

func (m *Model) v41LayerRoutedTokenFull(in v41LayerMoEInput, xn []float32, picks []routePick, seq int) ([]float32, error) {
	cfg := m.Cfg
	l, st, scratch := in.l, in.st, in.scratch
	outputs := make([][]float32, len(picks))
	var routedExpert v41RoutedExpertFunc
	var gateUp v41ExpertGateUpFunc
	var down v41ExpertDownFunc
	if st != nil {
		routedExpert, gateUp, down = st.routedExpert, st.expertGateUp, st.expertDown
	}
	var err error
	for slot, pick := range picks {
		stem := "ffn.experts." + itoa(pick.expert)
		outputs[slot], err = m.v41FullRoutedExpertPublished(l, stem, xn, pick.weight, cfg, routedExpert, gateUp, down,
			func() ([]float32, []float32, []float32, error) { return m.v41ExpertTripleInto(l, stem, scratch, true) },
			func() ([]float32, error) { return m.hostExpertDown(l, stem, scratch) }, seq > 1, scratch.gatherOps.bf16PublishFunc())
		if err != nil {
			return nil, err
		}
	}
	return v41FullRoutedSumWith(l, picks, outputs, in.H, scratch.routedSumFunc())
}

// v41LayerRoutedPickHost accumulates one reduced-path pick into routed.
//
// #13358: the pick is offered to the session's optional device gate/up seam
// first. A handled result returns the I-wide fused intermediate from the
// backend, so the host down contraction runs over it WITHOUT materializing the
// gate/up f32 weights. A decline (or no callback) keeps the historical host
// triple byte-for-byte. A selected device failure must surface.
func (m *Model) v41LayerRoutedPickHost(in v41LayerMoEInput, xn []float32, pick routePick, routed []float32, contract func(w1, w3, w2, x []float32, inter, dim int, cfg Config) []float32) error {
	cfg := m.Cfg
	l, H, st, scratch := in.l, in.H, in.st, in.scratch
	stem := "ffn.experts." + itoa(pick.expert)
	if st != nil && st.expertGateUp != nil {
		h, outcome, gerr := st.expertGateUp(l, stem, xn)
		switch outcome {
		case v41GateUpError:
			return v41StageErr(v41StageMoE, l, gerr)
		case v41GateUpHandled:
			// #13704: offer the I-wide intermediate to the device down seam. When
			// it handles, all three projections ran on the backend and no host
			// expert GEMM remains; a decline keeps the host f32 down contraction.
			if st.expertDown != nil {
				yd, dOutcome, derr := st.expertDown(l, stem, h)
				switch dOutcome {
				case v41DownError:
					return v41StageErr(v41StageMoE, l, derr)
				case v41DownHandled:
					contractOpen := m.v41NowNanos()
					if contractOpen != 0 {
						m.v41NoteExpertContractionNanos(m.v41NowNanos() - contractOpen)
					} else {
						m.v41NoteExpertContraction()
					}
					_ = ffn.AddScaled(routed, yd, pick.weight)
					return nil
				}
			}
			w2, err := m.hostExpertDown(l, stem, scratch)
			if err != nil {
				return err
			}
			contractOpen := m.v41NowNanos()
			y := matRows(w2, h, H, cfg.MoEIntermediateSize)
			if contractOpen != 0 {
				m.v41NoteExpertContractionNanos(m.v41NowNanos() - contractOpen)
			} else {
				m.v41NoteExpertContraction()
			}
			_ = ffn.AddScaled(routed, y, pick.weight)
			return nil
		}
	}
	w1, w3, w2, err := m.v41ExpertTripleInto(l, stem, scratch, true)
	if err != nil {
		return err
	}
	// #13299: time the routed-expert contraction separately from the
	// fault/dequant that produced its weights, so the ledger can attribute the
	// 492 s first token (fak#13294) to scalar contraction vs tier IO. Inert with
	// no clock.
	contractOpen := m.v41NowNanos()
	y := contract(w1, w3, w2, xn, cfg.MoEIntermediateSize, H, cfg)
	if contractOpen != 0 {
		m.v41NoteExpertContractionNanos(m.v41NowNanos() - contractOpen)
	} else {
		m.v41NoteExpertContraction()
	}
	_ = ffn.AddScaled(routed, y, pick.weight)
	return nil
}

// v41LayerSharedExpertAndPost adds the shared expert to each routed sum, runs
// the mHC post-mix, and publishes the next streams once every row succeeded.
func (m *Model) v41LayerSharedExpertAndPost(in v41LayerMoEInput, panel v41LayerFFNPanel, routedByToken [][]float32) error {
	cfg := m.Cfg
	l, H, st, scratch, resident := in.l, in.H, in.st, in.scratch, in.resident
	seq := len(in.x)
	var nextPanel [][][]float32
	if in.full && resident == nil {
		nextPanel = make([][][]float32, seq)
	}
	var moeRows [][]float32
	if resident != nil {
		moeRows = make([][]float32, seq)
	}
	var err error
	var sharedChain v41ColdSharedResidentFunc
	var sharedPanel []float32
	var sharedPanelSelected bool
	var sharedPanelFault *v41SharedPanelDeferredFault
	if seq > 1 && in.full && st != nil && st.callbackOwner != nil && st.callbackOwner.M == m &&
		scratch.denseProjection() != nil && scratch.sharedActivation != nil && scratch.gatherOps.bf16PublishFunc() != nil {
		sharedChain = st.callbackOwner.v41ColdSharedExpertChain(l)
		sharedPanel, sharedPanelSelected, err = st.callbackOwner.v41ColdSharedExpertPanel(l)(panel.flat, seq)
		if err != nil {
			if !errors.As(err, &sharedPanelFault) {
				return err
			}
			err = nil
		}
	}
	for t := 0; t < seq; t++ {
		if sharedPanelFault != nil && t == sharedPanelFault.Row {
			return sharedPanelFault.finalize()
		}
		xn := panel.inputs[t]
		routed := routedByToken[t]
		var shared []float32
		var sharedResident bool
		if sharedPanelSelected {
			shared = sharedPanel[t*H : (t+1)*H]
			sharedResident = true
		} else if sharedChain != nil {
			shared, sharedResident, err = sharedChain(xn)
		}
		if err == nil && !sharedResident {
			shared, err = m.v41SharedExpertSwiGLUWithPublish(l, xn, cfg, scratch.denseProjection(), scratch.sharedActivation, scratch.gatherOps.bf16PublishFunc())
		}
		if err != nil {
			return err
		}
		moe, err := v41SharedExpertAddWith(l, routed, shared, in.routeCfg, scratch.sharedAdd)
		if err != nil {
			return v41StageErr(v41StageMoE, l, err)
		}
		// #13325 moe_sum: captured BEFORE the delta/residual mixing so an
		// expert-contraction fault is distinguishable from an mHC post-mix fault.
		in.trace.record(l, t, v41TraceStageMoESum, moe)

		if resident != nil {
			moeRows[t] = moe
		} else if in.full {
			nextPanel[t], err = v41MHCPostBF16On(in.mhcDevice, l, moe, panel.residual[t], panel.mixes[t])
			if err != nil {
				return err
			}
		} else {
			delta := make([]float32, H)
			for i := range delta {
				delta[i] = in.attnOut[t][i] + moe[i]
			}
			pre := in.preByPos[t]
			residual := [][]float32{pre, pre, pre, pre}
			next, err := v41MHCPost(delta, residual, in.hcByPos[t].post, in.hcByPos[t].comb)
			if err != nil {
				return v41StageErr(v41StageMHC, l, err)
			}
			copy(in.x[t], next[0])
		}
	}
	if resident != nil {
		// Every token row succeeded: publish the pending panel and its carry.
		if err := resident.finishLayer(l, moeRows); err != nil {
			return err
		}
		if err := resident.publish(l); err != nil {
			return err
		}
		scratch.mhcCarry.nextLayer++
		return nil
	}
	if in.full {
		// All token rows succeeded: stream and carry publication has no failing work.
		for t := range nextPanel {
			for h := 0; h < 4; h++ {
				copy(in.streams[t][h], nextPanel[t][h])
			}
			copy(in.x[t], nextPanel[t][0])
			scratch.mhcCarry.pre[t] = panel.mixes[t].pre
		}
		scratch.mhcCarry.nextLayer++
	}
	return nil
}
