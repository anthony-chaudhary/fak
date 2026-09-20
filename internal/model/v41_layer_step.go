package model

// v41_layer_step.go — the fak#13306 single-position plain-layer seam for the
// V4.1 native decode path (halo-ds41-100-30 packet 07).
//
// v41Layer (v41_forward.go) is a FULL-SEQUENCE assembly: it projects every
// prompt position, seeds per-layer temporal state, and contracts attention over
// the whole panel. That is exactly what a prefill needs, and exactly what a
// decode step must not do -- re-running a seq-token panel to produce one token
// re-projections and re-contracts the entire history (the 100/30 plan's
// "Session decode recomputes all history" finding).
//
// v41LayerStep is the one-position counterpart for PLAIN, UNCOMPRESSED layer
// roles. It accepts the current position's hidden row, the four mHC streams, the
// absolute position and that layer's retained V41AttentionState, and advances
// the layer by exactly one row. It introduces NO new arithmetic: the mHC split,
// pre-collapse, attention contraction, grouped output projection, router picks,
// routed/shared expert SwiGLU and post-mix are the same functions v41Layer calls
// with seq == 1. The only structural difference is the attention KV row set:
// v41Layer builds it from the in-panel `kvRows` (and leaf 13's
// v41PlainWindowKeys window); a step builds it from the layer's RETAINED rows
// (seeded by leaf 06's seedTemporal, windowed by leaf 13's semantics) plus the
// one newly projected row. For a LOCAL (positive-window) layer the retained
// prefix always covers every visible key, so the ordered key set and therefore
// the score/softmax/value math is identical; a GLOBAL (non-positive-window)
// layer is bounded by the fixed 128-row retained ring and is refused once the
// prefix no longer fits it (see below), so no truncated-context logits escape.
//
// Fail-closed contract (never a silent fall-through):
//   - a compressed, source or reader role is refused BEFORE any mutation, with
//     the typed ErrV41ForwardStage, because their KV stream is not the per-layer
//     window this seam contracts;
//   - a non-append position (pos != state.nextWindowPos) is refused, so a caller
//     cannot silently skip or replay history;
//   - a GLOBAL layer whose full causal prefix no longer fits the retained ring
//     is refused before the KV row is committed, because contracting only the
//     surviving tail would emit logits from a truncated context the checkpoint
//     never declared;
//   - the state is advanced ONLY after the layer arithmetic succeeded, via the
//     atomic V41AttentionState.Step, so a failed step leaves the retained state
//     untouched.
//
// Session.Step is deliberately NOT switched here (leaf 10 owns activation); this
// file adds the seam and its parity witness only.

import "fmt"

// maxInt is used by the attention option builders below; it is defined in
// v41_forward.go for the full-sequence path and reused here so the two call
// sites cannot drift.

// v41LayerStep advances one plain layer by exactly one position. x is the SINGLE
// current hidden row (length cfg.HiddenSize) and is updated in place. streams is
// the position's four persistent mHC streams and is updated in place. pos is the
// absolute position of the row (0-based). layerState is the layer's retained
// temporal state, seeded from the prefix by leaf 06 and advanced here by one row.
//
// It returns the typed ErrV41ForwardStage for an unsupported role, an
// out-of-order position, or a malformed geometry. State is committed only after
// the full layer arithmetic succeeds.
func (m *Model) v41LayerStep(l int, x []float32, streams [][]float32, pos int, layerState *V41AttentionState, scratch *v41ProjScratch) error {
	cfg := m.Cfg
	if scratch == nil {
		scratch = &v41ProjScratch{}
	}
	if len(x) != cfg.HiddenSize {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: layer step hidden row has %d values, want %d", ErrV41ForwardStage, len(x), cfg.HiddenSize))
	}
	if len(streams) != 4 {
		return v41StageErr(v41StageMHC, l,
			fmt.Errorf("%w: layer step carries %d mHC streams, want 4", ErrV41ForwardStage, len(streams)))
	}
	for s := range streams {
		if len(streams[s]) != cfg.HiddenSize {
			return v41StageErr(v41StageMHC, l,
				fmt.Errorf("%w: layer step stream %d has %d values, want %d", ErrV41ForwardStage, s, len(streams[s]), cfg.HiddenSize))
		}
	}

	// Resolve the layer's execution plan and refuse everything that is not a
	// plain per-layer contraction BEFORE touching any state. A compressed ratio
	// (CED/CSA2 pooling) or a shared-source/readere role contracts a different KV
	// stream; silently running the per-layer window path for those would emit
	// logits for a schedule the checkpoint never declared.
	plan, err := v41AttentionPlanFor(cfg, l, m.v41AttentionRolesCached())
	if err != nil {
		return err
	}
	if plan.Role != V41AttentionRolePerLayer || plan.Ratio > 1 {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: layer %d step requires a plain uncompressed role, got role=%d ratio=%d",
				ErrV41ForwardStage, l, plan.Role, plan.Ratio))
	}
	if layerState == nil {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: layer %d step requires retained decode state", ErrV41ForwardStage, l))
	}
	// Append-only: the step position must be the next retained position. This
	// refuses a caller that skipped or replayed history instead of silently
	// producing a token from a wrong causal context.
	if pos != layerState.nextWindowPos {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: layer %d step position %d is not the next retained position %d",
				ErrV41ForwardStage, l, pos, layerState.nextWindowPos))
	}

	H, hd, nH := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads
	eps := float32(cfg.RMSNormEps)
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		return err
	}

	attnNorm := m.tensor(layerName(l, "attn_norm.weight"))
	ffnNorm := m.tensor(layerName(l, "ffn_norm.weight"))
	wMix, err := m.v41MHCMixF32Into(l, scratch.mhc)
	if err != nil {
		return err
	}
	scratch.mhc = wMix
	mixBase := m.tensor(layerName(l, "mhc.base"))
	mixScale := m.tensor(layerName(l, "mhc.scale"))
	woA, err := m.v41ProjF32Into(l, "attn.wo_a.weight", scratch.woA)
	if err != nil {
		return err
	}
	scratch.woA = woA
	woB, err := m.v41ProjF32Into(l, "attn.wo_b.weight", scratch.woB)
	if err != nil {
		return err
	}
	scratch.woB = woB
	for _, leaf := range []string{
		"attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight",
		"ffn.gate.weight",
		"ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight",
		"ffn.shared_experts.w2.weight",
	} {
		if !m.hasResidentWeight(layerName(l, leaf)) {
			return v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, layerName(l, leaf)))
		}
	}
	sink := m.tensor(layerName(l, "attn.sink"))
	gateBias := m.tensor(layerName(l, "ffn.gate.e_score_correction_bias"))

	// ---- mHC coefficient split (one position) ----
	mhcFlat, mhcTransposed, mhcOK := m.v41MHCWeightLayout(l)
	if !mhcOK {
		return v41StageErr(v41StageMHC, l,
			fmt.Errorf("%w: mHC mix weight holds no admitted geometry", ErrV41ForwardStage))
	}
	xn := rmsnormCfg(x, attnNorm, eps, cfg)
	var mixes []float32
	if mhcFlat {
		if mixes, err = v41MHCProjectFull(wMix, streams, H, eps, mhcTransposed); err != nil {
			return v41StageErr(v41StageMHC, l, err)
		}
	} else {
		mixes = matRows(wMix, xn, v41MHCMixWidth, H)
	}
	mix, err := v41MHCSplit(mixes, mixScale, mixBase, 4, hcItersOrDefault(cfg), hcEpsOrDefault(cfg))
	if err != nil {
		return v41StageErr(v41StageMHC, l, err)
	}
	// The pre-collapse reads the four DISTINCT persistent streams on the full
	// path and the reduced stand-in (four identical copies of the normalized
	// input) on the reduced path -- byte-identical to v41Layer's seq==1 branch.
	streams4 := streams
	if !full {
		streams4 = [][]float32{xn, xn, xn, xn}
	}
	collapsed, err := v41MHCPre(streams4, mix.pre)
	if err != nil {
		return v41StageErr(v41StageMHC, l, err)
	}

	// ---- attention for the one position ----
	ropeDim := cfg.QKRopeHeadDim
	if ropeDim <= 0 || ropeDim%2 != 0 || ropeDim > hd {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: attention qk_rope_head_dim must be a positive even value <= head_dim %d, got %d", ErrV41ForwardStage, hd, ropeDim))
	}
	qLat, err := m.v41ProjMatRows(l, "attn.wq_a.weight", collapsed, cfg.QLoraRank, H)
	if err != nil {
		return err
	}
	if full {
		qNorm := m.tensor(layerName(l, "attn.wq_a_norm.weight"))
		if len(qNorm) != cfg.QLoraRank {
			return v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: q-lora norm has %d values, want %d", ErrV41ForwardStage, len(qNorm), cfg.QLoraRank))
		}
		copy(qLat, rmsnormCfg(qLat, qNorm, eps, cfg))
	}
	q, err := m.v41ProjMatRows(l, "attn.wq_b.weight", qLat, nH*hd, cfg.QLoraRank)
	if err != nil {
		return err
	}
	var kv []float32
	if full {
		if hd > v41KVLoraRank {
			return v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: attention head_dim %d exceeds full KV latent rank %d", ErrV41ForwardStage, hd, v41KVLoraRank))
		}
		kvFull, err := m.v41ProjMatRows(l, "attn.wkv.weight", collapsed, v41KVLoraRank, H)
		if err != nil {
			return err
		}
		kv = kvFull[:hd]
	} else {
		kv, err = m.v41ProjMatRows(l, "attn.wkv.weight", collapsed, hd, H)
		if err != nil {
			return err
		}
	}
	if full {
		kvNorm := m.tensor(layerName(l, "attn.kv_norm.weight"))
		if len(kvNorm) != v41KVLoraRank {
			return v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: kv norm has %d values, want %d", ErrV41ForwardStage, len(kvNorm), v41KVLoraRank))
		}
		kv = rmsnormCfg(kv, kvNorm, eps, cfg)
	}
	cos, sin := v41RopeTableForLayer(cfg, l, pos)
	for h := 0; h < nH; h++ {
		applyRopeTailInterleaved(q[h*hd:(h+1)*hd], cos, sin, ropeDim)
	}
	applyRopeTailInterleaved(kv, cos, sin, ropeDim)

	// The ordered absolute key-row IDs this position attends under leaf 13's
	// configured-window semantics, intersected with what the retained ring
	// actually holds. The ring is a fixed `windowSize`-row circular buffer and
	// `nextWindowPos` is the count of appended rows, so the populated tail after
	// any number of Prefill/Step calls is exactly min(nextWindowPos, windowSize)
	// rows ending at nextWindowPos. That tail is derived HERE from the ring
	// occupancy rather than from V41AttentionState.retainedWindowRows (which
	// seedTemporal sets but the incremental Step does not grow), so a
	// multi-step decode sees the correct retained prefix.
	windowKeys := v41PlainWindowKeys(pos, cfg.windowForLayer(l))
	retained := layerState.retainedTailRows()
	// Fail closed whenever the configured visible key set is not fully retained.
	// That covers a GLOBAL (non-positive-window) layer whose causal prefix has
	// outgrown the fixed ring, AND a LOCAL layer configured with a window wider
	// than the ring. In either case the oldest visible keys are gone, and
	// silently contracting the surviving tail would emit logits from a truncated
	// context the checkpoint never declared. Refuse before the KV row is
	// committed. `windowKeys` is [lo..pos], so its oldest key is windowKeys[0];
	// if that key predates the retained tail's first row the window is not
	// covered.
	if len(windowKeys) > 0 {
		retainedLoKey := pos - len(retained)
		if firstVisible := int(windowKeys[0]); firstVisible < retainedLoKey {
			return v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: layer %d cannot step position %d from a %d-row retained ring: visible keys [%d,%d] are not fully retained (oldest %d < %d)",
					ErrV41ForwardStage, l, pos, len(retained), firstVisible, int(windowKeys[len(windowKeys)-1]), firstVisible, retainedLoKey))
		}
	}
	// Build the ordered visible key set: because pos is append-only, retained
	// rows are exactly the suffix of windowKeys already available, in ascending
	// order, followed by the current position.
	visibleKeys := make([]int32, 0, len(windowKeys))
	retainedLo := pos - len(retained)
	for _, k := range windowKeys {
		if int(k) < pos {
			if int(k) < retainedLo {
				continue
			}
			visibleKeys = append(visibleKeys, k)
		} else {
			// pos itself, appended below.
			visibleKeys = append(visibleKeys, int32(pos))
		}
	}
	if len(visibleKeys) == 0 {
		// Defensive: a non-negative pos always yields at least itself.
		visibleKeys = append(visibleKeys, int32(pos))
	}

	// Assemble the ordered KV row set: retained rows for the visible prefix
	// (oldest first) followed by the newly projected row. This reproduces the
	// relative order a full forward's `kvRows` window would present.
	flatKV := make([]float32, 0, len(visibleKeys)*hd)
	for _, k := range visibleKeys {
		if int(k) == pos {
			flatKV = append(flatKV, kv...)
			continue
		}
		rowIdx := int(k) - retainedLo
		if rowIdx < 0 || rowIdx >= len(retained) {
			return v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: layer %d step key %d is outside retained window [%d,%d)",
					ErrV41ForwardStage, l, k, retainedLo, pos))
		}
		flatKV = append(flatKV, retained[rowIdx]...)
	}
	idx := v41PlainWindowIndexList(visibleKeys)
	scale := cfg.attnScale()
	o, err := V41SparseAttentionSink(q, flatKV, sink, idx, V41SparseAttentionSinkOptions{
		B: 1, M: 1, Heads: nH, HeadDim: hd, TopK: len(visibleKeys) + 1, N: len(visibleKeys), Softmax: scale,
	})
	if err != nil {
		return v41StageErr(v41StageAttention, l, err)
	}
	attnOut, err := V41GroupedOutputProjection(o, woA, woB, 1, 1, nH, hd, cfg.OGroups, cfg.OLoraRank, H)
	if err != nil {
		return v41StageErr(v41StageAttention, l, err)
	}

	// ---- MoE for the one position ----
	routeCfg, err := v41RouterConfigFullGeometry(cfg)
	if err != nil {
		return err
	}
	ffnX := rmsnormCfg(x, ffnNorm, eps, cfg)
	routerLogits, err := m.v41ProjMatRows(l, "ffn.gate.weight", ffnX, cfg.NumExperts, H)
	if err != nil {
		return err
	}
	picks, err := v41Route(routerLogits, gateBias, routeCfg)
	if err != nil {
		return v41StageErr(v41StageMoE, l, err)
	}
	routed := make([]float32, H)
	for _, pick := range picks {
		stem := "ffn.experts." + itoa(pick.expert)
		w1, w3, w2, err := m.v41ExpertTripleInto(l, stem, scratch)
		if err != nil {
			return err
		}
		contractOpen := m.v41NowNanos()
		y := v41SwiGLU(w1, w3, w2, ffnX, cfg.MoEIntermediateSize, H, cfg)
		if contractOpen != 0 {
			m.v41NoteExpertContractionNanos(m.v41NowNanos() - contractOpen)
		} else {
			m.v41NoteExpertContraction()
		}
		for i := range routed {
			routed[i] += pick.weight * y[i]
		}
	}
	shared, err := m.v41SharedExpertSwiGLU(l, ffnX, cfg)
	if err != nil {
		return err
	}
	moe, err := v41SharedExpertAdd(routed, shared, routeCfg)
	if err != nil {
		return v41StageErr(v41StageMoE, l, err)
	}
	delta := make([]float32, H)
	for i := 0; i < H; i++ {
		delta[i] = attnOut[i] + moe[i]
	}
	residual := [][]float32{collapsed, collapsed, collapsed, collapsed}
	if full {
		residual = streams
	}
	next, err := v41MHCPost(delta, residual, mix.post, mix.comb)
	if err != nil {
		return v41StageErr(v41StageMHC, l, err)
	}

	// Commit the position: append the one projected KV row to the retained
	// temporal state. At position 0 there is no prefix to step from, so the
	// first row seeds the empty state through Prefill; every later position
	// appends through Step. Both are atomic, so a failure here leaves the state
	// unchanged and the caller's x/streams writes below are not reached.
	var commitErr error
	if pos == 0 {
		commitErr = layerState.Prefill([][]float32{kv}, nil)
	} else {
		commitErr = layerState.Step(kv, nil)
	}
	if commitErr != nil {
		return v41StageErr(v41StageAttention, l, commitErr)
	}
	if full {
		for h := 0; h < 4; h++ {
			copy(streams[h], next[h])
		}
		copy(x, next[0])
	} else {
		copy(x, next[0])
	}
	return nil
}

// hcItersOrDefault / hcEpsOrDefault mirror v41Layer's per-forward mHC Sinkhorn
// parameters so a step and a full forward resolve the same constants from the
// same config.
func hcItersOrDefault(cfg Config) int {
	if cfg.DeepSeekV41 != nil && cfg.DeepSeekV41.HCSinkhornIters > 0 {
		return cfg.DeepSeekV41.HCSinkhornIters
	}
	return 1
}

// retainedTailRows returns the populated rows of the attention ring in
// oldest-first order, derived from the ring's occupancy. V41AttentionState
// exposes retainedWindowKV(), but that reads its retainedWindowRows counter,
// which seedTemporal sets once and the incremental Step never grows -- so after
// a decode step it under-reports the tail. The ring geometry is authoritative:
// `nextWindowPos` rows have been appended into a `windowSize`-row circular
// buffer, so exactly min(nextWindowPos, windowSize) rows are populated, at
// absolute row IDs [nextWindowPos-tail, nextWindowPos). Reading the ring here
// keeps the step's view of history current without mutating the state. Rows are
// returned as inspection copies (matching retainedWindowKV), so a caller cannot
// alias and corrupt the live ring.
func (s *V41AttentionState) retainedTailRows() [][]float32 {
	if s == nil || s.nextWindowPos == 0 || s.windowSize <= 0 {
		return nil
	}
	tail := s.nextWindowPos
	if tail > s.windowSize {
		tail = s.windowSize
	}
	start := s.nextWindowPos - tail
	out := make([][]float32, tail)
	for i := 0; i < tail; i++ {
		out[i] = append([]float32(nil), s.window[(start+i)%s.windowSize]...)
	}
	return out
}

func hcEpsOrDefault(cfg Config) float32 {
	if cfg.DeepSeekV41 != nil && cfg.DeepSeekV41.HCEps > 0 {
		return float32(cfg.DeepSeekV41.HCEps)
	}
	return float32(1e-6)
}
