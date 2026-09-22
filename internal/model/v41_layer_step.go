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
	return m.v41LayerStepWithRegistry(l, x, streams, pos, layerState, nil, scratch)
}

// v41LayerStepWithRegistry is v41LayerStep plus the session-owned shared source
// registry a compressed/source/reader role layer publishes into and resolves
// from (#13480). A nil registry preserves the historical plain-layer-only
// contract: the role composition refuses, so a caller that does not carry the
// shared state can never step a role layer.
func (m *Model) v41LayerStepWithRegistry(l int, x []float32, streams [][]float32, pos int, layerState, registry *V41AttentionState, scratch *v41ProjScratch) error {
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
	if layerState == nil {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: layer %d step requires retained decode state", ErrV41ForwardStage, l))
	}
	// A compressed / shared-source / reader role contracts the shared compressed
	// stream rather than this layer's own per-position window. It is stepped by a
	// separate one-position composition (#13480) so the per-layer window path
	// below stays byte-for-byte the plain-layer arithmetic: silently running the
	// window contraction for a role layer would emit logits from a schedule the
	// checkpoint never declared. The role step still receives the layer's own
	// retained state (its compressor group cursor) and returns the same typed
	// ErrV41ForwardStage on any refusal, so a failed step leaves state untouched.
	if plan.Role != V41AttentionRolePerLayer || plan.Ratio > 1 {
		return m.v41LayerStepRole(l, plan, x, streams, pos, layerState, registry, scratch)
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

// v41LayerStepRole advances ONE position of a compressed / shared-source / reader
// V4.1 layer through the retained compressed stream instead of the layer's own
// per-position window (#13480). It is the role-aware counterpart of v41LayerStep:
// the mHC split/pre-collapse, the query projection, the MoE block and the mHC
// post are the SAME arithmetic v41LayerStep's plain branch runs for seq == 1, so
// the only structural difference is the attention KV row set and its commit.
//
// Every arithmetic path mirrors v41Layer's compressed branch (v41_forward.go
// #13006/#12896) at seq == 1:
//   - a ratio > 1 (source) layer projects its one pre-attention carrier through
//     attn.compressor.wkv/wgate and appends it to the retained compressor group;
//     when the append completes a group the pooled row is published to the
//     session registry (appendCompressorSource) exactly as the full path does;
//   - an index source projects its emitted row through the index weight once per
//     completed group (appendCompressorSourceIndex) and stages its per-position
//     top-k selection so a later reader can reuse it;
//   - a reader layer resolves its source's published compressed rows via
//     KVSourceRows and reuses the source's published top-k exactly as v41Layer's
//     reader branch does.
//
// The retained stream is the shared state's, not this layer's window ring, so the
// step never touches the per-layer window cursor. The step is append-only and
// atomic: a typed ErrV41ForwardStage from the shared append leaves the compressor
// group, the publication registries and the layer's retained state unchanged, and
// only a fully successful step commits the mHC streams and the new hidden row.
func (m *Model) v41LayerStepRole(l int, plan V41AttentionPlan, x []float32, streams [][]float32, pos int, layerState, registry *V41AttentionState, scratch *v41ProjScratch) error {
	cfg := m.Cfg
	if len(x) != cfg.HiddenSize {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: layer role step hidden row has %d values, want %d", ErrV41ForwardStage, len(x), cfg.HiddenSize))
	}
	if len(streams) != 4 {
		return v41StageErr(v41StageMHC, l,
			fmt.Errorf("%w: layer role step carries %d mHC streams, want 4", ErrV41ForwardStage, len(streams)))
	}
	for s := range streams {
		if len(streams[s]) != cfg.HiddenSize {
			return v41StageErr(v41StageMHC, l,
				fmt.Errorf("%w: layer role step stream %d has %d values, want %d", ErrV41ForwardStage, s, len(streams[s]), cfg.HiddenSize))
		}
	}
	if registry == nil {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: layer %d role step requires the shared source registry", ErrV41ForwardStage, l))
	}
	if scratch == nil {
		scratch = &v41ProjScratch{}
	}
	if pos != layerState.nextWindowPos {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: layer %d role step position %d is not the next retained position %d",
				ErrV41ForwardStage, l, pos, layerState.nextWindowPos))
	}

	H, hd, nH := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads
	eps := float32(cfg.RMSNormEps)
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		return err
	}
	if !full {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: layer %d role step requires the full V4.1 geometry", ErrV41ForwardStage, l))
	}

	attnNorm := m.tensor(layerName(l, "attn_norm.weight"))
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

	// ---- mHC coefficient split (one position), byte-identical to the plain branch ----
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
	collapsed, err := v41MHCPre(streams, mix.pre)
	if err != nil {
		return v41StageErr(v41StageMHC, l, err)
	}

	// ---- attention query for the one position (mirrors v41Layer's full branch) ----
	ropeDim := cfg.QKRopeHeadDim
	if ropeDim <= 0 || ropeDim%2 != 0 || ropeDim > hd {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: attention qk_rope_head_dim must be a positive even value <= head_dim %d, got %d", ErrV41ForwardStage, hd, ropeDim))
	}
	qLat, err := m.v41ProjMatRows(l, "attn.wq_a.weight", collapsed, cfg.QLoraRank, H)
	if err != nil {
		return err
	}
	qNorm := m.tensor(layerName(l, "attn.wq_a_norm.weight"))
	if len(qNorm) != cfg.QLoraRank {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: q-lora norm has %d values, want %d", ErrV41ForwardStage, len(qNorm), cfg.QLoraRank))
	}
	copy(qLat, rmsnormCfg(qLat, qNorm, eps, cfg))
	q, err := m.v41ProjMatRows(l, "attn.wq_b.weight", qLat, nH*hd, cfg.QLoraRank)
	if err != nil {
		return err
	}
	if hd > v41KVLoraRank {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: attention head_dim %d exceeds full KV latent rank %d", ErrV41ForwardStage, hd, v41KVLoraRank))
	}
	kvFull, err := m.v41ProjMatRows(l, "attn.wkv.weight", collapsed, v41KVLoraRank, H)
	if err != nil {
		return err
	}
	kv := kvFull[:hd]
	kvNorm := m.tensor(layerName(l, "attn.kv_norm.weight"))
	if len(kvNorm) != v41KVLoraRank {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: kv norm has %d values, want %d", ErrV41ForwardStage, len(kvNorm), v41KVLoraRank))
	}
	kv = rmsnormCfg(kv, kvNorm, eps, cfg)
	cos, sin := v41RopeTableForLayer(cfg, l, pos)
	for h := 0; h < nH; h++ {
		applyRopeTailInterleaved(q[h*hd:(h+1)*hd], cos, sin, ropeDim)
	}
	applyRopeTailInterleaved(kv, cos, sin, ropeDim)

	// ---- shared compressed stream: source publishes, reader resolves ----
	//
	// The shared registry (st.attn) is the model-scoped publication store the full
	// forward uses, while this layer's own state carries the compressor group
	// cursor. A source appends through ITS state and then mirrors the completed
	// rows into the registry so a later reader resolves them; a reader reads the
	// registry only. This mirrors v41Layer's source-then-consumer ordering (#12896)
	// at seq == 1.
	var sharedKV [][]float32
	if plan.Ratio > 1 {
		width := v41CompressorWidth(cfg)
		wkv := m.tensor(layerName(l, "attn.compressor.wkv.weight"))
		wgate := m.tensor(layerName(l, "attn.compressor.wgate.weight"))
		normWeight := m.tensor(layerName(l, "attn.compressor.norm.weight"))
		pool, perr := NewV41CompressorPool(plan.Ratio, width)
		if perr != nil {
			return v41StageErr(v41StageCompress, l, perr)
		}
		projectKV := func(in []float32) ([]float32, error) { return matRows(wkv, in, width, H), nil }
		projectScore := func(in []float32) ([]float32, error) { return matRows(wgate, in, width, H), nil }
		var indexPub bool
		var projectIndex func([]float32) ([]float32, error)
		if indexSourceAt(cfg.DeepSeekV41, l) {
			wproj := m.tensor(layerName(l, "indexer.wk.weight"))
			kNorm := m.tensor(layerName(l, "indexer.k_norm.weight"))
			indexDim := cfg.IndexHeadDim
			if indexDim <= 0 {
				return v41StageErr(v41StageIndexer, l,
					fmt.Errorf("%w: indexer geometry headDim=%d is not declared", ErrV41ForwardStage, indexDim))
			}
			indexPub = true
			projectIndex = func(row []float32) ([]float32, error) {
				projected := matRows(wproj, row, indexDim, len(row))
				if len(kNorm) == indexDim {
					projected = rmsnormCfg(projected, kNorm, eps, cfg)
				}
				return projected, nil
			}
		}
		var latent []float32
		var indexKey []float32
		var emitted bool
		var start, end int
		if indexPub {
			latent, indexKey, emitted, start, end, err = layerState.appendCompressorSourceIndex(
				l, plan.Ratio, pos, collapsed, pool, projectKV, projectScore, projectIndex, normWeight, eps)
		} else {
			latent, emitted, start, end, err = layerState.appendCompressorSource(
				l, plan.Ratio, pos, collapsed, pool, projectKV, projectScore, normWeight, eps)
		}
		if err != nil {
			return v41StageErr(v41StageCompress, l, err)
		}
		if emitted && len(latent) > 0 {
			// Mirror the completed group into the shared registry so a later
			// reader resolves it. The layer's own publication already exists on
			// its state; the registry entry is the cross-layer view.
			ref := V41AttentionStateRef{LayerID: l, Ratio: plan.Ratio, IsKVSource: true, IsIndexSource: indexPub}
			upd := V41AttentionStateUpdate{Ref: ref, Latent: latent}
			if indexPub {
				upd.IndexKey = indexKey
			}
			if err := registry.publishUpdates([]V41AttentionStateUpdate{upd}); err != nil {
				return v41StageErr(v41StageCompress, l, err)
			}
			sharedKV = [][]float32{latent}
			_ = start
			_ = end
		}
		if indexPub && emitted {
			// Publish the index source's per-position top-k so a later reader
			// reuses it without recomputing the scoring path (mirrors v41Layer's
			// index-source publication at seq == 1).
			if err := m.v41RoleStepPublishTopK(registry, l, plan, qLat, collapsed, sharedKV); err != nil {
				return err
			}
		}
		if len(sharedKV) == 0 {
			// No completed group yet: the layer contracts nothing for this position,
			// exactly as the full path's empty compressed stream does for an
			// incomplete group. The compressor cursor has advanced; finish the
			// MoE/mHC tail with a zero attention output so the position advances.
			return m.v41LayerStepRoleFinish(l, x, streams, mix, make([]float32, H), scratch)
		}
	} else {
		// Reader (or a ratio<=1 shared layer): resolve the nearest preceding
		// source's completed compressed rows from the shared registry.
		if plan.KVSourceLayer < 0 {
			return v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: layer %d role step resolves no shared KV source", ErrV41ForwardStage, l))
		}
		rows, ok := registry.KVSourceRows(plan.KVSourceLayer)
		if !ok || len(rows) == 0 {
			return v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: layer %d role step reads source %d but no compressed rows are published", ErrV41ForwardStage, l, plan.KVSourceLayer))
		}
		sharedKV = rows
	}

	// ---- compressed contraction over the shared rows (block-causal) ----
	opt := V41AttentionSharedKVOptions{
		Layer: l, Ratio: maxInt(plan.Ratio, 1), Groups: len(sharedKV),
		HeadDim: hd, Heads: nH, Softmax: cfg.attnScale(), Sink: m.tensor(layerName(l, "attn.sink")),
	}
	if plan.TopKWidth > 0 {
		idx, ierr := m.v41RoleStepIndex(registry, l, plan, qLat, collapsed, len(sharedKV))
		if ierr != nil {
			return ierr
		}
		if idx != nil {
			opt.Idx = idx
			opt.IndexTopK = plan.TopKWidth
			opt.TopK = plan.TopKWidth
		}
	}
	o, err := V41AttentionCompressedForward(q, sharedKV, opt)
	if err != nil {
		return v41StageErr(v41StageAttention, l, err)
	}
	attnProjected, err := V41GroupedOutputProjection(o, woA, woB, 1, 1, nH, hd, cfg.OGroups, cfg.OLoraRank, H)
	if err != nil {
		return v41StageErr(v41StageAttention, l, err)
	}

	return m.v41LayerStepRoleFinish(l, x, streams, mix, attnProjected, scratch)
}

// v41LayerStepRoleFinish runs the MoE block and the mHC post for one position and
// commits the position's mHC streams and hidden row. It is shared by the source
// (which may have no attention output yet) and the reader path so the tail after
// attention is byte-identical to the plain branch's seq == 1 arithmetic. The
// per-layer window cursor is deliberately NOT advanced here: the role path owns
// the shared compressed stream, not this layer's window ring.
func (m *Model) v41LayerStepRoleFinish(l int, x []float32, streams [][]float32, mix v41MHCMix, attnOut []float32, scratch *v41ProjScratch) error {
	cfg := m.Cfg
	H := cfg.HiddenSize
	eps := float32(cfg.RMSNormEps)
	ffnNorm := m.tensor(layerName(l, "ffn_norm.weight"))
	ffnX := rmsnormCfg(x, ffnNorm, eps, cfg)
	routeCfg, err := v41RouterConfigFullGeometry(cfg)
	if err != nil {
		return err
	}
	routerLogits, err := m.v41ProjMatRows(l, "ffn.gate.weight", ffnX, cfg.NumExperts, H)
	if err != nil {
		return err
	}
	gateBias := m.tensor(layerName(l, "ffn.gate.e_score_correction_bias"))
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
	next, err := v41MHCPost(delta, streams, mix.post, mix.comb)
	if err != nil {
		return v41StageErr(v41StageMHC, l, err)
	}
	for h := 0; h < 4; h++ {
		copy(streams[h], next[h])
	}
	copy(x, next[0])
	return nil
}

// v41RoleStepIndex resolves the per-position index selection a compressed
// contraction consumes. A layer that IS the index source publishes its own
// selection (computed here from the source's own completed stream) and a reader
// reuses its source's published top-k, matching v41AttentionIndexList's
// precedence at seq == 1.
func (m *Model) v41RoleStepIndex(state *V41AttentionState, l int, plan V41AttentionPlan, qLat, hidden []float32, groups int) ([]int32, error) {
	if plan.TopKWidth <= 0 || groups <= 0 {
		return nil, nil
	}
	if indexSourceAt(m.Cfg.DeepSeekV41, l) {
		rows, ok := state.KVSourceRows(l)
		if !ok || len(rows) == 0 {
			return nil, nil
		}
		idx, err := m.v41IndexRows(l, qLat, hidden, rows)
		if err != nil {
			return nil, err
		}
		return m.v41RoleStepNormalizeIndex(l, plan, idx, groups)
	}
	published, _, ok := state.TopK()
	if !ok || len(published) == 0 {
		return nil, v41StageErr(v41StageIndexer, l,
			fmt.Errorf("%w: layer %d reads index source %d but no top-k selection was published", ErrV41ForwardStage, l, plan.IndexSourceLayer))
	}
	// The newest published row is the causal superset for this position.
	return m.v41RoleStepNormalizeIndex(l, plan, published[len(published)-1], groups)
}

// v41RoleStepNormalizeIndex validates one selection row against the compressed
// stream and pads a short top-k with -1 (the reference's empty-slot marker),
// matching v41AttentionIndexList's normalization.
func (m *Model) v41RoleStepNormalizeIndex(l int, plan V41AttentionPlan, row []int32, groups int) ([]int32, error) {
	if len(row) > plan.TopKWidth {
		return nil, v41StageErr(v41StageIndexer, l,
			fmt.Errorf("%w: layer %d index row width %d exceeds stride %d", ErrV41ForwardStage, l, len(row), plan.TopKWidth))
	}
	normalized := make([]int32, plan.TopKWidth)
	for i := range normalized {
		normalized[i] = -1
	}
	for i, id := range row {
		if id < -1 || int(id) >= groups {
			return nil, v41StageErr(v41StageIndexer, l,
				fmt.Errorf("%w: layer %d index row %d out of range for %d groups", ErrV41ForwardStage, l, id, groups))
		}
		normalized[i] = id
	}
	return normalized, nil
}

// v41RoleStepPublishTopK publishes the index source's per-position top-k
// selection for a just-completed group so a later reader reuses it. It mirrors
// v41Layer's index-source publication at seq == 1: one selection row for the
// newest position, the causal superset across the stream.
func (m *Model) v41RoleStepPublishTopK(state *V41AttentionState, l int, plan V41AttentionPlan, qLat, hidden []float32, sharedKV [][]float32) error {
	if len(sharedKV) == 0 {
		return nil
	}
	idx, err := m.v41IndexRows(l, qLat, hidden, sharedKV)
	if err != nil {
		return err
	}
	if idx == nil {
		return nil
	}
	row, err := m.v41RoleStepNormalizeIndex(l, plan, idx, len(sharedKV))
	if err != nil {
		return err
	}
	return state.PublishTopK(plan.Ratio, [][]int32{row})
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

// v41IndexReaderStep resolves the ONE committed index publication a reader
// layer may consume for the group-closing absolute position pos (#13309).
//
// It is the read half of appendCompressorSourceIndex: the source seam stages a
// (KV latent, index key) pair keyed by [source layer, absolute half-open range),
// and this seam returns the index key of the publication whose range ends at
// pos+1, for the reader's RESOLVED index source. sourceLayer is the candidate
// source named by the caller; it is checked against the layer's plan rather than
// trusted, so naming another layer's stream refuses instead of silently reading
// it.
//
// Fail-closed. Missing, stale (the publication ends behind pos+1),
// duplicate/out-of-order (a hole in the contiguous set) and wrong-source
// publications each refuse with the typed ErrV41ForwardStage before any reader
// mutation, and every refusal is a pure read: no publication registry, retained
// group or position cursor is changed. A hole is detected by the same
// contiguity oracle IndexKeys applies, so the two agree on what a complete
// stream is.
//
// The returned key is a COPY: mutating it cannot alter the publication, so a
// reader may hold it while the source layer keeps appending.
func (m *Model) v41IndexReaderStep(layer, sourceLayer, pos int, state *V41AttentionState) ([]float32, error) {
	if state == nil {
		return nil, fmt.Errorf("%w: index reader layer %d has no attention state", ErrV41ForwardStage, layer)
	}
	if layer < 0 {
		return nil, fmt.Errorf("%w: index reader layer %d is negative", ErrV41ForwardStage, layer)
	}
	if sourceLayer < 0 {
		return nil, fmt.Errorf("%w: index reader layer %d names no index source", ErrV41ForwardStage, layer)
	}
	if pos < 0 {
		return nil, fmt.Errorf("%w: index reader layer %d position %d is negative", ErrV41ForwardStage, layer, pos)
	}

	// The plan is the authority on which source this layer may read. An
	// undecodable schedule refuses rather than falling back to the caller's guess.
	plan, err := m.v41AttentionPlan(layer)
	if err != nil {
		return nil, err
	}
	if plan.IndexSourceLayer < 0 {
		return nil, fmt.Errorf("%w: index reader layer %d resolves no index source", ErrV41ForwardStage, layer)
	}
	if sourceLayer != plan.IndexSourceLayer {
		return nil, fmt.Errorf("%w: index reader layer %d declares index source %d, not %d", ErrV41ForwardStage, layer, plan.IndexSourceLayer, sourceLayer)
	}

	// The position must close a group under the source's ratio: only a
	// group-closing position carries a committed row.
	ratio := plan.Ratio
	if ratio <= 1 {
		return nil, fmt.Errorf("%w: index reader layer %d declares ratio %d, not a compressed regime", ErrV41ForwardStage, layer, ratio)
	}
	if (pos+1)%ratio != 0 {
		return nil, fmt.Errorf("%w: index reader layer %d position %d does not close a ratio-%d group", ErrV41ForwardStage, layer, pos, ratio)
	}

	// The reader's own span must be covered: the source must have committed the
	// group that ends exactly at pos+1 (end index pos+1-ratio+1). A publication
	// that stops short is stale; one whose rows are not contiguous is holed.
	group := (pos + 1) / ratio
	start := group - 1
	end := group
	if end < 1 {
		return nil, fmt.Errorf("%w: index reader layer %d position %d has no preceding group", ErrV41ForwardStage, layer, pos)
	}
	publishedEnd := state.indexPublishedEnd[sourceLayer]
	if publishedEnd == 0 {
		return nil, fmt.Errorf("%w: index reader layer %d has no committed index publication from source %d", ErrV41ForwardStage, layer, sourceLayer)
	}
	if publishedEnd < end {
		return nil, fmt.Errorf("%w: index reader layer %d position %d is stale: source %d published %d row(s), need %d", ErrV41ForwardStage, layer, pos, sourceLayer, publishedEnd, end)
	}

	// Contiguity: every row in [0,end) must be present. A hole or an
	// out-of-order/duplicate staging refuses, matching IndexKeys' oracle.
	key := v41AttentionPublicationKey{sourceLayer: sourceLayer, start: start, end: end}
	row, ok := state.indexPublications[key]
	if !ok {
		return nil, fmt.Errorf("%w: index reader layer %d position %d: no publication for group [%d,%d)", ErrV41ForwardStage, layer, pos, start, end)
	}
	for i := 0; i < end; i++ {
		if _, present := state.indexPublications[v41AttentionPublicationKey{sourceLayer: sourceLayer, start: i, end: i + 1}]; !present {
			return nil, fmt.Errorf("%w: index reader layer %d position %d: publication set has a hole at row %d", ErrV41ForwardStage, layer, pos, i)
		}
	}
	if len(row) != state.headDim {
		return nil, fmt.Errorf("%w: index reader layer %d position %d: published row width %d, want %d", ErrV41ForwardStage, layer, pos, len(row), state.headDim)
	}

	// A pure read: return a copy so the caller cannot alias the publication.
	return append([]float32(nil), row...), nil
}

// v41AttentionPlan resolves one layer's attention plan from the receiver's config
// and the cached role map. It mirrors v41AttentionPlanFor's authority so a
// reader validates its source against the SAME schedule the execution path uses.
func (m *Model) v41AttentionPlan(layer int) (V41AttentionPlan, error) {
	if m == nil {
		return V41AttentionPlan{}, fmt.Errorf("%w: nil model", ErrV41ForwardStage)
	}
	if m.Cfg.DeepSeekV41 == nil {
		return V41AttentionPlan{}, fmt.Errorf("%w: layer %d has no V4.1 attention schedule", ErrV41ForwardStage, layer)
	}
	return v41AttentionPlanFor(m.Cfg, layer, m.v41AttentionRolesCached())
}

func hcEpsOrDefault(cfg Config) float32 {
	if cfg.DeepSeekV41 != nil && cfg.DeepSeekV41.HCEps > 0 {
		return float32(cfg.DeepSeekV41.HCEps)
	}
	return float32(1e-6)
}
