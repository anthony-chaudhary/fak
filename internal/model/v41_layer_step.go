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

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

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
	// Role layers combine their own window with shared compressed rows. Their
	// source append already advances the compressor cursor, so the role path
	// commits the window itself without calling State.Step a second time.
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

	if full {
		if err := scratch.mhcCarry.validate(l, 1); err != nil {
			return err
		}
		if err := m.v41AdmitMHC(l); err != nil {
			return err
		}
	}
	attnNorm := m.tensor(layerName(l, "attn_norm.weight"))
	mixBase := m.tensor(layerName(l, "mhc.base"))
	mixScale := m.tensor(layerName(l, "mhc.scale"))
	projectOutput := m.v41GroupedOutputProjector(l, nH, hd, cfg.OGroups, cfg.OLoraRank, H, scratch)
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
	input := xn
	if mhcFlat {
		width, ok := checkedMulInt(4, H)
		if !ok || len(streams) != 4 {
			return v41StageErr(v41StageMHC, l, errV41ProjectionResult)
		}
		input = make([]float32, 0, width)
		for _, stream := range streams {
			if len(stream) != H {
				return v41StageErr(v41StageMHC, l, errV41ProjectionResult)
			}
			input = append(input, stream...)
		}
	}
	projectMHC := m.v41MHCProjector(l, H, eps, mhcFlat, mhcTransposed, scratch)
	mixes, err := projectMHC(input)
	if err != nil {
		return err
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
	pre := mix.pre
	if full {
		pre = scratch.mhcCarry.pre[0]
	}
	collapsed, err := v41MHCPre(streams4, pre)
	if err != nil {
		return v41StageErr(v41StageMHC, l, err)
	}
	if full {
		collapsed, err = m.v41AttentionInputNorm(l, collapsed, eps)
		if err != nil {
			return err
		}
	}

	// ---- attention for the one position ----
	ropeDim := cfg.QKRopeHeadDim
	if ropeDim <= 0 || ropeDim%2 != 0 || ropeDim > hd {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: attention qk_rope_head_dim must be a positive even value <= head_dim %d, got %d", ErrV41ForwardStage, hd, ropeDim))
	}
	qLat, err := m.v41ProjMatRowsWithProjection(l, "attn.wq_a.weight", collapsed, cfg.QLoraRank, H, scratch.denseProjection)
	if err != nil {
		return err
	}
	if full {
		if err := m.v41QueryNormInPlace(l, qLat, eps, scratch.queryNorm); err != nil {
			return err
		}
	}
	q, err := m.v41ProjMatRowsWithProjection(l, "attn.wq_b.weight", qLat, nH*hd, cfg.QLoraRank, scratch.denseProjection)
	if err != nil {
		return err
	}
	var kv []float32
	if full {
		if hd > v41KVLoraRank {
			return v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: attention head_dim %d exceeds full KV latent rank %d", ErrV41ForwardStage, hd, v41KVLoraRank))
		}
		kvFull, err := m.v41ProjMatRowsWithProjection(l, "attn.wkv.weight", collapsed, v41KVLoraRank, H, scratch.denseProjection)
		if err != nil {
			return err
		}
		kv = kvFull[:hd]
	} else {
		kv, err = m.v41ProjMatRowsWithProjection(l, "attn.wkv.weight", collapsed, hd, H, scratch.denseProjection)
		if err != nil {
			return err
		}
	}
	if full {
		if err := m.v41KVNormInPlace(l, kv, eps, scratch.kvNorm); err != nil {
			return err
		}
	}
	cos, sin := v41RopeTableForLayer(cfg, l, pos)
	if err := v41TailRoPEInPlace(l, q, kv, cos, sin, nH, hd, ropeDim, scratch.tailRoPE); err != nil {
		return err
	}

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
	attentionOpened := m.v41NowNanos()
	o, err := v41SparseAttentionSinkWithDevice(l, q, flatKV, sink, idx, V41SparseAttentionSinkOptions{
		B: 1, M: 1, Heads: nH, HeadDim: hd, TopK: len(visibleKeys) + 1, N: len(visibleKeys), Softmax: scale,
	}, scratch.sharedAttention)
	m.v41NoteAttentionContraction(attentionOpened)
	if err != nil {
		return v41StageErr(v41StageAttention, l, err)
	}
	v41InverseAttentionOutputInPlace(cfg, l, pos, o, nH, hd)
	attnOut, err := projectOutput(o)
	if err != nil {
		return v41StageErr(v41StageAttention, l, err)
	}

	// ---- MoE for the one position ----
	routeCfg, err := v41RouterConfigFullGeometry(cfg)
	if err != nil {
		return err
	}
	var ffnX []float32
	var ffnResidual [][]float32
	var ffnMix v41MHCMix
	if full {
		projectFFN, ferr := m.v41FFNMHCProjector(l, scratch)
		if ferr != nil {
			return ferr
		}
		ffnResidual, ffnMix, ffnX, err = m.v41FullFFNInput(l, streams, mix, attnOut, projectFFN, scratch)
	} else {
		ffnX, err = m.v41FFNNorm(l, x, eps, scratch.ffnNorm)
	}
	if err != nil {
		return err
	}
	routerLogits, err := m.v41ProjMatRowsWithProjection(l, "ffn.gate.weight", ffnX, cfg.NumExperts, H, scratch.denseProjection)
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
		y, err := m.v41IncrementalExpert(l, stem, ffnX, scratch)
		if err != nil {
			return err
		}
		for i := range routed {
			routed[i] += pick.weight * y[i]
		}
	}
	shared, err := m.v41SharedExpertSwiGLUWithActivation(l, ffnX, cfg, scratch.denseProjection, scratch.sharedActivation)
	if err != nil {
		return err
	}
	moe, err := v41SharedExpertAdd(routed, shared, routeCfg)
	if err != nil {
		return v41StageErr(v41StageMoE, l, err)
	}
	var next [][]float32
	if full {
		next, err = v41MHCPostBF16(l, moe, ffnResidual, ffnMix)
	} else {
		delta := make([]float32, H)
		for i := range delta {
			delta[i] = attnOut[i] + moe[i]
		}
		residual := [][]float32{collapsed, collapsed, collapsed, collapsed}
		next, err = v41MHCPost(delta, residual, mix.post, mix.comb)
	}
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
		scratch.mhcCarry.pre[0] = ffnMix.pre
		scratch.mhcCarry.nextLayer++
	} else {
		copy(x, next[0])
	}
	return nil
}

// v41LayerStepRole advances ONE position of a compressed / shared-source / reader
// V4.1 layer through its own retained window plus the shared compressed stream.
// It is the role-aware counterpart of v41LayerStep:
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
// The window ring commits only after the layer arithmetic succeeds. The outer
// composition restores window, compressor and publication state on failure.
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
	if pos < 0 || pos == int(^uint(0)>>1) || layerState.windowSize <= 0 ||
		len(layerState.window) != layerState.windowSize || layerState.headDim != cfg.HeadDim ||
		layerState.retainedWindowRows < 0 || layerState.retainedWindowRows > min(pos, layerState.windowSize) ||
		len(layerState.window[pos%layerState.windowSize]) != cfg.HeadDim {
		return v41StageErr(v41StageAttention, l, fmt.Errorf("%w: role window state has invalid geometry", ErrV41ForwardStage))
	}
	windowStart := 0
	if window := cfg.windowForLayer(l); window > 0 {
		windowStart = max(0, pos-window+1)
	}
	if windowStart < pos-layerState.retainedWindowRows {
		return v41StageErr(v41StageAttention, l, fmt.Errorf("%w: role window is not fully retained", ErrV41ForwardStage))
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

	if full {
		if err := scratch.mhcCarry.validate(l, 1); err != nil {
			return err
		}
		if err := m.v41AdmitMHC(l); err != nil {
			return err
		}
	}
	attnNorm := m.tensor(layerName(l, "attn_norm.weight"))
	mixBase := m.tensor(layerName(l, "mhc.base"))
	mixScale := m.tensor(layerName(l, "mhc.scale"))
	projectOutput := m.v41GroupedOutputProjector(l, nH, hd, cfg.OGroups, cfg.OLoraRank, H, scratch)

	// ---- mHC coefficient split (one position), byte-identical to the plain branch ----
	mhcFlat, mhcTransposed, mhcOK := m.v41MHCWeightLayout(l)
	if !mhcOK {
		return v41StageErr(v41StageMHC, l,
			fmt.Errorf("%w: mHC mix weight holds no admitted geometry", ErrV41ForwardStage))
	}
	xn := rmsnormCfg(x, attnNorm, eps, cfg)
	input := xn
	if mhcFlat {
		width, ok := checkedMulInt(4, H)
		if !ok || len(streams) != 4 {
			return v41StageErr(v41StageMHC, l, errV41ProjectionResult)
		}
		input = make([]float32, 0, width)
		for _, stream := range streams {
			if len(stream) != H {
				return v41StageErr(v41StageMHC, l, errV41ProjectionResult)
			}
			input = append(input, stream...)
		}
	}
	projectMHC := m.v41MHCProjector(l, H, eps, mhcFlat, mhcTransposed, scratch)
	mixes, err := projectMHC(input)
	if err != nil {
		return err
	}
	mix, err := v41MHCSplit(mixes, mixScale, mixBase, 4, hcItersOrDefault(cfg), hcEpsOrDefault(cfg))
	if err != nil {
		return v41StageErr(v41StageMHC, l, err)
	}
	collapsed, err := v41MHCPre(streams, scratch.mhcCarry.pre[0])
	if err != nil {
		return v41StageErr(v41StageMHC, l, err)
	}
	collapsed, err = m.v41AttentionInputNorm(l, collapsed, eps)
	if err != nil {
		return err
	}

	// ---- attention query for the one position (mirrors v41Layer's full branch) ----
	ropeDim := cfg.QKRopeHeadDim
	if ropeDim <= 0 || ropeDim%2 != 0 || ropeDim > hd {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: attention qk_rope_head_dim must be a positive even value <= head_dim %d, got %d", ErrV41ForwardStage, hd, ropeDim))
	}
	qLat, err := m.v41ProjMatRowsWithProjection(l, "attn.wq_a.weight", collapsed, cfg.QLoraRank, H, scratch.denseProjection)
	if err != nil {
		return err
	}
	if err := m.v41QueryNormInPlace(l, qLat, eps, scratch.queryNorm); err != nil {
		return err
	}
	q, err := m.v41ProjMatRowsWithProjection(l, "attn.wq_b.weight", qLat, nH*hd, cfg.QLoraRank, scratch.denseProjection)
	if err != nil {
		return err
	}
	if hd > v41KVLoraRank {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: attention head_dim %d exceeds full KV latent rank %d", ErrV41ForwardStage, hd, v41KVLoraRank))
	}
	kvFull, err := m.v41ProjMatRowsWithProjection(l, "attn.wkv.weight", collapsed, v41KVLoraRank, H, scratch.denseProjection)
	if err != nil {
		return err
	}
	kv := kvFull[:hd]
	if err := m.v41KVNormInPlace(l, kv, eps, scratch.kvNorm); err != nil {
		return err
	}
	cos, sin := v41RopeTableForLayer(cfg, l, pos)
	if err := v41TailRoPEInPlace(l, q, kv, cos, sin, nH, hd, ropeDim, scratch.tailRoPE); err != nil {
		return err
	}
	window := make([][]float32, 0, pos-windowStart+1)
	for key := windowStart; key < pos; key++ {
		window = append(window, layerState.window[key%layerState.windowSize])
	}
	window = append(window, kv)

	// ---- shared compressed stream: source publishes, reader resolves ----
	//
	// The shared registry (st.attn) is the model-scoped publication store the full
	// forward uses, while this layer's own state carries the compressor group
	// cursor. A source appends through ITS state and then mirrors the completed
	// rows into the registry so a later reader resolves them; a reader reads the
	// registry only. This mirrors v41Layer's source-then-consumer ordering (#12896)
	// at seq == 1.
	var sharedKV, indexKeys [][]float32
	var sourceIdx []int32
	if v41OwnsCompressedRows(plan) {
		width := v41CompressorWidth(cfg)
		normWeight := m.tensor(layerName(l, "attn.compressor.norm.weight"))
		pool, perr := NewV41CompressorPool(plan.Ratio, width)
		if perr != nil {
			return v41StageErr(v41StageCompress, l, perr)
		}
		if normalize := scratch.compressorNorm; normalize != nil {
			pool.normalize = func(pooled, gain []float32, eps float32) ([]float32, error) {
				return normalize(l, pooled, gain, eps)
			}
		}
		projectKV := func(in []float32) ([]float32, error) {
			return m.v41ProjMatRowsWithProjection(l, "attn.compressor.wkv.weight", in, width, H, scratch.denseProjection)
		}
		projectScore := func(in []float32) ([]float32, error) {
			return m.v41ProjMatRowsWithProjection(l, "attn.compressor.wgate.weight", in, width, H, scratch.denseProjection)
		}
		var indexPub bool
		var projectIndex func([]float32) ([]float32, error)
		if indexSourceAt(cfg.DeepSeekV41, l) {
			indexPub = true
			projectIndex = func(row []float32) ([]float32, error) {
				keys, err := m.v41IndexKeysWithOperations(l, [][]float32{row}, scratch.denseProjection, scratch.indexKeyNorm)
				if err != nil {
					return nil, err
				}
				return keys[0], nil
			}
		}
		_, _, _, _, _, err = layerState.appendCompressorSourcePublication(
			l, plan.Ratio, pos, collapsed, pool, projectKV, projectScore, normWeight, eps,
			v41CompressorPublication{
				projectIndex: projectIndex,
				finalize: func(start int, latent, key []float32) error {
					return m.v41CompressedPublicationRoPE(l, start, latent, key)
				},
				registry: registry,
				ref:      V41AttentionStateRef{LayerID: l, Ratio: plan.Ratio, IsKVSource: plan.KVSourceLayer == l, IsIndexSource: indexPub},
			})
		if err != nil {
			return v41StageErr(v41StageCompress, l, err)
		}
		sharedKV, _ = layerState.KVSourceRows(l)
		if indexPub {
			var ok bool
			indexKeys, ok = layerState.IndexKeys(l)
			if len(sharedKV) != len(indexKeys) || (len(sharedKV) > 0 && !ok) {
				return v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: own index history is incomplete", ErrV41ForwardStage))
			}
		}
	} else {
		sharedKV, indexKeys, err = m.v41CompressedReaderRows(plan, registry, pos+1)
		if err != nil {
			return err
		}
	}
	if indexSourceAt(cfg.DeepSeekV41, l) {
		sourceIdx, err = m.v41IndexRowsWithOperations(l, pos, qLat, collapsed, indexKeys, scratch.denseProjection, scratch.indexerScore, scratch.indexScoreHealth)
		if err != nil {
			return err
		}
		row, err := m.v41RoleStepNormalizeIndex(l, plan, sourceIdx, len(indexKeys))
		if err != nil {
			return err
		}
		if err := registry.PublishTopK(plan.Ratio, [][]int32{row}); err != nil {
			return err
		}
	}
	// Canonical compressed IDs remain unoffset until the common composer adds
	// this layer's window width. Padding and repeated selections are preserved.
	var ids []int32
	if plan.TopKWidth > 0 {
		var ierr error
		if indexSourceAt(cfg.DeepSeekV41, l) {
			ids, ierr = m.v41RoleStepNormalizeIndex(l, plan, sourceIdx, len(sharedKV))
		} else {
			ids, ierr = m.v41RoleStepIndex(registry, l, pos, plan, qLat, collapsed, len(sharedKV))
		}
		if ierr != nil {
			return ierr
		}
		if ids == nil {
			ids, ierr = m.v41RoleStepNormalizeIndex(l, plan, nil, len(sharedKV))
			if ierr != nil {
				return ierr
			}
		}
	}
	if ids == nil {
		ids = make([]int32, len(sharedKV))
		for group := range ids {
			ids[group] = int32(group)
		}
	}
	attentionOpened := m.v41NowNanos()
	o, err := v41CombinedAttention(l, pos, plan.kvGroupSize(cfg), nH, hd, q, m.tensor(layerName(l, "attn.sink")), window, sharedKV, ids, cfg.attnScale(), scratch.sharedAttention)
	m.v41NoteAttentionContraction(attentionOpened)
	if err != nil {
		return v41StageErr(v41StageAttention, l, err)
	}
	v41InverseAttentionOutputInPlace(cfg, l, pos, o, nH, hd)
	attnProjected, err := projectOutput(o)
	if err != nil {
		return v41StageErr(v41StageAttention, l, err)
	}

	if err := m.v41LayerStepRoleFinish(l, x, streams, mix, attnProjected, scratch); err != nil {
		return err
	}
	// No error-returning operation remains. State.Step would advance an owner's
	// already-committed compressor cursor twice. Retain this layer's own row,
	// including on reader layers and before the first completed source group.
	copy(layerState.window[pos%layerState.windowSize], kv)
	layerState.nextWindowPos, layerState.nextCompressRow = pos+1, pos+1
	retained := min(layerState.retainedWindowRows+1, layerState.windowSize)
	if configured := cfg.windowForLayer(l); configured > 0 {
		retained = min(retained, configured)
	}
	layerState.retainedCopies += retained - layerState.retainedWindowRows
	layerState.retainedWindowRows = retained
	return nil
}

// v41LayerStepRoleFinish runs the MoE block and the mHC post for one position and
// commits the position's mHC streams and hidden row. It is shared by the source
// and reader paths so the tail after attention is byte-identical to the plain
// branch's seq == 1 arithmetic. The caller commits its window after this returns.
func (m *Model) v41LayerStepRoleFinish(l int, x []float32, streams [][]float32, mix v41MHCMix, attnOut []float32, scratch *v41ProjScratch) error {
	cfg := m.Cfg
	H := cfg.HiddenSize
	projectFFN, err := m.v41FFNMHCProjector(l, scratch)
	if err != nil {
		return err
	}
	ffnResidual, ffnMix, ffnX, err := m.v41FullFFNInput(l, streams, mix, attnOut, projectFFN, scratch)
	if err != nil {
		return err
	}
	routeCfg, err := v41RouterConfigFullGeometry(cfg)
	if err != nil {
		return err
	}
	routerLogits, err := m.v41ProjMatRowsWithProjection(l, "ffn.gate.weight", ffnX, cfg.NumExperts, H, scratch.denseProjection)
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
		y, err := m.v41IncrementalExpert(l, stem, ffnX, scratch)
		if err != nil {
			return err
		}
		for i := range routed {
			routed[i] += pick.weight * y[i]
		}
	}
	shared, err := m.v41SharedExpertSwiGLUWithActivation(l, ffnX, cfg, scratch.denseProjection, scratch.sharedActivation)
	if err != nil {
		return err
	}
	moe, err := v41SharedExpertAdd(routed, shared, routeCfg)
	if err != nil {
		return v41StageErr(v41StageMoE, l, err)
	}
	next, err := v41MHCPostBF16(l, moe, ffnResidual, ffnMix)
	if err != nil {
		return v41StageErr(v41StageMHC, l, err)
	}
	for h := 0; h < 4; h++ {
		copy(streams[h], next[h])
	}
	copy(x, next[0])
	scratch.mhcCarry.pre[0] = ffnMix.pre
	scratch.mhcCarry.nextLayer++
	return nil
}

// v41RoleStepIndex resolves the per-position index selection a compressed
// contraction consumes. A layer that IS the index source publishes its own
// selection (computed here from the source's own completed stream) and a reader
// reuses its source's published top-k, matching v41AttentionIndexList's
// precedence at seq == 1.
func (m *Model) v41RoleStepIndex(state *V41AttentionState, l, pos int, plan V41AttentionPlan, qLat, hidden []float32, groups int) ([]int32, error) {
	if plan.TopKWidth <= 0 || groups <= 0 {
		return nil, nil
	}
	if indexSourceAt(m.Cfg.DeepSeekV41, l) {
		rows, ok := state.IndexKeys(l)
		if !ok || len(rows) == 0 {
			return nil, nil
		}
		idx, err := m.v41IndexRowsProjected(l, pos, qLat, hidden, rows, nil)
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
func (m *Model) v41RoleStepPublishTopK(state *V41AttentionState, l, pos int, plan V41AttentionPlan, qLat, hidden []float32, keys [][]float32) error {
	idx, err := m.v41IndexRowsProjected(l, pos, qLat, hidden, keys, nil)
	if err != nil {
		return err
	}
	row, err := m.v41RoleStepNormalizeIndex(l, plan, idx, len(keys))
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
	if len(row) != state.indexWidth() {
		return nil, fmt.Errorf("%w: index reader layer %d position %d: published row width %d, want %d", ErrV41ForwardStage, layer, pos, len(row), state.indexWidth())
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

var errV41ExpertResult = errors.New("model: invalid selected V4.1 expert result")

type V41ExpertOperationError struct {
	Layer int
	Stage string
	Cause error
}

func (e *V41ExpertOperationError) Error() string {
	return fmt.Sprintf("model: selected V4.1 expert operation %s failed at layer %d: %v", e.Stage, e.Layer, e.Cause)
}

func (e *V41ExpertOperationError) Unwrap() error { return e.Cause }

func (e *V41ExpertOperationError) SelectedExpertOperation() bool { return true }

func v41ExpertOperationErr(l int, stage string, cause error) error {
	if cause == nil {
		cause = errV41ExpertResult
	}
	return &V41ExpertOperationError{Layer: l, Stage: stage, Cause: v41StageErr(v41StageMoE, l, cause)}
}

func (m *Model) v41IncrementalExpert(l int, stem string, xn []float32, scratch *v41ProjScratch) (out []float32, resultErr error) {
	cfg := m.Cfg
	var dispatchOpen int64
	var dispatchActive, gateUp bool
	defer func() {
		if r := recover(); r != nil {
			if cause, ok := r.(error); ok && dispatchActive {
				var operation *BackendForwardOperationError
				if errors.As(cause, &operation) {
					m.v41NoteIncrementalDeviceDispatch(gateUp, dispatchOpen)
					panic(r)
				}
				var backendError *compute.BackendError
				if errors.As(cause, &backendError) {
					m.v41NoteIncrementalDeviceDispatch(gateUp, dispatchOpen)
					stage := "down"
					if gateUp {
						stage = "gate/up"
					}
					out, resultErr = nil, v41ExpertOperationErr(l, stage, cause)
					return
				}
			}
			panic(r)
		}
	}()
	if scratch.expertGateUp != nil {
		dispatchOpen, dispatchActive, gateUp = m.v41NowNanos(), true, true
		h, outcome, err := scratch.expertGateUp(l, stem, xn)
		dispatchActive = false
		if outcome == v41GateUpHandled || outcome == v41GateUpError {
			m.v41NoteIncrementalDeviceDispatch(true, dispatchOpen)
		}
		switch outcome {
		case v41GateUpError:
			return nil, v41ExpertOperationErr(l, "gate/up", err)
		case v41GateUpHandled:
			if err != nil {
				return nil, v41ExpertOperationErr(l, "gate/up", err)
			}
			if len(h) != cfg.MoEIntermediateSize {
				return nil, v41ExpertOperationErr(l, "gate/up", fmt.Errorf("%w: intermediate width %d, want %d", errV41ExpertResult, len(h), cfg.MoEIntermediateSize))
			}
			if scratch.expertDown != nil {
				dispatchOpen, dispatchActive, gateUp = m.v41NowNanos(), true, false
				y, downOutcome, downErr := scratch.expertDown(l, stem, h)
				dispatchActive = false
				if downOutcome == v41DownHandled || downOutcome == v41DownError {
					m.v41NoteIncrementalDeviceDispatch(false, dispatchOpen)
				}
				switch downOutcome {
				case v41DownError:
					return nil, v41ExpertOperationErr(l, "down", downErr)
				case v41DownHandled:
					if downErr != nil {
						return nil, v41ExpertOperationErr(l, "down", downErr)
					}
					if len(y) != cfg.HiddenSize {
						return nil, v41ExpertOperationErr(l, "down", fmt.Errorf("%w: output width %d, want %d", errV41ExpertResult, len(y), cfg.HiddenSize))
					}
					m.v41NoteExpertContraction()
					return y, nil
				case v41DownDeclined:
				default:
					return nil, v41ExpertOperationErr(l, "down", errV41ExpertResult)
				}
			}
			w2, err := m.hostExpertDown(l, stem, scratch)
			if err != nil {
				return nil, err
			}
			contractOpen := m.v41NowNanos()
			y := matRows(w2, h, cfg.HiddenSize, cfg.MoEIntermediateSize)
			if contractOpen != 0 {
				m.v41NoteExpertContractionNanos(m.v41NowNanos() - contractOpen)
			} else {
				m.v41NoteExpertContraction()
			}
			return y, nil
		case v41GateUpDeclined:
		default:
			return nil, v41ExpertOperationErr(l, "gate/up", errV41ExpertResult)
		}
	}
	w1, w3, w2, err := m.v41ExpertTripleInto(l, stem, scratch, true)
	if err != nil {
		return nil, err
	}
	contractOpen := m.v41NowNanos()
	y := v41SwiGLU(w1, w3, w2, xn, cfg.MoEIntermediateSize, cfg.HiddenSize, cfg)
	if contractOpen != 0 {
		m.v41NoteExpertContractionNanos(m.v41NowNanos() - contractOpen)
	} else {
		m.v41NoteExpertContraction()
	}
	return y, nil
}
