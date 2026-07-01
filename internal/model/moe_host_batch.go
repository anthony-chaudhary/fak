package model

// moe_host_batch.go — lever 2: batch the GLM MoE host-expert dispatch.
//
// The per-expert loop (glmMoeFFN.apply -> expertSwiGLU) issues one parFor per
// (expert, projection): ~3*K tiny dispatches per MoE layer. parFor's per-call cost
// (global dispatch mutex + worker wake + busy-wait drain + per-chunk atomic cursor
// contention) dominates when each call is small, capping parallel scaling far below the
// core count (measured ~6-9x of 32 cores looped). This module runs ALL active experts'
// rows for a projection in ONE parFor over a flattened (expert, row) index space,
// splitting each work chunk on expert boundaries and calling the SAME resident Range
// kernel the per-expert path calls. Every output row is computed by the identical
// reduction in the identical order, so the result is BIT-IDENTICAL to the loop (pinned by
// TestBatchedExpertDeltaMatchesLoop); only the parFor dispatch COUNT changes (3 per layer
// instead of ~3*K). See docs/notes/GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md (lever 2).

// batchExpertRows runs fn(e, rlo, rhi) for every (expert, contiguous row-range) chunk in the
// flattened [0, K*rowsPer) index space, splitting each parFor work chunk on expert boundaries
// — the shared dispatch scaffold of q4kBatchRows / kQuantBatchRows. inDim is ts[0].in, used
// only for the parThreshold serial/parallel gate; fn owns the per-row reduction, so every
// output row is computed in the identical order regardless of how the chunk is split.
func batchExpertRows(K, rowsPer, inDim int, fn func(e, rlo, rhi int)) {
	if K == 0 || rowsPer == 0 {
		return
	}
	total := K * rowsPer
	body := func(lo, hi int) {
		for g := lo; g < hi; {
			e := g / rowsPer
			rlo := g - e*rowsPer
			rhi := rowsPer
			if e*rowsPer+rhi > hi {
				rhi = hi - e*rowsPer
			}
			fn(e, rlo, rhi)
			g = e*rowsPer + rhi
		}
	}
	parForRange(total, total*inDim, body)
}

// q4kBatchRows computes, for K resident q4kTensors that all share the SAME activation xn
// (shape [rowsPer, in]), every tensor's [rowsPer] output, in ONE parFor over the flattened
// [0, K*rowsPer) row space. y[e] receives tensor e's output. The per-row reduction is the
// same q4kMatRowsRange / q4kMatRowsRangeInt8 the looped q4kMatRows uses (same int8/f32 gate,
// same once-quantized activation), so y[e] is bit-identical to q4kMatRows(ts[e], xn).
func q4kBatchRows(ts []*q4kTensor, xn []float32, rowsPer int, y [][]float32) {
	K := len(ts)
	if K == 0 || rowsPer == 0 {
		return
	}
	useInt8 := q4kSDOTEnabled()
	var qv q8Vec
	if useInt8 {
		qv = quantizeVecQ8(xn)
	}
	batchExpertRows(K, rowsPer, ts[0].in, func(e, rlo, rhi int) {
		if useInt8 {
			q4kMatRowsRangeInt8(ts[e], qv, y[e], rlo, rhi)
		} else {
			q4kMatRowsRange(ts[e], xn, y[e], rlo, rhi)
		}
	})
}

// kQuantBatchRows is the q4kBatchRows twin for resident Q5_K/Q6_K tensors with PER-EXPERT
// activations (the down projection: each expert's input is its own SwiGLU result). acts[e]
// is tensor e's activation; y[e] receives its [rowsPer] output. Each tensor's int8/f32 gate
// and per-row reduction match kQuantMatRows(ts[e], acts[e]) exactly, so the result is
// bit-identical to the per-expert call.
func kQuantBatchRows(ts []*kQuantTensor, acts [][]float32, rowsPer int, y [][]float32) {
	K := len(ts)
	if K == 0 || rowsPer == 0 {
		return
	}
	useInt8 := make([]bool, K)
	qvs := make([]q8Vec, K)
	for e := range ts {
		useInt8[e] = kQuantSDOTEnabled(ts[e].kind)
		if useInt8[e] {
			qvs[e] = quantizeVecQ8(acts[e])
		}
	}
	batchExpertRows(K, rowsPer, ts[0].in, func(e, rlo, rhi int) {
		if useInt8[e] {
			ranger := q5kMatRowsRangeInt8
			if ts[e].kind == kindQ6K {
				ranger = q6kMatRowsRangeInt8
			}
			ranger(ts[e], qvs[e], y[e], rlo, rhi)
		} else {
			kQuantMatRowsRange(ts[e], acts[e], y[e], rlo, rhi)
		}
	})
}

// batchedExpertDelta accumulates the gate-weighted SwiGLU sum of the picked experts into
// delta, batching each projection's GEMVs across experts (q4kBatchRows for gate+up which
// share xn, then per-expert SwiGLU, then kQuantBatchRows for down). The SwiGLU math, the
// activation, and the gate-weighted accumulation order are identical to the per-expert
// expertSwiGLU loop, so delta is bit-identical to it. Pure (no Model lookup) so the
// bit-identity test can drive it with random tensors.
func batchedExpertDelta(cfg Config, picks []routePick, gate, up []*q4kTensor, down []*kQuantTensor, xn, delta []float32) {
	K := len(picks)
	H := cfg.HiddenSize
	MI := cfg.expertIntermediate()
	gOut := make([][]float32, K)
	uOut := make([][]float32, K)
	dOut := make([][]float32, K)
	for i := 0; i < K; i++ {
		gOut[i] = make([]float32, MI)
		uOut[i] = make([]float32, MI)
		dOut[i] = make([]float32, H)
	}
	q4kBatchRows(gate, xn, MI, gOut)
	q4kBatchRows(up, xn, MI, uOut)
	swig := make([][]float32, K)
	for i := 0; i < K; i++ {
		g, u := gOut[i], uOut[i]
		for j := 0; j < MI; j++ {
			g[j] = act(g[j], cfg) * u[j]
		}
		swig[i] = g
	}
	kQuantBatchRows(down, swig, H, dOut)
	for i, pk := range picks {
		w := pk.weight
		d := dOut[i]
		for j := 0; j < H; j++ {
			delta[j] += w * d[j]
		}
	}
}

// hostBatchedGLMExperts is the Model-bound entry: it gathers the picked experts' resident
// gate/up (q4kw) + down (kqw) tensors and, when every one is present in that exact resident
// shape with no per-expert bias and no active LoRA, accumulates the batched delta and returns
// true. Otherwise it writes NOTHING and returns false, so the caller falls back to the proven
// per-expert loop. The guards keep the fast path bit-identical: a bias add or a LoRA delta the
// loop applies are NOT modeled here, so their presence declines the fast path rather than
// diverging.
func (m *Model) hostBatchedGLMExperts(layer int, xn, delta []float32, picks []routePick) bool {
	if m.lora != nil {
		return false
	}
	K := len(picks)
	if K == 0 {
		return true
	}
	gate := make([]*q4kTensor, K)
	up := make([]*q4kTensor, K)
	down := make([]*kQuantTensor, K)
	for i, pk := range picks {
		g := m.q4kw[expertName(layer, pk.expert, "gate_proj.weight")]
		u := m.q4kw[expertName(layer, pk.expert, "up_proj.weight")]
		d := m.kqw[expertName(layer, pk.expert, "down_proj.weight")]
		if g == nil || u == nil || d == nil {
			return false
		}
		if m.has(expertName(layer, pk.expert, "gate_proj.bias")) ||
			m.has(expertName(layer, pk.expert, "up_proj.bias")) ||
			m.has(expertName(layer, pk.expert, "down_proj.bias")) {
			return false
		}
		gate[i], up[i], down[i] = g, u, d
	}
	batchedExpertDelta(m.Cfg, picks, gate, up, down, xn, delta)
	return true
}

// batchedMetalExperts is the Metal-decode twin of hostBatchedGLMExperts (#1382): it gathers the
// picked experts' gate/up/down tensor NAMES and runs all of them through the session's batched
// fused-MLP (q4kFusedMLPBatch — one Metal command buffer for the whole layer's top-k experts),
// then accumulates the gate-weighted sum into delta in route order. It returns true only when the
// batch actually ran on the GPU; otherwise it writes NOTHING and returns false so moeFFN falls back
// to the per-expert loop (where each expert still takes the single-expert q4kFusedMLP). The guards
// mirror the loop's fused-path gate exactly (no LoRA, no per-expert bias), so a config the fused
// path cannot serve declines rather than diverging; the host-order accumulation keeps delta
// bit-identical to the loop up to Metal float-order, pinned by the decode parity gate.
func (m *Model) batchedMetalExperts(s *Session, layer int, xn, delta []float32, picks []routePick) bool {
	if m.lora != nil {
		return false
	}
	K := len(picks)
	if K == 0 {
		return true
	}
	gate := make([]string, K)
	up := make([]string, K)
	down := make([]string, K)
	for i, pk := range picks {
		if m.has(expertName(layer, pk.expert, "gate_proj.bias")) ||
			m.has(expertName(layer, pk.expert, "up_proj.bias")) ||
			m.has(expertName(layer, pk.expert, "down_proj.bias")) {
			return false
		}
		gate[i] = expertName(layer, pk.expert, "gate_proj.weight")
		up[i] = expertName(layer, pk.expert, "up_proj.weight")
		down[i] = expertName(layer, pk.expert, "down_proj.weight")
	}
	outs := s.q4kFusedMLPBatch(gate, up, down, xn)
	if outs == nil {
		return false
	}
	H := m.Cfg.HiddenSize
	for i, pk := range picks {
		o := outs[i]
		if len(o) < H {
			return false // defensive: a short row means the batch geometry disagreed
		}
		for j := 0; j < H; j++ {
			delta[j] += pk.weight * o[j]
		}
	}
	return true
}
