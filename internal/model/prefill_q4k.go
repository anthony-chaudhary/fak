package model

// prefill_q4k.go — the resident-Q4_K PREFILL lane: a BATCHED prefill that is the structural
// twin of prefillBatchedQ (quant_forward.go), differing in exactly ONE way — each per-layer
// projection GEMM dispatches by resident format, exactly as the per-token sessionQ4KKernel
// does for decode:
//
//   - the q4_k_m majority (self_attn.v_proj/o_proj, mlp.gate/up/down — every identity-
//     normalized matmul weight the loader held raw) runs q4kGemmDispatch, the batched Q4_K
//     GEMM, on the f32 activation: the CPU q4kGemm by default, and under -tags fakmetal with
//     s.MetalQ4K the Metal q4_k dequant-GEMM on the GPU — the same GPU route decode's GEMV
//     already took, so the resident-Q4_K prefill stops running the slow CPU GEMM that timed
//     out real prompts (#1071). Each weight super-block is dequantized ONCE and reused across
//     all P prompt tokens, instead of re-streaming the whole weight matrix P times as the
//     per-token GEMV prefill does. Prefill is compute-bound, so amortizing the dequant +
//     weight bandwidth across the P free axes is what closes the gap to llama.cpp-Metal on
//     the q4_k_m artifact (QWEN36-NATIVE-PERF-PLAN-2026-06-19.md P3).
//   - the resident K-quant minority (kqw: Q5_K/Q6_K weights such as v_proj or down_proj
//     in mixed-quant artifacts) runs kQuantGemmDispatch on the f32 activation.
//   - the normalize-sensitive / Q8 minority (self_attn.q_proj/k_proj always, plus any Q8
//     weight or un-quantized f32 manifest tensor) runs q8GemmDispatch against the
//     q8w store — CPU qGemm8 by default, Metal Q8 GEMM when MetalQ4K is enabled — on a
//     Q8-quantized activation panel.
//
// Everything else — RMSNorm, RoPE, the causal GQA attention over the f32 KV cache, SwiGLU,
// the residuals — is the identical f32 math prefillBatchedQ runs. The cache it builds is the
// same f32 object (Kraw pre-RoPE, K post-RoPE, V, pos), so Evict/Clone and the proven KV
// rungs are unaffected.
//
// Correctness contract vs the per-token Q4K decode path (tokenHiddenQ via sessionQ4KKernel):
//   - For a Q4_K-resident projection, q4kGemm[o,t] is BIT-IDENTICAL to q4kMatRows(row o,
//     activation t) — same per-super-block dequant, same 4-accumulator dot, same super-block
//     order (TestQ4KGemmMatchesMatRows). So the q4_k_m majority produces byte-for-byte the
//     same projection as the proven per-token Q4K path.
//   - For a Q8-minority projection, qGemm8 is the SAME register-blocked tile kernel
//     prefillBatchedQ uses; its relationship to the per-token Q8 GEMV (qMatRows/qdot8) is
//     the documented deferred-reduction / single-rounded-FMA drift already covered by the
//     Q8 path's own gate (argmax-exact vs the oracle, logit-cosine-tight) — NOT a new
//     numerical surface introduced here.
// The end-to-end Q4_K correctness gate is unchanged: greedy-continuation agreement with the
// llama.cpp q4_k_m artifact + first-token id parity (248068), the standard the plan holds
// the whole Q4_K lane to.

import (
	"fmt"
	"os"
	"time"
)

// prefillBatchedQ4K ingests `ids` as a batch through the resident-Q4_K path, appending P
// positions to the cache and returning the LAST token's post-final-norm hidden (caller
// applies the head). It assumes the q4k-hybrid load: every matmul weight is resident in
// EITHER q4kw (raw Q4_K majority) or q8w (Q8 minority); the per-projection dispatch picks
// the right one. Fills the same f32 KV cache the per-token / f32 / Q8 paths build.
func (s *Session) prefillBatchedQ4K(ids []int) []float32 {
	// The dense layer graph bounds one command buffer's panel; walk a longer prompt in
	// panels. Each panel appends its positions before the next reads them as prefix, so the
	// causal result is the single-pass result.
	useGraph := s.denseQ4KPrefillGraphEligible()
	if useGraph && len(ids) > denseQ4KPrefillGraphMaxRows {
		var last []float32
		for lo := 0; lo < len(ids); lo += denseQ4KPrefillGraphMaxRows {
			last = s.prefillBatchedQ4KPanel(ids[lo:min(lo+denseQ4KPrefillGraphMaxRows, len(ids))], true)
		}
		return last
	}
	return s.prefillBatchedQ4KPanel(ids, useGraph)
}

// prefillBatchedQ4KPanel is one batched pass over ids. useGraph admits the dense layer graph
// (metal_dense_prefill_graph.go): each layer boundary — o_proj, residual, post-norm, MLP,
// residual, next layer's input norm and q/k/v — runs as ONE command buffer, with the host
// attention between graphs. A declined segment falls back, for the rest of the panel, to the
// per-projection host route, which is the reference this route is gated against.
func (s *Session) prefillBatchedQ4KPanel(ids []int, useGraph bool) []float32 {
	dispatchWorkers := currentWorkerCount()
	m, cfg := s.M, s.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := cfg.attnScale()
	attnCap := float32(cfg.AttnSoftcap)
	P := len(ids)
	base := s.Cache.Len()

	var tQuant, tGemm, tAttn, tGraph time.Duration
	var graphGPUms float64
	graphCBs := 0
	t0 := time.Now()
	tic := func() time.Time {
		if qprofOn {
			return time.Now()
		}
		return time.Time{}
	}
	toc := func(d *time.Duration, t time.Time) {
		if qprofOn {
			*d += time.Since(t)
		}
	}
	// One reused Q8 activation-panel scratch for the minority projections that need it
	// (q/k always; plus any Q6_K minority such as down_proj). Each panel is fully consumed
	// before the next is built, so a single buffer is safe — same discipline as prefillBatchedQ.
	scratch := &q8Panel{}
	qz := func(X []float32, P, width int) *q8Panel {
		t := tic()
		quantizeBatchPanelInto(scratch, X, P, width)
		toc(&tQuant, t)
		return scratch
	}
	// lazyQ8 builds the shared Q8 panel of one f32 activation only when a projection that
	// reads it is actually on the Q8 path: in the q4k-hybrid load o/gate/up/down are almost
	// always q4kw/kqw-resident, so their panels were built and thrown away (#13599 Step A).
	// The built panel is cached for the activation's other consumers (q/k/v share one).
	type lazyQ8 struct {
		X     []float32
		width int
		p     *q8Panel
	}
	panel := func(lz *lazyQ8) *q8Panel {
		if lz.p == nil {
			lz.p = qz(lz.X, P, lz.width)
		}
		return lz.p
	}
	// proj dispatches a batched projection [P,out] by resident format: q4kw-resident →
	// q4kGemmDispatch on the f32 activation Xf; kqw-resident → kQuantGemmDispatch;
	// otherwise → q8GemmDispatch on the activation's lazily-built Q8 panel (with m.q8
	// quantizing the weight on demand from the f32 manifest if un-quantized). The width is
	// inferred from the resident tensor's .out, so the caller does not pass it.
	//
	// q4kGemmDispatch is the CPU q4kGemm by default (pure-Go build, bit-identical to before);
	// under -tags fakmetal with s.MetalQ4K set it routes the q4_k-majority batched GEMM to the
	// Metal q4_k dequant-GEMM, the same GPU path decode's GEMV already takes — so the resident-
	// Q4_K prefill runs on the GPU instead of the slow CPU GEMM that timed out real prompts
	// (#1071), mirroring the already-dispatched qwen35-hybrid prefill (qwen35_prefill_q4k.go).
	proj := func(name string, lz *lazyQ8) []float32 {
		q8Path := m.q4kw[name] == nil && m.kqw[name] == nil && m.q2w[name] == nil
		if q8Path && (m.prism == nil || m.prism.weightWidth[name] == 0) {
			panel(lz) // build outside the gemm window so quant is not double-counted
		}
		t := tic()
		var r []float32
		Xf := lz.X
		Xrot := m.prismProjectPanel(name, Xf, P)
		if qt := m.q4kw[name]; qt != nil {
			r = s.q4kGemmDispatch(name, qt, Xrot, P)
		} else if qt := m.kqw[name]; qt != nil {
			r = s.kQuantGemmDispatch(name, qt, Xrot, P)
		} else if qt := m.q2w[name]; qt != nil {
			r = q2MatRowsBatch(qt, Xrot, P)
		} else {
			var Xq *q8Panel
			if m.prism != nil && m.prism.weightWidth[name] != 0 {
				Xq = &q8Panel{}
				quantizeBatchPanelInto(Xq, Xrot, P, m.prism.weightWidth[name])
			} else {
				Xq = panel(lz)
			}
			r = s.q8GemmDispatch(name, m.q8(name), Xq)
		}
		toc(&tGemm, t)
		return r
	}
	// projGroup runs a same-activation projection set, first through the one-command-buffer
	// Metal group dispatchers (Q8 members on the shared panel, Q4_K members on the f32
	// activation), then fills every member they left nil through proj. Both group
	// dispatchers decline (nil) off-Metal, so the CPU route is the per-weight loop unchanged.
	projGroup := func(names []string, lz *lazyQ8) [][]float32 {
		out := make([][]float32, len(names))
		q8Members := 0
		for _, name := range names {
			if m.q4kw[name] == nil && m.kqw[name] == nil && m.q2w[name] == nil && m.q8w[name] != nil {
				q8Members++
			}
		}
		var Xq *q8Panel
		if q8Members >= 2 && s.MetalQ4K && m.prism == nil {
			Xq = panel(lz) // built outside the gemm window so quant is not double-counted
		}
		t := tic()
		if Xq != nil {
			for i, r := range s.q8GemmGroupDispatchDirect(names, Xq, P) {
				out[i] = r
			}
		}
		for i, r := range s.q4kGemmGroupDispatch(names, lz.X, P) {
			if r != nil {
				out[i] = r
			}
		}
		toc(&tGemm, t)
		for i, name := range names {
			if out[i] == nil {
				out[i] = proj(name, lz)
			}
		}
		return out
	}

	embed := m.embedRows()
	X := make([]float32, P*H)
	for t, id := range ids {
		copy(X[t*H:(t+1)*H], embed[id*H:(id+1)*H])
		m.prismInverseEmbeddingRow("model.embed_tokens.weight", X[t*H:(t+1)*H])
		scaleEmbedInPlace(X[t*H:(t+1)*H], cfg) // Gemma; no-op for Llama/Qwen
	}

	cosP := make([][]float32, P)
	sinP := make([][]float32, P)
	for t := 0; t < P; t++ {
		cosP[t], sinP[t] = ropeRow(cfg, base+t)
	}

	if s.MetalQ4K {
		m.metalQ4KWeights() // upload all Q4_K weights upfront — avoids per-call GPU round-trips (#1113)
		m.metalQ8Weights()  // upload Q8-minority projection weights upfront for Metal Q8 prefill (#1087)
	}

	// Normalization fully overwrites this request-local panel; synchronous projections
	// finish consuming it before the next normalization, including across layers.
	normPanel := make([]float32, P*H)

	// hostLayerIn is layer l's host head: input norm, then q/k/v.
	hostLayerIn := func(l int) (Q, K, V []float32) {
		lp := func(str string) string { return layerName(l, str) }
		// kv.go:670 routes here on !q8PrefillNeedsTokenLoop, which has no LayerNorm term, so a
		// resident-Q4_K PreNorm LayerNorm family prefills HERE while decoding through the
		// bias-aware blockStep. The learned input_layernorm.bias must ride along; rmsnormCfg
		// hard-passes nil.
		Xn := normPanel
		parForWork(P, dispatchWorkers, H, func(lo, hi int) {
			wIn := m.tensor(lp("input_layernorm.weight"))
			bIn := m.tensorOptional(lp("input_layernorm.bias"))
			for t := lo; t < hi; t++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn[t*H:(t+1)*H], normCfg(X[t*H:(t+1)*H], wIn, bIn, eps, cfg))
				} else {
					rmsnormInto(Xn[t*H:(t+1)*H], X[t*H:(t+1)*H], wIn, eps)
				}
			}
		})
		// One lazily-built Q8 panel feeds the Q8-minority projections (q/k in the q4k-hybrid
		// load); q+k share one grouped command buffer, and a q4k-resident v reads raw f32 Xn.
		qkv := projGroup([]string{lp("self_attn.q_proj.weight"), lp("self_attn.k_proj.weight"), lp("self_attn.v_proj.weight")}, &lazyQ8{X: Xn, width: H})
		return qkv[0], qkv[1], qkv[2]
	}

	// hostLayerOut is layer l's host tail: o_proj on attnOut, residual, post-norm, SwiGLU MLP,
	// residual — all into X.
	hostLayerOut := func(l int, attnOut []float32) {
		lp := func(str string) string { return layerName(l, str) }
		O := proj(lp("self_attn.o_proj.weight"), &lazyQ8{X: attnOut, width: nH * hd})
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(O[t*H:(t+1)*H], lp("self_attn.o_proj.bias"))
		}
		parForWork(len(X), dispatchWorkers, 1, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += O[i]
			}
		})

		Xn2 := normPanel
		parForWork(P, dispatchWorkers, H, func(lo, hi int) {
			wPost := m.tensor(lp("post_attention_layernorm.weight"))
			bPost := m.tensorOptional(lp("post_attention_layernorm.bias"))
			for t := lo; t < hi; t++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn2[t*H:(t+1)*H], normCfg(X[t*H:(t+1)*H], wPost, bPost, eps, cfg))
				} else {
					rmsnormInto(Xn2[t*H:(t+1)*H], X[t*H:(t+1)*H], wPost, eps)
				}
			}
		})
		I := cfg.IntermediateSize
		gu := projGroup([]string{lp("mlp.gate_proj.weight"), lp("mlp.up_proj.weight")}, &lazyQ8{X: Xn2, width: H})
		G, U := gu[0], gu[1]
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(G[t*I:(t+1)*I], lp("mlp.gate_proj.bias"))
			m.addBiasIfPresent(U[t*I:(t+1)*I], lp("mlp.up_proj.bias"))
		}
		parForWork(len(G), dispatchWorkers, 1, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				G[i] = act(G[i], cfg) * U[i]
			}
		})
		Down := proj(lp("mlp.down_proj.weight"), &lazyQ8{X: G, width: I})
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(Down[t*H:(t+1)*H], lp("mlp.down_proj.bias"))
		}
		parForWork(len(X), dispatchWorkers, 1, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += Down[i]
			}
		})
	}

	// graphStep runs one dense layer-graph segment (tail of prev, head of next). A decline
	// disables the graph for the rest of this panel and is recorded; the caller then runs
	// the same segment on the host route from the unchanged X.
	graphStep := func(prev int, attnOut []float32, next int) (Q, K, V []float32, ok bool) {
		t := tic()
		Q, K, V, receipt, ok := s.denseQ4KGraphSegment(prev, attnOut, next, X, P)
		toc(&tGraph, t)
		if !ok {
			useGraph = false
			s.recordMetalFallback(MetalFallbackDensePrefillGraphHost)
			return nil, nil, nil, false
		}
		graphCBs++
		graphGPUms += receipt.GPUMilliseconds
		return Q, K, V, true
	}

	var pending []float32 // layer l-1's attention output, awaiting its o_proj/MLP tail
	for l := 0; l < cfg.NumLayers; l++ {
		var Q, K, V []float32
		viaGraph := false
		if useGraph {
			Q, K, V, viaGraph = graphStep(l-1, pending, l)
		}
		if !viaGraph {
			if l > 0 {
				hostLayerOut(l-1, pending)
			}
			Q, K, V = hostLayerIn(l)
		}
		for t := 0; t < P; t++ {
			m.applyProjBias(l, Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w], V[t*w:(t+1)*w])
			m.applyLayerQKNorm(l, Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w])
		}

		// Stash raw (pre-RoPE, post-qk-norm) K straight into the cache, THEN RoPE K in place —
		// same bytes the per-token path's Kraw captures, no extra alloc+copy per layer.
		s.Cache.Kraw[l] = append(s.Cache.Kraw[l], K...)
		parForWork(P, dispatchWorkers, (nH+nKV)*hd, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				ropeRowQKInto(Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w], cosP[t], sinP[t], hd, nH, nKV)
			}
		})

		s.Cache.appendBatchedKV(l, K, V, P, w)
		Kl, Vl := s.Cache.attentionRows(l)

		attnOut := make([]float32, P*nH*hd)
		tA := tic()
		attnPrefillInto(attnOut, Q, Kl, Vl, P, base, nH, hd, w, grp, cfg.windowForLayer(l), l, scale, attnCap, fdot, nil)
		toc(&tAttn, tA)
		pending = attnOut
	}
	viaGraph := false
	if useGraph {
		_, _, _, viaGraph = graphStep(cfg.NumLayers-1, pending, -1)
	}
	if !viaGraph {
		hostLayerOut(cfg.NumLayers-1, pending)
	}

	for t := 0; t < P; t++ {
		s.Cache.appendPosition(base+t, ids[t])
	}
	if qprofOn {
		total := time.Since(t0)
		rest := total - tGemm - tAttn - tQuant - tGraph
		ms := func(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }
		fmt.Fprintf(os.Stderr, "[q4kprof P=%d] total=%.1f  gemm=%.1f  attn=%.1f  quant=%.1f  rest(norm/rope/resid)=%.1f  graph=%.1f (gpu=%.1f cbs=%d) ms\n",
			P, ms(total), ms(tGemm), ms(tAttn), ms(tQuant), ms(rest), ms(tGraph), graphGPUms, graphCBs)
	}
	last := X[(P-1)*H : P*H]
	// finalNorm, not a hand-rolled normCfg: it is the ONE place the final-norm weight, its
	// optional bias, and eps are bound together, so this lane cannot drift from the per-token
	// path again the way the hard-coded nil bias here did.
	return m.finalNorm(last)
}
