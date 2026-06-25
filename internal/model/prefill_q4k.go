package model

// prefill_q4k.go — the resident-Q4_K PREFILL lane: a BATCHED prefill that is the structural
// twin of prefillBatchedQ (quant_forward.go), differing in exactly ONE way — each per-layer
// projection GEMM dispatches by resident format, exactly as the per-token sessionQ4KKernel
// does for decode:
//
//   - the q4_k_m majority (self_attn.v_proj/o_proj, mlp.gate/up/down — every identity-
//     normalized matmul weight the loader held raw) runs q4kGemm, the batched Q4_K GEMM,
//     on the f32 activation. Each weight super-block is dequantized ONCE and reused across
//     all P prompt tokens, instead of re-streaming the whole weight matrix P times as the
//     per-token GEMV prefill does. Prefill is compute-bound, so amortizing the dequant +
//     weight bandwidth across the P free axes is what closes the gap to llama.cpp-Metal on
//     the q4_k_m artifact (QWEN36-NATIVE-PERF-PLAN-2026-06-19.md P3).
//   - the normalize-sensitive / Q6_K minority (self_attn.q_proj/k_proj always, plus any Q6_K
//     weight such as mlp.down_proj or lm_head in a q4_k_m mix) runs the proven batched Q8
//     GEMM qGemm8 against the q8w store — the SAME path prefillBatchedQ takes, on a Q8-
//     quantized activation panel.
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

	var tQuant, tGemm, tAttn time.Duration
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
	// proj dispatches a batched projection [P,out] by resident format: q4kw-resident →
	// q4kGemm on the f32 activation Xf; otherwise → qGemm8 on the Q8 panel Xq. The width is
	// inferred from the resident tensor's .out, so the caller does not pass it. Xq may be
	// nil when the caller knows the projection is q4k-resident (it is only read on the q8
	// branch); passing the matching panel is the caller's responsibility for minority names.
	proj := func(name string, Xf []float32, Xq *q8Panel) []float32 {
		t := tic()
		var r []float32
		if qt := m.q4kw[name]; qt != nil {
			r = q4kGemm(qt, Xf, P)
		} else {
			r = qGemm8(m.q8(name), Xq)
		}
		toc(&tGemm, t)
		return r
	}

	embed := m.embedRows()
	X := make([]float32, P*H)
	for t, id := range ids {
		copy(X[t*H:(t+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[t*H:(t+1)*H], cfg) // Gemma; no-op for Llama/Qwen
	}

	cosP := make([][]float32, P)
	sinP := make([][]float32, P)
	for t := 0; t < P; t++ {
		cosP[t], sinP[t] = ropeRow(cfg, base+t)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }

		Xn := make([]float32, P*H)
		parFor(P, numWorkers, func(lo, hi int) {
			wIn := m.tensor(lp("input_layernorm.weight"))
			for t := lo; t < hi; t++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn[t*H:(t+1)*H], rmsnormCfg(X[t*H:(t+1)*H], wIn, eps, cfg))
				} else {
					rmsnormInto(Xn[t*H:(t+1)*H], X[t*H:(t+1)*H], wIn, eps)
				}
			}
		})
		// Xnq feeds the Q8-minority projections (q/k at minimum). The q4_k_m majority reads
		// raw f32 Xn directly, so the panel is built once and consumed by whichever of
		// q/k/v are on the Q8 path; a q4k-resident v just ignores it.
		Xnq := qz(Xn, P, H)

		Q := proj(lp("self_attn.q_proj.weight"), Xn, Xnq)
		K := proj(lp("self_attn.k_proj.weight"), Xn, Xnq)
		V := proj(lp("self_attn.v_proj.weight"), Xn, Xnq)
		for t := 0; t < P; t++ {
			m.applyProjBias(l, Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w], V[t*w:(t+1)*w])
			m.applyLayerQKNorm(l, Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w])
		}

		// Stash raw (pre-RoPE, post-qk-norm) K straight into the cache, THEN RoPE K in place —
		// same bytes the per-token path's Kraw captures, no extra alloc+copy per layer.
		s.Cache.Kraw[l] = append(s.Cache.Kraw[l], K...)
		parFor(P, numWorkers, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				ropeRowQKInto(Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w], cosP[t], sinP[t], hd, nH, nKV)
			}
		})

		s.Cache.K[l] = append(s.Cache.K[l], K...)
		s.Cache.V[l] = append(s.Cache.V[l], V...)
		Kl, Vl := s.Cache.K[l], s.Cache.V[l]

		attnOut := make([]float32, P*nH*hd)
		tA := tic()
		attnPrefillInto(attnOut, Q, Kl, Vl, P, base, nH, hd, w, grp, cfg.windowForLayer(l), l, scale, attnCap, fdot, nil)
		toc(&tAttn, tA)

		O := proj(lp("self_attn.o_proj.weight"), attnOut, qz(attnOut, P, nH*hd))
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(O[t*H:(t+1)*H], lp("self_attn.o_proj.bias"))
		}
		parFor(len(X), numWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += O[i]
			}
		})

		Xn2 := make([]float32, P*H)
		parFor(P, numWorkers, func(lo, hi int) {
			wPost := m.tensor(lp("post_attention_layernorm.weight"))
			for t := lo; t < hi; t++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn2[t*H:(t+1)*H], rmsnormCfg(X[t*H:(t+1)*H], wPost, eps, cfg))
				} else {
					rmsnormInto(Xn2[t*H:(t+1)*H], X[t*H:(t+1)*H], wPost, eps)
				}
			}
		})
		I := cfg.IntermediateSize
		Xn2q := qz(Xn2, P, H)
		G := proj(lp("mlp.gate_proj.weight"), Xn2, Xn2q)
		U := proj(lp("mlp.up_proj.weight"), Xn2, Xn2q)
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(G[t*I:(t+1)*I], lp("mlp.gate_proj.bias"))
			m.addBiasIfPresent(U[t*I:(t+1)*I], lp("mlp.up_proj.bias"))
		}
		parFor(len(G), numWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				G[i] = act(G[i], cfg) * U[i]
			}
		})
		Down := proj(lp("mlp.down_proj.weight"), G, qz(G, P, I))
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(Down[t*H:(t+1)*H], lp("mlp.down_proj.bias"))
		}
		parFor(len(X), numWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += Down[i]
			}
		})
	}

	for t := 0; t < P; t++ {
		s.Cache.pos = append(s.Cache.pos, base+t)
	}
	if qprofOn {
		total := time.Since(t0)
		rest := total - tGemm - tAttn - tQuant
		ms := func(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }
		fmt.Fprintf(os.Stderr, "[q4kprof P=%d] total=%.1f  gemm=%.1f  attn=%.1f  quant=%.1f  rest(norm/rope/resid)=%.1f ms\n",
			P, ms(total), ms(tGemm), ms(tAttn), ms(tQuant), ms(rest))
	}
	last := X[(P-1)*H : P*H]
	return rmsnormCfg(last, m.tensor("model.norm.weight"), eps, cfg)
}
