package model

// prefill_batch.go — the prefill performance lane: a BATCHED, parallel prefill that
// fills the same kernel-owned KV cache as the per-token path, but processes all P prompt
// tokens' projections as one GEMM (each weight row read once and reused across all P
// tokens) instead of GEMV-per-token (re-streaming all 537 MB of weights P times). This
// is the structural fix for the 22-147x prefill gap measured in MODEL-BASELINE-RESULTS.md:
// HF/llama.cpp prefill is fast precisely because it batches; fak did not.
//
// It is held BIT-IDENTICAL to the per-token tokenHidden loop by construction (every
// output element is the same sum_i w*x in the same i-order — only the loop nest over
// (token,row) and the assignment of rows to cores is reordered) and enforced by
// TestPrefillBatchedMatchesSerial. So the cache it builds is byte-for-byte what the
// proven per-token Prefill builds, and R2/R3/R14 (which assert exact fak-vs-fak
// identity) stay green whether prefill ran batched or per-token.
//
// "By construction" is a claim about the ARITHMETIC only; it is not self-enforcing, and
// this lane has twice been caught doing the same thing to it — silently omitting a term
// the per-token path applies, on an axis no fixture exercised. First the learned norm
// biases (fixed; TestPrefillBatchedMatchesSerialWithNormBias), then the projection biases
// self_attn.o_proj.bias and mlp.{gate,up,down}_proj.bias (fixed;
// TestPrefillBatchedProjBias). Both classes hid because TestPrefillBatchedMatchesSerial
// runs on an RMSNorm, bias-free SmolLM2 export where the missing term is identically zero,
// so the identity assertion held vacuously — and that test is export-gated (it SKIPs
// without a local .cache SmolLM2 export), so on a bare checkout it does not even run. Any
// term added to blockStep must be added here AND witnessed on a SYNTHETIC fixture that
// actually carries it, or this header is false again.

// prefillBatched ingests `ids` as a batch, appending P positions to the cache starting
// at the current absolute position (Cache.Len(), so a prior Evict() compaction shifts
// these down exactly as the per-token path does), and returns the LAST token's hidden
// vector (post-final-norm, pre-head) — the caller applies the head once.
func (s *Session) prefillBatched(ids []int) []float32 {
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
	base := s.Cache.Len() // absolute position of the first new token

	// embeddings: X is flat [P*H], X[t] is position t's working hidden vector.
	embed := m.embedRows()
	X := make([]float32, P*H)
	for t, id := range ids {
		copy(X[t*H:(t+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[t*H:(t+1)*H], cfg) // Gemma; no-op for Llama
	}

	// precompute RoPE rows for the P absolute positions once.
	cosP := make([][]float32, P)
	sinP := make([][]float32, P)
	for t := 0; t < P; t++ {
		cosP[t], sinP[t] = ropeRow(cfg, base+t)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }

		// pre-attn norm, per position (parallel across tokens). LayerNorm families
		// (StableLM, GPT-NeoX, Falcon, MPT, biased Cohere) carry a learned norm BIAS;
		// normCfg must receive it or this lane silently disagrees with Forward, which
		// passes n.preBias (arch.go:635). rmsnormCfg hard-passes nil and must not be
		// used here.
		Xn := make([]float32, P*H)
		parFor(P, dispatchWorkers, func(lo, hi int) {
			wIn := m.tensor(lp("input_layernorm.weight"))
			bIn := m.tensorOptional(lp("input_layernorm.bias"))
			for t := lo; t < hi; t++ {
				copy(Xn[t*H:(t+1)*H], normCfg(X[t*H:(t+1)*H], wIn, bIn, eps, cfg))
			}
		})

		// batched q/k/v projections: each [P, *].
		Q := matMulBatch(m.tensor(lp("self_attn.q_proj.weight")), Xn, nH*hd, H, P)
		K := matMulBatch(m.tensor(lp("self_attn.k_proj.weight")), Xn, w, H, P)
		V := matMulBatch(m.tensor(lp("self_attn.v_proj.weight")), Xn, w, H, P)
		for t := 0; t < P; t++ {
			m.applyProjBias(l, Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w], V[t*w:(t+1)*w])
			// qk-norm AFTER projection, BEFORE RoPE; no-op for Llama.
			m.applyLayerQKNorm(l, Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w])
		}

		// Kraw (pre-RoPE, post-qk-norm K) must be stashed BEFORE roping K, exactly like the
		// per-token path, so eviction can later reposition a survivor in a single rotation.
		Kraw := append([]float32(nil), K...)
		// RoPE q,k per head at each token's absolute position (parallel across tokens),
		// each row through the shared single-row builder.
		parFor(P, dispatchWorkers, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				ropeRowQKInto(Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w], cosP[t], sinP[t], hd, nH, nKV)
			}
		})

		// append all P positions' K/V (and pre-RoPE Kraw) to the kernel-owned cache.
		Kl, Vl, Wl, attnOut := preparePrefillAttention(s.Cache, l, Kraw, K, V, cfg, P, nH, hd)

		// causal GQA attention for each new position t (absolute base+t), attending to
		// cached keys [j0, base+t] inclusive — identical to the per-token path. Parallel
		// across tokens (each token's softmax reduction stays in-order = bit-identical).
		// SWA: j0 = windowLo for the layer's window (j0=0 → full causal). During prefill the
		// cache is contiguous (pos[j]==j: a prior Evict renumbers pos[i]=i and prefill
		// appends at Cache.Len()), so the index IS the absolute position and the lower
		// bound max(0, base+t-W+1) equals the keyed-off-pos[] bound.
		parFor(P, dispatchWorkers, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				nPos := base + t + 1
				j0 := windowLoContig(nPos, base+t, Wl)
				for h := 0; h < nH; h++ {
					kvh := h / grp
					qh := packedHead(Q, t, nH*hd, h, hd)
					scores := make([]float32, nPos-j0)
					fillAttentionScores(scores, qh, Kl, j0, nPos, w, kvh, hd, scale, dot)
					softcapInPlace(scores, attnCap)
					softmaxInPlace(scores)
					if m.attnObs != nil { // #852: emit the post-softmax row (copy-out, math untouched)
						emitAttnRow(m.attnObs, l, base+t, h, j0, scores)
					}
					out := packedHead(attnOut, t, nH*hd, h, hd)
					accumulateAttentionValues(out, Vl, scores, j0, nPos, w, kvh, hd)
				}
			}
		})

		// batched output projection + residual. The optional self_attn.o_proj.bias must be
		// added per row before the residual, exactly as the per-token path does (kv.go's
		// addBiasIfPresent after mat.mul); dropping it here was a silent numerics divergence
		// on every checkpoint that carries the tensor.
		O := matMulBatch(m.tensor(lp("self_attn.o_proj.weight")), attnOut, H, nH*hd, P)
		for t := 0; t < P; t++ {
			m.addBiasIfPresent(O[t*H:(t+1)*H], lp("self_attn.o_proj.bias"))
		}
		for i := range X {
			X[i] += O[i]
		}

		// MLP (SwiGLU), batched + residual.
		Xn2 := make([]float32, P*H)
		parFor(P, dispatchWorkers, func(lo, hi int) {
			wPost := m.tensor(lp("post_attention_layernorm.weight"))
			bPost := m.tensorOptional(lp("post_attention_layernorm.bias"))
			for t := lo; t < hi; t++ {
				copy(Xn2[t*H:(t+1)*H], normCfg(X[t*H:(t+1)*H], wPost, bPost, eps, cfg))
			}
		})
		Down := m.batchedGatedMLP(lp, Xn2, P, H, cfg.IntermediateSize, cfg)
		for i := range X {
			X[i] += Down[i]
		}
	}

	// record the P new absolute positions, then return the LAST token's normed hidden.
	for t := 0; t < P; t++ {
		s.Cache.appendPosition(base+t, ids[t])
	}
	last := X[(P-1)*H : P*H]
	// finalNorm, not a hand-rolled normCfg: it is the ONE place the final-norm weight,
	// its optional bias, and eps are bound together, so this lane cannot drift from the
	// per-token path again the way the hard-coded nil bias here did.
	return m.finalNorm(last)
}
