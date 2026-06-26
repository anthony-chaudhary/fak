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

// prefillBatched ingests `ids` as a batch, appending P positions to the cache starting
// at the current absolute position (Cache.Len(), so a prior Evict() compaction shifts
// these down exactly as the per-token path does), and returns the LAST token's hidden
// vector (post-final-norm, pre-head) — the caller applies the head once.
func (s *Session) prefillBatched(ids []int) []float32 {
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

		// pre-attn norm, per position (parallel across tokens).
		Xn := make([]float32, P*H)
		parFor(P, numWorkers, func(lo, hi int) {
			wIn := m.tensor(lp("input_layernorm.weight"))
			for t := lo; t < hi; t++ {
				copy(Xn[t*H:(t+1)*H], rmsnormCfg(X[t*H:(t+1)*H], wIn, eps, cfg))
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
		parFor(P, numWorkers, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				ropeRowQKInto(Q[t*nH*hd:(t+1)*nH*hd], K[t*w:(t+1)*w], cosP[t], sinP[t], hd, nH, nKV)
			}
		})

		// append all P positions' K/V (and pre-RoPE Kraw) to the kernel-owned cache.
		s.Cache.Kraw[l] = append(s.Cache.Kraw[l], Kraw...)
		s.Cache.K[l] = append(s.Cache.K[l], K...)
		s.Cache.V[l] = append(s.Cache.V[l], V...)
		Kl, Vl := s.Cache.K[l], s.Cache.V[l]

		// causal GQA attention for each new position t (absolute base+t), attending to
		// cached keys [j0, base+t] inclusive — identical to the per-token path. Parallel
		// across tokens (each token's softmax reduction stays in-order = bit-identical).
		// SWA: j0 = windowLo for the layer's window (j0=0 → full causal). During prefill the
		// cache is contiguous (pos[j]==j: a prior Evict renumbers pos[i]=i and prefill
		// appends at Cache.Len()), so the index IS the absolute position and the lower
		// bound max(0, base+t-W+1) equals the keyed-off-pos[] bound.
		Wl := cfg.windowForLayer(l)
		attnOut := make([]float32, P*nH*hd)
		parFor(P, numWorkers, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				nPos := base + t + 1
				j0 := windowLoContig(nPos, base+t, Wl)
				for h := 0; h < nH; h++ {
					kvh := h / grp
					qh := Q[t*nH*hd+h*hd : t*nH*hd+(h+1)*hd]
					scores := make([]float32, nPos-j0)
					for j := j0; j < nPos; j++ {
						kh := Kl[j*w+kvh*hd : j*w+(kvh+1)*hd]
						scores[j-j0] = dot(qh, kh) * scale
					}
					softcapInPlace(scores, attnCap)
					softmaxInPlace(scores)
					if m.attnObs != nil { // #852: emit the post-softmax row (copy-out, math untouched)
						emitAttnRow(m.attnObs, l, base+t, h, j0, scores)
					}
					out := attnOut[t*nH*hd+h*hd : t*nH*hd+(h+1)*hd]
					for j := j0; j < nPos; j++ {
						vh := Vl[j*w+kvh*hd : j*w+(kvh+1)*hd]
						wj := scores[j-j0]
						for d := 0; d < hd; d++ {
							out[d] += wj * vh[d]
						}
					}
				}
			}
		})

		// batched output projection + residual.
		O := matMulBatch(m.tensor(lp("self_attn.o_proj.weight")), attnOut, H, nH*hd, P)
		for i := range X {
			X[i] += O[i]
		}

		// MLP (SwiGLU), batched + residual.
		Xn2 := make([]float32, P*H)
		parFor(P, numWorkers, func(lo, hi int) {
			wPost := m.tensor(lp("post_attention_layernorm.weight"))
			for t := lo; t < hi; t++ {
				copy(Xn2[t*H:(t+1)*H], rmsnormCfg(X[t*H:(t+1)*H], wPost, eps, cfg))
			}
		})
		I := cfg.IntermediateSize
		G := matMulBatch(m.tensor(lp("mlp.gate_proj.weight")), Xn2, I, H, P)
		U := matMulBatch(m.tensor(lp("mlp.up_proj.weight")), Xn2, I, H, P)
		for i := range G {
			G[i] = act(G[i], cfg) * U[i]
		}
		Down := matMulBatch(m.tensor(lp("mlp.down_proj.weight")), G, H, I, P)
		for i := range X {
			X[i] += Down[i]
		}
	}

	// record the P new absolute positions, then return the LAST token's normed hidden.
	for t := 0; t < P; t++ {
		s.Cache.pos = append(s.Cache.pos, base+t)
	}
	last := X[(P-1)*H : P*H]
	return rmsnormCfg(last, m.tensor("model.norm.weight"), eps, cfg)
}
