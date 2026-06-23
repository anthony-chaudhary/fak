package model

// verify.go — the single-pass batched + tree-attention VERIFY execution (rung #533 of the
// poly-model serving epic #529; docs/serving/polymodel-prefill-share-plan.md §5/§7). It is
// the throughput half that turns the already-shipped accept DECISION (internal/polymodel:
// AcceptGreedy / AcceptTree) and bit-exact rollback (internal/spec: the ProvisionalSink +
// model.KVCache.Evict) into a real single forward pass over the candidate tokens — instead
// of one bandwidth-bound decode step per candidate.
//
// VerifyForward runs the P candidate tokens in ids through the model in ONE pass and
// returns each panel position's next-token logits (post-head), so the caller takes argmax
// per position and feeds the result to AcceptGreedy (chain) or AcceptTree (tree). Two
// attention shapes share the one forward:
//
//   - CHAIN (the linear / greedy-speculation case): pos == nil and allow == nil. Each
//     panel token attends causally to the committed prefix plus every earlier panel token
//     (honoring the layer's sliding window), exactly like prefillBatched. The per-position
//     logits AND the appended KV are bit-identical to P sequential Session.Step calls
//     (VerifyForward shares prefillBatched's math, proven bit-exact to the per-token loop
//     by TestPrefillBatchedMatchesSerial, and re-asserted directly by
//     TestVerifyForwardMatchesSerial), so it is a lossless drop-in for the sequential verify.
//   - TREE (the Medusa / EAGLE-2 / SpecInfer case): pos gives each panel token's absolute
//     RoPE position and allow(q,k) reports whether panel query q may attend to panel key k.
//     Siblings share a position (base+depth-1) and allow is the ancestor relation, so each
//     node attends only to its ancestor chain + the committed prefix — tree-attention masks.
//     The accepted path is then token-identical to plain greedy decode (its context at every
//     depth IS the greedy context), witnessed in internal/spec.
//
// The committed prefix [0, base) is ALWAYS attended in both shapes (it is real context,
// never masked). The supported regime is the plain f32 PreNorm path (standard GQA + RoPE,
// no backend / quant / MoE / Alibi / SWA-hybrid / Qwen-hybrid / per-layer-RoPE) — the
// regime internal/spec and cmd/polymodelbench use; for the chain, any other regime falls
// back to P sequential Steps (still correct, still returns per-position logits, just not
// single-pass). A tree (allow != nil) needs the masked attention and so requires the
// supported regime (returns nil otherwise).
func (s *Session) VerifyForward(ids []int, pos []int, allow func(q, k int) bool) [][]float32 {
	P := len(ids)
	if P == 0 {
		return nil
	}
	if !verifyForwardBatchedOK(s) {
		if allow != nil {
			return nil // tree verify needs the masked batched attention; unsupported regime
		}
		return s.verifyForwardSequential(ids) // chain fallback: correct, universal, not single-pass
	}
	return s.verifyForwardBatched(ids, pos, allow)
}

// verifyForwardSequential is the always-correct chain fallback: P sequential Steps, each
// returning that position's logits. It is the pre-#533 verify path (one step per
// candidate), kept so VerifyForward never regresses a model the batched path does not cover.
func (s *Session) verifyForwardSequential(ids []int) [][]float32 {
	out := make([][]float32, len(ids))
	for i, id := range ids {
		out[i] = s.Step(id)
	}
	return out
}

// verifyForwardBatchedOK reports whether the batched f32 PreNorm verify path supports this
// session. It mirrors the dispatch in Prefill (kv.go): the plain PreNorm standard path with
// no backend / quant / MoE / Alibi / Qwen-hybrid / non-PreNorm / per-layer-RoPE.
func verifyForwardBatchedOK(s *Session) bool {
	if s.Backend != nil || s.Quant || s.Q4 || s.Q4K || s.Metal || s.PrecisionPolicy != nil {
		return false
	}
	cfg := s.M.Cfg
	if cfg.isGLMMoeDsa() || q8PrefillNeedsTokenLoop(cfg) {
		return false
	}
	return true
}

// verifyForwardBatched is the single-pass forward, structurally a generalization of
// prefillBatched: the embed / projection / RoPE / MLP cores are identical, and the
// attention is (a) the exact contiguous-causal loop from prefillBatched when allow == nil
// (bit-identical, the chain) or (b) an explicit per-query allowed-key set when allow != nil
// (the tree-attention mask). pos (nil ⇒ base..base+P-1) gives each panel token's absolute
// RoPE position; the cache appends P positions with those absolute positions. The final
// norm + head mirror Prefill's head(prefillBatched(...)) per panel token so the chain
// logits are bit-identical to serial head(finalNorm(tokenHidden(...))).
func (s *Session) verifyForwardBatched(ids []int, pos []int, allow func(q, k int) bool) [][]float32 {
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
	if pos == nil {
		pos = make([]int, P)
		for q := 0; q < P; q++ {
			pos[q] = base + q
		}
	}

	embed := m.embedRows()
	X := make([]float32, P*H)
	for q, id := range ids {
		copy(X[q*H:(q+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[q*H:(q+1)*H], cfg)
	}

	cosP := make([][]float32, P)
	sinP := make([][]float32, P)
	for q := 0; q < P; q++ {
		cosP[q], sinP[q] = ropeRow(cfg, pos[q])
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }

		Xn := make([]float32, P*H)
		parFor(P, numWorkers, func(lo, hi int) {
			wIn := m.tensor(lp("input_layernorm.weight"))
			for q := lo; q < hi; q++ {
				copy(Xn[q*H:(q+1)*H], rmsnormCfg(X[q*H:(q+1)*H], wIn, eps, cfg))
			}
		})

		Q := matMulBatch(m.tensor(lp("self_attn.q_proj.weight")), Xn, nH*hd, H, P)
		K := matMulBatch(m.tensor(lp("self_attn.k_proj.weight")), Xn, w, H, P)
		V := matMulBatch(m.tensor(lp("self_attn.v_proj.weight")), Xn, w, H, P)
		for q := 0; q < P; q++ {
			m.applyProjBias(l, Q[q*nH*hd:(q+1)*nH*hd], K[q*w:(q+1)*w], V[q*w:(q+1)*w])
			m.applyLayerQKNorm(l, Q[q*nH*hd:(q+1)*nH*hd], K[q*w:(q+1)*w])
		}

		// Kraw (pre-RoPE, post-qk-norm K) stashed before roping K, exactly like the
		// per-token path, so a later Evict can reposition a survivor in a single rotation.
		Kraw := append([]float32(nil), K...)
		parFor(P, numWorkers, func(lo, hi int) {
			for q := lo; q < hi; q++ {
				ropeRowQKInto(Q[q*nH*hd:(q+1)*nH*hd], K[q*w:(q+1)*w], cosP[q], sinP[q], hd, nH, nKV)
			}
		})

		s.Cache.Kraw[l] = append(s.Cache.Kraw[l], Kraw...)
		s.Cache.K[l] = append(s.Cache.K[l], K...)
		s.Cache.V[l] = append(s.Cache.V[l], V...)
		Kl, Vl := s.Cache.K[l], s.Cache.V[l]

		Wl := cfg.windowForLayer(l)
		attnOut := make([]float32, P*nH*hd)
		if allow == nil {
			// CHAIN: byte-identical to prefillBatched's contiguous-causal attention — query
			// q (absolute base+q) attends to cached keys [j0, base+q] inclusive.
			parFor(P, numWorkers, func(lo, hi int) {
				for q := lo; q < hi; q++ {
					nPos := base + q + 1
					j0 := windowLoContig(nPos, base+q, Wl)
					for h := 0; h < nH; h++ {
						kvh := h / grp
						qh := Q[q*nH*hd+h*hd : q*nH*hd+(h+1)*hd]
						scores := make([]float32, nPos-j0)
						for j := j0; j < nPos; j++ {
							kh := Kl[j*w+kvh*hd : j*w+(kvh+1)*hd]
							scores[j-j0] = dot(qh, kh) * scale
						}
						softcapInPlace(scores, attnCap)
						softmaxInPlace(scores)
						out := attnOut[q*nH*hd+h*hd : q*nH*hd+(h+1)*hd]
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
		} else {
			// TREE: query q attends to the committed prefix [0,base) plus the panel keys
			// allow admits (its ancestor chain). The set is sparse and per-query; iterate
			// it in index order for a deterministic reduction. Siblings are NOT ancestors
			// of each other, so two candidate continuations never attend to one another —
			// the structural difference from the chain.
			parFor(P, numWorkers, func(lo, hi int) {
				for q := lo; q < hi; q++ {
					for h := 0; h < nH; h++ {
						kvh := h / grp
						qh := Q[q*nH*hd+h*hd : q*nH*hd+(h+1)*hd]
						keys := make([]int, 0, base+P)
						for j := 0; j < base; j++ {
							keys = append(keys, j)
						}
						for k := 0; k < P; k++ {
							if allow(q, k) {
								keys = append(keys, base+k)
							}
						}
						scores := make([]float32, len(keys))
						for idx, j := range keys {
							kh := Kl[j*w+kvh*hd : j*w+(kvh+1)*hd]
							scores[idx] = dot(qh, kh) * scale
						}
						softcapInPlace(scores, attnCap)
						softmaxInPlace(scores)
						out := attnOut[q*nH*hd+h*hd : q*nH*hd+(h+1)*hd]
						for idx, j := range keys {
							vh := Vl[j*w+kvh*hd : j*w+(kvh+1)*hd]
							wj := scores[idx]
							for d := 0; d < hd; d++ {
								out[d] += wj * vh[d]
							}
						}
					}
				}
			})
		}

		O := matMulBatch(m.tensor(lp("self_attn.o_proj.weight")), attnOut, H, nH*hd, P)
		for i := range X {
			X[i] += O[i]
		}

		Xn2 := make([]float32, P*H)
		parFor(P, numWorkers, func(lo, hi int) {
			wPost := m.tensor(lp("post_attention_layernorm.weight"))
			for q := lo; q < hi; q++ {
				copy(Xn2[q*H:(q+1)*H], rmsnormCfg(X[q*H:(q+1)*H], wPost, eps, cfg))
			}
		})
		inter := cfg.IntermediateSize
		G := matMulBatch(m.tensor(lp("mlp.gate_proj.weight")), Xn2, inter, H, P)
		U := matMulBatch(m.tensor(lp("mlp.up_proj.weight")), Xn2, inter, H, P)
		for i := range G {
			G[i] = act(G[i], cfg) * U[i]
		}
		Down := matMulBatch(m.tensor(lp("mlp.down_proj.weight")), G, H, inter, P)
		for i := range X {
			X[i] += Down[i]
		}
	}

	for q := 0; q < P; q++ {
		s.Cache.pos = append(s.Cache.pos, pos[q])
	}
	out := make([][]float32, P)
	for q := 0; q < P; q++ {
		out[q] = s.head(m.finalNorm(X[q*H : (q+1)*H]))
	}
	return out
}
