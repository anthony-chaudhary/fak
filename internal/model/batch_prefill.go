package model

// PrefillEach ingests each user's (possibly distinct) prompt into that user's own cache and
// returns each user's last-token logits — the distribution over its first generated token.
// Prefill is per-user (prompts have different lengths); the throughput win this file is about
// is in the DECODE phase (StepBatch), which is the memory-bound regime an agent loop lives in.
func (bs *BatchSession) PrefillEach(prompts [][]int) [][]float32 {
	if len(prompts) != len(bs.Seqs) {
		panic("model: PrefillEach prompt count != batch size")
	}
	if P, ok := rectangularPrefillLen(prompts); ok && batchRectFastPathOK(bs.M.Cfg, bs.Quant) {
		if bs.Quant {
			return bs.prefillEachRectQ(prompts, P, true)
		}
		return bs.prefillEachRectF32(prompts, P, true)
	}
	out := make([][]float32, len(prompts))
	for b, p := range prompts {
		out[b] = bs.Seqs[b].Prefill(p)
	}
	return out
}

// PrefillEachNoLogits ingests each user's prompt into its own cache and intentionally skips
// final-token logits when the rectangular fast path can do so. Fleet result-ingest uses this:
// the tool/result tokens must extend KV state, but their post-prefill next-token distribution
// is discarded before the next decode turn starts. Non-PreNorm topologies (and the
// non-rectangular case) fall back to the per-session topology-aware PrefillNoLogits.
func (bs *BatchSession) PrefillEachNoLogits(prompts [][]int) {
	if len(prompts) != len(bs.Seqs) {
		panic("model: PrefillEachNoLogits prompt count != batch size")
	}
	if P, ok := rectangularNoLogitsPrefillLen(prompts); ok && batchRectFastPathOK(bs.M.Cfg, bs.Quant) {
		if bs.Quant {
			bs.prefillEachRectQ(prompts, P, false)
			return
		}
		bs.prefillEachRectF32(prompts, P, false)
		return
	}
	for b, p := range prompts {
		bs.Seqs[b].PrefillNoLogits(p)
	}
}

func rectangularNoLogitsPrefillLen(prompts [][]int) (int, bool) {
	return rectangularPrefillLenMin(prompts, 1)
}

func rectangularPrefillLen(prompts [][]int) (int, bool) {
	return rectangularPrefillLenMin(prompts, 2)
}

// rectangularPrefillLenMin returns the common per-prompt length P (and true) when at least
// minPrompts prompts are present and every prompt is the same non-empty length within the
// rectangular-prefill token cap; otherwise (0,false). The two callers differ only in
// minPrompts (the no-logits path admits a single prompt, the batched path needs >=2).
func rectangularPrefillLenMin(prompts [][]int, minPrompts int) (int, bool) {
	if len(prompts) < minPrompts {
		return 0, false
	}
	P := len(prompts[0])
	if P == 0 || P > batchRectPrefillMaxTokens {
		return 0, false
	}
	for _, p := range prompts[1:] {
		if len(p) != P {
			return 0, false
		}
	}
	return P, true
}

func (bs *BatchSession) prefillEachRectF32(prompts [][]int, P int, wantLogits bool) [][]float32 {
	m, cfg := bs.M, bs.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := cfg.attnScale()
	B, N := len(prompts), len(prompts)*P

	baseB, caches, cosN, sinN := bs.rectPrefillGeometry(P)
	embed := m.embedRows()
	X := make([]float32, N*H)
	for b, p := range prompts {
		for t, id := range p {
			row := b*P + t
			copy(X[row*H:(row+1)*H], embed[id*H:(id+1)*H])
			scaleEmbedInPlace(X[row*H:(row+1)*H], cfg) // Gemma; no-op for Llama
		}
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(s string) string { return layerName(l, s) }

		Xn := make([]float32, N*H)
		wIn := m.tensor(lp("input_layernorm.weight"))
		parFor(N, numWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				copy(Xn[row*H:(row+1)*H], rmsnormCfg(X[row*H:(row+1)*H], wIn, eps, cfg))
			}
		})

		Q := matMulBatch(m.tensor(lp("self_attn.q_proj.weight")), Xn, nH*hd, H, N)
		K := matMulBatch(m.tensor(lp("self_attn.k_proj.weight")), Xn, w, H, N)
		V := matMulBatch(m.tensor(lp("self_attn.v_proj.weight")), Xn, w, H, N)
		for row := 0; row < N; row++ {
			m.applyProjBias(l, Q[row*nH*hd:(row+1)*nH*hd], K[row*w:(row+1)*w], V[row*w:(row+1)*w])
			m.applyLayerQKNorm(l, Q[row*nH*hd:(row+1)*nH*hd], K[row*w:(row+1)*w])
		}

		for b, c := range caches {
			for t := 0; t < P; t++ {
				row := b*P + t
				c.Kraw[l] = append(c.Kraw[l], K[row*w:(row+1)*w]...)
			}
		}
		parFor(N, numWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				ropeRowQKInto(Q[row*nH*hd:(row+1)*nH*hd], K[row*w:(row+1)*w], cosN[row], sinN[row], hd, nH, nKV)
			}
		})
		for b, c := range caches {
			for t := 0; t < P; t++ {
				row := b*P + t
				c.K[l] = append(c.K[l], K[row*w:(row+1)*w]...)
				c.V[l] = append(c.V[l], V[row*w:(row+1)*w]...)
			}
		}

		// F32 prefill keeps the plain (allocating) attention path: the windowed
		// attnPrefillMultiInto covers both full-causal (W<0) and SWA layers. The fresh make
		// below starts attnOut zeroed (the saxpy accumulation requires it), and nil scratch
		// means no pooling here — the pooled GQA-fused path is the Q8 hot lane only.
		attnOut := make([]float32, N*nH*hd)
		attnPrefillMultiInto(attnOut, Q, caches, baseB, l, P, nH, hd, w, grp, cfg.windowForLayer(l), scale, dot, nil)

		O := matMulBatch(m.tensor(lp("self_attn.o_proj.weight")), attnOut, H, nH*hd, N)
		for row := 0; row < N; row++ {
			m.addBiasIfPresent(O[row*H:(row+1)*H], lp("self_attn.o_proj.bias"))
		}
		for i := range X {
			X[i] += O[i]
		}

		Xn2 := make([]float32, N*H)
		wPost := m.tensor(lp("post_attention_layernorm.weight"))
		parFor(N, numWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				copy(Xn2[row*H:(row+1)*H], rmsnormCfg(X[row*H:(row+1)*H], wPost, eps, cfg))
			}
		})
		I := cfg.IntermediateSize
		G := matMulBatch(m.tensor(lp("mlp.gate_proj.weight")), Xn2, I, H, N)
		U := matMulBatch(m.tensor(lp("mlp.up_proj.weight")), Xn2, I, H, N)
		for row := 0; row < N; row++ {
			m.addBiasIfPresent(G[row*I:(row+1)*I], lp("mlp.gate_proj.bias"))
			m.addBiasIfPresent(U[row*I:(row+1)*I], lp("mlp.up_proj.bias"))
		}
		for i := range G {
			G[i] = act(G[i], cfg) * U[i]
		}
		Down := matMulBatch(m.tensor(lp("mlp.down_proj.weight")), G, H, I, N)
		for row := 0; row < N; row++ {
			m.addBiasIfPresent(Down[row*H:(row+1)*H], lp("mlp.down_proj.bias"))
		}
		for i := range X {
			X[i] += Down[i]
		}
	}

	bs.finishRectPrefillPositions(baseB, P)
	if !wantLogits {
		return nil
	}
	Xnorm := make([]float32, B*H)
	normW := m.tensor("model.norm.weight")
	for b := 0; b < B; b++ {
		row := b*P + P - 1
		copy(Xnorm[b*H:(b+1)*H], rmsnormCfg(X[row*H:(row+1)*H], normW, eps, cfg))
	}
	Logits := matMulBatch(m.lmHead(), Xnorm, cfg.VocabSize, H, B)
	out := splitLogits(Logits, B, cfg.VocabSize)
	for b := range out {
		logitScaleInPlace(out[b], cfg) // Cohere/Gemma2; no-op for Llama
	}
	return out
}

func (bs *BatchSession) prefillEachRectQ(prompts [][]int, P int, wantLogits bool) [][]float32 {
	m, cfg := bs.M, bs.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := cfg.attnScale()
	B, N := len(prompts), len(prompts)*P
	if bs.scratch == nil {
		bs.scratch = &q8Panel{}
	}
	if bs.pbuf == nil {
		bs.pbuf = &batchRectPrefillBuf{}
	}
	pb := bs.pbuf

	baseB := growInts(pb.base, B)
	pb.base = baseB
	caches := growCaches(pb.caches, B)
	pb.caches = caches
	cosN := grow2D(pb.cos, N, hd/2)
	pb.cos = cosN
	sinN := grow2D(pb.sin, N, hd/2)
	pb.sin = sinN
	inv := cachedInvFreq(cfg, 0)
	for b, s := range bs.Seqs {
		baseB[b] = s.Cache.Len()
		caches[b] = s.Cache
		for t := 0; t < P; t++ {
			row := b*P + t
			ropeRowInto(cosN[row], sinN[row], inv, baseB[b]+t)
		}
	}
	embed := m.embedRows()
	X := grow(pb.X, N*H)
	pb.X = X
	for b, p := range prompts {
		for t, id := range p {
			row := b*P + t
			copy(X[row*H:(row+1)*H], embed[id*H:(id+1)*H])
			scaleEmbedInPlace(X[row*H:(row+1)*H], cfg) // Gemma; no-op for Llama
		}
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(s string) string { return layerName(l, s) }
		ql := m.q8Layer(l)

		Xn := grow(pb.Xn, N*H)
		pb.Xn = Xn
		wIn := m.tensor(lp("input_layernorm.weight"))
		parFor(N, numWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn[row*H:(row+1)*H], rmsnormCfg(X[row*H:(row+1)*H], wIn, eps, cfg))
				} else {
					rmsnormInto(Xn[row*H:(row+1)*H], X[row*H:(row+1)*H], wIn, eps)
				}
			}
		})
		quantizeBatchPanelInto(bs.scratch, Xn, N, H)
		// Fused q/k/v: one quantized Xn panel drives three tile GEMMs into pooled dsts (perf,
		// numerically identical to three separate qGemm8 calls). Bias + QK-norm are applied
		// via the config-driven helpers so non-Llama archs (AttentionBias, QKNorm) are correct.
		Q := grow(pb.Q, N*nH*hd)
		pb.Q = Q
		K := grow(pb.K, N*w)
		pb.K = K
		V := grow(pb.V, N*w)
		pb.V = V
		qGemm8IntoMany(bs.scratch,
			qgemm8Target{qt: ql.qProj, Y: Q},
			qgemm8Target{qt: ql.kProj, Y: K},
			qgemm8Target{qt: ql.vProj, Y: V},
		)
		for row := 0; row < N; row++ {
			m.applyProjBias(l, Q[row*nH*hd:(row+1)*nH*hd], K[row*w:(row+1)*w], V[row*w:(row+1)*w])
			m.applyLayerQKNorm(l, Q[row*nH*hd:(row+1)*nH*hd], K[row*w:(row+1)*w])
		}

		for b, c := range caches {
			for t := 0; t < P; t++ {
				row := b*P + t
				c.Kraw[l] = append(c.Kraw[l], K[row*w:(row+1)*w]...)
			}
		}
		parFor(N, numWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				ropeRowQKInto(Q[row*nH*hd:(row+1)*nH*hd], K[row*w:(row+1)*w], cosN[row], sinN[row], hd, nH, nKV)
			}
		})
		for b, c := range caches {
			for t := 0; t < P; t++ {
				row := b*P + t
				c.K[l] = append(c.K[l], K[row*w:(row+1)*w]...)
				c.V[l] = append(c.V[l], V[row*w:(row+1)*w]...)
			}
		}

		// Attention. pb.attn is a reused buffer and the helper += accumulates into it, so it
		// must be cleared first. The GQA-fused helper carries the layer window bound.
		attnOut := grow(pb.attn, N*nH*hd)
		pb.attn = attnOut
		clear(attnOut)
		scoreDot3 := fdot3scalar
		if attnFdot3SIMD && B >= attnFdot3SIMDMinBatch {
			scoreDot3 = fdot3SIMD
		}
		pb.scores = attnPrefillMultiGQAInto(attnOut, Q, caches, baseB, l, P, nH, hd, w, grp, cfg.windowForLayer(l), scale, fdot, scoreDot3, pb.scores)

		quantizeBatchPanelInto(bs.scratch, attnOut, N, nH*hd)
		O := grow(pb.O, N*H)
		pb.O = O
		qGemm8Into(ql.oProj, bs.scratch, O)
		for row := 0; row < N; row++ {
			m.addBiasIfPresent(O[row*H:(row+1)*H], lp("self_attn.o_proj.bias"))
		}
		for i := range X {
			X[i] += O[i]
		}

		Xn2 := grow(pb.Xn2, N*H)
		pb.Xn2 = Xn2
		wPost := m.tensor(lp("post_attention_layernorm.weight"))
		parFor(N, numWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn2[row*H:(row+1)*H], rmsnormCfg(X[row*H:(row+1)*H], wPost, eps, cfg))
				} else {
					rmsnormInto(Xn2[row*H:(row+1)*H], X[row*H:(row+1)*H], wPost, eps)
				}
			}
		})
		I := cfg.IntermediateSize
		quantizeBatchPanelInto(bs.scratch, Xn2, N, H)
		// Fused gate/up GEMM (perf) into pooled dsts, then the config-driven activation. For
		// Llama act==silu so `act(G)*U` is byte-identical to swigluInPlace(G,U); for Gemma it
		// is the correct GeGLU. The fused GEMM is orthogonal to the activation choice.
		G := grow(pb.G, N*I)
		pb.G = G
		U := grow(pb.U, N*I)
		pb.U = U
		qGemm8IntoMany(bs.scratch,
			qgemm8Target{qt: ql.gateProj, Y: G},
			qgemm8Target{qt: ql.upProj, Y: U},
		)
		for row := 0; row < N; row++ {
			m.addBiasIfPresent(G[row*I:(row+1)*I], lp("mlp.gate_proj.bias"))
			m.addBiasIfPresent(U[row*I:(row+1)*I], lp("mlp.up_proj.bias"))
		}
		for i := range G {
			G[i] = act(G[i], cfg) * U[i]
		}
		quantizeBatchPanelInto(bs.scratch, G, N, I)
		Down := grow(pb.Down, N*H)
		pb.Down = Down
		qGemm8Into(ql.downProj, bs.scratch, Down)
		for row := 0; row < N; row++ {
			m.addBiasIfPresent(Down[row*H:(row+1)*H], lp("mlp.down_proj.bias"))
		}
		for i := range X {
			X[i] += Down[i]
		}
	}

	bs.finishRectPrefillPositions(baseB, P)
	if !wantLogits {
		return nil
	}
	Xnorm := grow(pb.Xnorm, B*H)
	pb.Xnorm = Xnorm
	normW := m.tensor("model.norm.weight")
	for b := 0; b < B; b++ {
		row := b*P + P - 1
		if cfg.NormGain1p || cfg.LayerNorm {
			copy(Xnorm[b*H:(b+1)*H], rmsnormCfg(X[row*H:(row+1)*H], normW, eps, cfg))
		} else {
			rmsnormInto(Xnorm[b*H:(b+1)*H], X[row*H:(row+1)*H], normW, eps)
		}
	}
	quantizeBatchPanelInto(bs.scratch, Xnorm, B, H)
	Logits := qGemm8(m.q8(m.headName()), bs.scratch)
	out := splitLogits(Logits, B, cfg.VocabSize)
	for b := range out {
		logitScaleInPlace(out[b], cfg) // Cohere/Gemma2; no-op for Llama
	}
	return out
}

func (bs *BatchSession) rectPrefillGeometry(P int) ([]int, []*KVCache, [][]float32, [][]float32) {
	B := len(bs.Seqs)
	baseB := make([]int, B)
	caches := make([]*KVCache, B)
	cosN := make([][]float32, B*P)
	sinN := make([][]float32, B*P)
	for b, s := range bs.Seqs {
		baseB[b] = s.Cache.Len()
		caches[b] = s.Cache
		for t := 0; t < P; t++ {
			row := b*P + t
			cosN[row], sinN[row] = ropeRow(bs.M.Cfg, baseB[b]+t)
		}
	}
	return baseB, caches, cosN, sinN
}

func (bs *BatchSession) finishRectPrefillPositions(baseB []int, P int) {
	for b, s := range bs.Seqs {
		for t := 0; t < P; t++ {
			s.Cache.pos = append(s.Cache.pos, baseB[b]+t)
		}
	}
}

func splitLogits(logits []float32, B, vocab int) [][]float32 {
	out := make([][]float32, B)
	for b := 0; b < B; b++ {
		out[b] = logits[b*vocab : (b+1)*vocab]
	}
	return out
}
