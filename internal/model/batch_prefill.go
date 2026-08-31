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
	dispatchWorkers := currentWorkerCount()
	m, cfg := bs.M, bs.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := cfg.attnScale()
	B, N := len(prompts), len(prompts)*P

	baseB, caches, cosN, sinN := bs.rectPrefillGeometry(P)
	X := make([]float32, N*H)
	m.embedRectRowsInto(X, prompts, P, H, cfg)

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(s string) string { return layerName(l, s) }

		// LayerNorm families (StableLM, GPT-NeoX, Falcon, biased Cohere) carry a learned norm
		// BIAS; normCfg must receive it or this lane silently disagrees with the per-token path,
		// which passes n.preBias (arch.go:635). rmsnormCfg hard-passes nil and must not be used
		// here — batchPreNormFastPathOK (batch.go:109) has no LayerNorm term, so a PreNorm
		// LayerNorm arch routes straight into this lane.
		Xn := make([]float32, N*H)
		wIn := m.tensor(lp("input_layernorm.weight"))
		bIn := m.tensorOptional(lp("input_layernorm.bias"))
		parFor(N, dispatchWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				copy(Xn[row*H:(row+1)*H], normCfg(X[row*H:(row+1)*H], wIn, bIn, eps, cfg))
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
		parFor(N, dispatchWorkers, func(lo, hi int) {
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
		bPost := m.tensorOptional(lp("post_attention_layernorm.bias"))
		parFor(N, dispatchWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				copy(Xn2[row*H:(row+1)*H], normCfg(X[row*H:(row+1)*H], wPost, bPost, eps, cfg))
			}
		})
		Down := m.batchedGatedMLP(lp, Xn2, N, H, cfg.IntermediateSize, cfg)
		for i := range X {
			X[i] += Down[i]
		}
	}

	bs.finishRectPrefillPositions(baseB, prompts, P)
	if !wantLogits {
		return nil
	}
	Xnorm := make([]float32, B*H)
	for b := 0; b < B; b++ {
		row := b*P + P - 1
		// finalNorm, not a hand-rolled normCfg: it is the ONE place the final-norm weight, its
		// optional bias, and eps are bound together, so this lane cannot drift from the per-token
		// path again the way the hard-coded nil bias here did.
		copy(Xnorm[b*H:(b+1)*H], m.finalNorm(X[row*H:(row+1)*H]))
	}
	Logits := matMulBatch(m.lmHead(), Xnorm, cfg.VocabSize, H, B)
	return splitScaledLogits(nil, Logits, B, cfg.VocabSize, cfg)
}

func (bs *BatchSession) prefillEachRectQ(prompts [][]int, P int, wantLogits bool) [][]float32 {
	dispatchWorkers := currentWorkerCount()
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
	X := grow(pb.X, N*H)
	pb.X = X
	m.embedRectRowsInto(X, prompts, P, H, cfg)

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(s string) string { return layerName(l, s) }
		ql := m.q8Layer(l)

		Xn := grow(pb.Xn, N*H)
		pb.Xn = Xn
		wIn := m.tensor(lp("input_layernorm.weight"))
		// The bias is nil for every RMSNorm arch, so this is a no-op there; it matters only on the
		// cfg.LayerNorm disjunct of the branch below, which q8FastPreNormOK (quant_forward.go:145)
		// currently keeps out of this quantized lane. Passing it anyway keeps the norm call
		// identical to the f32 twin, so relaxing that gate can never silently drop the bias.
		bIn := m.tensorOptional(lp("input_layernorm.bias"))
		parFor(N, dispatchWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn[row*H:(row+1)*H], normCfg(X[row*H:(row+1)*H], wIn, bIn, eps, cfg))
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
		parFor(N, dispatchWorkers, func(lo, hi int) {
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
		bPost := m.tensorOptional(lp("post_attention_layernorm.bias"))
		parFor(N, dispatchWorkers, func(lo, hi int) {
			for row := lo; row < hi; row++ {
				if cfg.NormGain1p || cfg.LayerNorm {
					copy(Xn2[row*H:(row+1)*H], normCfg(X[row*H:(row+1)*H], wPost, bPost, eps, cfg))
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
		m.fuseGatedMLPPanels(lp, G, U, N, I, cfg)
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

	bs.finishRectPrefillPositions(baseB, prompts, P)
	if !wantLogits {
		return nil
	}
	Xnorm := grow(pb.Xnorm, B*H)
	pb.Xnorm = Xnorm
	normW := m.tensor("model.norm.weight")
	for b := 0; b < B; b++ {
		row := b*P + P - 1
		if cfg.NormGain1p || cfg.LayerNorm {
			copy(Xnorm[b*H:(b+1)*H], m.finalNorm(X[row*H:(row+1)*H]))
		} else {
			rmsnormInto(Xnorm[b*H:(b+1)*H], X[row*H:(row+1)*H], normW, eps)
		}
	}
	quantizeBatchPanelInto(bs.scratch, Xnorm, B, H)
	Logits := qGemm8(m.q8(m.headName()), bs.scratch)
	return splitScaledLogits(nil, Logits, B, cfg.VocabSize, cfg)
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

func (bs *BatchSession) finishRectPrefillPositions(baseB []int, prompts [][]int, P int) {
	for b, s := range bs.Seqs {
		for t := 0; t < P; t++ {
			s.Cache.appendPosition(baseB[b]+t, prompts[b][t])
		}
	}
}

// splitScaledLogits slices the flat [B,vocab] batched logits into B ALIASING per-sequence
// rows (no copy) and applies the config's logit scale to each (Cohere/Gemma2; no-op for
// Llama). All four batched lanes — rect prefill f32/Q8 and batched decode f32/Q8 — end
// with this same slice-then-scale tail, which used to be four copies of the loop. `rows`
// lets the decode lanes hand in their reused row-header buffer instead of allocating; a
// nil/short `rows` is allocated fresh, exactly as the allocating callers did.
func splitScaledLogits(rows [][]float32, logits []float32, B, vocab int, cfg Config) [][]float32 {
	if len(rows) < B {
		rows = make([][]float32, B)
	}
	for b := 0; b < B; b++ {
		rows[b] = logits[b*vocab : (b+1)*vocab]
		logitScaleInPlace(rows[b], cfg)
	}
	return rows
}
