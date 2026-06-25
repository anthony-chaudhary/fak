package model

// StepBatch decodes ONE token for each user (ids[b] is user b's already-chosen token) and
// returns B next-token logit vectors. In the f32 path each user's logits are bit-for-bit
// identical to bs.Seqs[b].Step(ids[b]) run serially; the only difference is that the weight
// stream is shared across all B users instead of re-streamed per user.
func (bs *BatchSession) StepBatch(ids []int) [][]float32 {
	if len(ids) != len(bs.Seqs) {
		panic("model: StepBatch id count != batch size")
	}
	// Reset the per-step MAC count: a serial-fallback step (a single lane, or a config the
	// batched fast path does not cover) does no shared-weight batched projection work, so it
	// honestly reports 0 rather than the previous call's stale count. stepBatchF32/Q set it.
	bs.lastStepMACs = 0
	if len(ids) == 1 || !batchDecodeFastPathOK(bs.M.Cfg, bs.Quant) {
		if len(ids) == 1 {
			return [][]float32{bs.Seqs[0].Step(ids[0])}
		}
		out := make([][]float32, len(ids))
		for b, id := range ids {
			out[b] = bs.Seqs[b].Step(id)
		}
		return out
	}
	if bs.Quant {
		return bs.stepBatchQ(ids)
	}
	return bs.stepBatchF32(ids)
}

// LastStepMACs returns the exact multiply-accumulate count of the B-proportional projection
// GEMMs (QKV + attention-output + SwiGLU gate/up/down + the LM head) performed by the most
// recent StepBatch / StepBatchActive. This is the weight-streaming decode work the batch
// dimension amortises and the ragged (active-lane-masked) path compacts: it scales EXACTLY
// with the active batch size, so a fleet with K of C lanes idle reports (C−K)/C of the
// full-batch count — the work-elimination the ragged path buys, witnessed as an exact integer
// ratio (TestRaggedBatchIdleLaneSkip). Closed form from the model shape and the active B,
// computed once per step; it does NOT include attention (per-user, cache-length-dependent),
// so the projection count is the precise lever domain and yields the exact ratio #520 names.
func (bs *BatchSession) LastStepMACs() int64 { return bs.lastStepMACs }

// recordStepMACs stamps the B-proportional projection MAC count for a step of B active lanes.
// Per layer the seven projections (q/k/v/o + gate/up/down) cost H·(2·nH·hd + 2·nKV·hd + 3·I)
// MACs, the head costs H·VocabSize, and the panel multiplies both by B.
func (bs *BatchSession) recordStepMACs(B int) {
	cfg := bs.M.Cfg
	H := int64(cfg.HiddenSize)
	perLayer := H * (2*int64(cfg.NumHeads)*int64(cfg.HeadDim) +
		2*int64(cfg.NumKVHeads)*int64(cfg.HeadDim) +
		3*int64(cfg.IntermediateSize))
	head := H * int64(cfg.VocabSize)
	bs.lastStepMACs = int64(B) * (int64(cfg.NumLayers)*perLayer + head)
}

// StepBatchActive is the ragged-batch decode step (#520): it decodes ONLY the lanes whose
// active[b] is true, compacting the batch dimension to the active count so a fleet with K of
// C lanes idle this turn does (C−K)/C of the full batch's decode work. Idle lanes are left
// untouched — no KV append, no position advance, no logits — so a lane blocked on a tool,
// finished (EOS), or simply idle this turn can be reactivated on a later step without paying
// for work it did not need. out[b] is the next-token logits for an active lane b and nil for
// an idle one; a caller consumes the active logits before the next call (same aliasing
// discipline as StepBatch).
//
// Correctness is the all-active invariant lifted to a subset. The compaction gathers the
// active lanes into a contiguous sub-panel and runs the EXACT stepBatchF32/stepBatchQ
// machinery over it (the same weight-stream-sharing GEMMs, the same per-user attention),
// sharing each active lane's own *Session so the KV append and position advance land in the
// right per-lane cache. For every active lane b the returned logits and the appended K/V/pos
// are therefore bit-for-bit identical to a full StepBatch run with all lanes active and the
// idle lanes' results discarded — compaction changes only WHICH rows share the weight stream,
// never a single rounding (the same property that makes StepBatch bit-identical to serial
// Step). When every lane is active this IS StepBatch (byte-identical), which the witness pins.
// Per-agent KV ownership is preserved by construction: each lane's cache is its own object,
// appended only when active, so Evict/Clone behave exactly as in the serial path.
func (bs *BatchSession) StepBatchActive(ids []int, active []bool) [][]float32 {
	C := len(bs.Seqs)
	if len(ids) != C {
		panic("model: StepBatchActive id count != batch size")
	}
	if len(active) != C {
		panic("model: StepBatchActive active-mask length != batch size")
	}
	idx := make([]int, 0, C)
	allActive := true
	for b, a := range active {
		if a {
			idx = append(idx, b)
		} else {
			allActive = false
		}
	}
	if allActive {
		return bs.StepBatch(ids) // byte-identical to today's full batch
	}
	if len(idx) == 0 {
		bs.lastStepMACs = 0 // no active lane decoded a token this step
		return make([][]float32, C)
	}
	// Compact: a transient BatchSession over ONLY the active lanes, sharing each active lane's
	// *Session (so KV append + position advance land in the right cache) and the reused
	// scratch/dbuf (so compaction is not a buffer-realloc regression). The idle lanes' sessions
	// are absent from sub.Seqs, so they are neither projected over nor attended — zero work on
	// them, which is exactly the (C−K)/C compaction LastStepMACs witnesses.
	sub := &BatchSession{
		M:       bs.M,
		Seqs:    make([]*Session, len(idx)),
		Quant:   bs.Quant,
		scratch: bs.scratch,
		dbuf:    bs.dbuf,
	}
	subIDs := make([]int, len(idx))
	for i, b := range idx {
		sub.Seqs[i] = bs.Seqs[b]
		subIDs[i] = ids[b]
	}
	subLogits := sub.StepBatch(subIDs)
	// Propagate any grown/shared buffers + the active-step MAC count back to the parent session
	// so a following StepBatch / StepBatchActive reuses them and LastStepMACs reflects this step.
	bs.scratch = sub.scratch
	bs.dbuf = sub.dbuf
	bs.lastStepMACs = sub.lastStepMACs
	out := make([][]float32, C)
	for i, b := range idx {
		out[b] = subLogits[i]
	}
	return out
}

// GenerateBatch greedily decodes up to n tokens for every user in lockstep after their
// prompts, returning each user's generated ids. A user that emits EOS stops contributing new
// tokens (its slot is re-fed its own EOS so the batch geometry stays rectangular — cheap, and
// it keeps the per-user output bit-identical to serial Generate for the non-EOS users). This
// is STATIC batching (fixed B for the run); the per-step primitive (StepBatch) is exactly
// what a continuous-batching scheduler would call after admitting/evicting users between steps.
func (bs *BatchSession) GenerateBatch(prompts [][]int, n int) [][]int {
	B := len(bs.Seqs)
	logits := bs.PrefillEach(prompts)
	out := make([][]int, B)
	done := make([]bool, B)
	// eosSlot is the id re-fed into a finished slot to keep the batch rectangular. The
	// first EOS id (scalar or list head) is fine for this; stop detection uses isEOS.
	eosSlot := bs.M.Cfg.EOSTokenID
	if len(bs.M.Cfg.EOSTokenIDs) > 0 {
		eosSlot = bs.M.Cfg.EOSTokenIDs[0]
	}
	next := make([]int, B)
	for i := 0; i < n; i++ {
		anyLive := false
		for b := 0; b < B; b++ {
			if done[b] {
				next[b] = eosSlot // keep slot rectangular; its logits are ignored once done
				continue
			}
			t := argmaxF32(logits[b])
			out[b] = append(out[b], t)
			next[b] = t
			if bs.M.Cfg.IsEOS(t) {
				done[b] = true
			} else {
				anyLive = true
			}
		}
		if !anyLive {
			break
		}
		logits = bs.StepBatch(next)
	}
	return out
}

// stepBatchF32 is the f32 multi-user decode step. Structure mirrors tokenHidden exactly,
// hoisted to operate on a [B, *] panel: the projections/MLP/head become matMulBatch GEMMs
// (weight read once, reused across all B), attention stays per-user over each user's cache.
func (bs *BatchSession) stepBatchF32(ids []int) [][]float32 {
	m, cfg := bs.M, bs.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := cfg.attnScale()
	B := len(ids)
	if bs.dbuf == nil {
		bs.dbuf = &batchDecodeBuf{}
	}
	db := bs.dbuf

	// Each user decodes at its OWN absolute position (its current cache length), so RoPE is
	// per-user. (Captured before any append; pos is recorded at the end, matching tokenHidden.)
	posB := growInts(db.pos, B)
	db.pos = posB
	cosB := grow2D(db.cos, B, hd/2)
	db.cos = cosB
	sinB := grow2D(db.sin, B, hd/2)
	db.sin = sinB
	inv := cachedInvFreq(cfg, 0)
	for b := 0; b < B; b++ {
		posB[b] = bs.Seqs[b].Cache.Len()
		ropeRowInto(cosB[b], sinB[b], inv, posB[b])
	}

	// embedding lookup: X is [B, H], one working hidden row per user.
	embed := m.embedRows()
	X := make([]float32, B*H)
	for b, id := range ids {
		copy(X[b*H:(b+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[b*H:(b+1)*H], cfg) // Gemma; no-op for Llama
	}
	caches := growCaches(db.caches, B)
	db.caches = caches
	for b := 0; b < B; b++ {
		caches[b] = bs.Seqs[b].Cache
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(s string) string { return layerName(l, s) }

		// pre-attn RMSNorm, per user.
		Xn := make([]float32, B*H)
		wIn := m.tensor(lp("input_layernorm.weight"))
		for b := 0; b < B; b++ {
			copy(Xn[b*H:(b+1)*H], rmsnormCfg(X[b*H:(b+1)*H], wIn, eps, cfg))
		}

		// batched q/k/v projections: one weight stream, B rows.
		Q := matMulBatch(m.tensor(lp("self_attn.q_proj.weight")), Xn, nH*hd, H, B)
		K := matMulBatch(m.tensor(lp("self_attn.k_proj.weight")), Xn, w, H, B)
		V := matMulBatch(m.tensor(lp("self_attn.v_proj.weight")), Xn, w, H, B)
		for b := 0; b < B; b++ {
			m.applyProjBias(l, Q[b*nH*hd:(b+1)*nH*hd], K[b*w:(b+1)*w], V[b*w:(b+1)*w])
			m.applyLayerQKNorm(l, Q[b*nH*hd:(b+1)*nH*hd], K[b*w:(b+1)*w])
		}

		// per-user RoPE + append k/v to each user's own cache (cheap serial pass; the append
		// mutates each user's distinct cache so it can't be raced). The per-user single-row
		// rotate-and-stash is the same one the decode block funnels through (ropeRowQK).
		for b := 0; b < B; b++ {
			qb := Q[b*nH*hd : (b+1)*nH*hd]
			kb := K[b*w : (b+1)*w]
			vb := V[b*w : (b+1)*w]
			c := caches[b]
			c.Kraw[l] = append(c.Kraw[l], kb...) // pre-RoPE, for lossless eviction
			ropeRowQKInto(qb, kb, cosB[b], sinB[b], hd, nH, nKV)
			c.K[l] = append(c.K[l], kb...)
			c.V[l] = append(c.V[l], vb...)
		}
		// causal GQA attention, each user over its own cache (parallel, allocation-light).
		attnOut := make([]float32, B*nH*hd)
		db.scores = attnDecodeBatch(attnOut, Q, caches, l, B, nH, hd, w, grp, cfg.windowForLayer(l), scale, dot, nil, db.scores, m.attnObs)

		// batched output projection + residual.
		O := matMulBatch(m.tensor(lp("self_attn.o_proj.weight")), attnOut, H, nH*hd, B)
		for i := range X {
			X[i] += O[i]
		}

		// batched MLP (SwiGLU) + residual.
		Xn2 := make([]float32, B*H)
		wPost := m.tensor(lp("post_attention_layernorm.weight"))
		for b := 0; b < B; b++ {
			copy(Xn2[b*H:(b+1)*H], rmsnormCfg(X[b*H:(b+1)*H], wPost, eps, cfg))
		}
		I := cfg.IntermediateSize
		G := matMulBatch(m.tensor(lp("mlp.gate_proj.weight")), Xn2, I, H, B)
		U := matMulBatch(m.tensor(lp("mlp.up_proj.weight")), Xn2, I, H, B)
		for i := range G {
			G[i] = act(G[i], cfg) * U[i]
		}
		Down := matMulBatch(m.tensor(lp("mlp.down_proj.weight")), G, H, I, B)
		for i := range X {
			X[i] += Down[i]
		}
	}

	// final norm per user; record each user's new absolute position; batched LM head.
	normW := m.tensor("model.norm.weight")
	Xnorm := make([]float32, B*H)
	for b := 0; b < B; b++ {
		copy(Xnorm[b*H:(b+1)*H], rmsnormCfg(X[b*H:(b+1)*H], normW, eps, cfg))
		bs.Seqs[b].Cache.pos = append(bs.Seqs[b].Cache.pos, posB[b])
	}
	// the 113 MB tied-embedding head streamed ONCE for all B users — the single biggest
	// per-token weight, and so the single biggest batching beneficiary at decode.
	Logits := matMulBatch(m.lmHead(), Xnorm, cfg.VocabSize, H, B)
	out := make([][]float32, B)
	for b := 0; b < B; b++ {
		out[b] = Logits[b*cfg.VocabSize : (b+1)*cfg.VocabSize]
		logitScaleInPlace(out[b], cfg) // Cohere/Gemma2; no-op for Llama
	}
	bs.recordStepMACs(B) // B-proportional projection MAC count for this step (see LastStepMACs)
	return out
}

// qgemmBatch quantizes a [B, width] activation panel into the session's reused scratch and
// runs the register-blocked Q8_0 tile GEMM against the named weight, returning [B, out].
func (bs *BatchSession) qgemmBatch(name string, X []float32, B, width int) []float32 {
	if bs.scratch == nil {
		bs.scratch = &q8Panel{}
	}
	quantizeBatchPanelInto(bs.scratch, X, B, width)
	return qGemm8(bs.M.q8(name), bs.scratch)
}

// qgemmBatchInto is qgemmBatch writing the GEMM result into a caller-provided dst (reused
// across decode steps). Bit-identical to qgemmBatch.
func (bs *BatchSession) qgemmBatchInto(name string, X []float32, B, width int, dst []float32) {
	if bs.scratch == nil {
		bs.scratch = &q8Panel{}
	}
	quantizeBatchPanelInto(bs.scratch, X, B, width)
	qGemm8Into(bs.M.q8(name), bs.scratch, dst)
}

func (bs *BatchSession) qgemmBatchTensorInto(qt *q8Tensor, X []float32, B, width int, dst []float32) {
	if bs.scratch == nil {
		bs.scratch = &q8Panel{}
	}
	quantizeBatchPanelInto(bs.scratch, X, B, width)
	qGemm8Into(qt, bs.scratch, dst)
}

// stepBatchQ is the Q8_0 multi-user decode step: the structural twin of stepBatchF32 with the
// projections + head run as quantized tile GEMMs over the batch panel. Attention is the same
// f32 math over the f32 KV cache (identical to tokenHiddenQ). Not bit-identical to the serial
// qdot8 decode (the tile reduces in a different order) but clears the same Q8 gate the prefill
// path does — argmax-exact + cosine vs f32 (TestBatchedDecodeQMatchesF32).
func (bs *BatchSession) stepBatchQ(ids []int) [][]float32 {
	m, cfg := bs.M, bs.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := cfg.attnScale()
	B := len(ids)

	if bs.dbuf == nil {
		bs.dbuf = &batchDecodeBuf{}
	}
	db := bs.dbuf

	posB := growInts(db.pos, B)
	db.pos = posB
	cosB := grow2D(db.cos, B, hd/2)
	db.cos = cosB
	sinB := grow2D(db.sin, B, hd/2)
	db.sin = sinB
	inv := cachedInvFreq(cfg, 0)
	for b := 0; b < B; b++ {
		posB[b] = bs.Seqs[b].Cache.Len()
		ropeRowInto(cosB[b], sinB[b], inv, posB[b])
	}

	embed := m.embedRows()
	X := grow(db.X, B*H)
	db.X = X
	for b, id := range ids {
		copy(X[b*H:(b+1)*H], embed[id*H:(id+1)*H])
		scaleEmbedInPlace(X[b*H:(b+1)*H], cfg) // Gemma; no-op for Llama
	}
	caches := growCaches(db.caches, B)
	db.caches = caches
	for b := 0; b < B; b++ {
		caches[b] = bs.Seqs[b].Cache
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(s string) string { return layerName(l, s) }
		ql := m.q8Layer(l)

		Xn := grow(db.Xn, B*H)
		db.Xn = Xn
		wIn := m.tensor(lp("input_layernorm.weight"))
		for b := 0; b < B; b++ {
			if cfg.NormGain1p || cfg.LayerNorm {
				copy(Xn[b*H:(b+1)*H], rmsnormCfg(X[b*H:(b+1)*H], wIn, eps, cfg))
			} else {
				rmsnormInto(Xn[b*H:(b+1)*H], X[b*H:(b+1)*H], wIn, eps)
			}
		}
		// q/k/v share one quantized panel of Xn (built once, reused across the three GEMMs).
		if bs.scratch == nil {
			bs.scratch = &q8Panel{}
		}
		quantizeBatchPanelInto(bs.scratch, Xn, B, H)
		Q := grow(db.Q, B*nH*hd)
		db.Q = Q
		K := grow(db.K, B*w)
		db.K = K
		V := grow(db.V, B*w)
		db.V = V
		qGemm8IntoMany(bs.scratch,
			qgemm8Target{qt: ql.qProj, Y: Q},
			qgemm8Target{qt: ql.kProj, Y: K},
			qgemm8Target{qt: ql.vProj, Y: V},
		)
		for b := 0; b < B; b++ {
			m.applyProjBias(l, Q[b*nH*hd:(b+1)*nH*hd], K[b*w:(b+1)*w], V[b*w:(b+1)*w])
			m.applyLayerQKNorm(l, Q[b*nH*hd:(b+1)*nH*hd], K[b*w:(b+1)*w])
		}

		for b := 0; b < B; b++ {
			qb := Q[b*nH*hd : (b+1)*nH*hd]
			kb := K[b*w : (b+1)*w]
			vb := V[b*w : (b+1)*w]
			c := caches[b]
			c.Kraw[l] = append(c.Kraw[l], kb...)
			ropeRowQKInto(qb, kb, cosB[b], sinB[b], hd, nH, nKV)
			c.K[l] = append(c.K[l], kb...)
			c.V[l] = append(c.V[l], vb...)
		}
		// attention is the same f32 math over the f32 KV cache as the f32 lane.
		attnOut := grow(db.attn, B*nH*hd)
		db.attn = attnOut
		scoreDot3 := fdot3scalar
		if attnFdot3SIMD && B >= attnFdot3SIMDMinBatch {
			scoreDot3 = fdot3SIMD
		}
		db.scores = attnDecodeBatch(attnOut, Q, caches, l, B, nH, hd, w, grp, cfg.windowForLayer(l), scale, fdot, scoreDot3, db.scores, m.attnObs)

		O := grow(db.O, B*H)
		db.O = O
		bs.qgemmBatchTensorInto(ql.oProj, attnOut, B, nH*hd, O)
		for i := range X {
			X[i] += O[i]
		}

		Xn2 := grow(db.Xn2, B*H)
		db.Xn2 = Xn2
		wPost := m.tensor(lp("post_attention_layernorm.weight"))
		for b := 0; b < B; b++ {
			if cfg.NormGain1p || cfg.LayerNorm {
				copy(Xn2[b*H:(b+1)*H], rmsnormCfg(X[b*H:(b+1)*H], wPost, eps, cfg))
			} else {
				rmsnormInto(Xn2[b*H:(b+1)*H], X[b*H:(b+1)*H], wPost, eps)
			}
		}
		I := cfg.IntermediateSize
		quantizeBatchPanelInto(bs.scratch, Xn2, B, H)
		G := grow(db.G, B*I)
		db.G = G
		U := grow(db.U, B*I)
		db.U = U
		qGemm8IntoMany(bs.scratch,
			qgemm8Target{qt: ql.gateProj, Y: G},
			qgemm8Target{qt: ql.upProj, Y: U},
		)
		for i := range G {
			G[i] = act(G[i], cfg) * U[i]
		}
		Down := grow(db.Down, B*H)
		db.Down = Down
		bs.qgemmBatchTensorInto(ql.downProj, G, B, I, Down)
		for i := range X {
			X[i] += Down[i]
		}
	}

	normW := m.tensor("model.norm.weight")
	Xnorm := grow(db.Xnorm, B*H)
	db.Xnorm = Xnorm
	for b := 0; b < B; b++ {
		if cfg.NormGain1p || cfg.LayerNorm {
			copy(Xnorm[b*H:(b+1)*H], rmsnormCfg(X[b*H:(b+1)*H], normW, eps, cfg))
		} else {
			rmsnormInto(Xnorm[b*H:(b+1)*H], X[b*H:(b+1)*H], normW, eps)
		}
		bs.Seqs[b].Cache.pos = append(bs.Seqs[b].Cache.pos, posB[b])
	}
	Logits := grow(db.Logits, B*cfg.VocabSize)
	db.Logits = Logits
	bs.qgemmBatchTensorInto(m.q8Head(), Xnorm, B, H, Logits)
	out := growLogitRows(db.out, B)
	db.out = out
	for b := 0; b < B; b++ {
		out[b] = Logits[b*cfg.VocabSize : (b+1)*cfg.VocabSize]
		logitScaleInPlace(out[b], cfg) // Cohere/Gemma2; no-op for Llama
	}
	bs.recordStepMACs(B) // B-proportional projection MAC count for this step (see LastStepMACs)
	return out
}
