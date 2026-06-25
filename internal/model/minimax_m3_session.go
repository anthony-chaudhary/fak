package model

// minimax_m3_session.go — MiniMax-M3 "MiniMax Sparse Attention" (MSA) incremental
// Session/KV-cache path (cached decode, prefix reuse, eviction). It is the MSA analogue
// of glm_dsa_session.go: the cacheless full-prefill Forward (minimax_m3.go) ran the MSA
// math over the whole sequence; this re-derives the SAME math one position at a time
// against the kernel-owned KVCache, so a split Prefill/Step, PrefillNoLogits,
// SessionFromPrefix, and greedy Generate all agree with the cacheless Forward
// (TestMiniMaxM3MSASessionCacheMatchesCacheless / the optional HF oracle).
//
// MSA keeps the REAL uncompressed GQA K/V (NOT GLM's MLA latent), so the main K/V/Kraw
// rows live in the ordinary KVCache.K/Kraw/V and reuse the proven softmax-KV eviction
// (re-RoPE of a repositioned survivor). The only extra per-layer state MSA needs is the
// lightning-indexer key: like GLM's IndexK/IndexKraw, the post-(k_norm,partial-RoPE) key
// is cached per cached position so each step re-scores its freshly-projected indexer
// query against the cached keys; the post-norm/pre-RoPE raw is kept so an Evict can
// re-RoPE a repositioned survivor in a single rotation (minimaxKVCache below). A
// full_attention layer caches no index key (it runs dense GQA), so its IndexK slot stays
// empty and eviction skips it.

// minimaxKVCache holds the per-layer lightning-indexer key cache for the MiniMax-M3 MSA
// Session path. It lives beside the ordinary K/Kraw/V rows in KVCache (the main GQA K/V
// uses those directly); only the sparse minimax_m3_sparse layers populate an entry.
type minimaxKVCache struct {
	IndexK    [][]float32 // [layer] -> flat IndexHeadDim per cached position (post k_norm + partial RoPE)
	IndexKraw [][]float32 // [layer] -> the SAME entries post-k_norm, PRE-RoPE, so Evict can re-RoPE a survivor
}

// newMinimaxKVCache allocates the MSA index-key cache for a MiniMax-M3 sparse-attention
// model, or nil for any other family (so the ordinary cache stays untouched).
func newMinimaxKVCache(cfg Config) *minimaxKVCache {
	if !cfg.isMiniMaxSparseAttn() {
		return nil
	}
	return &minimaxKVCache{
		IndexK:    make([][]float32, cfg.NumLayers),
		IndexKraw: make([][]float32, cfg.NumLayers),
	}
}

func (c *minimaxKVCache) cloneWithReserve(cfg Config, extraPositions int) *minimaxKVCache {
	if c == nil {
		return nil
	}
	if extraPositions < 0 {
		extraPositions = 0
	}
	idxExtra := extraPositions * cfg.IndexHeadDim
	out := &minimaxKVCache{
		IndexK:    make([][]float32, len(c.IndexK)),
		IndexKraw: make([][]float32, len(c.IndexKraw)),
	}
	for l := range c.IndexK {
		out.IndexK[l] = cloneFloat32WithReserve(c.IndexK[l], idxExtra)
		out.IndexKraw[l] = cloneFloat32WithReserve(c.IndexKraw[l], idxExtra)
	}
	return out
}

func (c *minimaxKVCache) reserve(cfg Config, extraPositions int) {
	if c == nil || extraPositions <= 0 {
		return
	}
	idxExtra := extraPositions * cfg.IndexHeadDim
	for l := range c.IndexK {
		c.IndexK[l] = reserveFloat32(c.IndexK[l], idxExtra)
		c.IndexKraw[l] = reserveFloat32(c.IndexKraw[l], idxExtra)
	}
}

func (c *minimaxKVCache) evict(cfg Config, from, end int) {
	if c == nil {
		return
	}
	idxStride := cfg.IndexHeadDim
	for l := range c.IndexK {
		c.IndexK[l] = evictFloat32Rows(c.IndexK[l], from, end, idxStride)
		c.IndexKraw[l] = evictFloat32Rows(c.IndexKraw[l], from, end, idxStride)
	}
}

// rerotateSurvivor re-derives the cached index key for a survivor that compaction moved
// to absolute position pos: copy the post-k_norm, pre-RoPE raw and apply partial RoPE at
// the new position in a single rotation (bit-exact to a prefill that never saw the span),
// exactly as the main K is re-RoPEd. A dense (full_attention) layer has no cached index
// key, so its empty slot is skipped.
func (c *minimaxKVCache) rerotateSurvivor(cfg Config, pos int) {
	if c == nil {
		return
	}
	rot := cfg.rotaryDim()
	idxStride := cfg.IndexHeadDim
	for l := range c.IndexK {
		if len(c.IndexKraw[l]) == 0 {
			continue
		}
		cos, sin := ropeRowForLayer(cfg, l, pos)
		off := pos * idxStride
		dst := c.IndexK[l][off : off+idxStride]
		copy(dst, c.IndexKraw[l][off:off+idxStride]) // post-k_norm, pre-RoPE
		applyRopeRow(dst[:rot], cos, sin)            // single rotation at the new position
	}
}

// requireMiniMaxSession lazily allocates the MSA index-key cache and fails closed on the
// session modes the MSA path does not yet wire. The incremental MSA path is the CPU f32
// resident path (it mirrors the cacheless f32 Forward bit-for-bit); the quantized, HAL,
// Metal, and dynamic-precision paths are a separate gate, so they panic loudly rather than
// silently running dense GQA on the sparse layers.
func (s *Session) requireMiniMaxSession() {
	if s.Quant || s.Q4 || s.Q4K || s.Backend != nil || s.Metal || s.MetalQ4K || s.PrecisionPolicy != nil {
		panic("model: MiniMax-M3 MSA Session: only the CPU f32 resident path is wired " +
			"(Quant/Q4/Q4K/Backend/Metal/PrecisionPolicy unsupported)")
	}
	if s.Cache.msa == nil {
		s.Cache.msa = newMinimaxKVCache(s.M.Cfg)
	}
}

// tokenHiddenMiniMax runs ONE incremental decode step at absolute position pos through
// every MiniMax-M3 decoder layer against this session's cache, appending this position's
// main K/V (and, on sparse layers, the indexer key) and returning the post-final-norm
// hidden. It is the single-position analogue of the cacheless layerMiniMax: a
// full_attention layer runs dense causal GQA, a minimax_m3_sparse layer runs the
// lightning-indexer block-sparse path, then the SwiGLU-OAI MoE FFN — each composed under
// the same pre/post-norm topology (two sequential composeSublayer calls, exactly as
// layerMiniMax composes its two whole-sequence sublayers).
func (s *Session) tokenHiddenMiniMax(id, pos int) []float32 {
	m, cfg := s.M, s.M.Cfg
	H := cfg.HiddenSize
	eps := float32(cfg.RMSNormEps)
	embed := m.embedRows()
	x := append([]float32(nil), embed[id*H:(id+1)*H]...)
	scaleEmbedInPlace(x, cfg) // Gemma sqrt(hidden); no-op for MiniMax
	mat := residentKernel{m}
	for l := 0; l < cfg.NumLayers; l++ {
		layer := l
		cos, sin := ropeRowForLayer(cfg, l, pos)
		attnBody := func(xn []float32) []float32 {
			if cfg.isMSALayer(layer) {
				out, ok := m.minimaxMSAStep(s.Cache, layer, pos, xn, cos, sin)
				if !ok {
					panic("model: minimax_m3 MSA session step failed")
				}
				return out
			}
			return m.minimaxDenseAttnStep(s.Cache, layer, pos, xn, cos, sin)
		}
		mlpBody := func(xn []float32) []float32 {
			return m.ffnForLayer(layer).apply(m, layer, mat.prep(xn), mat)
		}
		composeSublayer(cfg.BlockTopology, x, m.attentionNorms(layer), eps, cfg, attnBody)
		composeSublayer(cfg.BlockTopology, x, m.mlpNorms(layer), eps, cfg, mlpBody)
	}
	s.Cache.pos = append(s.Cache.pos, pos)
	return m.finalNorm(x)
}

// minimaxDenseAttnStep runs the dense causal GQA sublayer for a full_attention MiniMax-M3
// layer at one position over the cached K/V, mirroring the cacheless attnSeq dense path
// (per-head qk-norm + partial RoPE, scaled-dot GQA, o_proj + bias) but appending this
// position's K/Kraw/V to the kernel-owned cache and attending the full causal range.
func (m *Model) minimaxDenseAttnStep(cache *KVCache, layer, pos int, xn, cos, sin []float32) []float32 {
	cfg := m.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	scale := cfg.attnScale()
	w := nKV * hd
	p := func(s string) string { return layerName(layer, s) }

	q := m.residentMatRows(p("self_attn.q_proj.weight"), xn, nH*hd, H)
	k := m.residentMatRows(p("self_attn.k_proj.weight"), xn, w, H)
	v := m.residentMatRows(p("self_attn.v_proj.weight"), xn, w, H)
	m.applyProjBias(layer, q, k, v)
	m.applyLayerQKNorm(layer, q, k)
	kraw := append([]float32(nil), k...) // post-qk-norm, pre-RoPE (Evict reposition source)
	ropeRowQKInto(q, k, cos, sin, hd, nH, nKV)
	cache.K[layer] = append(cache.K[layer], k...)
	cache.Kraw[layer] = append(cache.Kraw[layer], kraw...)
	cache.V[layer] = append(cache.V[layer], v...)

	nPos := len(cache.K[layer]) / w
	attnOut := make([]float32, nH*hd)
	for h := 0; h < nH; h++ {
		kvh := h / grp
		qh := q[h*hd : (h+1)*hd]
		scores := make([]float32, nPos)
		for j := 0; j < nPos; j++ {
			kh := cache.K[layer][j*w+kvh*hd : j*w+(kvh+1)*hd]
			scores[j] = dot(qh, kh) * scale
		}
		m.softmaxAttentionScores(layer, h, scores)
		o := attnOut[h*hd : (h+1)*hd]
		for j := 0; j < nPos; j++ {
			vh := cache.V[layer][j*w+kvh*hd : j*w+(kvh+1)*hd]
			wj := scores[j]
			for d := 0; d < hd; d++ {
				o[d] += wj * vh[d]
			}
		}
	}
	out := m.residentMatRows(p("self_attn.o_proj.weight"), attnOut, H, nH*hd)
	m.addBiasIfPresent(out, p("self_attn.o_proj.bias"))
	return out
}

// minimaxMSAStep runs the MiniMax-M3 MSA (block-sparse GQA) sublayer for a sparse layer at
// one position over the cached K/V: it projects the main q/k/v (per-head qk-norm + partial
// RoPE, appended to the cache exactly like the dense path), appends this position's
// lightning-indexer key to the index cache, re-scores a freshly-projected indexer query
// against ALL cached index keys to choose the admitted causal key positions, and runs GQA
// softmax over only those admitted keys. The selection math is byte-for-byte the cacheless
// minimaxIndexerSelectBlocks/minimaxMSAAttnSeq, so a cached step agrees with the full
// Forward.
func (m *Model) minimaxMSAStep(cache *KVCache, layer, pos int, xn, cos, sin []float32) ([]float32, bool) {
	cfg := m.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	if cache == nil || cache.msa == nil || len(xn) != H {
		return nil, false
	}
	w := nKV * hd
	p := func(s string) string { return layerName(layer, s) }

	q := m.residentMatRows(p("self_attn.q_proj.weight"), xn, nH*hd, H)
	k := m.residentMatRows(p("self_attn.k_proj.weight"), xn, w, H)
	v := m.residentMatRows(p("self_attn.v_proj.weight"), xn, w, H)
	m.applyProjBias(layer, q, k, v)
	m.applyLayerQKNorm(layer, q, k)
	kraw := append([]float32(nil), k...)
	ropeRowQKInto(q, k, cos, sin, hd, nH, nKV)
	cache.K[layer] = append(cache.K[layer], k...)
	cache.Kraw[layer] = append(cache.Kraw[layer], kraw...)
	cache.V[layer] = append(cache.V[layer], v...)

	if !m.minimaxIndexerAppendKey(cache.msa, layer, pos, xn, cos, sin) {
		return nil, false
	}
	admitted, ok := m.minimaxSelectCachedKeys(cache.msa, layer, pos, xn, cos, sin)
	if !ok {
		return nil, false
	}

	scale := cfg.attnScale()
	attnOut := make([]float32, nH*hd)
	for h := 0; h < nH; h++ {
		kvh := h / grp
		qh := q[h*hd : (h+1)*hd]
		scores := make([]float32, len(admitted))
		for i, kp := range admitted {
			kh := cache.K[layer][kp*w+kvh*hd : kp*w+(kvh+1)*hd]
			scores[i] = dot(qh, kh) * scale
		}
		softmaxInPlace(scores)
		o := attnOut[h*hd : (h+1)*hd]
		for i, kp := range admitted {
			vh := cache.V[layer][kp*w+kvh*hd : kp*w+(kvh+1)*hd]
			wt := scores[i]
			for d := 0; d < hd; d++ {
				o[d] += wt * vh[d]
			}
		}
	}
	out := m.residentMatRows(p("self_attn.o_proj.weight"), attnOut, H, nH*hd)
	m.addBiasIfPresent(out, p("self_attn.o_proj.bias"))
	return out, true
}

// minimaxIndexerAppendKey projects this position's single shared lightning-indexer key
// (self_attn.indexer.k_proj, one head of IndexHeadDim), RMS-norms it (k_norm), applies
// partial RoPE, and appends both the post-RoPE key and its post-norm/pre-RoPE raw to the
// index cache. It mirrors the k-side of the cacheless minimaxIndexerProject for one row.
func (m *Model) minimaxIndexerAppendKey(msa *minimaxKVCache, layer, pos int, xn, cos, sin []float32) bool {
	cfg := m.Cfg
	H := cfg.HiddenSize
	idxDim := cfg.IndexHeadDim
	rot := cfg.rotaryDim()
	eps := float32(cfg.RMSNormEps)
	ip := func(s string) string { return layerName(layer, "self_attn.indexer."+s) }
	kf := m.residentMatRows(ip("k_proj.weight"), xn, idxDim, H)
	applyRMSNormInPlaceCfg(kf, m.tensor(ip("k_norm.weight")), eps, cfg)
	kraw := append([]float32(nil), kf...) // post-k_norm, pre-RoPE
	applyRopeRow(kf[:rot], cos, sin)
	msa.IndexK[layer] = append(msa.IndexK[layer], kf...)
	msa.IndexKraw[layer] = append(msa.IndexKraw[layer], kraw...)
	return len(msa.IndexK[layer]) == (pos+1)*idxDim && len(msa.IndexKraw[layer]) == (pos+1)*idxDim
}

// minimaxSelectCachedKeys projects this position's indexer query (IndexNHeads heads of
// IndexHeadDim, q_norm + partial RoPE), scores it against EVERY cached index key, max-pools
// per index head into blocks, merges those per-head block scores with max (HF amax over
// index heads), selects the top-IndexTopKBlocks blocks with the always-on local window, and
// broadcasts the block choice back onto the admitted causal key positions. It is the
// single-query form of the cacheless minimaxIndexerSelectBlocks + minimaxIndexerSelect,
// scoring against the CACHED keys rather than recomputing them, so the admitted set matches.
func (m *Model) minimaxSelectCachedKeys(msa *minimaxKVCache, layer, pos int, xn, cos, sin []float32) ([]int, bool) {
	cfg := m.Cfg
	H := cfg.HiddenSize
	nIdx := cfg.IndexNHeads
	idxDim := cfg.IndexHeadDim
	rot := cfg.rotaryDim()
	eps := float32(cfg.RMSNormEps)
	blockSize := cfg.IndexBlockSize
	ip := func(s string) string { return layerName(layer, "self_attn.indexer."+s) }
	qNorm := m.tensor(ip("q_norm.weight"))

	qf := m.residentMatRows(ip("q_proj.weight"), xn, nIdx*idxDim, H)
	idxQ := make([][]float32, nIdx)
	for h := 0; h < nIdx; h++ {
		head := qf[h*idxDim : (h+1)*idxDim]
		applyRMSNormInPlaceCfg(head, qNorm, eps, cfg)
		applyRopeRow(head[:rot], cos, sin)
		idxQ[h] = head
	}

	nKeys := pos + 1
	keyPos := make([]int, nKeys)
	for i := range keyPos {
		keyPos[i] = i
	}
	// Block score = max over (all index heads, all keys in the block) — the per-head block
	// max-pool (msaBlockScores) then the amax over index heads (the merge below).
	merged := map[int]float64{}
	for g := 0; g < nIdx; g++ {
		scores := make([]float64, nKeys)
		for kk := 0; kk < nKeys; kk++ {
			key := msa.IndexK[layer][kk*idxDim : (kk+1)*idxDim]
			scores[kk] = float64(dot(idxQ[g], key))
		}
		bs, ok := msaBlockScores(scores, keyPos, pos, blockSize)
		if !ok {
			return nil, false
		}
		for b, s := range bs {
			if cur, seen := merged[b]; !seen || s > cur {
				merged[b] = s
			}
		}
	}
	blocks := minimaxSelectBlocks(merged, pos, blockSize, cfg.IndexTopKBlocks, cfg.IndexLocalBlocks)
	admitted, ok := msaSelectedKeyPositions(keyPos, pos, blockSize, blocks)
	if !ok {
		return nil, false
	}
	return admitted, true
}
