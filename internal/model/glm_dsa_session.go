package model

import (
	"fmt"
	"math"
)

type glmDsaKVCache struct {
	K         [][]float32
	Kraw      [][]float32
	V         [][]float32
	IndexK    [][]float64
	IndexKraw [][]float64
}

func newGLMDsaKVCache(cfg Config) *glmDsaKVCache {
	if !cfg.isGLMMoeDsa() {
		return nil
	}
	return &glmDsaKVCache{
		K:         make([][]float32, cfg.NumLayers),
		Kraw:      make([][]float32, cfg.NumLayers),
		V:         make([][]float32, cfg.NumLayers),
		IndexK:    make([][]float64, cfg.NumLayers),
		IndexKraw: make([][]float64, cfg.NumLayers),
	}
}

func (c *glmDsaKVCache) cloneWithReserve(cfg Config, extraPositions int) *glmDsaKVCache {
	if c == nil {
		return nil
	}
	if extraPositions < 0 {
		extraPositions = 0
	}
	kExtra := extraPositions * glmDsaAttentionKStride(cfg)
	vExtra := extraPositions * glmDsaAttentionVStride(cfg)
	idxExtra := extraPositions * cfg.IndexHeadDim
	out := &glmDsaKVCache{
		K:         make([][]float32, len(c.K)),
		Kraw:      make([][]float32, len(c.Kraw)),
		V:         make([][]float32, len(c.V)),
		IndexK:    make([][]float64, len(c.IndexK)),
		IndexKraw: make([][]float64, len(c.IndexKraw)),
	}
	for l := range c.K {
		out.K[l] = cloneFloat32WithReserve(c.K[l], kExtra)
		out.Kraw[l] = cloneFloat32WithReserve(c.Kraw[l], kExtra)
		out.V[l] = cloneFloat32WithReserve(c.V[l], vExtra)
		out.IndexK[l] = cloneFloat64WithReserve(c.IndexK[l], idxExtra)
		out.IndexKraw[l] = cloneFloat64WithReserve(c.IndexKraw[l], idxExtra)
	}
	return out
}

func (c *glmDsaKVCache) reserve(cfg Config, extraPositions int) {
	if c == nil || extraPositions <= 0 {
		return
	}
	kExtra := extraPositions * glmDsaAttentionKStride(cfg)
	vExtra := extraPositions * glmDsaAttentionVStride(cfg)
	idxExtra := extraPositions * cfg.IndexHeadDim
	for l := range c.K {
		c.K[l] = reserveFloat32(c.K[l], kExtra)
		c.Kraw[l] = reserveFloat32(c.Kraw[l], kExtra)
		c.V[l] = reserveFloat32(c.V[l], vExtra)
		c.IndexK[l] = reserveFloat64(c.IndexK[l], idxExtra)
		c.IndexKraw[l] = reserveFloat64(c.IndexKraw[l], idxExtra)
	}
}

func (c *glmDsaKVCache) evict(cfg Config, from, end int) {
	if c == nil {
		return
	}
	kStride := glmDsaAttentionKStride(cfg)
	vStride := glmDsaAttentionVStride(cfg)
	idxStride := cfg.IndexHeadDim
	for l := range c.K {
		c.K[l] = evictFloat32Rows(c.K[l], from, end, kStride)
		c.Kraw[l] = evictFloat32Rows(c.Kraw[l], from, end, kStride)
		c.V[l] = evictFloat32Rows(c.V[l], from, end, vStride)
		c.IndexK[l] = evictFloat64Rows(c.IndexK[l], from, end, idxStride)
		c.IndexKraw[l] = evictFloat64Rows(c.IndexKraw[l], from, end, idxStride)
	}
}

func evictFloat32Rows(s []float32, from, end, stride int) []float32 {
	if len(s) == 0 {
		return s
	}
	return append(s[:from*stride], s[end*stride:]...)
}

func evictFloat64Rows(s []float64, from, end, stride int) []float64 {
	if len(s) == 0 {
		return s
	}
	return append(s[:from*stride], s[end*stride:]...)
}

func (c *glmDsaKVCache) rerotateSurvivor(cfg Config, pos int) {
	if c == nil {
		return
	}
	qkNope, qkRope := cfg.QKNopeHeadDim, cfg.QKRopeHeadDim
	qkHead := qkNope + qkRope
	kStride := glmDsaAttentionKStride(cfg)
	idxStride := cfg.IndexHeadDim
	for l := range c.K {
		cos, sin := ropeRowForLayer(cfg, l, pos)
		for h := 0; h < cfg.NumHeads; h++ {
			off := pos*kStride + h*qkHead
			raw := c.Kraw[l][off : off+qkHead]
			dst := c.K[l][off : off+qkHead]
			copy(dst[:qkNope], raw[:qkNope])
			copy(dst[qkNope:], glmDsaApplyInterleavedRoPE(raw[qkNope:], cos, sin))
		}
		if len(c.IndexKraw[l]) == 0 {
			continue
		}
		off := pos * idxStride
		raw := float64To32(c.IndexKraw[l][off : off+idxStride])
		glmDsaApplyIndexerRoPE(raw[:qkRope], cos, sin)
		for i, v := range raw {
			c.IndexK[l][off+i] = float64(v)
		}
	}
}

func glmDsaAttentionKStride(cfg Config) int {
	return cfg.NumHeads * (cfg.QKNopeHeadDim + cfg.QKRopeHeadDim)
}

func glmDsaAttentionVStride(cfg Config) int {
	return cfg.NumHeads * cfg.VHeadDim
}

func (s *Session) tokenHiddenGLMDsa(id, pos int) []float32 {
	xf, err := s.decodeBandGLMDsa(id, nil, 0, s.M.Cfg.NumLayers, pos, true, true)
	if err != nil {
		panic(err)
	}
	// Token boundary: return this position's transient device op-output buffers to the backend pool
	// — the same recycle the HAL decode path (recycleHALToken) does every token. GLM-DSA's
	// Prefill/Step loops (kv.go) drive tokenHiddenGLMDsa DIRECTLY, not through the HAL, so without
	// this a multi-token Prefill never recycles and the device op buffers grow per position until
	// cudaMalloc fails. xf is host-resident (finalNorm), so recycling the device pool is safe here.
	// No-op on the host/cpu-ref backend (it advertises no Recycle), so host forwards stay byte-exact.
	if r, ok := s.Backend.(interface{ Recycle() }); ok {
		r.Recycle()
	}
	return xf
}

// decodeBandGLMDsa runs ONE incremental decode step through transformer-layer band
// [lo,hi) at absolute position pos, mutating only this session's DSA cache slots in
// [lo,hi). isFirst embeds id into the initial hidden; otherwise x is the hidden state
// handed over from the previous stage and id is ignored. isLast appends the pos ledger
// entry and applies finalNorm, returning the head's input; a non-last stage returns the
// raw band-output hidden for the next stage and does NOT touch Cache.pos. pos is the
// caller's absolute position — never read from Cache.Len(), which only the tail stage
// advances. sharedTopK is band-local: reset here every call, and because every band
// begins on a full-indexer layer (PartitionPlan rule, re-guarded below) the first layer
// recomputes its own top-k, so a fresh slice is correct and no top-k crosses a boundary.
// tokenHiddenGLMDsa delegates to this over [0,NumLayers) so there is ONE instruction
// stream (the FMA-fusion no-op gate) — the monolithic path is byte-for-byte unchanged.
func (s *Session) decodeBandGLMDsa(id int, x []float32, lo, hi, pos int, isFirst, isLast bool) ([]float32, error) {
	m, cfg := s.M, s.M.Cfg
	if lo < 0 || hi <= lo || hi > cfg.NumLayers {
		return nil, fmt.Errorf("model: decodeBandGLMDsa range [%d,%d) invalid for %d layers", lo, hi, cfg.NumLayers)
	}
	if glmDsaIndexerIsShared(cfg, lo) {
		return nil, fmt.Errorf("model: decodeBandGLMDsa cannot start at GLM shared-indexer layer %d (band must begin on a full-indexer layer)", lo)
	}
	if isFirst {
		H := cfg.HiddenSize
		embed := m.embedRows()
		x = append([]float32(nil), embed[id*H:(id+1)*H]...)
		scaleEmbedInPlace(x, cfg)
	}
	s.glmDsaSharedTopK = nil
	eps := float32(cfg.RMSNormEps)
	// #86 (partial): with a compute.Backend attached, route the dense GEMMs through it (the GPU pure
	// kernels) via backendKernel; otherwise the host residentKernel. Both the MoE/FFN GEMMs (mlpBody)
	// AND the DSA attention's dense projections — q_a/q_b, kv_a/kv_b, indexer wq_b/wk/weights_proj,
	// and o_proj (mat threaded into glmDsaAttentionStep) — run on the kernel; only the small DSA
	// index-score dots + top-k selection + sparse softmax/ΣwV glue and the DSA KV cache stay
	// host-resident (the genuinely sparse inner loop — the labeled #86/#413 residual). On the host
	// path residentKernel.mul ≡ residentMatRows, so CPU sessions stay byte-for-byte unchanged.
	// CPUOffloadExperts swaps in a splitKernel that keeps the expert GEMMs on host RAM while dense
	// stays on the device (the --n-cpu-moe hybrid); see glmDsaMatKernel (moe_offload.go).
	mat := s.glmDsaMatKernel()
	for l := lo; l < hi; l++ {
		layer := l
		attnBody := func(xn []float32) []float32 {
			out, topK, ok := m.glmDsaAttentionStep(s.Cache.glm, layer, pos, xn, s.glmDsaSharedTopK, mat)
			if !ok {
				panic("model: glm_moe_dsa attention step failed")
			}
			if !glmDsaIndexerIsShared(cfg, layer) {
				s.glmDsaSharedTopK = append(s.glmDsaSharedTopK[:0], topK...)
			}
			return out
		}
		mlpBody := func(xn []float32) []float32 {
			return m.ffnForLayer(layer).apply(m, layer, mat.prep(xn), mat)
		}
		attnNorm := m.attentionNorms(layer)
		mlpNorm := m.mlpNorms(layer)
		if cfg.BlockTopology == ParallelResidual {
			mlpNorm = m.parallelMLPNorms(layer, attnNorm)
		}
		composeBlock(cfg.BlockTopology, x, attnNorm, mlpNorm, eps, cfg, attnBody, mlpBody)
	}
	if isLast {
		s.Cache.pos = append(s.Cache.pos, pos)
		return m.finalNorm(x), nil
	}
	return x, nil
}

// glmDsaAttentionStep runs the GLM-MoE-DSA attention sublayer for one position. mat selects
// where its dense projection GEMMs execute: residentKernel (host, byte-for-byte residentMatRows)
// or backendKernel (the compute.Backend — q8 -> k_q8_gemm, the GPU pure kernel). The learned-index
// score dots, top-k selection, and sparse softmax/ΣwV over the selected keys stay host-resident in
// either case (the genuinely sparse glue), so only the projections move to the device.
func (m *Model) glmDsaAttentionStep(cache *glmDsaKVCache, layer, pos int, xn []float32, sharedTopK []int, mat matKernel) ([]float32, []int, bool) {
	cfg := m.Cfg
	if cache == nil || !cfg.isGLMMoeDsa() || len(xn) != cfg.HiddenSize {
		return nil, nil, false
	}
	var topK []int
	if glmDsaIndexerIsShared(cfg, layer) {
		if len(sharedTopK) == 0 {
			return nil, nil, false
		}
		topK = append([]int(nil), sharedTopK...)
	} else {
		if !glmDsaIndexerIsFull(cfg, layer) {
			return nil, nil, false
		}
		var ok bool
		topK, ok = m.glmDsaIndexStep(cache, layer, pos, xn, mat)
		if !ok {
			return nil, nil, false
		}
	}
	query, ok := m.glmDsaAppendAttentionKV(cache, layer, pos, xn, mat)
	if !ok {
		return nil, nil, false
	}
	out, ok := m.glmDsaAttendCached(cache, layer, pos, query, topK, mat)
	if !ok {
		return nil, nil, false
	}
	return out, topK, true
}

func (m *Model) glmDsaIndexStep(cache *glmDsaKVCache, layer, pos int, xn []float32, mat matKernel) ([]int, bool) {
	cfg := m.Cfg
	H, qLora := cfg.HiddenSize, cfg.QLoraRank
	indexHeads, indexDim := cfg.IndexNHeads, cfg.IndexHeadDim
	qkRope := cfg.QKRopeHeadDim
	if H == 0 || qLora == 0 || indexHeads == 0 || indexDim == 0 || qkRope <= 0 || qkRope > indexDim || qkRope%2 != 0 {
		return nil, false
	}
	// proj runs a named projection GEMM through the active kernel (host residentMatRows or the
	// compute.Backend); residentKernel makes it byte-for-byte residentMatRows.
	proj := func(name string, x []float32, out, in int) []float32 { return mat.mul(name, mat.prep(x), out, in) }

	ap := layerPrefix(layer) + "self_attn."
	qResid := proj(ap+"q_a_proj.weight", xn, qLora, H)
	addOptionalBias(qResid, m.tensorOptional(ap+"q_a_proj.bias"))
	qResid = rmsnorm(qResid, m.tensor(ap+"q_a_layernorm.weight"), glmDsaInnerNormEps)
	qFull := proj(ap+"indexer.wq_b.weight", qResid, indexHeads*indexDim, qLora)

	k := layernorm(proj(ap+"indexer.wk.weight", xn, indexDim, H),
		m.tensor(ap+"indexer.k_norm.weight"), m.tensor(ap+"indexer.k_norm.bias"), glmDsaInnerNormEps)
	weights := proj(ap+"indexer.weights_proj.weight", xn, indexHeads, H)
	weightScale := float32(1.0 / math.Sqrt(float64(indexHeads)))
	for i := range weights {
		weights[i] *= weightScale
	}

	cos, sin := ropeRowForLayer(cfg, layer, pos)
	indexQ := make([][]float64, indexHeads)
	for h := 0; h < indexHeads; h++ {
		head := append([]float32(nil), qFull[h*indexDim:(h+1)*indexDim]...)
		glmDsaApplyIndexerRoPE(head[:qkRope], cos, sin)
		indexQ[h] = float32To64(head)
	}
	k = append([]float32(nil), k...)
	kRaw := append([]float32(nil), k...)
	glmDsaApplyIndexerRoPE(k[:qkRope], cos, sin)
	k64 := float32To64(k)
	cache.IndexKraw[layer] = append(cache.IndexKraw[layer], float32To64(kRaw)...)
	cache.IndexK[layer] = append(cache.IndexK[layer], k64...)
	if len(cache.IndexK[layer]) != (pos+1)*indexDim || len(cache.IndexKraw[layer]) != (pos+1)*indexDim {
		return nil, false
	}

	scale := 1.0 / math.Sqrt(float64(indexDim))

	// Device path (OPTIONAL): when the active kernel's backend advertises compute.DSAIndexBackend,
	// the indexer score + top-k SELECTION runs on the device (k_dsa_index_score + k_dsa_index_topk).
	// The cache stores f64 that are losslessly-widened f32 (the keys/queries originated f32, see the
	// float32To64 above), so narrowing back to f32 for the device is exact; the device accumulates
	// the score dot in f64, so the selected positions are bit-identical to the host loop below — the
	// selection-stability boundary is satisfied, not bypassed. Any non-conforming result falls back.
	if ik, ok := mat.(dsaIndexKernel); ok {
		nKeys := pos + 1
		idxQ := make([]float32, indexHeads*indexDim)
		for h := 0; h < indexHeads; h++ {
			copy(idxQ[h*indexDim:(h+1)*indexDim], float64To32(indexQ[h]))
		}
		idxK := float64To32(cache.IndexK[layer][:nKeys*indexDim])
		if sel, ok := ik.indexSelect(idxQ, idxK, weights, nKeys, indexHeads, indexDim, pos, cfg.IndexTopK, float32(scale)); ok {
			if glmDsaValidSelection(sel, pos) {
				return sel, true
			}
		}
	}

	scores := make([]float64, pos+1)
	for keyPos := 0; keyPos <= pos; keyPos++ {
		key := cache.IndexK[layer][keyPos*indexDim : (keyPos+1)*indexDim]
		var score float64
		for h := 0; h < indexHeads; h++ {
			headScore := dot64(indexQ[h], key) * scale
			if math.IsNaN(headScore) {
				return nil, false
			}
			if headScore < 0 {
				headScore = 0
			}
			score += float64(weights[h]) * headScore
		}
		scores[keyPos] = score
	}
	topK, ok := dsaTopKIndices([][]float64{scores}, []int{pos}, glmDsaPositions(pos+1), cfg.IndexTopK)
	if !ok || len(topK) != 1 {
		return nil, false
	}
	return topK[0], true
}

func (m *Model) glmDsaAppendAttentionKV(cache *glmDsaKVCache, layer, pos int, xn []float32, mat matKernel) ([]float32, bool) {
	cfg := m.Cfg
	H, nH := cfg.HiddenSize, cfg.NumHeads
	qLora, kvLora := cfg.QLoraRank, cfg.KVLoraRank
	qkNope, qkRope, vHead := cfg.QKNopeHeadDim, cfg.QKRopeHeadDim, cfg.VHeadDim
	qkHead := qkNope + qkRope
	if H == 0 || nH == 0 || qLora == 0 || kvLora == 0 || qkNope == 0 || qkRope == 0 || vHead == 0 || qkRope%2 != 0 {
		return nil, false
	}
	proj := func(name string, x []float32, out, in int) []float32 { return mat.mul(name, mat.prep(x), out, in) }

	ap := layerPrefix(layer) + "self_attn."
	qResid := proj(ap+"q_a_proj.weight", xn, qLora, H)
	addOptionalBias(qResid, m.tensorOptional(ap+"q_a_proj.bias"))
	qResid = rmsnorm(qResid, m.tensor(ap+"q_a_layernorm.weight"), glmDsaInnerNormEps)
	qFull := proj(ap+"q_b_proj.weight", qResid, nH*qkHead, qLora)

	compressedKV := proj(ap+"kv_a_proj_with_mqa.weight", xn, kvLora+qkRope, H)
	addOptionalBias(compressedKV, m.tensorOptional(ap+"kv_a_proj_with_mqa.bias"))
	kvLatent := rmsnorm(compressedKV[:kvLora], m.tensor(ap+"kv_a_layernorm.weight"), glmDsaInnerNormEps)
	kvFull := proj(ap+"kv_b_proj.weight", kvLatent, nH*(qkNope+vHead), kvLora)
	kRotRaw := compressedKV[kvLora:]

	cos, sin := ropeRowForLayer(cfg, layer, pos)
	query := make([]float32, nH*qkHead)
	key := make([]float32, nH*qkHead)
	keyRaw := make([]float32, nH*qkHead)
	value := make([]float32, nH*vHead)
	for h := 0; h < nH; h++ {
		qSrc := qFull[h*qkHead : (h+1)*qkHead]
		qDst := query[h*qkHead : (h+1)*qkHead]
		copy(qDst[:qkNope], qSrc[:qkNope])
		copy(qDst[qkNope:], glmDsaApplyInterleavedRoPE(qSrc[qkNope:], cos, sin))

		kvSrc := kvFull[h*(qkNope+vHead) : (h+1)*(qkNope+vHead)]
		kDst := key[h*qkHead : (h+1)*qkHead]
		kRawDst := keyRaw[h*qkHead : (h+1)*qkHead]
		copy(kDst[:qkNope], kvSrc[:qkNope])
		copy(kRawDst[:qkNope], kvSrc[:qkNope])
		copy(kRawDst[qkNope:], kRotRaw)
		copy(kDst[qkNope:], glmDsaApplyInterleavedRoPE(kRawDst[qkNope:], cos, sin))
		copy(value[h*vHead:(h+1)*vHead], kvSrc[qkNope:])
	}
	cache.K[layer] = append(cache.K[layer], key...)
	cache.Kraw[layer] = append(cache.Kraw[layer], keyRaw...)
	cache.V[layer] = append(cache.V[layer], value...)
	if len(cache.K[layer]) != (pos+1)*nH*qkHead ||
		len(cache.Kraw[layer]) != (pos+1)*nH*qkHead ||
		len(cache.V[layer]) != (pos+1)*nH*vHead {
		return nil, false
	}
	return query, true
}

func (m *Model) glmDsaAttendCached(cache *glmDsaKVCache, layer, pos int, query []float32, topK []int, mat matKernel) ([]float32, bool) {
	cfg := m.Cfg
	nH := cfg.NumHeads
	qkHead := cfg.QKNopeHeadDim + cfg.QKRopeHeadDim
	vHead := cfg.VHeadDim
	H := cfg.HiddenSize
	selected, ok := glmDsaSelectedCausalKeys(topK, pos, pos+1)
	if !ok || len(selected) == 0 || len(query) != nH*qkHead {
		return nil, false
	}
	scale := float32(1.0 / math.Sqrt(float64(qkHead)))
	attnConcat := m.glmDsaSparseAttend(cache, layer, query, selected, nH, qkHead, vHead, scale, mat)
	ap := layerPrefix(layer) + "self_attn."
	out := mat.mul(ap+"o_proj.weight", mat.prep(attnConcat), H, nH*vHead)
	addOptionalBias(out, m.tensorOptional(ap+"o_proj.bias"))
	return out, true
}

// glmDsaSparseAttend computes the GLM-DSA attention output over the already-SELECTED causal
// keys: per head, softmax(scale·dot(q_h, k_h))·v_h. When mat is a backendKernel whose backend
// advertises device sparse attention (compute.DSASparseBackend — the cpu-ref and, on the GPU,
// k_dsa_sparse_attend), it gathers the selected K/V rows contiguous and runs the attention math
// ON THE KERNEL; otherwise it runs the byte-for-byte host loop. The selection itself (which keys)
// is host-computed and fixed before this point, so the device path attends the SAME keys — its
// only divergence is f32 reduction order (the Approx class), keeping the forward argmax-exact.
// The host fallback is the exact loop glmDsaAttendCached carried before the device seam, so any
// session without a sparse-capable backend is unchanged.
func (m *Model) glmDsaSparseAttend(cache *glmDsaKVCache, layer int, query []float32, selected []int, nH, qkHead, vHead int, scale float32, mat matKernel) []float32 {
	if sk, ok := mat.(dsaSparseKernel); ok {
		nSel := len(selected)
		selK := make([]float32, nSel*nH*qkHead)
		selV := make([]float32, nSel*nH*vHead)
		for i, keyPos := range selected {
			copy(selK[i*nH*qkHead:(i+1)*nH*qkHead], cache.K[layer][keyPos*nH*qkHead:(keyPos+1)*nH*qkHead])
			copy(selV[i*nH*vHead:(i+1)*nH*vHead], cache.V[layer][keyPos*nH*vHead:(keyPos+1)*nH*vHead])
		}
		if out, ok := sk.sparseAttend(query, selK, selV, nSel, nH, qkHead, vHead, scale); ok {
			return out
		}
	}
	attnConcat := make([]float32, nH*vHead)
	for h := 0; h < nH; h++ {
		qh := query[h*qkHead : (h+1)*qkHead]
		scores := make([]float32, len(selected))
		for i, keyPos := range selected {
			kh := cache.K[layer][keyPos*nH*qkHead+h*qkHead : keyPos*nH*qkHead+(h+1)*qkHead]
			scores[i] = dot(qh, kh) * scale
		}
		softmaxInPlace(scores)
		oh := attnConcat[h*vHead : (h+1)*vHead]
		for i, keyPos := range selected {
			vh := cache.V[layer][keyPos*nH*vHead+h*vHead : keyPos*nH*vHead+(h+1)*vHead]
			w := scores[i]
			for d := 0; d < vHead; d++ {
				oh[d] += w * vh[d]
			}
		}
	}
	return attnConcat
}

func glmDsaPositions(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}
