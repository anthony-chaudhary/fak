package model

// PackedPrefillMaxTokens bounds the complete heterogeneous panel, not each
// request. It is shared with the native scheduler's selected serving lane.
const PackedPrefillMaxTokens = 32

// PackedPrefillEligible reports the narrow host dense-pre-norm capability. The
// caller must exclusively own the session until the operation returns; this is
// not permission to race Cache, Close, mode changes, or model configuration.
func (s *Session) PackedPrefillEligible() bool {
	if s == nil || s.M == nil || s.Cache == nil || s.Backend != nil ||
		s.executionPolicy == ExecutionPolicyDeviceOnly || s.M.attnObs != nil ||
		s.Q4K || s.Q4 || s.F16 || s.GPTQ || s.Metal || s.MetalQ4K ||
		s.DenseGPULayers != 0 || s.GPULayers != 0 || s.PrecisionPolicy != nil ||
		s.tap != nil || s.tapActive != nil || s.captureTargetHidden ||
		s.cacheGeometryFailed || s.branchCacheRetired ||
		!batchPreNormFastPathOK(s.M.Cfg, s.Quant) ||
		s.M.Cfg.isGemma4() || s.M.Cfg.isMiniMaxSparseAttn() {
		return false
	}
	cfg, c := s.M.Cfg, s.Cache
	if cfg.HiddenSize <= 0 || cfg.IntermediateSize <= 0 || cfg.NumLayers <= 0 ||
		cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 || cfg.HeadDim%2 != 0 ||
		cfg.NumHeads%cfg.NumKVHeads != 0 || cfg.VocabSize <= 0 ||
		c.cfg.HiddenSize != cfg.HiddenSize || c.cfg.NumLayers != cfg.NumLayers ||
		c.cfg.NumHeads != cfg.NumHeads || c.cfg.NumKVHeads != cfg.NumKVHeads ||
		c.cfg.HeadDim != cfg.HeadDim {
		return false
	}
	maxInt := int(^uint(0) >> 1)
	for _, width := range []int{cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize, cfg.HeadDim} {
		if width > maxInt/PackedPrefillMaxTokens {
			return false
		}
	}
	if cfg.NumHeads > maxInt/cfg.HeadDim/PackedPrefillMaxTokens ||
		cfg.NumKVHeads > maxInt/cfg.HeadDim/PackedPrefillMaxTokens {
		return false
	}
	if len(c.Kraw) != cfg.NumLayers || len(c.K) != cfg.NumLayers || len(c.V) != cfg.NumLayers {
		return false
	}
	if c.Precision() != KVPrecisionFP32 {
		return false
	}
	for i, pos := range c.pos {
		if pos != i {
			return false
		}
	}
	width := cfg.NumKVHeads * cfg.HeadDim
	if c.Len() > maxInt/width-PackedPrefillMaxTokens {
		return false
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if len(c.Kraw[l]) != c.Len()*width || len(c.K[l]) != c.Len()*width || len(c.V[l]) != c.Len()*width {
			return false
		}
	}
	return true
}

// PrefillPacked executes one bounded unpadded panel and returns each request's
// own final-token logits. false means NO session/cache effect; no serial fallback
// is hidden inside this API. Sessions/caches must be distinct, caller-owned, and
// use the same model and weight mode. Returned rows are independent of scratch.
//
// Adapted from vLLM GPUModelRunner._prepare_inputs at 5c8ae7c7:
// cumulative query offsets, per-request positions, and last-row logits gather.
// This is synchronous host execution; no GPU/async scheduler behavior is claimed.
func (bs *BatchSession) PrefillPacked(prompts [][]int) ([][]float32, bool) {
	if bs == nil || bs.M == nil || len(prompts) == 0 ||
		len(prompts) != len(bs.Seqs) || len(prompts) > PackedPrefillMaxTokens {
		return nil, false
	}
	offsets := make([]int, len(prompts)+1)
	seen := make(map[*KVCache]bool, len(prompts))
	for b, p := range prompts {
		s := bs.Seqs[b]
		if len(p) == 0 || len(p) > PackedPrefillMaxTokens-offsets[b] ||
			!s.PackedPrefillEligible() || s.M != bs.M || s.Quant != bs.Quant || seen[s.Cache] {
			return nil, false
		}
		for _, id := range p {
			if id < 0 || id >= bs.M.Cfg.VocabSize {
				return nil, false
			}
		}
		seen[s.Cache] = true
		offsets[b+1] = offsets[b] + len(p)
	}
	if bs.Quant {
		return bs.prefillEachPackedQ(prompts, offsets, 0, true), true
	}
	return bs.prefillEachPackedF32(prompts, offsets, 0, true), true
}

func packedUniformOffsets(batch, tokens int) []int {
	offsets := make([]int, batch+1)
	for i := range offsets {
		offsets[i] = i * tokens
	}
	return offsets
}

func (m *Model) embedPackedRowsInto(dst []float32, prompts [][]int, offsets []int, rectangular, hidden int, cfg Config, workers int) {
	if rectangular > 0 {
		if !m.tryPrefillEmbedRectRows(dst, prompts, rectangular, hidden, cfg, workers) {
			m.embedRectRowsInto(dst, prompts, rectangular, hidden, cfg)
		}
		return
	}
	for b, prompt := range prompts {
		m.embedRectRowsInto(dst[offsets[b]*hidden:offsets[b+1]*hidden], [][]int{prompt}, len(prompt), hidden, cfg)
	}
}

func attnPrefillPackedF32Into(dst, q []float32, caches []*KVCache, bases, offsets []int, layer, rectangular, heads, dim, width, group, window int, scale float32) {
	if rectangular > 0 {
		attnPrefillMultiBuiltinInto(dst, q, caches, bases, layer, rectangular, heads, dim, width, group, window, scale, nil)
		return
	}
	for b := range caches {
		start, end := offsets[b]*heads*dim, offsets[b+1]*heads*dim
		attnPrefillMultiBuiltinInto(dst[start:end], q[start:end], caches[b:b+1], bases[b:b+1], layer, offsets[b+1]-offsets[b], heads, dim, width, group, window, scale, nil)
	}
}

func attnPrefillPackedQInto(dst, q, k, v []float32, caches []*KVCache, bases, offsets []int, layer, rectangular, heads, dim, width, group, window int, scale float32, scores [][]float32) [][]float32 {
	if rectangular > 0 {
		return attnPrefillMultiGQALoanedInto(dst, q, k, v, caches, bases, layer, rectangular, heads, dim, width, group, window, scale, scores)
	}
	for b := range caches {
		start, end := offsets[b]*heads*dim, offsets[b+1]*heads*dim
		kvStart, kvEnd := offsets[b]*width, offsets[b+1]*width
		scores = attnPrefillMultiGQALoanedInto(dst[start:end], q[start:end], k[kvStart:kvEnd], v[kvStart:kvEnd], caches[b:b+1], bases[b:b+1], layer, offsets[b+1]-offsets[b], heads, dim, width, group, window, scale, scores)
	}
	return scores
}
