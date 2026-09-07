package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestQSALayerDetectionAndGating verifies QSA layer classification and dynamic gating heuristics (Issue #465).
func TestQSALayerDetectionAndGating(t *testing.T) {
	// Qwen3.5 hybrid config with 3 linear_attention layers and 1 full_attention layer (layer 3).
	cfg := Config{
		NumLayers: 4,
		LayerTypes: []string{
			"linear_attention",
			"linear_attention",
			"linear_attention",
			"full_attention",
		},
	}

	if !cfg.IsQwen35Hybrid() {
		t.Fatal("cfg must be a Qwen3.5 hybrid config")
	}
	if !cfg.HasQSASparseAttn() {
		t.Fatal("cfg.HasQSASparseAttn() must be true for Qwen3.5 hybrid with full_attention layer")
	}

	// Verify IsQSALayer for each layer
	for l := 0; l < 3; l++ {
		if cfg.IsQSALayer(l) {
			t.Errorf("layer %d (linear_attention) must NOT be QSA layer", l)
		}
	}
	if !cfg.IsQSALayer(3) {
		t.Errorf("layer 3 (full_attention) must be QSA layer")
	}
	if cfg.IsQSALayer(-1) || cfg.IsQSALayer(4) {
		t.Errorf("out-of-bounds layer indices must NOT be QSA layer")
	}

	// Negative control: pure linear attention hybrid with no full_attention or qsa layer
	pureLinearCfg := Config{
		NumLayers:  2,
		LayerTypes: []string{"linear_attention", "linear_attention"},
	}
	if pureLinearCfg.HasQSASparseAttn() {
		t.Errorf("pureLinearCfg.HasQSASparseAttn() must be false")
	}

	// Verify ShouldUseQSASparseGather:
	// 1. Below 16,384 tokens (e.g. 10,000) -> returns false (dynamic gating).
	if cfg.ShouldUseQSASparseGather(3, 10000, 1) {
		t.Errorf("ShouldUseQSASparseGather(layer 3, 10000, 1) = true, want false (dynamic gating)")
	}
	if cfg.ShouldUseQSASparseGather(3, 16383, 1) {
		t.Errorf("ShouldUseQSASparseGather(layer 3, 16383, 1) = true, want false (dynamic gating boundary)")
	}

	// 2. At or above 16,384 tokens (e.g. 16,384, 78,000) with batchSize=1 -> returns true.
	if !cfg.ShouldUseQSASparseGather(3, 16384, 1) {
		t.Errorf("ShouldUseQSASparseGather(layer 3, 16384, 1) = false, want true")
	}
	if !cfg.ShouldUseQSASparseGather(3, 78000, 1) {
		t.Errorf("ShouldUseQSASparseGather(layer 3, 78000, 1) = false, want true")
	}

	// 3. Multi-sequence safety: with batchSize=2 -> returns false (falls back to dense masked attention).
	if cfg.ShouldUseQSASparseGather(3, 16384, 2) {
		t.Errorf("ShouldUseQSASparseGather(layer 3, 16384, 2) = true, want false (multi-sequence safety)")
	}
	if cfg.ShouldUseQSASparseGather(3, 20000, 2) {
		t.Errorf("ShouldUseQSASparseGather(layer 3, 20000, 2) = true, want false (multi-sequence safety)")
	}
	if cfg.ShouldUseQSASparseGather(3, 78000, 4) {
		t.Errorf("ShouldUseQSASparseGather(layer 3, 78000, 4) = true, want false (multi-sequence safety)")
	}

	// 4. On non-QSA layer (e.g. linear_attention) -> returns false regardless of token count.
	for l := 0; l < 3; l++ {
		if cfg.ShouldUseQSASparseGather(l, 16384, 1) {
			t.Errorf("ShouldUseQSASparseGather(layer %d, 16384, 1) = true, want false (non-QSA layer)", l)
		}
		if cfg.ShouldUseQSASparseGather(l, 78000, 1) {
			t.Errorf("ShouldUseQSASparseGather(layer %d, 78000, 1) = true, want false (non-QSA layer)", l)
		}
	}
}

// TestQSASparseRowGather_ParityAndNumericalEquivalence verifies numerical equivalence between
// dense masked attention over 20,000 tokens and true QSA sparse row gather attention (Issue #465).
func TestQSASparseRowGather_ParityAndNumericalEquivalence(t *testing.T) {
	// Configuration: N_kv = 20,000 tokens, 2 KV heads, 4 query heads (grp=2), HeadDim = 64
	totalTokens := 20000
	numKVHeads := 2
	numQueryHeads := 4
	headDim := 64
	grp := numQueryHeads / numKVHeads
	w := numKVHeads * headDim // row width = 128
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	cfg := Config{
		HiddenSize: numQueryHeads * headDim,
		HeadDim:    headDim,
		NumHeads:   numQueryHeads,
		NumKVHeads: numKVHeads,
		NumLayers:  2,
		LayerTypes: []string{"linear_attention", "full_attention"},
	}

	targetLayer := 1
	cache := NewKVCache(cfg)
	cache.pos = make([]int, totalTokens)
	for i := range cache.pos {
		cache.pos[i] = i
	}
	cache.K[targetLayer] = make([]float32, totalTokens*w)
	cache.V[targetLayer] = make([]float32, totalTokens*w)

	// Construct Query Q
	Q := make([]float32, numQueryHeads*headDim)
	for h := 0; h < numQueryHeads; h++ {
		for d := 0; d < headDim; d++ {
			Q[h*headDim+d] = float32(math.Sin(float64(d+1)*0.1)) + 0.5
		}
	}

	// Populate K/V with realistic synthetic values where specific token blocks have higher affinity with Q.
	// Total blocks for 20,000 tokens: (20000 + 64 - 1) / 64 = 313 blocks.
	// Tail blocks: 309, 310, 311, 312 (4 blocks = 256 tokens).
	// Candidate pool: blocks 0..308 (309 blocks).
	// Select 32 specific candidate blocks to have high affinity with Q.
	blockSize := compute.QSABlockSize
	totalBlocks := (totalTokens + blockSize - 1) / blockSize
	tailBlocks := compute.QSALocalTailTokens / blockSize // 4
	tailStartBlock := totalBlocks - tailBlocks           // 309

	highAffinityBlocks := make(map[int]bool, 32)
	for i := 0; i < 32; i++ {
		highAffinityBlocks[i*8] = true // blocks 0, 8, 16, ..., 248
	}

	for tTok := 0; tTok < totalTokens; tTok++ {
		b := tTok / blockSize
		isHighAffinity := highAffinityBlocks[b] || (b >= tailStartBlock)

		for i := 0; i < w; i++ {
			idx := tTok*w + i
			kvh := i / headDim
			dim := i % headDim
			sinVal := float32(math.Sin(float64(tTok*w+i)*0.017 + float64(b)*0.1))

			// Query head corresponding to this KV head
			qh := kvh * grp
			qVal := Q[qh*headDim+dim]

			if isHighAffinity {
				// Strongly aligned with Q: dot product is positive and large
				cache.K[targetLayer][idx] = qVal + sinVal*0.01
			} else {
				// Strongly opposing Q: dot product is negative and large
				cache.K[targetLayer][idx] = -qVal*2.0 + sinVal*0.01
			}
			cache.V[targetLayer][idx] = float32(math.Cos(float64(tTok*w+i) * 0.013))
		}
	}

	// 1. Run dense attention over all 20,000 tokens (reference)
	denseAttnOut := make([]float32, numQueryHeads*headDim)
	denseScoresScratch := make([][]float32, grp)
	for g := range denseScoresScratch {
		denseScoresScratch[g] = make([]float32, totalTokens)
	}
	attnDecodeOne(denseAttnOut, Q, cache, targetLayer, numQueryHeads, headDim, w, grp, scale, fdot, fdot3scalar, denseScoresScratch)

	// 2. Run QSA sparse row gather attention:
	//    - Score blocks
	//    - Select top-k 32 blocks (2048 tok) + 4 tail blocks (256 tok)
	//    - Gather rows into contiguous scratch
	//    - Evaluate attention over gathered rows
	s := &Session{
		M:     &Model{Cfg: cfg},
		Cache: cache,
		Quant: true,
	}
	db := &qDecodeBuf{}

	// (a) Score blocks and verify selection
	q0 := vectorHead(Q, 0, headDim)
	blockScores := make([]float32, totalBlocks)
	for b := 0; b < totalBlocks; b++ {
		midToken := b*blockSize + (blockSize / 2)
		if midToken >= totalTokens {
			midToken = totalTokens - 1
		}
		kMid := packedHead(cache.K[targetLayer], midToken, w, 0, headDim)
		blockScores[b] = fdot(q0, kMid) * scale
	}

	// (b) Select top-k 32 blocks + 4 tail blocks via RadixTopKBlockSelect
	topKBlocks := compute.QSABaseTopKTokens / blockSize
	selectedBlocks, receipt, err := compute.RadixTopKBlockSelect(blockScores, totalBlocks, topKBlocks, tailBlocks)
	if err != nil {
		t.Fatalf("RadixTopKBlockSelect: %v", err)
	}
	if receipt.DynamicGatingBypassed {
		t.Fatal("receipt.DynamicGatingBypassed must be false for 20,000 tokens")
	}
	if len(selectedBlocks) != 36 {
		t.Fatalf("len(selectedBlocks) = %d, want 36 (32 top-k + 4 tail)", len(selectedBlocks))
	}
	if receipt.SelectedBlocks != 36 {
		t.Fatalf("receipt.SelectedBlocks = %d, want 36", receipt.SelectedBlocks)
	}

	// Verify all tail blocks are selected
	for tb := tailStartBlock; tb < totalBlocks; tb++ {
		found := false
		for _, b := range selectedBlocks {
			if int(b) == tb {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("tail block %d missing from selectedBlocks", tb)
		}
	}

	// Verify all 32 high affinity blocks are selected
	for hab := range highAffinityBlocks {
		found := false
		for _, b := range selectedBlocks {
			if int(b) == hab {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("high affinity block %d missing from selectedBlocks", hab)
		}
	}

	// (c) Gather rows into contiguous scratch
	gatheredK, gatheredV, err := compute.SparseRowGatherKV(
		cache.K[targetLayer], cache.V[targetLayer],
		selectedBlocks, blockSize, numKVHeads, headDim, totalTokens,
	)
	if err != nil {
		t.Fatalf("SparseRowGatherKV: %v", err)
	}
	expectedGatherTokens := 0
	for _, b := range selectedBlocks {
		toks := blockSize
		if int(b)*blockSize+toks > totalTokens {
			toks = totalTokens - int(b)*blockSize
		}
		if toks > 0 {
			expectedGatherTokens += toks
		}
	}
	expectedGatherElements := expectedGatherTokens * w
	if len(gatheredK) != expectedGatherElements || len(gatheredV) != expectedGatherElements {
		t.Fatalf("gathered KV elements = (%d, %d), want %d", len(gatheredK), len(gatheredV), expectedGatherElements)
	}

	// (d) Evaluate attention over gathered rows via session's attnDecodeQSA
	sparseAttnOut := make([]float32, numQueryHeads*headDim)
	s.attnDecodeQSA(sparseAttnOut, Q, cache, db, targetLayer, numQueryHeads, headDim, w, grp, scale, fdot, fdot3scalar)

	// 3. Assert numerical equivalence: ValidateQSAPerplexityParity reports relative delta < 0.05% (0.0005)
	relL2Attn, err := ValidateQSAPerplexityParity(denseAttnOut, sparseAttnOut)
	if err != nil {
		t.Fatalf("ValidateQSAPerplexityParity(attnOut) returned error: %v (relL2=%e)", err, relL2Attn)
	}
	t.Logf("QSA sparse vs dense attention relative L2 delta = %e (tolerance = %e)", relL2Attn, compute.QSAPerplexityDeltaTolerance)
	if relL2Attn >= compute.QSAPerplexityDeltaTolerance {
		t.Fatalf("QSA relative L2 delta %e >= tolerance %e (0.05%%)", relL2Attn, compute.QSAPerplexityDeltaTolerance)
	}

	// 4. Assert top token logits/argmax matches reference
	vocabSize := 1000
	denseLogits := make([]float32, vocabSize)
	sparseLogits := make([]float32, vocabSize)

	for v := 0; v < vocabSize; v++ {
		var dotDense, dotSparse float32
		for d := 0; d < numQueryHeads*headDim; d++ {
			weight := float32(math.Sin(float64((v+1)*(d+1)) * 0.007))
			dotDense += weight * denseAttnOut[d]
			dotSparse += weight * sparseAttnOut[d]
		}
		denseLogits[v] = dotDense
		sparseLogits[v] = dotSparse
	}

	argmaxDense := 0
	maxValDense := denseLogits[0]
	for v := 1; v < vocabSize; v++ {
		if denseLogits[v] > maxValDense {
			maxValDense = denseLogits[v]
			argmaxDense = v
		}
	}

	argmaxSparse := 0
	maxValSparse := sparseLogits[0]
	for v := 1; v < vocabSize; v++ {
		if sparseLogits[v] > maxValSparse {
			maxValSparse = sparseLogits[v]
			argmaxSparse = v
		}
	}

	t.Logf("Top token logits: dense=%d (val=%f), sparse=%d (val=%f)", argmaxDense, maxValDense, argmaxSparse, maxValSparse)
	if argmaxDense != argmaxSparse {
		t.Fatalf("top token argmax mismatch: dense=%d, sparse=%d (denseVal=%f, sparseVal=%f)",
			argmaxDense, argmaxSparse, maxValDense, maxValSparse)
	}

	relL2Logits, err := ValidateQSAPerplexityParity(denseLogits, sparseLogits)
	if err != nil {
		t.Fatalf("ValidateQSAPerplexityParity(logits) returned error: %v (relL2=%e)", err, relL2Logits)
	}
	t.Logf("QSA sparse vs dense logits relative L2 delta = %e (tolerance = %e)", relL2Logits, compute.QSAPerplexityDeltaTolerance)
	if relL2Logits >= compute.QSAPerplexityDeltaTolerance {
		t.Fatalf("Logits relative L2 delta %e >= tolerance %e (0.05%%)", relL2Logits, compute.QSAPerplexityDeltaTolerance)
	}
}

// TestQSADecodeSessionBufferReservation verifies pre-allocation of QSA scratch buffers upon Session.Reserve (Issue #465).
func TestQSADecodeSessionBufferReservation(t *testing.T) {
	cfg := Config{
		HiddenSize:       256,
		HeadDim:          64,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 512,
		VocabSize:        1000,
		NumLayers:        4,
		LayerTypes: []string{
			"linear_attention",
			"linear_attention",
			"linear_attention",
			"full_attention",
		},
	}

	m := &Model{Cfg: cfg}
	s := &Session{
		M:     m,
		Cache: NewKVCache(cfg),
		Quant: true,
	}

	extraPositions := 20000
	s.Reserve(extraPositions)

	if s.qDecode == nil {
		t.Fatal("s.qDecode is nil after Reserve")
	}

	w := cfg.NumKVHeads * cfg.HeadDim                                                    // 2 * 64 = 128
	expectedGatherLen := compute.QSAMaxGatherTokens * w                                  // 2304 * 128 = 294,912
	expectedBlocks := (extraPositions + compute.QSABlockSize - 1) / compute.QSABlockSize // (20000 + 63) / 64 = 313

	if len(s.qDecode.qsaGatheredK) != expectedGatherLen {
		t.Errorf("len(qsaGatheredK) = %d, want %d", len(s.qDecode.qsaGatheredK), expectedGatherLen)
	}
	if len(s.qDecode.qsaGatheredV) != expectedGatherLen {
		t.Errorf("len(qsaGatheredV) = %d, want %d", len(s.qDecode.qsaGatheredV), expectedGatherLen)
	}
	if len(s.qDecode.qsaBlockScores) != expectedBlocks {
		t.Errorf("len(qsaBlockScores) = %d, want %d", len(s.qDecode.qsaBlockScores), expectedBlocks)
	}

	// Negative control: verify non-QSA config does not allocate QSA buffers
	nonQSACfg := Config{
		HiddenSize:       256,
		HeadDim:          64,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 512,
		VocabSize:        1000,
		NumLayers:        2,
		LayerTypes:       []string{"linear_attention", "linear_attention"},
	}
	sNonQSA := &Session{
		M:     &Model{Cfg: nonQSACfg},
		Cache: NewKVCache(nonQSACfg),
		Quant: true,
	}
	sNonQSA.Reserve(extraPositions)
	if sNonQSA.qDecode == nil {
		t.Fatal("sNonQSA.qDecode is nil after Reserve")
	}
	if len(sNonQSA.qDecode.qsaGatheredK) != 0 {
		t.Errorf("non-QSA qsaGatheredK should be empty, got len %d", len(sNonQSA.qDecode.qsaGatheredK))
	}
	if len(sNonQSA.qDecode.qsaGatheredV) != 0 {
		t.Errorf("non-QSA qsaGatheredV should be empty, got len %d", len(sNonQSA.qDecode.qsaGatheredV))
	}
	if len(sNonQSA.qDecode.qsaBlockScores) != 0 {
		t.Errorf("non-QSA qsaBlockScores should be empty, got len %d", len(sNonQSA.qDecode.qsaBlockScores))
	}
}
