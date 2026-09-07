package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestQSA_ConfigAndLayerIdentification verifies config layer identification and dynamic gating rules:
// - HasQSASparseAttn() and IsQSALayer(l) on hybrid config with linear_attention and full_attention / qsa.
// - ShouldUseQSASparseGather(l, nKV, batchSize) dynamic threshold gating, layer gating, and batch safety.
func TestQSA_ConfigAndLayerIdentification(t *testing.T) {
	hybridCfg := Config{
		NumLayers: 4,
		LayerTypes: []string{
			"linear_attention",
			"full_attention",
			"linear_attention",
			"qsa",
		},
	}

	if !hybridCfg.HasQSASparseAttn() {
		t.Fatal("hybridCfg.HasQSASparseAttn() = false, want true")
	}

	// Non-hybrid configs must return false for HasQSASparseAttn
	nonHybrid := Config{
		NumLayers:  2,
		LayerTypes: []string{"full_attention", "qsa"},
	}
	if nonHybrid.HasQSASparseAttn() {
		t.Error("nonHybrid.HasQSASparseAttn() = true, want false (missing linear_attention)")
	}

	linearOnly := Config{
		NumLayers:  2,
		LayerTypes: []string{"linear_attention", "linear_attention"},
	}
	if linearOnly.HasQSASparseAttn() {
		t.Error("linearOnly.HasQSASparseAttn() = true, want false (missing full_attention/qsa)")
	}

	// IsQSALayer checks
	if hybridCfg.IsQSALayer(0) {
		t.Errorf("hybridCfg.IsQSALayer(0) = true, want false for linear_attention")
	}
	if !hybridCfg.IsQSALayer(1) {
		t.Errorf("hybridCfg.IsQSALayer(1) = false, want true for full_attention")
	}
	if hybridCfg.IsQSALayer(2) {
		t.Errorf("hybridCfg.IsQSALayer(2) = true, want false for linear_attention")
	}
	if !hybridCfg.IsQSALayer(3) {
		t.Errorf("hybridCfg.IsQSALayer(3) = false, want true for qsa")
	}
	if hybridCfg.IsQSALayer(-1) {
		t.Errorf("hybridCfg.IsQSALayer(-1) = true, want false")
	}
	if hybridCfg.IsQSALayer(4) {
		t.Errorf("hybridCfg.IsQSALayer(4) = true, want false (out of bounds)")
	}

	// ShouldUseQSASparseGather:
	// 1. Returns false if not QSA layer
	if hybridCfg.ShouldUseQSASparseGather(0, 16384, 1) {
		t.Error("ShouldUseQSASparseGather(0, 16384, 1) = true, want false for non-QSA layer")
	}
	if hybridCfg.ShouldUseQSASparseGather(2, 20000, 1) {
		t.Error("ShouldUseQSASparseGather(2, 20000, 1) = true, want false for non-QSA layer")
	}

	// 2. Returns false if N_kv < 16,384 tokens (dynamic threshold gating)
	threshold := compute.QSADynamicGatingThreshold
	if threshold != 16384 {
		t.Fatalf("compute.QSADynamicGatingThreshold = %d, want 16384", threshold)
	}
	for _, nKV := range []int{0, 64, 1024, 8192, 16383} {
		if hybridCfg.ShouldUseQSASparseGather(1, nKV, 1) {
			t.Errorf("ShouldUseQSASparseGather(1, %d, 1) = true, want false for nKV < 16,384", nKV)
		}
		if hybridCfg.ShouldUseQSASparseGather(3, nKV, 1) {
			t.Errorf("ShouldUseQSASparseGather(3, %d, 1) = true, want false for nKV < 16,384", nKV)
		}
	}

	// 3. Returns false if batchSize > 1 (multi-sequence safety fallback)
	for _, bs := range []int{2, 3, 4, 8, 16} {
		if hybridCfg.ShouldUseQSASparseGather(1, 16384, bs) {
			t.Errorf("ShouldUseQSASparseGather(1, 16384, %d) = true, want false for batchSize > 1", bs)
		}
	}

	// 4. Returns true when layer is QSA layer, N_kv >= 16,384, and batchSize == 1
	for _, l := range []int{1, 3} {
		for _, nKV := range []int{16384, 16385, 32768, 65536, 131072} {
			if !hybridCfg.ShouldUseQSASparseGather(l, nKV, 1) {
				t.Errorf("ShouldUseQSASparseGather(%d, %d, 1) = false, want true", l, nKV)
			}
		}
	}
}

// TestQSA_SparseRowGatherDecodeEquivalence constructs a synthetic model/session with a QSA layer,
// populates the KV cache with N_kv = 16,384 tokens (256 blocks of 64 tokens) with deterministic
// sinusoidal values, executes both dense attention (attnDecodeOne) and sparse attention (attnDecodeQSA),
// and asserts relative L2 delta < 0.05% via ValidateQSAPerplexityParity.
func TestQSA_SparseRowGatherDecodeEquivalence(t *testing.T) {
	hd := 64
	nH := 4
	nKV := 2
	grp := 2
	w := nKV * hd
	scale := float32(1.0 / math.Sqrt(float64(hd)))

	cfg := Config{
		HeadDim:    hd,
		NumHeads:   nH,
		NumKVHeads: nKV,
		NumLayers:  2,
		LayerTypes: []string{"linear_attention", "qsa"},
	}

	totalTokens := 16384
	cache := NewKVCache(cfg)
	cache.pos = make([]int, totalTokens)
	for i := range cache.pos {
		cache.pos[i] = i
	}
	cache.K[1] = make([]float32, totalTokens*w)
	cache.V[1] = make([]float32, totalTokens*w)

	Q := make([]float32, nH*hd)
	for i := 0; i < nH*hd; i++ {
		Q[i] = float32(math.Sin(float64(i+1) * 0.1))
	}

	// Populate KV cache with N_kv = 16,384 tokens (256 blocks of 64 tokens)
	// using deterministic sinusoidal values. In accordance with QSA sparse attention
	// semantics, the selected top blocks and tail blocks concentrate attention mass
	// while unselected blocks receive low attention logits.
	for tTok := 0; tTok < totalTokens; tTok++ {
		b := tTok / compute.QSABlockSize
		isSelected := (b < compute.QSABaseTopKTokens/compute.QSABlockSize) ||
			(b >= (totalTokens-compute.QSALocalTailTokens)/compute.QSABlockSize)

		for i := 0; i < w; i++ {
			idx := tTok*w + i
			h := i / hd
			dim := i % hd
			sinVal := float32(math.Sin(float64(tTok*w+i)*0.017 + float64(b)*0.1))

			if isSelected {
				cache.K[1][idx] = Q[h*hd+dim] + sinVal*0.01
				cache.V[1][idx] = float32(math.Cos(float64(tTok*w+i) * 0.013))
			} else {
				cache.K[1][idx] = -Q[h*hd+dim]*2.0 + sinVal*0.01
				cache.V[1][idx] = float32(math.Cos(float64(tTok*w+i) * 0.013))
			}
		}
	}

	s := &Session{M: &Model{Cfg: cfg}, Cache: cache}
	db := &qDecodeBuf{}

	denseScores := make([]float32, nH*hd)
	sparseScores := make([]float32, nH*hd)

	_ = attnDecodeOne(denseScores, Q, cache, 1, nH, hd, w, grp, scale, fdot, fdot3scalar, nil)
	_ = s.attnDecodeQSA(sparseScores, Q, cache, db, 1, nH, hd, w, grp, scale, fdot, fdot3scalar)

	relL2, err := ValidateQSAPerplexityParity(denseScores, sparseScores)
	if err != nil {
		t.Fatalf("ValidateQSAPerplexityParity: %v", err)
	}

	if relL2 >= compute.QSAPerplexityDeltaTolerance {
		t.Fatalf("relative L2 delta %f >= QSAPerplexityDeltaTolerance (%f)", relL2, compute.QSAPerplexityDeltaTolerance)
	}
	t.Logf("QSA sparse vs dense relative L2 delta = %f (tolerance = %f)", relL2, compute.QSAPerplexityDeltaTolerance)
}

// TestQSA_MultiSequenceSafety verifies that when 2 or more sequences are batched,
// ShouldUseQSASparseGather returns false, guaranteeing fallback to dense masked attention
// to prevent cross-stream divergence.
func TestQSA_MultiSequenceSafety(t *testing.T) {
	cfg := Config{
		NumLayers:  2,
		LayerTypes: []string{"linear_attention", "qsa"},
	}

	for _, batchSize := range []int{2, 3, 4, 8, 16, 32, 64} {
		for _, nKV := range []int{16384, 20000, 32768, 65536, 131072} {
			if cfg.ShouldUseQSASparseGather(1, nKV, batchSize) {
				t.Errorf("ShouldUseQSASparseGather(1, %d, %d) = true, want false (multi-sequence safety fallback)", nKV, batchSize)
			}
		}
	}

	// Boundary check: single sequence with batchSize == 1 must succeed
	if !cfg.ShouldUseQSASparseGather(1, 16384, 1) {
		t.Errorf("ShouldUseQSASparseGather(1, 16384, 1) = false, want true for single sequence")
	}
}

// TestQSA_PerplexityParityTolerance verifies ValidateQSAPerplexityParity:
// - Passes for exact match (relL2 == 0) and small perturbations (< 0.05%).
// - Fails with error for divergences exceeding 0.05% (compute.QSAPerplexityDeltaTolerance).
// - Validates empty slice error guards.
func TestQSA_PerplexityParityTolerance(t *testing.T) {
	n := 256
	dense := make([]float32, n)
	for i := range dense {
		dense[i] = float32(math.Sin(float64(i+1)*0.1)) + 1.5
	}

	// 1. Exact match (0% delta)
	relL2Exact, err := ValidateQSAPerplexityParity(dense, dense)
	if err != nil {
		t.Errorf("exact match returned error: %v", err)
	}
	if relL2Exact != 0 {
		t.Errorf("exact match relL2 = %f, want 0", relL2Exact)
	}

	// 2. Small perturbation (< 0.05%, e.g. 0.01% perturbation = 0.0001)
	smallPerturbed := make([]float32, n)
	for i := range dense {
		smallPerturbed[i] = dense[i] * 1.0001
	}
	relL2Small, errSmall := ValidateQSAPerplexityParity(dense, smallPerturbed)
	if errSmall != nil {
		t.Errorf("small perturbation (< 0.05%%) returned error: %v", errSmall)
	}
	if relL2Small >= compute.QSAPerplexityDeltaTolerance {
		t.Errorf("small perturbation relL2 = %f >= tolerance %f", relL2Small, compute.QSAPerplexityDeltaTolerance)
	}

	// 3. Large perturbation (> 0.05%, e.g. 0.10% perturbation = 0.0010)
	largePerturbed := make([]float32, n)
	for i := range dense {
		largePerturbed[i] = dense[i] * 1.0010
	}
	relL2Large, errLarge := ValidateQSAPerplexityParity(dense, largePerturbed)
	if errLarge == nil {
		t.Errorf("large perturbation (> 0.05%%) expected error, got nil (relL2=%f)", relL2Large)
	}
	if relL2Large <= compute.QSAPerplexityDeltaTolerance {
		t.Errorf("large perturbation relL2 = %f <= tolerance %f", relL2Large, compute.QSAPerplexityDeltaTolerance)
	}

	// 4. Empty slice handling
	if _, err := ValidateQSAPerplexityParity(nil, dense); err == nil {
		t.Error("ValidateQSAPerplexityParity(nil, dense) expected error, got nil")
	}
	if _, err := ValidateQSAPerplexityParity(dense, nil); err == nil {
		t.Error("ValidateQSAPerplexityParity(dense, nil) expected error, got nil")
	}
	if _, err := ValidateQSAPerplexityParity([]float32{}, []float32{}); err == nil {
		t.Error("ValidateQSAPerplexityParity([], []) expected error, got nil")
	}

	// 5. Zero vector handling
	zeroVec := make([]float32, n)
	relL2Zero, errZero := ValidateQSAPerplexityParity(zeroVec, zeroVec)
	if errZero != nil {
		t.Errorf("ValidateQSAPerplexityParity(zero, zero) returned error: %v", errZero)
	}
	if relL2Zero != 0 {
		t.Errorf("zero vector relL2 = %f, want 0", relL2Zero)
	}
}
