package model

import "testing"

// mla_latent_kv_test.go — the golden correctness witness for the latent-resident MLA KV
// cache (issue #4364, epic #4352 colibri). It proves the cache is LOSSLESS: storing only
// the compressed [c_KV | k_R_raw] row per token and RECONSTRUCTING per-head K/V at read
// yields bit-for-bit the same K/V a naive cache would have MATERIALIZED and stored, and it
// witnesses the ~57x memory shrink at DeepSeek geometry numerically. No GPU and no weights
// on disk: the tiny synthetic MLA fixture (newSyntheticMLA) is the full correctness witness.

// materializedMLAKVRef is the INDEPENDENT reference: from a token's hidden state it
// materializes full per-head K and V the eager (pre-residency) way — down-project the
// hidden to the latent, up-project the latent to per-head K/V, then broadcast the rotated
// shared decoupled key into the front RopeDim lanes of every head's K. It builds from the
// ORIGINAL hidden (not from the cache's stored latent), so matching it proves the cache's
// store-latent-then-reconstruct path loses nothing. It reduces with the same in-order
// mlaMatRows primitive the production read path uses, so the agreement is bit-exact.
func materializedMLAKVRef(m *Model, hv []float32, pos int) (k, v []float32) {
	cfg := m.Cfg
	mla := m.MLA
	H, hd, nKV := cfg.HiddenSize, cfg.HeadDim, cfg.NumKVHeads
	w := nKV * hd

	cKV := mlaMatRows(mla.DownKV, hv, mla.KVLatentDim, H)
	k = mlaMatRows(mla.UpK, cKV, w, mla.KVLatentDim)
	v = mlaMatRows(mla.UpV, cKV, w, mla.KVLatentDim)

	kR := mlaMatRows(mla.DownR, hv, mla.RopeDim, H)
	cos, sin := ropeRowFromInv(mlaRopeInv(mla.RopeDim, cfg.RopeTheta), pos)
	applyRopeRow(kR, cos, sin)
	for h := 0; h < nKV; h++ {
		copy(k[h*hd:h*hd+mla.RopeDim], kR)
	}
	return k, v
}

// TestLatentMLAKVReconstructMatchesMaterialized is the #4364 golden: over a small synthetic
// GQA MLA fixture, every token's reconstruct-from-latent equals the independent materialized
// per-head K/V reference bit-for-bit, and the resident row is genuinely narrower than the
// materialized K/V it replaces.
func TestLatentMLAKVReconstructMatchesMaterialized(t *testing.T) {
	cfg := Config{
		HiddenSize:        16,
		NumLayers:         1,
		NumHeads:          4,
		NumKVHeads:        2, // GQA: reconstruction fills NumKVHeads*HeadDim, shared across query heads
		HeadDim:           6,
		IntermediateSize:  32,
		VocabSize:         40,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
	const latent, ropeDim = 6, 2
	m := newSyntheticMLA(cfg, latent, ropeDim)
	c := NewLatentMLAKVCache(m)

	// A tiny multi-token context, each hidden distinct and at a distinct absolute RoPE
	// position so the decoupled rotation genuinely varies per token.
	hiddens := [][]float32{
		makeVec(cfg.HiddenSize, 0.11, 1),
		makeVec(cfg.HiddenSize, -0.07, 2),
		makeVec(cfg.HiddenSize, 0.05, 3),
		makeVec(cfg.HiddenSize, -0.03, 4),
	}
	positions := []int{0, 1, 2, 3}

	for j, hv := range hiddens {
		if got := c.AppendHidden(hv, positions[j]); got != j {
			t.Fatalf("AppendHidden returned index %d, want %d", got, j)
		}
	}
	if c.Len() != len(hiddens) {
		t.Fatalf("cache Len()=%d, want %d", c.Len(), len(hiddens))
	}

	// The resident row is genuinely the compressed latent+k_R, narrower than the
	// materialized per-head K/V — otherwise the lossless witness would be vacuous.
	if c.ResidentFloatsPerToken() != latent+ropeDim {
		t.Fatalf("ResidentFloatsPerToken()=%d, want %d", c.ResidentFloatsPerToken(), latent+ropeDim)
	}
	if c.ResidentFloatsPerToken() >= c.MaterializedFloatsPerToken() {
		t.Fatalf("resident %d not smaller than materialized %d — fixture not exercising the shrink",
			c.ResidentFloatsPerToken(), c.MaterializedFloatsPerToken())
	}

	// GOLDEN: for every resident token, reconstruct-from-latent must equal the independent
	// materialized reference, bit-for-bit (both reduce with the same in-order primitive).
	for j, hv := range hiddens {
		kGot, vGot := c.ReconstructKV(j)
		kWant, vWant := materializedMLAKVRef(m, hv, positions[j])

		if len(kGot) != len(kWant) || len(vGot) != len(vWant) {
			t.Fatalf("token %d width mismatch: K %d/%d V %d/%d", j, len(kGot), len(kWant), len(vGot), len(vWant))
		}
		dk, ik := maxAbsDiff(kWant, kGot)
		dv, iv := maxAbsDiff(vWant, vGot)
		t.Logf("token %d: reconstructed vs materialized — max|dK|=%.3e@%d max|dV|=%.3e@%d", j, dk, ik, dv, iv)
		if dk != 0 {
			t.Fatalf("token %d reconstructed K != materialized K: max|d|=%.3e at lane %d", j, dk, ik)
		}
		if dv != 0 {
			t.Fatalf("token %d reconstructed V != materialized V: max|d|=%.3e at lane %d", j, dv, iv)
		}
	}

	// The resident rows are directly the (rows, positions) the production MLA read paths
	// consume (attendOne / attendOneAbsorbed) — reuse, not a parallel format.
	rows, pos := c.Rows()
	if len(rows) != c.Len() || len(pos) != c.Len() {
		t.Fatalf("Rows() returned %d rows / %d positions, want %d", len(rows), len(pos), c.Len())
	}
	if pos[len(pos)-1] != positions[len(positions)-1] {
		t.Fatalf("Rows() last position = %d, want %d", pos[len(pos)-1], positions[len(positions)-1])
	}
}

// TestLatentMLAKVCompressionRatioAtDeepSeekGeometry witnesses the "~57x smaller than
// materialized per-head K/V" claim numerically, at DeepSeek-V3 geometry (kv_lora_rank 512,
// rope 64; full-MHA baseline 128 heads x 128 dim). The accounting is a pure function of the
// geometry, so it needs no projection weights — a bare Model carrying only Cfg + MLA dims.
func TestLatentMLAKVCompressionRatioAtDeepSeekGeometry(t *testing.T) {
	m := &Model{
		Cfg: Config{NumKVHeads: 128, HeadDim: 128},
		MLA: &MLAConfig{KVLatentDim: 512, RopeDim: 64},
	}
	c := NewLatentMLAKVCache(m)

	if got := c.ResidentFloatsPerToken(); got != 576 {
		t.Errorf("ResidentFloatsPerToken()=%d, want 576 (512 latent + 64 rope)", got)
	}
	if got := c.MaterializedFloatsPerToken(); got != 32768 {
		t.Errorf("MaterializedFloatsPerToken()=%d, want 32768 (2 x 128 x 128)", got)
	}
	ratio := c.CompressionRatio()
	t.Logf("latent-resident MLA KV compression: %.1fx (resident %d f32/tok vs materialized %d f32/tok)",
		ratio, c.ResidentFloatsPerToken(), c.MaterializedFloatsPerToken())
	if ratio < 56 || ratio > 58 {
		t.Errorf("CompressionRatio()=%.2f, want ~57 (in [56,58])", ratio)
	}

	// Total footprint scales with token count; the per-token ratio is invariant.
	for i := 0; i < 10; i++ {
		c.AppendRow(make([]float32, c.ResidentFloatsPerToken()), i)
	}
	if c.ResidentFloats() != 10*576 || c.MaterializedFloats() != 10*32768 {
		t.Errorf("totals over 10 tokens: resident=%d materialized=%d, want %d/%d",
			c.ResidentFloats(), c.MaterializedFloats(), 10*576, 10*32768)
	}
}
