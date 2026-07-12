package model

import "testing"

// TestMLAAbsorbedMatchesNaive is the parity witness for issue #4356: the absorbed MLA
// read path (attendOneAbsorbed — W_UK folded into the query, W_UV folded in after
// softmax, scored directly against the compressed latent) must equal the naive
// reconstruct-then-attend path (attendOne over mlaKVLayout.reconstructKV) on the same
// cache. They are mathematically identical; only the reduction order differs, so the
// gate is a float tolerance, not bit-exact.
//
// The fixture is GQA (NumHeads=4, NumKVHeads=2, grp=2) so the h/GroupSize head mapping
// is exercised — TestMLANaiveMatchesReference only covers MHA (grp=1) — and the latent
// is narrower than per-head K so absorption is doing the shrink it exists for.
func TestMLAAbsorbedMatchesNaive(t *testing.T) {
	cfg := Config{
		HiddenSize:        24,
		NumLayers:         1,
		NumHeads:          4,
		NumKVHeads:        2, // GQA: grp = 2, so multiple query heads share a kv head
		HeadDim:           8,
		IntermediateSize:  48,
		VocabSize:         50,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
	const latent, ropeDim = 6, 4
	m := newSyntheticMLA(cfg, latent, ropeDim)

	// A small decode context (S <= 4 cached positions), each a distinct hidden at a
	// distinct absolute RoPE position so the decoupled rotation genuinely varies.
	hiddens := [][]float32{
		makeVec(cfg.HiddenSize, 0.13, 1),
		makeVec(cfg.HiddenSize, -0.09, 2),
		makeVec(cfg.HiddenSize, 0.04, 3),
		makeVec(cfg.HiddenSize, -0.02, 4),
	}
	positions := []int{0, 1, 2, 3}

	rows := make([][]float32, len(hiddens))
	for j, hv := range hiddens {
		rows[j] = m.mlaProject(hv, positions[j])
	}
	// Query for the last (decode) position, built the same way the naive test does.
	q := buildMLAQuery(m, hiddens[len(hiddens)-1], positions[len(positions)-1])

	naive := attendOne(m, mlaKVLayout{}, 0, q, rows, positions)
	absorbed := attendOneAbsorbed(m, 0, q, rows, positions)

	if len(absorbed) != len(naive) {
		t.Fatalf("absorbed output width = %d, want %d", len(absorbed), len(naive))
	}
	d, idx := maxAbsDiff(naive, absorbed)
	t.Logf("absorbed vs naive MLA attention: max|Δ|=%.3e at lane %d", d, idx)
	// Reduction order differs (fold-then-score vs score-then-fold); tolerate float32
	// rounding but nothing larger — a real divergence blows past this immediately.
	if d > 1e-5 {
		t.Fatalf("absorbed MLA attention != naive: max|Δ|=%.3e at lane %d\n absorbed=%v\n    naive=%v",
			d, idx, absorbed, naive)
	}

	// The witness is non-vacuous only if the latent is narrower than per-head K — i.e.
	// absorption is actually scoring a compressed context, not an equal-width one.
	if w := cfg.NumKVHeads * cfg.HeadDim; latent+ropeDim >= w {
		t.Fatalf("fixture not exercising the shrink: latent+rope=%d >= per-head K width=%d",
			latent+ropeDim, w)
	}
}

// TestAbsorbedMLASelected pins the auto-select gate (issue #4356: S<=4 & kv_lora<=512).
// Absorption is a decode-side win, so it is chosen only for small decode/verify batches
// against a compressed latent; prefill (large batch), wide latents, and non-MLA models
// keep the naive reconstruct-then-attend path.
func TestAbsorbedMLASelected(t *testing.T) {
	cfg := Config{
		HiddenSize: 24, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 48, VocabSize: 50, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	mla := newSyntheticMLA(cfg, 6, 4)

	cases := []struct {
		name      string
		m         *Model
		batchSize int
		want      bool
	}{
		{"decode-batch-1", mla, 1, true},
		{"verify-batch-4", mla, 4, true},
		{"prefill-batch-5", mla, 5, false},
		{"prefill-large-batch", mla, 64, false},
		{"non-mla-model", NewSynthetic(cfg), 1, false},
	}
	for _, tc := range cases {
		if got := absorbedMLASelected(tc.m, tc.batchSize); got != tc.want {
			t.Errorf("absorbedMLASelected(%s, S=%d) = %v, want %v",
				tc.name, tc.batchSize, got, tc.want)
		}
	}

	// The wide-latent cutoff: an otherwise decode-shaped step whose latent exceeds the
	// kv_lora<=512 bound stays naive even at batch 1.
	wide := newSyntheticMLA(cfg, 513, 4)
	if absorbedMLASelected(wide, 1) {
		t.Errorf("absorbedMLASelected wide latent (KVLatentDim=513) = true, want false")
	}
}
