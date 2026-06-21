package model

import (
	"math"
	"testing"
)

// TestStandardLayoutNoOp is the load-bearing no-op gate (issue #25 acceptance: the
// Llama GQA default kvLayout must stay byte-identical). It proves the standard
// kvLayout is a faithful description of the per-head layout blockStep already builds:
// reconstructing K/V through standardKVLayout and scoring through attendOne yields
// EXACTLY the inline blockStep attention output, max|Δ|=0. If the kvLayout seam ever
// drifts the Llama read path, this fails bit-for-bit.
func TestStandardLayoutNoOp(t *testing.T) {
	cfg := Config{
		HiddenSize:        32,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         97,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
	m := NewSynthetic(cfg)

	if got := m.modelLayout().name(); got != "standard" {
		t.Fatalf("Llama model selected layout %q, want standard", got)
	}
	if str := m.modelLayout().cacheStride(m); str != cfg.NumKVHeads*cfg.HeadDim {
		t.Fatalf("standard cacheStride = %d, want %d", str, cfg.NumKVHeads*cfg.HeadDim)
	}

	prompt := []int{3, 17, 5, 23, 41, 2, 19}
	s := m.NewSession()
	s.Prefill(prompt)

	layout := standardKVLayout{}
	w := cfg.NumKVHeads * cfg.HeadDim
	nPos := s.Cache.Len()

	// For every layer, recompute the LAST query position's attention two ways:
	//   (ref)  the verbatim inline blockStep attention loop, reading the cache directly.
	//   (got)  attendOne over standard-layout rows built from the same cache.
	// Assert max|Δ|=0 across all heads/lanes/layers — the no-op witness.
	var maxD float64
	for l := 0; l < cfg.NumLayers; l++ {
		// A deterministic query for this layer (any fixed vector exercises the path; we
		// reuse the cached K of the last position so heads have non-degenerate scores).
		q := make([]float32, cfg.NumHeads*cfg.HeadDim)
		for i := range q {
			q[i] = float32(math.Sin(float64(i+1)*0.37 + float64(l)))
		}

		ref := inlineStandardAttention(m, l, q, s.Cache, nPos)

		// Build standard-layout rows [K|V] per cached position from the live cache.
		rows := make([][]float32, nPos)
		positions := make([]int, nPos)
		for j := 0; j < nPos; j++ {
			row := make([]float32, 2*w)
			copy(row[:w], s.Cache.K[l][j*w:(j+1)*w])
			copy(row[w:], s.Cache.V[l][j*w:(j+1)*w])
			rows[j] = row
			positions[j] = s.Cache.pos[j]
		}
		got := attendOne(m, layout, l, q, rows, positions)

		d, _ := maxAbsDiff(ref, got)
		if d > maxD {
			maxD = d
		}
	}
	t.Logf("standard kvLayout vs inline blockStep attention: max|Δ|=%.3e", maxD)
	if maxD != 0 {
		t.Fatalf("standard kvLayout attention drifted from inline path: max|Δ|=%.3e, want 0", maxD)
	}
}

// inlineStandardAttention is the verbatim per-head causal GQA read loop from
// blockStep (kv.go), isolated as the reference the standard kvLayout must reproduce
// bit-for-bit. It scores q against the cache's stored post-RoPE K and weighted-sums V.
func inlineStandardAttention(m *Model, l int, q []float32, cache *KVCache, nPos int) []float32 {
	cfg := m.Cfg
	nH, hd := cfg.NumHeads, cfg.HeadDim
	grp := cfg.GroupSize()
	w := cfg.NumKVHeads * hd
	scale := float32(1.0 / math.Sqrt(float64(hd)))

	out := make([]float32, nH*hd)
	for h := 0; h < nH; h++ {
		kvh := h / grp
		qh := q[h*hd : (h+1)*hd]
		scores := make([]float32, nPos)
		for j := 0; j < nPos; j++ {
			kh := cache.K[l][j*w+kvh*hd : j*w+(kvh+1)*hd]
			scores[j] = dot(qh, kh) * scale
		}
		softmaxInPlace(scores)
		oh := out[h*hd : (h+1)*hd]
		for j := 0; j < nPos; j++ {
			vh := cache.V[l][j*w+kvh*hd : j*w+(kvh+1)*hd]
			wj := scores[j]
			for d := 0; d < hd; d++ {
				oh[d] += wj * vh[d]
			}
		}
	}
	return out
}

// newSyntheticMLA builds a tiny in-test MLA geometry with deterministic LCG weights
// (the synthetic.go discipline: no torch, no HF, no DeepSeek download). It returns a
// Model whose MLA field is set, so modelLayout() picks the MLA kvLayout. The widths
// are deliberately small so the naive reconstruction is hand-checkable.
func newSyntheticMLA(cfg Config, latent, ropeDim int) *Model {
	m := NewSynthetic(cfg)
	H := cfg.HiddenSize
	w := cfg.NumKVHeads * cfg.HeadDim

	seed := uint64(0xD1B54A32D192ED03)
	next := func() float32 {
		seed = seed*6364136223846793005 + 1442695040888963407
		u := float32(seed>>40) / float32(1<<24)
		return (u*2 - 1) * 0.1
	}
	fill := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = next()
		}
		return v
	}
	m.MLA = &MLAConfig{
		KVLatentDim: latent,
		RopeDim:     ropeDim,
		DownKV:      fill(latent * H),
		UpK:         fill(w * latent),
		UpV:         fill(w * latent),
		DownR:       fill(ropeDim * H),
	}
	return m
}

// TestMLANaiveMatchesReference is acceptance gate #2's structural core (the naive
// decompress-then-attend form, hand-checkable). On a tiny synthetic MLA config it
// runs the production naive path (mlaProject write + mlaKVLayout.reconstructKV read +
// attendOne) and asserts it equals an INDEPENDENT straight-line reference computed in
// the test — the whole point of pinning the naive form is that it reduces to ordinary
// MHA and is therefore checkable without weights or an HF oracle.
func TestMLANaiveMatchesReference(t *testing.T) {
	cfg := Config{
		HiddenSize:        16,
		NumLayers:         1,
		NumHeads:          2,
		NumKVHeads:        2, // MHA (grp=1) keeps the reference maximally simple
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

	if got := m.modelLayout().name(); got != "mla" {
		t.Fatalf("MLA model selected layout %q, want mla", got)
	}
	if str := m.modelLayout().cacheStride(m); str != latent+ropeDim {
		t.Fatalf("MLA cacheStride = %d, want %d", str, latent+ropeDim)
	}

	H, hd, nKV, nH := cfg.HiddenSize, cfg.HeadDim, cfg.NumKVHeads, cfg.NumHeads
	w := nKV * hd

	// A tiny 3-position sequence. Each position's hidden is a fixed deterministic
	// vector (a stand-in for a normed hidden); positions carry distinct absolute RoPE
	// indices so the decoupled-RoPE rotation actually differs per position.
	hiddens := [][]float32{
		makeVec(H, 0.11, 1),
		makeVec(H, -0.07, 2),
		makeVec(H, 0.05, 3),
	}
	positions := []int{0, 1, 2}

	layout := mlaKVLayout{}

	// ---- production naive path: write cache rows, then attend the LAST position ----
	rows := make([][]float32, len(hiddens))
	for j, hv := range hiddens {
		rows[j] = m.mlaProject(hv, positions[j])
		if len(rows[j]) != latent+ropeDim {
			t.Fatalf("mlaProject row width = %d, want %d", len(rows[j]), latent+ropeDim)
		}
	}
	// Query for the last position: content q from q_proj over the last hidden, with the
	// front RopeDim lanes of each head replaced by the rotated decoupled query q_R. The
	// reference below builds the SAME query, so the test pins both write and read.
	q := buildMLAQuery(m, hiddens[len(hiddens)-1], positions[len(positions)-1])
	got := attendOne(m, layout, 0, q, rows, positions)

	// ---- independent straight-line reference (no calls into kvlayout.go) ----------
	want := mlaNaiveReference(m, q, hiddens, positions)

	d, idx := maxAbsDiff(want, got)
	t.Logf("MLA naive path vs hand reference: max|Δ|=%.3e at lane %d", d, idx)
	if d > 1e-6 {
		t.Fatalf("MLA naive attention != reference: max|Δ|=%.3e at lane %d\n got=%v\nwant=%v",
			d, idx, got, want)
	}

	// Sanity: the cached row is genuinely the latent+k_R, far smaller than per-head
	// K (the data-model shrink MLA is for). latent+ropeDim must be < NumKVHeads*HeadDim
	// for the witness to be non-vacuous on this fixture.
	if latent+ropeDim >= w {
		t.Fatalf("fixture not exercising the shrink: latent+rope=%d >= per-head K width=%d",
			latent+ropeDim, w)
	}
	_ = nH
}

// buildMLAQuery constructs the MLA query for one position the SAME way the reference
// expects: content query from q_proj per head, with the front RopeDim lanes of each
// head overwritten by the rotated decoupled query q_R (derived from DownR, like k_R).
func buildMLAQuery(m *Model, hv []float32, pos int) []float32 {
	cfg := m.Cfg
	mla := m.MLA
	H, hd, nH := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads

	q := mlaMatRows(m.tensor(layerName(0, "self_attn.q_proj.weight")), hv, nH*hd, H)
	// decoupled query: same DownR projection as the key, rotated at pos.
	qR := mlaMatRows(mla.DownR, hv, mla.RopeDim, H)
	cos, sin := ropeRowFromInv(mlaRopeInv(mla.RopeDim, cfg.RopeTheta), pos)
	applyRopeRow(qR, cos, sin)
	for h := 0; h < nH; h++ {
		copy(q[h*hd:h*hd+mla.RopeDim], qR)
	}
	return q
}

// mlaNaiveReference recomputes the naive MLA attention output for the LAST position
// from scratch — up-project each cached latent to per-head K/V, broadcast the rotated
// decoupled key, score, softmax, weighted-sum V — using only local arithmetic and the
// raw MLA weights. It deliberately does NOT call into kvlayout.go so it is a genuine
// independent witness of the production path.
func mlaNaiveReference(m *Model, q []float32, hiddens [][]float32, positions []int) []float32 {
	cfg := m.Cfg
	mla := m.MLA
	H, hd, nKV, nH := cfg.HiddenSize, cfg.HeadDim, cfg.NumKVHeads, cfg.NumHeads
	grp := cfg.GroupSize()
	w := nKV * hd
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	nPos := len(hiddens)

	// Reconstruct per-head K/V for every cached position, by hand.
	ks := make([][]float32, nPos)
	vs := make([][]float32, nPos)
	for j := 0; j < nPos; j++ {
		// down-project to latent, then up-project to per-head K/V.
		cKV := mlaMatRows(mla.DownKV, hiddens[j], mla.KVLatentDim, H)
		k := mlaMatRows(mla.UpK, cKV, w, mla.KVLatentDim)
		v := mlaMatRows(mla.UpV, cKV, w, mla.KVLatentDim)
		// rotated decoupled key, broadcast into the front RopeDim lanes of every head.
		kR := mlaMatRows(mla.DownR, hiddens[j], mla.RopeDim, H)
		cos, sin := ropeRowFromInv(mlaRopeInv(mla.RopeDim, cfg.RopeTheta), positions[j])
		applyRopeRow(kR, cos, sin)
		for h := 0; h < nKV; h++ {
			copy(k[h*hd:h*hd+mla.RopeDim], kR)
		}
		ks[j], vs[j] = k, v
	}

	out := make([]float32, nH*hd)
	for h := 0; h < nH; h++ {
		kvh := h / grp
		qh := q[h*hd : (h+1)*hd]
		scores := make([]float32, nPos)
		for j := 0; j < nPos; j++ {
			scores[j] = dot(qh, ks[j][kvh*hd:(kvh+1)*hd]) * scale
		}
		softmaxInPlace(scores)
		oh := out[h*hd : (h+1)*hd]
		for j := 0; j < nPos; j++ {
			vh := vs[j][kvh*hd : (kvh+1)*hd]
			for d := 0; d < hd; d++ {
				oh[d] += scores[j] * vh[d]
			}
		}
	}
	return out
}

// makeVec builds a deterministic vector of width n (a stand-in hidden state).
func makeVec(n int, base float32, salt int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = base + float32(math.Sin(float64((i+1)*(salt+1))*0.21))*0.1
	}
	return v
}
