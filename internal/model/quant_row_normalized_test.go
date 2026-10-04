package model

import (
	"math/rand"
	"strings"
	"testing"
)

// quant_row_normalized_test.go — model-level witnesses for fak#13567: the qwen35 row-permuted
// projections (GDN in_proj_qkv/z/a/b, full-attention q/k) served from the native q4kw/kqw stores
// must produce the same logits as the same weights served from the Q8 store, on the batched
// prefill AND the per-token decode loop, without any reader panicking on a missing Q8 copy.

type rowNormalizedFixture struct {
	name string
	out  int
	q6   bool
}

func rowNormalizedFixtures(cfg Config) []rowNormalizedFixture {
	_, nV, _, _, _, valDim, convDim := cfg.linearAttnDims()
	var fx []rowNormalizedFixture
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			fx = append(fx,
				// Layer 1 ships its GDN QKV as Q6_K (the q4_k_m mix), the rest as Q4_K.
				rowNormalizedFixture{layerName(l, "linear_attn.in_proj_qkv.weight"), convDim, l == 1},
				rowNormalizedFixture{layerName(l, "linear_attn.in_proj_z.weight"), valDim, false},
				rowNormalizedFixture{layerName(l, "linear_attn.in_proj_a.weight"), nV, false},
				rowNormalizedFixture{layerName(l, "linear_attn.in_proj_b.weight"), nV, false},
			)
			continue
		}
		fx = append(fx,
			rowNormalizedFixture{layerName(l, "self_attn.q_proj.weight"), 2 * cfg.NumHeads * cfg.HeadDim, false},
			rowNormalizedFixture{layerName(l, "self_attn.k_proj.weight"), cfg.NumKVHeads * cfg.HeadDim, false},
		)
	}
	return fx
}

// rowNormalizedRaw returns random native blocks for one projection plus their exact f32 decode.
func rowNormalizedRaw(rng *rand.Rand, f rowNormalizedFixture, in int) ([]byte, []float32) {
	nblk := in / qkK
	values := make([]float32, f.out*in)
	if f.q6 {
		bb := kindQ6K.blockBytes()
		raw := make([]byte, f.out*nblk*bb)
		rng.Read(raw)
		for b := 0; b < f.out*nblk; b++ {
			raw[b*bb+bb-1] = 0x2C | (raw[b*bb+bb-1] & 0x03)
			q6kDequantSuperBlock(values[b*qkK:(b+1)*qkK], raw[b*bb:(b+1)*bb])
		}
		return raw, values
	}
	raw := make([]byte, f.out*nblk*q4kBlockBytes)
	for b := 0; b < f.out*nblk; b++ {
		blk := raw[b*q4kBlockBytes : (b+1)*q4kBlockBytes]
		randQ4KBlockBounded(rng, blk, 2, 6)
		q4kDequantSuperBlock(values[b*qkK:(b+1)*qkK], blk)
	}
	return raw, values
}

// TestQwen35RowNormalizedNativeStoresMatchQ8 runs the hybrid forward twice over the SAME
// decoded weights: once with every row-permuted projection in the Q8 store (the historical
// loader route: dequant -> normalize -> Q8) and once with them in q4kw/kqw (the native-row
// route). The only difference is Q8 weight rounding, so logits agree within quant tolerance and
// argmax; a reader that ignored q4kw/kqw would panic ("q8 tensor not built"), and a wrong store
// lookup would diverge O(1).
func TestQwen35RowNormalizedNativeStoresMatchQ8(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	rng := rand.New(rand.NewSource(13567))
	H := cfg.HiddenSize
	type native struct {
		raw []byte
		f   rowNormalizedFixture
	}
	var natives []native
	for _, f := range rowNormalizedFixtures(cfg) {
		if !Qwen35RowNormalizedProjection(cfg, f.name) {
			t.Fatalf("%s is not classified as a row-normalized projection", f.name)
		}
		raw, values := rowNormalizedRaw(rng, f, H)
		m.q8w[f.name] = quantizeQ8(values, f.out, H) // baseline: the Q8 route over the same weights
		natives = append(natives, native{raw, f})
	}
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}
	follow := []int{67, 71, 73}
	// batched: Prefill (the batched resident-Q4_K hybrid lane) + Step decode.
	// loop: the per-token decode loop (tokenHiddenQ via sessionQ4KKernel) for every position.
	batched := func() [][]float32 {
		s := m.NewSession()
		s.Q4K = true
		out := [][]float32{append([]float32(nil), s.Prefill(prompt)...)}
		for _, id := range follow {
			out = append(out, append([]float32(nil), s.Step(id)...))
		}
		return out
	}
	loop := func() [][]float32 {
		s := m.NewSession()
		s.Q4K = true
		var h []float32
		for _, id := range prompt {
			h = s.tokenHiddenQ(id, s.Cache.Len())
		}
		out := [][]float32{append([]float32(nil), s.headResident(h)...)}
		for _, id := range follow {
			h = s.tokenHiddenQ(id, s.Cache.Len())
			out = append(out, append([]float32(nil), s.headResident(h)...))
		}
		return out
	}
	want := batched()

	if m.kqw == nil {
		m.kqw = map[string]*kQuantTensor{}
	}
	for _, n := range natives {
		delete(m.q8w, n.f.name)
		if n.f.q6 {
			m.kqw[n.f.name] = quantizeKQuantFromRaw(n.raw, n.f.out, H, kindQ6K)
		} else {
			m.q4kw[n.f.name] = quantizeQ4KFromRaw(n.raw, n.f.out, H)
		}
	}
	got, gotLoop := batched(), loop()
	for i := range got {
		// Self-consistency: the batched prefill/Step and the per-token loop read the native
		// stores identically (both dispatch q4kw/kqw before q8w).
		assertQuantLogitsClose(t, "native-row batched vs decode loop", gotLoop[i], got[i])
		// Against the Q8-served baseline the only difference is the Q8 rounding of these
		// weights. Measured at this seed: prefill cosine 0.99988, Steps 0.888/0.99994/0.99995 —
		// the first Step's input is ill-conditioned in this random synthetic model (the same
		// 0.888 appears between the two stores' token loops, while each store's batched and loop
		// paths agree to 1.0), so the cosine floor is asserted on the prefill call and argmax
		// on every call. A wrong row order measures cosine -0.04 here (mutation witness).
		t.Logf("call %d native-vs-Q8 cosine %.7f", i, cosine(want[i], got[i]))
		if argmax(want[i]) != argmax(got[i]) {
			t.Fatalf("call %d argmax native=%d Q8=%d", i, argmax(got[i]), argmax(want[i]))
		}
	}
	assertCosineAtLeast(t, "native-row vs Q8 prefill logits", want[0], got[0], 0.999)
}

// TestAddCanonicalRowNormalizedRefusals pins the narrow builder entry point's contract: it stores
// only Qwen3.5 row-permuted projections with an exact native payload, never outside that set,
// never for the exact Qwen3.8 Q8 runtime, and never over an existing representation.
func TestAddCanonicalRowNormalizedRefusals(t *testing.T) {
	cfg := qwen35HybridQ4KTestCfg()
	H := cfg.HiddenSize
	q4 := func(out int) []byte { return make([]byte, out*H/qkK*q4kBlockBytes) }
	b := NewQuantBuilder(cfg, cfg.TieWordEmbeddings)
	qkv := layerName(0, "linear_attn.in_proj_qkv.weight")
	_, _, _, _, _, _, convDim := cfg.linearAttnDims()
	if err := b.AddCanonicalRowNormalizedQ4K(layerName(0, "self_attn.qkv_proj.weight"), []int{convDim, H}, q4(convDim)); err != nil {
		t.Fatalf("source-chain qkv name refused: %v", err)
	}
	if !b.m.HasQ4K(qkv) {
		t.Fatal("qkv not stored under its resolved canonical name")
	}
	if err := b.AddCanonicalRowNormalizedQ4K(qkv, []int{convDim, H}, q4(convDim)); err == nil || !strings.Contains(err.Error(), "already") {
		t.Fatalf("duplicate store err = %v", err)
	}
	q6 := make([]byte, 4*H/qkK*kindQ6K.blockBytes())
	if err := b.AddCanonicalRowNormalizedQ6K(layerName(1, "linear_attn.in_proj_a.weight"), []int{4, H}, q6); err != nil || !b.m.HasKQuant(layerName(1, "linear_attn.in_proj_a.weight")) {
		t.Fatalf("Q6_K row-normalized store err=%v", err)
	}
	for _, bad := range []struct {
		name  string
		shape []int
		raw   []byte
	}{
		{layerName(0, "linear_attn.out_proj.weight"), []int{H, H}, q4(H)}, // column transform
		{layerName(0, "mlp.down_proj.weight"), []int{H, H}, q4(H)},        // identity: ResidentQ4KEligible's job
		{layerName(2, "linear_attn.in_proj_z.weight"), []int{256, H}, q4(255)},
		{"mtp.layers.0.self_attn.q_proj.weight", []int{512, H}, q4(512)},
	} {
		if err := b.AddCanonicalRowNormalizedQ4K(bad.name, bad.shape, bad.raw); err == nil {
			t.Errorf("%s shape %v accepted", bad.name, bad.shape)
		}
	}
	exact := cfg
	exact.NumLayers = 64
	exact.LayerTypes = make([]string, 64)
	for l := range exact.LayerTypes {
		exact.LayerTypes[l] = "linear_attention"
		if (l+1)%4 == 0 {
			exact.LayerTypes[l] = "full_attention"
		}
	}
	if err := NewQuantBuilder(exact, true).AddCanonicalRowNormalizedQ4K(qkv, []int{convDim, H}, q4(convDim)); err == nil {
		t.Error("exact Qwen3.8 runtime accepted a native in_proj_qkv")
	}
}
