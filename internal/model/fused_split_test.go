package model

import (
	"testing"
)

// phiSplitCfg is a tiny Phi-shaped config: GQA (nH=4, nKV=2), head_dim 8, so the
// fused qkv_proj has nH*hd + 2*nKV*hd = 32 + 16 + 16 = 64 rows, and gate_up_proj
// has 2*I = 2*24 = 48 rows.
func phiSplitCfg() Config {
	return Config{
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          8,
		IntermediateSize: 24,
		VocabSize:        11,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       -1,
	}
}

// rampRows builds `rows`*`in` floats whose value encodes (row, col) so a
// mislaid byte range is detectable: v = base + row*1000 + col.
func rampRows(base float64, rows, in int) []float32 {
	out := make([]float32, rows*in)
	for r := 0; r < rows; r++ {
		for c := 0; c < in; c++ {
			out[r*in+c] = float32(base + float64(r)*1000.0 + float64(c))
		}
	}
	return out
}

// TestFusedSplitMatchesSeparate is the gate-(1) witness: a load-time split of a
// synthetic fused qkv_proj / gate_up_proj is Float32bits-equal to the same weights
// stored as separate q/k/v / gate/up tensors. The split is a pure contiguous
// byte-range cut, so it must be bit-identical — no arithmetic, no new numeric claim.
func TestFusedSplitMatchesSeparate(t *testing.T) {
	cfg := phiSplitCfg()
	nH, nKV, hd, H, I := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.HiddenSize, cfg.IntermediateSize
	qRows, kRows, vRows := nH*hd, nKV*hd, nKV*hd

	// Per-layer reference component weights (the "separately stored" truth).
	type layerRef struct {
		q, k, v, gate, up []float32
	}
	refs := make([]layerRef, cfg.NumLayers)

	// Build the FUSED model: one qkv_proj (rows = q++k++v) and one gate_up_proj
	// (rows = gate++up) per layer, plus the non-fused tensors the forward pass needs.
	fused := []NamedTensorF32{
		{Name: "model.embed_tokens.weight", Shape: []int{cfg.VocabSize, H}, Data: rampRows(0.01, cfg.VocabSize, H)},
	}
	for l := 0; l < cfg.NumLayers; l++ {
		pre := layerPrefix(l)
		q := rampRows(1.0+float64(l)*0.1, qRows, H)
		k := rampRows(2.0+float64(l)*0.1, kRows, H)
		v := rampRows(3.0+float64(l)*0.1, vRows, H)
		gate := rampRows(4.0+float64(l)*0.1, I, H)
		up := rampRows(5.0+float64(l)*0.1, I, H)
		refs[l] = layerRef{q, k, v, gate, up}

		qkv := concatF32(q, k, v)     // rows q++k++v, axis-0
		gateUp := concatF32(gate, up) // rows gate++up, axis-0
		fused = append(fused,
			NamedTensorF32{Name: pre + "input_layernorm.weight", Shape: []int{H}, Data: rampRows(0.1, 1, H)},
			NamedTensorF32{Name: pre + suffixQKVProj, Shape: []int{qRows + kRows + vRows, H}, Data: qkv},
			NamedTensorF32{Name: pre + "self_attn.o_proj.weight", Shape: []int{H, nH * hd}, Data: rampRows(0.2, H, nH*hd)},
			NamedTensorF32{Name: pre + "post_attention_layernorm.weight", Shape: []int{H}, Data: rampRows(0.3, 1, H)},
			NamedTensorF32{Name: pre + suffixGateUpProj, Shape: []int{2 * I, H}, Data: gateUp},
			NamedTensorF32{Name: pre + "mlp.down_proj.weight", Shape: []int{H, I}, Data: rampRows(0.4, H, I)},
		)
	}
	fused = append(fused, NamedTensorF32{Name: "model.norm.weight", Shape: []int{H}, Data: rampRows(0.5, 1, H)})

	m, err := NewFromF32Tensors(cfg, fused)
	if err != nil {
		t.Fatalf("NewFromF32Tensors(fused): %v", err)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		p := func(s string) string { return layerName(l, s) }
		// The fused entries must be GONE — the forward pass never names them.
		if m.has(p(suffixQKVProj)) {
			t.Fatalf("layer %d: fused qkv_proj still present after split", l)
		}
		if m.has(p(suffixGateUpProj)) {
			t.Fatalf("layer %d: fused gate_up_proj still present after split", l)
		}
		// The component tensors must be present and bit-identical to the reference.
		assertFloat32BitsEqual(t, "q_proj L"+itoa(l), refs[l].q, m.tensor(p(suffixQProj)))
		assertFloat32BitsEqual(t, "k_proj L"+itoa(l), refs[l].k, m.tensor(p(suffixKProj)))
		assertFloat32BitsEqual(t, "v_proj L"+itoa(l), refs[l].v, m.tensor(p(suffixVProj)))
		assertFloat32BitsEqual(t, "gate_proj L"+itoa(l), refs[l].gate, m.tensor(p(suffixGateProj)))
		assertFloat32BitsEqual(t, "up_proj L"+itoa(l), refs[l].up, m.tensor(p(suffixUpProj)))
	}
}

// TestFusedSplitForwardEqualsUnfused proves the split is invisible to the forward
// pass: a Phi-style fused checkpoint and a separately-stored checkpoint carrying the
// SAME weights produce Float32bits-identical prefill logits. This is the end-to-end
// form of "the core runs unchanged" — the split changed only the manifest, not bytes.
func TestFusedSplitForwardEqualsUnfused(t *testing.T) {
	cfg := phiSplitCfg()
	nH, nKV, hd, H, I := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.HiddenSize, cfg.IntermediateSize
	qRows, kRows, vRows := nH*hd, nKV*hd, nKV*hd

	var fused, separate []NamedTensorF32
	embed := NamedTensorF32{Name: "model.embed_tokens.weight", Shape: []int{cfg.VocabSize, H}, Data: rampRows(0.01, cfg.VocabSize, H)}
	fused = append(fused, embed)
	separate = append(separate, embed)
	for l := 0; l < cfg.NumLayers; l++ {
		pre := layerPrefix(l)
		q := rampRows(1.0+float64(l)*0.1, qRows, H)
		k := rampRows(2.0+float64(l)*0.1, kRows, H)
		v := rampRows(3.0+float64(l)*0.1, vRows, H)
		gate := rampRows(4.0+float64(l)*0.1, I, H)
		up := rampRows(5.0+float64(l)*0.1, I, H)
		inLN := NamedTensorF32{Name: pre + "input_layernorm.weight", Shape: []int{H}, Data: rampRows(0.1, 1, H)}
		oProj := NamedTensorF32{Name: pre + "self_attn.o_proj.weight", Shape: []int{H, nH * hd}, Data: rampRows(0.2, H, nH*hd)}
		postLN := NamedTensorF32{Name: pre + "post_attention_layernorm.weight", Shape: []int{H}, Data: rampRows(0.3, 1, H)}
		down := NamedTensorF32{Name: pre + "mlp.down_proj.weight", Shape: []int{H, I}, Data: rampRows(0.4, H, I)}

		fused = append(fused, inLN,
			NamedTensorF32{Name: pre + suffixQKVProj, Shape: []int{qRows + kRows + vRows, H}, Data: concatF32(q, k, v)},
			oProj, postLN,
			NamedTensorF32{Name: pre + suffixGateUpProj, Shape: []int{2 * I, H}, Data: concatF32(gate, up)},
			down)
		separate = append(separate, inLN,
			NamedTensorF32{Name: pre + suffixQProj, Shape: []int{qRows, H}, Data: q},
			NamedTensorF32{Name: pre + suffixKProj, Shape: []int{kRows, H}, Data: k},
			NamedTensorF32{Name: pre + suffixVProj, Shape: []int{vRows, H}, Data: v},
			oProj, postLN,
			NamedTensorF32{Name: pre + suffixGateProj, Shape: []int{I, H}, Data: gate},
			NamedTensorF32{Name: pre + suffixUpProj, Shape: []int{I, H}, Data: up},
			down)
	}
	norm := NamedTensorF32{Name: "model.norm.weight", Shape: []int{H}, Data: rampRows(0.5, 1, H)}
	fused = append(fused, norm)
	separate = append(separate, norm)

	mf, err := NewFromF32Tensors(cfg, fused)
	if err != nil {
		t.Fatalf("fused load: %v", err)
	}
	ms, err := NewFromF32Tensors(cfg, separate)
	if err != nil {
		t.Fatalf("separate load: %v", err)
	}
	prompt := []int{1, 4, 7, 2, 9}
	got := mf.NewSession().Prefill(prompt)
	want := ms.NewSession().Prefill(prompt)
	assertFloat32BitsEqual(t, "fused-vs-separate prefill logits", want, got)
}

// TestFusedSplitRejectsRowMismatch is the fail-closed guard: a fused tensor whose
// row count does not equal the config-implied q+k+v sum is rejected at load, not
// silently mis-cut.
func TestFusedSplitRejectsRowMismatch(t *testing.T) {
	cfg := phiSplitCfg()
	H := cfg.HiddenSize
	man := map[string]tensorMeta{
		// 5 rows where config implies nH*hd+2*nKV*hd = 64.
		layerName(0, suffixQKVProj): {Dtype: "f32", Shape: []int{5, H}, Offset: 0, Nbytes: 5 * H * 4},
	}
	err := splitFusedProjections(cfg, man)
	if err == nil {
		t.Fatalf("expected row-mismatch error, got nil")
	}
}

func concatF32(parts ...[]float32) []float32 {
	var out []float32
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}
