package model

import "testing"

func TestTensorAliasesMaterializeCanonicalNames(t *testing.T) {
	cfg := Config{
		TensorAliases: map[string]string{
			"model.embed_tokens.weight": "source.embed.weight",
			"model.norm.weight":         "source.final_norm.weight",
		},
	}
	m, err := NewFromF32Tensors(cfg, []NamedTensorF32{
		{Name: "source.embed.weight", Shape: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}},
		{Name: "source.final_norm.weight", Shape: []int{3}, Data: []float32{7, 8, 9}},
	})
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}
	assertFloat32BitsEqual(t, "canonical embed alias", m.tensor("source.embed.weight"), m.tensor("model.embed_tokens.weight"))
	assertFloat32BitsEqual(t, "canonical norm alias", m.tensor("source.final_norm.weight"), m.tensor("model.norm.weight"))
}

func TestTensorAliasesFailClosedOnMissingSource(t *testing.T) {
	_, err := NewFromF32Tensors(Config{
		TensorAliases: map[string]string{
			"model.embed_tokens.weight": "missing.embed.weight",
		},
	}, []NamedTensorF32{
		{Name: "other.weight", Shape: []int{1}, Data: []float32{1}},
	})
	if err == nil {
		t.Fatal("missing tensor alias source accepted")
	}
}

func TestTensorAliasesFeedFusedProjectionSplit(t *testing.T) {
	cfg := Config{
		HiddenSize: 3,
		NumLayers:  1,
		NumHeads:   2,
		NumKVHeads: 1,
		HeadDim:    2,
		TensorAliases: map[string]string{
			"model.layers.0.self_attn.qkv_proj.weight": "source.layers.0.attn.packed_qkv.weight",
		},
	}
	fused := make([]float32, 8*3)
	for i := range fused {
		fused[i] = float32(i + 1)
	}
	m, err := NewFromF32Tensors(cfg, []NamedTensorF32{
		{Name: "source.layers.0.attn.packed_qkv.weight", Shape: []int{8, 3}, Data: fused},
	})
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}
	assertFloat32BitsEqual(t, "q alias split", fused[:12], m.tensor("model.layers.0.self_attn.q_proj.weight"))
	assertFloat32BitsEqual(t, "k alias split", fused[12:18], m.tensor("model.layers.0.self_attn.k_proj.weight"))
	assertFloat32BitsEqual(t, "v alias split", fused[18:24], m.tensor("model.layers.0.self_attn.v_proj.weight"))
}

func TestGPTNeoXInterleavedQKVMaterialization(t *testing.T) {
	cfg := Config{
		HiddenSize:       4,
		NumLayers:        1,
		NumHeads:         2,
		NumKVHeads:       2,
		HeadDim:          2,
		IntermediateSize: 8,
		VocabSize:        16,
		ModelType:        "gpt_neox",
	}
	qkv := rampRows(10, 12, cfg.HiddenSize)
	bias := []float32{100, 101, 200, 201, 300, 301, 110, 111, 210, 211, 310, 311}

	m, err := NewFromF32Tensors(cfg, []NamedTensorF32{
		{Name: "gpt_neox.layers.0.attention.query_key_value.weight", Shape: []int{12, 4}, Data: qkv},
		{Name: "gpt_neox.layers.0.attention.query_key_value.bias", Shape: []int{12}, Data: bias},
	})
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}

	row := func(r int) []float32 { return qkv[r*cfg.HiddenSize : (r+1)*cfg.HiddenSize] }
	assertFloat32BitsEqual(t, "gpt-neox q weight", concatF32(row(0), row(1), row(6), row(7)), m.tensor("model.layers.0.self_attn.q_proj.weight"))
	assertFloat32BitsEqual(t, "gpt-neox k weight", concatF32(row(2), row(3), row(8), row(9)), m.tensor("model.layers.0.self_attn.k_proj.weight"))
	assertFloat32BitsEqual(t, "gpt-neox v weight", concatF32(row(4), row(5), row(10), row(11)), m.tensor("model.layers.0.self_attn.v_proj.weight"))
	assertFloat32BitsEqual(t, "gpt-neox q bias", []float32{100, 101, 110, 111}, m.tensor("model.layers.0.self_attn.q_proj.bias"))
	assertFloat32BitsEqual(t, "gpt-neox k bias", []float32{200, 201, 210, 211}, m.tensor("model.layers.0.self_attn.k_proj.bias"))
	assertFloat32BitsEqual(t, "gpt-neox v bias", []float32{300, 301, 310, 311}, m.tensor("model.layers.0.self_attn.v_proj.bias"))
}

func TestFalconTensorMaterializationFeedsFusedSplit(t *testing.T) {
	cfg := Config{
		HiddenSize:       4,
		NumLayers:        1,
		NumHeads:         2,
		NumKVHeads:       1,
		HeadDim:          2,
		IntermediateSize: 8,
		VocabSize:        16,
		ModelType:        "falcon",
	}
	q := rampRows(1, 4, cfg.HiddenSize)
	k := rampRows(2, 2, cfg.HiddenSize)
	v := rampRows(3, 2, cfg.HiddenSize)
	qkv := concatF32(q, k, v)

	m, err := NewFromF32Tensors(cfg, []NamedTensorF32{
		{Name: "transformer.word_embeddings.weight", Shape: []int{16, 4}, Data: rampRows(0.1, 16, 4)},
		{Name: "transformer.ln_f.weight", Shape: []int{4}, Data: []float32{1, 2, 3, 4}},
		{Name: "transformer.ln_f.bias", Shape: []int{4}, Data: []float32{0.1, 0.2, 0.3, 0.4}},
		{Name: "transformer.h.0.input_layernorm.weight", Shape: []int{4}, Data: []float32{5, 6, 7, 8}},
		{Name: "transformer.h.0.input_layernorm.bias", Shape: []int{4}, Data: []float32{-0.1, -0.2, -0.3, -0.4}},
		{Name: "transformer.h.0.self_attention.query_key_value.weight", Shape: []int{8, 4}, Data: qkv},
		{Name: "transformer.h.0.self_attention.dense.weight", Shape: []int{4, 4}, Data: rampRows(4, 4, 4)},
		{Name: "transformer.h.0.mlp.dense_h_to_4h.weight", Shape: []int{8, 4}, Data: rampRows(5, 8, 4)},
		{Name: "transformer.h.0.mlp.dense_4h_to_h.weight", Shape: []int{4, 8}, Data: rampRows(6, 4, 8)},
	})
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}
	assertFloat32BitsEqual(t, "falcon q split", q, m.tensor("model.layers.0.self_attn.q_proj.weight"))
	assertFloat32BitsEqual(t, "falcon k split", k, m.tensor("model.layers.0.self_attn.k_proj.weight"))
	assertFloat32BitsEqual(t, "falcon v split", v, m.tensor("model.layers.0.self_attn.v_proj.weight"))
	assertFloat32BitsEqual(t, "falcon final norm bias", []float32{0.1, 0.2, 0.3, 0.4}, m.tensor("model.norm.bias"))
	assertFloat32BitsEqual(t, "falcon layernorm bias", []float32{-0.1, -0.2, -0.3, -0.4}, m.tensor("model.layers.0.input_layernorm.bias"))
	assertFloat32BitsEqual(t, "falcon dense h to 4h", rampRows(5, 8, 4), m.tensor("model.layers.0.mlp.gate_proj.weight"))
}

func TestMPTTensorMaterializationFeedsFusedSplit(t *testing.T) {
	cfg := Config{
		HiddenSize:       4,
		NumLayers:        1,
		NumHeads:         2,
		NumKVHeads:       2,
		HeadDim:          2,
		IntermediateSize: 8,
		VocabSize:        16,
		ModelType:        "mpt",
	}
	q := rampRows(1, 4, cfg.HiddenSize)
	k := rampRows(2, 4, cfg.HiddenSize)
	v := rampRows(3, 4, cfg.HiddenSize)
	qkv := concatF32(q, k, v)

	m, err := NewFromF32Tensors(cfg, []NamedTensorF32{
		{Name: "transformer.wte.weight", Shape: []int{16, 4}, Data: rampRows(0.1, 16, 4)},
		{Name: "transformer.norm_f.weight", Shape: []int{4}, Data: []float32{1, 2, 3, 4}},
		{Name: "transformer.blocks.0.norm_1.weight", Shape: []int{4}, Data: []float32{5, 6, 7, 8}},
		{Name: "transformer.blocks.0.norm_2.weight", Shape: []int{4}, Data: []float32{9, 10, 11, 12}},
		{Name: "transformer.blocks.0.attn.Wqkv.weight", Shape: []int{12, 4}, Data: qkv},
		{Name: "transformer.blocks.0.attn.out_proj.weight", Shape: []int{4, 4}, Data: rampRows(4, 4, 4)},
		{Name: "transformer.blocks.0.ffn.up_proj.weight", Shape: []int{8, 4}, Data: rampRows(5, 8, 4)},
		{Name: "transformer.blocks.0.ffn.down_proj.weight", Shape: []int{4, 8}, Data: rampRows(6, 4, 8)},
	})
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}
	assertFloat32BitsEqual(t, "mpt q split", q, m.tensor("model.layers.0.self_attn.q_proj.weight"))
	assertFloat32BitsEqual(t, "mpt k split", k, m.tensor("model.layers.0.self_attn.k_proj.weight"))
	assertFloat32BitsEqual(t, "mpt v split", v, m.tensor("model.layers.0.self_attn.v_proj.weight"))
	assertFloat32BitsEqual(t, "mpt norm 1", []float32{5, 6, 7, 8}, m.tensor("model.layers.0.input_layernorm.weight"))
	assertFloat32BitsEqual(t, "mpt norm 2", []float32{9, 10, 11, 12}, m.tensor("model.layers.0.post_attention_layernorm.weight"))
	assertFloat32BitsEqual(t, "mpt up proj", rampRows(5, 8, 4), m.tensor("model.layers.0.mlp.gate_proj.weight"))
}
