package model

import "testing"

func TestBatchedMoEExpertsSplitToCanonicalViews(t *testing.T) {
	cfg := Config{
		HiddenSize:       3,
		IntermediateSize: 2,
		NumLayers:        1,
		NumExperts:       2,
		NumExpertsPerTok: 1,
	}
	gateUp := make([]float32, cfg.NumExperts*2*cfg.IntermediateSize*cfg.HiddenSize)
	for i := range gateUp {
		gateUp[i] = float32(i + 1)
	}
	down := make([]float32, cfg.NumExperts*cfg.HiddenSize*cfg.IntermediateSize)
	for i := range down {
		down[i] = float32(100 + i)
	}
	m, err := NewFromF32Tensors(cfg, []NamedTensorF32{
		{
			Name:  layerName(0, "mlp.experts.gate_up_proj"),
			Shape: []int{cfg.NumExperts, 2 * cfg.IntermediateSize, cfg.HiddenSize},
			Data:  gateUp,
		},
		{
			Name:  layerName(0, "mlp.experts.down_proj"),
			Shape: []int{cfg.NumExperts, cfg.HiddenSize, cfg.IntermediateSize},
			Data:  down,
		},
	})
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}
	I, H := cfg.IntermediateSize, cfg.HiddenSize
	for e := 0; e < cfg.NumExperts; e++ {
		base := e * 2 * I * H
		assertFloat32BitsEqual(t, "expert gate", gateUp[base:base+I*H], m.tensor(expertName(0, e, "gate_proj.weight")))
		assertFloat32BitsEqual(t, "expert up", gateUp[base+I*H:base+2*I*H], m.tensor(expertName(0, e, "up_proj.weight")))
		dbase := e * H * I
		assertFloat32BitsEqual(t, "expert down", down[dbase:dbase+H*I], m.tensor(expertName(0, e, "down_proj.weight")))
	}
	if m.has(layerName(0, "mlp.experts.gate_up_proj")) || m.has(layerName(0, "mlp.experts.down_proj")) {
		t.Fatal("batched MoE source tensors still present after split")
	}
}

func TestBatchedMoEExpertsRejectShapeMismatch(t *testing.T) {
	cfg := Config{
		HiddenSize:       3,
		IntermediateSize: 2,
		NumLayers:        1,
		NumExperts:       2,
		NumExpertsPerTok: 1,
	}
	_, err := NewFromF32Tensors(cfg, []NamedTensorF32{
		{
			Name:  layerName(0, "mlp.experts.gate_up_proj"),
			Shape: []int{cfg.NumExperts, cfg.IntermediateSize, cfg.HiddenSize},
			Data:  make([]float32, cfg.NumExperts*cfg.IntermediateSize*cfg.HiddenSize),
		},
	})
	if err == nil {
		t.Fatal("bad batched MoE gate/up shape accepted")
	}
}

func TestMixtralBlockSparseExpertsAliasCanonicalNames(t *testing.T) {
	cfg := Config{
		HiddenSize:       3,
		IntermediateSize: 2,
		NumLayers:        1,
		NumExperts:       2,
		NumExpertsPerTok: 1,
		ModelType:        "mixtral",
	}
	m, err := NewFromF32Tensors(cfg, []NamedTensorF32{
		{Name: layerName(0, "block_sparse_moe.gate.weight"), Shape: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}},
		{Name: layerName(0, "block_sparse_moe.experts.0.w1.weight"), Shape: []int{2, 3}, Data: []float32{10, 11, 12, 13, 14, 15}},
		{Name: layerName(0, "block_sparse_moe.experts.0.w2.weight"), Shape: []int{3, 2}, Data: []float32{20, 21, 22, 23, 24, 25}},
		{Name: layerName(0, "block_sparse_moe.experts.0.w3.weight"), Shape: []int{2, 3}, Data: []float32{30, 31, 32, 33, 34, 35}},
		{Name: layerName(0, "block_sparse_moe.experts.1.w1.weight"), Shape: []int{2, 3}, Data: []float32{40, 41, 42, 43, 44, 45}},
		{Name: layerName(0, "block_sparse_moe.experts.1.w2.weight"), Shape: []int{3, 2}, Data: []float32{50, 51, 52, 53, 54, 55}},
		{Name: layerName(0, "block_sparse_moe.experts.1.w3.weight"), Shape: []int{2, 3}, Data: []float32{60, 61, 62, 63, 64, 65}},
	})
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}
	assertFloat32BitsEqual(t, "mixtral router", []float32{1, 2, 3, 4, 5, 6}, m.tensor(routerName(0)))
	assertFloat32BitsEqual(t, "mixtral e0 gate", []float32{10, 11, 12, 13, 14, 15}, m.tensor(expertName(0, 0, "gate_proj.weight")))
	assertFloat32BitsEqual(t, "mixtral e0 down", []float32{20, 21, 22, 23, 24, 25}, m.tensor(expertName(0, 0, "down_proj.weight")))
	assertFloat32BitsEqual(t, "mixtral e0 up", []float32{30, 31, 32, 33, 34, 35}, m.tensor(expertName(0, 0, "up_proj.weight")))
	assertFloat32BitsEqual(t, "mixtral e1 gate", []float32{40, 41, 42, 43, 44, 45}, m.tensor(expertName(0, 1, "gate_proj.weight")))
	assertFloat32BitsEqual(t, "mixtral e1 down", []float32{50, 51, 52, 53, 54, 55}, m.tensor(expertName(0, 1, "down_proj.weight")))
	assertFloat32BitsEqual(t, "mixtral e1 up", []float32{60, 61, 62, 63, 64, 65}, m.tensor(expertName(0, 1, "up_proj.weight")))
}

func TestGPTOSSExpertMaterializationTransposesInterleavedWeights(t *testing.T) {
	cfg := Config{
		HiddenSize:       2,
		IntermediateSize: 3,
		NumLayers:        1,
		NumExperts:       2,
		NumExpertsPerTok: 1,
		ModelType:        "gpt_oss",
	}
	gateUp := []float32{
		// expert 0, hidden 0: gate0, up0, gate1, up1, gate2, up2
		1, 101, 2, 102, 3, 103,
		// expert 0, hidden 1
		4, 104, 5, 105, 6, 106,
		// expert 1, hidden 0
		11, 111, 12, 112, 13, 113,
		// expert 1, hidden 1
		14, 114, 15, 115, 16, 116,
	}
	gateUpBias := []float32{
		7, 107, 8, 108, 9, 109,
		17, 117, 18, 118, 19, 119,
	}
	down := []float32{
		// expert 0, intermediate rows [I,H]
		201, 202,
		203, 204,
		205, 206,
		// expert 1
		211, 212,
		213, 214,
		215, 216,
	}
	downBias := []float32{
		301, 302,
		311, 312,
	}
	m, err := NewFromF32Tensors(cfg, []NamedTensorF32{
		{Name: layerName(0, "mlp.router.weight"), Shape: []int{2, 2}, Data: []float32{1, 2, 3, 4}},
		{Name: layerName(0, "mlp.router.bias"), Shape: []int{2}, Data: []float32{5, 6}},
		{Name: layerName(0, "mlp.experts.gate_up_proj"), Shape: []int{2, 2, 6}, Data: gateUp},
		{Name: layerName(0, "mlp.experts.gate_up_proj_bias"), Shape: []int{2, 6}, Data: gateUpBias},
		{Name: layerName(0, "mlp.experts.down_proj"), Shape: []int{2, 3, 2}, Data: down},
		{Name: layerName(0, "mlp.experts.down_proj_bias"), Shape: []int{2, 2}, Data: downBias},
	})
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}
	assertFloat32BitsEqual(t, "gpt-oss router weight alias", []float32{1, 2, 3, 4}, m.tensor(routerName(0)))
	assertFloat32BitsEqual(t, "gpt-oss router bias alias", []float32{5, 6}, m.tensor(routerBiasName(0)))
	assertFloat32BitsEqual(t, "gpt-oss e0 gate", []float32{1, 4, 2, 5, 3, 6}, m.tensor(expertName(0, 0, "gate_proj.weight")))
	assertFloat32BitsEqual(t, "gpt-oss e0 up", []float32{101, 104, 102, 105, 103, 106}, m.tensor(expertName(0, 0, "up_proj.weight")))
	assertFloat32BitsEqual(t, "gpt-oss e1 gate", []float32{11, 14, 12, 15, 13, 16}, m.tensor(expertName(0, 1, "gate_proj.weight")))
	assertFloat32BitsEqual(t, "gpt-oss e1 up", []float32{111, 114, 112, 115, 113, 116}, m.tensor(expertName(0, 1, "up_proj.weight")))
	assertFloat32BitsEqual(t, "gpt-oss e0 gate bias", []float32{7, 8, 9}, m.tensor(expertName(0, 0, "gate_proj.bias")))
	assertFloat32BitsEqual(t, "gpt-oss e0 up bias", []float32{107, 108, 109}, m.tensor(expertName(0, 0, "up_proj.bias")))
	assertFloat32BitsEqual(t, "gpt-oss e0 down", []float32{201, 203, 205, 202, 204, 206}, m.tensor(expertName(0, 0, "down_proj.weight")))
	assertFloat32BitsEqual(t, "gpt-oss e1 down", []float32{211, 213, 215, 212, 214, 216}, m.tensor(expertName(0, 1, "down_proj.weight")))
	assertFloat32BitsEqual(t, "gpt-oss e1 down bias", []float32{311, 312}, m.tensor(expertName(0, 1, "down_proj.bias")))
	for _, name := range []string{
		layerName(0, "mlp.experts.gate_up_proj"),
		layerName(0, "mlp.experts.gate_up_proj_bias"),
		layerName(0, "mlp.experts.down_proj"),
		layerName(0, "mlp.experts.down_proj_bias"),
	} {
		if m.has(name) {
			t.Fatalf("gpt-oss source tensor %s still present after materialization", name)
		}
	}
}
