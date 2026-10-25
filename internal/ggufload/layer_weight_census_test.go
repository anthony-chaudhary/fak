package ggufload

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestCensusLayerWeightsGroupsTensors(t *testing.T) {
	f := &File{Tensors: []TensorInfo{
		{Name: "token_embd.weight", Type: TensorQ4_K, Dims: []uint64{256, 10}},  // 10 x 144 B
		{Name: "blk.0.attn_q.weight", Type: TensorQ4_K, Dims: []uint64{256, 2}}, // 288 B
		{Name: "blk.0.attn_norm.weight", Type: TensorF32, Dims: []uint64{8}},    // 32 B
		{Name: "blk.1.ffn_up.weight", Type: TensorQ6_K, Dims: []uint64{256, 1}}, // 210 B
		{Name: "blk.2.nextn.eh_proj.weight", Type: TensorF32, Dims: []uint64{4}},
		{Name: "output_norm.weight", Type: TensorF32, Dims: []uint64{8}},
		{Name: "output.weight", Type: TensorQ6_K, Dims: []uint64{256, 3}},
	}}
	c, err := CensusLayerWeights(f, 2)
	if err != nil {
		t.Fatal(err)
	}
	if c.Layers[0] != 320 || c.Layers[1] != 210 {
		t.Fatalf("layers = %v, want [320 210]", c.Layers)
	}
	if c.EmbedBytes != 1440 || c.HeadBytes != 630+32 || c.TiedHead || c.OtherBytes != 16 {
		t.Fatalf("census = %+v", c)
	}

	f.Tensors = f.Tensors[:len(f.Tensors)-1] // drop output.weight: the head is tied
	c, err = CensusLayerWeights(f, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !c.TiedHead || c.HeadBytes != 1440+32 {
		t.Fatalf("tied census = %+v", c)
	}
	if _, err := CensusLayerWeights(f, 0); err == nil {
		t.Fatal("numLayers 0 accepted")
	}
}
