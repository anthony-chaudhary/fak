package ggufload

import "testing"

func qwen35MoEConfigMetadata(experts, topK, width uint64, norm *bool) map[string]Value {
	const p = "qwen35moe."
	meta := map[string]Value{
		"general.architecture":                 {Type: TypeString, Value: "qwen35moe"},
		p + "embedding_length":                 {Type: TypeUint64, Value: uint64(32)},
		p + "block_count":                      {Type: TypeUint64, Value: uint64(2)},
		p + "attention.head_count":             {Type: TypeUint64, Value: uint64(4)},
		p + "feed_forward_length":              {Type: TypeUint64, Value: uint64(64)},
		p + "attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
		p + "expert_count":                     {Type: TypeUint64, Value: experts},
		p + "expert_used_count":                {Type: TypeUint64, Value: topK},
		p + "expert_feed_forward_length":       {Type: TypeUint64, Value: width},
	}
	if norm != nil {
		meta[p+"expert_weights_norm"] = Value{Type: TypeBool, Value: *norm}
	}
	return meta
}

func TestQwen35MoEGGUFNormalization(t *testing.T) {
	no := false
	yes := true
	tests := []struct {
		name                 string
		experts, topK, width uint64
		norm                 *bool
		wantNorm             bool
	}{
		{name: "35B absent defaults true", experts: 256, topK: 8, width: 512, wantNorm: true},
		{name: "397B absent defaults true", experts: 512, topK: 10, width: 1024, wantNorm: true},
		{name: "explicit false stays false", experts: 256, topK: 8, width: 512, norm: &no},
		{name: "explicit true stays true", experts: 512, topK: 10, width: 1024, norm: &yes, wantNorm: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg, err := (&File{Metadata: qwen35MoEConfigMetadata(test.experts, test.topK, test.width, test.norm)}).Config()
			if err != nil {
				t.Fatalf("Config: %v", err)
			}
			if cfg.ModelType != "qwen35moe" || cfg.NumExperts != int(test.experts) || cfg.NumExpertsPerTok != int(test.topK) || cfg.MoEIntermediateSize != int(test.width) {
				t.Fatalf("axes = type:%q experts:%d top-k:%d width:%d, want qwen35moe/%d/%d/%d",
					cfg.ModelType, cfg.NumExperts, cfg.NumExpertsPerTok, cfg.MoEIntermediateSize,
					test.experts, test.topK, test.width)
			}
			if cfg.NormTopKProb != test.wantNorm {
				t.Fatalf("NormTopKProb = %t, want %t", cfg.NormTopKProb, test.wantNorm)
			}
		})
	}
}

func TestQwen35DenseGGUFDoesNotInheritMoENormalization(t *testing.T) {
	meta := qwen35MoEConfigMetadata(0, 0, 0, nil)
	meta["general.architecture"] = Value{Type: TypeString, Value: "qwen35"}
	for key, value := range meta {
		if len(key) > len("qwen35moe.") && key[:len("qwen35moe.")] == "qwen35moe." {
			delete(meta, key)
			meta["qwen35."+key[len("qwen35moe."):]] = value
		}
	}
	cfg, err := (&File{Metadata: meta}).Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.NormTopKProb {
		t.Fatal("dense qwen35 absent expert_weights_norm unexpectedly defaulted true")
	}
}
