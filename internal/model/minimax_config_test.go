package model

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1s
func TestMiniMaxM31PublishedConfigSerialization(t *testing.T) {
	const digest = "df6a3668e2de5b8dc0e883e3767495ad17af7e4d69fabe9683088be015fda9ab"
	if got := fmt.Sprintf("%x", sha256.Sum256([]byte(minimaxM31PublishedConfig))); got != digest {
		t.Fatalf("published config digest = %s, want %s", got, digest)
	}
	var cfg Config
	if err := json.Unmarshal([]byte(minimaxM31PublishedConfig), &cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.ModelType != "minimax_m3_vl" || cfg.HiddenSize != 6144 || cfg.NumLayers != 60 || !cfg.isMiniMaxSparseAttn() {
		t.Fatalf("published MiniMax geometry/family = %s/%d/%d", cfg.ModelType, cfg.HiddenSize, cfg.NumLayers)
	}
	if cfg.IndexNHeads != 4 || cfg.IndexHeadDim != 128 || cfg.IndexBlockSize != 128 || cfg.IndexTopKBlocks != 16 || cfg.IndexLocalBlocks != 1 {
		t.Fatalf("published sparse geometry = heads:%d dim:%d block:%d topk:%d local:%d", cfg.IndexNHeads, cfg.IndexHeadDim, cfg.IndexBlockSize, cfg.IndexTopKBlocks, cfg.IndexLocalBlocks)
	}
	for layer := range cfg.NumLayers {
		if !cfg.isMSALayer(layer) {
			t.Fatalf("published sparse layer %d was decoded as %q", layer, cfg.layerType(layer))
		}
		wantMoE := 1
		if layer < 3 {
			wantMoE = 0
		}
		if len(cfg.MoELayerSchedule) != cfg.NumLayers || cfg.MoELayerSchedule[layer] != wantMoE {
			t.Fatalf("published MoE schedule at layer %d = %v", layer, cfg.MoELayerSchedule)
		}
	}
	if cfg.MoELayerFreq != 0 || !cfg.NormGain1p || !cfg.QKNorm || cfg.rotaryDim() != 64 || cfg.RopeTheta != 10000000 {
		t.Fatal("published MiniMax norm/rope or schedule identity was lost")
	}
	roundtrip, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Config
	if err := json.Unmarshal(roundtrip, &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded.MoELayerSchedule, cfg.MoELayerSchedule) || !reflect.DeepEqual(decoded.LayerTypes, cfg.LayerTypes) || decoded.ModelType != cfg.ModelType {
		t.Fatal("normalized export lost MiniMax schedules or family")
	}
}

// fak-test:runtime fast est=1s
func TestMiniMaxConfigScalarFrequencyCompatibility(t *testing.T) {
	for _, family := range []string{"minimax_m3", "llama", "deepseek_v3"} {
		t.Run(family, func(t *testing.T) {
			input := fmt.Sprintf(`{"model_type":%q,"hidden_size":16,"num_hidden_layers":3,"num_attention_heads":2,"moe_layer_freq":2}`, family)
			var cfg Config
			if err := json.Unmarshal([]byte(input), &cfg); err != nil {
				t.Fatal(err)
			}
			if cfg.MoELayerFreq != 2 || cfg.MoELayerSchedule != nil {
				t.Fatalf("scalar frequency decoded as %d/%v", cfg.MoELayerFreq, cfg.MoELayerSchedule)
			}
			encoded, err := json.Marshal(cfg)
			if err != nil {
				t.Fatal(err)
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(encoded, &fields); err != nil {
				t.Fatal(err)
			}
			if string(fields["moe_layer_freq"]) != "2" {
				t.Fatalf("legacy scalar serialization changed to %s", fields["moe_layer_freq"])
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestMiniMaxConfigSparseScheduleAndRefusals(t *testing.T) {
	mutate := func(t *testing.T, change func(map[string]any)) []byte {
		t.Helper()
		var fields map[string]any
		if err := json.Unmarshal([]byte(minimaxM31PublishedConfig), &fields); err != nil {
			t.Fatal(err)
		}
		change(fields["text_config"].(map[string]any))
		encoded, err := json.Marshal(fields)
		if err != nil {
			t.Fatal(err)
		}
		return encoded
	}
	t.Run("wrapper norm overrides nested", func(t *testing.T) {
		var fields map[string]any
		if err := json.Unmarshal([]byte(minimaxM31PublishedConfig), &fields); err != nil {
			t.Fatal(err)
		}
		fields["norm_gain_1p"] = false
		encoded, err := json.Marshal(fields)
		if err != nil {
			t.Fatal(err)
		}
		var cfg Config
		if err := json.Unmarshal(encoded, &cfg); err != nil {
			t.Fatal(err)
		}
		if cfg.NormGain1p {
			t.Fatal("wrapper norm_gain_1p=false was lost to nested use_gemma_norm=true")
		}
	})
	t.Run("mixed sparse schedule", func(t *testing.T) {
		input := mutate(t, func(text map[string]any) {
			sparse := text["sparse_attention_config"].(map[string]any)
			sparse["sparse_attention_freq"].([]any)[1] = float64(0)
			text["moe_layer_freq"].([]any)[4] = float64(0)
			text["use_gemma_norm"] = false
		})
		var cfg Config
		if err := json.Unmarshal(input, &cfg); err != nil {
			t.Fatal(err)
		}
		if cfg.isMSALayer(1) || !cfg.isMSALayer(0) || cfg.MoELayerSchedule[4] != 0 || cfg.MoELayerSchedule[5] != 1 || cfg.NormGain1p {
			t.Fatal("independent attention/MoE schedules or explicit norm flag was lost")
		}
	})
	for _, tc := range []struct {
		name string
		edit func(map[string]any)
	}{
		{"short MoE schedule", func(text map[string]any) { text["moe_layer_freq"] = []int{0, 1} }},
		{"noninteger MoE schedule", func(text map[string]any) { text["moe_layer_freq"].([]any)[0] = 0.5 }},
		{"short sparse schedule", func(text map[string]any) {
			text["sparse_attention_config"].(map[string]any)["sparse_attention_freq"] = []int{1}
		}},
		{"zero block size", func(text map[string]any) { text["sparse_attention_config"].(map[string]any)["sparse_block_size"] = 0 }},
		{"negative topk", func(text map[string]any) { text["sparse_attention_config"].(map[string]any)["sparse_topk_blocks"] = -1 }},
		{"index smaller than rotary", func(text map[string]any) { text["sparse_attention_config"].(map[string]any)["sparse_index_dim"] = 32 }},
		{"unsupported score", func(text map[string]any) {
			text["sparse_attention_config"].(map[string]any)["sparse_score_type"] = "sum"
		}},
		{"unsupported initial block", func(text map[string]any) { text["sparse_attention_config"].(map[string]any)["sparse_init_block"] = 1 }},
		{"unsupported index value", func(text map[string]any) {
			text["sparse_attention_config"].(map[string]any)["sparse_disable_index_value"].([]any)[0] = float64(0)
		}},
		{"conflicting layer types", func(text map[string]any) { text["layer_types"] = []string{"full_attention"} }},
		{"conflicting flat geometry", func(text map[string]any) { text["index_block_size"] = 16 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var cfg Config
			if err := json.Unmarshal(mutate(t, tc.edit), &cfg); !errors.Is(err, ErrMiniMaxConfig) {
				t.Fatalf("malformed MiniMax metadata error = %v, want ErrMiniMaxConfig", err)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestMiniMaxPublishedMetadataKeepsSessionFences(t *testing.T) {
	var cfg Config
	if err := json.Unmarshal([]byte(minimaxM31PublishedConfig), &cfg); err != nil {
		t.Fatal(err)
	}
	for _, flag := range []string{"Quant", "Q4", "Q4K", "Metal", "MetalQ4K"} {
		t.Run(flag, func(t *testing.T) {
			session := &Session{M: &Model{Cfg: cfg}, Cache: NewKVCache(cfg)}
			switch flag {
			case "Quant":
				session.Quant = true
			case "Q4":
				session.Q4 = true
			case "Q4K":
				session.Q4K = true
			case "Metal":
				session.Metal = true
			case "MetalQ4K":
				session.MetalQ4K = true
			}
			defer func() {
				if recover() == nil {
					t.Fatal("published metadata admitted an unsupported MiniMax session mode")
				}
				if session.Cache.Len() != 0 {
					t.Fatal("unsupported MiniMax session appended tokens")
				}
			}()
			session.requireMiniMaxSession()
		})
	}
}

// Source data: MiniMaxAI/MiniMax-M3.1-Flash config.json at
// 54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e; model metadata, no remote code.

const minimaxM31PublishedConfig = `{
  "architectures": [
    "MiniMaxM3SparseForConditionalGeneration"
  ],
  "auto_map": {
    "AutoConfig": "configuration_minimax_m3_vl.MiniMaxM3VLConfig"
  },
  "model_type": "minimax_m3_vl",
  "text_config": {
    "hidden_size": 6144,
    "intermediate_size": 3072,
    "num_hidden_layers": 60,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 200064,
    "max_position_embeddings": 1048576,
    "rms_norm_eps": 1e-06,
    "use_gemma_norm": true,
    "attention_output_gate": false,
    "rope_theta": 10000000,
    "rotary_dim": 64,
    "partial_rotary_factor": 0.5,
    "hidden_act": "swigluoai",
    "use_qk_norm": true,
    "tie_word_embeddings": false,
    "dense_intermediate_size": 12288,
    "shared_intermediate_size": 3072,
    "num_local_experts": 128,
    "num_experts_per_tok": 4,
    "n_shared_experts": 1,
    "scoring_func": "sigmoid",
    "use_routing_bias": true,
    "moe_layer_freq": [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    "qk_norm_type": "per_head",
    "swiglu_alpha": 1.702,
    "swiglu_limit": 7.0,
    "routed_scaling_factor": 2.0,
    "sparse_attention_config": {
      "use_sparse_attention": true,
      "sparse_index_dim": 128,
      "sparse_num_index_heads": 4,
      "sparse_topk_blocks": 16,
      "sparse_block_size": 128,
      "sparse_score_type": "max",
      "sparse_init_block": 0,
      "sparse_local_block": 1,
      "sparse_attention_freq": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1
      ],
      "sparse_disable_index_value": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1
      ]
    },
    "architectures": [
      "MiniMaxM3SparseForCausalLM"
    ]
  },
  "vision_config": {
    "hidden_size": 1280,
    "num_attention_heads": 16,
    "num_hidden_layers": 32,
    "intermediate_size": 5120,
    "patch_size": 14,
    "image_size": 672,
    "projection_dim": 6144,
    "position_embedding_type": "rope",
    "rope_mode": "3d",
    "rope_theta": 10000.0,
    "attention_dropout": 0.0,
    "hidden_act": "gelu",
    "initializer_factor": 1.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-05,
    "model_type": "clip_vision_model",
    "num_channels": 3,
    "vocab_size": 32000,
    "img_token_compression_config": {
      "image_token_compression_method": "patch_merge",
      "spatial_merge_size": 2,
      "temporal_patch_size": 2
    },
    "vision_segment_max_frames": 4
  },
  "img_token_compression_config": {
    "image_token_compression_method": "patch_merge",
    "spatial_merge_size": 2,
    "temporal_patch_size": 2
  },
  "image_grid_pinpoints": "[(336, 336), (336, 672), (336, 1008), (336, 1344), (336, 1680), (336, 2016), (672, 336), (672, 672), (672, 1008), (672, 1344), (672, 1680), (672, 2016), (1008, 336), (1008, 672), (1008, 1008), (1008, 1344), (1008, 1680), (1008, 2016), (1344, 336), (1344, 672), (1344, 1008), (1344, 1344), (1344, 1680), (1344, 2016), (1680, 336), (1680, 672), (1680, 1008), (1680, 1344), (1680, 1680), (1680, 2016), (2016, 336), (2016, 672), (2016, 1008), (2016, 1344), (2016, 1680), (2016, 2016)]",
  "image_seq_length": 576,
  "image_token_index": 200025,
  "video_token_index": 200026,
  "multimodal_projector_bias": true,
  "num_reward_heads": 0,
  "process_image_mode": "dynamic_res",
  "projector_hidden_act": "gelu",
  "vision_feature_layer": -1,
  "vision_feature_select_strategy": "full",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.52.4",
  "projector_hidden_size": 6144,
  "quantization_config": {
    "quant_method": "mxfp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [
      1,
      32
    ],
    "ignored_layers": [
      "language_model.model.layers.10.block_sparse_moe.gate",
      "language_model.model.layers.11.block_sparse_moe.gate",
      "language_model.model.layers.12.block_sparse_moe.gate",
      "language_model.model.layers.13.block_sparse_moe.gate",
      "language_model.model.layers.14.block_sparse_moe.gate",
      "language_model.model.layers.15.block_sparse_moe.gate",
      "language_model.model.layers.16.block_sparse_moe.gate",
      "language_model.model.layers.17.block_sparse_moe.gate",
      "language_model.model.layers.18.block_sparse_moe.gate",
      "language_model.model.layers.19.block_sparse_moe.gate",
      "language_model.model.layers.20.block_sparse_moe.gate",
      "language_model.model.layers.21.block_sparse_moe.gate",
      "language_model.model.layers.22.block_sparse_moe.gate",
      "language_model.model.layers.23.block_sparse_moe.gate",
      "language_model.model.layers.24.block_sparse_moe.gate",
      "language_model.model.layers.25.block_sparse_moe.gate",
      "language_model.model.layers.26.block_sparse_moe.gate",
      "language_model.model.layers.27.block_sparse_moe.gate",
      "language_model.model.layers.28.block_sparse_moe.gate",
      "language_model.model.layers.29.block_sparse_moe.gate",
      "language_model.model.layers.3.block_sparse_moe.gate",
      "language_model.model.layers.30.block_sparse_moe.gate",
      "language_model.model.layers.31.block_sparse_moe.gate",
      "language_model.model.layers.32.block_sparse_moe.gate",
      "language_model.model.layers.33.block_sparse_moe.gate",
      "language_model.model.layers.34.block_sparse_moe.gate",
      "language_model.model.layers.35.block_sparse_moe.gate",
      "language_model.model.layers.36.block_sparse_moe.gate",
      "language_model.model.layers.37.block_sparse_moe.gate",
      "language_model.model.layers.38.block_sparse_moe.gate",
      "language_model.model.layers.39.block_sparse_moe.gate",
      "language_model.model.layers.4.block_sparse_moe.gate",
      "language_model.model.layers.40.block_sparse_moe.gate",
      "language_model.model.layers.41.block_sparse_moe.gate",
      "language_model.model.layers.42.block_sparse_moe.gate",
      "language_model.model.layers.43.block_sparse_moe.gate",
      "language_model.model.layers.44.block_sparse_moe.gate",
      "language_model.model.layers.45.block_sparse_moe.gate",
      "language_model.model.layers.46.block_sparse_moe.gate",
      "language_model.model.layers.47.block_sparse_moe.gate",
      "language_model.model.layers.48.block_sparse_moe.gate",
      "language_model.model.layers.49.block_sparse_moe.gate",
      "language_model.model.layers.5.block_sparse_moe.gate",
      "language_model.model.layers.50.block_sparse_moe.gate",
      "language_model.model.layers.51.block_sparse_moe.gate",
      "language_model.model.layers.52.block_sparse_moe.gate",
      "language_model.model.layers.53.block_sparse_moe.gate",
      "language_model.model.layers.54.block_sparse_moe.gate",
      "language_model.model.layers.55.block_sparse_moe.gate",
      "language_model.model.layers.56.block_sparse_moe.gate",
      "language_model.model.layers.57.block_sparse_moe.gate",
      "language_model.model.layers.58.block_sparse_moe.gate",
      "language_model.model.layers.59.block_sparse_moe.gate",
      "language_model.model.layers.6.block_sparse_moe.gate",
      "language_model.model.layers.7.block_sparse_moe.gate",
      "language_model.model.layers.8.block_sparse_moe.gate",
      "language_model.model.layers.9.block_sparse_moe.gate",
      "lm_head",
      "model.embed_tokens",
      "multi_modal_projector",
      "patch_merge_mlp",
      "vision_tower"
    ],
    "group_size": 16,
    "moe_quant_algo": "NVFP4",
    "moe_quant_format": "nvfp4-pack-quantized"
  }
}
`
