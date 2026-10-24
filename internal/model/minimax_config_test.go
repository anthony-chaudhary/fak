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
