package model

import (
	"encoding/json"
	"fmt"
	"testing"
)

// kimiK31ConfigFixture builds a config.json shaped like the published Kimi K3.1
// multimodal wrapper: the top level carries architectures/model_type, the image
// placeholder id, eos and vision_config; text_config carries the LM axes, the
// K3.1 MoE spellings, and the nested linear_attn_config / dsa_config objects
// with 1-BASED layer membership lists. The lists are generated with the
// published rule (l%4==0 or the final layer -> DSA, else KDA) so the fixture
// tracks NumLayers instead of hand-copying 93 entries.
func kimiK31ConfigFixture(numLayers int) string {
	var kda, dsa []int
	for l := 1; l <= numLayers; l++ {
		if l%4 == 0 || l == numLayers {
			dsa = append(dsa, l)
		} else {
			kda = append(kda, l)
		}
	}
	// jsonInts renders a list as a real JSON array — %v would emit the
	// space-separated Go form, which is not valid JSON.
	jsonInts := func(list []int) string {
		b, err := json.Marshal(list)
		if err != nil {
			return "[]"
		}
		return string(b)
	}
	return fmt.Sprintf(`{
		"architectures": ["KimiK31ForConditionalGeneration"],
		"model_type": "kimi_k31",
		"media_placeholder_token_id": 163605,
		"eos_token_id": 163586,
		"vision_config": {"depth": 5},
		"text_config": {
			"model_type": "kimi_k31",
			"hidden_size": 7168,
			"num_hidden_layers": %d,
			"num_attention_heads": 96,
			"num_key_value_heads": 96,
			"intermediate_size": 33792,
			"vocab_size": 163840,
			"rms_norm_eps": 1e-05,
			"max_position_embeddings": 1048576,
			"tie_word_embeddings": false,
			"hidden_act": "situ",
			"activation_situ_beta": 4.0,
			"activation_situ_linear_beta": 25.0,
			"attn_res_block_size": 12,
			"mla_use_nope": true,
			"mla_use_output_gate": true,
			"q_lora_rank": 1536,
			"kv_lora_rank": 512,
			"qk_nope_head_dim": 128,
			"qk_rope_head_dim": 64,
			"v_head_dim": 128,
			"first_k_dense_replace": 1,
			"moe_layer_freq": 1,
			"moe_intermediate_size": 3072,
			"routed_expert_hidden_size": 3584,
			"latent_moe_use_norm": true,
			"moe_renormalize": true,
			"moe_router_activation_func": "sigmoid",
			"num_experts": 896,
			"num_experts_per_token": 16,
			"num_shared_experts": 2,
			"num_expert_group": 1,
			"topk_method": "noaux_tc",
			"num_nextn_predict_layers": 0,
			"dsa_config": {
				"dsa_layers": %v,
				"index_num_heads": 32,
				"index_head_dim": 128,
				"index_topk": 2048,
				"index_k_norm_type": "layernorm",
				"index_skip_topk_offset": 1,
				"index_topk_freq": 1
			},
			"linear_attn_config": {
				"num_heads": 96,
				"head_dim": 128,
				"short_conv_kernel_size": 4,
				"gate_lower_bound": -5.0,
				"use_full_rank_gate": true,
				"kda_layers": %v,
				"full_attn_layers": %v
			}
		}
	}`, numLayers, jsonInts(dsa), jsonInts(kda), jsonInts(dsa))
}

// TestConfigKimiK31PublishedAxes pins the published K3.1 axes end to end: the
// wrapper unwrap, the alt-spelling MoE folds, the nested KDA/DSA geometry, and
// the 0-based LayerTypes rebuild. The geometry asserts are the axes the L8
// loaders feed into KimiK31KDAParams (NumHeads 96, HeadDim 128, ConvKernel 4,
// HiddenSize 7168) and KimiK31DSAParams (NumHeads 96, QLoRA 1536, KVLoRA 512,
// QkNope 128, QkRope 64, VDim 128, Hidden 7168, IdxHeads 32, IdxDim 128,
// TopK 2048); weight-level validation stays with the loader leaves.
func TestConfigKimiK31PublishedAxes(t *testing.T) {
	var cfg Config
	if err := json.Unmarshal([]byte(kimiK31ConfigFixture(93)), &cfg); err != nil {
		t.Fatalf("unmarshal K3.1 config: %v", err)
	}
	if !cfg.IsKimiK31() {
		t.Fatalf("family key = %q, want IsKimiK31() true", cfg.archFamilyKey())
	}
	if cfg.HeadDim != 192 {
		t.Fatalf("HeadDim = %d, want 192 (MLA derive qk_nope 128 + qk_rope 64)", cfg.HeadDim)
	}
	if cfg.HiddenSize != 7168 || cfg.NumLayers != 93 || cfg.NumHeads != 96 || cfg.NumKVHeads != 96 ||
		cfg.IntermediateSize != 33792 || cfg.VocabSize != 163840 ||
		cfg.RMSNormEps != 1e-5 || cfg.MaxPositionEmbeddings != 1048576 || cfg.TieWordEmbeddings {
		t.Fatalf("base axes = hidden:%d layers:%d heads:%d kv:%d inter:%d vocab:%d eps:%v maxpos:%d tie:%v",
			cfg.HiddenSize, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.IntermediateSize,
			cfg.VocabSize, cfg.RMSNormEps, cfg.MaxPositionEmbeddings, cfg.TieWordEmbeddings)
	}
	if cfg.HiddenAct != "situ" || cfg.ActivationSituBeta != 4.0 || cfg.ActivationSituLinearBeta != 25.0 {
		t.Fatalf("situ activation = act:%q beta:%v linear_beta:%v, want situ/4.0/25.0",
			cfg.HiddenAct, cfg.ActivationSituBeta, cfg.ActivationSituLinearBeta)
	}
	if cfg.AttnResBlockSize != 12 || !cfg.MLAUseNope || !cfg.MLAUseOutputGate {
		t.Fatalf("MLA variant axes = res_block:%d use_nope:%v use_gate:%v, want 12/true/true",
			cfg.AttnResBlockSize, cfg.MLAUseNope, cfg.MLAUseOutputGate)
	}
	if cfg.QLoraRank != 1536 || cfg.KVLoraRank != 512 || cfg.QKNopeHeadDim != 128 ||
		cfg.QKRopeHeadDim != 64 || cfg.VHeadDim != 128 {
		t.Fatalf("MLA widths = q_lora:%d kv_lora:%d nope:%d rope:%d v:%d, want 1536/512/128/64/128",
			cfg.QLoraRank, cfg.KVLoraRank, cfg.QKNopeHeadDim, cfg.QKRopeHeadDim, cfg.VHeadDim)
	}
	if !cfg.IsMoE() || cfg.NumExperts != 896 || cfg.NumExpertsPerTok != 16 || cfg.NSharedExperts != 2 ||
		cfg.NGroup != 1 || !cfg.NormTopKProb || cfg.ScoringFunc != "sigmoid" ||
		cfg.MoEIntermediateSize != 3072 || cfg.RoutedExpertHiddenSize != 3584 ||
		!cfg.LatentMoEUseNorm || cfg.FirstKDenseReplace != 1 || cfg.MoELayerFreq != 1 ||
		cfg.TopKMethod != "noaux_tc" {
		t.Fatalf("MoE axes = IsMoE:%v experts:%d topk:%d shared:%d group:%d norm:%v scoring:%q "+
			"moe_inter:%d routed_hidden:%d latent_norm:%v first_dense:%d freq:%d method:%q",
			cfg.IsMoE(), cfg.NumExperts, cfg.NumExpertsPerTok, cfg.NSharedExperts, cfg.NGroup,
			cfg.NormTopKProb, cfg.ScoringFunc, cfg.MoEIntermediateSize, cfg.RoutedExpertHiddenSize,
			cfg.LatentMoEUseNorm, cfg.FirstKDenseReplace, cfg.MoELayerFreq, cfg.TopKMethod)
	}
	if cfg.IndexNHeads != 32 || cfg.IndexHeadDim != 128 || cfg.IndexTopK != 2048 ||
		cfg.IndexKNormType != "layernorm" || cfg.IndexSkipTopkOffset != 1 || cfg.IndexTopkFreq != 1 {
		t.Fatalf("DSA indexer = heads:%d dim:%d topk:%d knorm:%q skip_offset:%d freq:%d, "+
			"want 32/128/2048/layernorm/1/1",
			cfg.IndexNHeads, cfg.IndexHeadDim, cfg.IndexTopK, cfg.IndexKNormType,
			cfg.IndexSkipTopkOffset, cfg.IndexTopkFreq)
	}
	if cfg.LinearNumKeyHeads != 96 || cfg.LinearNumValueHeads != 96 || cfg.LinearKeyHeadDim != 128 ||
		cfg.LinearValueHeadDim != 128 || cfg.LinearConvKernelDim != 4 ||
		cfg.LinearGateLowerBound != -5.0 || !cfg.LinearUseFullRankGate {
		t.Fatalf("KDA linear axes = key_heads:%d value_heads:%d key_dim:%d value_dim:%d conv:%d "+
			"gate_lb:%v full_rank_gate:%v, want 96/96/128/128/4/-5.0/true",
			cfg.LinearNumKeyHeads, cfg.LinearNumValueHeads, cfg.LinearKeyHeadDim,
			cfg.LinearValueHeadDim, cfg.LinearConvKernelDim, cfg.LinearGateLowerBound,
			cfg.LinearUseFullRankGate)
	}
	// LayerTypes: 0-based rebuild of the 1-based lists — 69 kda + 24 dsa, and
	// ZERO "full_attention" tokens because dsa_layers wins the precedence over
	// the identical full_attn_layers list.
	counts := map[string]int{}
	for _, tok := range cfg.LayerTypes {
		counts[tok]++
	}
	if len(cfg.LayerTypes) != 93 || counts[KimiK31LayerTypeKDA] != 69 ||
		counts[KimiK31LayerTypeDSA] != 24 || counts["full_attention"] != 0 {
		t.Fatalf("LayerTypes = len:%d counts:%v, want len 93 with kda 69 / dsa 24 / full_attention 0",
			len(cfg.LayerTypes), counts)
	}
	for _, spot := range []struct {
		idx  int
		want string
	}{
		{0, KimiK31LayerTypeKDA}, {3, KimiK31LayerTypeDSA}, {90, KimiK31LayerTypeKDA},
		{91, KimiK31LayerTypeDSA}, {92, KimiK31LayerTypeDSA},
	} {
		if got := cfg.LayerTypes[spot.idx]; got != spot.want {
			t.Fatalf("LayerTypes[%d] = %q, want %q (0-based rebuild of the 1-based lists)",
				spot.idx, got, spot.want)
		}
	}
	if cfg.EOSTokenID != 163586 || cfg.ImageTokenID != 163605 || !cfg.HasVision {
		t.Fatalf("token/flag axes = eos:%d image:%d has_vision:%v, want 163586/163605/true",
			cfg.EOSTokenID, cfg.ImageTokenID, cfg.HasVision)
	}
	if cfg.HasMTPHead() {
		t.Fatalf("HasMTPHead = true, want false (num_nextn_predict_layers 0)")
	}
	if cfg.IsQwen35Hybrid() {
		t.Fatalf("IsQwen35Hybrid = true, want false (no linear_attention tokens)")
	}
}

// TestConfigKimiK31FamilyKeyBothSpellings proves the family key matches both
// published spellings and rejects the flat K3 ("kimi_linear") family.
func TestConfigKimiK31FamilyKeyBothSpellings(t *testing.T) {
	cases := []struct {
		name, modelType, arch string
		want                  bool
	}{
		{"model_type only", "kimi_k31", "", true},
		{"architectures only", "", "KimiK31ForConditionalGeneration", true},
		{"flat kimi_linear (K3) does not match", "kimi_linear", "KimiLinearForCausalLM", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			js := fmt.Sprintf(`{"model_type":%q,"architectures":[%q]}`, tc.modelType, tc.arch)
			var cfg Config
			if err := json.Unmarshal([]byte(js), &cfg); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got := cfg.IsKimiK31(); got != tc.want {
				t.Fatalf("IsKimiK31(model_type=%q arch=%q) = %v, want %v (family key %q)",
					tc.modelType, tc.arch, got, tc.want, cfg.archFamilyKey())
			}
		})
	}
}

// TestConfigKimiK31NormalizeGuards pins the no-op contract of the LayerTypes
// rebuild — the guard that keeps GLM-5.3-Flash (which ships the same nested
// lists 0-BASED together with flat layer_types) byte-identical.
func TestConfigKimiK31NormalizeGuards(t *testing.T) {
	t.Run("flat layer_types wins over the nested lists", func(t *testing.T) {
		var cfg Config
		if err := json.Unmarshal([]byte(`{
			"model_type": "kimi_k31",
			"num_hidden_layers": 2,
			"layer_types": ["linear_attention", "full_attention"],
			"linear_attn_config": {"kda_layers": [1], "full_attn_layers": [2]}
		}`), &cfg); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if len(cfg.LayerTypes) != 2 || cfg.LayerTypes[0] != "linear_attention" {
			t.Fatalf("LayerTypes = %v, want the flat list untouched", cfg.LayerTypes)
		}
	})
	t.Run("wrapper without vision_config clears HasVision", func(t *testing.T) {
		var cfg Config
		if err := json.Unmarshal([]byte(`{
			"model_type": "kimi_k31",
			"num_hidden_layers": 2,
			"vision_config": null,
			"text_config": {"num_hidden_layers": 2}
		}`), &cfg); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if cfg.HasVision {
			t.Fatal("HasVision = true, want false (vision_config null counts as absent)")
		}
	})
	t.Run("no membership lists leaves LayerTypes empty", func(t *testing.T) {
		var cfg Config
		if err := json.Unmarshal([]byte(`{
			"model_type": "kimi_k31",
			"num_hidden_layers": 2,
			"linear_attn_config": {"num_heads": 96, "head_dim": 128}
		}`), &cfg); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if len(cfg.LayerTypes) != 0 {
			t.Fatalf("LayerTypes = %v, want empty (no membership lists)", cfg.LayerTypes)
		}
	})
}
