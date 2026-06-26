package model

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"
)

func TestConfigUnmarshalEOSTokenIDScalarOrList(t *testing.T) {
	base := `{
		"hidden_size": 8,
		"num_hidden_layers": 1,
		"num_attention_heads": 2,
		"num_key_value_heads": 1,
		"head_dim": 4,
		"intermediate_size": 16,
		"vocab_size": 32,
		"rms_norm_eps": 0.00001,
		"rope_theta": 10000,
		"tie_word_embeddings": true,
		"attention_bias": false,
		"model_type": "synthetic",
		"eos_token_id": %s
	}`
	tests := []struct {
		name     string
		eosJSON  string
		wantID   int
		wantList []int
		hits     []int
		misses   []int
	}{
		{name: "scalar", eosJSON: `2`, wantID: 2, wantList: []int{2}, hits: []int{2}, misses: []int{3}},
		{name: "list", eosJSON: `[2, 128001, 128008]`, wantID: 2, wantList: []int{2, 128001, 128008}, hits: []int{2, 128001, 128008}, misses: []int{128009}},
		{name: "null", eosJSON: `null`, wantID: 0, hits: []int{0}, misses: []int{1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cfg Config
			if err := json.Unmarshal([]byte(fmt.Sprintf(base, tt.eosJSON)), &cfg); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if cfg.EOSTokenID != tt.wantID {
				t.Fatalf("EOSTokenID=%d want %d", cfg.EOSTokenID, tt.wantID)
			}
			if len(cfg.EOSTokenIDs) != len(tt.wantList) {
				t.Fatalf("EOSTokenIDs=%v want %v", cfg.EOSTokenIDs, tt.wantList)
			}
			for i := range tt.wantList {
				if cfg.EOSTokenIDs[i] != tt.wantList[i] {
					t.Fatalf("EOSTokenIDs=%v want %v", cfg.EOSTokenIDs, tt.wantList)
				}
			}
			for _, id := range tt.hits {
				if !cfg.IsEOS(id) {
					t.Fatalf("IsEOS(%d)=false, want true", id)
				}
			}
			for _, id := range tt.misses {
				if cfg.IsEOS(id) {
					t.Fatalf("IsEOS(%d)=true, want false", id)
				}
			}
		})
	}
}

func TestConfigUnmarshalRejectsBadEOSTokenID(t *testing.T) {
	var cfg Config
	err := json.Unmarshal([]byte(`{"eos_token_id": "bad"}`), &cfg)
	if err == nil {
		t.Fatal("bad eos_token_id accepted")
	}
}

func TestConfigDerivesArchitectureAxesFromMetadata(t *testing.T) {
	decode := func(t *testing.T, js string) Config {
		t.Helper()
		var cfg Config
		if err := json.Unmarshal([]byte(js), &cfg); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		return cfg
	}

	gemma := decode(t, `{
		"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
		"intermediate_size": 128, "vocab_size": 32, "rms_norm_eps": 1e-6,
		"rope_theta": 10000, "model_type": "gemma2",
		"architectures": ["Gemma2ForCausalLM"],
		"hidden_activation": "gelu_pytorch_tanh"
	}`)
	if gemma.HeadDim != 16 || gemma.NumKVHeads != 4 {
		t.Fatalf("derived head/KV dims = %d/%d, want 16/4", gemma.HeadDim, gemma.NumKVHeads)
	}
	if gemma.BlockTopology != SandwichNorm {
		t.Fatalf("gemma topology = %v, want SandwichNorm", gemma.BlockTopology)
	}
	if !gemma.NormGain1p || !gemma.ActGeluTanh {
		t.Fatalf("gemma norm/activation flags = %v/%v, want true/true", gemma.NormGain1p, gemma.ActGeluTanh)
	}
	if gemma.EmbedScale != math.Sqrt(64) {
		t.Fatalf("gemma embed scale = %v, want sqrt(hidden)", gemma.EmbedScale)
	}

	olmo := decode(t, `{
		"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
		"num_key_value_heads": 4, "head_dim": 16, "intermediate_size": 128,
		"vocab_size": 32, "rms_norm_eps": 1e-6, "rope_theta": 10000,
		"model_type": "olmo2"
	}`)
	if olmo.BlockTopology != PostNorm || !olmo.QKNorm {
		t.Fatalf("olmo2 topology/qknorm = %v/%v, want PostNorm/true", olmo.BlockTopology, olmo.QKNorm)
	}

	neox := decode(t, `{
		"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
		"intermediate_size": 128, "vocab_size": 32, "rms_norm_eps": 1e-6,
		"rope_theta": 10000, "architectures": ["GPTNeoXForCausalLM"],
		"hidden_act": "gelu",
		"rope_parameters": {
			"rope_theta": 10000, "rope_type": "default", "partial_rotary_factor": 0.25
		}
	}`)
	if neox.BlockTopology != ParallelResidual || !neox.LayerNorm || !neox.DenseMLP || !neox.ActGeluErf || neox.PartialRotaryFactor != 0.25 {
		t.Fatalf("gpt-neox axes = topology:%v layernorm:%v dense:%v gelu:%v partial:%v; want ParallelResidual/LayerNorm/DenseMLP/GELU/0.25",
			neox.BlockTopology, neox.LayerNorm, neox.DenseMLP, neox.ActGeluErf, neox.PartialRotaryFactor)
	}

	cohere := decode(t, `{
		"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
		"intermediate_size": 128, "vocab_size": 32, "rms_norm_eps": 1e-6,
		"rope_theta": 10000, "architectures": ["CohereForCausalLM"],
		"hidden_act": "gelu"
	}`)
	if cohere.BlockTopology != ParallelResidual || cohere.LogitScale != 0.0625 || !cohere.LayerNorm || !cohere.ActGeluErf {
		t.Fatalf("cohere topology/logit/norm/act = %v/%v/%v/%v, want ParallelResidual/0.0625/LayerNorm/GELU",
			cohere.BlockTopology, cohere.LogitScale, cohere.LayerNorm, cohere.ActGeluErf)
	}

	falcon := decode(t, `{
		"hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2,
		"num_key_value_heads": 1, "head_dim": 4, "intermediate_size": 32,
		"vocab_size": 32, "rms_norm_eps": 1e-5, "rope_theta": 10000,
		"model_type": "falcon", "architectures": ["FalconForCausalLM"],
		"hidden_act": "gelu", "parallel_attn": true
	}`)
	if falcon.BlockTopology != ParallelResidual || !falcon.LayerNorm || !falcon.DenseMLP || !falcon.ActGeluErf {
		t.Fatalf("falcon axes = topology:%v layernorm:%v dense:%v gelu:%v; want ParallelResidual/LayerNorm/DenseMLP/GELU",
			falcon.BlockTopology, falcon.LayerNorm, falcon.DenseMLP, falcon.ActGeluErf)
	}

	rawFalcon := decode(t, `{
		"hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2,
		"vocab_size": 65024, "layer_norm_epsilon": 1e-5,
		"model_type": "falcon", "architectures": ["FalconForCausalLM"],
		"multi_query": true, "parallel_attn": true
	}`)
	if rawFalcon.NumKVHeads != 1 || rawFalcon.IntermediateSize != 32 || rawFalcon.RMSNormEps != 1e-5 {
		t.Fatalf("raw falcon aliases = kv:%d intermediate:%d eps:%v, want 1/32/1e-5",
			rawFalcon.NumKVHeads, rawFalcon.IntermediateSize, rawFalcon.RMSNormEps)
	}

	mpt := decode(t, `{
		"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
		"num_key_value_heads": 4, "head_dim": 8, "intermediate_size": 128,
		"vocab_size": 32, "rms_norm_eps": 1e-5, "rope_theta": 10000,
		"model_type": "mpt", "architectures": ["MptForCausalLM"],
		"alibi": true, "alibi_bias_max": 8
	}`)
	if !mpt.LayerNorm || !mpt.DenseMLP || !mpt.ActGeluErf || !mpt.Alibi {
		t.Fatalf("mpt axes = layernorm:%v dense:%v gelu:%v alibi:%v; want all true",
			mpt.LayerNorm, mpt.DenseMLP, mpt.ActGeluErf, mpt.Alibi)
	}

	stable := decode(t, `{
		"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
		"num_key_value_heads": 4, "head_dim": 16, "intermediate_size": 37,
		"vocab_size": 32, "rms_norm_eps": 1e-5, "rope_theta": 10000,
		"model_type": "stablelm", "architectures": ["StableLmForCausalLM"],
		"hidden_act": "gelu",
		"rope_parameters": {
			"rope_theta": 10000, "rope_type": "default", "partial_rotary_factor": 0.25
		}
	}`)
	if !stable.LayerNorm || !stable.ActGeluErf || stable.PartialRotaryFactor != 0.25 {
		t.Fatalf("stablelm axes = layernorm:%v gelu:%v partial:%v; want LayerNorm/GELU/0.25",
			stable.LayerNorm, stable.ActGeluErf, stable.PartialRotaryFactor)
	}

	gptoss := decode(t, `{
		"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
		"num_key_value_heads": 1, "head_dim": 32, "intermediate_size": 64,
		"vocab_size": 201088, "rms_norm_eps": 1e-5, "rope_theta": 150000,
		"model_type": "gpt_oss", "architectures": ["GptOssForCausalLM"],
		"layer_types": ["sliding_attention", "full_attention"],
		"sliding_window": 128, "num_local_experts": 32, "num_experts_per_tok": 4,
		"rope_parameters": {
			"rope_type": "yarn", "factor": 32, "beta_fast": 32, "beta_slow": 1,
			"original_max_position_embeddings": 4096, "truncate": false,
			"rope_theta": 150000
		}
	}`)
	if !gptoss.IsMoE() || gptoss.NumExperts != 32 || gptoss.NumExpertsPerTok != 4 {
		t.Fatalf("gpt-oss moe fields = experts:%d topk:%d IsMoE:%v; want 32/4/true",
			gptoss.NumExperts, gptoss.NumExpertsPerTok, gptoss.IsMoE())
	}
	if gptoss.RopeScaling != "yarn" || gptoss.RopeFactor != 32 || gptoss.RopeOrigContext != 4096 {
		t.Fatalf("gpt-oss yarn fields = scaling:%q factor:%v orig:%d; want yarn/32/4096",
			gptoss.RopeScaling, gptoss.RopeFactor, gptoss.RopeOrigContext)
	}
	if len(gptoss.Window) != 2 || gptoss.Window[0] != 128 || gptoss.Window[1] != -1 {
		t.Fatalf("gpt-oss layer windows = %v, want [128 -1]", gptoss.Window)
	}

	llama3 := decode(t, `{
		"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
		"intermediate_size": 128, "vocab_size": 32, "rms_norm_eps": 1e-6,
		"rope_theta": 500000, "model_type": "llama",
		"rope_scaling": {
			"rope_type": "llama3", "factor": 8,
			"low_freq_factor": 1, "high_freq_factor": 4,
			"original_max_position_embeddings": 8192
		}
	}`)
	if llama3.RopeScaling != "llama3" || llama3.RopeFactor != 8 || llama3.RopeOrigContext != 8192 {
		t.Fatalf("nested llama3 rope scaling not flattened: %+v", llama3)
	}

	swa := decode(t, `{
		"hidden_size": 64, "num_hidden_layers": 3, "num_attention_heads": 4,
		"intermediate_size": 128, "vocab_size": 32, "rms_norm_eps": 1e-6,
		"rope_theta": 10000, "model_type": "mistral", "sliding_window": 5
	}`)
	if len(swa.Window) != 3 || swa.Window[0] != 5 || swa.Window[1] != 5 || swa.Window[2] != 5 {
		t.Fatalf("scalar sliding_window expanded to %v, want [5 5 5]", swa.Window)
	}

	qkn := decode(t, `{
		"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
		"intermediate_size": 128, "vocab_size": 32, "rms_norm_eps": 1e-6,
		"rope_theta": 10000, "model_type": "custom", "use_qk_norm": true
	}`)
	if !qkn.QKNorm {
		t.Fatal("use_qk_norm=true did not derive QKNorm")
	}

	// GLM-5.2 (zai-org, model_type "glm_moe_dsa"): a MoE family. The loader derives
	// the glm family and the dsa variant from model_type, and the generic MoE config
	// block populates NumExperts/NumExpertsPerTok. DSA sparse-attention forward is
	// research-grade; this case pins the family, MoE, and indexer metadata the
	// loader/oracle boundary keys off.
	glm := decode(t, `{
		"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
		"num_key_value_heads": 2, "head_dim": 8, "intermediate_size": 64,
		"vocab_size": 97, "rms_norm_eps": 1e-5, "rope_theta": 10000,
		"model_type": "glm_moe_dsa", "architectures": ["GlmMoeDsaForCausalLM"],
		"num_local_experts": 4, "num_experts_per_tok": 2, "norm_topk_prob": true,
		"n_group": 2, "topk_group": 1, "routed_scaling_factor": 2.5,
		"index_n_heads": 4, "index_head_dim": 16, "index_topk": 8,
		"indexer_types": ["full", "shared"]
	}`)
	if !glm.isGLM() || !glm.isGLMMoeDsa() {
		t.Fatalf("glm_moe_dsa family = %q, want glm+dsa (isGLM=%v isGLMMoeDsa=%v)",
			glm.archFamilyKey(), glm.isGLM(), glm.isGLMMoeDsa())
	}
	if !glm.IsMoE() || glm.NumExperts != 4 || glm.NumExpertsPerTok != 2 {
		t.Fatalf("glm MoE fields = experts:%d topk:%d IsMoE:%v; want 4/2/true",
			glm.NumExperts, glm.NumExpertsPerTok, glm.IsMoE())
	}
	if glm.IndexNHeads != 4 || glm.IndexHeadDim != 16 || glm.IndexTopK != 8 ||
		len(glm.IndexerTypes) != 2 || glm.IndexerTypes[1] != "shared" {
		t.Fatalf("glm DSA indexer fields = heads:%d dim:%d topk:%d types:%v, want 4/16/8 [full shared]",
			glm.IndexNHeads, glm.IndexHeadDim, glm.IndexTopK, glm.IndexerTypes)
	}
	if glm.NGroup != 2 || glm.TopKGroup != 1 || glm.RoutedScalingFactor != 2.5 {
		t.Fatalf("glm MoE routing fields = n_group:%d topk_group:%d scale:%v, want 2/1/2.5",
			glm.NGroup, glm.TopKGroup, glm.RoutedScalingFactor)
	}
}

func TestConfigDerivationHonorsExplicitOverrides(t *testing.T) {
	var cfg Config
	err := json.Unmarshal([]byte(`{
		"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
		"intermediate_size": 128, "vocab_size": 32, "rms_norm_eps": 1e-6,
		"rope_theta": 10000, "model_type": "gemma2",
		"block_topology": "pre_norm",
		"norm_gain_1p": false,
		"act_gelu_tanh": false,
		"embed_scale": 1
	}`), &cfg)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if cfg.BlockTopology != PreNorm || cfg.NormGain1p || cfg.ActGeluTanh || cfg.EmbedScale != 1 {
		t.Fatalf("explicit overrides not honored: topology=%v norm=%v act=%v embed=%v",
			cfg.BlockTopology, cfg.NormGain1p, cfg.ActGeluTanh, cfg.EmbedScale)
	}

	var bad Config
	if err := json.Unmarshal([]byte(`{"block_topology": "inside_out"}`), &bad); err == nil {
		t.Fatal("unknown block_topology accepted")
	}
}

func TestConfigDerivesGemma3LayerAttentionAxes(t *testing.T) {
	var cfg Config
	err := json.Unmarshal([]byte(`{
		"hidden_size": 64, "num_hidden_layers": 6, "num_attention_heads": 4,
		"intermediate_size": 128, "vocab_size": 32, "rms_norm_eps": 1e-6,
		"rope_theta": 1000000, "model_type": "gemma3",
		"architectures": ["Gemma3ForCausalLM"],
		"sliding_window": 5,
		"sliding_window_pattern": 3,
		"rope_parameters": {
			"sliding_attention": {"rope_type": "default", "rope_theta": 10000},
			"full_attention": {"rope_type": "default", "rope_theta": 1000000}
		}
	}`), &cfg)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	wantTypes := []string{"sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"}
	if len(cfg.LayerTypes) != len(wantTypes) {
		t.Fatalf("LayerTypes len=%d want %d: %v", len(cfg.LayerTypes), len(wantTypes), cfg.LayerTypes)
	}
	for i := range wantTypes {
		if cfg.LayerTypes[i] != wantTypes[i] {
			t.Fatalf("LayerTypes[%d]=%q want %q (%v)", i, cfg.LayerTypes[i], wantTypes[i], cfg.LayerTypes)
		}
	}
	wantWindow := []int{5, 5, -1, 5, 5, -1}
	for i := range wantWindow {
		if cfg.Window[i] != wantWindow[i] {
			t.Fatalf("Window[%d]=%d want %d (%v)", i, cfg.Window[i], wantWindow[i], cfg.Window)
		}
	}
	wantTheta := []float64{10000, 10000, 1000000, 10000, 10000, 1000000}
	for i := range wantTheta {
		if cfg.RopeThetaPerLayer[i] != wantTheta[i] {
			t.Fatalf("RopeThetaPerLayer[%d]=%v want %v (%v)", i, cfg.RopeThetaPerLayer[i], wantTheta[i], cfg.RopeThetaPerLayer)
		}
	}
}

func TestConfigAcceptsFlatRopeParameters(t *testing.T) {
	var cfg Config
	err := json.Unmarshal([]byte(`{
		"hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 4,
		"intermediate_size": 16, "vocab_size": 32, "rms_norm_eps": 1e-6,
		"model_type": "qwen2",
		"rope_parameters": {"rope_theta": 1000000, "rope_type": "default"}
	}`), &cfg)
	if err != nil {
		t.Fatalf("unmarshal flat rope_parameters: %v", err)
	}
	if cfg.RopeParameters["default"].RopeTheta != 1000000 {
		t.Fatalf("flat rope_parameters decoded as %+v, want default rope theta", cfg.RopeParameters)
	}
}

func TestConfigDerivesQwen25ProductionCheckpoints(t *testing.T) {
	tests := []struct {
		name      string
		js        string
		hidden    int
		layers    int
		heads     int
		kvHeads   int
		inter     int
		groupSize int
	}{
		{
			name: "7b",
			js: `{
				"architectures": ["Qwen2ForCausalLM"],
				"eos_token_id": 151645,
				"hidden_act": "silu",
				"hidden_size": 3584,
				"intermediate_size": 18944,
				"max_position_embeddings": 32768,
				"max_window_layers": 28,
				"model_type": "qwen2",
				"num_attention_heads": 28,
				"num_hidden_layers": 28,
				"num_key_value_heads": 4,
				"rms_norm_eps": 1e-6,
				"rope_theta": 1000000.0,
				"sliding_window": 131072,
				"tie_word_embeddings": false,
				"use_sliding_window": false,
				"vocab_size": 152064
			}`,
			hidden: 3584, layers: 28, heads: 28, kvHeads: 4, inter: 18944, groupSize: 7,
		},
		{
			name: "32b",
			js: `{
				"architectures": ["Qwen2ForCausalLM"],
				"eos_token_id": 151645,
				"hidden_act": "silu",
				"hidden_size": 5120,
				"intermediate_size": 27648,
				"max_position_embeddings": 32768,
				"max_window_layers": 70,
				"model_type": "qwen2",
				"num_attention_heads": 40,
				"num_hidden_layers": 64,
				"num_key_value_heads": 8,
				"rms_norm_eps": 1e-6,
				"rope_theta": 1000000.0,
				"sliding_window": 131072,
				"tie_word_embeddings": false,
				"use_sliding_window": false,
				"vocab_size": 152064
			}`,
			hidden: 5120, layers: 64, heads: 40, kvHeads: 8, inter: 27648, groupSize: 5,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cfg Config
			if err := json.Unmarshal([]byte(tt.js), &cfg); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if !strings.Contains(cfg.archFamilyKey(), "qwen2") {
				t.Fatalf("family = %q, want qwen2", cfg.archFamilyKey())
			}
			if cfg.HiddenSize != tt.hidden || cfg.NumLayers != tt.layers || cfg.NumHeads != tt.heads ||
				cfg.NumKVHeads != tt.kvHeads || cfg.IntermediateSize != tt.inter {
				t.Fatalf("dims = H:%d L:%d heads:%d kv:%d I:%d, want H:%d L:%d heads:%d kv:%d I:%d",
					cfg.HiddenSize, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.IntermediateSize,
					tt.hidden, tt.layers, tt.heads, tt.kvHeads, tt.inter)
			}
			if cfg.HeadDim != 128 {
				t.Fatalf("HeadDim = %d, want 128 derived from hidden/heads", cfg.HeadDim)
			}
			if cfg.GroupSize() != tt.groupSize {
				t.Fatalf("GQA group size = %d, want %d", cfg.GroupSize(), tt.groupSize)
			}
			if cfg.activationName() != "silu" || cfg.ActGeluTanh || cfg.ActGeluErf || cfg.DenseMLP {
				t.Fatalf("activation axes = name:%q tanh:%v erf:%v dense:%v, want SiLU SwiGLU",
					cfg.activationName(), cfg.ActGeluTanh, cfg.ActGeluErf, cfg.DenseMLP)
			}
			if !cfg.AttentionBias {
				t.Fatal("Qwen2 omitted attention_bias did not derive legacy q/k/v projection bias")
			}
			if len(cfg.Window) != 0 || cfg.windowForLayer(0) != -1 || cfg.windowForLayer(tt.layers-1) != -1 {
				t.Fatalf("use_sliding_window=false derived Window=%v, want full causal attention", cfg.Window)
			}
			if cfg.RopeTheta != 1000000 || cfg.MaxPositionEmbeddings != 32768 || cfg.VocabSize != 152064 {
				t.Fatalf("rope/context/vocab = %v/%d/%d, want 1000000/32768/152064",
					cfg.RopeTheta, cfg.MaxPositionEmbeddings, cfg.VocabSize)
			}
			if cfg.TieWordEmbeddings {
				t.Fatal("tie_word_embeddings=true, want false for production Qwen2.5 checkpoints")
			}
		})
	}
}

func TestConfigDerivesQwenLegacyBiasWithoutBreakingQwen36(t *testing.T) {
	var qwen2 Config
	err := json.Unmarshal([]byte(`{
		"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
		"num_key_value_heads": 2, "intermediate_size": 128, "vocab_size": 32,
		"rms_norm_eps": 1e-6, "rope_theta": 1000000, "model_type": "qwen2"
	}`), &qwen2)
	if err != nil {
		t.Fatalf("unmarshal qwen2: %v", err)
	}
	if !qwen2.AttentionBias {
		t.Fatal("qwen2 omitted attention_bias did not derive the legacy projection-bias default")
	}

	var qwen36 Config
	err = json.Unmarshal([]byte(`{
		"architectures": ["Qwen3_5ForConditionalGeneration"],
		"model_type": "qwen3_5",
		"text_config": {
			"attention_bias": false,
			"attn_output_gate": true,
			"eos_token_id": 248044,
			"full_attention_interval": 4,
			"head_dim": 256,
			"hidden_act": "silu",
			"hidden_size": 5120,
			"intermediate_size": 17408,
			"layer_types": [
				"linear_attention", "linear_attention", "linear_attention", "full_attention"
			],
			"linear_conv_kernel_dim": 4,
			"linear_key_head_dim": 128,
			"linear_num_key_heads": 16,
			"linear_num_value_heads": 48,
			"linear_value_head_dim": 128,
			"max_position_embeddings": 262144,
			"model_type": "qwen3_5_text",
			"num_attention_heads": 24,
			"num_hidden_layers": 4,
			"num_key_value_heads": 4,
			"partial_rotary_factor": 0.25,
			"rms_norm_eps": 1e-6,
			"rope_parameters": {
				"partial_rotary_factor": 0.25,
				"rope_theta": 10000000,
				"rope_type": "default"
			},
			"tie_word_embeddings": false,
			"vocab_size": 248320
		}
	}`), &qwen36)
	if err != nil {
		t.Fatalf("unmarshal qwen36: %v", err)
	}
	if qwen36.AttentionBias {
		t.Fatal("qwen36 explicit attention_bias=false was overwritten")
	}
	if !qwen36.IsQwen35Hybrid() || !qwen36.AttnOutputGate || !qwen36.NormGain1p {
		t.Fatalf("qwen36 hybrid axes missing: hybrid=%v gate=%v norm_gain_1p=%v",
			qwen36.IsQwen35Hybrid(), qwen36.AttnOutputGate, qwen36.NormGain1p)
	}
	if qwen36.NumLayers != 4 || qwen36.NumHeads != 24 || qwen36.NumKVHeads != 4 || qwen36.HeadDim != 256 {
		t.Fatalf("qwen36 text_config dims not preserved: layers=%d heads=%d kv=%d head_dim=%d",
			qwen36.NumLayers, qwen36.NumHeads, qwen36.NumKVHeads, qwen36.HeadDim)
	}
	if qwen36.PartialRotaryFactor != 0.25 || qwen36.RopeTheta != 10000000 {
		t.Fatalf("qwen36 rope axes = partial:%v theta:%v, want 0.25/10000000",
			qwen36.PartialRotaryFactor, qwen36.RopeTheta)
	}
}
