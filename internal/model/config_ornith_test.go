package model

import (
	"encoding/json"
	"fmt"
	"testing"
)

// TestConfigOrnithMoEPublishedAxes pins the published 35B and 397B
// text-config axes. Their configs omit norm_topk_prob, while the Qwen3.5-MoE router
// always renormalizes the selected weights; only absence receives that family default.
func TestConfigOrnithMoEPublishedAxes(t *testing.T) {
	tests := []struct {
		name, normField      string
		wrapperNormField     string
		experts, topK, width int
		wantNorm             bool
	}{
		{name: "35B absent defaults true", experts: 256, topK: 8, width: 512, wantNorm: true},
		{name: "397B absent defaults true", experts: 512, topK: 10, width: 1024, wantNorm: true},
		{name: "35B explicit false stays false", normField: `,"norm_topk_prob":false`, experts: 256, topK: 8, width: 512},
		{name: "397B explicit true stays true", normField: `,"norm_topk_prob":true`, experts: 512, topK: 10, width: 1024, wantNorm: true},
		{name: "wrapper explicit false overrides absent", wrapperNormField: `,"norm_topk_prob":false`, experts: 256, topK: 8, width: 512},
		{name: "wrapper explicit false overrides nested true", normField: `,"norm_topk_prob":true`, wrapperNormField: `,"norm_topk_prob":false`, experts: 512, topK: 10, width: 1024},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			js := fmt.Sprintf(`{
				"model_type":"qwen3_5_moe",
				"architectures":["Qwen3_5MoeForConditionalGeneration"]%s,
				"text_config":{
					"model_type":"qwen3_5_moe_text",
					"hidden_size":2048,
					"num_attention_heads":16,
					"num_hidden_layers":4,
					"vocab_size":248320,
					"intermediate_size":6144,
					"num_experts":%d,
					"num_experts_per_tok":%d,
					"moe_intermediate_size":%d,
					"shared_expert_intermediate_size":%d%s
				}
			}`, test.wrapperNormField, test.experts, test.topK, test.width, test.width, test.normField)
			var cfg Config
			if err := json.Unmarshal([]byte(js), &cfg); err != nil {
				t.Fatalf("unmarshal published-axis Ornith MoE config: %v", err)
			}
			if !cfg.IsMoE() || cfg.NumExperts != test.experts || cfg.NumExpertsPerTok != test.topK || cfg.MoEIntermediateSize != test.width || cfg.SharedIntermediateSize != test.width {
				t.Fatalf("axes = MoE:%t experts:%d top-k:%d routed:%d shared:%d, want true/%d/%d/%d/%d",
					cfg.IsMoE(), cfg.NumExperts, cfg.NumExpertsPerTok, cfg.MoEIntermediateSize, cfg.SharedIntermediateSize,
					test.experts, test.topK, test.width, test.width)
			}
			if cfg.NormTopKProb != test.wantNorm {
				t.Fatalf("NormTopKProb = %t, want %t", cfg.NormTopKProb, test.wantNorm)
			}
		})
	}
}

// TestConfigCanonicalExpertKeysUnchanged proves the alias overlay does not disturb the
// existing canonical-key families: a Mixtral-style num_local_experts config still loads
// as MoE with its own count, and a dense llama config stays dense (NumExperts==0).
func TestConfigCanonicalExpertKeysUnchanged(t *testing.T) {
	mixtral := `{
		"model_type": "mixtral",
		"architectures": ["MixtralForCausalLM"],
		"hidden_size": 4096,
		"num_attention_heads": 32,
		"num_hidden_layers": 4,
		"vocab_size": 32000,
		"intermediate_size": 14336,
		"num_local_experts": 8,
		"num_experts_per_tok": 2
	}`
	var mx Config
	if err := json.Unmarshal([]byte(mixtral), &mx); err != nil {
		t.Fatalf("unmarshal Mixtral config: %v", err)
	}
	if !mx.IsMoE() || mx.NumExperts != 8 || mx.NumExpertsPerTok != 2 {
		t.Fatalf("Mixtral fields = experts:%d topk:%d IsMoE:%v; want 8/2/true",
			mx.NumExperts, mx.NumExpertsPerTok, mx.IsMoE())
	}

	dense := `{
		"model_type": "llama",
		"architectures": ["LlamaForCausalLM"],
		"hidden_size": 4096,
		"num_attention_heads": 32,
		"num_hidden_layers": 4,
		"vocab_size": 32000,
		"intermediate_size": 11008
	}`
	var d Config
	if err := json.Unmarshal([]byte(dense), &d); err != nil {
		t.Fatalf("unmarshal dense llama config: %v", err)
	}
	if d.IsMoE() || d.NumExperts != 0 {
		t.Fatalf("dense llama fields = experts:%d IsMoE:%v; want 0/false", d.NumExperts, d.IsMoE())
	}

	// A non-Qwen family using the generic num_experts alias does not inherit the
	// Qwen3.5-MoE normalization decision.
	var other Config
	if err := json.Unmarshal([]byte(`{"model_type":"other_moe","architectures":["Qwen3_5MoeForConditionalGeneration"],"num_experts":4,"num_experts_per_tok":2}`), &other); err != nil {
		t.Fatalf("unmarshal non-Qwen MoE: %v", err)
	}
	if other.NormTopKProb {
		t.Fatal("non-Qwen absent norm_topk_prob unexpectedly defaulted true")
	}
}
