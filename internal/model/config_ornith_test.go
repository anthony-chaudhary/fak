package model

import (
	"encoding/json"
	"testing"
)

// TestConfigOrnithMoEExpertKeyAliases pins the keystone Ornith fix (#1027): the
// qwen3_5_moe checkpoints serialize their expert count under "num_experts" and the
// shared-expert FFN width under "shared_expert_intermediate_size". Before the alias
// overlay these keys went unread, NumExperts resolved to 0, IsMoE() returned false,
// and the 35B/397B silently loaded as a dense model. The fixture mirrors the real
// Ornith-1.0-35B config.json shape.
func TestConfigOrnithMoEExpertKeyAliases(t *testing.T) {
	js := `{
		"model_type": "qwen3_5_moe",
		"architectures": ["Qwen3_5MoeForCausalLM"],
		"hidden_size": 2048,
		"num_attention_heads": 16,
		"num_hidden_layers": 4,
		"vocab_size": 248320,
		"intermediate_size": 6144,
		"num_experts": 256,
		"num_experts_per_tok": 8,
		"moe_intermediate_size": 512,
		"shared_expert_intermediate_size": 512
	}`
	var cfg Config
	if err := json.Unmarshal([]byte(js), &cfg); err != nil {
		t.Fatalf("unmarshal Ornith MoE config: %v", err)
	}
	if !cfg.IsMoE() {
		t.Fatalf("IsMoE() = false; want true (num_experts:256 must populate NumExperts)")
	}
	if cfg.NumExperts != 256 {
		t.Errorf("NumExperts = %d; want 256 (from num_experts alias)", cfg.NumExperts)
	}
	if cfg.NumExpertsPerTok != 8 {
		t.Errorf("NumExpertsPerTok = %d; want 8 (canonical num_experts_per_tok)", cfg.NumExpertsPerTok)
	}
	if cfg.SharedIntermediateSize != 512 {
		t.Errorf("SharedIntermediateSize = %d; want 512 (from shared_expert_intermediate_size alias)", cfg.SharedIntermediateSize)
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
}
