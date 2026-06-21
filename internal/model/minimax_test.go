package model

import (
	"encoding/json"
	"testing"
)

// minimax_test.go — host-tractable witnesses for MiniMax-M3 family recognition and
// load-path behavior. Like glm_test.go, these prove the deterministic config->axis
// mapping and the multimodal/MTP tensor skip from in-repo JSON only, with no HF
// download and no real checkpoint. The MSA selection numerics are witnessed
// separately in msa_index_test.go.
//
// CLAIMED here: the MiniMax family is recognized from model_type + architectures; the
// M3 sparse-attention variant and its per-layer MSA/full split are derived; per-head
// qk-norm is enabled; the MSA block-selection config (index_block_size /
// index_topk_blocks / index_local_blocks) decodes; and the vision tower + MTP head are
// dropped at load on both the f32 and quant paths while attention/MoE tensors are kept.
// NOT CLAIMED: a wired MSA forward, the learned lightning indexer / SwiGLU-OAI MoE, or
// real-checkpoint numeric parity — those remain a separate (GPU/artifact) gate.

func TestMiniMaxFamilyDerivationFromConfig(t *testing.T) {
	m3 := `{
		"hidden_size": 16, "num_hidden_layers": 4, "num_attention_heads": 4,
		"num_key_value_heads": 2, "head_dim": 4, "intermediate_size": 32,
		"vocab_size": 64, "rms_norm_eps": 1e-5, "rope_theta": 10000,
		"model_type": "minimax_m3", "architectures": ["MiniMaxM3ForCausalLM"],
		"num_local_experts": 8, "num_experts_per_tok": 2, "norm_topk_prob": true,
		"index_block_size": 128, "index_topk_blocks": 8, "index_local_blocks": 1,
		"layer_types": ["full_attention", "minimax_m3_sparse", "minimax_m3_sparse", "full_attention"]
	}`
	var cfg Config
	if err := json.Unmarshal([]byte(m3), &cfg); err != nil {
		t.Fatalf("unmarshal minimax_m3: %v", err)
	}
	if !cfg.isMiniMax() {
		t.Fatalf("minimax_m3: isMiniMax=false, want true (family key=%q)", cfg.archFamilyKey())
	}
	if !cfg.isMiniMaxSparseAttn() {
		t.Fatalf("minimax_m3: isMiniMaxSparseAttn=false, want true (family key=%q)", cfg.archFamilyKey())
	}
	if cfg.IndexBlockSize != 128 || cfg.IndexTopKBlocks != 8 || cfg.IndexLocalBlocks != 1 {
		t.Fatalf("MSA config = block:%d topk:%d local:%d, want 128/8/1",
			cfg.IndexBlockSize, cfg.IndexTopKBlocks, cfg.IndexLocalBlocks)
	}
	if !cfg.QKNorm {
		t.Fatalf("minimax_m3: QKNorm=false, want true (per-head q/k norm derived)")
	}
	if !cfg.IsMoE() || cfg.NumExperts != 8 || cfg.NumExpertsPerTok != 2 {
		t.Fatalf("minimax_m3 MoE = experts:%d topk:%d IsMoE:%v, want 8/2/true",
			cfg.NumExperts, cfg.NumExpertsPerTok, cfg.IsMoE())
	}
	// Per-layer MSA/full split from layer_types.
	for l, want := range []bool{false, true, true, false} {
		if got := cfg.isMSALayer(l); got != want {
			t.Fatalf("isMSALayer(%d)=%v, want %v (layer_type=%q)", l, got, want, cfg.layerType(l))
		}
	}

	// A MiniMax model whose family key lacks "m3" is still recognized as the sparse
	// variant when a layer is typed minimax_m3_sparse (the layer-type signal).
	var byLayer Config
	if err := json.Unmarshal([]byte(`{
		"hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2,
		"head_dim": 4, "intermediate_size": 16, "vocab_size": 32,
		"model_type": "minimax",
		"layer_types": ["full_attention", "minimax_m3_sparse"]
	}`), &byLayer); err != nil {
		t.Fatalf("unmarshal minimax-by-layer: %v", err)
	}
	if !byLayer.isMiniMax() || !byLayer.isMiniMaxSparseAttn() {
		t.Fatalf("minimax-by-layer: isMiniMax=%v sparse=%v, want true/true",
			byLayer.isMiniMax(), byLayer.isMiniMaxSparseAttn())
	}

	// A non-M3 MiniMax (no m3 key, no sparse layer types) is MiniMax but NOT the MSA
	// variant — guards against treating M1/M2 as M3.
	var m2 Config
	if err := json.Unmarshal([]byte(`{
		"hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2,
		"head_dim": 4, "intermediate_size": 16, "vocab_size": 32,
		"model_type": "minimax", "layer_types": ["full_attention", "full_attention"]
	}`), &m2); err != nil {
		t.Fatalf("unmarshal minimax (non-m3): %v", err)
	}
	if !m2.isMiniMax() {
		t.Fatalf("minimax (non-m3): isMiniMax=false, want true")
	}
	if m2.isMiniMaxSparseAttn() {
		t.Fatalf("minimax (non-m3): isMiniMaxSparseAttn=true, want false")
	}
	if m2.isMSALayer(0) || m2.isMSALayer(1) {
		t.Fatalf("minimax (non-m3): a full_attention layer must not be an MSA layer")
	}

	// A non-MiniMax family is neither (substring false-positive guard).
	var llama Config
	if err := json.Unmarshal([]byte(`{
		"hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
		"head_dim": 4, "intermediate_size": 16, "vocab_size": 32, "model_type": "llama"
	}`), &llama); err != nil {
		t.Fatalf("unmarshal llama: %v", err)
	}
	if llama.isMiniMax() || llama.isMiniMaxSparseAttn() || llama.isMSALayer(0) {
		t.Fatalf("llama: minimax flags = %v/%v/%v, want all false",
			llama.isMiniMax(), llama.isMiniMaxSparseAttn(), llama.isMSALayer(0))
	}
}

// TestMiniMaxQKNormOptOut proves the derived per-head qk-norm can be explicitly
// disabled by an export that pins qk_norm=false, mirroring the other qk-norm families.
func TestMiniMaxQKNormOptOut(t *testing.T) {
	var cfg Config
	if err := json.Unmarshal([]byte(`{
		"hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
		"head_dim": 4, "intermediate_size": 16, "vocab_size": 32,
		"model_type": "minimax_m3", "qk_norm": false
	}`), &cfg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if cfg.QKNorm {
		t.Fatalf("explicit qk_norm=false was overridden to true")
	}
}

// TestMiniMaxDropsMtpAndVisualTensorsAtLoad proves the MiniMax-M3 load path drops the
// vision tower ("model.visual.") and MTP head ("mtp.") tensors on both the f32 and
// quant paths, while keeping the text attention/MLP/expert tensors, and that a dense
// Llama config is completely unaffected (the Llama-invariance gate).
func TestMiniMaxDropsMtpAndVisualTensorsAtLoad(t *testing.T) {
	cfg := Config{ModelType: "minimax_m3", Architectures: []string{"MiniMaxM3ForCausalLM"}}
	if !cfg.dropsMtpAndVisualAtLoad() {
		t.Fatalf("minimax_m3: dropsMtpAndVisualAtLoad=false, want true")
	}
	for _, name := range []string{"model.visual.encoder.weight", "mtp.0.embed.weight", "mtp.head.weight"} {
		if !skipLoadTensor(cfg, name) {
			t.Fatalf("minimax: skipLoadTensor(%q)=false, want true", name)
		}
		if got, keep := quantSourceTensorName(cfg, name); keep || got != "" {
			t.Fatalf("minimax quant: quantSourceTensorName(%q)=(%q,%v), want dropped", name, got, keep)
		}
	}
	for _, name := range []string{
		"model.embed_tokens.weight",
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.mlp.gate_proj.weight",
		"model.layers.0.mlp.experts.0.gate_proj.weight",
	} {
		if skipLoadTensor(cfg, name) {
			t.Fatalf("minimax: skipLoadTensor(%q)=true, want false (kept tensor)", name)
		}
		if got, keep := quantSourceTensorName(cfg, name); !keep || got != name {
			t.Fatalf("minimax quant: quantSourceTensorName(%q)=(%q,%v), want kept unchanged", name, got, keep)
		}
	}

	llama := Config{ModelType: "llama"}
	if llama.dropsMtpAndVisualAtLoad() {
		t.Fatalf("llama: dropsMtpAndVisualAtLoad=true, want false")
	}
	if skipLoadTensor(llama, "mtp.0.embed.weight") {
		t.Fatalf("llama: skipLoadTensor(mtp.*)=true, want false (skip must be family-gated)")
	}
}
