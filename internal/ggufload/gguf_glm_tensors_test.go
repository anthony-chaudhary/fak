package ggufload

import "testing"

// gguf_glm_tensors_test.go — the golden for the glm_moe_dsa tensor-name map
// (gguf_glm_tensors.go). It pins, from the package boundary, that every 1:1 GLM-specific
// per-layer GGUF tensor resolves to the canonical HF name the native glm_dsa forward reads,
// that the GLM-shared tensors still fall through to the base map under the glm_moe_dsa arch,
// that the map is arch-gated (the GLM names do NOT resolve as Llama), and that the batched
// routed experts are deliberately left unmapped (the loud-failure the splitter slice closes).

func TestGLMMoeDsaCanonicalTensorNames(t *testing.T) {
	const arch = "glm_moe_dsa"

	// 1:1 GLM-specific tensors -> canonical, at layer 3 (a real layer index round-trips).
	glm := map[string]string{
		"blk.3.attn_q_a.weight":             "model.layers.3.self_attn.q_a_proj.weight",
		"blk.3.attn_q_a.bias":               "model.layers.3.self_attn.q_a_proj.bias",
		"blk.3.attn_q_a_norm.weight":        "model.layers.3.self_attn.q_a_layernorm.weight",
		"blk.3.attn_q_b.weight":             "model.layers.3.self_attn.q_b_proj.weight",
		"blk.3.attn_kv_a_mqa.weight":        "model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
		"blk.3.attn_kv_a_mqa.bias":          "model.layers.3.self_attn.kv_a_proj_with_mqa.bias",
		"blk.3.attn_kv_a_norm.weight":       "model.layers.3.self_attn.kv_a_layernorm.weight",
		"blk.3.attn_kv_b.weight":            "model.layers.3.self_attn.kv_b_proj.weight",
		"blk.3.attn_output.bias":            "model.layers.3.self_attn.o_proj.bias",
		"blk.3.ffn_gate_inp.weight":         "model.layers.3.mlp.gate.weight",
		"blk.3.exp_probs_b.bias":            "model.layers.3.mlp.gate.e_score_correction_bias",
		"blk.3.ffn_gate_shexp.weight":       "model.layers.3.mlp.shared_experts.gate_proj.weight",
		"blk.3.ffn_up_shexp.weight":         "model.layers.3.mlp.shared_experts.up_proj.weight",
		"blk.3.ffn_down_shexp.weight":       "model.layers.3.mlp.shared_experts.down_proj.weight",
		"blk.3.attn_indexer_q_b.weight":     "model.layers.3.self_attn.indexer.wq_b.weight",
		"blk.3.attn_indexer_k.weight":       "model.layers.3.self_attn.indexer.wk.weight",
		"blk.3.attn_indexer_k_norm.weight":  "model.layers.3.self_attn.indexer.k_norm.weight",
		"blk.3.attn_indexer_k_norm.bias":    "model.layers.3.self_attn.indexer.k_norm.bias",
		"blk.3.attn_indexer_weights.weight": "model.layers.3.self_attn.indexer.weights_proj.weight",
	}
	for gguf, want := range glm {
		got, ok := CanonicalTensorNameArch(gguf, arch)
		if !ok {
			t.Errorf("CanonicalTensorNameArch(%q, glm_moe_dsa) did not resolve, want %q", gguf, want)
			continue
		}
		if got != want {
			t.Errorf("CanonicalTensorNameArch(%q, glm_moe_dsa) = %q, want %q", gguf, got, want)
		}
	}

	// GLM-shared tensors fall through to the base map under the glm arch: the attention/FFN
	// norms, o_proj.weight, the global tensors, and the leading-dense-layer MLP projections
	// (FirstKDenseReplace layers carry ffn_gate/up/down, not experts).
	shared := map[string]string{
		"token_embd.weight":        "model.embed_tokens.weight",
		"output_norm.weight":       "model.norm.weight",
		"output.weight":            "lm_head.weight",
		"blk.0.attn_norm.weight":   "model.layers.0.input_layernorm.weight",
		"blk.0.ffn_norm.weight":    "model.layers.0.post_attention_layernorm.weight",
		"blk.0.attn_output.weight": "model.layers.0.self_attn.o_proj.weight",
		"blk.0.ffn_gate.weight":    "model.layers.0.mlp.gate_proj.weight",
		"blk.0.ffn_up.weight":      "model.layers.0.mlp.up_proj.weight",
		"blk.0.ffn_down.weight":    "model.layers.0.mlp.down_proj.weight",
	}
	for gguf, want := range shared {
		got, ok := CanonicalTensorNameArch(gguf, arch)
		if !ok || got != want {
			t.Errorf("CanonicalTensorNameArch(%q, glm_moe_dsa) = (%q,%v), want (%q,true) via base-map fall-through", gguf, got, ok, want)
		}
	}

	// Arch-gating: the GLM-specific names must NOT resolve as Llama (arch=="") — they belong
	// only to glm_moe_dsa, so the branch never leaks into the default family.
	for gguf := range glm {
		if gguf == "blk.3.attn_output.bias" {
			continue // attn_output.bias is GLM-only here, but other archs lack it too; skip the shared-name edge
		}
		if _, ok := CanonicalTensorNameArch(gguf, ""); ok {
			t.Errorf("CanonicalTensorNameArch(%q, \"\") resolved as Llama, want no mapping (GLM-only tensor)", gguf)
		}
	}

	// The batched routed experts are DELIBERATELY unmapped (the 1->E split the loader-side
	// expert splitter will do) — they must fail loud, not resolve to a wrong dense name.
	for _, gguf := range []string{
		"blk.3.ffn_gate_exps.weight",
		"blk.3.ffn_up_exps.weight",
		"blk.3.ffn_down_exps.weight",
	} {
		if got, ok := CanonicalTensorNameArch(gguf, arch); ok {
			t.Errorf("CanonicalTensorNameArch(%q, glm_moe_dsa) resolved to %q, want NO mapping until the expert splitter lands", gguf, got)
		}
	}
}
