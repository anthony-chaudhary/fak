package ggufload

// gguf_glm_tensors.go — the glm_moe_dsa (GLM-5.2) per-layer GGUF tensor-name map, the
// Pillar-1 "tensor names" slice of the native-753B track (docs/notes/
// native-753b-track-staged-plan.md). It maps the 1:1 GLM-specific GGUF tensor suffixes to
// the canonical HF names internal/model's native glm_dsa forward already consumes
// (self_attn.q_a_proj / kv_a_proj_with_mqa / kv_b_proj …, the indexer wq_b/wk/weights_proj,
// the router mlp.gate.weight + its e_score_correction_bias, and mlp.shared_experts.*).
//
// PROVISIONAL KEY SPELLINGS — read this. No real GLM-5.2 GGUF exists on disk and upstream
// llama.cpp ships no glm_moe_dsa converter yet, so the GGUF-side spellings here are NOT
// validated against a real checkpoint. The MLA + MoE + shared-expert names follow llama.cpp's
// established deepseek2.* convention (GLM-DSA attention IS DeepSeek MLA), so those are
// high-confidence; the THREE DSA-indexer names have NO upstream precedent and are a best-guess
// mirror of the canonical names — they are the single most fragile assumption and are pinned in
// the named block below so the closing follow-on (a golden against a real GGUF header) only
// re-pins this one block. This mirrors exactly how applyGLMMoeDsaConfig pinned the metadata-key
// spellings in gguf_config.go.
//
// NOT mapped here (by design): the batched ROUTED experts ffn_gate_exps / ffn_up_exps /
// ffn_down_exps. Each is a single [E,…] blob that must split into E per-expert canonical
// tensors (mlp.experts.<e>.{gate,up,down}_proj.weight) — a 1→E expansion CanonicalTensorNameArch
// (one name in, one name out) structurally cannot express. Leaving them unmapped makes a
// glm_moe_dsa GGUF that reaches the routed experts fail LOUD ("no canonical mapping") rather
// than load a silently-wrong dense-shaped model; the loader-side expert splitter is the next
// slice, after which the load completes end to end.

// glmMoeDsaGGUFSuffix is the provisional GGUF-side spelling of each 1:1 glm_moe_dsa per-layer
// tensor (the part after "blk.<L>."). Grouped so the high-confidence deepseek2-convention
// names and the best-guess indexer names are visibly separate. RE-PIN THE INDEXER BLOCK
// against a real GGUF header before treating the GGUF load as validated.
const (
	// MLA latent attention (deepseek2 convention).
	glmGGUFAttnQADown  = "attn_q_a.weight"       // q_a_proj   (down-projection to q_lora_rank)
	glmGGUFAttnQADownB = "attn_q_a.bias"         // q_a_proj.bias (optional)
	glmGGUFAttnQBUp    = "attn_q_b.weight"       // q_b_proj   (up-projection to heads)
	glmGGUFAttnKVAMQA  = "attn_kv_a_mqa.weight"  // kv_a_proj_with_mqa
	glmGGUFAttnKVAMQAB = "attn_kv_a_mqa.bias"    // kv_a_proj_with_mqa.bias (optional)
	glmGGUFAttnKVANorm = "attn_kv_a_norm.weight" // kv_a_layernorm
	glmGGUFAttnKVB     = "attn_kv_b.weight"      // kv_b_proj
	glmGGUFAttnOutputB = "attn_output.bias"      // o_proj.bias (the .weight is the base map's)

	// MoE router (deepseek2 convention): ffn_gate_inp is the router gate matmul; exp_probs_b
	// is the per-expert score-correction bias added to the router logits before top-k.
	glmGGUFRouter     = "ffn_gate_inp.weight" // mlp.gate.weight
	glmGGUFRouterBias = "exp_probs_b.bias"    // mlp.gate.e_score_correction_bias

	// Shared experts (deepseek2 convention): the always-on expert run beside the routed ones.
	glmGGUFSharedGate = "ffn_gate_shexp.weight" // mlp.shared_experts.gate_proj.weight
	glmGGUFSharedUp   = "ffn_up_shexp.weight"   // mlp.shared_experts.up_proj.weight
	glmGGUFSharedDown = "ffn_down_shexp.weight" // mlp.shared_experts.down_proj.weight

	// DSA learned indexer — PROVISIONAL, NO UPSTREAM CONVERTER. Best-guess mirror of the
	// canonical self_attn.indexer.{wq_b,wk,weights_proj} names. Re-pin against a real header.
	glmGGUFIndexerWQB     = "attn_indexer_q_b.weight"     // indexer.wq_b
	glmGGUFIndexerWK      = "attn_indexer_k.weight"       // indexer.wk
	glmGGUFIndexerWeights = "attn_indexer_weights.weight" // indexer.weights_proj
)

// glmMoeDsaCanonicalSuffix maps a glm_moe_dsa per-layer GGUF tensor suffix (after "blk.<L>.")
// to the canonical HF suffix (after "model.layers.<L>.") the native glm_dsa forward reads.
// Returns ok=false for any suffix that is not GLM-specific so CanonicalTensorNameArch falls
// through to the shared base map (attn_norm, ffn_norm, attn_output.weight, and the leading
// dense layers' ffn_gate/up/down), and for the batched routed experts (intentionally
// unmapped — see the file header).
func glmMoeDsaCanonicalSuffix(suffix string) (string, bool) {
	mapped, ok := map[string]string{
		glmGGUFAttnQADown:  "self_attn.q_a_proj.weight",
		glmGGUFAttnQADownB: "self_attn.q_a_proj.bias",
		glmGGUFAttnQBUp:    "self_attn.q_b_proj.weight",
		glmGGUFAttnKVAMQA:  "self_attn.kv_a_proj_with_mqa.weight",
		glmGGUFAttnKVAMQAB: "self_attn.kv_a_proj_with_mqa.bias",
		glmGGUFAttnKVANorm: "self_attn.kv_a_layernorm.weight",
		glmGGUFAttnKVB:     "self_attn.kv_b_proj.weight",
		glmGGUFAttnOutputB: "self_attn.o_proj.bias",

		glmGGUFRouter:     "mlp.gate.weight",
		glmGGUFRouterBias: "mlp.gate.e_score_correction_bias",

		glmGGUFSharedGate: "mlp.shared_experts.gate_proj.weight",
		glmGGUFSharedUp:   "mlp.shared_experts.up_proj.weight",
		glmGGUFSharedDown: "mlp.shared_experts.down_proj.weight",

		glmGGUFIndexerWQB:     "self_attn.indexer.wq_b.weight",
		glmGGUFIndexerWK:      "self_attn.indexer.wk.weight",
		glmGGUFIndexerWeights: "self_attn.indexer.weights_proj.weight",
	}[suffix]
	return mapped, ok
}
