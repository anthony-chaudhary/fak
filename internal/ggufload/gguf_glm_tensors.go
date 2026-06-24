package ggufload

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// gguf_glm_tensors.go — the glm_moe_dsa (GLM-5.2) per-layer GGUF tensor-name map, the
// Pillar-1 "tensor names" slice of the native-753B track (docs/notes/
// native-753b-track-staged-plan.md). It maps the 1:1 GLM-specific GGUF tensor suffixes to
// the canonical HF names internal/model's native glm_dsa forward already consumes
// (self_attn.q_a_proj / kv_a_proj_with_mqa / kv_b_proj …, the indexer wq_b/wk/weights_proj,
// the router mlp.gate.weight + its e_score_correction_bias, and mlp.shared_experts.*).
//
// VALIDATED KEY SPELLINGS (2026-06-24). The GGUF-side spellings were re-pinned against the real
// community GLM-5.2-Q4_K_M GGUF on the lab GPU server (general.architecture "glm-dsa", llama.cpp
// LLM_ARCH_GLM_DSA — fak normalizes that to "glm_moe_dsa", see canonicalGGUFArch). The MLA + MoE
// + shared-expert names matched the guessed deepseek2.* convention; the DSA-indexer names did
// NOT — the real file uses an "indexer." sub-namespace (indexer.attn_q_b / indexer.attn_k /
// indexer.k_norm / indexer.proj), so the earlier best-guess attn_indexer_* names were corrected.
//
// KNOWN REMAINING GAP (next slice): the real file splits the KV-b up-projection into SEPARATE
// blk.<L>.attn_k_b + blk.<L>.attn_v_b tensors, whereas fak's forward consumes ONE combined
// self_attn.kv_b_proj.weight. Those two names are deliberately left unmapped here so a real load
// fails LOUD rather than silently mis-shaping — the per-head 2→1 merge (DeepSeek [qkNope]+[vHead]
// per head) is the next loader slice.
//
// NOT mapped here (by design): the batched ROUTED experts ffn_gate_exps / ffn_up_exps /
// ffn_down_exps. Each is a single [E,…] blob that must split into E per-expert canonical
// tensors (mlp.experts.<e>.{gate,up,down}_proj.weight) — a 1→E expansion CanonicalTensorNameArch
// (one name in, one name out) structurally cannot express. The loader-side expert splitter
// (glmMoeDsaBatchedExpert) handles that before CanonicalTensorNameArch is consulted.

// glmMoeDsaGGUFSuffix is the provisional GGUF-side spelling of each 1:1 glm_moe_dsa per-layer
// tensor (the part after "blk.<L>."). Grouped so the high-confidence deepseek2-convention
// names and the best-guess indexer names are visibly separate. RE-PIN THE INDEXER BLOCK
// against a real GGUF header before treating the GGUF load as validated.
const (
	// MLA latent attention (deepseek2 convention).
	glmGGUFAttnQADown  = "attn_q_a.weight"       // q_a_proj   (down-projection to q_lora_rank)
	glmGGUFAttnQADownB = "attn_q_a.bias"         // q_a_proj.bias (optional)
	glmGGUFAttnQANorm  = "attn_q_a_norm.weight"  // q_a_layernorm (the RMSNorm on the q latent)
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

	// DSA learned indexer — VALIDATED 2026-06-24 against the real GLM-5.2 (glm-dsa) Q4_K_M
	// GGUF on the lab GPU server (llama.cpp LLM_ARCH_GLM_DSA). The real spellings live under
	// an "indexer." sub-namespace and use "proj" (not "weights"); these replace the earlier
	// best-guess attn_indexer_* names, which the real file did NOT use.
	glmGGUFIndexerWQB     = "indexer.attn_q_b.weight" // indexer.wq_b
	glmGGUFIndexerWK      = "indexer.attn_k.weight"   // indexer.wk
	glmGGUFIndexerKNorm   = "indexer.k_norm.weight"   // indexer.k_norm.weight (RMSNorm on the index key)
	glmGGUFIndexerKNormB  = "indexer.k_norm.bias"     // indexer.k_norm.bias
	glmGGUFIndexerWeights = "indexer.proj.weight"     // indexer.weights_proj
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
		glmGGUFAttnQANorm:  "self_attn.q_a_layernorm.weight",
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
		glmGGUFIndexerKNorm:   "self_attn.indexer.k_norm.weight",
		glmGGUFIndexerKNormB:  "self_attn.indexer.k_norm.bias",
		glmGGUFIndexerWeights: "self_attn.indexer.weights_proj.weight",
	}[suffix]
	return mapped, ok
}

// Batched routed-expert GGUF tensors (deepseek2 convention): one [E,…] blob per layer that the
// loader must SPLIT 1→E. CanonicalTensorNameArch leaves these unmapped on purpose — a single name
// cannot become E per-expert names — so the split lives here + the loader (gguf_weightsource.go).
const (
	glmGGUFExpertsGate = "ffn_gate_exps.weight" // -> mlp.experts.<e>.gate_proj.weight
	glmGGUFExpertsUp   = "ffn_up_exps.weight"   // -> mlp.experts.<e>.up_proj.weight
	glmGGUFExpertsDown = "ffn_down_exps.weight" // -> mlp.experts.<e>.down_proj.weight
)

// glmMoeDsaBatchedExpert reports whether a glm_moe_dsa GGUF tensor name is a batched routed-expert
// blob and, if so, returns its layer index and the per-expert canonical projection name
// (gate_proj/up_proj/down_proj). These are the tensors the loader splits into E per-expert 2-D
// tensors (mlp.experts.<e>.<proj>.weight) — the form internal/model's MoE forward consumes
// (moe.go expertName). Detected from the GGUF name (not the canonical map) so the split happens
// BEFORE CanonicalTensorNameArch, which deliberately returns no mapping for these.
func glmMoeDsaBatchedExpert(name string) (layer int, proj string, ok bool) {
	if !strings.HasPrefix(name, "blk.") {
		return 0, "", false
	}
	rest := strings.TrimPrefix(name, "blk.")
	dot := strings.IndexByte(rest, '.')
	if dot <= 0 {
		return 0, "", false
	}
	l, err := strconv.Atoi(rest[:dot])
	if err != nil {
		return 0, "", false
	}
	switch rest[dot+1:] {
	case glmGGUFExpertsGate:
		return l, "gate_proj", true
	case glmGGUFExpertsUp:
		return l, "up_proj", true
	case glmGGUFExpertsDown:
		return l, "down_proj", true
	}
	return 0, "", false
}

// splitGLMMoeDsaExperts expands a batched routed-expert tensor — already dequantized to f32, model
// shape [E, out, in] (the GGUF stores [in, out, E]; modelShapeFromGGUFDims reverses it) — into E
// per-expert canonical 2-D tensors model.layers.<layer>.mlp.experts.<e>.<proj>.weight of shape
// [out, in]. Expert e is the contiguous block data[e*out*in : (e+1)*out*in], so the split is a pure
// row-major reslice along the leading expert axis (bit-equal to manual slicing). Each expert's slice
// is copied into its own backing array so a later quantize/normalize cannot alias across experts.
func splitGLMMoeDsaExperts(layer int, proj string, shape []int, data []float32) ([]model.NamedTensorF32, error) {
	if len(shape) != 3 {
		return nil, fmt.Errorf("gguf: glm_moe_dsa batched expert tensor must be 3-D [E,out,in], got shape %v", shape)
	}
	e, out, in := shape[0], shape[1], shape[2]
	if e <= 0 || out <= 0 || in <= 0 {
		return nil, fmt.Errorf("gguf: glm_moe_dsa batched expert tensor has non-positive dim in [%d,%d,%d]", e, out, in)
	}
	per := out * in
	if len(data) != e*per {
		return nil, fmt.Errorf("gguf: glm_moe_dsa expert blob [%d,%d,%d] has %d values, want %d", e, out, in, len(data), e*per)
	}
	tensors := make([]model.NamedTensorF32, e)
	for x := 0; x < e; x++ {
		seg := make([]float32, per)
		copy(seg, data[x*per:(x+1)*per])
		tensors[x] = model.NamedTensorF32{
			Name:  fmt.Sprintf("model.layers.%d.mlp.experts.%d.%s.weight", layer, x, proj),
			Shape: []int{out, in},
			Data:  seg,
		}
	}
	return tensors, nil
}
