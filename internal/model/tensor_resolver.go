package model

// Family-aware tensor-name resolver (issue #473).
//
// The core forward pass reads a fixed set of CANONICAL tensor names — the Llama /
// SmolLM2 / Qwen vocabulary: model.embed_tokens.weight, model.norm.weight, and per
// layer model.layers.<l>.self_attn.{q,k,v,o}_proj.weight, the input/post-attention
// norms, and mlp.{gate,up,down}_proj.weight (see forward.go, batch.go, hal.go, arch.go).
// A real non-Llama checkpoint stores the SAME weights under DIFFERENT names (GPT-NeoX's
// gpt_neox.layers.<l>.attention.query_key_value.weight) or carries EXTRA required tensors
// the Llama set never had (Gemma's pre/post feed-forward norms, OLMo2/Gemma3/Cohere2
// qk-norms). Without a name map the config can select the right axis yet the loader still
// hunts for Llama-style names and silently misses the weight.
//
// This file is the DECLARATIVE companion to the imperative materialize* passes in
// weights.go. Those passes DO the byte-level work — aliasing a source name to a canonical
// one, splitting a fused query_key_value into q/k/v, validating shape via requireF32Shape.
// This resolver DESCRIBES, per family, exactly which canonical tensors are required and
// which source names in a manifest satisfy each one, so a synthetic manifest can be proven
// to resolve every required canonical tensor — or fail with a precise error naming the
// missing tensor and the source names that were searched. It reuses the existing seam
// (layerPrefix/layerName/itoa/archFamilyKey) rather than duplicating it, and it is a
// STANDALONE read-only API: newModel does not call it, so the Llama/SmolLM2 load path is
// byte-for-byte unchanged (the resolver is a no-op for Llama by construction — every
// required canonical tensor resolves to ITSELF, with zero aliases).
//
// Scope: this proves the loader can FIND the right tensors for a family. NUMERIC
// correctness — that the family runs to f32 parity with a HF oracle — is out of scope and
// gated by #47; shape validation is the materialize* passes' requireF32Shape.

import (
	"fmt"
	"strings"
)

// tensorReq is one canonical tensor the core forward pass reads, plus the ordered list of
// additional source manifest names that can satisfy it (renames, or a fused parent that a
// materialize* pass would split). The canonical name is always an implicit FIRST candidate
// — the HF-standard identity — so a Llama manifest resolves every req to itself with no
// aliases. An optional req that resolves to nothing is dropped, not an error (q/k/v bias,
// a tied lm_head, a norm bias a family may omit).
type tensorReq struct {
	canonical string
	aliases   []string
	optional  bool
}

// candidates is the canonical name followed by its aliases — the full set of manifest
// names that resolve this req, searched in order (first present wins).
func (r tensorReq) candidates() []string {
	return append([]string{r.canonical}, r.aliases...)
}

// resolve returns the first candidate present in the manifest, or ("", false).
func (r tensorReq) resolve(man map[string]tensorMeta) (string, bool) {
	for _, name := range r.candidates() {
		if _, ok := man[name]; ok {
			return name, true
		}
	}
	return "", false
}

// resolverSpec is a family's full required-tensor description: a human-readable family
// label, the global (non-layer) canonical tensors, and a per-layer template. It is the
// table the resolver walks.
type resolverSpec struct {
	family   string
	globals  []tensorReq
	perLayer func(layer int) []tensorReq
}

// Resolution is the result of ResolveTensorNames: for every REQUIRED canonical tensor that
// resolved, the source manifest name that provides it (the canonical name itself when the
// match is identity). Optional tensors that were absent do not appear.
type Resolution struct {
	Family   string
	Resolved map[string]string
}

// SourceFor returns the source manifest name that resolves a canonical tensor, or ""
// if it was not required/resolved.
func (r *Resolution) SourceFor(canonical string) string { return r.Resolved[canonical] }

// ResolveTensorNames maps a checkpoint's (model_type, architectures) + tensor manifest onto
// the canonical names the core forward pass reads. For each required canonical tensor it
// records the resolving source name; a required tensor with no source is a precise error
// naming the family, the missing canonical tensor, and the candidate source names searched.
// It inspects manifest KEYS only (presence), never tensor bytes, so callers can resolve a
// manifest before any weights are materialized.
//
// Llama / SmolLM2 / Qwen route to the identity spec: every required canonical tensor
// resolves to itself, so the result is a pure no-op map.
func ResolveTensorNames(cfg Config, manifest map[string]tensorMeta) (*Resolution, error) {
	spec := resolveSpecFor(cfg)
	res := &Resolution{Family: spec.family, Resolved: make(map[string]string)}
	reqs := append([]tensorReq(nil), spec.globals...)
	if spec.perLayer != nil {
		for l := 0; l < cfg.NumLayers; l++ {
			reqs = append(reqs, spec.perLayer(l)...)
		}
	}
	for _, r := range reqs {
		src, ok := r.resolve(manifest)
		if ok {
			res.Resolved[r.canonical] = src
			continue
		}
		if r.optional {
			continue
		}
		return nil, fmt.Errorf("model: %s family: required canonical tensor %q has no source in the manifest (searched: %s)",
			spec.family, r.canonical, strings.Join(r.candidates(), ", "))
	}
	return res, nil
}

// resolveSpecFor picks the family spec from the lowercased, separator-stripped
// model_type+architectures key (archFamilyKey). The default — and the only path Llama /
// SmolLM2 / Qwen take — is the identity spec, so those checkpoints are unaffected.
func resolveSpecFor(cfg Config) resolverSpec {
	fam := cfg.archFamilyKey()
	switch {
	case strings.Contains(fam, "gptneox"):
		return gptNeoXSpec(cfg)
	case strings.Contains(fam, "falcon"):
		return falconSpec(cfg)
	case strings.Contains(fam, "mpt"):
		return mptSpec(cfg)
	case strings.Contains(fam, "stablelm"):
		return stableLMSpec(cfg)
	case strings.Contains(fam, "olmo2"):
		return olmo2Spec(cfg)
	case strings.Contains(fam, "cohere"):
		return cohereSpec(cfg)
	case strings.Contains(fam, "gemma"):
		return gemmaSpec(cfg)
	case strings.Contains(fam, "gptoss"):
		return gptOSSSpec(cfg)
	case strings.Contains(fam, "deepseek"):
		return deepSeekMLASpec(cfg)
	default:
		return identitySpec(cfg)
	}
}

// ---- shared building blocks --------------------------------------------------------

// baseGlobals is the global tensor set every dense family shares under canonical naming:
// the token embedding, the final norm (optional bias), and an optional untied lm_head.
func baseGlobals() []tensorReq {
	return []tensorReq{
		{canonical: "model.embed_tokens.weight"},
		{canonical: "model.norm.weight"},
		{canonical: "model.norm.bias", optional: true},
		{canonical: "lm_head.weight", optional: true},
	}
}

// stdAttnProjections is the standard split q/k/v/o projection set under canonical naming,
// with q/k/v/o bias optional (presence-driven; Qwen2/StableLM carry q/k/v bias, most do
// not). aliases, when non-nil, supply per-projection source names (e.g. a fused parent).
func stdAttnProjections(p string, qAlias, kAlias, vAlias, oAlias []string) []tensorReq {
	return []tensorReq{
		{canonical: p + "self_attn.q_proj.weight", aliases: qAlias},
		{canonical: p + "self_attn.q_proj.bias", optional: true},
		{canonical: p + "self_attn.k_proj.weight", aliases: kAlias},
		{canonical: p + "self_attn.k_proj.bias", optional: true},
		{canonical: p + "self_attn.v_proj.weight", aliases: vAlias},
		{canonical: p + "self_attn.v_proj.bias", optional: true},
		{canonical: p + "self_attn.o_proj.weight", aliases: oAlias},
		{canonical: p + "self_attn.o_proj.bias", optional: true},
	}
}

// swigluMLP is the dense SwiGLU FFN tensor set (gate/up/down) under canonical naming,
// shared by every family that is not a DenseMLP (NeoX/Falcon/MPT) or MoE model.
func swigluMLP(p string) []tensorReq {
	return []tensorReq{
		{canonical: p + "mlp.gate_proj.weight"},
		{canonical: p + "mlp.up_proj.weight"},
		{canonical: p + "mlp.down_proj.weight"},
	}
}

// qkNorm appends the per-head qk-norm reqs (self_attn.{q,k}_norm.weight), which arch.go's
// applyQKNorm reads when Config.QKNorm is on (OLMo2 / Gemma3 / Cohere2 / Qwen3).
func qkNorm(p string, optional bool) []tensorReq {
	return []tensorReq{
		{canonical: p + "self_attn.q_norm.weight", optional: optional},
		{canonical: p + "self_attn.k_norm.weight", optional: optional},
	}
}

// ---- family specs ------------------------------------------------------------------

// identitySpec is the Llama / SmolLM2 / Qwen contract: the canonical names ARE the source
// names, so every req resolves to itself. This mirrors exactly the tensor set NewSynthetic
// builds and the forward pass reads (PreNorm: input_layernorm + post_attention_layernorm,
// split q/k/v/o, SwiGLU gate/up/down).
func identitySpec(cfg Config) resolverSpec {
	return resolverSpec{
		family:  "llama",
		globals: baseGlobals(),
		perLayer: func(l int) []tensorReq {
			p := layerPrefix(l)
			reqs := []tensorReq{{canonical: p + "input_layernorm.weight"}, {canonical: p + "input_layernorm.bias", optional: true}}
			reqs = append(reqs, stdAttnProjections(p, nil, nil, nil, nil)...)
			reqs = append(reqs,
				tensorReq{canonical: p + "post_attention_layernorm.weight"},
				tensorReq{canonical: p + "post_attention_layernorm.bias", optional: true},
			)
			reqs = append(reqs, swigluMLP(p)...)
			return reqs
		},
	}
}

// gemmaSpec covers Gemma 2 and Gemma 3. Gemma keeps HF-standard split q/k/v/o and GeGLU
// gate/up/down (identity names), but its SandwichNorm block (arch.go) reads FOUR norms per
// layer that the Llama set never had: input_layernorm + post_attention_layernorm around
// attention, and pre_feedforward_layernorm + post_feedforward_layernorm around the FFN
// (these exact names are read by mlpNorms in weights.go). Gemma 3 additionally carries
// qk-norms (Config.QKNorm), which Gemma 2 does not.
func gemmaSpec(cfg Config) resolverSpec {
	return resolverSpec{
		family:  "gemma",
		globals: baseGlobals(),
		perLayer: func(l int) []tensorReq {
			p := layerPrefix(l)
			reqs := []tensorReq{{canonical: p + "input_layernorm.weight"}}
			reqs = append(reqs, stdAttnProjections(p, nil, nil, nil, nil)...)
			if cfg.QKNorm {
				reqs = append(reqs, qkNorm(p, false)...)
			}
			reqs = append(reqs,
				tensorReq{canonical: p + "post_attention_layernorm.weight"},
				tensorReq{canonical: p + "pre_feedforward_layernorm.weight"},
				tensorReq{canonical: p + "post_feedforward_layernorm.weight"},
			)
			reqs = append(reqs, swigluMLP(p)...)
			return reqs
		},
	}
}

// olmo2Spec covers OLMo 2: HF-standard split q/k/v/o and SwiGLU, but a POST-norm block
// (arch.go PostNorm, derived in deriveConfigAxes) — so it has NO input_layernorm and NO
// pre_feedforward_layernorm. The norm applied after attention is post_attention_layernorm
// and after the FFN is post_feedforward_layernorm (the names attentionNorms/mlpNorms fall
// back to). OLMo 2 also always carries qk-norms (self_attn.{q,k}_norm.weight).
func olmo2Spec(cfg Config) resolverSpec {
	return resolverSpec{
		family:  "olmo2",
		globals: baseGlobals(),
		perLayer: func(l int) []tensorReq {
			p := layerPrefix(l)
			reqs := stdAttnProjections(p, nil, nil, nil, nil)
			reqs = append(reqs, qkNorm(p, false)...)
			reqs = append(reqs,
				tensorReq{canonical: p + "post_attention_layernorm.weight"},
				tensorReq{canonical: p + "post_feedforward_layernorm.weight"},
			)
			reqs = append(reqs, swigluMLP(p)...)
			return reqs
		},
	}
}

// cohereSpec covers Cohere Command-R / Command-R+ / Command-R7B. Cohere is a
// ParallelResidual block (arch.go) where attention and the FFN read ONE shared
// input_layernorm, so there is a single required norm per layer; a separate
// post_attention_layernorm is optional (parallelMLPNorms falls back to the shared norm
// when it is absent). q/k/v/o and SwiGLU are HF-standard. Command-R7B / Cohere2 add
// qk-norms (Config.QKNorm); the original Command-R does not, so they are required only
// when QKNorm is on. The final LM head is tied (lm_head optional).
func cohereSpec(cfg Config) resolverSpec {
	return resolverSpec{
		family:  "cohere",
		globals: baseGlobals(),
		perLayer: func(l int) []tensorReq {
			p := layerPrefix(l)
			reqs := []tensorReq{{canonical: p + "input_layernorm.weight"}, {canonical: p + "input_layernorm.bias", optional: true}}
			reqs = append(reqs, stdAttnProjections(p, nil, nil, nil, nil)...)
			reqs = append(reqs, qkNorm(p, !cfg.QKNorm)...) // optional unless QKNorm (Cohere2) is on
			reqs = append(reqs,
				tensorReq{canonical: p + "post_attention_layernorm.weight", optional: true},
			)
			reqs = append(reqs, swigluMLP(p)...)
			return reqs
		},
	}
}

// gptNeoXSpec covers GPT-NeoX. The source vocabulary is gpt_neox.*: a FUSED
// attention.query_key_value.weight (materializeGPTNeoXTensors splits it into q/k/v),
// attention.dense -> o_proj, a DENSE MLP (dense_h_to_4h -> gate_proj, dense_4h_to_h ->
// down_proj, NO up_proj), and ParallelResidual with distinct input/post-attention
// LayerNorms that carry bias. The alias source strings here are the SAME strings
// materializeGPTNeoXTensors uses, kept in sync as the declarative contract.
func gptNeoXSpec(cfg Config) resolverSpec {
	return resolverSpec{
		family: "gptneox",
		globals: []tensorReq{
			{canonical: "model.embed_tokens.weight", aliases: []string{"gpt_neox.embed_in.weight"}},
			{canonical: "model.norm.weight", aliases: []string{"gpt_neox.final_layer_norm.weight"}},
			{canonical: "model.norm.bias", aliases: []string{"gpt_neox.final_layer_norm.bias"}, optional: true},
			{canonical: "lm_head.weight", aliases: []string{"embed_out.weight"}, optional: true},
		},
		perLayer: func(l int) []tensorReq {
			p := layerPrefix(l)
			src := "gpt_neox.layers." + itoa(l) + "."
			fusedQKV := []string{src + "attention.query_key_value.weight"}
			return []tensorReq{
				{canonical: p + "input_layernorm.weight", aliases: []string{src + "input_layernorm.weight"}},
				{canonical: p + "input_layernorm.bias", aliases: []string{src + "input_layernorm.bias"}, optional: true},
				{canonical: p + "post_attention_layernorm.weight", aliases: []string{src + "post_attention_layernorm.weight"}},
				{canonical: p + "post_attention_layernorm.bias", aliases: []string{src + "post_attention_layernorm.bias"}, optional: true},
				{canonical: p + "self_attn.q_proj.weight", aliases: fusedQKV},
				{canonical: p + "self_attn.k_proj.weight", aliases: fusedQKV},
				{canonical: p + "self_attn.v_proj.weight", aliases: fusedQKV},
				{canonical: p + "self_attn.o_proj.weight", aliases: []string{src + "attention.dense.weight"}},
				{canonical: p + "self_attn.o_proj.bias", aliases: []string{src + "attention.dense.bias"}, optional: true},
				{canonical: p + "mlp.gate_proj.weight", aliases: []string{src + "mlp.dense_h_to_4h.weight"}},
				{canonical: p + "mlp.down_proj.weight", aliases: []string{src + "mlp.dense_4h_to_h.weight"}},
			}
		},
	}
}

// falconSpec covers Falcon. Source vocabulary is transformer.h.*: a fused
// self_attention.query_key_value.weight (split by materializeFalconTensors),
// self_attention.dense -> o_proj, a dense MLP (dense_h_to_4h/dense_4h_to_h), and a single
// input_layernorm (+bias) per layer. Embedding is transformer.word_embeddings; final norm
// is transformer.ln_f (+bias). Alias strings mirror materializeFalconTensors.
func falconSpec(cfg Config) resolverSpec {
	return resolverSpec{
		family: "falcon",
		globals: []tensorReq{
			{canonical: "model.embed_tokens.weight", aliases: []string{"transformer.word_embeddings.weight"}},
			{canonical: "model.norm.weight", aliases: []string{"transformer.ln_f.weight"}},
			{canonical: "model.norm.bias", aliases: []string{"transformer.ln_f.bias"}, optional: true},
			{canonical: "lm_head.weight", optional: true},
		},
		perLayer: func(l int) []tensorReq {
			p := layerPrefix(l)
			src := "transformer.h." + itoa(l) + "."
			fusedQKV := []string{src + "self_attention.query_key_value.weight"}
			return []tensorReq{
				{canonical: p + "input_layernorm.weight", aliases: []string{src + "input_layernorm.weight"}},
				{canonical: p + "input_layernorm.bias", aliases: []string{src + "input_layernorm.bias"}, optional: true},
				{canonical: p + "self_attn.q_proj.weight", aliases: fusedQKV},
				{canonical: p + "self_attn.k_proj.weight", aliases: fusedQKV},
				{canonical: p + "self_attn.v_proj.weight", aliases: fusedQKV},
				{canonical: p + "self_attn.o_proj.weight", aliases: []string{src + "self_attention.dense.weight"}},
				{canonical: p + "self_attn.o_proj.bias", aliases: []string{src + "self_attention.dense.bias"}, optional: true},
				{canonical: p + "mlp.gate_proj.weight", aliases: []string{src + "mlp.dense_h_to_4h.weight"}},
				{canonical: p + "mlp.down_proj.weight", aliases: []string{src + "mlp.dense_4h_to_h.weight"}},
			}
		},
	}
}

// mptSpec covers MPT. Source vocabulary is transformer.blocks.*: a fused attn.Wqkv.weight
// (split by materializeMPTTensors), attn.out_proj -> o_proj, a dense MLP (ffn.up_proj ->
// gate_proj, ffn.down_proj -> down_proj), and two LayerNorms norm_1/norm_2 -> input /
// post-attention (MPT is no-bias). Embedding is transformer.wte; final norm is
// transformer.norm_f. Alias strings mirror materializeMPTTensors.
func mptSpec(cfg Config) resolverSpec {
	return resolverSpec{
		family: "mpt",
		globals: []tensorReq{
			{canonical: "model.embed_tokens.weight", aliases: []string{"transformer.wte.weight"}},
			{canonical: "model.norm.weight", aliases: []string{"transformer.norm_f.weight"}},
			{canonical: "lm_head.weight", optional: true},
		},
		perLayer: func(l int) []tensorReq {
			p := layerPrefix(l)
			src := "transformer.blocks." + itoa(l) + "."
			fusedQKV := []string{src + "attn.Wqkv.weight"}
			return []tensorReq{
				{canonical: p + "input_layernorm.weight", aliases: []string{src + "norm_1.weight"}},
				{canonical: p + "post_attention_layernorm.weight", aliases: []string{src + "norm_2.weight"}},
				{canonical: p + "self_attn.q_proj.weight", aliases: fusedQKV},
				{canonical: p + "self_attn.k_proj.weight", aliases: fusedQKV},
				{canonical: p + "self_attn.v_proj.weight", aliases: fusedQKV},
				{canonical: p + "self_attn.o_proj.weight", aliases: []string{src + "attn.out_proj.weight"}},
				{canonical: p + "mlp.gate_proj.weight", aliases: []string{src + "ffn.up_proj.weight"}},
				{canonical: p + "mlp.down_proj.weight", aliases: []string{src + "ffn.down_proj.weight"}},
			}
		},
	}
}

// stableLMSpec covers StableLM / StableLM-2. Unlike NeoX/Falcon/MPT, StableLM does NOT
// fuse or rename q/k/v: it uses HF-standard split q/k/v/o and SwiGLU, with optional q/k/v
// projection bias and a LayerNorm-with-bias block (input_layernorm + post_attention, both
// optionally biased). Its distinguishing axes — LayerNorm vs RMSNorm and partial RoPE —
// are config axes (deriveConfigAxes / PartialRotaryFactor), not tensor renames, so at the
// NAME level the required set is the canonical identity set. (The StableLM-2 per-head
// qk-LayerNorm naming variant, self_attn.{q,k}_layernorm.*, is NOT asserted here: it needs
// a real manifest to pin and is left to the real-checkpoint smoke.)
func stableLMSpec(cfg Config) resolverSpec {
	spec := identitySpec(cfg)
	spec.family = "stablelm"
	return spec
}

// gptOSSSpec — SCAFFOLD, dependency-gated by #24 (MXFP4 format). gpt-oss is a MoE model
// (router + per-expert gate_up/down, attention sinks) whose real weights ship in the MXFP4
// 4-bit block format that #24 has not yet exposed to the loader. The f32 expert-split path
// exists (materializeGPTOSSTensors), but proving a real gpt-oss manifest resolves requires
// the MXFP4 names + dtype #24 will introduce. The required-tensor table is intentionally
// left minimal (the canonical globals only) so this entry never claims unverified MXFP4
// names; the family-completeness proof is the t.Skip("#24") test in the suite.
func gptOSSSpec(cfg Config) resolverSpec {
	return resolverSpec{
		family:  "gptoss",
		globals: baseGlobals(),
		// perLayer deliberately nil: gpt-oss MoE expert + attention-sink names are gated by
		// #24 (MXFP4). See TestResolverGPTOSSScaffold_Skip.
	}
}

// deepSeekMLASpec — SCAFFOLD, dependency-gated by #25 (real MLA layout). DeepSeek V2/V3 use
// Multi-head Latent Attention: low-rank q_a/q_b and kv_a/kv_b projections plus
// q_a_layernorm / kv_a_layernorm, with a latent KV cache — a different attention tensor
// shape entirely (the MLAConfig fields in weights.go are exported/audited but #25 has not
// moved the runtime past the synthetic MLA layout). The MLA projection names are NOT
// asserted here: pinning them needs a real DeepSeek manifest this task does not have, and a
// wrong canonical mapping would silently mis-load. The completeness proof is the
// t.Skip("#25") test.
func deepSeekMLASpec(cfg Config) resolverSpec {
	return resolverSpec{
		family:  "deepseek-mla",
		globals: baseGlobals(),
		// perLayer deliberately nil: MLA q_a/q_b/kv_a/kv_b projection names are gated by #25.
		// See TestResolverDeepSeekMLAScaffold_Skip.
	}
}
