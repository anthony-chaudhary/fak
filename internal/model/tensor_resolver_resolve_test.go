package model

// Fixture tests for the family-aware tensor-name resolver (issue #473).
//
// These prove the resolver contract directly: given a checkpoint's (model_type,
// architectures) + a SYNTHETIC tensor manifest, ResolveTensorNames maps every REQUIRED
// canonical tensor onto a source name present in the manifest, OR fails with a precise
// error naming the family, the missing canonical tensor, and the source names searched.
// Manifests are presence-only (the resolver inspects manifest KEYS, never bytes), so each
// fixture is a tiny set of names — no weights, no real checkpoint.
//
// The Llama/SmolLM2/Qwen no-op is proven by TestResolverLlamaIdentityNoOp: every required
// canonical tensor resolves to ITSELF (zero renames), which is the load-path-unchanged
// guarantee at the name level (the numeric no-op is TestArchLlamaNoOp). gpt-oss (#24) and
// DeepSeek MLA (#25) are dependency-gated scaffolds whose per-family completeness proof is a
// t.Skip naming the missing artifact — see the bottom of this file.

import (
	"os"
	"strings"
	"testing"
)

// manifestKeys builds a presence-only manifest from a list of tensor names. The tensorMeta
// value is irrelevant to ResolveTensorNames, which checks key presence only.
func manifestKeys(names ...string) map[string]tensorMeta {
	man := make(map[string]tensorMeta, len(names))
	for _, n := range names {
		man[n] = tensorMeta{Dtype: "F32"}
	}
	return man
}

// assertResolvesTo runs the resolver and asserts each canonical->source pair holds. It also
// asserts the resolved family label matches.
func assertResolvesTo(t *testing.T, cfg Config, man map[string]tensorMeta, wantFamily string, pairs map[string]string) *Resolution {
	t.Helper()
	res, err := ResolveTensorNames(cfg, man)
	if err != nil {
		t.Fatalf("ResolveTensorNames(%s): unexpected error: %v", wantFamily, err)
	}
	if res.Family != wantFamily {
		t.Fatalf("family = %q, want %q", res.Family, wantFamily)
	}
	for canonical, wantSrc := range pairs {
		if got := res.SourceFor(canonical); got != wantSrc {
			t.Errorf("%s: canonical %q resolved to %q, want %q", wantFamily, canonical, got, wantSrc)
		}
	}
	return res
}

// assertResolveError asserts the resolver fails and the error mentions every wanted
// substring (family label, the missing canonical tensor, and the "searched:" trailer).
func assertResolveError(t *testing.T, cfg Config, man map[string]tensorMeta, wantSubstrs ...string) {
	t.Helper()
	res, err := ResolveTensorNames(cfg, man)
	if err == nil {
		t.Fatalf("expected resolve error, got nil (resolved family %q)", res.Family)
	}
	for _, sub := range wantSubstrs {
		if !strings.Contains(err.Error(), sub) {
			t.Errorf("error %q does not contain %q", err.Error(), sub)
		}
	}
}

// ---- Llama / SmolLM2 / Qwen identity no-op -----------------------------------------

// TestResolverLlamaIdentityNoOp proves the load path is unchanged at the NAME level for
// Llama/SmolLM2: a synthetic Llama manifest (built by NewSynthetic, the same layout Load
// produces) resolves every required canonical tensor to ITSELF, with zero renames.
func TestResolverLlamaIdentityNoOp(t *testing.T) {
	cfg := Config{
		HiddenSize: 8, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 2,
		IntermediateSize: 16, VocabSize: 32, ModelType: "llama",
	}
	m := NewSynthetic(cfg)
	res, err := ResolveTensorNames(m.Cfg, m.manifest)
	if err != nil {
		t.Fatalf("Llama synthetic manifest must resolve: %v", err)
	}
	if res.Family != "llama" {
		t.Fatalf("family = %q, want llama", res.Family)
	}
	if len(res.Resolved) == 0 {
		t.Fatal("no tensors resolved")
	}
	for canonical, src := range res.Resolved {
		if src != canonical {
			t.Errorf("identity broken: canonical %q resolved to source %q (expected identity)", canonical, src)
		}
	}
	// Spot-check the required canonical tensors appear (identity) across both layers.
	for _, want := range []string{
		"model.embed_tokens.weight",
		"model.norm.weight",
		"model.layers.0.input_layernorm.weight",
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.self_attn.o_proj.weight",
		"model.layers.0.post_attention_layernorm.weight",
		"model.layers.1.mlp.gate_proj.weight",
		"model.layers.1.mlp.up_proj.weight",
		"model.layers.1.mlp.down_proj.weight",
	} {
		if res.SourceFor(want) != want {
			t.Errorf("required canonical tensor %q not resolved to itself (got %q)", want, res.SourceFor(want))
		}
	}
}

// TestResolverQwenRoutesToIdentity confirms a Qwen checkpoint (HF-standard names) routes to
// the identity (no-op) spec, like Llama/SmolLM2 — it carries q/k/v projection bias, which
// resolves via the optional bias reqs.
func TestResolverQwenRoutesToIdentity(t *testing.T) {
	p := layerPrefix(0)
	man := manifestKeys(
		"model.embed_tokens.weight", "model.norm.weight",
		p+"input_layernorm.weight",
		p+"self_attn.q_proj.weight", p+"self_attn.q_proj.bias",
		p+"self_attn.k_proj.weight", p+"self_attn.k_proj.bias",
		p+"self_attn.v_proj.weight", p+"self_attn.v_proj.bias",
		p+"self_attn.o_proj.weight",
		p+"post_attention_layernorm.weight",
		p+"mlp.gate_proj.weight", p+"mlp.up_proj.weight", p+"mlp.down_proj.weight",
	)
	cfg := Config{NumLayers: 1, ModelType: "qwen2"}
	assertResolvesTo(t, cfg, man, "llama", map[string]string{
		p + "self_attn.q_proj.weight": p + "self_attn.q_proj.weight",
		p + "self_attn.q_proj.bias":   p + "self_attn.q_proj.bias", // optional bias resolves to itself
	})
}

// ---- GPT-NeoX / Falcon / MPT / StableLM (fused or renamed q/k/v) --------------------

// TestResolverGPTNeoX proves the gpt_neox.* source vocabulary resolves: a fused
// attention.query_key_value satisfies all three of q/k/v, attention.dense -> o_proj, the
// dense MLP renames, the embed/final-norm aliases, and the embed_out -> lm_head untie.
func TestResolverGPTNeoX(t *testing.T) {
	p := layerPrefix(0)
	src := "gpt_neox.layers.0."
	fused := src + "attention.query_key_value.weight"
	man := manifestKeys(
		"gpt_neox.embed_in.weight",
		"gpt_neox.final_layer_norm.weight",
		"gpt_neox.final_layer_norm.bias",
		"embed_out.weight",
		src+"input_layernorm.weight", src+"input_layernorm.bias",
		src+"post_attention_layernorm.weight", src+"post_attention_layernorm.bias",
		fused,
		src+"attention.dense.weight", src+"attention.dense.bias",
		src+"mlp.dense_h_to_4h.weight",
		src+"mlp.dense_4h_to_h.weight",
	)
	cfg := Config{NumLayers: 1, ModelType: "gpt_neox"}
	assertResolvesTo(t, cfg, man, "gptneox", map[string]string{
		"model.embed_tokens.weight":   "gpt_neox.embed_in.weight",
		"model.norm.weight":           "gpt_neox.final_layer_norm.weight",
		"model.norm.bias":             "gpt_neox.final_layer_norm.bias",
		"lm_head.weight":              "embed_out.weight",
		p + "input_layernorm.weight":  src + "input_layernorm.weight",
		p + "self_attn.q_proj.weight": fused,
		p + "self_attn.k_proj.weight": fused,
		p + "self_attn.v_proj.weight": fused,
		p + "self_attn.o_proj.weight": src + "attention.dense.weight",
		p + "mlp.gate_proj.weight":    src + "mlp.dense_h_to_4h.weight",
		p + "mlp.down_proj.weight":    src + "mlp.dense_4h_to_h.weight",
	})
}

// TestResolverGPTNeoXMissingQKVIsPreciseError proves a NeoX manifest missing the fused
// query_key_value fails naming both the canonical tensor AND the fused source searched.
func TestResolverGPTNeoXMissingQKVIsPreciseError(t *testing.T) {
	src := "gpt_neox.layers.0."
	man := manifestKeys(
		"gpt_neox.embed_in.weight",
		"gpt_neox.final_layer_norm.weight",
		src+"input_layernorm.weight",
		src+"post_attention_layernorm.weight",
		// fused query_key_value deliberately absent
		src+"attention.dense.weight",
		src+"mlp.dense_h_to_4h.weight",
		src+"mlp.dense_4h_to_h.weight",
	)
	cfg := Config{NumLayers: 1, ModelType: "gpt_neox"}
	assertResolveError(t, cfg, man,
		"gptneox family",
		"model.layers.0.self_attn.q_proj.weight",
		src+"attention.query_key_value.weight",
		"searched:",
	)
}

// TestResolverFalcon proves the transformer.h.* source vocabulary resolves: fused
// query_key_value -> q/k/v, self_attention.dense -> o_proj, dense MLP, word_embeddings and
// ln_f aliases (with final-norm bias).
func TestResolverFalcon(t *testing.T) {
	p := layerPrefix(0)
	src := "transformer.h.0."
	fused := src + "self_attention.query_key_value.weight"
	man := manifestKeys(
		"transformer.word_embeddings.weight",
		"transformer.ln_f.weight", "transformer.ln_f.bias",
		src+"input_layernorm.weight", src+"input_layernorm.bias",
		fused,
		src+"self_attention.dense.weight",
		src+"mlp.dense_h_to_4h.weight",
		src+"mlp.dense_4h_to_h.weight",
	)
	cfg := Config{NumLayers: 1, ModelType: "falcon"}
	assertResolvesTo(t, cfg, man, "falcon", map[string]string{
		"model.embed_tokens.weight":   "transformer.word_embeddings.weight",
		"model.norm.weight":           "transformer.ln_f.weight",
		"model.norm.bias":             "transformer.ln_f.bias",
		p + "input_layernorm.weight":  src + "input_layernorm.weight",
		p + "self_attn.q_proj.weight": fused,
		p + "self_attn.k_proj.weight": fused,
		p + "self_attn.v_proj.weight": fused,
		p + "self_attn.o_proj.weight": src + "self_attention.dense.weight",
		p + "mlp.gate_proj.weight":    src + "mlp.dense_h_to_4h.weight",
		p + "mlp.down_proj.weight":    src + "mlp.dense_4h_to_h.weight",
	})
}

// TestResolverMPT proves the transformer.blocks.* source vocabulary resolves: fused
// attn.Wqkv -> q/k/v, attn.out_proj -> o_proj, ffn.up_proj/down_proj -> gate/down, the
// norm_1/norm_2 -> input/post-attention norms, and wte/norm_f globals (MPT is no-bias).
func TestResolverMPT(t *testing.T) {
	p := layerPrefix(0)
	src := "transformer.blocks.0."
	fused := src + "attn.Wqkv.weight"
	man := manifestKeys(
		"transformer.wte.weight",
		"transformer.norm_f.weight",
		src+"norm_1.weight",
		src+"norm_2.weight",
		fused,
		src+"attn.out_proj.weight",
		src+"ffn.up_proj.weight",
		src+"ffn.down_proj.weight",
	)
	cfg := Config{NumLayers: 1, ModelType: "mpt"}
	assertResolvesTo(t, cfg, man, "mpt", map[string]string{
		"model.embed_tokens.weight":           "transformer.wte.weight",
		"model.norm.weight":                   "transformer.norm_f.weight",
		p + "input_layernorm.weight":          src + "norm_1.weight",
		p + "post_attention_layernorm.weight": src + "norm_2.weight",
		p + "self_attn.q_proj.weight":         fused,
		p + "self_attn.k_proj.weight":         fused,
		p + "self_attn.v_proj.weight":         fused,
		p + "self_attn.o_proj.weight":         src + "attn.out_proj.weight",
		p + "mlp.gate_proj.weight":            src + "ffn.up_proj.weight",
		p + "mlp.down_proj.weight":            src + "ffn.down_proj.weight",
	})
}

// TestResolverStableLM proves StableLM uses the canonical identity name set (it does NOT
// fuse or rename q/k/v — its LayerNorm-vs-RMSNorm and partial-RoPE axes are config, not
// tensor renames). Its optional q/k/v projection bias resolves when present.
func TestResolverStableLM(t *testing.T) {
	p := layerPrefix(0)
	man := manifestKeys(
		"model.embed_tokens.weight", "model.norm.weight", "model.norm.bias",
		p+"input_layernorm.weight", p+"input_layernorm.bias",
		p+"self_attn.q_proj.weight", p+"self_attn.q_proj.bias",
		p+"self_attn.k_proj.weight", p+"self_attn.k_proj.bias",
		p+"self_attn.v_proj.weight", p+"self_attn.v_proj.bias",
		p+"self_attn.o_proj.weight",
		p+"post_attention_layernorm.weight", p+"post_attention_layernorm.bias",
		p+"mlp.gate_proj.weight", p+"mlp.up_proj.weight", p+"mlp.down_proj.weight",
	)
	cfg := Config{NumLayers: 1, ModelType: "stablelm"}
	assertResolvesTo(t, cfg, man, "stablelm", map[string]string{
		p + "self_attn.q_proj.weight": p + "self_attn.q_proj.weight",
		p + "self_attn.q_proj.bias":   p + "self_attn.q_proj.bias",
		p + "input_layernorm.bias":    p + "input_layernorm.bias",
		"model.norm.bias":             "model.norm.bias",
	})
	// Dropping a required projection is a precise error naming the stablelm family.
	delete(man, p+"self_attn.v_proj.weight")
	assertResolveError(t, cfg, man, "stablelm family", p+"self_attn.v_proj.weight", "searched:")
}

// ---- OLMo2 (post-norm + qk-norm) ---------------------------------------------------

// olmo2Manifest builds a complete single-layer OLMo2 manifest (post-norm block: no
// input_layernorm and no pre_feedforward_layernorm; qk-norm always present).
func olmo2Manifest() map[string]tensorMeta {
	p := layerPrefix(0)
	return manifestKeys(
		"model.embed_tokens.weight", "model.norm.weight",
		p+"self_attn.q_proj.weight", p+"self_attn.k_proj.weight",
		p+"self_attn.v_proj.weight", p+"self_attn.o_proj.weight",
		p+"self_attn.q_norm.weight", p+"self_attn.k_norm.weight",
		p+"post_attention_layernorm.weight", p+"post_feedforward_layernorm.weight",
		p+"mlp.gate_proj.weight", p+"mlp.up_proj.weight", p+"mlp.down_proj.weight",
	)
}

func TestResolverOLMo2(t *testing.T) {
	p := layerPrefix(0)
	cfg := Config{NumLayers: 1, ModelType: "olmo2"}
	assertResolvesTo(t, cfg, olmo2Manifest(), "olmo2", map[string]string{
		p + "self_attn.q_norm.weight":           p + "self_attn.q_norm.weight",
		p + "self_attn.k_norm.weight":           p + "self_attn.k_norm.weight",
		p + "post_attention_layernorm.weight":   p + "post_attention_layernorm.weight",
		p + "post_feedforward_layernorm.weight": p + "post_feedforward_layernorm.weight",
	})
}

// TestResolverOLMo2MissingQKNormIsPreciseError proves OLMo2's qk-norm is REQUIRED: dropping
// q_norm fails naming the olmo2 family and the missing canonical tensor.
func TestResolverOLMo2MissingQKNormIsPreciseError(t *testing.T) {
	p := layerPrefix(0)
	man := olmo2Manifest()
	delete(man, p+"self_attn.q_norm.weight")
	cfg := Config{NumLayers: 1, ModelType: "olmo2"}
	assertResolveError(t, cfg, man, "olmo2 family", p+"self_attn.q_norm.weight", "searched:")
}

// TestResolverOLMo2MissingPostFFNNormIsPreciseError proves the post-FFN norm is REQUIRED for
// the OLMo2 post-norm block.
func TestResolverOLMo2MissingPostFFNNormIsPreciseError(t *testing.T) {
	p := layerPrefix(0)
	man := olmo2Manifest()
	delete(man, p+"post_feedforward_layernorm.weight")
	cfg := Config{NumLayers: 1, ModelType: "olmo2"}
	assertResolveError(t, cfg, man, "olmo2 family", p+"post_feedforward_layernorm.weight")
}

// ---- Gemma 2 / Gemma 3 (sandwich norms; Gemma3 adds qk-norm) -----------------------

// gemmaManifest builds a single-layer Gemma manifest. withQKNorm adds the Gemma3 per-head
// qk-norms. Gemma's SandwichNorm reads FOUR norms per layer that the Llama set never had.
func gemmaManifest(withQKNorm bool) map[string]tensorMeta {
	p := layerPrefix(0)
	names := []string{
		"model.embed_tokens.weight", "model.norm.weight",
		p + "input_layernorm.weight",
		p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
		p + "self_attn.v_proj.weight", p + "self_attn.o_proj.weight",
		p + "post_attention_layernorm.weight",
		p + "pre_feedforward_layernorm.weight",
		p + "post_feedforward_layernorm.weight",
		p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight", p + "mlp.down_proj.weight",
	}
	if withQKNorm {
		names = append(names, p+"self_attn.q_norm.weight", p+"self_attn.k_norm.weight")
	}
	return manifestKeys(names...)
}

// TestResolverGemma2 proves a Gemma 2 manifest (no qk-norm) resolves all four sandwich
// norms, and that dropping the pre-feedforward norm is a precise error.
func TestResolverGemma2(t *testing.T) {
	p := layerPrefix(0)
	cfg := Config{NumLayers: 1, ModelType: "gemma2", QKNorm: false}
	assertResolvesTo(t, cfg, gemmaManifest(false), "gemma", map[string]string{
		p + "input_layernorm.weight":            p + "input_layernorm.weight",
		p + "post_attention_layernorm.weight":   p + "post_attention_layernorm.weight",
		p + "pre_feedforward_layernorm.weight":  p + "pre_feedforward_layernorm.weight",
		p + "post_feedforward_layernorm.weight": p + "post_feedforward_layernorm.weight",
	})
	man := gemmaManifest(false)
	delete(man, p+"pre_feedforward_layernorm.weight")
	assertResolveError(t, cfg, man, "gemma family", p+"pre_feedforward_layernorm.weight", "searched:")
}

// TestResolverGemma3QKNorm proves Gemma 3 (QKNorm on) REQUIRES the per-head qk-norms in
// addition to the four sandwich norms: dropping k_norm is a precise error.
func TestResolverGemma3QKNorm(t *testing.T) {
	p := layerPrefix(0)
	cfg := Config{NumLayers: 1, ModelType: "gemma3", QKNorm: true}
	assertResolvesTo(t, cfg, gemmaManifest(true), "gemma", map[string]string{
		p + "self_attn.q_norm.weight": p + "self_attn.q_norm.weight",
		p + "self_attn.k_norm.weight": p + "self_attn.k_norm.weight",
	})
	man := gemmaManifest(true)
	delete(man, p+"self_attn.k_norm.weight")
	assertResolveError(t, cfg, man, "gemma family", p+"self_attn.k_norm.weight")
}

// ---- Cohere Command-R / R+ / R7B (parallel residual; R7B adds qk-norm) -------------

// cohereManifest builds a single-layer Cohere manifest. Cohere's ParallelResidual block
// reads ONE shared input_layernorm for both attention and FFN, so post_attention_layernorm
// is optional. withQKNorm adds the Command-R7B / Cohere2 per-head qk-norms.
func cohereManifest(withQKNorm bool) map[string]tensorMeta {
	p := layerPrefix(0)
	names := []string{
		"model.embed_tokens.weight", "model.norm.weight",
		p + "input_layernorm.weight",
		p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight",
		p + "self_attn.v_proj.weight", p + "self_attn.o_proj.weight",
		p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight", p + "mlp.down_proj.weight",
	}
	if withQKNorm {
		names = append(names, p+"self_attn.q_norm.weight", p+"self_attn.k_norm.weight")
	}
	return manifestKeys(names...)
}

// TestResolverCohereSharedNorm proves the original Command-R / R+ (no qk-norm) resolves with
// a SINGLE shared input_layernorm and no separate post_attention_layernorm (optional) and no
// qk-norm (optional when QKNorm is off).
func TestResolverCohereSharedNorm(t *testing.T) {
	p := layerPrefix(0)
	cfg := Config{NumLayers: 1, ModelType: "cohere", QKNorm: false}
	res := assertResolvesTo(t, cfg, cohereManifest(false), "cohere", map[string]string{
		p + "input_layernorm.weight": p + "input_layernorm.weight",
	})
	// post_attention_layernorm and qk-norm are optional here and were absent: they must NOT
	// appear in the resolution.
	for _, absent := range []string{
		p + "post_attention_layernorm.weight",
		p + "self_attn.q_norm.weight",
		p + "self_attn.k_norm.weight",
	} {
		if got := res.SourceFor(absent); got != "" {
			t.Errorf("optional absent tensor %q unexpectedly resolved to %q", absent, got)
		}
	}
}

// TestResolverCohere2QKNormRequired proves Command-R7B / Cohere2 (QKNorm on) REQUIRES the
// per-head qk-norms: absent -> precise error; present -> resolves.
func TestResolverCohere2QKNormRequired(t *testing.T) {
	p := layerPrefix(0)
	cfg := Config{NumLayers: 1, ModelType: "cohere2", QKNorm: true}
	// Missing qk-norm is a precise error.
	assertResolveError(t, cfg, cohereManifest(false), "cohere family", p+"self_attn.q_norm.weight", "searched:")
	// With qk-norm present it resolves.
	assertResolvesTo(t, cfg, cohereManifest(true), "cohere", map[string]string{
		p + "self_attn.q_norm.weight": p + "self_attn.q_norm.weight",
		p + "self_attn.k_norm.weight": p + "self_attn.k_norm.weight",
	})
}

// ---- generic precise-error contract -------------------------------------------------

// TestResolverPreciseErrorNamesFamilyAndTensor proves the precise-error contract on the
// identity path: a Llama manifest missing one required projection fails with a message
// naming the family, the exact missing canonical tensor, and the searched source list.
func TestResolverPreciseErrorNamesFamilyAndTensor(t *testing.T) {
	p := layerPrefix(0)
	man := manifestKeys(
		"model.embed_tokens.weight", "model.norm.weight",
		p+"input_layernorm.weight",
		p+"self_attn.q_proj.weight", p+"self_attn.k_proj.weight",
		// v_proj deliberately absent
		p+"self_attn.o_proj.weight",
		p+"post_attention_layernorm.weight",
		p+"mlp.gate_proj.weight", p+"mlp.up_proj.weight", p+"mlp.down_proj.weight",
	)
	cfg := Config{NumLayers: 1, ModelType: "llama"}
	assertResolveError(t, cfg, man,
		"llama family",
		p+"self_attn.v_proj.weight",
		"searched:",
		p+"self_attn.v_proj.weight", // the searched candidate is the canonical name itself
	)
}

// ---- dependency-gated scaffolds (#24 gpt-oss MXFP4, #25 DeepSeek MLA) ---------------

// TestResolverGPTOSSScaffold_Skip proves gpt-oss routes to its (globals-only) scaffold spec,
// then skips the per-layer completeness proof: gpt-oss MoE expert + attention-sink weights
// ship in the MXFP4 4-bit block format that #24 has not yet exposed, and no real gpt-oss
// checkpoint is available to pin those names without risking a wrong canonical mapping.
func TestResolverGPTOSSScaffold_Skip(t *testing.T) {
	res, err := ResolveTensorNames(
		Config{NumLayers: 0, ModelType: "gpt_oss"},
		manifestKeys("model.embed_tokens.weight", "model.norm.weight"),
	)
	if err != nil {
		t.Fatalf("gpt-oss scaffold globals must resolve: %v", err)
	}
	if res.Family != "gptoss" {
		t.Fatalf("family = %q, want gptoss", res.Family)
	}
	t.Skip("#24: gpt-oss MoE expert + attention-sink tensors ship in the MXFP4 4-bit block " +
		"format #24 has not yet exposed to the loader; the per-family required-tensor table is " +
		"deferred to a real gpt-oss artifact rather than invented from memory.")
}

// TestResolverDeepSeekMLAScaffold_Skip proves DeepSeek routes to its (globals-only) scaffold
// spec, then skips the per-layer completeness proof: MLA's low-rank q_a/q_b/kv_a/kv_b
// projections + latent KV cache are a different attention tensor shape entirely, gated by #25
// (which has not moved the runtime past the synthetic MLA layout), and pinning their names
// needs a real DeepSeek manifest this task does not have.
func TestResolverDeepSeekMLAScaffold_Skip(t *testing.T) {
	res, err := ResolveTensorNames(
		Config{NumLayers: 0, ModelType: "deepseek_v3"},
		manifestKeys("model.embed_tokens.weight", "model.norm.weight"),
	)
	if err != nil {
		t.Fatalf("deepseek scaffold globals must resolve: %v", err)
	}
	if res.Family != "deepseek-mla" {
		t.Fatalf("family = %q, want deepseek-mla", res.Family)
	}
	t.Skip("#25: DeepSeek MLA q_a/q_b/kv_a/kv_b projection + latent-cache tensor names are " +
		"gated by #25 (runtime still on the synthetic MLA layout); they are deferred to a real " +
		"DeepSeek artifact rather than invented from memory, since a wrong canonical mapping " +
		"would silently mis-load weights.")
}

// ---- real-checkpoint smoke (skip when no artifact is present) -----------------------

// TestResolverRealCheckpointSmoke_SkipWhenAbsent runs ResolveTensorNames against a REAL
// fak-exported checkpoint (config.json + manifest.json + weights.f32, the layout Load reads)
// when FAK_RESOLVER_CHECKPOINT_DIR points at one; otherwise it skips. This is the
// "when artifacts are available" smoke the acceptance criteria asks for, wired so CI without
// model weights stays green.
func TestResolverRealCheckpointSmoke_SkipWhenAbsent(t *testing.T) {
	dir := os.Getenv("FAK_RESOLVER_CHECKPOINT_DIR")
	if dir == "" {
		t.Skip("set FAK_RESOLVER_CHECKPOINT_DIR to a fak-exported checkpoint dir " +
			"(config.json + manifest.json + weights.f32) to run the real-checkpoint resolver smoke")
	}
	m, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%q): %v", dir, err)
	}
	res, err := ResolveTensorNames(m.Cfg, m.manifest)
	if err != nil {
		t.Fatalf("ResolveTensorNames on real checkpoint %q: %v", dir, err)
	}
	t.Logf("resolved %d required canonical tensors for family %q from %s", len(res.Resolved), res.Family, dir)
}
