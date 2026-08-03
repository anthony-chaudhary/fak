package model

import "strings"

// gemma4_synthetic.go — the in-memory gemma4 checkpoint the SERVING-path witnesses need.
//
// Why a dedicated constructor. NewSynthetic (synthetic.go) sizes every layer's q/k/v/o
// from the SCALAR cfg.HeadDim / cfg.NumKVHeads, which is exactly the assumption gemma4
// breaks: local (sliding) layers carry a small head_dim with several kv heads, global
// (full-attention) layers a large head_dim with a single kv head whose k_proj output also
// serves as V (no v_proj tensor at all). A synthetic model built on the scalar geometry
// would therefore mis-shape every projection the dedicated forward reads, so it cannot
// stand in for a gemma4 checkpoint.
//
// What it is for. The gemma4 numerics are already witnessed against a faithful GGUF in
// internal/ggufload (gemma4_test.go / gemma4_session_test.go). What that fixture cannot
// reach is the SERVE wiring: internal/agent's InKernelPlanner and its radix prefix-reuse
// path live in a package the ggufload test helpers are not importable from. This is the
// no-file, no-download gemma4 model those planner-level witnesses run on — the same
// role NewSyntheticGLMDsa plays for the GLM-MoE-DSA planner arms in
// internal/agent/inkernel_reuse_test.go.
//
// Faithfulness. The tensor set mirrors internal/ggufload's tinyGemma4GGUF: per-layer
// head_dim / kv-head counts, q/k norms of per-head width, no v_proj on the global layer,
// sandwich-norm tensors on both sub-layers, a per-layer output scale, and the shared
// rope_freqs vector the global layers divide their frequencies by. The weights are
// meaningless (a fixed-seed LCG, as in every NewSynthetic*); the SHAPES and the tensor
// NAMES are what make the dedicated forward run its real instruction stream.
//
// cfg must already be a gemma4 Config — Config.isGemma4() true, with LayerTypes,
// HeadDimPerLayer and NumKVHeadsPerLayer populated exactly as ggufload's
// applyGemma4Config would build them from the checkpoint metadata.
func NewSyntheticGemma4(cfg Config) *Model {
	if !cfg.isGemma4() {
		panic("model: NewSyntheticGemma4 needs a gemma4 Config (ModelType/Architectures)")
	}
	H, I, V, nH := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize, cfg.NumHeads

	type ts = synthTensor
	tensors := []ts{{"model.embed_tokens.weight", []int{V, H}}}

	// The global (full-attention) layers divide each RoPE frequency by this shared
	// per-frequency factor; its width is half the widest rotary span in the model.
	ropeFreqHalf := 0
	for l := 0; l < cfg.NumLayers; l++ {
		if h := cfg.ropeDimForLayer(l) / 2; h > ropeFreqHalf {
			ropeFreqHalf = h
		}
	}
	if ropeFreqHalf > 0 {
		tensors = append(tensors, ts{"model.rope_freqs.weight", []int{ropeFreqHalf}})
	}

	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		hd := cfg.headDimForLayer(l)
		nKV := cfg.numKVHeadsForLayer(l)
		tensors = append(tensors,
			ts{p + "input_layernorm.weight", []int{H}},
			ts{p + "self_attn.q_proj.weight", []int{nH * hd, H}},
			ts{p + "self_attn.k_proj.weight", []int{nKV * hd, H}},
		)
		if cfg.gemma4LayerIsSliding(l) {
			// Global layers carry no v_proj: V is the raw k_proj output (gemma4.go).
			tensors = append(tensors, ts{p + "self_attn.v_proj.weight", []int{nKV * hd, H}})
		}
		tensors = append(tensors,
			ts{p + "self_attn.o_proj.weight", []int{H, nH * hd}},
			// Per-HEAD q/k norm width (the baked-(1+w) GGUF gain), not the projection width.
			ts{p + "self_attn.q_norm.weight", []int{hd}},
			ts{p + "self_attn.k_norm.weight", []int{hd}},
			ts{p + "post_attention_layernorm.weight", []int{H}},
			ts{p + "pre_feedforward_layernorm.weight", []int{H}},
			ts{p + "post_feedforward_layernorm.weight", []int{H}},
			ts{p + "mlp.gate_proj.weight", []int{I, H}},
			ts{p + "mlp.up_proj.weight", []int{I, H}},
			ts{p + "mlp.down_proj.weight", []int{H, I}},
			ts{p + "layer_output_scale.weight", []int{1}},
		)
	}
	tensors = append(tensors, ts{"model.norm.weight", []int{H}})

	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		if gemma4SyntheticUnitTensor(name) {
			return 1.0
		}
		return synthMatmulFill(name, next)
	})

	cfg.TieWordEmbeddings = true // synthetic head is tied to the embedding
	return &Model{Cfg: cfg, manifest: man, raw: raw}
}

// gemma4SyntheticUnitTensor reports the tensors a synthetic gemma4 checkpoint fills with
// exactly 1.0 rather than an LCG draw: every norm gain (so each RMSNorm is
// well-conditioned), the per-layer block output scale (so it is a true identity rather
// than a random amplifier compounding over layers), and the rope_freqs factor (so the
// global layers' proportional-rope division is exercised without perturbing the table).
func gemma4SyntheticUnitTensor(name string) bool {
	switch {
	case strings.HasSuffix(name, "layernorm.weight"),
		strings.HasSuffix(name, "q_norm.weight"),
		strings.HasSuffix(name, "k_norm.weight"),
		strings.HasSuffix(name, "layer_output_scale.weight"),
		name == "model.norm.weight",
		name == "model.rope_freqs.weight":
		return true
	}
	return false
}
