package model

import (
	"encoding/binary"
	"math"
	"strings"
)

// NewSynthetic builds an in-memory Model with deterministic pseudo-random weights
// for an arbitrary (small) Config — no files, no torch, no 538MB HF export.
//
// It exists because the cache mechanics this package proves — append, evict,
// re-RoPE, renumber — are correct for ANY weights: the property "an evicted span
// leaves the cache byte-identical to a run that never saw it" is structural, not
// numeric. So a tiny synthetic checkpoint exercises the KV-quarantine *wiring*
// (e.g. internal/kvmmu) deterministically on a CI box that has no model weights,
// while the oracle test (internal/model/oracle_test.go) separately proves the
// *numerics* equal HuggingFace on the real SmolLM2/Qwen export.
//
// This is NOT a real checkpoint: the logits are meaningless. What is faithful is
// the data layout (the same flat-f32 manifest+raw that Load produces, read by the
// same m.tensor() views), so a Session built on it runs the KV-CACHE path —
// Prefill / token / Step / Evict — through the identical code the HF-verified
// model uses. (The cacheless full-prefill Forward reads the same layout but is
// not the path the KV-quarantine bridge or its witness exercise.)
//
// Norm weights are set to exactly 1.0 (so RMSNorm is well-conditioned); all other
// tensors are filled in [-scale, scale] from a fixed-seed LCG, so distinct token
// ids produce distinct hidden states (a poisoned span genuinely perturbs decode —
// which is what makes a KV-eviction witness non-vacuous).
func NewSynthetic(cfg Config) *Model {
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize

	type ts struct {
		name  string
		shape []int
	}
	var tensors []ts
	tensors = append(tensors, ts{"model.embed_tokens.weight", []int{V, H}})
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		qRows := nH * hd
		if cfg.AttnOutputGate && !cfg.isLinearAttnLayer(l) {
			qRows *= 2
		}
		tensors = append(tensors,
			ts{p + "input_layernorm.weight", []int{H}},
		)
		if cfg.isLinearAttnLayer(l) {
			nK := cfg.LinearNumKeyHeads
			nV := cfg.LinearNumValueHeads
			kHd := cfg.LinearKeyHeadDim
			vHd := cfg.LinearValueHeadDim
			keyDim := nK * kHd
			valDim := nV * vHd
			convDim := 2*keyDim + valDim
			K := cfg.LinearConvKernelDim
			tensors = append(tensors,
				ts{p + "linear_attn.in_proj_qkv.weight", []int{convDim, H}},
				ts{p + "linear_attn.in_proj_z.weight", []int{valDim, H}},
				ts{p + "linear_attn.in_proj_b.weight", []int{nV, H}},
				ts{p + "linear_attn.in_proj_a.weight", []int{nV, H}},
				ts{p + "linear_attn.conv1d.weight", []int{convDim * K}},
				ts{p + "linear_attn.A_log", []int{nV}},
				ts{p + "linear_attn.dt_bias", []int{nV}},
				ts{p + "linear_attn.norm.weight", []int{vHd}},
				ts{p + "linear_attn.out_proj.weight", []int{H, valDim}},
			)
		} else {
			tensors = append(tensors,
				ts{p + "self_attn.q_proj.weight", []int{qRows, H}},
				ts{p + "self_attn.k_proj.weight", []int{nKV * hd, H}},
				ts{p + "self_attn.v_proj.weight", []int{nKV * hd, H}},
				ts{p + "self_attn.o_proj.weight", []int{H, nH * hd}},
			)
		}
		tensors = append(tensors,
			ts{p + "post_attention_layernorm.weight", []int{H}},
			ts{p + "mlp.gate_proj.weight", []int{I, H}},
			ts{p + "mlp.up_proj.weight", []int{I, H}},
			ts{p + "mlp.down_proj.weight", []int{H, I}},
		)
	}
	tensors = append(tensors, ts{"model.norm.weight", []int{H}})

	man := make(map[string]tensorMeta, len(tensors))
	off := 0
	for _, t := range tensors {
		n := 1
		for _, d := range t.shape {
			n *= d
		}
		man[t.name] = tensorMeta{Dtype: "F32", Shape: t.shape, Offset: off, Nbytes: n * 4}
		off += n * 4
	}

	raw := make([]byte, off)
	seed := uint64(0x9E3779B97F4A7C15)
	next := func() float32 {
		seed = seed*6364136223846793005 + 1442695040888963407
		u := float32(seed>>40) / float32(1<<24) // [0,1)
		return u*2 - 1                          // [-1,1)
	}
	for _, t := range tensors {
		m := man[t.name]
		n := m.Nbytes / 4
		norm := strings.HasSuffix(t.name, "layernorm.weight") || strings.HasSuffix(t.name, "linear_attn.norm.weight") || t.name == "model.norm.weight"
		scale := float32(0.1)
		if strings.Contains(t.name, "embed_tokens") {
			scale = 0.2 // wider so distinct ids separate cleanly
		}
		for i := 0; i < n; i++ {
			var f float32
			if norm {
				f = 1.0
			} else {
				f = next() * scale
			}
			binary.LittleEndian.PutUint32(raw[m.Offset+i*4:], math.Float32bits(f))
		}
	}

	cfg.TieWordEmbeddings = true // synthetic head is tied to the embedding
	return &Model{Cfg: cfg, manifest: man, raw: raw}
}

// NewSyntheticMoE builds an in-memory MoE Model: the same layout as NewSynthetic
// but each layer carries a router (mlp.gate.weight) plus NumExperts expert SwiGLU
// triples (mlp.experts.<e>.{gate,up,down}_proj.weight) instead of one dense FFN.
// It needs cfg.NumExperts>0; cfg.NumExpertsPerTok is the top-k. Used by the MoE
// wiring test so the router -> top-k -> per-expert -> weighted-sum dataflow runs
// with no real Mixtral/Qwen3-MoE download.
func NewSyntheticMoE(cfg Config) *Model {
	if cfg.NumExperts <= 0 {
		panic("model: NewSyntheticMoE needs cfg.NumExperts > 0")
	}
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	H, I, V, E := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize, cfg.NumExperts

	type ts struct {
		name  string
		shape []int
	}
	var tensors []ts
	tensors = append(tensors, ts{"model.embed_tokens.weight", []int{V, H}})
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		tensors = append(tensors,
			ts{p + "input_layernorm.weight", []int{H}},
			ts{p + "self_attn.q_proj.weight", []int{nH * hd, H}},
			ts{p + "self_attn.k_proj.weight", []int{nKV * hd, H}},
			ts{p + "self_attn.v_proj.weight", []int{nKV * hd, H}},
			ts{p + "self_attn.o_proj.weight", []int{H, nH * hd}},
			ts{p + "post_attention_layernorm.weight", []int{H}},
			ts{routerName(l), []int{E, H}}, // router: [num_experts, hidden]
		)
		for e := 0; e < E; e++ {
			tensors = append(tensors,
				ts{expertName(l, e, "gate_proj.weight"), []int{I, H}},
				ts{expertName(l, e, "up_proj.weight"), []int{I, H}},
				ts{expertName(l, e, "down_proj.weight"), []int{H, I}},
			)
		}
	}
	tensors = append(tensors, ts{"model.norm.weight", []int{H}})

	man := make(map[string]tensorMeta, len(tensors))
	off := 0
	for _, t := range tensors {
		n := 1
		for _, d := range t.shape {
			n *= d
		}
		man[t.name] = tensorMeta{Dtype: "F32", Shape: t.shape, Offset: off, Nbytes: n * 4}
		off += n * 4
	}

	raw := make([]byte, off)
	seed := uint64(0x9E3779B97F4A7C15)
	next := func() float32 {
		seed = seed*6364136223846793005 + 1442695040888963407
		u := float32(seed>>40) / float32(1<<24) // [0,1)
		return u*2 - 1                          // [-1,1)
	}
	for _, t := range tensors {
		mm := man[t.name]
		n := mm.Nbytes / 4
		norm := strings.HasSuffix(t.name, "layernorm.weight") || t.name == "model.norm.weight"
		scale := float32(0.1)
		if strings.Contains(t.name, "embed_tokens") {
			scale = 0.2
		}
		for i := 0; i < n; i++ {
			var f float32
			if norm {
				f = 1.0
			} else {
				f = next() * scale
			}
			binary.LittleEndian.PutUint32(raw[mm.Offset+i*4:], math.Float32bits(f))
		}
	}

	cfg.TieWordEmbeddings = true
	return &Model{Cfg: cfg, manifest: man, raw: raw}
}

// NewSyntheticGLMDsa builds an in-memory GLM-MoE-DSA Model — the same flat-f32
// manifest+raw layout Load produces, but with the MLA attention (q_a/q_b/kv_a/kv_b)
// + Dynamic-Sparse-Attention indexer tensor set instead of the Llama q/k/v_proj.
// It exists so a runnable GLM-DSA generation path (cmd/pipelinegen -selfcheck, and
// any non-test driver) can exercise the native DSA forward + the pipeline-parallel
// handoff with NO external checkpoint — the test fixture that builds this tensor
// set lives in _test.go and cannot be imported from cmd/.
//
// It builds the DENSE FFN form (one mlp.{gate,up,down}_proj per layer); the router
// path is not needed to make the model runnable and the dense MLP keeps the cfg
// minimal. The indexer block is emitted ONLY on full-indexer layers: a shared
// layer reuses the previous full layer's top-k and reads no indexer weight (see
// glmDsaAttnSeqShared / glmDsaAttentionStep), so emitting them there would be dead
// weight. Embeddings are tied (no lm_head). cfg.IndexerTypes must have one entry
// per layer and layer 0 must be "full" (a leading shared layer has no predecessor
// index and the forward path rejects it).
//
// Like NewSynthetic the logits are meaningless; what is faithful is the layout and
// the GLM-DSA *code path* it drives. Norm weights (every *_layernorm.weight, the
// indexer k_norm.weight, model.norm.weight) init to 1.0; k_norm.bias to 0.0; all
// matmul weights to a deterministic LCG in [-scale, scale].
func NewSyntheticGLMDsa(cfg Config) *Model {
	if !cfg.isGLMMoeDsa() {
		panic("model: NewSyntheticGLMDsa needs a glm_moe_dsa cfg (ModelType/Architectures)")
	}
	if len(cfg.IndexerTypes) != cfg.NumLayers {
		panic("model: NewSyntheticGLMDsa needs len(IndexerTypes) == NumLayers")
	}
	if cfg.NumLayers > 0 && glmDsaIndexerIsShared(cfg, 0) {
		panic("model: NewSyntheticGLMDsa layer 0 must be a full indexer (no predecessor to share)")
	}
	nH := cfg.NumHeads
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize
	qkHead := cfg.QKNopeHeadDim + cfg.QKRopeHeadDim

	type ts struct {
		name  string
		shape []int
	}
	var tensors []ts
	add := func(name string, shape ...int) { tensors = append(tensors, ts{name, shape}) }

	add("model.embed_tokens.weight", V, H) // tied head; no lm_head tensor
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		ap := p + "self_attn."
		add(p+"input_layernorm.weight", H)
		add(ap+"q_a_proj.weight", cfg.QLoraRank, H)
		add(ap+"q_a_layernorm.weight", cfg.QLoraRank)
		add(ap+"q_b_proj.weight", nH*qkHead, cfg.QLoraRank)
		add(ap+"kv_a_proj_with_mqa.weight", cfg.KVLoraRank+cfg.QKRopeHeadDim, H)
		add(ap+"kv_a_layernorm.weight", cfg.KVLoraRank)
		add(ap+"kv_b_proj.weight", nH*(cfg.QKNopeHeadDim+cfg.VHeadDim), cfg.KVLoraRank)
		add(ap+"o_proj.weight", H, nH*cfg.VHeadDim)
		// Indexer weights live only on full layers (shared layers reuse the index).
		if glmDsaIndexerIsFull(cfg, l) {
			add(ap+"indexer.wq_b.weight", cfg.IndexNHeads*cfg.IndexHeadDim, cfg.QLoraRank)
			add(ap+"indexer.wk.weight", cfg.IndexHeadDim, H)
			add(ap+"indexer.k_norm.weight", cfg.IndexHeadDim)
			add(ap+"indexer.k_norm.bias", cfg.IndexHeadDim)
			add(ap+"indexer.weights_proj.weight", cfg.IndexNHeads, H)
		}
		add(p+"post_attention_layernorm.weight", H)
		add(p+"mlp.gate_proj.weight", I, H)
		add(p+"mlp.up_proj.weight", I, H)
		add(p+"mlp.down_proj.weight", H, I)
	}
	add("model.norm.weight", H)

	man := make(map[string]tensorMeta, len(tensors))
	off := 0
	for _, t := range tensors {
		n := 1
		for _, d := range t.shape {
			n *= d
		}
		man[t.name] = tensorMeta{Dtype: "F32", Shape: t.shape, Offset: off, Nbytes: n * 4}
		off += n * 4
	}

	raw := make([]byte, off)
	seed := uint64(0x9E3779B97F4A7C15)
	next := func() float32 {
		seed = seed*6364136223846793005 + 1442695040888963407
		u := float32(seed>>40) / float32(1<<24) // [0,1)
		return u*2 - 1                          // [-1,1)
	}
	for _, t := range tensors {
		mm := man[t.name]
		n := mm.Nbytes / 4
		isOne := strings.HasSuffix(t.name, "layernorm.weight") ||
			strings.HasSuffix(t.name, "k_norm.weight") ||
			t.name == "model.norm.weight"
		isZero := strings.HasSuffix(t.name, "k_norm.bias")
		scale := float32(0.1)
		if strings.Contains(t.name, "embed_tokens") {
			scale = 0.2
		}
		for i := 0; i < n; i++ {
			var f float32
			switch {
			case isOne:
				f = 1.0
			case isZero:
				f = 0.0
			default:
				f = next() * scale
			}
			binary.LittleEndian.PutUint32(raw[mm.Offset+i*4:], math.Float32bits(f))
		}
	}

	cfg.TieWordEmbeddings = true
	return &Model{Cfg: cfg, manifest: man, raw: raw}
}
