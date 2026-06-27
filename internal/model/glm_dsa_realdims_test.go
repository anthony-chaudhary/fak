package model

import (
	"math"
	"testing"
)

// glm_dsa_realdims_test.go — regression for the GLM-5.2 CONTEXT-BLIND decode bug witnessed on the
// real 466GB model (every prompt -> the same repeated token "apel"), while the existing synthetic
// GLM-DSA tests PASS. The smoking-gun structural difference: the synthetic fixture
// (tinyGLMDsaSafetensorsFixtureN) uses QKNopeHeadDim==QKRopeHeadDim (4==4) and VHeadDim==2*qkNope,
// so any code that confuses the nope/rope boundary or the qkHead/vHead strides is MASKED — the dims
// are symmetric. Real GLM-5.2 is ASYMMETRIC: qkNope=192, qkRope=64 (nope=3x rope), vHead=256. This
// fixture rebuilds the tiny model with the SAME asymmetry (qkNope != qkRope, vHead != qkNope) so a
// boundary/stride bug that only bites on asymmetric dims is exercised.
//
// The decisive property a correct attention forward MUST have: the output DEPENDS ON THE INPUT.
// A context-blind forward returns ~the same logits for different prompts. So the test prefills two
// DIFFERENT prompts and asserts the last-token logits DIFFER (low cosine / different argmax). If the
// forward is context-blind at asymmetric dims, this fails — localizing the bug to the host forward's
// asymmetric-dim slicing, no external oracle needed.

func tinyGLMDsaAsymmetricCfg() Config {
	// Asymmetric per-head dims mirroring real GLM-5.2's structure at tiny scale:
	//   qkNope(12) != qkRope(4)  (real 192 != 64),  vHead(16) != qkNope(12)  (real 256 != 192).
	// kvLora(16) and the index dims kept small but with indexHeadDim != qkHead so the index path's
	// own slicing is exercised independently of the attention head dims.
	return Config{
		HiddenSize:        32,
		NumLayers:         2,
		NumHeads:          3,
		NumKVHeads:        3,
		HeadDim:           16, // == vHead for the generic-path sanity; the DSA path uses the *_mla dims below
		IntermediateSize:  64,
		VocabSize:         41,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		EOSTokenID:        -1,
		ModelType:         "glm_moe_dsa",
		Architectures:     []string{"GlmMoeDsaForCausalLM"},
		QLoraRank:         32,
		KVLoraRank:        16,
		QKNopeHeadDim:     12, // != QKRopeHeadDim — the asymmetry the synthetic fixture lacks
		QKRopeHeadDim:     4,
		VHeadDim:          16, // != QKNopeHeadDim
		IndexNHeads:       2,
		IndexHeadDim:      8, // != qkHead(16) and != vHead(16)
		IndexTopK:         8,
		IndexerTypes:      []string{"full", "shared"},
		TieWordEmbeddings: false,
	}
}

// TestGLMDsaAsymmetricDimsContextDependent prefills two DIFFERENT prompts through the host GLM-DSA
// forward at ASYMMETRIC per-head dims and asserts the resulting logits actually DIFFER. A
// context-blind forward (the real-model "apel" bug) returns near-identical logits regardless of
// prompt; that is what this fails on. Pure host path (no backend), so it runs anywhere.
// buildGLMDsaTensorsFromCfg builds the dense (non-MoE) GLM-DSA safetensors set for an arbitrary cfg,
// mirroring tinyGLMDsaSafetensorsFixtureN's tensor layout but parameterized by the asymmetric cfg
// above (kept inline so this regression touches only one file).
func buildGLMDsaTensorsFromCfg(t *testing.T, dtype string, cfg Config) map[string]tinySTTensor {
	t.Helper()
	tensors := map[string]tinySTTensor{}
	add := func(name string, shape []int, vals []float32) {
		t.Helper()
		n := 1
		for _, d := range shape {
			n *= d
		}
		if len(vals) != n {
			t.Fatalf("tensor %s values = %d, want %d", name, len(vals), n)
		}
		tensors[name] = tinySTTensor{dtype: dtype, shape: shape, data: glmTinyTensorBytes(t, dtype, vals)}
	}
	seq := float32(0.001)
	addSeq := func(name string, shape []int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		add(name, shape, sequenceFloats(n, seq))
		seq += 0.017
	}
	addOnes := func(name string, n int) {
		vals := make([]float32, n)
		for i := range vals {
			vals[i] = 1
		}
		add(name, []int{n}, vals)
	}
	addZeros := func(name string, n int) { add(name, []int{n}, make([]float32, n)) }

	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize
	nH := cfg.NumHeads
	qkHead := cfg.QKNopeHeadDim + cfg.QKRopeHeadDim
	addSeq("model.embed_tokens.weight", []int{V, H})
	if !cfg.TieWordEmbeddings {
		addSeq("lm_head.weight", []int{V, H})
	}
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		ap := p + "self_attn."
		addOnes(p+"input_layernorm.weight", H)
		addSeq(ap+"q_a_proj.weight", []int{cfg.QLoraRank, H})
		addOnes(ap+"q_a_layernorm.weight", cfg.QLoraRank)
		addSeq(ap+"q_b_proj.weight", []int{nH * qkHead, cfg.QLoraRank})
		addSeq(ap+"kv_a_proj_with_mqa.weight", []int{cfg.KVLoraRank + cfg.QKRopeHeadDim, H})
		addOnes(ap+"kv_a_layernorm.weight", cfg.KVLoraRank)
		addSeq(ap+"kv_b_proj.weight", []int{nH * (cfg.QKNopeHeadDim + cfg.VHeadDim), cfg.KVLoraRank})
		addSeq(ap+"o_proj.weight", []int{H, nH * cfg.VHeadDim})
		if !glmDsaIndexerIsShared(cfg, l) {
			addSeq(ap+"indexer.wq_b.weight", []int{cfg.IndexNHeads * cfg.IndexHeadDim, cfg.QLoraRank})
			addSeq(ap+"indexer.wk.weight", []int{cfg.IndexHeadDim, H})
			addOnes(ap+"indexer.k_norm.weight", cfg.IndexHeadDim)
			addZeros(ap+"indexer.k_norm.bias", cfg.IndexHeadDim)
			addSeq(ap+"indexer.weights_proj.weight", []int{cfg.IndexNHeads, H})
		}
		addOnes(p+"post_attention_layernorm.weight", H)
		addSeq(p+"mlp.gate_proj.weight", []int{I, H})
		addSeq(p+"mlp.up_proj.weight", []int{I, H})
		addSeq(p+"mlp.down_proj.weight", []int{H, I})
	}
	addOnes("model.norm.weight", H)
	return tensors
}

func TestGLMDsaAsymmetricDimsContextDependent(t *testing.T) {
	cfg := tinyGLMDsaAsymmetricCfg()
	tensors := buildGLMDsaTensorsFromCfg(t, "F32", cfg)
	path := writeTinySafetensors(t, tensors)
	lean, err := LoadSafetensorsQuant(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}
	if !lean.Cfg.isGLMMoeDsa() {
		t.Fatalf("family = %q, want glm_moe_dsa", lean.Cfg.archFamilyKey())
	}

	logitsFor := func(prompt []int) []float32 {
		s := lean.NewSession()
		return s.Prefill(prompt)
	}

	// Two clearly different prompts (different tokens, different length-1 context).
	lA := logitsFor([]int{3, 17, 5})
	lB := logitsFor([]int{29, 7, 31})
	if len(lA) != cfg.VocabSize || len(lB) != cfg.VocabSize {
		t.Fatalf("logits shape A=%d B=%d want vocab=%d", len(lA), len(lB), cfg.VocabSize)
	}

	// A correct, context-dependent forward gives DIFFERENT distributions for different prompts.
	// The context-blind bug makes them ~identical (cosine ~1, same argmax). Gate on both.
	cos := realDimsCosine(lA, lB)
	aA, aB := glmDsaArgmax(lA), glmDsaArgmax(lB)
	t.Logf("GLM-DSA asymmetric-dims: prompt-A argmax=%d prompt-B argmax=%d cosine(A,B)=%.6f", aA, aB, cos)
	if cos > 0.9999 {
		t.Fatalf("CONTEXT-BLIND: two different prompts gave cosine %.6f (~identical logits) at asymmetric dims — the forward ignores its input (the real GLM-5.2 'apel' bug class)", cos)
	}

	// Also assert the forward isn't producing a degenerate constant logit vector (all-equal logits
	// would also yield the same argmax regardless of input). The std-dev of the logits must be > 0.
	if std := stddev(lA); std < 1e-6 {
		t.Fatalf("degenerate logits: prompt-A logit std-dev %.3e ~0 (constant output)", std)
	}
}

func realDimsCosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func stddev(v []float32) float64 {
	if len(v) == 0 {
		return 0
	}
	var mean float64
	for _, x := range v {
		mean += float64(x)
	}
	mean /= float64(len(v))
	var s float64
	for _, x := range v {
		d := float64(x) - mean
		s += d * d
	}
	return math.Sqrt(s / float64(len(v)))
}
