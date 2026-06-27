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
	// Asymmetric per-head dims mirroring real GLM-5.2's structure, but every REDUCTION dim a
	// multiple of 32 (the Q8_0 quantize-at-load constraint — real GLM-5.2 dims 192/64/256/512 are
	// all 32-aligned). The asymmetry the synthetic fixture lacks:
	//   qkNope(64) != qkRope(32)  (real 192 != 64),  vHead(96) != qkNope(64)  (real 256 != 192),
	//   indexHeadDim(32) != qkHead(96) and != vHead(96).
	// Reduction dims: q_b_proj in=qLora(64); kv_b_proj in=kvLora(32); o_proj in=nH*vHead(2*96=192);
	// indexer.wk in=H(64); all multiples of 32.
	return Config{
		HiddenSize:        64,
		NumLayers:         2,
		NumHeads:          2,
		NumKVHeads:        2,
		HeadDim:           96, // == vHead for the generic-path sanity; the DSA path uses the *_mla dims
		IntermediateSize:  64,
		VocabSize:         41,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		EOSTokenID:        -1,
		ModelType:         "glm_moe_dsa",
		Architectures:     []string{"GlmMoeDsaForCausalLM"},
		QLoraRank:         64,
		KVLoraRank:        32,
		QKNopeHeadDim:     64, // != QKRopeHeadDim — the asymmetry the synthetic fixture lacks
		QKRopeHeadDim:     32,
		VHeadDim:          96, // != QKNopeHeadDim
		IndexNHeads:       2,
		IndexHeadDim:      32, // != qkHead(96) and != vHead(96)
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
	// Varied pseudo-random weights (LCG, ~N(0,0.1)) — NOT the near-constant sequenceFloats ramp,
	// whose nearly-identical rows collapse the forward to a context-blind output (a degenerate
	// fixture). Real weights vary per row, so the attention/MLP actually depend on their input.
	var lcg uint64 = 0x243F6A8885A308D3
	randf := func() float32 {
		lcg = lcg*6364136223846793005 + 1442695040888963407
		// top 24 bits -> [-0.1, 0.1)
		u := float32(lcg>>40) / float32(1<<24)
		return (u - 0.5) * 0.2
	}
	addSeq := func(name string, shape []int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		vals := make([]float32, n)
		for i := range vals {
			vals[i] = randf()
		}
		add(name, shape, vals)
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

// promptCosine builds a GLM-DSA model from cfg and returns cosine(logits(promptA), logits(promptB))
// + the two argmaxes. A context-DEPENDENT forward gives cosine < 1 (different prompts -> different
// logits); the context-blind bug gives cosine == 1.
func promptCosine(t *testing.T, cfg Config) (float64, int, int) {
	t.Helper()
	tensors := buildGLMDsaTensorsFromCfg(t, "F32", cfg)
	path := writeTinySafetensors(t, tensors)
	lean, err := LoadSafetensorsQuant(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}
	if !lean.Cfg.isGLMMoeDsa() {
		t.Fatalf("family = %q, want glm_moe_dsa", lean.Cfg.archFamilyKey())
	}
	logitsFor := func(prompt []int) []float32 { return lean.NewSession().Prefill(prompt) }
	lA := logitsFor([]int{3, 17, 5})
	lB := logitsFor([]int{29, 7, 31})
	if len(lA) != cfg.VocabSize || len(lB) != cfg.VocabSize {
		t.Fatalf("logits shape A=%d B=%d want vocab=%d", len(lA), len(lB), cfg.VocabSize)
	}
	if std := stddev(lA); std < 1e-6 {
		t.Fatalf("degenerate logits: prompt-A logit std-dev %.3e ~0 (constant output)", std)
	}
	return realDimsCosine(lA, lB), glmDsaArgmax(lA), glmDsaArgmax(lB)
}

// tinyGLMDsaSymmetricCfg is the asymmetric cfg made SYMMETRIC (qkNope==qkRope, vHead==qkNope) — the
// control: same weights/structure, only the per-head dims are balanced like the existing synthetic
// fixture. It must stay context-DEPENDENT. If both this and the asymmetric case were context-blind,
// the bug would be in the fixture, not the asymmetric-dim handling.
func tinyGLMDsaSymmetricCfg() Config {
	c := tinyGLMDsaAsymmetricCfg()
	c.QKNopeHeadDim = 32 // == QKRopeHeadDim
	c.QKRopeHeadDim = 32
	c.VHeadDim = 32 // == qkNope; qkHead = 64
	// keep reduction dims 32-aligned: o_proj in = nH*vHead = 2*32 = 64 (ok), kv_b in = kvLora = 32 (ok).
	return c
}

// TestGLMDsaAsymmetricDimsContextDependent reproduces the real GLM-5.2 "apel" context-blind bug in a
// fast CPU unit test: at ASYMMETRIC per-head dims (qkNope != qkRope, like real 192 != 64) the forward
// returns IDENTICAL logits for different prompts, while the SYMMETRIC control (qkNope == qkRope, the
// shape the existing synthetic fixture uses) stays context-dependent. The symmetric control proves
// the bug is the asymmetric-dim handling, not the fixture.
func TestGLMDsaAsymmetricDimsContextDependent(t *testing.T) {
	// Control: symmetric dims MUST be context-dependent (cosine < 1).
	symCos, symA, symB := promptCosine(t, tinyGLMDsaSymmetricCfg())
	t.Logf("SYMMETRIC control: argmax A=%d B=%d cosine=%.6f", symA, symB, symCos)
	if symCos > 0.9999 {
		t.Fatalf("the SYMMETRIC control is ALSO context-blind (cosine %.6f) — the fixture is degenerate, not the asymmetric-dim handling; fix the fixture before trusting the asymmetric case", symCos)
	}

	// The real bug: asymmetric dims go context-blind.
	asymCos, aA, aB := promptCosine(t, tinyGLMDsaAsymmetricCfg())
	t.Logf("ASYMMETRIC: argmax A=%d B=%d cosine=%.6f", aA, aB, asymCos)
	if asymCos > 0.9999 {
		t.Fatalf("CONTEXT-BLIND: at ASYMMETRIC dims two different prompts gave cosine %.6f (identical logits) while the symmetric control gave %.6f — the GLM-DSA forward ignores its input at qkNope != qkRope (the real GLM-5.2 'apel' bug)", asymCos, symCos)
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
