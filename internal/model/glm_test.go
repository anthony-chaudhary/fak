package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// glm_test.go — best-effort native-kernel support witnesses for the GLM family
// (zai-org GLM-5.2, model_type "glm_moe_dsa": a MoE model with Dynamic Sparse
// Attention + IndexShare + an MTP head). These tests are the "synthetic-witnessed"
// tier of the MODEL-ARCH-SUPPORT-AUDIT: they prove loader/family/MoE wiring and
// a quant-loaded GLM DSA execution smoke from in-repo artifacts only, with no HF
// download and no 753B checkpoint. The matching real-artifact GLM-MoE-DSA
// witnesses live in oracle_test.go (boundary, DSA trace reproduction, dense-prefix
// layer parity, and cacheless Forward parity) and are skipped until a tiny
// glm_moe_dsa oracle is exported.
//
// What is claimed here and what is NOT:
//   - CLAIMED: the GLM family is recognized; the MTP/vision tensors are dropped
//     at load; the generic MoE FFN forward path runs finite for a GLM-shaped
//     MoE config; and a tiny GLM DSA safetensors fixture can quantize/drop its
//     MLA/indexer projection f32 weights while Forward and Session still run.
//   - NOT CLAIMED by these synthetic weights: HF numeric parity for DSA. The real
//     DSA boundary and cacheless attention math are witnessed by the optional
//     tiny GLM-MoE-DSA oracle tests in oracle_test.go.

// TestGLMFamilyDerivationFromConfig proves the GLM family is recognized from the
// HF config metadata the loader reads: model_type + architectures. The
// archFamilyKey lowercases and strips separators, so "glm_moe_dsa" -> "glmmoedsa".
// A plain dense "glm" model_type is GLM but NOT the DSA variant; a non-GLM
// model_type (llama) is neither. This is the deterministic mapping the rest of
// the GLM load path (mtp skip, boundary witness) keys off.
func TestGLMFamilyDerivationFromConfig(t *testing.T) {
	dsa := `{
		"hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2,
		"num_key_value_heads": 1, "head_dim": 4, "intermediate_size": 16,
		"vocab_size": 32, "rms_norm_eps": 1e-5, "rope_theta": 10000,
		"model_type": "glm_moe_dsa", "architectures": ["GlmMoeDsaForCausalLM"],
		"num_local_experts": 4, "num_experts_per_tok": 2, "norm_topk_prob": true
	}`
	var cfg Config
	if err := json.Unmarshal([]byte(dsa), &cfg); err != nil {
		t.Fatalf("unmarshal glm_moe_dsa: %v", err)
	}
	if !cfg.isGLM() {
		t.Fatalf("glm_moe_dsa: isGLM=false, want true (family key=%q)", cfg.archFamilyKey())
	}
	if !cfg.isGLMMoeDsa() {
		t.Fatalf("glm_moe_dsa: isGLMMoeDsa=false, want true (family key=%q)", cfg.archFamilyKey())
	}
	if !cfg.IsMoE() || cfg.NumExperts != 4 || cfg.NumExpertsPerTok != 2 {
		t.Fatalf("glm_moe_dsa MoE = experts:%d topk:%d IsMoE:%v, want 4/2/true",
			cfg.NumExperts, cfg.NumExpertsPerTok, cfg.IsMoE())
	}

	// A plain dense GLM model_type is GLM but not the DSA variant.
	var dense Config
	if err := json.Unmarshal([]byte(`{
		"hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
		"head_dim": 4, "intermediate_size": 16, "vocab_size": 32,
		"model_type": "glm4"
	}`), &dense); err != nil {
		t.Fatalf("unmarshal glm4: %v", err)
	}
	if !dense.isGLM() {
		t.Fatalf("glm4: isGLM=false, want true")
	}
	if dense.isGLMMoeDsa() {
		t.Fatalf("glm4: isGLMMoeDsa=true, want false (no dsa in family key)")
	}

	// A non-GLM family is neither. Guards against a substring false-positive.
	var llama Config
	if err := json.Unmarshal([]byte(`{
		"hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
		"head_dim": 4, "intermediate_size": 16, "vocab_size": 32,
		"model_type": "llama"
	}`), &llama); err != nil {
		t.Fatalf("unmarshal llama: %v", err)
	}
	if llama.isGLM() {
		t.Fatalf("llama: isGLM=true, want false (family key=%q)", llama.archFamilyKey())
	}
}

// TestGLMDropsMtpAndVisualTensorsAtLoad proves the GLM load path drops the
// vision tower ("model.visual.") and MTP head ("mtp.") tensors before they are
// decoded into the f32 buffer — the same OOM-avoidance skip Qwen3.5 has, because
// GLM-5.2 ships a multimodal vision encoder and an MTP speculative-decoding head
// that the text causal-LM forward never reads. Attention and MLP tensors are
// kept, and a Llama config is completely unaffected (the Llama-invariance gate).
func TestGLMDropsMtpAndVisualTensorsAtLoad(t *testing.T) {
	glmCfg := Config{ModelType: "glm_moe_dsa", Architectures: []string{"GlmMoeDsaForCausalLM"}}
	if !glmCfg.dropsMtpAndVisualAtLoad() {
		t.Fatalf("glm_moe_dsa: dropsMtpAndVisualAtLoad=false, want true")
	}
	for _, name := range []string{"model.visual.encoder.weight", "mtp.0.embed.weight", "mtp.head.weight"} {
		if !skipLoadTensor(glmCfg, name) {
			t.Fatalf("glm: skipLoadTensor(%q)=false, want true", name)
		}
		if got, keep := quantSourceTensorName(glmCfg, name); keep || got != "" {
			t.Fatalf("glm quant: quantSourceTensorName(%q)=(%q,%v), want dropped", name, got, keep)
		}
	}
	for _, name := range []string{
		"model.embed_tokens.weight",
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.mlp.gate_proj.weight",
		"model.layers.0.mlp.experts.0.gate_proj.weight",
	} {
		if skipLoadTensor(glmCfg, name) {
			t.Fatalf("glm: skipLoadTensor(%q)=true, want false (kept tensor)", name)
		}
		if got, keep := quantSourceTensorName(glmCfg, name); !keep || got != name {
			t.Fatalf("glm quant: quantSourceTensorName(%q)=(%q,%v), want kept unchanged", name, got, keep)
		}
	}

	// Llama invariance: a dense Llama config keeps everything, including any
	// hypothetical mtp.* name (Llama has no MTP head, but the skip must be
	// gated on the family, not on the prefix alone).
	llamaCfg := Config{ModelType: "llama"}
	if llamaCfg.dropsMtpAndVisualAtLoad() {
		t.Fatalf("llama: dropsMtpAndVisualAtLoad=true, want false")
	}
	if skipLoadTensor(llamaCfg, "mtp.0.embed.weight") {
		t.Fatalf("llama: skipLoadTensor(mtp.*)=true, want false (skip is family-gated)")
	}
	if got, keep := quantSourceTensorName(llamaCfg, "mtp.0.embed.weight"); !keep || got != "mtp.0.embed.weight" {
		t.Fatalf("llama quant: quantSourceTensorName(mtp.*)=(%q,%v), want kept unchanged", got, keep)
	}
}

// TestGLMMoEForwardRunsThroughNativeKernel proves the generic MoE FFN forward
// path dispatches for a GLM-family MoE config and yields finite logits through
// the native in-kernel Prefill + decode. GLM-family MoE expert FFNs use the same
// router -> top-k -> per-expert SwiGLU -> weighted-sum dataflow the canonical MoE
// path already proves for Mixtral/Qwen3-MoE/gpt-oss. This witness runs that path
// on a synthetic GLM-shaped checkpoint (NewSyntheticMoE with a glm_moe config),
// so it needs no HF download and no 753B weights.
//
// It is deliberately a MoE-FFN witness only: it does NOT exercise the Dynamic
// Sparse Attention forward, which needs GLM-MoE-DSA q_a/q_b/kv_a/kv_b and indexer
// tensors. The DSA path is covered by the optional real-artifact oracle tests.
func TestGLMMoEForwardRunsThroughNativeKernel(t *testing.T) {
	cfg := Config{
		HiddenSize:       32,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          8,
		IntermediateSize: 64,
		VocabSize:        97,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		NumExperts:       4,
		NumExpertsPerTok: 2,
		NormTopKProb:     true,
		EOSTokenID:       -1,
		ModelType:        "glm_moe",
		Architectures:    []string{"GlmMoeForCausalLM"},
	}
	m := NewSyntheticMoE(cfg)
	if !m.Cfg.isGLM() {
		t.Fatalf("synthetic MoE config not recognized as GLM family (key=%q)", m.Cfg.archFamilyKey())
	}
	if m.Cfg.isGLMMoeDsa() {
		t.Fatalf("synthetic MoE config recognized as glm_moe_dsa; synthetic attention tensors are standard GQA")
	}
	if !m.Cfg.IsMoE() {
		t.Fatalf("synthetic GLM config not MoE")
	}

	// Cacheless full forward: finite per-layer hidden states and per-position logits.
	prompt := []int{3, 17, 5, 23, 11, 7}
	act := m.Forward(prompt)
	if len(act.Hidden) != cfg.NumLayers+1 {
		t.Fatalf("Forward hidden len = %d, want %d", len(act.Hidden), cfg.NumLayers+1)
	}
	if len(act.Logits) != len(prompt) {
		t.Fatalf("Forward logits len = %d, want seq %d", len(act.Logits), len(prompt))
	}
	for l, h := range act.Hidden {
		for i, v := range h {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("Forward hidden[%d][%d] not finite: %v", l, i, v)
			}
		}
	}
	for pos, row := range act.Logits {
		if len(row) != cfg.VocabSize {
			t.Fatalf("Forward logits[%d] len = %d, want vocab %d", pos, len(row), cfg.VocabSize)
		}
		for i, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("Forward logits[%d][%d] not finite: %v", pos, i, v)
			}
		}
	}

	// Cached path (the one the KV-quarantine bridge exercises): Prefill then a
	// couple of decode steps must agree with the cacheless forward at the last
	// position and stay finite.
	s := m.NewSession()
	cached := s.Prefill(prompt)
	last := len(prompt) - 1
	if d, _ := maxAbsDiff(cached, act.Logits[last]); d > 1e-4 {
		t.Fatalf("cached Prefill disagrees with Forward last logits: max|Δ|=%.3e", d)
	}
	for _, id := range []int{11, 29} {
		step := s.Step(id)
		if len(step) != cfg.VocabSize {
			t.Fatalf("Step logits len = %d, want vocab %d", len(step), cfg.VocabSize)
		}
		for i, v := range step {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("Step logit[%d] not finite: %v", i, v)
			}
		}
	}
}

func TestGLMMoeDsaQuantLoadRunsResidentDSAProjections(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensors(t)
	regular, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	lean, err := LoadSafetensorsQuant(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}
	if !lean.Cfg.isGLMMoeDsa() {
		t.Fatalf("lean model family = %q, want glm_moe_dsa", lean.Cfg.archFamilyKey())
	}

	quantized := []string{
		layerName(0, "self_attn.q_a_proj.weight"),
		layerName(0, "self_attn.q_b_proj.weight"),
		layerName(0, "self_attn.kv_a_proj_with_mqa.weight"),
		layerName(0, "self_attn.kv_b_proj.weight"),
		layerName(0, "self_attn.o_proj.weight"),
		layerName(0, "self_attn.indexer.wq_b.weight"),
		layerName(0, "self_attn.indexer.wk.weight"),
		layerName(0, "self_attn.indexer.weights_proj.weight"),
		layerName(0, "mlp.gate_proj.weight"),
		layerName(0, "mlp.up_proj.weight"),
		layerName(0, "mlp.down_proj.weight"),
	}
	for _, name := range quantized {
		if !regular.has(name) {
			t.Fatalf("regular model missing expected f32 tensor %s", name)
		}
		if lean.has(name) {
			t.Fatalf("lean GLM DSA model kept f32 tensor %s; want q8-only residency", name)
		}
		if lean.q8w[name] == nil {
			t.Fatalf("lean GLM DSA model missing q8 tensor %s", name)
		}
	}

	prompt := []int{3, 17, 5, 23}
	act := lean.Forward(prompt)
	if got := len(act.Logits); got != len(prompt) {
		t.Fatalf("lean GLM DSA Forward logits rows = %d, want %d", got, len(prompt))
	}
	assertFiniteGLM(t, "lean Forward last logits", act.Logits[len(prompt)-1])
	wantQ := quantHeadLogitsFromForward(lean, act, len(prompt)-1)

	s := lean.NewSession()
	s.Quant = true
	prefill := s.Prefill(prompt)
	if s.Cache.Len() != len(prompt) {
		t.Fatalf("lean GLM DSA cache len after Prefill = %d, want %d", s.Cache.Len(), len(prompt))
	}
	if d, at := maxAbsDiff(prefill, wantQ); d > 1e-4 {
		t.Fatalf("lean GLM DSA Prefill disagrees with cacheless Q8 head: max|delta|=%.3e at %d", d, at)
	}
	next := 11
	nextAct := lean.Forward(append(append([]int{}, prompt...), next))
	want := quantHeadLogitsFromForward(lean, nextAct, len(prompt))
	got := s.Step(next)
	if s.Cache.Len() != len(prompt)+1 {
		t.Fatalf("lean GLM DSA cache len after Step = %d, want %d", s.Cache.Len(), len(prompt)+1)
	}
	if d, at := maxAbsDiff(got, want); d > 1e-4 {
		t.Fatalf("lean GLM DSA Step disagrees with cacheless Q8 head: max|delta|=%.3e at %d", d, at)
	}
}

func TestGLMMoeDsaRegularQuantizeBuildsResidentDSAProjections(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensors(t)
	regular, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	lean, err := LoadSafetensorsQuant(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}

	regular.Quantize()
	for _, name := range append(glmDsaQuantFixtureWeights(cfg), regular.headName()) {
		if !regular.has(name) {
			t.Fatalf("regular GLM DSA model dropped f32 tensor %s during Quantize", name)
		}
		got := regular.q8w[name]
		if got == nil {
			t.Fatalf("regular GLM DSA Quantize missing q8 tensor %s", name)
		}
		want := lean.q8w[name]
		if want == nil {
			t.Fatalf("lean GLM DSA quant loader missing comparison q8 tensor %s", name)
		}
		assertQ8TensorEqualGLM(t, name, got, want)
	}

	prompt := []int{3, 17, 5, 23}
	act := regular.Forward(prompt)
	s := regular.NewSession()
	s.Quant = true
	prefill := s.Prefill(prompt)
	if d, at := maxAbsDiff(prefill, quantHeadLogitsFromForward(regular, act, len(prompt)-1)); d > 1e-4 {
		t.Fatalf("regular-quant GLM DSA Prefill disagrees with cacheless Q8 head: max|delta|=%.3e at %d", d, at)
	}
	next := 11
	nextAct := regular.Forward(append(append([]int{}, prompt...), next))
	got := s.Step(next)
	if d, at := maxAbsDiff(got, quantHeadLogitsFromForward(regular, nextAct, len(prompt))); d > 1e-4 {
		t.Fatalf("regular-quant GLM DSA Step disagrees with cacheless Q8 head: max|delta|=%.3e at %d", d, at)
	}
}

func TestGLMMoeDsaQuantSessionUsesUntiedQ8Head(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensors(t, false)
	lean, err := LoadSafetensorsQuant(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}
	if lean.has("lm_head.weight") {
		t.Fatalf("lean GLM DSA model kept f32 lm_head.weight; want q8-only head residency")
	}
	if lean.q8w["lm_head.weight"] == nil {
		t.Fatalf("lean GLM DSA model missing q8 lm_head.weight")
	}

	prompt := []int{3, 17, 5, 23}
	act := lean.Forward(prompt)
	s := lean.NewSession()
	s.Quant = true
	prefill := s.Prefill(prompt)
	if d, at := maxAbsDiff(prefill, act.Logits[len(prompt)-1]); d > 1e-4 {
		t.Fatalf("quant GLM DSA Prefill disagrees with resident Forward: max|delta|=%.3e at %d", d, at)
	}
	next := 11
	want := lean.Forward(append(append([]int{}, prompt...), next)).Logits[len(prompt)]
	got := s.Step(next)
	if d, at := maxAbsDiff(got, want); d > 1e-4 {
		t.Fatalf("quant GLM DSA Step disagrees with resident Forward: max|delta|=%.3e at %d", d, at)
	}
}

func TestGLMMoeDsaQuantSessionCoversNoLogitsPrefixAndGenerate(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensors(t, false)
	lean, err := LoadSafetensorsQuant(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}

	prompt := []int{3, 17, 5, 23}
	noLogits := lean.NewSession()
	noLogits.Quant = true
	noLogits.PrefillNoLogits(prompt[:len(prompt)-1])
	if got, want := noLogits.Cache.Len(), len(prompt)-1; got != want {
		t.Fatalf("quant GLM DSA PrefillNoLogits cache len = %d, want %d", got, want)
	}
	wantLast := lean.Forward(prompt).Logits[len(prompt)-1]
	if d, at := maxAbsDiff(noLogits.Step(prompt[len(prompt)-1]), wantLast); d > 1e-4 {
		t.Fatalf("quant GLM DSA PrefillNoLogits/Step disagrees with resident Forward: max|delta|=%.3e at %d", d, at)
	}

	prefix := lean.NewSession()
	prefix.Quant = true
	prefix.PrefillNoLogits(prompt[:len(prompt)-1])
	reuse := lean.SessionFromPrefix(prefix.Cache)
	reuse.Quant = true
	if got, want := reuse.Cache.Len(), len(prompt)-1; got != want {
		t.Fatalf("quant GLM DSA SessionFromPrefix cache len = %d, want %d", got, want)
	}
	if d, at := maxAbsDiff(reuse.Step(prompt[len(prompt)-1]), wantLast); d > 1e-4 {
		t.Fatalf("quant GLM DSA SessionFromPrefix/Step disagrees with resident Forward: max|delta|=%.3e at %d", d, at)
	}

	gotIDs := lean.NewSession()
	gotIDs.Quant = true
	got := gotIDs.Generate(prompt, 3)
	want := greedyByForward(lean, prompt, 3)
	if len(got) != len(want) {
		t.Fatalf("quant GLM DSA Generate len = %d, want %d (got %v, want %v)", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("quant GLM DSA Generate[%d] = %d, want %d (got %v, want %v)", i, got[i], want[i], got, want)
		}
	}
}

func TestGLMMoeDsaQuantEvictMatchesNeverSawAndReropes(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensors(t, false)
	lean, err := LoadSafetensorsQuant(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}

	prefix := []int{3, 17}
	poison := []int{5, 23}
	query := []int{11, 7}
	next := 19

	s := lean.NewSession()
	s.Quant = true
	s.PrefillNoLogits(prefix)
	s.PrefillNoLogits(poison)
	if removed := s.Cache.Evict(len(prefix), len(poison)); removed != len(poison) {
		t.Fatalf("quant GLM DSA Evict removed %d, want %d", removed, len(poison))
	}
	if got, want := s.Cache.Len(), len(prefix); got != want {
		t.Fatalf("quant GLM DSA post-evict cache len = %d, want %d", got, want)
	}

	never := append(append([]int{}, prefix...), query...)
	wantPrefill := lean.Forward(never).Logits[len(never)-1]
	if d, at := maxAbsDiff(s.Prefill(query), wantPrefill); d > 1e-4 {
		t.Fatalf("quant GLM DSA write-time Evict Prefill disagrees with never-saw Forward: max|delta|=%.3e at %d", d, at)
	}
	wantStep := lean.Forward(append(append([]int{}, never...), next)).Logits[len(never)]
	if d, at := maxAbsDiff(s.Step(next), wantStep); d > 1e-4 {
		t.Fatalf("quant GLM DSA write-time Evict Step disagrees with never-saw Forward: max|delta|=%.3e at %d", d, at)
	}

	mid := lean.NewSession()
	mid.Quant = true
	all := append(append(append([]int{}, prefix...), poison...), query...)
	mid.PrefillNoLogits(all)
	if removed := mid.Cache.Evict(len(prefix), len(poison)); removed != len(poison) {
		t.Fatalf("quant GLM DSA middle Evict removed %d, want %d", removed, len(poison))
	}
	if got, want := mid.Cache.Len(), len(prefix)+len(query); got != want {
		t.Fatalf("quant GLM DSA middle post-evict cache len = %d, want %d", got, want)
	}
	assertGLMDsaCacheReroped(t, mid.Cache)
}

func TestGLMMoeDsaQuantLoadBF16MatchesDecodedQ8(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensorsDType(t, "BF16", true)
	regular, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	lean, err := LoadSafetensorsQuant(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}

	for _, name := range glmDsaQuantFixtureWeights(cfg) {
		meta := regular.manifest[name]
		want := quantizeQ8(regular.tensor(name), meta.Shape[0], meta.Shape[1])
		got := lean.q8w[name]
		if got == nil {
			t.Fatalf("lean BF16 GLM DSA model missing q8 tensor %s", name)
		}
		assertQ8TensorEqualGLM(t, name, got, want)
		if lean.has(name) {
			t.Fatalf("lean BF16 GLM DSA model kept f32 tensor %s; want q8-only residency", name)
		}
	}

	prompt := []int{3, 17, 5, 23}
	act := lean.Forward(prompt)
	s := lean.NewSession()
	s.Quant = true
	prefill := s.Prefill(prompt)
	if d, at := maxAbsDiff(prefill, quantHeadLogitsFromForward(lean, act, len(prompt)-1)); d > 1e-4 {
		t.Fatalf("BF16 quant GLM DSA Prefill disagrees with cacheless Q8 head: max|delta|=%.3e at %d", d, at)
	}
}

func TestGLMMoeDsaQuantizeAllowsSharedLayersWithoutIndexerTensors(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, true, false, false)
	regular, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	lean, err := LoadSafetensorsQuant(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}

	regular.Quantize()
	for _, name := range []string{
		layerName(0, "self_attn.indexer.wq_b.weight"),
		layerName(0, "self_attn.indexer.wk.weight"),
		layerName(0, "self_attn.indexer.weights_proj.weight"),
	} {
		if regular.q8w[name] == nil {
			t.Fatalf("regular GLM DSA Quantize missing full-indexer q8 tensor %s", name)
		}
		if lean.q8w[name] == nil {
			t.Fatalf("lean GLM DSA quant loader missing full-indexer q8 tensor %s", name)
		}
	}
	for _, name := range []string{
		layerName(1, "self_attn.indexer.wq_b.weight"),
		layerName(1, "self_attn.indexer.wk.weight"),
		layerName(1, "self_attn.indexer.weights_proj.weight"),
	} {
		if regular.has(name) || regular.q8w[name] != nil {
			t.Fatalf("regular GLM DSA shared layer unexpectedly retained/quantized indexer tensor %s", name)
		}
		if lean.has(name) || lean.q8w[name] != nil {
			t.Fatalf("lean GLM DSA shared layer unexpectedly retained/quantized indexer tensor %s", name)
		}
	}

	prompt := []int{3, 17, 5, 23}
	act := regular.Forward(prompt)
	s := regular.NewSession()
	s.Quant = true
	prefill := s.Prefill(prompt)
	if d, at := maxAbsDiff(prefill, quantHeadLogitsFromForward(regular, act, len(prompt)-1)); d > 1e-4 {
		t.Fatalf("regular shared-indexerless GLM DSA Prefill disagrees with cacheless Q8 head: max|delta|=%.3e at %d", d, at)
	}

	leanAct := lean.Forward(prompt)
	leanS := lean.NewSession()
	leanS.Quant = true
	leanPrefill := leanS.Prefill(prompt)
	if d, at := maxAbsDiff(leanPrefill, quantHeadLogitsFromForward(lean, leanAct, len(prompt)-1)); d > 1e-4 {
		t.Fatalf("lean shared-indexerless GLM DSA Prefill disagrees with cacheless Q8 head: max|delta|=%.3e at %d", d, at)
	}
}

func TestGLMMoeDsaQuantizeBuildsSharedExperts(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, false, true, true)
	regular, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	lean, err := LoadSafetensorsQuant(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}

	regular.Quantize()
	for _, name := range glmDsaMoEQuantFixtureWeights(cfg) {
		got := regular.q8w[name]
		if got == nil {
			t.Fatalf("regular GLM DSA Quantize missing q8 tensor %s", name)
		}
		want := lean.q8w[name]
		if want == nil {
			t.Fatalf("lean GLM DSA quant loader missing q8 tensor %s", name)
		}
		assertQ8TensorEqualGLM(t, name, got, want)
		if lean.has(name) {
			t.Fatalf("lean GLM DSA model kept f32 tensor %s; want q8-only residency", name)
		}
	}

	prompt := []int{3, 17, 5, 23}
	act := regular.Forward(prompt)
	s := regular.NewSession()
	s.Quant = true
	prefill := s.Prefill(prompt)
	if d, at := maxAbsDiff(prefill, quantHeadLogitsFromForward(regular, act, len(prompt)-1)); d > 1e-4 {
		t.Fatalf("regular GLM DSA shared-expert Prefill disagrees with cacheless Q8 head: max|delta|=%.3e at %d", d, at)
	}

	leanAct := lean.Forward(prompt)
	leanS := lean.NewSession()
	leanS.Quant = true
	leanPrefill := leanS.Prefill(prompt)
	if d, at := maxAbsDiff(leanPrefill, quantHeadLogitsFromForward(lean, leanAct, len(prompt)-1)); d > 1e-4 {
		t.Fatalf("lean GLM DSA shared-expert Prefill disagrees with cacheless Q8 head: max|delta|=%.3e at %d", d, at)
	}
}

func TestGLMMoeDsaQuantDirLoadsShardedBF16Weights(t *testing.T) {
	dir, cfg := writeTinyGLMDsaShardedSafetensorsDir(t, "BF16", false, true, true, true)
	regular, err := LoadSafetensorsDir(dir, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsDir: %v", err)
	}
	lean, err := LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuantDir: %v", err)
	}

	var checked int
	for name, meta := range regular.manifest {
		if !isQuantWeight(name) || len(meta.Shape) != 2 {
			continue
		}
		want := quantizeQ8(regular.tensor(name), meta.Shape[0], meta.Shape[1])
		got := lean.q8w[name]
		if got == nil {
			t.Fatalf("lean sharded BF16 GLM DSA model missing q8 tensor %s", name)
		}
		assertQ8TensorEqualGLM(t, name, got, want)
		if lean.has(name) {
			t.Fatalf("lean sharded BF16 GLM DSA model kept f32 tensor %s; want q8-only residency", name)
		}
		checked++
	}
	if checked == 0 {
		t.Fatalf("sharded BF16 GLM DSA fixture checked no quantizable tensors")
	}
	for _, small := range []string{
		"model.embed_tokens.weight",
		layerName(0, "input_layernorm.weight"),
		layerName(1, "post_attention_layernorm.weight"),
		"model.norm.weight",
	} {
		if !lean.has(small) {
			t.Fatalf("lean sharded BF16 GLM DSA model dropped small f32 tensor %s", small)
		}
	}

	prompt := []int{3, 17, 5, 23}
	act := lean.Forward(prompt)
	s := lean.NewSession()
	s.Quant = true
	prefill := s.Prefill(prompt)
	if d, at := maxAbsDiff(prefill, act.Logits[len(prompt)-1]); d > 1e-4 {
		t.Fatalf("sharded BF16 GLM DSA Prefill disagrees with resident Forward: max|delta|=%.3e at %d", d, at)
	}
}

func quantHeadLogitsFromForward(m *Model, act *Activations, pos int) []float32 {
	H := m.Cfg.HiddenSize
	layerHidden := act.Hidden[len(act.Hidden)-1]
	xf := m.finalNorm(layerHidden[pos*H : (pos+1)*H])
	logits := qMatRows(m.q8Head(), quantizeVecQ8(xf))
	logitScaleInPlace(logits, m.Cfg)
	return logits
}

func greedyByForward(m *Model, prompt []int, n int) []int {
	ids := append([]int(nil), prompt...)
	out := make([]int, 0, n)
	for i := 0; i < n; i++ {
		act := m.Forward(ids)
		next := argmaxF32(act.Logits[len(ids)-1])
		out = append(out, next)
		if m.Cfg.IsEOS(next) {
			break
		}
		ids = append(ids, next)
	}
	return out
}

func writeTinyGLMDsaSafetensors(t *testing.T, tiedHead ...bool) (string, Config) {
	t.Helper()
	tied := true
	if len(tiedHead) > 0 {
		tied = tiedHead[0]
	}
	return writeTinyGLMDsaSafetensorsDType(t, "F32", tied)
}

func writeTinyGLMDsaSafetensorsDType(t *testing.T, dtype string, tied bool) (string, Config) {
	t.Helper()
	return writeTinyGLMDsaSafetensorsFixture(t, dtype, tied, false, false, false)
}

func writeTinyGLMDsaSafetensorsFixture(t *testing.T, dtype string, tied, omitSharedIndexer, withMoE, withSharedExperts bool) (string, Config) {
	t.Helper()
	tensors, cfg := tinyGLMDsaSafetensorsFixture(t, dtype, tied, omitSharedIndexer, withMoE, withSharedExperts)
	return writeTinySafetensors(t, tensors), cfg
}

func tinyGLMDsaSafetensorsFixture(t *testing.T, dtype string, tied, omitSharedIndexer, withMoE, withSharedExperts bool) (map[string]tinySTTensor, Config) {
	return tinyGLMDsaSafetensorsFixtureN(t, dtype, 2, []string{"full", "shared"}, tied, omitSharedIndexer, withMoE, withSharedExperts)
}

// tinyGLMDsaSafetensorsFixtureN is the layer-count-parameterized form of the GLM
// DSA fixture. numLayers and indexerTypes (one "full"/"shared" per layer) let a
// pipeline-partition test build a checkpoint with a valid interior cut on a FULL
// indexer layer (e.g. 3 layers ["full","shared","full"], cut at layer 2), which
// the default 2-layer ["full","shared"] fixture cannot express (its only interior
// boundary is the shared layer, correctly rejected). The 2-layer caller above is
// byte-identical to the prior hardcoded fixture.
func tinyGLMDsaSafetensorsFixtureN(t *testing.T, dtype string, numLayers int, indexerTypes []string, tied, omitSharedIndexer, withMoE, withSharedExperts bool) (map[string]tinySTTensor, Config) {
	t.Helper()
	if len(indexerTypes) != numLayers {
		t.Fatalf("tinyGLMDsaSafetensorsFixtureN: indexerTypes len = %d, want numLayers = %d", len(indexerTypes), numLayers)
	}
	cfg := Config{
		HiddenSize:        32,
		NumLayers:         numLayers,
		NumHeads:          4,
		NumKVHeads:        4,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         41,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		EOSTokenID:        -1,
		ModelType:         "glm_moe_dsa",
		Architectures:     []string{"GlmMoeDsaForCausalLM"},
		QLoraRank:         32,
		KVLoraRank:        32,
		QKNopeHeadDim:     4,
		QKRopeHeadDim:     4,
		VHeadDim:          8,
		IndexNHeads:       4,
		IndexHeadDim:      8,
		IndexTopK:         2,
		IndexerTypes:      indexerTypes,
		TieWordEmbeddings: tied,
	}
	if withMoE {
		cfg.NumExperts = 2
		cfg.NumExpertsPerTok = 1
		cfg.NGroup = 1
		cfg.TopKGroup = 1
		cfg.RoutedScalingFactor = 1
	}
	if withSharedExperts {
		cfg.NSharedExperts = 1
		cfg.MoEIntermediateSize = cfg.IntermediateSize
	}
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
	addZeros := func(name string, n int) {
		add(name, []int{n}, make([]float32, n))
	}

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
		if !omitSharedIndexer || !glmDsaIndexerIsShared(cfg, l) {
			addSeq(ap+"indexer.wq_b.weight", []int{cfg.IndexNHeads * cfg.IndexHeadDim, cfg.QLoraRank})
			addSeq(ap+"indexer.wk.weight", []int{cfg.IndexHeadDim, H})
			addOnes(ap+"indexer.k_norm.weight", cfg.IndexHeadDim)
			addZeros(ap+"indexer.k_norm.bias", cfg.IndexHeadDim)
			addSeq(ap+"indexer.weights_proj.weight", []int{cfg.IndexNHeads, H})
		}
		addOnes(p+"post_attention_layernorm.weight", H)
		if cfg.IsMoE() {
			addSeq(routerName(l), []int{cfg.NumExperts, H})
			for e := 0; e < cfg.NumExperts; e++ {
				addSeq(expertName(l, e, "gate_proj.weight"), []int{I, H})
				addSeq(expertName(l, e, "up_proj.weight"), []int{I, H})
				addSeq(expertName(l, e, "down_proj.weight"), []int{H, I})
			}
			if cfg.NSharedExperts > 0 {
				sharedI := cfg.MoEIntermediateSize * cfg.NSharedExperts
				addSeq(p+"mlp.shared_experts.gate_proj.weight", []int{sharedI, H})
				addSeq(p+"mlp.shared_experts.up_proj.weight", []int{sharedI, H})
				addSeq(p+"mlp.shared_experts.down_proj.weight", []int{H, sharedI})
			}
		} else {
			addSeq(p+"mlp.gate_proj.weight", []int{I, H})
			addSeq(p+"mlp.up_proj.weight", []int{I, H})
			addSeq(p+"mlp.down_proj.weight", []int{H, I})
		}
	}
	addOnes("model.norm.weight", H)
	return tensors, cfg
}

func writeTinyGLMDsaShardedSafetensorsDir(t *testing.T, dtype string, tied, omitSharedIndexer, withMoE, withSharedExperts bool) (string, Config) {
	t.Helper()
	tensors, cfg := tinyGLMDsaSafetensorsFixture(t, dtype, tied, omitSharedIndexer, withMoE, withSharedExperts)
	return writeShardedDirFromTensors(t, tensors, cfg)
}

// writeTinyGLMDsaShardedSafetensorsDirN is the layer-parameterized sibling: it
// writes an N-layer GLM DSA sharded checkpoint so a pipeline-partition test can
// load each stage standalone via WithLayerWindow with a valid full-indexer cut.
func writeTinyGLMDsaShardedSafetensorsDirN(t *testing.T, dtype string, numLayers int, indexerTypes []string, tied, omitSharedIndexer, withMoE, withSharedExperts bool) (string, Config) {
	t.Helper()
	tensors, cfg := tinyGLMDsaSafetensorsFixtureN(t, dtype, numLayers, indexerTypes, tied, omitSharedIndexer, withMoE, withSharedExperts)
	return writeShardedDirFromTensors(t, tensors, cfg)
}

// writeShardedDirFromTensors round-robins a tensor set across two safetensors
// shards + an index.json and returns the temp dir — the shared body of both GLM
// DSA sharded-dir writers above.
func writeShardedDirFromTensors(t *testing.T, tensors map[string]tinySTTensor, cfg Config) (string, Config) {
	t.Helper()
	dir := t.TempDir()
	shardNames := []string{"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
	shards := []map[string]tinySTTensor{{}, {}}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	weightMap := map[string]string{}
	for i, name := range names {
		shard := i % len(shards)
		shards[shard][name] = tensors[name]
		weightMap[name] = shardNames[shard]
	}
	for i, shard := range shards {
		path := filepath.Join(dir, shardNames[i])
		if err := os.WriteFile(path, tinySafetensorsBytes(t, shard), 0o644); err != nil {
			t.Fatalf("write GLM DSA shard %s: %v", shardNames[i], err)
		}
	}
	index, err := json.Marshal(struct {
		WeightMap map[string]string `json:"weight_map"`
	}{WeightMap: weightMap})
	if err != nil {
		t.Fatalf("marshal GLM DSA shard index: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), index, 0o644); err != nil {
		t.Fatalf("write GLM DSA shard index: %v", err)
	}
	return dir, cfg
}

func glmTinyTensorBytes(t *testing.T, dtype string, vals []float32) []byte {
	t.Helper()
	switch dtype {
	case "F32":
		return f32TestBytes(vals)
	case "BF16":
		return bf16TestBytes(vals)
	default:
		t.Fatalf("unsupported tiny GLM tensor dtype %q", dtype)
		return nil
	}
}

func glmDsaQuantFixtureWeights(cfg Config) []string {
	names := []string{}
	for l := 0; l < cfg.NumLayers; l++ {
		names = append(names,
			layerName(l, "self_attn.q_a_proj.weight"),
			layerName(l, "self_attn.q_b_proj.weight"),
			layerName(l, "self_attn.kv_a_proj_with_mqa.weight"),
			layerName(l, "self_attn.kv_b_proj.weight"),
			layerName(l, "self_attn.o_proj.weight"),
			layerName(l, "self_attn.indexer.wq_b.weight"),
			layerName(l, "self_attn.indexer.wk.weight"),
			layerName(l, "self_attn.indexer.weights_proj.weight"),
			layerName(l, "mlp.gate_proj.weight"),
			layerName(l, "mlp.up_proj.weight"),
			layerName(l, "mlp.down_proj.weight"),
		)
	}
	return names
}

func glmDsaMoEQuantFixtureWeights(cfg Config) []string {
	names := []string{}
	for l := 0; l < cfg.NumLayers; l++ {
		names = append(names, routerName(l))
		for e := 0; e < cfg.NumExperts; e++ {
			names = append(names,
				expertName(l, e, "gate_proj.weight"),
				expertName(l, e, "up_proj.weight"),
				expertName(l, e, "down_proj.weight"),
			)
		}
		if cfg.NSharedExperts > 0 {
			names = append(names,
				layerName(l, "mlp.shared_experts.gate_proj.weight"),
				layerName(l, "mlp.shared_experts.up_proj.weight"),
				layerName(l, "mlp.shared_experts.down_proj.weight"),
			)
		}
	}
	return names
}

func assertQ8TensorEqualGLM(t *testing.T, name string, got, want *q8Tensor) {
	t.Helper()
	if got.out != want.out || got.in != want.in || got.nblk != want.nblk {
		t.Fatalf("%s shape: got (%d,%d,%d), want (%d,%d,%d)",
			name, got.out, got.in, got.nblk, want.out, want.in, want.nblk)
	}
	for i := range want.q {
		if got.q[i] != want.q[i] {
			t.Fatalf("%s code[%d]: got %d, want %d", name, i, got.q[i], want.q[i])
		}
	}
	for i := range want.d {
		if math.Float32bits(got.d[i]) != math.Float32bits(want.d[i]) {
			t.Fatalf("%s scale[%d]: got %v, want %v", name, i, got.d[i], want.d[i])
		}
	}
}

func assertFiniteGLM(t *testing.T, label string, xs []float32) {
	t.Helper()
	for i, v := range xs {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("%s[%d] is not finite: %v", label, i, v)
		}
	}
}
