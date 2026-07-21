package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// Kimi K2 / K3 (Moonshot) arch recognition.
//
// Kimi K2 is a scaled-up DeepSeek-V3: the same MLA latent attention + DeepSeekMoE backbone
// (sigmoid router + exp_probs_b score-correction bias + routed shared experts), only wider —
// 384 routed experts vs V3's 256, 8 active, 1 shared, and fewer attention heads. K2's released
// GGUFs already declare general.architecture = "deepseek2" (HF model_type "kimi_k2"), so a stock
// K2 file loads via fak's deepseek2 MLA+MoE path untouched — fak has no >256-expert cap, the one
// real llama.cpp K2 blocker. canonicalGGUFArch additionally normalizes the Moonshot-branded arch
// spellings a repack or a future dedicated llama.cpp LLM_ARCH_KIMI_* could emit onto "deepseek2",
// so recognition fires (usesMLAMoELayout → ForwardGLMDsaMLA) instead of the #934 empty/attnSeq
// refusal, while the metadata-key PREFIX stays the file's own spelling. Kimi carries no DSA
// indexer, so it collapses to deepseek2 outright rather than earning its own model_type. These
// tests are the recognition seam K2 rides today and K3 is expected to ride when it ships (re-pin
// the tensor/metadata spellings against a real K3 header before treating a K3 load as validated).

// synthKimiMeta is the minimal Kimi-K2 (DeepSeek-V3-family) MLA+MoE metadata scaled down, keyed on
// the file's OWN "<arch>." prefix (not "deepseek2.") to prove the metadata reads use the raw arch
// prefix while recognition normalizes the arch string. Shape mirrors the deepseek2-derived case in
// TestGLMMoeDsaConfig, MINUS the DSA-indexer keys (Kimi has no lightning indexer) and with the
// Kimi-distinctive no-group routing (expert_group_count=1) + K2's real routed_scaling_factor 2.827.
func synthKimiMeta(arch string) map[string]Value {
	p := arch + "."
	return map[string]Value{
		"general.architecture":                 {Type: TypeString, Value: arch},
		p + "context_length":                   {Type: TypeUint64, Value: uint64(16)},
		p + "embedding_length":                 {Type: TypeUint64, Value: uint64(32)},
		p + "block_count":                      {Type: TypeUint64, Value: uint64(4)},
		p + "feed_forward_length":              {Type: TypeUint64, Value: uint64(64)},
		p + "attention.head_count":             {Type: TypeUint64, Value: uint64(4)},
		p + "attention.head_count_kv":          {Type: TypeUint64, Value: uint64(1)},
		p + "attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
		p + "rope.freq_base":                   {Type: TypeFloat32, Value: float32(10000)},
		// MoE FFN axis (scaled-down "384 experts / 8 active / 1 shared").
		p + "expert_count":                      {Type: TypeUint64, Value: uint64(8)},
		p + "expert_used_count":                 {Type: TypeUint64, Value: uint64(2)},
		p + "expert_feed_forward_length":        {Type: TypeUint64, Value: uint64(48)},
		p + "expert_shared_count":               {Type: TypeUint64, Value: uint64(1)},
		p + "expert_shared_feed_forward_length": {Type: TypeUint64, Value: uint64(48)},
		p + "leading_dense_block_count":         {Type: TypeUint64, Value: uint64(1)},
		// Kimi uses no group-limited routing (n_group=1), unlike DeepSeek-V3's 8 groups.
		p + "expert_group_count":      {Type: TypeUint64, Value: uint64(1)},
		p + "expert_group_used_count": {Type: TypeUint64, Value: uint64(1)},
		p + "expert_weights_scale":    {Type: TypeFloat32, Value: float32(2.827)},
		// MLA latent-projection ranks + deepseek2-convention head-dim derivation
		// (n_embd_head_k under attention.key_length, rope portion under rope.dimension_count,
		// v_head_dim under attention.value_length — no explicit qk_* keys).
		p + "attention.q_lora_rank":  {Type: TypeUint64, Value: uint64(24)},
		p + "attention.kv_lora_rank": {Type: TypeUint64, Value: uint64(16)},
		p + "attention.key_length":   {Type: TypeUint64, Value: uint64(12)},
		p + "attention.value_length": {Type: TypeUint64, Value: uint64(8)},
		p + "rope.dimension_count":   {Type: TypeUint64, Value: uint64(4)},
	}
}

// TestCanonicalGGUFArchNormalizesKimi pins the arch-string normalization: every documented /
// plausible Kimi (Moonshot) spelling maps onto the internal "deepseek2" MLA+MoE model_type, while
// the already-canonical deepseek2 is a no-op and an unrelated arch is passed through untouched (no
// over-normalization outside the Moonshot Kimi namespace).
func TestCanonicalGGUFArchNormalizesKimi(t *testing.T) {
	for _, spelling := range []string{"kimi-k2", "kimi_k2", "kimi2", "kimi-k3", "kimi_k3", "kimi3", "kimi"} {
		if got := canonicalGGUFArch(spelling); got != "deepseek2" {
			t.Errorf("canonicalGGUFArch(%q) = %q, want deepseek2", spelling, got)
		}
	}
	// The already-canonical deepseek2 is a no-op, and an unrelated arch is untouched.
	if got := canonicalGGUFArch("deepseek2"); got != "deepseek2" {
		t.Errorf("canonicalGGUFArch(deepseek2) = %q, want deepseek2", got)
	}
	if got := canonicalGGUFArch("llama"); got != "llama" {
		t.Errorf("canonicalGGUFArch(llama) = %q, want llama (must not over-normalize)", got)
	}
}

// TestKimiConfigDerivesDeepSeekMLAMoE is the done-condition witness: a Kimi GGUF (branded
// general.architecture, own "kimi_k2." metadata prefix) derives a Config whose model_type is the
// canonical deepseek2 and whose MoE + MLA axes are populated from the raw-prefixed keys — so the
// shared MLA+MoE forward is enabled, NOT the #934 empty-layer_types refusal. Recognition is
// asserted through exported API only: usesMLAMoELayout() is unexported in package model, but it is
// exactly the predicate ClassifyForwardPath keys on to return ForwardGLMDsaMLA, so the
// classification below (a separate test) is the load-bearing recognition proof.
func TestKimiConfigDerivesDeepSeekMLAMoE(t *testing.T) {
	f := &File{Metadata: synthKimiMeta("kimi_k2")}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	// Arch normalized to the internal MLA+MoE model_type (the metadata prefix stayed "kimi_k2.").
	if cfg.ModelType != "deepseek2" {
		t.Fatalf("ModelType = %q, want deepseek2 (canonicalized from kimi_k2)", cfg.ModelType)
	}
	if !cfg.IsMoE() {
		t.Fatalf("IsMoE() = false; want true (NumExperts=%d)", cfg.NumExperts)
	}
	// MoE FFN axis: the scaled-down 384-expert / 8-active / 1-shared structure.
	if cfg.NumExperts != 8 || cfg.NumExpertsPerTok != 2 {
		t.Fatalf("MoE counts = experts:%d topk:%d, want 8/2", cfg.NumExperts, cfg.NumExpertsPerTok)
	}
	if cfg.MoEIntermediateSize != 48 || cfg.SharedIntermediateSize != 48 {
		t.Fatalf("MoE ffn = moe:%d shared:%d, want 48/48", cfg.MoEIntermediateSize, cfg.SharedIntermediateSize)
	}
	if cfg.NSharedExperts != 1 || cfg.FirstKDenseReplace != 1 {
		t.Fatalf("MoE structure = shared:%d firstKDense:%d, want 1/1", cfg.NSharedExperts, cfg.FirstKDenseReplace)
	}
	// Kimi's no-group routing (n_group=1) and K2's routed_scaling_factor 2.827 (float32-widened).
	if cfg.NGroup != 1 {
		t.Errorf("NGroup = %d, want 1 (Kimi no-group routing)", cfg.NGroup)
	}
	if cfg.RoutedScalingFactor < 2.8 || cfg.RoutedScalingFactor > 2.9 {
		t.Errorf("RoutedScalingFactor = %v, want ~2.827", cfg.RoutedScalingFactor)
	}
	// MLA latent-projection ranks + deepseek2-derived head dims (qk_nope = key_length - rope).
	if cfg.QLoraRank != 24 || cfg.KVLoraRank != 16 {
		t.Fatalf("MLA ranks = q:%d kv:%d, want 24/16", cfg.QLoraRank, cfg.KVLoraRank)
	}
	if cfg.QKRopeHeadDim != 4 || cfg.QKNopeHeadDim != 8 || cfg.VHeadDim != 8 {
		t.Errorf("MLA head dims = rope:%d nope:%d v:%d, want 4/8/8",
			cfg.QKRopeHeadDim, cfg.QKNopeHeadDim, cfg.VHeadDim)
	}
	// Kimi carries no DSA lightning indexer — the GLM-5.2-specific axis must stay zero.
	if cfg.IndexNHeads != 0 {
		t.Errorf("IndexNHeads = %d, want 0 (Kimi has no DSA indexer)", cfg.IndexNHeads)
	}
}

// TestKimiConfigClassifiesToRecognizedMLAMoEForward binds the recognition→dispatch chain end to
// end: a Kimi GGUF's derived Config must reach the wired GLM-DSA/DeepSeek MLA+MoE forward
// (ForwardGLMDsaMLA), NOT the #934 UnsupportedArchError an unrecognized arch would raise. This is
// the seam a serve/bench caller asserts before decoding a token, proven here from the raw
// "kimi_k2." metadata through canonicalGGUFArch to model.ClassifyForwardPath in one witness.
func TestKimiConfigClassifiesToRecognizedMLAMoEForward(t *testing.T) {
	f := &File{Metadata: synthKimiMeta("kimi_k2")}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	got, err := model.ClassifyForwardPath(cfg, nil)
	if err != nil {
		t.Fatalf("a recognized Kimi MLA+MoE model must classify, got refusal: %v", err)
	}
	if got != model.ForwardGLMDsaMLA {
		t.Fatalf("ClassifyForwardPath = %q, want %q (recognized MLA+MoE, not the #934 refusal)",
			got, model.ForwardGLMDsaMLA)
	}
}

// TestKimiConfigRecognizedRegardlessOfSpelling proves the recognition is spelling-robust: a Kimi
// file branded with any of the normalized spellings derives the same deepseek2 MLA+MoE Config and
// classifies to the same recognized forward as one branded "kimi_k2".
func TestKimiConfigRecognizedRegardlessOfSpelling(t *testing.T) {
	for _, arch := range []string{"kimi-k2", "kimi_k2", "kimi2", "kimi-k3", "kimi_k3", "kimi3", "kimi"} {
		f := &File{Metadata: synthKimiMeta(arch)}
		cfg, err := f.Config()
		if err != nil {
			t.Fatalf("Config(%q): %v", arch, err)
		}
		if cfg.ModelType != "deepseek2" {
			t.Errorf("arch %q: ModelType = %q, want deepseek2", arch, cfg.ModelType)
		}
		got, err := model.ClassifyForwardPath(cfg, nil)
		if err != nil {
			t.Errorf("arch %q: classify refused: %v", arch, err)
			continue
		}
		if got != model.ForwardGLMDsaMLA {
			t.Errorf("arch %q: ClassifyForwardPath = %q, want %q", arch, got, model.ForwardGLMDsaMLA)
		}
	}
}
