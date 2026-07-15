package ggufload

import "testing"

// Bonsai / Qwen3.6-27B arch recognition (epic #4867, child #4869).
//
// prism-ml Ternary-Bonsai-27B is Qwen3.6-27B with the architecture UNCHANGED — the same
// Gated-DeltaNet/SSM hybrid the qwen35 family drives — with only the weights re-quantized to
// ternary (Q2_0). The mainline Qwen3.6-27B GGUF already declares general.architecture =
// "qwen35" (gguf_qwen35_nextn_test.go), but a PrismML repack may brand its own arch string.
// Before this fix, Config() gated the hybrid schedule (LayerTypes/ssm axes/AttnOutputGate)
// on the RAW general.architecture matching exactly "qwen35"/"qwen35moe", so a Bonsai file
// declaring any other spelling would derive an EMPTY layer_types — the #934 state that makes
// IsQwen35Hybrid false and the forward refuse (arch_support.go). canonicalGGUFArch now
// normalizes the documented Bonsai/Qwen3.6 spellings onto "qwen35", and the hybrid gate keys
// on the canonical arch, so recognition fires while the metadata-key PREFIX stays the file's
// own spelling.

// synthBonsaiMeta is the minimal Bonsai/Qwen3.6-27B hybrid metadata scaled down, keyed on the
// file's OWN "bonsai." prefix (not "qwen35.") to prove the metadata reads use the raw arch
// prefix while recognition normalizes the arch string. Shape mirrors synthQwen35Meta plus the
// ssm.* axes the Gated-DeltaNet linear-attention layers need.
func synthBonsaiMeta(arch string) map[string]Value {
	p := arch + "."
	return map[string]Value{
		"general.architecture":                 {Type: TypeString, Value: arch},
		p + "context_length":                   {Type: TypeUint64, Value: uint64(16)},
		p + "embedding_length":                 {Type: TypeUint64, Value: uint64(32)},
		p + "block_count":                      {Type: TypeUint64, Value: uint64(8)},
		p + "feed_forward_length":              {Type: TypeUint64, Value: uint64(64)},
		p + "attention.head_count":             {Type: TypeUint64, Value: uint64(4)},
		p + "attention.head_count_kv":          {Type: TypeUint64, Value: uint64(2)},
		p + "attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
		p + "full_attention_interval":          {Type: TypeUint64, Value: uint64(4)},
		p + "ssm.conv_kernel":                  {Type: TypeUint64, Value: uint64(4)},
		p + "ssm.state_size":                   {Type: TypeUint64, Value: uint64(128)},
		p + "ssm.group_count":                  {Type: TypeUint64, Value: uint64(2)},
		p + "ssm.time_step_rank":               {Type: TypeUint64, Value: uint64(8)},
	}
}

// TestCanonicalGGUFArchNormalizesBonsai pins the arch-string normalization: every documented
// Bonsai/Qwen3.6 spelling maps onto the internal "qwen35" hybrid model_type, while an
// unrelated arch is passed through untouched (no over-normalization).
func TestCanonicalGGUFArchNormalizesBonsai(t *testing.T) {
	for _, spelling := range []string{"bonsai", "ternary-bonsai", "qwen3.6", "qwen36"} {
		if got := canonicalGGUFArch(spelling); got != "qwen35" {
			t.Errorf("canonicalGGUFArch(%q) = %q, want qwen35", spelling, got)
		}
	}
	// The already-canonical spelling is a no-op, and an unrelated arch is untouched.
	if got := canonicalGGUFArch("qwen35"); got != "qwen35" {
		t.Errorf("canonicalGGUFArch(qwen35) = %q, want qwen35", got)
	}
	if got := canonicalGGUFArch("llama"); got != "llama" {
		t.Errorf("canonicalGGUFArch(llama) = %q, want llama (must not over-normalize)", got)
	}
}

// TestBonsaiConfigDerivesQwen35Hybrid is the #4869 done-condition witness: a Bonsai GGUF
// (branded general.architecture, own "bonsai." metadata prefix) derives a Config whose layer
// schedule + attention/ssm axes match the qwen35 hybrid — so IsQwen35Hybrid is true and the
// forward is the recognized GDN path, NOT the #934 empty-layer_types refusal.
func TestBonsaiConfigDerivesQwen35Hybrid(t *testing.T) {
	f := &File{Metadata: synthBonsaiMeta("bonsai")}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	// Arch normalized to the internal hybrid model_type (the metadata prefix stayed "bonsai.").
	if cfg.ModelType != "qwen35" {
		t.Fatalf("ModelType = %q, want qwen35 (canonicalized from bonsai)", cfg.ModelType)
	}
	if !cfg.IsQwen35Hybrid() {
		t.Fatalf("IsQwen35Hybrid = false; layer_types = %v (hybrid schedule did not derive)", cfg.LayerTypes)
	}
	// Layer schedule: 8 blocks, full_attention every 4th, linear_attention otherwise.
	if len(cfg.LayerTypes) != 8 {
		t.Fatalf("LayerTypes = %v, want 8 entries", cfg.LayerTypes)
	}
	for l, got := range cfg.LayerTypes {
		want := "linear_attention"
		if (l+1)%4 == 0 {
			want = "full_attention"
		}
		if got != want {
			t.Errorf("LayerTypes[%d] = %q, want %q", l, got, want)
		}
	}
	// Attention/norm hybrid axes and the ssm.* linear-attention geometry must be populated
	// from the "bonsai." prefixed keys.
	if !cfg.AttnOutputGate || !cfg.NormGain1p || !cfg.QKNorm {
		t.Errorf("hybrid axes = gate:%v norm_gain_1p:%v qk_norm:%v, want all true",
			cfg.AttnOutputGate, cfg.NormGain1p, cfg.QKNorm)
	}
	if cfg.FullAttentionInterval != 4 {
		t.Errorf("FullAttentionInterval = %d, want 4", cfg.FullAttentionInterval)
	}
	if cfg.LinearConvKernelDim != 4 {
		t.Errorf("LinearConvKernelDim = %d, want 4 (ssm.conv_kernel)", cfg.LinearConvKernelDim)
	}
	if cfg.LinearKeyHeadDim != 128 || cfg.LinearValueHeadDim != 128 {
		t.Errorf("Linear{Key,Value}HeadDim = %d/%d, want 128/128 (ssm.state_size)",
			cfg.LinearKeyHeadDim, cfg.LinearValueHeadDim)
	}
	if cfg.LinearNumKeyHeads != 2 {
		t.Errorf("LinearNumKeyHeads = %d, want 2 (ssm.group_count)", cfg.LinearNumKeyHeads)
	}
	if cfg.LinearNumValueHeads != 8 {
		t.Errorf("LinearNumValueHeads = %d, want 8 (ssm.time_step_rank)", cfg.LinearNumValueHeads)
	}
}

// TestBonsaiConfigRecognizedRegardlessOfSpelling proves the recognition is spelling-robust:
// a Bonsai file branded "qwen36" derives the same hybrid schedule as one branded "bonsai".
func TestBonsaiConfigRecognizedRegardlessOfSpelling(t *testing.T) {
	for _, arch := range []string{"bonsai", "qwen36", "qwen3.6", "ternary-bonsai"} {
		f := &File{Metadata: synthBonsaiMeta(arch)}
		cfg, err := f.Config()
		if err != nil {
			t.Fatalf("Config(%q): %v", arch, err)
		}
		if !cfg.IsQwen35Hybrid() {
			t.Errorf("arch %q: IsQwen35Hybrid = false, want true (layer_types=%v)", arch, cfg.LayerTypes)
		}
	}
}
