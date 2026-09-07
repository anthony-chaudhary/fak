package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// Qwen3.8 architecture string recognition.
//
// Qwen3.8 checkpoints can declare general.architecture = "qwen3.8", "qwen38", "qwen-3.8",
// or "qwen-38". Like Qwen3.5 and Qwen3.6 (and the Bonsai repack), Qwen3.8 shares the hybrid
// Gated-DeltaNet / SSM architecture family ("qwen35"). canonicalGGUFArch normalizes all
// Qwen3.8 variants onto "qwen35" so that (*File).Config() sets ModelType = "qwen35", invokes
// the hybrid GDN configuration block (populating LayerTypes and ssm.* dimensions), and
// downstream model.ClassifyForwardPath reaches ForwardQwen35GDN instead of throwing
// UnsupportedArchError (#934). The metadata-key prefix stays the file's own spelling
// (e.g. "qwen3.8."), while ModelType and the hybrid gate normalize.

// synthQwen38Meta builds a minimal synthetic metadata map scaled down, keyed on the
// file's own arch prefix (e.g. "qwen3.8.") to verify that metadata reads use the raw
// prefix while recognition normalizes the arch string.
func synthQwen38Meta(arch string) map[string]Value {
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

// TestCanonicalGGUFArchQwen38 pins the arch-string normalization for Qwen3.8:
// all documented spellings ("qwen3.8", "qwen38", "qwen-3.8", "qwen-38") map onto
// the internal "qwen35" hybrid model_type, while the canonical "qwen35" is a no-op
// and unrelated architectures are passed through untouched.
func TestCanonicalGGUFArchQwen38(t *testing.T) {
	for _, spelling := range []string{"qwen3.8", "qwen38", "qwen-3.8", "qwen-38"} {
		if got := canonicalGGUFArch(spelling); got != "qwen35" {
			t.Errorf("canonicalGGUFArch(%q) = %q, want qwen35", spelling, got)
		}
	}
	if got := canonicalGGUFArch("qwen35"); got != "qwen35" {
		t.Errorf("canonicalGGUFArch(qwen35) = %q, want qwen35", got)
	}
	if got := canonicalGGUFArch("llama"); got != "llama" {
		t.Errorf("canonicalGGUFArch(llama) = %q, want llama (must not over-normalize)", got)
	}
}

// TestQwen38ConfigDerivesQwen35Hybrid verifies that a GGUF file declaring
// general.architecture = "qwen3.8" extracts a Config with ModelType = "qwen35",
// derives the hybrid GDN layer schedule, and populates the ssm.* and attention axes.
func TestQwen38ConfigDerivesQwen35Hybrid(t *testing.T) {
	f := &File{Metadata: synthQwen38Meta("qwen3.8")}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.ModelType != "qwen35" {
		t.Fatalf("ModelType = %q, want qwen35 (canonicalized from qwen3.8)", cfg.ModelType)
	}
	if !cfg.IsQwen35Hybrid() {
		t.Fatalf("IsQwen35Hybrid = false; layer_types = %v (hybrid schedule did not derive)", cfg.LayerTypes)
	}
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

// TestQwen38ConfigRecognizedRegardlessOfSpelling verifies that all Qwen3.8 spelling
// variants derive the hybrid schedule and map to ModelType = "qwen35".
func TestQwen38ConfigRecognizedRegardlessOfSpelling(t *testing.T) {
	for _, arch := range []string{"qwen3.8", "qwen38", "qwen-3.8", "qwen-38"} {
		f := &File{Metadata: synthQwen38Meta(arch)}
		cfg, err := f.Config()
		if err != nil {
			t.Fatalf("Config(%q): %v", arch, err)
		}
		if cfg.ModelType != "qwen35" {
			t.Errorf("arch %q: ModelType = %q, want qwen35", arch, cfg.ModelType)
		}
		if !cfg.IsQwen35Hybrid() {
			t.Errorf("arch %q: IsQwen35Hybrid = false, want true (layer_types=%v)", arch, cfg.LayerTypes)
		}
	}
}

// TestQwen38ConfigClassifiesToRecognizedGDNForward proves the end-to-end forward classification:
// a Qwen3.8 GGUF must reach ForwardQwen35GDN rather than failing with UnsupportedArchError (#934).
func TestQwen38ConfigClassifiesToRecognizedGDNForward(t *testing.T) {
	f := &File{Metadata: synthQwen38Meta("qwen3.8")}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	got, err := model.ClassifyForwardPath(cfg, nil)
	if err != nil {
		t.Fatalf("a recognized Qwen3.8 hybrid must classify, got refusal: %v", err)
	}
	if got != model.ForwardQwen35GDN {
		t.Fatalf("ClassifyForwardPath = %q, want %q (recognized qwen35 GDN, not the #934 refusal)",
			got, model.ForwardQwen35GDN)
	}
}
