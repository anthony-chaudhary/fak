package model

import (
	"errors"
	"strings"
	"testing"
)

// #934: the real Qwen3.6-27B GGUF is a Gated-DeltaNet/SSM hybrid (fused attn_qkv +
// per-layer ssm_*, no self_attn.q_proj). When its general.architecture is not
// recognized as qwen35, config.layer_types is left empty and the standard forward
// would panic on a missing self_attn.q_proj.weight on the first request. The load
// must instead fail with a typed, named UnsupportedArchError.

func TestRefuseUnsupportedHybridArchRefusesUnrecognizedGDN(t *testing.T) {
	// A manifest carrying the canonicalized ssm_* family (linear_attn.*) but a config
	// whose layer_types is empty (arch not recognized as qwen35) — the exact #934 state.
	man := map[string]tensorMeta{
		"model.layers.0.attn_norm.weight":             {},
		"model.layers.0.self_attn.qkv_proj.weight":    {}, // from fused attn_qkv (GDN in_proj)
		"model.layers.0.self_attn.q_gate_proj.weight": {},
		"model.layers.0.linear_attn.A_log":            {}, // from ssm_a
		"model.layers.0.linear_attn.out_proj.weight":  {},
	}
	cfg := Config{ModelType: "qwen3next", NumLayers: 1}
	if cfg.IsQwen35Hybrid() {
		t.Fatalf("precondition: empty layer_types must NOT be IsQwen35Hybrid")
	}

	err := refuseUnsupportedHybridArch(cfg, man)
	if err == nil {
		t.Fatalf("expected a typed refusal for a GDN/SSM checkpoint with empty layer_types, got nil")
	}
	var ua *UnsupportedArchError
	if !errors.As(err, &ua) {
		t.Fatalf("error is %T, want *UnsupportedArchError: %v", err, err)
	}
	if ua.Arch != "qwen3next" {
		t.Errorf("UnsupportedArchError.Arch = %q, want %q", ua.Arch, "qwen3next")
	}
	for _, want := range []string{"qwen3next", "self_attn.q_proj.weight", "#934", "linear_attn"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal message missing %q:\n%s", want, err.Error())
		}
	}
}

func TestRefuseUnsupportedHybridArchAllowsRecognizedHybrid(t *testing.T) {
	// Same GDN signature, but layer_types marks the linear-attention layers — the
	// supported qwen35-family path. Must NOT refuse.
	man := map[string]tensorMeta{
		"model.layers.0.linear_attn.A_log":           {},
		"model.layers.0.linear_attn.out_proj.weight": {},
	}
	cfg := Config{
		ModelType:  "qwen35",
		NumLayers:  2,
		LayerTypes: []string{"linear_attention", "full_attention"},
	}
	if !cfg.IsQwen35Hybrid() {
		t.Fatalf("precondition: layer_types with linear_attention must be IsQwen35Hybrid")
	}
	if err := refuseUnsupportedHybridArch(cfg, man); err != nil {
		t.Fatalf("recognized qwen35 hybrid must load, got refusal: %v", err)
	}
}

func TestRefuseUnsupportedHybridArchAllowsStandardArch(t *testing.T) {
	// A standard separate-projection attention checkpoint (no linear_attn.* tensors)
	// must not trip the GDN refusal even with empty layer_types.
	man := map[string]tensorMeta{
		"model.layers.0.self_attn.q_proj.weight": {},
		"model.layers.0.self_attn.k_proj.weight": {},
		"model.layers.0.self_attn.v_proj.weight": {},
		"model.layers.0.mlp.gate_proj.weight":    {},
	}
	cfg := Config{ModelType: "llama", NumLayers: 1}
	if err := refuseUnsupportedHybridArch(cfg, man); err != nil {
		t.Fatalf("standard arch must load, got refusal: %v", err)
	}
}

// TestNewFromF32TensorsRefusesUnsupportedHybridArch proves the public load path
// (every GGUF loader funnels through newModel) returns the typed refusal rather than
// constructing a Model that would panic in the forward — the #934 acceptance interim.
func TestNewFromF32TensorsRefusesUnsupportedHybridArch(t *testing.T) {
	tensors := []NamedTensorF32{
		{Name: "model.layers.0.linear_attn.A_log", Shape: []int{1}, Data: []float32{0}},
	}
	cfg := Config{ModelType: "qwen3next", NumLayers: 1}

	m, err := NewFromF32Tensors(cfg, tensors)
	if err == nil {
		t.Fatalf("NewFromF32Tensors must refuse an unrecognized GDN/SSM hybrid, got model %v", m)
	}
	if m != nil {
		t.Errorf("model must be nil on refusal, got %v", m)
	}
	var ua *UnsupportedArchError
	if !errors.As(err, &ua) {
		t.Fatalf("NewFromF32Tensors error is %T, want *UnsupportedArchError: %v", err, err)
	}
}
