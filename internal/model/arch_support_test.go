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

// #3814 (child of epic #3809): ClassifyForwardPath is the host-verifiable half of the
// small-GQA serve-verify — "confirm it maps to a supported forward path (attnSeq GQA
// for llama3/dense-Qwen2.5; gemma4.go for Gemma)". Each candidate the epic selects
// must resolve, at LOAD time, to its named supported path — pinning the naming↔dispatch
// mapping so a serve/bench readout (child #4/#6) cannot silently fall through to a wrong
// forward. The on-Metal tok/s+RSS smoke (DoD path a) is device-gated and remains the
// promotion evidence; this contract is what the smoke asserts before it runs.
func TestClassifyForwardPathSelectedSmallGQACandidates(t *testing.T) {
	cases := []struct {
		name string
		cfg  Config
		want ForwardPathKind
	}{
		{"Llama-3.2-3B (llama3 GQA)", Config{ModelType: "llama", NumLayers: 1}, ForwardAttnSeqGQA},
		{"Qwen2.5-Coder-7B (dense Qwen2.5 GQA)", Config{ModelType: "qwen2", NumLayers: 1}, ForwardAttnSeqGQA},
		{"Gemma-4-4B (heterogeneous geometry)", Config{ModelType: "gemma4", NumLayers: 1}, ForwardGemma4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ClassifyForwardPath(tc.cfg, nil)
			if err != nil {
				t.Fatalf("selected supported candidate must classify, got refusal: %v", err)
			}
			if got != tc.want {
				t.Errorf("ClassifyForwardPath = %q, want %q", got, tc.want)
			}
		})
	}
}

// A recognized qwen35 hybrid (layer_types marks the linear-attention layers) passes the
// load gate and classifies to the wired GDN forward — not the #934 refusal, and not a
// silent attnSeq fall-through that would panic on the missing self_attn.q_proj.weight.
func TestClassifyForwardPathRecognizedQwen35Hybrid(t *testing.T) {
	cfg := Config{
		ModelType:  "qwen35",
		NumLayers:  2,
		LayerTypes: []string{"linear_attention", "full_attention"},
	}
	if !cfg.IsQwen35Hybrid() {
		t.Fatalf("precondition: layer_types with linear_attention must be IsQwen35Hybrid")
	}
	got, err := ClassifyForwardPath(cfg, nil)
	if err != nil {
		t.Fatalf("recognized qwen35 hybrid must classify, got refusal: %v", err)
	}
	if got != ForwardQwen35GDN {
		t.Errorf("ClassifyForwardPath = %q, want %q", got, ForwardQwen35GDN)
	}
}

// TestClassifyForwardPathRefusesUnrecognizedGDN proves the classifier surfaces the SAME
// typed refusal newModel returns for the #934 GDN/SSM state (linear_attn.* present,
// layer_types empty) instead of naming a supported path — so a serve/bench caller
// refuses at classification time, never mid-request.
func TestClassifyForwardPathRefusesUnrecognizedGDN(t *testing.T) {
	man := map[string]tensorMeta{
		"model.layers.0.linear_attn.A_log": {},
	}
	cfg := Config{ModelType: "qwen3next", NumLayers: 1}
	got, err := ClassifyForwardPath(cfg, man)
	if err == nil {
		t.Fatalf("unrecognized GDN hybrid must refuse, got path %q", got)
	}
	if got != "" {
		t.Errorf("path must be empty on refusal, got %q", got)
	}
	var ua *UnsupportedArchError
	if !errors.As(err, &ua) {
		t.Fatalf("error is %T, want *UnsupportedArchError: %v", err, err)
	}
}
