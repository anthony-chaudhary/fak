package model

import (
	"errors"
	"strings"
	"testing"
)

// TestResidentQuantGemma4RefusesByName is the contract for issue #4274: a resident-quant CPU
// session over a gemma4-style checkpoint (per-layer head_dim) must refuse BY NAME — a typed
// *ResidentQuantUnsupportedError — instead of the cryptic "qk-norm weight length does not match
// head_dim" panic deep in the qk-norm band. The check reads only cfg + resident-quant flags,
// so it needs no weights and no forward run.
func TestResidentQuantGemma4RefusesByName(t *testing.T) {
	// gemma4-style geometry: local layers head_dim 128, global layers 256; scalar HeadDim 256.
	cfg := Config{
		ModelType:       "gemma4",
		HeadDim:         256,
		HeadDimPerLayer: []int{128, 256, 128, 256},
		QKNorm:          true,
	}
	s := &Session{M: &Model{Cfg: cfg}, Q4K: true}

	err := s.residentQuantForwardUnsupported()
	if err == nil {
		t.Fatal("resident Q4_K gemma4 session must refuse by name, got nil")
	}
	var rqe *ResidentQuantUnsupportedError
	if !errors.As(err, &rqe) {
		t.Fatalf("want *ResidentQuantUnsupportedError, got %T: %v", err, err)
	}
	if rqe.Format != "Q4_K" {
		t.Errorf("Format = %q, want Q4_K", rqe.Format)
	}
	if !strings.Contains(rqe.Arch, "gemma4") {
		t.Errorf("Arch = %q, want it to name gemma4", rqe.Arch)
	}
	if !strings.Contains(rqe.Error(), "#4274") {
		t.Errorf("Error() = %q, want it to cite issue #4274", rqe.Error())
	}

	// requireResidentQuantForwardSupported (the forward-entry guard) must panic with that SAME
	// typed error — the named refusal that replaces the deep band panic.
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("requireResidentQuantForwardSupported must panic for gemma4 resident-quant")
		}
		if _, ok := r.(*ResidentQuantUnsupportedError); !ok {
			t.Fatalf("panic value = %T, want *ResidentQuantUnsupportedError", r)
		}
	}()
	s.requireResidentQuantForwardSupported()
}

// TestResidentQuantUniformGeometryNotRefused pins the other half of the contract: the guard is
// inert for uniform-geometry arches (the shared resident-quant path must stay open for every
// non-gemma4 model), and inert for the f32 gemma4 session that legitimately uses the dedicated
// gemma4.go forward — the fix must not close the working path.
func TestResidentQuantUniformGeometryNotRefused(t *testing.T) {
	uniform := Config{ModelType: "qwen2", HeadDim: 128, QKNorm: true}
	for name, s := range map[string]*Session{
		"q4k":  {M: &Model{Cfg: uniform}, Q4K: true},
		"q4":   {M: &Model{Cfg: uniform}, Q4: true},
		"q8":   {M: &Model{Cfg: uniform}, Quant: true},
		"gptq": {M: &Model{Cfg: uniform}, GPTQ: true},
		"f32":  {M: &Model{Cfg: uniform}},
	} {
		if err := s.residentQuantForwardUnsupported(); err != nil {
			t.Errorf("uniform-geometry %s session must not be refused, got %v", name, err)
		}
	}

	// A homogeneous per-layer slice (every entry == scalar HeadDim) is still uniform geometry.
	homog := Config{ModelType: "qwen2", HeadDim: 128, HeadDimPerLayer: []int{128, 128}, QKNorm: true}
	if err := (&Session{M: &Model{Cfg: homog}, Q4K: true}).residentQuantForwardUnsupported(); err != nil {
		t.Errorf("homogeneous per-layer head_dim must not be refused, got %v", err)
	}

	// f32 gemma4 (no resident-quant flag) uses the dedicated gemma4.go path — the guard must
	// stay out of its way even though the geometry is heterogeneous.
	gemma4f32 := Config{ModelType: "gemma4", HeadDim: 256, HeadDimPerLayer: []int{128, 256}, QKNorm: true}
	if err := (&Session{M: &Model{Cfg: gemma4f32}}).residentQuantForwardUnsupported(); err != nil {
		t.Errorf("f32 gemma4 session must not be refused (dedicated forward), got %v", err)
	}
}

// TestHeterogeneousHeadDim is the pure predicate the refusal turns on.
func TestHeterogeneousHeadDim(t *testing.T) {
	cases := []struct {
		name string
		cfg  Config
		want bool
	}{
		{"none", Config{HeadDim: 128}, false},
		{"homogeneous", Config{HeadDim: 128, HeadDimPerLayer: []int{128, 128, 128}}, false},
		{"gemma4-local-global", Config{HeadDim: 256, HeadDimPerLayer: []int{128, 256}}, true},
		{"zero-padded-uniform", Config{HeadDim: 128, HeadDimPerLayer: []int{0, 128}}, false},
	}
	for _, tc := range cases {
		if got := tc.cfg.heterogeneousHeadDim(); got != tc.want {
			t.Errorf("%s: heterogeneousHeadDim() = %v, want %v", tc.name, got, tc.want)
		}
	}
}
