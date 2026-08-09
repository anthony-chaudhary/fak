package model

import (
	"errors"
	"strings"
	"testing"
)

// TestResidentQuantHeterogeneousGeometryRefusesByName is the contract for issue #4274, kept
// intact by #5495: the GENERIC uniform-geometry resident-quant band must refuse BY NAME — a
// typed *ResidentQuantUnsupportedError — for any per-layer-head_dim checkpoint that reaches
// it, instead of the cryptic "qk-norm weight length does not match head_dim" panic deep in
// the band. #5495 routes a gemma4 SESSION away from this band (Prefill/Step dispatch to the
// dedicated forward first), which shrinks the refusal's reachable domain; it does not and
// must not delete the guard, because the band is still reachable for a heterogeneous geometry
// with NO dedicated forward at all. The check reads only cfg + resident-quant flags, so it
// needs no weights and no forward run.
func TestResidentQuantHeterogeneousGeometryRefusesByName(t *testing.T) {
	cases := []struct {
		name      string
		modelType string
		// wantNoForward is true when the message must say no dedicated forward exists for
		// this geometry (the remedy is the f32 path), rather than naming a session mode.
		wantNoForward bool
	}{
		// A heterogeneous-head_dim architecture with no dedicated forward: the population the
		// refusal exists for after #5495. (archFamilyKey lowercases and strips separators, so
		// this name is spelled the way the key will read it.)
		{"unknown-heterogeneous-arch", "multiregime", true},
		// gemma4 reaching the generic band anyway (a lane #5495 did not route, e.g. the batched
		// BatchSession path): the guard must still fire rather than panic inside the band.
		{"gemma4-reaching-the-generic-band", "gemma4", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Local layers head_dim 128, global layers 256; scalar HeadDim 256.
			cfg := Config{
				ModelType:       tc.modelType,
				HeadDim:         256,
				HeadDimPerLayer: []int{128, 256, 128, 256},
				QKNorm:          true,
			}
			s := &Session{M: &Model{Cfg: cfg}, Q4K: true}

			err := s.residentQuantForwardUnsupported()
			if err == nil {
				t.Fatal("resident Q4_K heterogeneous-geometry session must refuse by name, got nil")
			}
			var rqe *ResidentQuantUnsupportedError
			if !errors.As(err, &rqe) {
				t.Fatalf("want *ResidentQuantUnsupportedError, got %T: %v", err, err)
			}
			if rqe.Format != "Q4_K" {
				t.Errorf("Format = %q, want Q4_K", rqe.Format)
			}
			if !strings.Contains(rqe.Arch, tc.modelType) {
				t.Errorf("Arch = %q, want it to name %q", rqe.Arch, tc.modelType)
			}
			if !strings.Contains(rqe.Error(), "#4274") {
				t.Errorf("Error() = %q, want it to cite issue #4274", rqe.Error())
			}
			if tc.wantNoForward && !strings.Contains(rqe.Error(), "No dedicated forward is wired") {
				t.Errorf("Error() = %q, want it to say no dedicated forward is wired for this geometry", rqe.Error())
			}

			// requireResidentQuantForwardSupported (the forward-entry guard) must panic with that
			// SAME typed error — the named refusal that replaces the deep band panic.
			func() {
				defer func() {
					r := recover()
					if r == nil {
						t.Fatal("requireResidentQuantForwardSupported must panic for a heterogeneous resident-quant geometry")
					}
					if _, ok := r.(*ResidentQuantUnsupportedError); !ok {
						t.Fatalf("panic value = %T, want *ResidentQuantUnsupportedError", r)
					}
				}()
				s.requireResidentQuantForwardSupported()
			}()
		})
	}
}

// TestGemma4UnwiredSessionModeRefusesByName is the second half of the #5495 domain shrink:
// the dedicated forward is wired for the HOST resident path only, so a gemma4 session that
// also carries a device/Metal/dynamic-precision mode must fail closed at the bridge entry —
// naming that mode — instead of silently running a uniform-geometry hand-copy of the block.
func TestGemma4UnwiredSessionModeRefusesByName(t *testing.T) {
	cfg := Config{ModelType: "gemma4", HeadDim: 256, HeadDimPerLayer: []int{128, 256}, QKNorm: true}
	cases := map[string]struct {
		mutate   func(*Session)
		wantMode string
	}{
		"metal":             {func(s *Session) { s.Metal = true }, "Metal prefill"},
		"metal-q4k":         {func(s *Session) { s.MetalQ4K = true }, "Metal Q4_K prefill"},
		"dynamic-precision": {func(s *Session) { s.PrecisionPolicy = &DynamicPrecisionPolicy{} }, "dynamic whole-token precision"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			s := &Session{M: &Model{Cfg: cfg}, Quant: true}
			tc.mutate(s)
			if s.gemma4SessionModeWired() {
				t.Fatalf("%s session must not report the gemma4 bridge as wired", name)
			}
			defer func() {
				r := recover()
				if r == nil {
					t.Fatalf("gemma4 %s session must fail closed, got no panic", name)
				}
				rqe, ok := r.(*ResidentQuantUnsupportedError)
				if !ok {
					t.Fatalf("panic value = %T, want *ResidentQuantUnsupportedError", r)
				}
				if rqe.Mode != tc.wantMode {
					t.Errorf("Mode = %q, want %q", rqe.Mode, tc.wantMode)
				}
				if rqe.Format != "Q8_0" {
					t.Errorf("Format = %q, want Q8_0", rqe.Format)
				}
				if !strings.Contains(rqe.Error(), tc.wantMode) || !strings.Contains(rqe.Error(), "#5495") {
					t.Errorf("Error() = %q, want it to name the mode and cite issue #5495", rqe.Error())
				}
			}()
			s.requireGemma4Session()
		})
	}

	// The host resident path IS wired for every resident store, so the bridge must not refuse
	// any of them — that is the capability #5495 opens.
	for name, s := range map[string]*Session{
		"f32":  {M: &Model{Cfg: cfg}},
		"q8":   {M: &Model{Cfg: cfg}, Quant: true},
		"q4":   {M: &Model{Cfg: cfg}, Q4: true},
		"q4k":  {M: &Model{Cfg: cfg}, Q4K: true},
		"gptq": {M: &Model{Cfg: cfg}, GPTQ: true},
	} {
		if !s.gemma4SessionModeWired() {
			t.Errorf("host resident gemma4 %s session must be wired", name)
		}
		s.requireGemma4Session() // must not panic
	}
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
