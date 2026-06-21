package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestHALSessionMatchesLegacyCPUReference(t *testing.T) {
	cfg := Config{
		HiddenSize:        16,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           4,
		IntermediateSize:  32,
		VocabSize:         64,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        63,
	}
	m := NewSynthetic(cfg)
	be := compute.Pick("cpu-ref")
	if !compute.RequireReference(be) {
		t.Fatal("cpu-ref must be a reference backend")
	}

	prompt := []int{3, 7, 11, 19, 23}
	legacy := m.NewSession()
	hal := m.NewBackendSession(be)
	assertSameF32(t, "prefill", legacy.Prefill(prompt), hal.Prefill(prompt))

	for i, id := range []int{5, 13, 21} {
		assertSameF32(t, "step", legacy.Step(id), hal.Step(id))
		if legacy.Cache.Len() != len(prompt)+i+1 {
			t.Fatalf("legacy cache length drifted at step %d", i)
		}
		if hal.halKV.Len() != len(prompt)+i+1 {
			t.Fatalf("HAL KV length drifted at step %d: got %d", i, hal.halKV.Len())
		}
	}

	want := m.NewSession().Generate(prompt, 4)
	got := m.NewBackendSession(be).Generate(prompt, 4)
	if len(got) != len(want) {
		t.Fatalf("Generate length mismatch: got %d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Generate[%d]=%d want %d", i, got[i], want[i])
		}
	}
}

func assertSameF32(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length mismatch: got %d want %d", label, len(got), len(want))
	}
	for i := range want {
		// Byte-exact on amd64 (fmaCrossPathTol==0); within FMA noise on arches where gc
		// auto-fuses multiply-add — the legacy session and the HAL backend are two distinct Go
		// code paths over the same f32 math, so on arm64 they diverge sub-ULP (~1e-6), exactly
		// like the reposition/profiler rungs (see fmatol_other_test.go). The token-level
		// Generate check below stays exact, so a real divergence is still caught.
		d := got[i] - want[i]
		if d < 0 {
			d = -d
		}
		if float64(d) > fmaCrossPathTol {
			t.Fatalf("%s[%d] drift: got %v want %v (|Δ|=%.3e > %.0e, bits %x vs %x)",
				label, i, got[i], want[i], d, fmaCrossPathTol, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}
