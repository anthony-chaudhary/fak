package model

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestPrefillSerialNormIntoIntegration(t *testing.T) {
	oldWorkers := NumWorkers()
	if err := SetWorkers(1); err != nil {
		t.Fatal(err)
	}
	defer SetWorkers(oldWorkers)
	bits := func(label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s lengths %d != %d", label, len(got), len(want))
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s[%d] bits %08x != %08x", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
	// Unsupported flags retain the original bias/gain-aware path, including
	// LayerNorm's precedence when both flags are true; plain RMS ignores bias.
	for _, cfg := range []Config{{}, {LayerNorm: true}, {NormGain1p: true}, {LayerNorm: true, NormGain1p: true}} {
		x, w, bias := []float32{1, -2, 3}, []float32{0.5, 2, -1}, []float32{0.25, 0.5, 0.75}
		got := make([]float32, len(x))
		prefillNormIntoDisjoint(got, x, w, bias, 1e-5, cfg)
		bits("configured fallback", got, normCfg(x, w, bias, 1e-5, cfg))
	}
	// Reach the actual ordinary-f32 batched route, with an independent per-token
	// reference. Existing norm-bias tests cover the unchanged configured branch.
	cfg := llamaArchConfig()
	if q8PrefillNeedsTokenLoop(cfg) {
		t.Fatal("synthetic fixture bypasses batched route")
	}
	m := NewSynthetic(cfg)
	prompt := []int{3, 17, 5, 23, 41, 2, 19}
	bat, ref := m.NewSession(), m.NewSession()
	var hidden []float32
	for _, id := range prompt {
		hidden = ref.tokenHidden(id, ref.Cache.Len())
	}
	bits("batched logits", bat.head(bat.prefillBatched(prompt)), ref.head(hidden))
	for l := 0; l < cfg.NumLayers; l++ {
		bits("batched K", bat.Cache.K[l], ref.Cache.K[l])
		bits("batched Kraw", bat.Cache.Kraw[l], ref.Cache.Kraw[l])
		bits("batched V", bat.Cache.V[l], ref.Cache.V[l])
	}
}
