package model

import "testing"

// TestPrefillExpertWarmEnvWiring pins the production wiring of the packed-prefill
// expert warm hook: the ring/offload setup arms it only under PrefillExpertWarmEnv,
// an armed hook fires once per layer in order on both packed lanes and asks for
// layer+1's last realized picks, and with the env off nothing is armed, nothing
// fires and logits stay bit-identical.
func TestPrefillExpertWarmEnvWiring(t *testing.T) {
	cfg := layerMajorTestConfig()
	const B, P = 2, 4
	prompts := layerMajorPrompts(B, P, cfg.VocabSize)
	placement := ExpertSpillPlacement{RingBytes: 1 << 20}

	for _, quant := range []bool{false, true} {
		name := "f32"
		if quant {
			name = "q8"
		}
		t.Run(name, func(t *testing.T) {
			if !batchRectFastPathOK(cfg, quant) {
				t.Fatal("premise broken: config no longer routes to the packed lane")
			}
			m := NewSynthetic(cfg)
			if quant {
				m.Quantize()
			}
			run := func(env string) ([][]float32, []*prefillNextLayerWarmer) {
				t.Setenv(PrefillExpertWarmEnv, env)
				bs := m.NewBatchSession(B)
				bs.SetQuant(quant)
				var ws []*prefillNextLayerWarmer
				for _, s := range bs.Seqs {
					s.ApplyExpertSpillPlacement(placement)
					if w := s.prefillWarmer(); w != nil {
						ws = append(ws, w)
					}
				}
				return bs.PrefillEach(prompts), ws
			}

			offLogits, offWarmers := run("")
			if len(offWarmers) != 0 {
				t.Fatalf("env off: %d sessions armed, want 0", len(offWarmers))
			}

			t.Setenv(PrefillExpertWarmEnv, "1")
			bs := m.NewBatchSession(B)
			bs.SetQuant(quant)
			var ws []*prefillNextLayerWarmer
			for _, s := range bs.Seqs {
				s.ApplyExpertSpillPlacement(placement)
				w := s.prefillWarmer()
				if w == nil {
					t.Fatal("env on with a ring budget: session not armed")
				}
				ws = append(ws, w)
			}
			// Seed a realized routing for every layer so each fire has a layer+1 set.
			var layers, asked []int
			ws[0].observe = func(l int, next []routePick) {
				layers = append(layers, l)
				if len(next) > 0 {
					asked = append(asked, next[0].expert)
				}
			}
			for l := 0; l < cfg.NumLayers; l++ {
				bs.Seqs[0].recordPrefillWarmPicks(l, []routePick{{expert: 100 + l}})
			}
			onLogits := bs.PrefillEach(prompts)

			if got := ws[0].fires.Load(); got != int64(cfg.NumLayers) {
				t.Fatalf("env on: %d fires, want %d (one per layer)", got, cfg.NumLayers)
			}
			for _, w := range ws[1:] {
				if got := w.fires.Load(); got != 0 {
					t.Fatalf("second panel session fired %d times; the panel warms once per layer", got)
				}
			}
			for i, l := range layers {
				if l != i {
					t.Fatalf("fire %d at layer %d, want %d (layers=%v)", i, l, i, layers)
				}
			}
			if len(asked) != cfg.NumLayers-1 {
				t.Fatalf("asked layer+1 sets on %d fires, want %d (last layer has no successor)", len(asked), cfg.NumLayers-1)
			}
			for i, e := range asked {
				if e != 100+i+1 {
					t.Fatalf("fire at layer %d staged expert %d, want layer %d's set (%d)", i, e, i+1, 100+i+1)
				}
			}
			assertBitIdenticalLogits(t, onLogits, offLogits, name+" env-on vs env-off")
		})
	}
}

// TestPrefillExpertWarmEnvNeedsRing pins that the env alone arms nothing: a session
// with no expert ring budget stays disarmed.
func TestPrefillExpertWarmEnvNeedsRing(t *testing.T) {
	t.Setenv(PrefillExpertWarmEnv, "on")
	s := NewSynthetic(layerMajorTestConfig()).NewSession()
	s.ApplyExpertSpillPlacement(ExpertSpillPlacement{})
	if s.prefillWarmer() != nil {
		t.Fatal("zero ring budget armed the warm hook")
	}
}
