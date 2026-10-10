package model

import (
	"fmt"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestQuantizedHybridStepPublishesTargetHidden(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	if !cfg.IsHybrid() && !cfg.IsQwen35Hybrid() {
		t.Fatal("fixture must enter the shared-block quantized branch")
	}
	m := NewSynthetic(cfg)
	m.Quantize()
	s, ordinary, ref := m.NewSession(), m.NewSession(), m.NewSession()
	t.Cleanup(s.Close)
	t.Cleanup(ordinary.Close)
	t.Cleanup(ref.Close)
	s.Quant, ordinary.Quant, ref.Quant = true, true, true

	// Construct the raw Q8 residual independently of tokenHiddenQ's publication.
	rawStep := func(token, pos int) []float32 {
		x := append([]float32(nil), m.embedRows()[token*cfg.HiddenSize:(token+1)*cfg.HiddenSize]...)
		scaleEmbedInPlace(x, cfg)
		for layer := 0; layer < cfg.NumLayers; layer++ {
			cos, sin := ropeRowForLayer(cfg, layer, pos)
			x = ref.blockStep(layer, pos, x, cos, sin, sessionQ8Kernel{ref})
		}
		ref.Cache.appendPosition(pos, token)
		return x
	}

	var first []float32
	var prefix, refPrefix *KVCache
	for pos, token := range []int{3, 7, 11, 5} {
		// A disabled prefix must stay unavailable when capture is later enabled.
		if pos == 2 {
			s.captureTargetHidden = true
		}
		logits := s.Step(token)
		assertFloat32BitsEqual(t, "capture does not change logits", ordinary.Step(token), logits)
		raw := rawStep(token, pos)
		assertFloat32BitsEqual(t, "Q8 reference logits", ref.headQ(m.finalNorm(raw)), logits)
		if len(ordinary.targetHidden) != 0 || len(ordinary.targetHiddenTokens) != 0 {
			t.Fatal("capture-disabled session published target history")
		}
		if pos < 2 {
			if len(s.targetHidden) != 0 || len(s.targetHiddenTokens) != 0 {
				t.Fatal("disabled prefix allocated target history")
			}
			continue
		}
		got, err := s.TargetHiddenAt(pos)
		if err != nil {
			t.Fatalf("committed position %d: %v", pos, err)
		}
		assertFloat32BitsEqual(t, "raw Q8 residual", raw, got)
		if reflect.DeepEqual(got, m.finalNorm(raw)) {
			t.Fatal("fixture does not distinguish raw from normalized hidden")
		}
		got[0]++
		again, err := s.TargetHiddenAt(pos)
		if err != nil {
			t.Fatal(err)
		}
		assertFloat32BitsEqual(t, "defensive read", raw, again)
		if pos == 2 {
			first = again
			prefix, refPrefix = s.Cache.Clone(), ref.Cache.Clone()
		}
	}
	for _, pos := range []int{0, 1, 4} {
		if _, err := s.TargetHiddenAt(pos); err == nil {
			t.Fatalf("uncaptured position %d became available", pos)
		}
	}
	stillFirst, err := s.TargetHiddenAt(2)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32BitsEqual(t, "earlier row survives later decode", first, stillFirst)
	if !reflect.DeepEqual(s.targetHiddenTokens, []int{-1, -1, 11, 5}) {
		t.Fatalf("token history = %v", s.targetHiddenTokens)
	}

	// Restore full prefix snapshots, including recurrent state. Hybrid caches
	// deliberately reject span eviction; hidden history still contains the old tail.
	if prefix == nil || refPrefix == nil {
		t.Fatal("missing prefix snapshots")
	}
	s.Cache, ref.Cache = prefix, refPrefix
	if _, err := s.TargetHiddenAt(3); err == nil {
		t.Fatal("hidden row past the restored prefix remained visible")
	}
	s.Step(17)
	want := rawStep(17, 3)
	got, err := s.TargetHiddenAt(3)
	if err != nil {
		t.Fatal(err)
	}
	assertFloat32BitsEqual(t, "replacement raw residual", want, got)
	if len(s.targetHidden) != 4 || !reflect.DeepEqual(s.targetHiddenTokens, []int{-1, -1, 11, 17}) {
		t.Fatalf("replacement history = %v (%d rows)", s.targetHiddenTokens, len(s.targetHidden))
	}
	s.Cache.lineage.ids[3] = 19
	if _, err := s.TargetHiddenAt(3); err == nil {
		t.Fatal("mismatched cache token exposed a stale hidden row")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestQuantizedHybridTargetHiddenFailureOrdering(t *testing.T) {
	for _, quant := range []bool{false, true} {
		for _, finalNorm := range []bool{false, true} {
			t.Run(fmt.Sprintf("quant=%t/finalNorm=%t", quant, finalNorm), func(t *testing.T) {
				cfg := qwen35HybridTestCfg()
				m := NewSynthetic(cfg)
				m.Quantize()
				s := m.NewSession()
				t.Cleanup(s.Close)
				s.Quant, s.captureTargetHidden = quant, true
				missing := layerName(cfg.NumLayers-1, "input_layernorm.weight")
				if finalNorm {
					missing = "model.norm.weight"
				} else {
					// attentionNorms permits the post-attention norm as a fallback.
					delete(m.manifest, layerName(cfg.NumLayers-1, "post_attention_layernorm.weight"))
				}
				delete(m.manifest, missing)
				func() {
					defer func() {
						if got := recover(); fmt.Sprint(got) != "model: missing tensor "+missing {
							t.Fatalf("panic = %v, want missing %s", got, missing)
						}
					}()
					s.tokenHidden(3, 0)
				}()
				if finalNorm {
					// Match F32: cache and raw hidden are published before finalNorm.
					if s.Cache.Len() != 1 || len(s.targetHidden) != 1 || !reflect.DeepEqual(s.targetHiddenTokens, []int{3}) {
						t.Fatal("finalNorm failure changed pre-norm publication ordering")
					}
					if _, err := s.TargetHiddenAt(0); err != nil {
						t.Fatal(err)
					}
				} else if len(s.targetHidden) != 0 || len(s.targetHiddenTokens) != 0 {
					t.Fatal("block failure published a raw hidden row")
				}
			})
		}
	}
}
