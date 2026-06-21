package model

import (
	"math"
	"testing"
)

// longropeCfg is a synthetic Phi-style longrope config: head_dim 8 (so half=4),
// with explicit short/long per-dim factor vectors and a long window that exceeds
// the original, so ropeLongFactor pins the LONG vector.
func longropeCfg() Config {
	return Config{
		HiddenSize:            16,
		NumLayers:             2,
		NumHeads:              4,
		NumKVHeads:            2,
		HeadDim:               8,
		IntermediateSize:      24,
		VocabSize:             11,
		RMSNormEps:            1e-5,
		RopeTheta:             10000,
		EOSTokenID:            -1,
		MaxPositionEmbeddings: 131072,
		LongRope: &RopeScaling{
			Type:                          "longrope",
			OriginalMaxPositionEmbeddings: 4096,
			ShortFactor:                   []float64{1.0, 1.1, 1.2, 1.3},
			LongFactor:                    []float64{1.5, 2.0, 3.0, 4.0},
		},
	}
}

// TestLongropeInvFreqAppliesFactorPerDim is the longrope numeric witness: each
// inv_freq[j] equals the plain-theta base divided by the PINNED (long) factor[j].
// It is checked against a hand-computed reference, not the implementation.
func TestLongropeInvFreqAppliesFactorPerDim(t *testing.T) {
	cfg := longropeCfg()
	half := cfg.HeadDim / 2
	long := cfg.LongRope.LongFactor

	got := invFreq(cfg, 0)
	if len(got) != half {
		t.Fatalf("inv_freq len = %d, want %d", len(got), half)
	}
	for j := 0; j < half; j++ {
		base := 1.0 / math.Pow(cfg.RopeTheta, float64(2*j)/float64(cfg.HeadDim))
		want := base / long[j]
		if got[j] != want {
			t.Errorf("inv_freq[%d] = %v, want base(%v)/factor(%v) = %v", j, got[j], base, long[j], want)
		}
	}
}

// TestLongropeSelectsLongVsShort checks the pinned selection rule: the long factor
// when the model's full window exceeds the original, the short factor otherwise —
// and that the selection is a function of Config alone (no live length input).
func TestLongropeSelectsLongVsShort(t *testing.T) {
	long := longropeCfg()
	if got := ropeLongFactor(long); !float64sEqual(got, long.LongRope.LongFactor) {
		t.Errorf("expected long factor for max>orig, got %v", got)
	}

	short := longropeCfg()
	short.MaxPositionEmbeddings = short.LongRope.OriginalMaxPositionEmbeddings // not extended
	if got := ropeLongFactor(short); !float64sEqual(got, short.LongRope.ShortFactor) {
		t.Errorf("expected short factor for max<=orig, got %v", got)
	}
}

// TestLongropeFactorPinnedGuard exercises the session-lifetime guard the design
// (§3 O3) mandates: the factor is well-formed (length head_dim/2) and fixed, so a
// future change that re-introduced length-dependence or a malformed vector trips here.
func TestLongropeFactorPinnedGuard(t *testing.T) {
	if !longropeFactorPinned(longropeCfg()) {
		t.Fatalf("well-formed longrope config should be pinned")
	}
	bad := longropeCfg()
	bad.LongRope.LongFactor = []float64{1.0, 2.0} // wrong length (half=4)
	if longropeFactorPinned(bad) {
		t.Fatalf("malformed factor length should NOT pass the pinned guard")
	}
}

// TestLongropeAttnScaleTemperature checks the HF-effective attention-temperature
// warm-up against the reference formula and confirms it multiplies the base
// 1/sqrt(head_dim) scale. HF scales both rotated q and rotated k, so the score
// multiplier is the square of the rotary scale.
func TestLongropeAttnScaleTemperature(t *testing.T) {
	cfg := longropeCfg()
	orig := float64(cfg.LongRope.OriginalMaxPositionEmbeddings)
	maxp := float64(cfg.MaxPositionEmbeddings)
	rotaryScale := math.Sqrt(1.0 + math.Log(maxp/orig)/math.Log(orig))
	wantMul := rotaryScale * rotaryScale
	if got := longropeAttnScaleMul(cfg); got != wantMul {
		t.Errorf("attn-scale mul = %v, want %v", got, wantMul)
	}
	wantScale := float32((1.0 / math.Sqrt(float64(cfg.HeadDim))) * wantMul)
	if got := cfg.attnScale(); got != wantScale {
		t.Errorf("attnScale = %v, want %v", got, wantScale)
	}
	if wantMul <= 1.0 {
		t.Fatalf("longrope temperature should warm the score (>1), got %v", wantMul)
	}
}

// TestLongropeLlamaNoOp is the Llama no-op gate the RoPE path is touched under: a
// config with NO rope_scaling must produce the plain-theta inv_freq and the exact
// base attention scale — bit-identical to the pre-longrope behaviour.
func TestLongropeLlamaNoOp(t *testing.T) {
	cfg := Config{HeadDim: 8, RopeTheta: 10000} // no RopeScaling
	if ropeLongFactor(cfg) != nil {
		t.Fatalf("non-longrope config must have nil factor")
	}
	if longropeAttnScaleMul(cfg) != 1.0 {
		t.Fatalf("non-longrope attn-scale mul must be identity 1.0")
	}
	// inv_freq is exactly the plain formula.
	half := cfg.HeadDim / 2
	got := invFreq(cfg, 0)
	for j := 0; j < half; j++ {
		want := 1.0 / math.Pow(cfg.RopeTheta, float64(2*j)/float64(cfg.HeadDim))
		if got[j] != want {
			t.Errorf("Llama inv_freq[%d] = %v, want plain %v", j, got[j], want)
		}
	}
	// attnScale equals the exact base 1/sqrt(head_dim) (no temperature).
	wantScale := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	if got := cfg.attnScale(); got != wantScale {
		t.Errorf("Llama attnScale = %v, want base %v", got, wantScale)
	}
}

// TestLongropeEvictRepositionBitExact is the longrope Evict re-rotation gate: after a
// MIDDLE-span eviction (survivors follow the evicted span, so they MUST be
// repositioned), every survivor's post-RoPE K equals a SINGLE rotation of its
// pre-RoPE Kraw at its NEW index, drawn from the SAME pinned longrope inv_freq the
// prefill used. Because the factor is pinned to the session regime (not the shrunken
// live length), the re-rotation does not flip the factor, so the reposition stays
// bit-exact — the asymmetry evict_test.go traps, here under longrope.
func TestLongropeEvictRepositionBitExact(t *testing.T) {
	cfg := longropeCfg()
	m := NewSynthetic(cfg)

	// Prefill prefix ++ poison ++ tail so the tail survivors sit AFTER the evicted
	// middle span and Evict must reposition them.
	prefix := []int{1, 4, 7}
	poison := []int{2, 9}
	tail := []int{3, 6, 8, 5}
	all := append(append(append([]int{}, prefix...), poison...), tail...)

	s := m.NewSession()
	s.Prefill(all)
	if s.Cache.Len() != len(all) {
		t.Fatalf("pre-evict len %d != %d", s.Cache.Len(), len(all))
	}
	removed := s.Cache.Evict(len(prefix), len(poison))
	if removed != len(poison) || s.Cache.Len() != len(prefix)+len(tail) {
		t.Fatalf("evict removed %d, len %d", removed, s.Cache.Len())
	}

	w, hd, nKV := s.Cache.kvStride(), cfg.HeadDim, cfg.NumKVHeads
	for i := 0; i < s.Cache.Len(); i++ {
		for l := 0; l < cfg.NumLayers; l++ {
			c, sn := ropeRowForLayer(cfg, l, i) // the SAME pinned longrope inv_freq as prefill
			for h := 0; h < nKV; h++ {
				want := append([]float32(nil), s.Cache.Kraw[l][i*w+h*hd:i*w+(h+1)*hd]...)
				applyRopeRow(want, c, sn)
				got := s.Cache.K[l][i*w+h*hd : i*w+(h+1)*hd]
				assertFloat32BitsEqual(t, "longrope reposition L"+itoa(l)+" h"+itoa(h)+" pos"+itoa(i), want, got)
			}
		}
	}
}

// TestLongropeEvictEqualsNeverSaw proves the stronger end-to-end claim: a longrope
// session that prefills [prefix ++ poison ++ query], evicts the poison, then continues
// is Float32bits-identical (logits) to a session that prefilled [prefix ++ query] and
// NEVER saw the poison. The pinned factor is what makes the repositioned K exact.
func TestLongropeEvictEqualsNeverSaw(t *testing.T) {
	cfg := longropeCfg()
	m := NewSynthetic(cfg)
	prefix := []int{1, 4, 7}
	poison := []int{2, 9}
	query := []int{3, 6}

	// Run A: saw poison, then evicted it before the query.
	a := m.NewSession()
	a.Prefill(prefix)
	a.Prefill(poison)
	a.Cache.Evict(len(prefix), len(poison))
	gotLogits := a.Prefill(query)

	// Run B: never saw the poison.
	b := m.NewSession()
	b.Prefill(prefix)
	wantLogits := b.Prefill(query)

	assertFloat32BitsEqual(t, "longrope evict==never logits", wantLogits, gotLogits)
}

func float64sEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
