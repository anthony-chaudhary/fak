package model

import (
	"math"
	"testing"
)

// recomputeGemma4Inv is the pre-cache inline build, kept verbatim as the witness
// oracle: gemma4InvFreq must return a table bit-identical to this for every key.
func recomputeGemma4Inv(cfg Config, l, ropeDim int, ropeFreqs []float64) []float64 {
	half := ropeDim / 2
	theta := cfg.ropeThetaForLayer(l)
	inv := make([]float64, half)
	for j := 0; j < half; j++ {
		inv[j] = 1.0 / math.Pow(theta, float64(2*j)/float64(ropeDim))
	}
	if !cfg.gemma4LayerIsSliding(l) && ropeFreqs != nil && !gemma4SkipRopeFreqs() {
		for j := 0; j < half && j < len(ropeFreqs); j++ {
			if ropeFreqs[j] != 0 {
				inv[j] /= ropeFreqs[j]
			}
		}
	}
	return inv
}

func bitsEqual(t *testing.T, name string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len got=%d want=%d", name, len(got), len(want))
	}
	for i := range got {
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
			t.Fatalf("%s: inv[%d] not bit-identical: got %v (%#x) want %v (%#x)",
				name, i, got[i], math.Float64bits(got[i]), want[i], math.Float64bits(want[i]))
		}
	}
}

// TestGemma4InvFreqBitIdentical proves the memoized inv_freq table equals a fresh
// inline recompute, bit-for-bit, across the axes that differ per layer: sliding vs
// non-sliding (rope_freqs division only on non-sliding), distinct per-layer theta
// and ropeDim, and the absent-rope_freqs case. A cache hit must also return the
// same table as the first miss.
func TestGemma4InvFreqBitIdentical(t *testing.T) {
	cfg := Config{
		HeadDim:           8,
		RopeTheta:         10000,
		RopeThetaPerLayer: []float64{1000000, 10000},
		RopeDimPerLayer:   []int{8, 4},
		// layer 0 sliding (no rope_freqs division), layer 1 full (divides).
		LayerTypes: []string{"sliding_attention", "full_attention"},
	}
	m := &Model{Cfg: cfg}
	ropeFreqs := []float64{1.5, 2.0, 0.0, 3.25}

	cases := []struct {
		name    string
		layer   int
		ropeDim int
		freqs   []float64
	}{
		{"sliding/with-freqs-ignored", 0, 8, ropeFreqs},
		{"full/with-freqs", 1, 4, ropeFreqs},
		{"full/nil-freqs", 1, 4, nil},
	}
	for _, c := range cases {
		want := recomputeGemma4Inv(cfg, c.layer, c.ropeDim, c.freqs)
		got1 := m.gemma4InvFreq(c.layer, c.ropeDim, c.freqs)
		bitsEqual(t, c.name+" (miss)", got1, want)
		got2 := m.gemma4InvFreq(c.layer, c.ropeDim, c.freqs)
		bitsEqual(t, c.name+" (hit)", got2, want)
	}
}
