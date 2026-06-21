package model

import (
	"math"
	"testing"
)

// TestLayerLogitsLastMatchesForward pins the logit lens to the real forward pass: the
// last layer's lens projection must equal Forward's own final logits at that position.
// out[last] runs the IDENTICAL finalNorm + head + scale on the IDENTICAL Hidden[last]
// vector that Forward uses, so they agree up to floating-point reassociation. If the
// lens ever drifts from the projection Forward actually performs, this goes red — so
// the visual debugger always shows the model's real math, never a stale copy.
func TestLayerLogitsLastMatchesForward(t *testing.T) {
	cfg := llamaArchConfig()
	m := NewSynthetic(cfg)
	prompt := []int{3, 17, 5, 23, 41, 2, 19}

	act := m.Forward(prompt)
	for _, pos := range []int{0, 3, len(prompt) - 1} {
		lens := m.LayerLogits(act, pos)
		if len(lens) != cfg.NumLayers+1 {
			t.Fatalf("pos %d: got %d layers, want %d", pos, len(lens), cfg.NumLayers+1)
		}
		last := lens[len(lens)-1]
		want := act.Logits[pos]
		if len(last) != len(want) {
			t.Fatalf("pos %d: lens vocab %d != forward vocab %d", pos, len(last), len(want))
		}
		var maxAbs float64
		for i := range want {
			d := math.Abs(float64(last[i] - want[i]))
			if d > maxAbs {
				maxAbs = d
			}
		}
		// Same inputs, same ops -> bit-identical on amd64, FMA-noise elsewhere.
		if maxAbs > 1e-3 {
			t.Fatalf("pos %d: lens[last] diverges from Forward logits, max|Δ|=%g", pos, maxAbs)
		}
		// argmax must match exactly (token-level correctness).
		if argmax(last) != argmax(want) {
			t.Fatalf("pos %d: lens[last] argmax %d != Forward argmax %d", pos, argmax(last), argmax(want))
		}
	}
}

// TestLayerLogitsShapeAndBounds checks the per-layer shape and that early layers are
// genuinely different from the final layer (the lens is reading evolving state, not a
// constant) — i.e. the hook is wired, not dead.
func TestLayerLogitsShapeAndBounds(t *testing.T) {
	cfg := llamaArchConfig()
	m := NewSynthetic(cfg)
	prompt := []int{3, 17, 5, 23, 41, 2, 19}
	act := m.Forward(prompt)
	pos := len(prompt) - 1

	lens := m.LayerLogits(act, pos)
	for l, lg := range lens {
		if len(lg) != cfg.VocabSize {
			t.Fatalf("layer %d: vocab %d != %d", l, len(lg), cfg.VocabSize)
		}
	}
	// Embedding-level (layer 0) prediction should differ from the final layer; if they
	// were identical the residual stream wouldn't be evolving and the lens would be
	// pointless to show.
	if argmax(lens[0]) == argmax(lens[len(lens)-1]) {
		// Not strictly guaranteed, but with random synthetic weights an identical
		// argmax across all depth is a strong signal the slicing is wrong.
		var diff float64
		for i := range lens[0] {
			diff += math.Abs(float64(lens[0][i] - lens[len(lens)-1][i]))
		}
		if diff < 1e-6 {
			t.Fatalf("layer 0 logits identical to final layer — lens not reading per-layer state")
		}
	}

	// Out-of-range position returns nil.
	if got := m.LayerLogits(act, act.Seq); got != nil {
		t.Fatalf("out-of-range pos: want nil, got %d layers", len(got))
	}
}

// TestTopK validates ranking, clamping, and that probabilities are a real softmax.
func TestTopK(t *testing.T) {
	logits := []float32{0.1, 3.0, -1.0, 2.5, 0.0}
	top := TopK(logits, 3)
	if len(top) != 3 {
		t.Fatalf("want 3, got %d", len(top))
	}
	if top[0].ID != 1 || top[1].ID != 3 || top[2].ID != 0 {
		t.Fatalf("ranking wrong: %+v", top)
	}
	// Descending probability.
	for i := 1; i < len(top); i++ {
		if top[i].Prob > top[i-1].Prob {
			t.Fatalf("not sorted by prob: %+v", top)
		}
	}
	// Full-vocab softmax: recompute the top prob and compare.
	var sum float64
	for _, v := range logits {
		sum += math.Exp(float64(v))
	}
	wantP0 := math.Exp(float64(logits[1])) / sum
	if math.Abs(float64(top[0].Prob)-wantP0) > 1e-4 {
		t.Fatalf("top prob %g != softmax %g", top[0].Prob, wantP0)
	}
	// k clamps to vocab.
	if got := TopK(logits, 100); len(got) != len(logits) {
		t.Fatalf("k>vocab: want %d, got %d", len(logits), len(got))
	}
	if got := TopK(nil, 5); got != nil {
		t.Fatalf("nil logits: want nil, got %v", got)
	}
}
