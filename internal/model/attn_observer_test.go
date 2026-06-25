package model

import (
	"math"
	"testing"
)

// attn_observer_test.go — acceptance tests for issue #852 (the attention-mass witness
// hook, rung 1 of #851). Two invariants the higher rungs build on:
//
//  1. Observer OFF == byte-identical: the logits of a forward pass with no observer set
//     are Float32bits-equal to one where the observer is installed but the pass is the
//     same. (We assert the observer does not perturb the math by comparing an
//     observed run's logits to an unobserved run's logits.)
//  2. Observer ON: every emitted row sums to ~1.0 (post-softmax invariant) and names
//     real, in-range, causal key positions.

// tinyObserverModel is a small dense Llama-style synthetic model (same shape family as
// tinyForwardBandModel) with real weights, so Forward runs the genuine attention seam.
func tinyObserverModel(t *testing.T) *Model {
	t.Helper()
	cfg := Config{
		HiddenSize: 16, NumLayers: 3, NumHeads: 4, NumKVHeads: 4, HeadDim: 4,
		IntermediateSize: 32, VocabSize: 24, RMSNormEps: 1e-5, RopeTheta: 10000,
		ModelType: "llama", EOSTokenID: -1,
	}
	return NewSynthetic(cfg)
}

// TestAttnObserverOffByteIdentical asserts the forward pass is Float32bits-identical
// with the observer absent vs. present-but-passive — i.e. emission never perturbs the
// math. The observed run installs a real observer that records rows; we then assert its
// logits equal the unobserved run's logits bit-for-bit.
func TestAttnObserverOffByteIdentical(t *testing.T) {
	ids := []int{1, 5, 2, 7, 3, 9}

	m1 := tinyObserverModel(t)
	if m1.AttnObserverSet() {
		t.Fatalf("fresh model reports an observer set; want none")
	}
	base := m1.Forward(ids)

	m2 := tinyObserverModel(t)
	var rows int
	m2.SetAttnObserver(func(layer, queryPos, head int, keyPositions []int, weights []float32) {
		rows++
	})
	if !m2.AttnObserverSet() {
		t.Fatalf("after SetAttnObserver, AttnObserverSet()=false; want true")
	}
	obs := m2.Forward(ids)

	if rows == 0 {
		t.Fatalf("observer installed but never invoked; the seam did not fire")
	}
	for t1 := range base.Logits {
		for i := range base.Logits[t1] {
			if math.Float32bits(base.Logits[t1][i]) != math.Float32bits(obs.Logits[t1][i]) {
				t.Fatalf("logits differ with observer on (pos %d idx %d): off=%v on=%v — emission perturbed the math",
					t1, i, base.Logits[t1][i], obs.Logits[t1][i])
			}
		}
	}
}

// TestAttnObserverNilZeroInvocation asserts that with no observer the seam does NOT
// allocate the per-row witness — there is nothing to observe. We can only check the
// invocation count indirectly (zero rows), which the byte-identical test also covers;
// here we make the zero-work intent explicit and assert the default really is off.
func TestAttnObserverNilDefaultOff(t *testing.T) {
	m := tinyObserverModel(t)
	if m.AttnObserverSet() {
		t.Fatalf("default model has an observer; want nil/off by default")
	}
	// A forward pass with no observer must not panic and must produce logits.
	act := m.Forward([]int{1, 2, 3})
	if len(act.Logits) != 3 {
		t.Fatalf("Forward produced %d logit rows; want 3", len(act.Logits))
	}
}

// TestAttnObserverOnRowInvariants asserts the post-softmax invariants on every emitted
// row: weights sum to ~1.0, key positions are in causal range [0, queryPos], strictly
// increasing and contiguous (the forward oracle path is full-causal, j0=0), and head /
// layer indices are in range.
func TestAttnObserverOnRowInvariants(t *testing.T) {
	m := tinyObserverModel(t)
	nH := m.Cfg.NumHeads
	nL := m.Cfg.NumLayers
	ids := []int{1, 5, 2, 7, 3, 9}

	var rows int
	m.SetAttnObserver(func(layer, queryPos, head int, keyPositions []int, weights []float32) {
		rows++
		if layer < 0 || layer >= nL {
			t.Errorf("layer %d out of range [0,%d)", layer, nL)
		}
		if head < 0 || head >= nH {
			t.Errorf("head %d out of range [0,%d)", head, nH)
		}
		if queryPos < 0 || queryPos >= len(ids) {
			t.Errorf("queryPos %d out of range [0,%d)", queryPos, len(ids))
		}
		if len(keyPositions) != len(weights) {
			t.Fatalf("keyPositions len %d != weights len %d", len(keyPositions), len(weights))
		}
		if len(weights) == 0 {
			t.Fatalf("empty attention row at layer %d pos %d head %d", layer, queryPos, head)
		}
		// post-softmax invariant: row sums to ~1.0
		var sum float64
		for _, w := range weights {
			if w < 0 || math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) {
				t.Errorf("weight %v not a valid probability", w)
			}
			sum += float64(w)
		}
		if math.Abs(sum-1.0) > 1e-4 {
			t.Errorf("attention row sum %v != 1.0 (layer %d pos %d head %d)", sum, layer, queryPos, head)
		}
		// key positions: causal, in-range, strictly increasing. The full-prefill oracle
		// path is full-causal so the row covers [0, queryPos] exactly.
		for i, kp := range keyPositions {
			if kp < 0 || kp > queryPos {
				t.Errorf("key position %d outside causal range [0,%d] (pos %d head %d)", kp, queryPos, queryPos, head)
			}
			if i > 0 && kp <= keyPositions[i-1] {
				t.Errorf("key positions not strictly increasing: %v", keyPositions)
			}
		}
		if keyPositions[0] != 0 || keyPositions[len(keyPositions)-1] != queryPos {
			t.Errorf("full-causal row should span [0,%d]; got [%d,%d]",
				queryPos, keyPositions[0], keyPositions[len(keyPositions)-1])
		}
	})

	m.Forward(ids)

	// every (layer, position, head) must have emitted exactly one row
	want := nL * len(ids) * nH
	if rows != want {
		t.Fatalf("emitted %d rows; want %d (nL=%d seq=%d nH=%d)", rows, want, nL, len(ids), nH)
	}
}

// TestAttnObserverOwnsBuffers asserts the observer receives freshly-allocated slices it
// may retain — mutating them after the callback must not affect the model or later rows
// (proves the copy-out, so a retained witness is safe for the higher rungs to keep).
func TestAttnObserverOwnsBuffers(t *testing.T) {
	m := tinyObserverModel(t)
	var kept [][]float32
	m.SetAttnObserver(func(layer, queryPos, head int, keyPositions []int, weights []float32) {
		kept = append(kept, weights) // retain
		for i := range weights {
			weights[i] = -999 // mutate the retained slice
		}
	})
	m.Forward([]int{1, 2, 3, 4})
	// If the slices were aliases into shared scratch, mutation would have corrupted
	// later rows or the math; the byte-identical test already covers the math. Here we
	// just assert each retained row is distinct backing memory (not all the same alias).
	if len(kept) < 2 {
		t.Fatalf("expected multiple rows, got %d", len(kept))
	}
	// the first row we mutated to -999 must still be -999 (we own it); a shared alias
	// would have been overwritten by a later row's copy.
	for _, w := range kept[0] {
		if w != -999 {
			t.Fatalf("retained row was overwritten (%v); observer does not own its buffer", w)
		}
	}
}
