package model

import (
	"math"
	"testing"
)

// route_observer_test.go — acceptance tests for issue #2623 (the MoE expert-routing
// witness hook, the analogue of AttnObserver #852). Two invariants the view layer builds
// on, mirroring attn_observer_test.go:
//
//  1. Observer OFF == byte-identical: the logits of a forward pass with no observer are
//     Float32bits-equal to one where the observer is installed but the pass is the same —
//     emission never perturbs the routing math.
//  2. Observer ON: every emitted token names real, in-range experts with valid gate
//     weights, one emission per (layer, token), and the observer owns the buffers it gets.

// tinyRouteModel is a small synthetic Mixtral-style MoE model (real f32 weights) so Forward
// runs the genuine router -> top-k -> weighted-sum seam route() lives on.
func tinyRouteModel(t *testing.T) *Model {
	t.Helper()
	cfg := Config{
		HiddenSize: 16, NumLayers: 2, NumHeads: 4, NumKVHeads: 4, HeadDim: 4,
		IntermediateSize: 32, VocabSize: 24, RMSNormEps: 1e-5, RopeTheta: 10000,
		NumExperts: 6, NumExpertsPerTok: 2, NormTopKProb: true,
		ModelType: "llama", EOSTokenID: -1,
	}
	return NewSyntheticMoE(cfg)
}

// TestRouteObserverOffByteIdentical asserts the forward pass is Float32bits-identical with
// the observer absent vs. present-but-passive — routing emission never perturbs the math.
func TestRouteObserverOffByteIdentical(t *testing.T) {
	ids := []int{1, 5, 2, 7, 3, 9}

	m1 := tinyRouteModel(t)
	if m1.RouteObserverSet() {
		t.Fatalf("fresh model reports a route observer set; want none")
	}
	base := m1.Forward(ids)

	m2 := tinyRouteModel(t)
	var fires int
	m2.SetRouteObserver(func(layer, tokenPos int, experts []int, gateWeights []float32) {
		fires++
	})
	if !m2.RouteObserverSet() {
		t.Fatalf("after SetRouteObserver, RouteObserverSet()=false; want true")
	}
	obs := m2.Forward(ids)

	if fires == 0 {
		t.Fatalf("route observer installed but never invoked; the seam did not fire")
	}
	for t1 := range base.Logits {
		for i := range base.Logits[t1] {
			if math.Float32bits(base.Logits[t1][i]) != math.Float32bits(obs.Logits[t1][i]) {
				t.Fatalf("logits differ with route observer on (pos %d idx %d): off=%v on=%v — emission perturbed routing",
					t1, i, base.Logits[t1][i], obs.Logits[t1][i])
			}
		}
	}
}

// TestRouteObserverNilDefaultOff asserts the default is nil/off: a forward pass with no
// observer must not panic, must produce logits, and must not fire the seam.
func TestRouteObserverNilDefaultOff(t *testing.T) {
	m := tinyRouteModel(t)
	if m.RouteObserverSet() {
		t.Fatalf("default model has a route observer; want nil/off by default")
	}
	act := m.Forward([]int{1, 2, 3})
	if len(act.Logits) != 3 {
		t.Fatalf("Forward produced %d logit rows; want 3", len(act.Logits))
	}
}

// TestRouteObserverOnRowInvariants asserts, on every emission: layer and tokenPos in range,
// experts in [0,NumExperts) with len == the routed top-k, gate weights valid and positive,
// and exactly one emission per (layer, token).
func TestRouteObserverOnRowInvariants(t *testing.T) {
	m := tinyRouteModel(t)
	nL := m.Cfg.NumLayers
	E := m.Cfg.NumExperts
	K := m.Cfg.NumExpertsPerTok
	ids := []int{1, 5, 2, 7, 3, 9}

	var fires int
	m.SetRouteObserver(func(layer, tokenPos int, experts []int, gateWeights []float32) {
		fires++
		if layer < 0 || layer >= nL {
			t.Errorf("layer %d out of range [0,%d)", layer, nL)
		}
		if tokenPos < 0 || tokenPos >= len(ids) {
			t.Errorf("tokenPos %d out of range [0,%d)", tokenPos, len(ids))
		}
		if len(experts) != len(gateWeights) {
			t.Fatalf("experts len %d != gateWeights len %d", len(experts), len(gateWeights))
		}
		if len(experts) != K {
			t.Errorf("emitted %d picks; want top-k=%d", len(experts), K)
		}
		seen := map[int]bool{}
		for i, e := range experts {
			if e < 0 || e >= E {
				t.Errorf("expert %d out of range [0,%d)", e, E)
			}
			if seen[e] {
				t.Errorf("expert %d selected twice in one token's top-k", e)
			}
			seen[e] = true
			w := gateWeights[i]
			if w < 0 || math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) {
				t.Errorf("gate weight %v not a valid non-negative weight", w)
			}
		}
	})

	m.Forward(ids)

	// route() fires once per (layer, token): every MoE layer routes every position.
	want := nL * len(ids)
	if fires != want {
		t.Fatalf("emitted %d routing rows; want %d (nL=%d seq=%d)", fires, want, nL, len(ids))
	}
}

// TestRouteObserverOwnsBuffers asserts the observer receives freshly-allocated slices it may
// retain — mutating them after the callback must not affect the model or later emissions
// (proves the copy-out, so a retained expert_hist witness is safe for the view layer to keep).
func TestRouteObserverOwnsBuffers(t *testing.T) {
	m := tinyRouteModel(t)
	var keptExperts [][]int
	m.SetRouteObserver(func(layer, tokenPos int, experts []int, gateWeights []float32) {
		keptExperts = append(keptExperts, experts) // retain
		for i := range experts {
			experts[i] = -999 // mutate the retained slice
		}
		for i := range gateWeights {
			gateWeights[i] = -999
		}
	})
	m.Forward([]int{1, 2, 3, 4})
	if len(keptExperts) < 2 {
		t.Fatalf("expected multiple emissions, got %d", len(keptExperts))
	}
	// the first row we mutated to -999 must still be -999 (we own it); a shared alias into
	// the routePick scratch would have been overwritten by a later token's copy.
	for _, e := range keptExperts[0] {
		if e != -999 {
			t.Fatalf("retained expert row was overwritten (%v); observer does not own its buffer", e)
		}
	}
}
