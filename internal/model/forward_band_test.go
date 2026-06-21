package model

import (
	"testing"
)

// forward_band_test.go — pipeline-parallel band execution. A pipeline-parallel
// worker runs only ITS contiguous band of layers and hands the hidden state to
// the next worker. ForwardBand is that unit; the correctness proof here is that
// splitting the monolithic Forward into bands and threading hidden states
// produces bit-identical logits — so partitioning the model across workers
// changes nothing numerically.

// TestForwardBandTwoWayMatchesMonolithic proves a two-band split (embed+[0,k)
// then [k,N)+head) reproduces the full Forward logits exactly. This is the
// pipeline-parallel correctness gate for a dense (Llama-style) model.
func TestForwardBandTwoWayMatchesMonolithic(t *testing.T) {
	m := tinyForwardBandModel(t)
	ids := []int{1, 2, 3, 4, 5}

	full := m.Forward(ids)

	// Band A: embed + layers [0, k). Band B: layers [k, N) + final norm + head.
	k := m.Cfg.NumLayers / 2
	if k == 0 {
		k = 1
	}
	xA := m.embedBand(ids)
	if _, _, err := m.ForwardBand(xA, 0, k, false); err != nil {
		t.Fatalf("band A [0,%d): %v", k, err)
	}
	_, logits, err := m.ForwardBand(xA, k, m.Cfg.NumLayers, true)
	if err != nil {
		t.Fatalf("band B [%d,%d): %v", k, m.Cfg.NumLayers, err)
	}

	if len(logits) != len(full.Logits) {
		t.Fatalf("band logits seq = %d, full = %d", len(logits), len(full.Logits))
	}
	var maxAbs float32
	for t2 := range logits {
		if len(logits[t2]) != len(full.Logits[t2]) {
			t.Fatalf("band logits[%d] len = %d, full = %d", t2, len(logits[t2]), len(full.Logits[t2]))
		}
		for i := range logits[t2] {
			d := logits[t2][i] - full.Logits[t2][i]
			if d < 0 {
				d = -d
			}
			if d > maxAbs {
				maxAbs = d
			}
		}
	}
	if maxAbs != 0 {
		t.Fatalf("two-band split logits differ from monolithic: max|delta|=%.3e (want bit-exact 0)", maxAbs)
	}
}

// TestForwardBandPerLayerMatchesMonolithic runs every layer as its own band and
// checks the result still matches — the maximally-partitioned case.
func TestForwardBandPerLayerMatchesMonolithic(t *testing.T) {
	m := tinyForwardBandModel(t)
	ids := []int{7, 1, 9}
	full := m.Forward(ids)

	x := m.embedBand(ids)
	N := m.Cfg.NumLayers
	for l := 0; l < N; l++ {
		isLast := l == N-1
		_, logits, err := m.ForwardBand(x, l, l+1, isLast)
		if err != nil {
			t.Fatalf("per-layer band [%d,%d): %v", l, l+1, err)
		}
		if isLast {
			var maxAbs float32
			for t2 := range logits {
				for i := range logits[t2] {
					d := logits[t2][i] - full.Logits[t2][i]
					if d < 0 {
						d = -d
					}
					if d > maxAbs {
						maxAbs = d
					}
				}
			}
			if maxAbs != 0 {
				t.Fatalf("per-layer band logits differ: max|delta|=%.3e (want 0)", maxAbs)
			}
		}
	}
}

// TestForwardBandRejectsBadRange proves ForwardBand fails closed on a malformed
// band rather than silently producing garbage.
func TestForwardBandRejectsBadRange(t *testing.T) {
	m := tinyForwardBandModel(t)
	x := m.embedBand([]int{1, 2})
	if _, _, err := m.ForwardBand(x, 2, 1, false); err == nil {
		t.Fatalf("ForwardBand(hi<lo) returned nil error; want a rejection")
	}
	if _, _, err := m.ForwardBand(x, -1, 1, false); err == nil {
		t.Fatalf("ForwardBand(lo<0) returned nil error; want a rejection")
	}
	if _, _, err := m.ForwardBand(x, 0, m.Cfg.NumLayers+1, true); err == nil {
		t.Fatalf("ForwardBand(hi>NumLayers) returned nil error; want a rejection")
	}
}

// tinyForwardBandModel builds a small dense (Llama-style) synthetic model with
// real weights so Forward and ForwardBand run the identical instruction stream.
func tinyForwardBandModel(t *testing.T) *Model {
	t.Helper()
	cfg := Config{
		HiddenSize: 16, NumLayers: 4, NumHeads: 4, NumKVHeads: 4, HeadDim: 4,
		IntermediateSize: 32, VocabSize: 24, RMSNormEps: 1e-5, RopeTheta: 10000,
		ModelType: "llama", EOSTokenID: -1,
	}
	return NewSynthetic(cfg)
}
