package model

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestV41HostArmCompressorEmission(t *testing.T) {
	for _, ratio := range []int{1, 4} {
		pool, reference := mustPool(t, ratio, 2), mustPool(t, ratio, 2)
		var stages []string
		observe := func(stage string) { stages = append(stages, stage) }
		for pos := 0; pos < ratio*2; pos++ {
			kv, score, gain := []float32{float32(pos) + 0.125, -2.75}, []float32{0.2, -0.3}, []float32{1, 0.75}
			got, emitted, err := pool.pushNormalizedObserved(pos, kv, score, gain, 1e-5, observe)
			want, wantEmitted, wantErr := reference.PushNormalized(pos, kv, score, gain, 1e-5)
			if err != nil || wantErr != nil || emitted != wantEmitted || !reflect.DeepEqual(got, want) {
				t.Fatalf("ratio=%d pos=%d emission/parity failure: %v %v", ratio, pos, err, wantErr)
			}
			groups := (pos + 1) / ratio
			var expected []string
			for i := 0; i < groups; i++ {
				if ratio > 1 {
					expected = append(expected, "compressor-gated-pool")
				}
				expected = append(expected, "compressor-norm", "bf16-boundary")
			}
			if !reflect.DeepEqual(stages, expected) {
				t.Fatalf("ratio=%d pos=%d stages=%v want=%v", ratio, pos, stages, expected)
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41HostArmCompressorSelectedCallbacks(t *testing.T) {
	fault := errors.New("selected compressor failure")
	for _, mode := range []string{"selected", "pool-error", "norm-error", "invalid"} {
		t.Run(mode, func(t *testing.T) {
			pool := mustPool(t, 2, 2)
			var stages []string
			observe := func(stage string) { stages = append(stages, stage) }
			pool.gatedPool = func(_, _ []float32) ([]float32, error) {
				if mode == "pool-error" {
					return nil, fault
				}
				return []float32{2, 3}, nil
			}
			pool.normalize = func(row, _ []float32, _ float32) ([]float32, error) {
				if mode == "norm-error" {
					return nil, fault
				}
				return row, nil
			}
			if mode == "norm-error" {
				pool.gatedPool = nil
			}
			for pos := 0; pos < 2; pos++ {
				gain := []float32{1, 1}
				if mode == "invalid" {
					gain[0] = float32(math.NaN())
				}
				_, emitted, err := pool.pushNormalizedObserved(pos, []float32{1, 2}, []float32{0, 0}, gain, 1e-5, observe)
				if mode == "invalid" {
					if err == nil || emitted {
						t.Fatal("invalid gain accepted")
					}
					break
				}
				if pos == 1 && mode != "selected" {
					if !errors.Is(err, fault) {
						t.Fatalf("lost selected failure: %v", err)
					}
				} else if err != nil {
					t.Fatal(err)
				}
			}
			var want []string
			if mode == "norm-error" {
				want = []string{"compressor-gated-pool"}
			}
			if !reflect.DeepEqual(stages, want) {
				t.Fatalf("stages=%v want=%v", stages, want)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41HostArmCompressorReplayObserverLifetime(t *testing.T) {
	f := v41LCFixtureValues()
	state, err := NewV41AttentionState(f.Width, f.Ratio)
	if err != nil {
		t.Fatal(err)
	}
	reference, err := NewV41AttentionState(f.Width, f.Ratio)
	if err != nil {
		t.Fatal(err)
	}
	projectKV := v41LCProjector(f.WKV, f.Width, f.Hidden)
	projectScore := v41LCProjector(f.WGate, f.Width, f.Hidden)
	var abandoned, completing []string
	for pos, input := range f.Inputs {
		var observe func(string)
		if pos < f.Ratio-1 {
			observe = func(stage string) { abandoned = append(abandoned, stage) }
		}
		if pos == f.Ratio-1 {
			observe = func(stage string) { completing = append(completing, stage) }
		}
		got, _, emitted, start, end, err := state.appendCompressorSourcePublicationObserved(0, f.Ratio, pos, input, mustPool(t, f.Ratio, f.Width), projectKV, projectScore, f.NormWeight, float32(f.Eps), v41CompressorPublication{}, observe)
		want, _, wantEmitted, wantStart, wantEnd, wantErr := reference.appendCompressorSourcePublication(0, f.Ratio, pos, input, mustPool(t, f.Ratio, f.Width), projectKV, projectScore, f.NormWeight, float32(f.Eps), v41CompressorPublication{})
		if err != nil || wantErr != nil || emitted != wantEmitted || start != wantStart || end != wantEnd || !reflect.DeepEqual(got, want) {
			t.Fatalf("pos=%d replay diverged: %v %v", pos, err, wantErr)
		}
	}
	if len(abandoned) != 0 || !reflect.DeepEqual(completing, []string{"compressor-gated-pool", "compressor-norm", "bf16-boundary"}) {
		t.Fatalf("retained observer: abandoned=%v completing=%v", abandoned, completing)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41HostArmCompressorIndexer(t *testing.T) {
	var stages []string
	observe := func(stage string) { stages = append(stages, stage) }
	pub, err := newV41IndexerPublicationObserved(0, []float32{2, -1}, []float32{1, 0, 0, 1, 2, 1}, []float32{1}, 1, 2, 3, 1, 2, 1, 0, observe)
	if err != nil {
		t.Fatal(err)
	}
	want, err := NewV41IndexerPublication(0, []float32{2, -1}, []float32{1, 0, 0, 1, 2, 1}, []float32{1}, 1, 2, 3, 1, 2, 1, 0)
	if err != nil || !reflect.DeepEqual(pub, want) {
		t.Fatalf("publication parity: %v", err)
	}
	if !reflect.DeepEqual(stages, []string{"attention-indexer-score", "attention-index-list", "attention-index-list"}) {
		t.Fatalf("stages=%v", stages)
	}
	stages = nil
	if _, err := v41IndexerScoreObserved([]float32{1}, nil, []float32{1}, 1, 1, 0, observe); err != nil {
		t.Fatal(err)
	}
	if _, err := v41SelectCandidateBlocksObserved(nil, 0, 1, 2, observe); err != nil {
		t.Fatal(err)
	}
	if _, err := v41SelectIndexRowsObserved(nil, 0, 1, 0, nil, observe); err != nil {
		t.Fatal(err)
	}
	if _, err := v41SelectIndexRowsObserved([]float32{float32(math.NaN())}, 1, 1, 0, nil, observe); err == nil {
		t.Fatal("NaN accepted")
	}
	if len(stages) != 0 {
		t.Fatalf("empty/invalid rows counted: %v", stages)
	}
	if _, err := v41SelectIndexRowsObserved([]float32{2, 1}, 2, 0, 0, nil, observe); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(stages, []string{"attention-index-list"}) {
		t.Fatalf("nonempty zero-top-k sorting stages=%v", stages)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41HostArmCompressorLaterStep(t *testing.T) {
	m, ring, _, host, _ := v41RingSessions(t)
	prefix, next := v41IncrementalPrefix(m.Cfg, 2)
	v41RingBitsEqualRows(t, "prefill", host.Prefill(prefix), ring.Prefill(prefix))
	var stages []string
	host.hostStageObserver = func(stage string) {
		if stage == "compressor-gated-pool" || stage == "compressor-norm" {
			stages = append(stages, stage)
		}
	}
	v41RingBitsEqualRows(t, "incomplete step", host.Step(next), ring.Step(next))
	if len(stages) != 0 {
		t.Fatalf("incomplete Step observed %v", stages)
	}
	next = (next*7 + 3) % m.Cfg.VocabSize
	v41RingBitsEqualRows(t, "closing step", host.Step(next), ring.Step(next))
	var want []string
	for l := 0; l < m.Cfg.NumLayers; l++ {
		want = append(want, "compressor-gated-pool", "compressor-norm")
	}
	if !reflect.DeepEqual(stages, want) {
		t.Fatalf("closing Step stages=%v want=%v", stages, want)
	}
}
