// Tests for the pure sampling helper in fakchat: argmax (temp<=0) and the
// temperature-scaled softmax draw (temp>0). Both paths are deterministic given
// their inputs — argmax needs no RNG, and a sharply-peaked distribution forces a
// single index regardless of the seeded RNG — so the expected values are computed
// by hand and the test fails if the selection logic regresses.
package main

import (
	"math"
	"math/rand"
	"testing"
)

func TestSampleArgmax(t *testing.T) {
	tests := []struct {
		name   string
		logits []float32
		temp   float64
		want   int
	}{
		{
			name:   "single element",
			logits: []float32{42},
			temp:   0,
			want:   0,
		},
		{
			name:   "max at end",
			logits: []float32{-1, 0, 3.5, 2},
			temp:   0,
			want:   2,
		},
		{
			name:   "max at start",
			logits: []float32{9, 1, 1, 1},
			temp:   0,
			want:   0,
		},
		{
			name:   "negative logits",
			logits: []float32{-5, -2, -9, -3},
			temp:   -1, // any temp <= 0 takes the argmax path
			want:   1,
		},
		{
			name:   "ties pick first max",
			logits: []float32{1, 5, 5, 2},
			temp:   0,
			want:   1, // strict '>' keeps the earliest maximum
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// rng is unused on the argmax path; pass a real one to match the signature.
			got := sample(tc.logits, tc.temp, rand.New(rand.NewSource(1)))
			if got != tc.want {
				t.Fatalf("sample(%v, %v) = %d, want %d", tc.logits, tc.temp, got, tc.want)
			}
		})
	}
}

// TestSamplePeakedTemperatureIsDeterministic exercises the temp>0 softmax path.
// With one logit dominating by ~1000 nats, the softmax mass on every other index
// underflows to (effectively) zero, so the cumulative draw r := rng.Float64()*sum
// can only land on the dominant index — no matter which seed feeds the RNG.
func TestSamplePeakedTemperatureIsDeterministic(t *testing.T) {
	logits := []float32{-1000, 0, -1000, -1000} // index 1 carries all the mass
	const want = 1
	for seed := int64(0); seed < 8; seed++ {
		rng := rand.New(rand.NewSource(seed))
		if got := sample(logits, 1.0, rng); got != want {
			t.Fatalf("seed %d: sample(...) = %d, want %d", seed, got, want)
		}
	}
}

// TestSampleTemperatureProbabilityRange checks the temp>0 branch never returns an
// out-of-range index and, over many draws of a flat distribution, actually visits
// more than one index (i.e. the RNG draw is wired into the cumulative sum, not a
// hidden argmax). A flat distribution makes each index equally likely.
func TestSampleTemperatureProbabilityRange(t *testing.T) {
	logits := []float32{0, 0, 0, 0}
	rng := rand.New(rand.NewSource(12345))
	seen := map[int]bool{}
	for i := 0; i < 200; i++ {
		got := sample(logits, 1.0, rng)
		if got < 0 || got >= len(logits) {
			t.Fatalf("iteration %d: sample returned out-of-range index %d", i, got)
		}
		seen[got] = true
	}
	if len(seen) < 2 {
		t.Fatalf("expected the flat-distribution draw to visit multiple indices, saw only %v", seen)
	}
}

// guard against accidental reliance on math being imported only transitively.
var _ = math.MaxFloat32
