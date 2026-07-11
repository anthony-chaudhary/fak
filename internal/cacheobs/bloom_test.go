package cacheobs

import (
	"math"
	"sync"
	"testing"
)

// testKey derives the i-th distinct deterministic test key: a counter fed through
// splitmix64 — a DIFFERENT mixer than the filter's own mix64 (fmix64), so the test
// stream cannot accidentally cohere with the probe derivation. No time-based
// randomness anywhere: the same i always yields the same key, so every run measures
// the same stream.
func testKey(i uint64) uint64 {
	x := i + 0x9e3779b97f4a7c15
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}

// #3396 (a): NO false negatives — every inserted key must report probablySeen on
// re-Observe, both immediately and after the whole load has been inserted (bits are
// only ever set, never cleared).
func TestBloomNoFalseNegatives(t *testing.T) {
	const n = 50_000
	b := NewBloomEstimator(n, 0.01)
	for i := uint64(0); i < n; i++ {
		b.Observe(testKey(i))
	}
	for i := uint64(0); i < n; i++ {
		if !b.Observe(testKey(i)) {
			t.Fatalf("false negative: key %d (0x%x) was inserted but re-Observe reported unseen", i, testKey(i))
		}
	}
}

// #3396 (b): the measured false-positive rate over a large stream of DISTINCT
// deterministic keys stays at or below ~2x the configured target. Every key is
// distinct, so every probablySeen=true is by definition a false positive; the
// stream length equals expectedItems, the load the target rate is sized for.
func TestBloomFalsePositiveRateBounded(t *testing.T) {
	const (
		n         = 100_000
		targetFPR = 0.01
	)
	b := NewBloomEstimator(n, targetFPR)
	var falsePositives int
	for i := uint64(0); i < n; i++ {
		if b.Observe(testKey(i)) {
			falsePositives++
		}
	}
	measured := float64(falsePositives) / float64(n)
	t.Logf("measured FPR %v (%d/%d) vs target %v", measured, falsePositives, n, targetFPR)
	if measured > 2*targetFPR {
		t.Fatalf("measured FPR %v (%d/%d) exceeds 2x target %v", measured, falsePositives, n, targetFPR)
	}
	// The bound must not pass vacuously: at this load a correct filter of this size
	// produces SOME false positives (deterministic stream — this is not flaky).
	if falsePositives == 0 {
		t.Fatalf("0 false positives over %d distinct keys — the measurement has no teeth (probe loop broken?)", n)
	}
	// ReusePotential over an all-distinct stream IS the measured FPR — the estimator's
	// over-report is bounded by the same target.
	if got := b.ReusePotential(); got != measured {
		t.Fatalf("ReusePotential = %v, want %v (probable repeats / total)", got, measured)
	}
}

// #3396 (c): ReusePotential on a crafted stream with a known repeat count returns
// the expected ratio. The filter is sized far beyond the stream (1000 items at
// 0.001) so the false-positive contribution is negligible and — with fixed keys
// and a fixed hash — deterministic: 60 distinct + 40 repeats -> exactly 0.40.
func TestBloomReusePotentialKnownRatio(t *testing.T) {
	b := NewBloomEstimator(1000, 0.001)
	const distinct, repeats = 60, 40
	for i := uint64(0); i < distinct; i++ {
		if b.Observe(testKey(i)) {
			t.Fatalf("fresh key %d reported seen in a near-empty oversized filter", i)
		}
	}
	for i := uint64(0); i < repeats; i++ {
		if !b.Observe(testKey(i % distinct)) { // re-observe already-seen keys
			t.Fatalf("repeat of key %d not recognized (false negative)", i%distinct)
		}
	}
	want := float64(repeats) / float64(distinct+repeats)
	if got := b.ReusePotential(); got != want {
		t.Fatalf("ReusePotential = %v, want %v (%d repeats / %d observations)", got, want, repeats, distinct+repeats)
	}
}

// The constructor sizes m and k with the standard formulas: n=10000, p=0.01 gives
// m = ceil(10000 * ln(100) / (ln 2)^2) = 95851 bits and k = round((m/n)*ln 2) = 7.
func TestBloomSizingFormulas(t *testing.T) {
	b := NewBloomEstimator(10_000, 0.01)
	if b.m != 95851 {
		t.Fatalf("m = %d, want 95851", b.m)
	}
	if b.k != 7 {
		t.Fatalf("k = %d, want 7", b.k)
	}
	if want := (b.m + 63) / 64; uint64(len(b.bits)) != want {
		t.Fatalf("bit words = %d, want %d", len(b.bits), want)
	}
}

// Degenerate constructor inputs are clamped to a usable filter, never a panic or a
// zero-width/zero-probe one: expectedItems < 1 acts as 1; targetFPR outside
// (0, 1) — including NaN — is clamped into [MinTargetFPR, MaxTargetFPR].
func TestBloomConstructorClamps(t *testing.T) {
	cases := []struct {
		name  string
		items int
		fpr   float64
	}{
		{"zero-items", 0, 0.01},
		{"negative-items", -5, 0.01},
		{"zero-fpr", 10, 0},
		{"negative-fpr", 10, -1},
		{"fpr-of-one", 10, 1},
		{"fpr-above-one", 10, 5},
		{"nan-fpr", 10, math.NaN()},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			b := NewBloomEstimator(c.items, c.fpr)
			if b.m < 1 || b.k < 1 || len(b.bits) < 1 {
				t.Fatalf("degenerate filter: m=%d k=%d words=%d", b.m, b.k, len(b.bits))
			}
			// It must actually work: fresh then repeat.
			b.Observe(testKey(1))
			if !b.Observe(testKey(1)) {
				t.Fatalf("clamped filter lost a recorded key (false negative)")
			}
		})
	}
}

// A nil or zero-value estimator is inert, matching the package's nil-Observer
// contract: no panic, no phantom repeat, zero potential.
func TestBloomNilAndZeroValueSafe(t *testing.T) {
	var nilB *BloomEstimator
	if nilB.Observe(42) {
		t.Fatalf("nil estimator reported a repeat")
	}
	if nilB.ReusePotential() != 0 {
		t.Fatalf("nil estimator potential = %v, want 0", nilB.ReusePotential())
	}
	var zero BloomEstimator
	if zero.Observe(42) {
		t.Fatalf("zero-value estimator reported a repeat")
	}
	if zero.ReusePotential() != 0 {
		t.Fatalf("zero-value estimator potential = %v, want 0", zero.ReusePotential())
	}
}

// Idle estimator: no observations -> ReusePotential 0, matching ReuseRatio's
// no-phantom-ratio contract.
func TestBloomIdlePotentialIsZero(t *testing.T) {
	if got := NewBloomEstimator(100, 0.01).ReusePotential(); got != 0 {
		t.Fatalf("idle potential = %v, want 0", got)
	}
}

// The observation counters saturate at MaxUint64 instead of wrapping, matching the
// Observer accumulators (in-package seeding is the only way to reach the ceiling).
func TestBloomCountersSaturate(t *testing.T) {
	b := NewBloomEstimator(16, 0.01)
	b.Observe(testKey(7)) // record once so the next Observe is a genuine repeat
	b.observations = math.MaxUint64
	b.probableRepeats = math.MaxUint64
	if !b.Observe(testKey(7)) {
		t.Fatalf("repeat not recognized")
	}
	if b.observations != math.MaxUint64 || b.probableRepeats != math.MaxUint64 {
		t.Fatalf("counters wrapped: observations=%d probableRepeats=%d, want both MaxUint64",
			b.observations, b.probableRepeats)
	}
	if got := b.ReusePotential(); got != 1.0 {
		t.Fatalf("potential at saturation = %v, want a sane 1.0", got)
	}
}

// Concurrent Observe is race-free (run under -race by CI): the estimator may be fed
// asynchronously from many goroutines per the off-hot-path contract.
func TestBloomConcurrentObserveIsRaceFree(t *testing.T) {
	b := NewBloomEstimator(10_000, 0.01)
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		g := g
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := uint64(0); i < 500; i++ {
				b.Observe(testKey(uint64(g)*500 + i))
			}
		}()
	}
	wg.Wait()
	b.mu.Lock()
	total := b.observations
	b.mu.Unlock()
	if total != 4000 {
		t.Fatalf("observations = %d, want 4000", total)
	}
}
