package main

// Witness tests for the aiperf-lifecycle-3 direct port: seeded arrival-pattern
// interval generators (Poisson / Gamma / Constant / ConcurrencyBurst) with live
// setRate for rate ramping. The manifest's verify command is
// `go test ./cmd/loadgen -run TestArrivalPatterns -count=1`.

import (
	"math"
	"testing"
	"time"
)

// fak-test:runtime fast est=2s
func TestArrivalPatterns(t *testing.T) {
	t.Parallel()

	t.Run("parse accepts closed set and empty", func(t *testing.T) {
		for _, name := range []string{"", "poisson", "gamma", "constant", "concurrency_burst"} {
			if _, err := parseArrivalPattern(name); err != nil {
				t.Fatalf("parseArrivalPattern(%q) = %v, want nil", name, err)
			}
		}
		for _, bad := range []string{"Poisson", "exponential", "burst", "wave", "poison"} {
			if _, err := parseArrivalPattern(bad); err == nil {
				t.Fatalf("parseArrivalPattern(%q) = nil, want refusal", bad)
			}
		}
	})

	t.Run("constant is exactly periodic and setRate rescales", func(t *testing.T) {
		g, err := newConstantGenerator(4) // period = 250ms
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 1000; i++ {
			if got := g.nextInterval(); math.Abs(got-0.25) > 1e-12 {
				t.Fatalf("constant interval[%d]=%v want 0.25", i, got)
			}
		}
		if err := g.setRate(8); err != nil { // period = 125ms
			t.Fatal(err)
		}
		if got := g.nextInterval(); math.Abs(got-0.125) > 1e-12 {
			t.Fatalf("constant after setRate=%v want 0.125", got)
		}
		if g.rate() != 8 {
			t.Fatalf("rate=%v want 8", g.rate())
		}
	})

	t.Run("burst is zero-delay and setRate is a no-op", func(t *testing.T) {
		g := newBurstGenerator()
		for i := 0; i < 100; i++ {
			if got := g.nextInterval(); got != 0 {
				t.Fatalf("burst interval[%d]=%v want 0", i, got)
			}
		}
		if err := g.setRate(123); err != nil {
			t.Fatalf("burst setRate = %v, want nil no-op", err)
		}
		if g.rate() != 0 {
			t.Fatalf("burst rate=%v want 0 (not applicable)", g.rate())
		}
	})

	// Ari: exponential inter-arrivals must average 1/rate. n=100k keeps the
	// standard error of the mean ~0.3% (exponential) so the 5% band is a
	// correctness gate, not a flake gate; a fixed seed removes run-to-run drift.
	for _, rate := range []float64{1, 10, 250} {
		g, err := newPoissonGenerator(rate, 20261112)
		if err != nil {
			t.Fatal(err)
		}
		mean := sampleMean(g, 100000)
		want := 1.0 / rate
		if rel := math.Abs(mean-want) / want; rel > 0.05 {
			t.Fatalf("poisson@%g mean=%v want %v (rel=%.4f > 5%%)", rate, mean, want, rel)
		}
	}

	// Gamma with smoothness == 1 is Poisson-equivalent, and every smoothness
	// preserves the mean 1/rate (scale = 1/(rate*smoothness)).
	for _, sm := range []float64{0.5, 1.0, 4.0} {
		g, err := newGammaGenerator(20, sm, 20261112)
		if err != nil {
			t.Fatal(err)
		}
		mean := sampleMean(g, 100000)
		want := 1.0 / 20
		if rel := math.Abs(mean-want) / want; rel > 0.05 {
			t.Fatalf("gamma(smoothness=%g) mean=%v want %v (rel=%.4f > 5%%)", sm, mean, want, rel)
		}
		if g.smoothness != sm {
			t.Fatalf("gamma smoothness=%v want %v", g.smoothness, sm)
		}
	}

	// Gamma burstiness is monotone in smoothness: lower smoothness clusters
	// arrivals (higher coefficient of variation = std/mean).
	cvLow := sampleCV(mustGamma(t, 20, 0.5), 200000)
	cvHigh := sampleCV(mustGamma(t, 20, 8.0), 200000)
	if !(cvLow > cvHigh) {
		t.Fatalf("gamma burstiness not monotone: cv(smoothness=0.5)=%v want > cv(smoothness=8)=%v", cvLow, cvHigh)
	}

	// A fixed seed reproduces the exact same stream; different seeds differ.
	t.Run("seeded reproducibility", func(t *testing.T) {
		a := mustPoisson(t, 50, 7)
		b := mustPoisson(t, 50, 7)
		c := mustPoisson(t, 50, 8)
		same := true
		differs := false
		for i := 0; i < 1000; i++ {
			av, bv, cv := a.nextInterval(), b.nextInterval(), c.nextInterval()
			if av != bv {
				same = false
			}
			if av != cv {
				differs = true
			}
		}
		if !same {
			t.Fatal("same seed produced a different poisson stream")
		}
		if !differs {
			t.Fatal("different seed produced an identical poisson stream")
		}
	})

	// setRate ramping takes effect on the next draw and rescaled means follow.
	t.Run("poisson setRate ramps the mean", func(t *testing.T) {
		g := mustPoisson(t, 100, 99)
		before := sampleMean(g, 20000)
		if rel := math.Abs(before-0.01) / 0.01; rel > 0.06 {
			t.Fatalf("pre-ramp mean=%v want ~0.01", before)
		}
		if err := g.setRate(1); err != nil {
			t.Fatal(err)
		}
		after := sampleMean(g, 20000)
		if rel := math.Abs(after-1.0) / 1.0; rel > 0.06 {
			t.Fatalf("post-ramp mean=%v want ~1.0", after)
		}
	})

	// Non-positive and non-finite rates are refused by construction.
	t.Run("non-positive rate refused", func(t *testing.T) {
		for _, bad := range []float64{0, -1, math.NaN(), math.Inf(1)} {
			if _, err := newPoissonGenerator(bad, 1); err == nil {
				t.Fatalf("newPoissonGenerator(%v) = nil, want refusal", bad)
			}
			if _, err := newConstantGenerator(bad); err == nil {
				t.Fatalf("newConstantGenerator(%v) = nil, want refusal", bad)
			}
			if _, err := newGammaGenerator(bad, 1, 1); err == nil {
				t.Fatalf("newGammaGenerator(rate=%v) = nil, want refusal", bad)
			}
		}
		if _, err := newGammaGenerator(1, 0, 1); err == nil {
			t.Fatal("newGammaGenerator(smoothness=0) = nil, want refusal")
		}
	})

	// The empty pattern is not a generator; only the caller's sentinel handles it.
	t.Run("empty pattern has no generator", func(t *testing.T) {
		if _, err := newIntervalGenerator("", 1, 1, 0); err == nil {
			t.Fatal("newIntervalGenerator(\"\") = nil, want refusal")
		}
	})
}

// TestArrivalPatternsDrivesPacer pins the wiring: an arrival generator supplies
// the per-step gap to the pacer, so a constant pattern reproduces the
// fixed-rate schedule exactly while a burst pattern emits every step with no
// sleep.
// fak-test:runtime fast est=100ms
func TestArrivalPatternsDrivesPacer(t *testing.T) {
	t.Parallel()
	start := time.Date(2026, 11, 12, 9, 0, 0, 0, time.UTC)

	t.Run("constant gap matches the fixed-interval pacer", func(t *testing.T) {
		const interval = 50 * time.Millisecond
		sim := newAdvSim(start)
		sim.steps = []advStep{
			{wake: start.Add(50 * time.Millisecond), ok: true},
			{wake: start.Add(100 * time.Millisecond), ok: true},
			{wake: start.Add(150 * time.Millisecond), ok: true},
		}
		gen, err := newConstantGenerator(20) // 1/20 s = 50ms
		if err != nil {
			t.Fatal(err)
		}
		p := newPacerFromGap(delaySource(gen), 10*time.Millisecond, sim.nowFn, sim.sleepFn)
		if got := p.run(t.Context(), 3, sim.issueFn); got != 3 {
			t.Fatalf("issued=%d want 3", got)
		}
		want := []time.Time{
			start.Add(50 * time.Millisecond),
			start.Add(100 * time.Millisecond),
			start.Add(150 * time.Millisecond),
		}
		advAssertTimes(t, "issuedAt", sim.issuedAt, want)
		_ = interval
	})

	t.Run("burst gap never sleeps", func(t *testing.T) {
		sim := newAdvSim(start)
		p := newPacerFromGap(delaySource(newBurstGenerator()), 10*time.Millisecond, sim.nowFn, sim.sleepFn)
		if got := p.run(t.Context(), 5, sim.issueFn); got != 5 {
			t.Fatalf("issued=%d want 5", got)
		}
		if len(sim.slept) != 0 {
			t.Fatalf("slept=%v want none for zero-delay burst", sim.slept)
		}
	})
}

// TestArrivalValidateConfig pins the fail-closed config contract: a rate-bearing
// pattern with no usable rate refuses before any network call, an unknown name
// refuses, and an empty pattern leaves the legacy config valid.
// fak-test:runtime fast est=50ms
func TestArrivalValidateConfig(t *testing.T) {
	t.Parallel()
	base := func() *Config {
		return &Config{Concurrencies: []int{1}, RequestsPer: 1}
	}

	if err := validateConfig(base()); err != nil {
		t.Fatalf("empty arrival: %v, want nil", err)
	}

	cfg := base()
	cfg.Arrival = arrivalPoisson // no rate anywhere
	if err := validateConfig(cfg); err == nil {
		t.Fatal("poisson with no rate: want refusal")
	}

	cfg = base()
	cfg.Arrival = arrivalPoisson
	cfg.RatePerSec = 3 // legacy rate is an accepted fallback
	if err := validateConfig(cfg); err != nil {
		t.Fatalf("poisson with legacy RatePerSec: %v, want nil", err)
	}

	cfg = base()
	cfg.Arrival = arrivalBurst // rate not applicable
	if err := validateConfig(cfg); err != nil {
		t.Fatalf("burst with no rate: %v, want nil", err)
	}

	cfg = base()
	cfg.Arrival = "wave" // not in the closed set
	if err := validateConfig(cfg); err == nil {
		t.Fatal("unknown pattern: want refusal")
	}

	cfg = base()
	cfg.Arrival = arrivalGamma
	cfg.ArrivalRate = 5
	cfg.ArrivalSmoothness = -1
	if err := validateConfig(cfg); err == nil {
		t.Fatal("negative smoothness: want refusal")
	}
}

// TestArrivalEchoRoundTrips pins that a shaped sweep records its shape in the
// top-level matrix JSON, so the receipt is self-describing.
// fak-test:runtime fast est=50ms
func TestArrivalEchoRoundTrips(t *testing.T) {
	t.Parallel()
	cfg := &Config{
		URL: "http://127.0.0.1:1/v1/chat/completions", Model: "m",
		Concurrencies: []int{1}, RequestsPer: 1,
		Arrival: arrivalGamma, ArrivalRate: 12.5, ArrivalSmoothness: 2.5, ArrivalSeed: 42,
	}
	mr := Run(t.Context(), cfg)
	if mr.Arrival == nil {
		t.Fatal("Arrival echo nil, want set for a shaped sweep")
	}
	if mr.Arrival.Pattern != string(arrivalGamma) || mr.Arrival.RatePerSec != 12.5 ||
		mr.Arrival.Smoothness != 2.5 || mr.Arrival.Seed != 42 {
		t.Fatalf("echo=%+v want gamma/12.5/2.5/42", *mr.Arrival)
	}

	// No pattern -> no echo (the block is omitted from JSON).
	plain := Run(t.Context(), &Config{
		URL: "http://127.0.0.1:1/v1/chat/completions", Model: "m",
		Concurrencies: []int{1}, RequestsPer: 1,
	})
	if plain.Arrival != nil {
		t.Fatalf("plain sweep echo=%+v want nil", *plain.Arrival)
	}
}

func sampleMean(g intervalGenerator, n int) float64 {
	var sum float64
	for i := 0; i < n; i++ {
		sum += g.nextInterval()
	}
	return sum / float64(n)
}

func sampleCV(g intervalGenerator, n int) float64 {
	xs := make([]float64, n)
	var sum float64
	for i := range xs {
		xs[i] = g.nextInterval()
		sum += xs[i]
	}
	mean := sum / float64(n)
	var varSum float64
	for _, x := range xs {
		d := x - mean
		varSum += d * d
	}
	return math.Sqrt(varSum/float64(n)) / mean
}

func mustPoisson(t *testing.T, rate float64, seed int64) *poissonGenerator {
	t.Helper()
	g, err := newPoissonGenerator(rate, seed)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func mustGamma(t *testing.T, rate, smoothness float64) *gammaGenerator {
	t.Helper()
	g, err := newGammaGenerator(rate, smoothness, 20261112)
	if err != nil {
		t.Fatal(err)
	}
	return g
}
