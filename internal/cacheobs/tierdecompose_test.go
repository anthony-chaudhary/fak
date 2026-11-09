package cacheobs

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

// TestTierDecomposeBasicSplit pins the hand-computed core contract on a window where every
// quantity is exactly representable: L3 = Δprefetch, L2 = max(0, Δloadback − Δprefetch),
// L1 = the residual, hit rate and miss from the prompt delta, status derived.
func TestTierDecomposeBasicSplit(t *testing.T) {
	before := CounterSnapshot{
		CachedTokens:     1000,
		PromptTokens:     2000,
		EvictedTokens:    5,
		LoadBackTokens:   50,
		BackupedTokens:   10,
		PrefetchedTokens: 20,
	}
	after := CounterSnapshot{
		CachedTokens:     1512, // +512
		PromptTokens:     3024, // +1024
		EvictedTokens:    8,    // +3
		LoadBackTokens:   80,   // +30
		BackupedTokens:   15,   // +5
		PrefetchedTokens: 70,   // +50
	}
	d := DecomposeTiers(before, after)
	if d.Status != LayerStatusDerived {
		t.Fatalf("moving window with tier shares must read derived, got %s", d.Status)
	}
	if d.L3 != 50 {
		t.Errorf("L3 must be Δprefetch = 50, got %v", d.L3)
	}
	if d.L2 != 0 {
		t.Errorf("L2 must be max(0, 30−50) = 0, got %v", d.L2)
	}
	if d.L1 != 462 {
		t.Errorf("L1 must be the residual 512−0−50 = 462, got %v", d.L1)
	}
	if d.HitRate != 0.5 {
		t.Errorf("hit rate must be 512/1024 = 0.5, got %v", d.HitRate)
	}
	if d.Miss != 512 {
		t.Errorf("miss must be 1024×(1−0.5) = 512, got %v", d.Miss)
	}
	if d.CachedDelta != 512 || d.PromptDelta != 1024 || d.LoadBackDelta != 30 || d.PrefetchDelta != 50 {
		t.Errorf("window deltas wrong: %+v", d)
	}
	if len(d.ResetCounters) != 0 {
		t.Errorf("monotone counters must name no resets, got %v", d.ResetCounters)
	}
}

// TestTierDecomposeThroughPathSubtraction pins the load-bearing counter fact: an L3 hit
// transits the host pool, so it increments BOTH the prefetch and the load-back counters.
// The through-path subtraction gives the host tier only the volume beyond the transit.
func TestTierDecomposeThroughPathSubtraction(t *testing.T) {
	// Every load-back transited storage: the host tier earns nothing.
	d := DecomposeTiers(
		CounterSnapshot{},
		CounterSnapshot{CachedTokens: 40, LoadBackTokens: 40, PrefetchedTokens: 40},
	)
	if d.L3 != 40 || d.L2 != 0 || d.L1 != 0 {
		t.Fatalf("full transit: want L3=40 L2=0 L1=0, got L3=%v L2=%v L1=%v", d.L3, d.L2, d.L1)
	}

	// Partial transit: only the load-back volume beyond the prefetch is host-tier.
	d = DecomposeTiers(
		CounterSnapshot{},
		CounterSnapshot{CachedTokens: 40, LoadBackTokens: 40, PrefetchedTokens: 25},
	)
	if d.L3 != 25 || d.L2 != 15 || d.L1 != 0 {
		t.Fatalf("partial transit: want L3=25 L2=15 L1=0, got L3=%v L2=%v L1=%v", d.L3, d.L2, d.L1)
	}

	// Load-back with no prefetch at all: the whole transfer volume is host-tier.
	d = DecomposeTiers(
		CounterSnapshot{},
		CounterSnapshot{CachedTokens: 30, LoadBackTokens: 30},
	)
	if d.L3 != 0 || d.L2 != 30 || d.L1 != 0 {
		t.Fatalf("no prefetch: want L3=0 L2=30 L1=0, got L3=%v L2=%v L1=%v", d.L3, d.L2, d.L1)
	}
}

// TestTierDecomposeOvershootRescale pins the rescale: when transfers exceed the cached
// volume (speculative/re-fetched blocks), L2 and L3 are scaled proportionally onto the
// cached total, the invariant holds EXACTLY, and the L2:L3 ratio is preserved.
func TestTierDecomposeOvershootRescale(t *testing.T) {
	// Δcached=100, Δloadback=150, Δprefetch=60 → raw L3=60, L2=90 (sum 150 > 100).
	// Proportional rescale: L3' = round(60×100/150) = 40, L2' = 100−40 = 60, L1 = 0.
	d := DecomposeTiers(
		CounterSnapshot{},
		CounterSnapshot{CachedTokens: 100, LoadBackTokens: 150, PrefetchedTokens: 60},
	)
	if d.Status != LayerStatusDerived {
		t.Fatalf("rescaled row with non-zero tiers must read derived, got %s", d.Status)
	}
	if d.L3 != 40 || d.L2 != 60 || d.L1 != 0 {
		t.Fatalf("want L1=0 L2=60 L3=40, got L1=%v L2=%v L3=%v", d.L1, d.L2, d.L3)
	}
	if sum := d.L1 + d.L2 + d.L3; sum != 100 {
		t.Fatalf("invariant must hold exactly after rescale, got L1+L2+L3 = %v", sum)
	}
	// Ratio preserved: raw 90:60 == rescaled 60:40.
	if math.Abs(d.L2/d.L3-1.5) > 1e-12 {
		t.Fatalf("rescale must preserve the L2:L3 ratio (90:60), got %v:%v", d.L2, d.L3)
	}
}

// TestTierDecomposeInvariantProperty asserts the rescale invariant over randomized delta
// inputs: L1, L2, L3 are non-negative, L3 never exceeds Δprefetch, L2 never exceeds
// Δloadback, and L1+L2+L3 equals Δcached BIT-EXACTLY — in the free-form regime and in the
// forced-overshoot regime where transfers exceed the cached volume.
func TestTierDecomposeInvariantProperty(t *testing.T) {
	rng := rand.New(rand.NewSource(1)) // deterministic seed: no wall clock, no entropy
	const maxDelta = 1 << 40           // far below 2^53: the documented exactness domain

	check := func(label string, dc, dp, dlb, dpf uint64) {
		base := uint64(1) << 30
		before := CounterSnapshot{
			CachedTokens: base, PromptTokens: base, EvictedTokens: base,
			LoadBackTokens: base, BackupedTokens: base, PrefetchedTokens: base,
		}
		after := CounterSnapshot{
			CachedTokens:     before.CachedTokens + dc,
			PromptTokens:     before.PromptTokens + dp,
			EvictedTokens:    before.EvictedTokens + rng.Uint64()%maxDelta,
			LoadBackTokens:   before.LoadBackTokens + dlb,
			BackupedTokens:   before.BackupedTokens + rng.Uint64()%maxDelta,
			PrefetchedTokens: before.PrefetchedTokens + dpf,
		}
		d := DecomposeTiers(before, after)
		if d.Status == LayerStatusCrashAffected || d.Status == LayerStatusServer {
			t.Fatalf("%s: monotone counters must never read crash_affected or server, got %s", label, d.Status)
		}
		if d.L1 < 0 || d.L2 < 0 || d.L3 < 0 {
			t.Fatalf("%s: tier shares must be non-negative, got L1=%v L2=%v L3=%v", label, d.L1, d.L2, d.L3)
		}
		if d.L3 > float64(dpf) {
			t.Fatalf("%s: L3=%v exceeds Δprefetch=%d", label, d.L3, dpf)
		}
		if d.L2 > float64(dlb) {
			t.Fatalf("%s: L2=%v exceeds Δloadback=%d", label, d.L2, dlb)
		}
		if sum := d.L1 + d.L2 + d.L3; sum != float64(dc) {
			t.Fatalf("%s: invariant violated: L1+L2+L3 = %v, want Δcached = %d (bit-exact)", label, sum, dc)
		}
		allZero := d.L1 == 0 && d.L2 == 0 && d.L3 == 0
		if allZero && d.Status != LayerStatusDerivedNoSignal && d.Status != LayerStatusNoMovement {
			t.Fatalf("%s: all-zero tiers must read derived_no_signal or no_movement, got %s", label, d.Status)
		}
		if !allZero && d.Status != LayerStatusDerived {
			t.Fatalf("%s: non-zero tiers must read derived, got %s", label, d.Status)
		}
	}

	// Regime 1: free-form deltas across the exactness domain.
	for i := 0; i < 500; i++ {
		check("free",
			rng.Uint64()%maxDelta, rng.Uint64()%maxDelta,
			rng.Uint64()%maxDelta, rng.Uint64()%maxDelta)
	}
	// Regime 2: forced overshoot — load-back drawn AT or ABOVE the cached volume.
	for i := 0; i < 500; i++ {
		dc := rng.Uint64() % maxDelta
		dlb := dc + rng.Uint64()%(maxDelta/2)
		check("overshoot", dc, rng.Uint64()%maxDelta, dlb, rng.Uint64()%maxDelta)
	}
}

// TestTierDecomposeNoMovement pins the no-movement row: the status says "nothing happened",
// the tier fields are MEASURED zeros (the counters were present and flat), and the ratios a
// 0/0 window cannot define stay NaN — never a silent zero row.
func TestTierDecomposeNoMovement(t *testing.T) {
	snap := CounterSnapshot{
		CachedTokens: 4321, PromptTokens: 98765, EvictedTokens: 7,
		LoadBackTokens: 3, BackupedTokens: 11, PrefetchedTokens: 2,
	}
	d := DecomposeTiers(snap, snap)
	if d.Status != LayerStatusNoMovement {
		t.Fatalf("flat counters must read no_movement, got %s", d.Status)
	}
	if d.L1 != 0 || d.L2 != 0 || d.L3 != 0 {
		t.Fatalf("flat counters are a measured zero, got L1=%v L2=%v L3=%v", d.L1, d.L2, d.L3)
	}
	if !math.IsNaN(d.HitRate) || !math.IsNaN(d.Miss) {
		t.Fatalf("0/0 hit rate and zero-prompt miss must be NaN (not measured), got hr=%v miss=%v", d.HitRate, d.Miss)
	}
	if len(d.ResetCounters) != 0 {
		t.Fatalf("flat counters are not resets, got %v", d.ResetCounters)
	}
}

// TestTierDecomposeNotMeasuredSentinels pins the NaN-not-zero discipline: fields whose
// defining inputs are undefined carry NaN, while fields that ARE measured may lawfully be
// zero — a measured zero and an absent measurement must stay distinguishable.
func TestTierDecomposeNotMeasuredSentinels(t *testing.T) {
	// Cache served tokens but the prompt counter did not move: the hit rate is undefined
	// (division by zero prompt) and the miss is unmeasured; the tier split is fully
	// measured.
	d := DecomposeTiers(CounterSnapshot{}, CounterSnapshot{CachedTokens: 500})
	if !math.IsNaN(d.HitRate) || !math.IsNaN(d.Miss) {
		t.Fatalf("zero prompt delta must leave hit rate and miss NaN, got hr=%v miss=%v", d.HitRate, d.Miss)
	}
	if d.L1 != 500 || d.L2 != 0 || d.L3 != 0 {
		t.Fatalf("the tier split is still a measurement, got L1=%v L2=%v L3=%v", d.L1, d.L2, d.L3)
	}
	if d.Status != LayerStatusDerived {
		t.Fatalf("a measured tier share must read derived, got %s", d.Status)
	}

	// Pure-miss window: prompt moved, nothing cached, no transfers. Hit rate is a MEASURED
	// zero (0.0 is a measurement), the miss is the full prompt, and no tier earned a share
	// — derived_no_signal, not a zero row.
	d = DecomposeTiers(CounterSnapshot{}, CounterSnapshot{PromptTokens: 777})
	if d.HitRate != 0 {
		t.Fatalf("an all-miss window must measure hit rate 0, got %v", d.HitRate)
	}
	if d.Miss != 777 {
		t.Fatalf("an all-miss window must miss the full prompt, got %v", d.Miss)
	}
	if d.L1 != 0 || d.L2 != 0 || d.L3 != 0 {
		t.Fatalf("no tier earned a share, got L1=%v L2=%v L3=%v", d.L1, d.L2, d.L3)
	}
	if d.Status != LayerStatusDerivedNoSignal {
		t.Fatalf("moving input with zero tier output must read derived_no_signal, got %s", d.Status)
	}

	// Speculative-only window: prefetch moved, nothing cached. The rescale drives both
	// transfer tiers to zero; the row is derived_no_signal with unmeasured ratios.
	d = DecomposeTiers(CounterSnapshot{}, CounterSnapshot{PrefetchedTokens: 100})
	if d.Status != LayerStatusDerivedNoSignal {
		t.Fatalf("a speculative-only window must read derived_no_signal, got %s", d.Status)
	}
	if d.L1 != 0 || d.L2 != 0 || d.L3 != 0 {
		t.Fatalf("speculative prefetch that cached nothing must attribute nothing, got L1=%v L2=%v L3=%v", d.L1, d.L2, d.L3)
	}
	if !math.IsNaN(d.HitRate) || !math.IsNaN(d.Miss) {
		t.Fatalf("no prompt movement leaves the ratios unmeasured, got hr=%v miss=%v", d.HitRate, d.Miss)
	}
}

// TestTierDecomposeCounterResetDetection pins reset detection: a DECREASING cumulative
// counter flips the row to crash_affected and names the offender, instead of clamping the
// negative delta to a silent zero.
func TestTierDecomposeCounterResetDetection(t *testing.T) {
	// One counter dropped: the load-back counter reset mid-window.
	d := DecomposeTiers(
		CounterSnapshot{CachedTokens: 100, PromptTokens: 100, LoadBackTokens: 50},
		CounterSnapshot{CachedTokens: 120, PromptTokens: 130, LoadBackTokens: 10},
	)
	if d.Status != LayerStatusCrashAffected {
		t.Fatalf("a decreasing counter must read crash_affected, got %s", d.Status)
	}
	if len(d.ResetCounters) != 1 || d.ResetCounters[0] != "load_back_tokens_total" {
		t.Fatalf("the reset offender must be named canonically, got %v", d.ResetCounters)
	}
	if d.LoadBackDelta != -40 {
		t.Fatalf("the negative delta is the reset evidence and stays reported, got %v", d.LoadBackDelta)
	}
	for name, v := range map[string]float64{"L1": d.L1, "L2": d.L2, "L3": d.L3, "miss": d.Miss, "hit_rate": d.HitRate} {
		if !math.IsNaN(v) {
			t.Fatalf("crash-affected %s must be NaN (untrustworthy), got %v", name, v)
		}
	}

	// Several counters dropped at once: every offender is named, in schema order.
	d = DecomposeTiers(
		CounterSnapshot{CachedTokens: 100, EvictedTokens: 10, BackupedTokens: 10},
		CounterSnapshot{CachedTokens: 50, EvictedTokens: 5, BackupedTokens: 2},
	)
	if d.Status != LayerStatusCrashAffected {
		t.Fatalf("multiple resets must read crash_affected, got %s", d.Status)
	}
	want := []string{"cached_tokens_total", "evicted_tokens_total", "backuped_tokens_total"}
	if !reflect.DeepEqual(d.ResetCounters, want) {
		t.Fatalf("reset counters must list every offender in schema order, got %v want %v", d.ResetCounters, want)
	}

	// Reset outranks movement: real forward motion elsewhere does not launder a reset.
	d = DecomposeTiers(
		CounterSnapshot{PromptTokens: 1000, PrefetchedTokens: 100},
		CounterSnapshot{PromptTokens: 2000, PrefetchedTokens: 50},
	)
	if d.Status != LayerStatusCrashAffected {
		t.Fatalf("a reset among moving counters must still read crash_affected, got %s", d.Status)
	}
}

// TestTierDecomposeStatusVocabularyClosed pins the closed lineage vocabulary: every status
// has its label spelling, the enumerator returns exactly the five declared statuses, and
// out-of-range values render unknown rather than panicking or inventing a sixth label.
func TestTierDecomposeStatusVocabularyClosed(t *testing.T) {
	want := map[LayerStatus]string{
		LayerStatusServer:          "server",
		LayerStatusDerived:         "derived",
		LayerStatusDerivedNoSignal: "derived_no_signal",
		LayerStatusNoMovement:      "no_movement",
		LayerStatusCrashAffected:   "crash_affected",
	}
	for s, label := range want {
		if s.String() != label {
			t.Fatalf("status %d must render %q, got %q", int(s), label, s.String())
		}
		if !s.valid() {
			t.Fatalf("declared status %d must be valid", int(s))
		}
	}
	if all := AllLayerStatuses(); len(all) != len(want) {
		t.Fatalf("the vocabulary must contain exactly the %d declared statuses, got %v", len(want), all)
	}
	for _, s := range AllLayerStatuses() {
		if _, ok := want[s]; !ok {
			t.Fatalf("enumerator returned undeclared status %v", s)
		}
	}
	for _, bad := range []LayerStatus{-1, numLayerStatuses, numLayerStatuses + 7} {
		if bad.String() != "unknown" {
			t.Fatalf("out-of-range status %d must render unknown, got %q", int(bad), bad.String())
		}
		if bad.valid() {
			t.Fatalf("out-of-range status %d must be invalid", int(bad))
		}
	}
}

// TestTierDecomposeMissFollowsHitRate pins the miss definition over randomized windows:
// whenever the hit rate is measured, it lies in [0,1] and the miss agrees with the integer
// complement Δprompt − Δcached within rounding — the formula and the arithmetic must not
// drift apart.
func TestTierDecomposeMissFollowsHitRate(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const maxDelta = 1 << 40
	for i := 0; i < 200; i++ {
		p := rng.Uint64() % maxDelta
		c := rng.Uint64() % (p + 1) // cached never exceeds prompt in sane windows
		d := DecomposeTiers(CounterSnapshot{}, CounterSnapshot{PromptTokens: p, CachedTokens: c})
		if p == 0 {
			continue
		}
		if math.IsNaN(d.HitRate) || math.IsNaN(d.Miss) {
			t.Fatalf("a non-zero prompt delta must measure both ratios, got hr=%v miss=%v", d.HitRate, d.Miss)
		}
		if d.HitRate < 0 || d.HitRate > 1 {
			t.Fatalf("hit rate must lie in [0,1], got %v", d.HitRate)
		}
		complement := float64(p) - float64(c)
		if diff := math.Abs(d.Miss - complement); diff > 1e-9*float64(p)+1e-9 {
			t.Fatalf("miss=%v must agree with Δprompt−Δcached=%v within rounding", d.Miss, complement)
		}
	}
}
