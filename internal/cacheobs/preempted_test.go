package cacheobs

import (
	"math"
	"testing"
)

// #3895 (a): an eviction-then-refetch of the same key books its recompute to the
// PREEMPTED bucket, not the cold bucket. Turn 1 serves key K fully warm; admission
// then evicts K while still warm; the refetch of the SAME key re-prefills the whole
// prompt, and the caller (who holds the lifecycle StateEvicted witness) taps it via
// ObservePreempted — so the self-inflicted loss is visible as such.
func TestPreemptedRefetchBooksToPreemptedNotCold(t *testing.T) {
	o := New()
	o.Observe(100, 100) // key K served fully warm: no miss at all
	// admission evicts K (still warm); the refetch recomputes all 100 tokens
	o.ObservePreempted(100, 0, 0, 100)
	s := o.Snapshot()
	if s.PreemptedReuseLostTokens != 100 {
		t.Fatalf("preempted = %d, want 100 (the whole self-inflicted refetch)", s.PreemptedReuseLostTokens)
	}
	if s.ColdMissTokens != 0 {
		t.Fatalf("cold miss = %d, want 0 (a preempted refetch must NOT masquerade as cold)", s.ColdMissTokens)
	}
	if s.PromptTokens != 200 || s.ReusedTokens != 100 || s.Turns != 2 {
		t.Fatalf("aggregate perturbed: prompt=%d reused=%d turns=%d, want 200/100/2",
			s.PromptTokens, s.ReusedTokens, s.Turns)
	}
}

// #3895 (b): a genuine cold miss — a key never cached — books to the COLD bucket,
// not preempted, on every tap that lacks (or denies) an eviction witness.
func TestGenuineColdMissBooksToColdNotPreempted(t *testing.T) {
	o := New()
	o.Observe(1000, 0)                 // legacy tap, first prefill of a fresh key
	o.ObserveSplit(500, 0, 0)          // split tap, still no eviction witness
	o.ObservePreempted(300, 0, 0, 0)   // preempt-aware tap that witnessed NO preemption
	o.ObservePreempted(200, 0, 150, 0) // partial reuse, un-reused remainder is cold
	s := o.Snapshot()
	if s.PreemptedReuseLostTokens != 0 {
		t.Fatalf("preempted = %d, want 0 (never a fabricated self-infliction)", s.PreemptedReuseLostTokens)
	}
	if want := uint64(1000 + 500 + 300 + 50); s.ColdMissTokens != want {
		t.Fatalf("cold miss = %d, want %d", s.ColdMissTokens, want)
	}
}

// #3895 (c): the pre-existing aggregate still reconciles. The same turn sequence fed
// through ObserveSplit and through ObservePreempted yields IDENTICAL pre-split stats —
// turns, token totals, both ratios, regime buckets, histogram — and the new split
// itself reconciles: preempted + cold == prompt - reused.
func TestPreemptedAttributionKeepsAggregateReconciled(t *testing.T) {
	turns := []struct{ prompt, cacheable, reused, preempted int }{
		{1000, 900, 300, 400}, // lookup-matched but evicted mid-flight: part self-inflicted
		{1000, 800, 800, 0},   // fully realized
		{100, 0, 0, 100},      // eviction-then-refetch: whole miss self-inflicted
		{200, 50, 50, 0},      // genuine cold remainder
	}
	split, pre := New(), New()
	for _, tn := range turns {
		split.ObserveSplit(tn.prompt, tn.cacheable, tn.reused)
		pre.ObservePreempted(tn.prompt, tn.cacheable, tn.reused, tn.preempted)
	}
	a, b := split.Snapshot(), pre.Snapshot()
	// Blank out the two new fields; everything else must match exactly.
	a.PreemptedReuseLostTokens, a.ColdMissTokens = 0, 0
	b.PreemptedReuseLostTokens, b.ColdMissTokens = 0, 0
	if a != b {
		t.Fatalf("attribution perturbed the aggregate:\nObserveSplit    = %+v\nObservePreempted = %+v", a, b)
	}
	s := pre.Snapshot()
	if got, want := s.PreemptedReuseLostTokens+s.ColdMissTokens, s.PromptTokens-s.ReusedTokens; got != want {
		t.Fatalf("preempted %d + cold %d = %d, want prompt-reused = %d",
			s.PreemptedReuseLostTokens, s.ColdMissTokens, got, want)
	}
	if s.PreemptedReuseLostTokens != 500 { // 400 + 0 + 100 + 0
		t.Fatalf("preempted = %d, want 500", s.PreemptedReuseLostTokens)
	}
	// And the ObserveSplit-only observer books its whole miss cold — same reconcile.
	c := split.Snapshot()
	if c.PreemptedReuseLostTokens != 0 || c.ColdMissTokens != c.PromptTokens-c.ReusedTokens {
		t.Fatalf("split-only observer: preempted=%d cold=%d, want 0 and %d",
			c.PreemptedReuseLostTokens, c.ColdMissTokens, c.PromptTokens-c.ReusedTokens)
	}
}

// #3895: legacy taps keep their exact pre-split behavior — Observe and ObserveSplit
// book the whole miss cold and never invent a preempted token.
func TestLegacyObservePathsBookMissCold(t *testing.T) {
	o := New()
	o.Observe(200, 150)
	o.ObserveSplit(1000, 900, 300)
	s := o.Snapshot()
	if s.PreemptedReuseLostTokens != 0 {
		t.Fatalf("preempted = %d, want 0 for eviction-blind taps", s.PreemptedReuseLostTokens)
	}
	if want := uint64(50 + 700); s.ColdMissTokens != want {
		t.Fatalf("cold miss = %d, want %d", s.ColdMissTokens, want)
	}
}

// #3895: preemptedLostTokens is clamped into [0, promptTokens - reusedPrefixTokens]
// after the ObserveSplit clamps — a miscount can never book more preempted loss than
// the turn actually missed, or a negative amount.
func TestObservePreemptedClamps(t *testing.T) {
	o := New()
	o.ObservePreempted(100, 0, 60, 999) // miss is 40: preempted clamped 999 -> 40
	o.ObservePreempted(100, 0, 0, -5)   // negative: clamped to 0, whole miss cold
	o.ObservePreempted(100, 0, 300, 10) // reused clamped to 100 first: miss 0, preempted 0
	s := o.Snapshot()
	if s.PreemptedReuseLostTokens != 40 {
		t.Fatalf("preempted = %d, want 40 (clamped to the turn's miss)", s.PreemptedReuseLostTokens)
	}
	if s.ColdMissTokens != 100 {
		t.Fatalf("cold miss = %d, want 100", s.ColdMissTokens)
	}
	if got, want := s.PreemptedReuseLostTokens+s.ColdMissTokens, s.PromptTokens-s.ReusedTokens; got != want {
		t.Fatalf("split does not reconcile: %d, want %d", got, want)
	}
}

// #3895: ObservePreempted keeps the ObserveSplit contract for everything it did not
// add — nil receiver and non-positive prompt are safe no-ops, and the regime buckets
// stay keyed on the REALIZED ratio.
func TestObservePreemptedContract(t *testing.T) {
	var nilObs *Observer
	nilObs.ObservePreempted(100, 0, 0, 100) // must not panic
	o := New()
	o.ObservePreempted(0, 0, 0, 50)  // non-positive prompt: ignored
	o.ObservePreempted(-1, 0, 0, 50) // ignored
	if s := o.Snapshot(); s.Turns != 0 || s.PreemptedReuseLostTokens != 0 || s.ColdMissTokens != 0 {
		t.Fatalf("ignored turns leaked into counters: %+v", s)
	}
	o.ObservePreempted(100, 0, 0, 100)  // realized 0.00 -> cold regime turn
	o.ObservePreempted(100, 50, 50, 25) // realized 0.50 -> partial
	o.ObservePreempted(100, 95, 95, 5)  // realized 0.95 -> frozen
	s := o.Snapshot()
	if s.ColdTurns != 1 || s.PartialTurns != 1 || s.FrozenTurns != 1 {
		t.Fatalf("regime buckets cold=%d partial=%d frozen=%d, want 1/1/1 (keyed on realized)",
			s.ColdTurns, s.PartialTurns, s.FrozenTurns)
	}
}

// #3895: the two new accumulators saturate at MaxUint64 like every other counter.
func TestPreemptedAndColdMissSaturate(t *testing.T) {
	o := New()
	o.preemptedReuseLostTokens = math.MaxUint64
	o.coldMissTokens = math.MaxUint64
	o.ObservePreempted(100, 0, 0, 60) // 60 preempted + 40 cold onto saturated counters
	s := o.Snapshot()
	if s.PreemptedReuseLostTokens != math.MaxUint64 {
		t.Fatalf("preempted = %d, want saturated at MaxUint64", s.PreemptedReuseLostTokens)
	}
	if s.ColdMissTokens != math.MaxUint64 {
		t.Fatalf("cold miss = %d, want saturated at MaxUint64", s.ColdMissTokens)
	}
}

// #3895: an idle observer reports zero in both new buckets — no phantom attribution.
func TestIdlePreemptedSplitIsZero(t *testing.T) {
	if s := New().Snapshot(); s.PreemptedReuseLostTokens != 0 || s.ColdMissTokens != 0 {
		t.Fatalf("idle split = preempted %d cold %d, want 0/0", s.PreemptedReuseLostTokens, s.ColdMissTokens)
	}
}
