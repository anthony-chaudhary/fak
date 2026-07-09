package cacheobs

import (
	"math"
	"sync"
	"testing"
)

func TestObserveAccumulatesAndBuckets(t *testing.T) {
	o := New()
	// turn 1: cold first prefill (nothing reused)
	o.Observe(100, 0)
	// turn 2: partial reuse
	o.Observe(200, 100)
	// turn 3: frozen-regime reuse (>= 90%)
	o.Observe(1000, 990)

	s := o.Snapshot()
	if s.Turns != 3 {
		t.Fatalf("turns = %d, want 3", s.Turns)
	}
	if s.PromptTokens != 1300 || s.ReusedTokens != 1090 {
		t.Fatalf("tokens = prompt %d reused %d, want 1300/1090", s.PromptTokens, s.ReusedTokens)
	}
	if s.ColdTurns != 1 || s.PartialTurns != 1 || s.FrozenTurns != 1 {
		t.Fatalf("buckets cold=%d partial=%d frozen=%d, want 1/1/1", s.ColdTurns, s.PartialTurns, s.FrozenTurns)
	}
	want := 1090.0 / 1300.0
	if d := s.ReuseRatio - want; d > 1e-9 || d < -1e-9 {
		t.Fatalf("reuse ratio = %v, want %v", s.ReuseRatio, want)
	}
}

func TestBucketBoundaries(t *testing.T) {
	// exactly FrozenFloor -> frozen; just under -> partial; exactly ColdCeil -> partial;
	// just under ColdCeil -> cold.
	cases := []struct {
		prompt, reused        int
		frozen, partial, cold uint64
	}{
		{100, 90, 1, 0, 0}, // 0.90 == FrozenFloor -> frozen
		{100, 89, 0, 1, 0}, // 0.89 -> partial
		{100, 10, 0, 1, 0}, // 0.10 == ColdCeil -> NOT cold (cold is strictly <)
		{100, 9, 0, 0, 1},  // 0.09 -> cold
	}
	for _, c := range cases {
		o := New()
		o.Observe(c.prompt, c.reused)
		s := o.Snapshot()
		if s.FrozenTurns != c.frozen || s.PartialTurns != c.partial || s.ColdTurns != c.cold {
			t.Errorf("Observe(%d,%d): buckets frozen=%d partial=%d cold=%d, want %d/%d/%d",
				c.prompt, c.reused, s.FrozenTurns, s.PartialTurns, s.ColdTurns, c.frozen, c.partial, c.cold)
		}
	}
}

func TestObserveClampsAndIgnores(t *testing.T) {
	o := New()
	o.Observe(0, 50)    // non-positive prompt: ignored
	o.Observe(-5, 1)    // negative prompt: ignored
	o.Observe(100, 250) // reused > prompt: clamped to prompt
	o.Observe(100, -10) // negative reused: clamped to 0
	s := o.Snapshot()
	if s.Turns != 2 {
		t.Fatalf("turns = %d, want 2 (the two zero/negative-prompt calls are ignored)", s.Turns)
	}
	if s.ReusedTokens != 100 { // 100 (clamped from 250) + 0 (clamped from -10)
		t.Fatalf("reused = %d, want 100 (clamped)", s.ReusedTokens)
	}
	if s.PromptTokens != 200 {
		t.Fatalf("prompt = %d, want 200", s.PromptTokens)
	}
}

func TestNilObserverSafe(t *testing.T) {
	var o *Observer
	o.Observe(100, 50) // must not panic
	if s := o.Snapshot(); s.Turns != 0 {
		t.Fatalf("nil observer snapshot turns = %d, want 0", s.Turns)
	}
}

func TestIdleRatioIsZero(t *testing.T) {
	if s := New().Snapshot(); s.ReuseRatio != 0 {
		t.Fatalf("idle reuse ratio = %v, want 0 (no phantom ratio)", s.ReuseRatio)
	}
}

// #1946: many turns that all hit the frozen ceiling (reuse == prompt every turn)
// must report ReuseRatio exactly 1.0 and every turn bucketed as frozen -- the
// headline regime the cliff metric exists to show, previously untested.
func TestAllFrozenRatioIsExactlyOne(t *testing.T) {
	o := New()
	const nTurns = 10_000
	for i := 0; i < nTurns; i++ {
		o.Observe(37, 37) // full reuse every turn: ratio == 1.0 exactly
	}
	s := o.Snapshot()
	if s.Turns != nTurns {
		t.Fatalf("turns = %d, want %d", s.Turns, nTurns)
	}
	if s.FrozenTurns != nTurns {
		t.Fatalf("frozen turns = %d, want %d (every turn hits the frozen ceiling)", s.FrozenTurns, nTurns)
	}
	if s.PartialTurns != 0 || s.ColdTurns != 0 {
		t.Fatalf("partial=%d cold=%d, want 0/0 for an all-frozen run", s.PartialTurns, s.ColdTurns)
	}
	if s.ReuseRatio != 1.0 {
		t.Fatalf("ReuseRatio = %v, want exactly 1.0", s.ReuseRatio)
	}
}

// #1946: the accumulators must saturate at math.MaxUint64 instead of silently
// wrapping back down to a small number once a long-lived process nears the
// ceiling. Directly seeding the unexported fields (this test is in-package) is
// the only way to reach the ceiling without actually accumulating 2^64 tokens.
func TestObserveSaturatesInsteadOfWrapping(t *testing.T) {
	o := New()
	o.turns = math.MaxUint64
	o.promptTokens = math.MaxUint64
	o.reusedTokens = math.MaxUint64
	o.frozen = math.MaxUint64

	o.Observe(100, 100) // ratio 1.0 -> would land in the already-saturated frozen bucket

	s := o.Snapshot()
	if s.Turns != math.MaxUint64 {
		t.Fatalf("turns = %d, want saturated at MaxUint64", s.Turns)
	}
	if s.PromptTokens != math.MaxUint64 || s.ReusedTokens != math.MaxUint64 {
		t.Fatalf("tokens did not saturate: prompt=%d reused=%d, want both MaxUint64", s.PromptTokens, s.ReusedTokens)
	}
	if s.FrozenTurns != math.MaxUint64 {
		t.Fatalf("frozen bucket = %d, want saturated at MaxUint64", s.FrozenTurns)
	}
	if s.ReuseRatio != 1.0 {
		t.Fatalf("reuse ratio at saturation = %v, want a sane 1.0, not a wrapped/NaN value", s.ReuseRatio)
	}
}

// #3367: the per-turn reuse-ratio histogram must expose the SHAPE of the partial
// range that the single o.partial scalar collapses — a bimodal run (turns piled at
// ~0.2 and ~0.8) must land in distinct buckets, not one lump.
func TestReuseHistogramSeparatesBimodalShape(t *testing.T) {
	o := New()
	for i := 0; i < 3; i++ {
		o.Observe(100, 15) // 0.15 -> le=0.2 bucket
	}
	for i := 0; i < 5; i++ {
		o.Observe(100, 75) // 0.75 -> le=0.8 bucket
	}
	s := o.Snapshot()
	if s.PartialTurns != 8 {
		t.Fatalf("partial turns = %d, want 8 (regime counter unchanged by histogram)", s.PartialTurns)
	}
	var want [len(ReuseRatioBuckets)]uint64
	want[1] = 3 // le=0.2
	want[7] = 5 // le=0.8
	if s.ReuseHistTurns != want {
		t.Fatalf("hist = %v, want %v (bimodal shape must be visible)", s.ReuseHistTurns, want)
	}
}

// #3367: bucket edges follow Prometheus `le` (<=) semantics, every observed turn is
// counted exactly once, and cold keeps its own strict (< ColdCeil) counter — a turn
// at exactly ColdCeil is le=0.1 in the histogram but NOT cold.
func TestReuseHistogramBoundariesAndColdCounter(t *testing.T) {
	o := New()
	o.Observe(100, 0)   // 0.00 -> le=0.1, cold
	o.Observe(100, 10)  // 0.10 == ColdCeil -> le=0.1, NOT cold (partial)
	o.Observe(100, 11)  // 0.11 -> le=0.2
	o.Observe(100, 90)  // 0.90 -> le=0.9, frozen
	o.Observe(100, 91)  // 0.91 -> le=1.0
	o.Observe(100, 100) // 1.00 -> le=1.0
	s := o.Snapshot()
	var want [len(ReuseRatioBuckets)]uint64
	want[0] = 2 // le=0.1: the cold turn + the exactly-ColdCeil turn
	want[1] = 1
	want[8] = 1
	want[9] = 2
	if s.ReuseHistTurns != want {
		t.Fatalf("hist = %v, want %v", s.ReuseHistTurns, want)
	}
	if s.ColdTurns != 1 {
		t.Fatalf("cold turns = %d, want 1 (cold stays strictly < ColdCeil)", s.ColdTurns)
	}
	var sum uint64
	for _, n := range s.ReuseHistTurns {
		sum += n
	}
	if sum != s.Turns {
		t.Fatalf("histogram total = %d, want Turns = %d (every turn counted once)", sum, s.Turns)
	}
}

// #3367: a saturated bucket must stick at MaxUint64, matching the other accumulators.
func TestReuseHistogramSaturates(t *testing.T) {
	o := New()
	o.reuseHist[len(ReuseRatioBuckets)-1] = math.MaxUint64
	o.Observe(100, 100) // ratio 1.0 -> the already-saturated le=1.0 bucket
	if got := o.Snapshot().ReuseHistTurns[len(ReuseRatioBuckets)-1]; got != math.MaxUint64 {
		t.Fatalf("le=1.0 bucket = %d, want saturated at MaxUint64", got)
	}
}

func TestConcurrentObserveIsRace_free(t *testing.T) {
	o := New()
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				o.Observe(100, 90)
			}
		}()
	}
	wg.Wait()
	if s := o.Snapshot(); s.Turns != 5000 {
		t.Fatalf("turns = %d, want 5000", s.Turns)
	}
}
