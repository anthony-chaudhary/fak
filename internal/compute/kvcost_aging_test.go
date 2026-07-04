package compute

import (
	"math"
	"testing"
)

// TestKVEvictionCostAgedReducesToDeAgedAtZeroStamp proves the reduction: when every span's
// AgeStamp is 0 (the zero value — no caller has ever stamped it), KVEvictionCostAged is
// byte-identical to KVEvictionCost and PickEvictionVictimAged picks the SAME victim as
// PickEvictionVictim. This is the "never a divergence on today's inputs" guarantee: any
// existing caller that leaves AgeStamp unset sees no behavior change from this file.
func TestKVEvictionCostAgedReducesToDeAgedAtZeroStamp(t *testing.T) {
	spans := []KVSpanStats{
		{Tokens: 100, Bytes: 400, Hits: 3},
		{Tokens: 10, Bytes: 40, Hits: 0},
		{Tokens: 50, Bytes: 100, Hits: 1, LastUsed: 5},
	}
	for i, s := range spans {
		if got, want := KVEvictionCostAged(s), KVEvictionCost(s); got != want {
			t.Fatalf("span %d: KVEvictionCostAged = %v, want %v (== KVEvictionCost at zero stamp)", i, got, want)
		}
	}
	wantIdx := PickEvictionVictim(spans)
	gotIdx, _ := PickEvictionVictimAged(spans)
	if gotIdx != wantIdx {
		t.Fatalf("PickEvictionVictimAged at zero stamp = %d, want %d (PickEvictionVictim's pick)", gotIdx, wantIdx)
	}
}

// TestPickEvictionVictimAgedEvictsStaleHotSpan is the anti-pollution witness: a span that
// was hot LONG AGO (high Hits, so a HIGH de-aged cost-of-losing) but has not been referenced
// since the pool's aging clock was near zero, competing against fresh one-shot spans stamped
// at the pool's current, much higher clock. Under the de-aged KVEvictionCost/
// PickEvictionVictim the stale-hot span's cost (11) beats every one-shot span's cost (1), so
// it is NEVER the victim — it is held forever regardless of how many fresh spans churn
// through, textbook cache pollution. Under the aged function the frozen low stamp (0) plus
// its own cost (11) is LOWER than a fresh span's inflated stamp (100) plus its cost (1), so
// the stale-hot span becomes the correct victim once the clock has advanced past it.
func TestPickEvictionVictimAgedEvictsStaleHotSpan(t *testing.T) {
	staleHot := KVSpanStats{Tokens: 100, Bytes: 100, Hits: 10, AgeStamp: 0}    // cost = 100*11/100 = 11
	freshOneShot := KVSpanStats{Tokens: 10, Bytes: 10, Hits: 0, AgeStamp: 100} // cost = 10*1/10 = 1

	spans := []KVSpanStats{staleHot, freshOneShot}

	// De-aged: the fresh span is always cheaper, so it is picked — staleHot is retained.
	if got := PickEvictionVictim(spans); got != 1 {
		t.Fatalf("de-aged PickEvictionVictim = %d, want 1 (fresh one-shot; stale-hot wrongly retained is the bug this issue fixes)", got)
	}

	// Aged: the stale-hot span's frozen stamp makes it the true victim.
	gotIdx, gotClock := PickEvictionVictimAged(spans)
	if gotIdx != 0 {
		t.Fatalf("aged PickEvictionVictimAged = %d, want 0 (stale-hot span; aging must let it become evictable)", gotIdx)
	}
	if wantClock := 11.0; gotClock != wantClock {
		t.Fatalf("newClock = %v, want %v (the evicted victim's own aged cost)", gotClock, wantClock)
	}

	// Multiple fresh spans churning through changes nothing under de-aged: staleHot is
	// STILL never picked, however many one-shot spans are considered alongside it — the
	// "held forever against a scan" failure mode named in the issue.
	manySpans := []KVSpanStats{
		staleHot,
		{Tokens: 10, Bytes: 10, Hits: 0, AgeStamp: 50, LastUsed: 1},
		{Tokens: 10, Bytes: 10, Hits: 0, AgeStamp: 75, LastUsed: 2},
		{Tokens: 10, Bytes: 10, Hits: 0, AgeStamp: 100, LastUsed: 3},
	}
	if got := PickEvictionVictim(manySpans); got == 0 {
		t.Fatalf("de-aged PickEvictionVictim picked the stale-hot span (idx 0) — test setup no longer demonstrates the bug")
	}
	if gotIdx, _ := PickEvictionVictimAged(manySpans); gotIdx != 0 {
		t.Fatalf("aged PickEvictionVictimAged over a scan of fresh spans = %d, want 0 (stale-hot must still be the cheapest once the clock has moved on)", gotIdx)
	}
}

// TestPickEvictionVictimAgedRespectsPinsAndLeases mirrors PickEvictionVictim's hard
// exclusions: a Pinned or Leased span is never a candidate regardless of its aged cost, and
// -1 (with newClock 0) signals nothing is evictable.
func TestPickEvictionVictimAgedRespectsPinsAndLeases(t *testing.T) {
	spans := []KVSpanStats{
		{Tokens: 1, Bytes: 100, Hits: 0, AgeStamp: 0, Pinned: true}, // cheapest cost (0.01) but pinned — must be skipped
		{Tokens: 1, Bytes: 100, Hits: 0, AgeStamp: 0, Leased: true}, // same cost but leased — must also be skipped
	}
	if idx, clock := PickEvictionVictimAged(spans); idx != -1 || clock != 0 {
		t.Fatalf("PickEvictionVictimAged over all-pinned/leased spans = (%d, %v), want (-1, 0)", idx, clock)
	}

	evictable := KVSpanStats{Tokens: 1, Bytes: 100, Hits: 0, AgeStamp: 0} // cheap: cost = 1*1/100 = 0.01
	mixed := append(append([]KVSpanStats{}, spans...), evictable)
	gotIdx, _ := PickEvictionVictimAged(mixed)
	if gotIdx != len(mixed)-1 {
		t.Fatalf("PickEvictionVictimAged = %d, want %d (the only unpinned/unleased span)", gotIdx, len(mixed)-1)
	}
}

// TestKVEvictionCostAgedPreservesUnknownBytesFailOpen: a span with unknown footprint still
// scores +Inf under the aged function regardless of AgeStamp, so a long-unstamped span never
// becomes preferable to an untracked-footprint one merely by aging.
func TestKVEvictionCostAgedPreservesUnknownBytesFailOpen(t *testing.T) {
	s := KVSpanStats{Tokens: 100, Bytes: 0, Hits: 5, AgeStamp: 1_000_000}
	if got := KVEvictionCostAged(s); !math.IsInf(got, 1) {
		t.Fatalf("unknown footprint should score +Inf even under a large AgeStamp, got %v", got)
	}
}
