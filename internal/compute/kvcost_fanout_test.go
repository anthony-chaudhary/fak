package compute

import "testing"

// TestKVEvictionCostTiesSharedAndLonelySpans is the repro: today's KVEvictionCost cannot see
// concurrent sharer fan-out, so a span shared by 8 live sessions scores IDENTICALLY to a
// one-shot leaf with the same Tokens/Hits/Bytes — the gap #2670 names.
func TestKVEvictionCostTiesSharedAndLonelySpans(t *testing.T) {
	shared := KVSpanStats{Tokens: 100, Bytes: 400, Hits: 0, Sharers: 8}
	lonely := KVSpanStats{Tokens: 100, Bytes: 400, Hits: 0, Sharers: 1}
	if cs, cl := KVEvictionCost(shared), KVEvictionCost(lonely); cs != cl {
		t.Fatalf("KVEvictionCost must be blind to Sharers (the gap): shared=%v lonely=%v", cs, cl)
	}
}

// TestKVEvictionCostFanoutSharedSpanCostsMoreToLose proves the fix: two spans with equal
// Tokens/Hits/Bytes, one Sharers=8 and one Sharers=1, must NOT tie under
// KVEvictionCostFanout — the shared span costs strictly more to lose, so
// PickEvictionVictimFanout keeps it and evicts the lonely one instead of mis-picking on the
// LRU tie-break.
func TestKVEvictionCostFanoutSharedSpanCostsMoreToLose(t *testing.T) {
	shared := KVSpanStats{Tokens: 100, Bytes: 400, Hits: 0, Sharers: 8, LastUsed: 1}
	lonely := KVSpanStats{Tokens: 100, Bytes: 400, Hits: 0, Sharers: 1, LastUsed: 2}

	cShared, cLonely := KVEvictionCostFanout(shared), KVEvictionCostFanout(lonely)
	if cShared <= cLonely {
		t.Fatalf("a span shared by 8 sessions must cost strictly more to lose than a lonely span: shared=%v lonely=%v", cShared, cLonely)
	}
	if want := cLonely * 8; cShared != want {
		t.Fatalf("shared cost = %v, want exactly %v (8x the lonely cost, uniform Tokens/Hits/Bytes)", cShared, want)
	}

	spans := []KVSpanStats{shared, lonely}
	idx := PickEvictionVictimFanout(spans)
	if idx != 1 {
		t.Fatalf("victim = span %d, want 1 (the lonely span) — the shared span must be kept despite its newer LastUsed making it the LRU pick", idx)
	}
}

// TestKVEvictionCostFanoutReducesToKVEvictionCostOnSingleConsumer proves the required
// reduction: Sharers <= 1 (including the zero value) must score byte-identically to
// KVEvictionCost, so every existing caller that never sets Sharers is unaffected.
func TestKVEvictionCostFanoutReducesToKVEvictionCostOnSingleConsumer(t *testing.T) {
	for _, sharers := range []int{0, 1} {
		s := KVSpanStats{Tokens: 50, Bytes: 200, Hits: 3, Sharers: sharers}
		if got, want := KVEvictionCostFanout(s), KVEvictionCost(s); got != want {
			t.Fatalf("Sharers=%d: KVEvictionCostFanout = %v, want %v (byte-identical reduction)", sharers, got, want)
		}
	}
}

// TestKVEvictionCostFanoutFailOpenOnUnknownBytes: the +Inf fail-open on unknown footprint is
// inherited unchanged — fan-out never makes an unknown-size span a preferred victim.
func TestKVEvictionCostFanoutFailOpenOnUnknownBytes(t *testing.T) {
	s := KVSpanStats{Tokens: 100, Bytes: 0, Hits: 5, Sharers: 8}
	if got := KVEvictionCostFanout(s); got != KVEvictionCost(s) {
		t.Fatalf("fail-open case: KVEvictionCostFanout = %v, want %v (+Inf, matching KVEvictionCost)", got, KVEvictionCost(s))
	}
}

// TestPickEvictionVictimFanoutReducesToPickEvictionVictimOnUniformSharers: when every span
// has Sharers <= 1, the fanout picker's choice must match the plain picker's exactly — no
// divergence on today's single-consumer inputs.
func TestPickEvictionVictimFanoutReducesToPickEvictionVictimOnUniformSharers(t *testing.T) {
	spans := []KVSpanStats{
		{Tokens: 100, Bytes: 400, Hits: 3, LastUsed: 10},
		{Tokens: 10, Bytes: 40, Hits: 0, LastUsed: 5},
		{Tokens: 50, Bytes: 400, Hits: 1, LastUsed: 1},
	}
	want := PickEvictionVictim(spans)
	got := PickEvictionVictimFanout(spans)
	if got != want {
		t.Fatalf("PickEvictionVictimFanout = %d, want %d (must match PickEvictionVictim when Sharers is unset everywhere)", got, want)
	}
}
