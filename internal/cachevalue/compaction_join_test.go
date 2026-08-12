package cachevalue

import "testing"

// TestJoinProviderUsageResolvesFireToItsOwnTurn is the #2788 acceptance: any fire joins 1:1 to
// its provider turn's cache_read/cache_creation. Two fires and two usage records arrive on
// independent axes and in DIFFERENT orders; each fire must pick up its OWN turn's counters, not
// its neighbour's — the failure the shared coordinate exists to make impossible.
func TestJoinProviderUsageResolvesFireToItsOwnTurn(t *testing.T) {
	k7 := CompactionJoinKey{TurnSeq: 7, MonotonicTSNano: 1_000}
	k9 := CompactionJoinKey{TurnSeq: 9, MonotonicTSNano: 4_000}
	samples := []CompactionSample{
		{ShedTokens: 100000, ShedWarmTokens: 0, JoinKey: k7},
		{ShedTokens: 100000, ShedWarmTokens: 100000, JoinKey: k9},
	}
	// Usage arrives in the opposite order — a positional pairing would cross the two fires.
	usage := []ProviderTurnUsage{
		{Key: k9, CacheReadTokens: 300000, CacheCreationTokens: 50000},
		{Key: k7, CacheReadTokens: 120000, CacheCreationTokens: 5000},
	}

	res := JoinProviderUsage(samples, usage)
	if !res.Clean() || res.Joined != 2 {
		t.Fatalf("join = %+v, want 2 joined and clean (every stamped fire resolved 1:1)", res)
	}
	if got := res.Samples[0]; got.RewriteCacheCreationTokens != 5000 || got.ObservedCacheReadTokens != 120000 || !got.RewriteKnown {
		t.Errorf("turn 7 fire joined to %d creation / %d read (known=%v), want 5000 / 120000 — it took the wrong turn's usage",
			got.RewriteCacheCreationTokens, got.ObservedCacheReadTokens, got.RewriteKnown)
	}
	if got := res.Samples[1]; got.RewriteCacheCreationTokens != 50000 || got.ObservedCacheReadTokens != 300000 || !got.RewriteKnown {
		t.Errorf("turn 9 fire joined to %d creation / %d read (known=%v), want 50000 / 300000",
			got.RewriteCacheCreationTokens, got.ObservedCacheReadTokens, got.RewriteKnown)
	}

	// The whole point of the join: each fire's net verdict is now scored against ITS OWN turn.
	// The cold shed with a small re-warm is net cheaper; the warm shed with a big one is hostile.
	if v := ClassifyCompaction(res.Samples[0]); v.Reason != ReasonCompactionNetCheaper {
		t.Errorf("turn 7 verdict = %q, want %q", v.Reason, ReasonCompactionNetCheaper)
	}
	if v := ClassifyCompaction(res.Samples[1]); v.Reason != ReasonCompactionCacheHostile {
		t.Errorf("turn 9 verdict = %q, want %q", v.Reason, ReasonCompactionCacheHostile)
	}
}

// TestJoinProviderUsageRefireSameTurnStaysDistinct guards the monotonic half of the key: a retry
// compacts the SAME turn sequence again at a later reading, and the two attempts must stay
// separable. Keyed on turn sequence alone both fires would collide and read Ambiguous.
func TestJoinProviderUsageRefireSameTurnStaysDistinct(t *testing.T) {
	first := CompactionJoinKey{TurnSeq: 12, MonotonicTSNano: 500}
	retry := CompactionJoinKey{TurnSeq: 12, MonotonicTSNano: 900}
	res := JoinProviderUsage(
		[]CompactionSample{{ShedTokens: 10000, JoinKey: first}, {ShedTokens: 20000, JoinKey: retry}},
		[]ProviderTurnUsage{
			{Key: first, CacheReadTokens: 1000, CacheCreationTokens: 100},
			{Key: retry, CacheReadTokens: 2000, CacheCreationTokens: 200},
		},
	)
	if !res.Clean() || res.Joined != 2 {
		t.Fatalf("join = %+v, want both re-fires of turn 12 joined 1:1", res)
	}
	if res.Samples[0].RewriteCacheCreationTokens != 100 || res.Samples[1].RewriteCacheCreationTokens != 200 {
		t.Errorf("re-fire creation = %d / %d, want 100 / 200 (the monotonic reading separates the attempts)",
			res.Samples[0].RewriteCacheCreationTokens, res.Samples[1].RewriteCacheCreationTokens)
	}
}

// TestJoinProviderUsageUnmatchedAbstains is the safety property: a stamped fire whose turn's
// usage never arrived must NOT keep a net claim. The join withdraws the counters so
// ClassifyCompaction abstains, rather than scoring against a turn it cannot prove.
func TestJoinProviderUsageUnmatchedAbstains(t *testing.T) {
	res := JoinProviderUsage(
		[]CompactionSample{{
			ShedTokens: 100000, ShedWarmTokens: 100000,
			// A caller pre-pasted a counter by convention; the join found no matching turn.
			RewriteCacheCreationTokens: 50000, RewriteKnown: true,
			JoinKey: CompactionJoinKey{TurnSeq: 3, MonotonicTSNano: 77},
		}},
		[]ProviderTurnUsage{{Key: CompactionJoinKey{TurnSeq: 4, MonotonicTSNano: 88}, CacheCreationTokens: 9}},
	)
	if res.Unmatched != 1 || res.Joined != 0 || res.Clean() {
		t.Fatalf("join = %+v, want 1 unmatched and not clean", res)
	}
	got := res.Samples[0]
	if got.RewriteKnown || got.RewriteCacheCreationTokens != 0 {
		t.Errorf("unmatched sample kept known=%v creation=%d, want the net claim withdrawn",
			got.RewriteKnown, got.RewriteCacheCreationTokens)
	}
	if v := ClassifyCompaction(got); v.Reason != ReasonCompactionEconomicsUnknown || v.NetKnown {
		t.Errorf("unmatched verdict = %q (net known=%v), want %q — an unproven join must not score a net",
			v.Reason, v.NetKnown, ReasonCompactionEconomicsUnknown)
	}
	// The WITNESSED shed story survives the abstain: only the net claim is withheld.
	if v := ClassifyCompaction(got); !within(v.ShedValueSavedTokEq, 10000, tol) {
		t.Errorf("shed value = %.3f, want 10000 even when the join failed", v.ShedValueSavedTokEq)
	}
}

// TestJoinProviderUsageAmbiguousRefusesBothSides: a key duplicated on EITHER side breaks the 1:1
// guarantee, so the join must refuse to pick a winner — for every sample carrying that key, not
// just the second one seen.
func TestJoinProviderUsageAmbiguousRefusesBothSides(t *testing.T) {
	dup := CompactionJoinKey{TurnSeq: 5, MonotonicTSNano: 42}

	// Duplicated on the FIRE side: two samples claim the same coordinate.
	res := JoinProviderUsage(
		[]CompactionSample{{ShedTokens: 100, JoinKey: dup}, {ShedTokens: 200, JoinKey: dup}},
		[]ProviderTurnUsage{{Key: dup, CacheCreationTokens: 7}},
	)
	if res.Ambiguous != 2 || res.Joined != 0 || res.Clean() {
		t.Fatalf("fire-side dup = %+v, want both samples refused as ambiguous", res)
	}
	for i, s := range res.Samples {
		if s.RewriteKnown {
			t.Errorf("ambiguous sample %d kept a net claim, want it withdrawn", i)
		}
	}

	// Duplicated on the USAGE side: one fire, two candidate turns, no winner.
	res = JoinProviderUsage(
		[]CompactionSample{{ShedTokens: 100, JoinKey: dup}},
		[]ProviderTurnUsage{{Key: dup, CacheCreationTokens: 7}, {Key: dup, CacheCreationTokens: 9}},
	)
	if res.Ambiguous != 1 || res.Joined != 0 || res.Samples[0].RewriteKnown {
		t.Fatalf("usage-side dup = %+v, want the sample refused rather than a coin flip", res)
	}
}

// TestJoinProviderUsageUnstampedPassesThrough: a sample with no turn coordinate is the honest
// state of a byte-level caller, not a failed join. It is counted apart and returned VERBATIM —
// its hand-assembled counters are not second-guessed, so the pre-join callers keep working.
func TestJoinProviderUsageUnstampedPassesThrough(t *testing.T) {
	legacy := CompactionSample{
		ShedTokens: 100000, ShedWarmTokens: 0,
		RewriteCacheCreationTokens: 5000, RewriteKnown: true,
	}
	res := JoinProviderUsage(
		[]CompactionSample{legacy},
		[]ProviderTurnUsage{{Key: CompactionJoinKey{TurnSeq: 1, MonotonicTSNano: 1}, CacheCreationTokens: 999}},
	)
	if res.Unstamped != 1 || res.Joined != 0 || !res.Clean() {
		t.Fatalf("join = %+v, want 1 unstamped, clean (an absent coordinate is not a failed join)", res)
	}
	if res.Samples[0] != legacy {
		t.Errorf("unstamped sample = %+v, want it returned verbatim %+v", res.Samples[0], legacy)
	}
	if v := ClassifyCompaction(res.Samples[0]); v.Reason != ReasonCompactionNetCheaper {
		t.Errorf("legacy verdict = %q, want %q — the pre-join path must be unchanged", v.Reason, ReasonCompactionNetCheaper)
	}
}

// TestCompactionJoinKeyIsZero pins the unstamped sentinel: only the all-zero key is unstamped, so
// a genuine turn 0 with a monotonic reading (or vice versa) still joins.
func TestCompactionJoinKeyIsZero(t *testing.T) {
	if !(CompactionJoinKey{}).IsZero() {
		t.Errorf("zero key reported stamped")
	}
	for _, k := range []CompactionJoinKey{{TurnSeq: 1}, {MonotonicTSNano: 1}, {TurnSeq: 1, MonotonicTSNano: 1}} {
		if k.IsZero() {
			t.Errorf("key %+v reported unstamped, want stamped", k)
		}
	}
}

// TestJoinProviderUsageDoesNotMutateInputs: the join is pure — a caller may fold the same sample
// slice twice, or hold it for the WITNESSED shed, without the first pass having rewritten it.
func TestJoinProviderUsageDoesNotMutateInputs(t *testing.T) {
	k := CompactionJoinKey{TurnSeq: 2, MonotonicTSNano: 20}
	samples := []CompactionSample{{ShedTokens: 100, JoinKey: k}}
	before := samples[0]
	JoinProviderUsage(samples, []ProviderTurnUsage{{Key: k, CacheReadTokens: 1, CacheCreationTokens: 2}})
	if samples[0] != before {
		t.Errorf("input sample mutated to %+v, want %+v untouched", samples[0], before)
	}
}
