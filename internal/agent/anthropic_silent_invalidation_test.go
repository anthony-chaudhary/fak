package agent

import "testing"

// TestSilentCacheInvalidation pins the #2791 signal: a fire whose protected prefix was proven
// byte-identical, yet whose provider re-created that prefix instead of reading it. The table is
// organised around the two ways the check could be WRONG rather than the one way it is right —
// a fire that legitimately bursts (#2785) must not be flagged, and an identity return carries no
// byte-equality witness to diverge from in the first place.
func TestSilentCacheInvalidation(t *testing.T) {
	fired := CompactOutcome{Reason: CompactReasonNone, Dropped: 3, ShedTokens: 900}

	tests := []struct {
		name string
		out  CompactOutcome
		u    Usage
		want bool
	}{
		{
			// The witness. bytes.Equal held (the fire proves it), but the provider read
			// nothing and re-ingested the prompt: TTL expiry / capacity eviction.
			name: "fired, prefix preserved, provider re-created it",
			out:  fired,
			u:    Usage{CacheReadInputTokens: 0, CacheCreationInputTokens: 40_000},
			want: true,
		},
		{
			// Confusion risk #1: the head-anchored fire BURSTS the recent breakpoint's cached
			// suffix on purpose (#1408/#2785). Creation is expected there — but the protected
			// head is still read, so a positive read disqualifies it. Not our signal.
			name: "fired with induced burst (#2785) — head still read, not silent",
			out:  fired,
			u:    Usage{CacheReadInputTokens: 30_000, CacheCreationInputTokens: 12_000},
			want: false,
		},
		{
			// The healthy warm fire: prefix preserved AND read back. Nothing to report.
			name: "fired and prefix read warm",
			out:  fired,
			u:    Usage{CacheReadInputTokens: 40_000, CacheCreationInputTokens: 0},
			want: false,
		},
		{
			// A cold turn with no creation is not an invalidation — there is nothing
			// evidencing the provider re-ingested the preserved prefix.
			name: "fired, no read and no creation",
			out:  fired,
			u:    Usage{CacheReadInputTokens: 0, CacheCreationInputTokens: 0},
			want: false,
		},
		{
			// Confusion risk #2: the prefix-mismatch bail is the case where bytes.Equal
			// FAILED, so fak shipped identity. Creation here is a known cache burst, not a
			// silent one — flagging it would invert the signal's meaning.
			name: "prefix_mismatch bail — bytes.Equal did not hold",
			out:  CompactOutcome{Reason: CompactReasonPrefixMismatch},
			u:    Usage{CacheReadInputTokens: 0, CacheCreationInputTokens: 40_000},
			want: false,
		},
		{
			name: "under_budget bail — no splice shipped",
			out:  CompactOutcome{Reason: CompactReasonUnderBudget, ProtectedPrefixTokens: 50_000},
			u:    Usage{CacheReadInputTokens: 0, CacheCreationInputTokens: 40_000},
			want: false,
		},
		{
			name: "burst_unprofitable bail — the fire never happened",
			out:  CompactOutcome{Reason: CompactReasonBurstUnprofitable},
			u:    Usage{CacheReadInputTokens: 0, CacheCreationInputTokens: 40_000},
			want: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := SilentCacheInvalidation(tc.out, tc.u); got != tc.want {
				t.Errorf("SilentCacheInvalidation(%+v, read=%d creation=%d) = %v, want %v",
					tc.out.Reason, tc.u.CacheReadInputTokens, tc.u.CacheCreationInputTokens, got, tc.want)
			}
		})
	}
}

// TestSilentCacheInvalidationFiresOnlyWithBothHalves proves the two clauses are conjunctive:
// neither a zero read alone nor a nonzero creation alone is sufficient. This is the guard against
// a later "simplification" collapsing the check into a bare creation>0 test, which would silently
// re-absorb the #2785 induced burst into this counter.
func TestSilentCacheInvalidationFiresOnlyWithBothHalves(t *testing.T) {
	fired := CompactOutcome{Reason: CompactReasonNone}

	if SilentCacheInvalidation(fired, Usage{CacheReadInputTokens: 0, CacheCreationInputTokens: 0}) {
		t.Error("zero read alone must not flag: no creation evidences no re-ingest")
	}
	if SilentCacheInvalidation(fired, Usage{CacheReadInputTokens: 1, CacheCreationInputTokens: 99}) {
		t.Error("creation alone must not flag: a positive read proves the prefix was served")
	}
	if !SilentCacheInvalidation(fired, Usage{CacheReadInputTokens: 0, CacheCreationInputTokens: 1}) {
		t.Error("both halves present must flag")
	}
}
