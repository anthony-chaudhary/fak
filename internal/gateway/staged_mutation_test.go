package gateway

import "testing"

// staged_mutation_test.go — the #2849 witness. It pins the two halves of the
// cache-aware state-mutation contract the primitive exists to offer once:
//
//  1. DEFERRED BY DEFAULT: a stable-prefix change goes through Mutate with no
//     `--now`, is queued (not applied), and induces no cache-rebuild cost. It
//     applies only at the session boundary, where the prefix cold-starts anyway
//     — so deferring is free.
//  2. `--now` CHARGES: the same change through Mutate with now=true applies
//     immediately and reports the measured cache_creation, priced by C1
//     (CachePricing) — a witnessed cost, byte-identical to the cache-price model
//     every other cache_creation token is valued with.

// stagedMutPricing is the C1 model the `--now` rebuild is charged against — the
// guarded Claude Opus 4.8 default the gateway resolves for the Claude path.
func stagedMutPricing(t *testing.T) CachePricing {
	t.Helper()
	p, _, ok := DefaultCachePricing("anthropic", "claude")
	if !ok {
		t.Fatalf("DefaultCachePricing(anthropic, claude) = _, _, false; want the Opus 4.8 default")
	}
	return p
}

// TestStagedMutationDeferredByDefault is witness half 1: a mutation staged with
// now=false is queued, not applied, and induces no cost — and applies only when
// the session boundary drains the queue.
func TestStagedMutationDeferredByDefault(t *testing.T) {
	q := NewStagedMutationQueue(stagedMutPricing(t))

	res := q.Mutate(StagedMutation{Name: "skills install", NewPrefixTokens: 20_000, WriteTTL: CacheTTL5m}, false)

	if res.Timing != ApplyDeferred {
		t.Errorf("deferred mutation timing = %v; want ApplyDeferred", res.Timing)
	}
	if res.Applied {
		t.Error("deferred mutation reports Applied=true; a queued change is not yet live")
	}
	if res.CacheCreationTokens != 0 {
		t.Errorf("deferred mutation induced %d cache_creation tokens; want 0 (the running cache is preserved)", res.CacheCreationTokens)
	}
	if res.RebuildCostUSD != 0 {
		t.Errorf("deferred mutation charged $%g; want $0 (deferring to the boundary is free)", res.RebuildCostUSD)
	}
	if got := q.PendingCount(); got != 1 {
		t.Fatalf("PendingCount after one deferred stage = %d; want 1", got)
	}

	// The session boundary applies the queued change with no induced cost.
	applied := q.ApplyAtBoundary()
	if len(applied) != 1 {
		t.Fatalf("ApplyAtBoundary drained %d mutations; want 1", len(applied))
	}
	if !applied[0].Applied {
		t.Error("mutation applied at the boundary reports Applied=false; it should be live now")
	}
	if applied[0].CacheCreationTokens != 0 || applied[0].RebuildCostUSD != 0 {
		t.Errorf("boundary-applied mutation induced cost (%d tok / $%g); want 0 — the new session cold-starts its prefix regardless",
			applied[0].CacheCreationTokens, applied[0].RebuildCostUSD)
	}
	if got := q.PendingCount(); got != 0 {
		t.Errorf("PendingCount after the boundary drained the queue = %d; want 0", got)
	}
}

// TestStagedMutationNowChargesRebuild is witness half 2: the `--now` path applies
// immediately, reports the induced cache_creation, and charges the measured
// rebuild cost — which must equal C1's CostUSD for that cache write exactly.
func TestStagedMutationNowChargesRebuild(t *testing.T) {
	pricing := stagedMutPricing(t)
	q := NewStagedMutationQueue(pricing)

	const prefixTokens = 20_000
	res := q.Mutate(StagedMutation{Name: "skills install", NewPrefixTokens: prefixTokens, WriteTTL: CacheTTL5m}, true)

	if res.Timing != ApplyNow {
		t.Errorf("--now mutation timing = %v; want ApplyNow", res.Timing)
	}
	if !res.Applied {
		t.Error("--now mutation reports Applied=false; it takes effect immediately")
	}
	if res.CacheCreationTokens != prefixTokens {
		t.Errorf("--now induced %d cache_creation tokens; want %d (the whole new prefix is re-written)", res.CacheCreationTokens, prefixTokens)
	}

	// The charged cost must BE the C1 price of that cache write — no more, no less.
	want := pricing.CostUSD(CacheUsage{CacheCreationTokens: prefixTokens, WriteTTL: CacheTTL5m})
	if res.RebuildCostUSD != want {
		t.Errorf("--now RebuildCostUSD = $%g; want C1 CostUSD $%g", res.RebuildCostUSD, want)
	}
	if want <= 0 {
		t.Fatalf("expected a positive measured rebuild cost; C1 CostUSD = $%g", want)
	}

	// A `--now` mutation is applied, never queued.
	if got := q.PendingCount(); got != 0 {
		t.Errorf("PendingCount after a --now mutation = %d; want 0 (it did not defer)", got)
	}

	// The 1h tier costs strictly more to rebuild than the 5m tier (2.0x vs 1.25x
	// write premium) — the contract prices the tier the harness actually chose.
	res1h := q.Mutate(StagedMutation{Name: "skills install", NewPrefixTokens: prefixTokens, WriteTTL: CacheTTL1h}, true)
	if !(res1h.RebuildCostUSD > res.RebuildCostUSD) {
		t.Errorf("1h rebuild $%g not greater than 5m rebuild $%g; the 1h write premium should cost more", res1h.RebuildCostUSD, res.RebuildCostUSD)
	}
}

// TestStagedMutationHarnessAdapter pins the external-harness opt-in: a harness
// that implements HarnessMutation routes through the SAME contract — deferred by
// default, charged on `--now` — without re-deriving the discipline.
func TestStagedMutationHarnessAdapter(t *testing.T) {
	pricing := stagedMutPricing(t)
	q := NewStagedMutationQueue(pricing)
	h := fakeHarnessMutation{name: "toggle tool", tokens: 12_000, tier: CacheTTL5m}

	deferred := StageHarnessMutation(q, h, false)
	if deferred.Timing != ApplyDeferred || deferred.Applied {
		t.Errorf("harness mutation without --now = %+v; want deferred, not applied", deferred)
	}
	if q.PendingCount() != 1 {
		t.Fatalf("harness deferred mutation did not queue; PendingCount = %d", q.PendingCount())
	}

	now := StageHarnessMutation(q, h, true)
	if now.Timing != ApplyNow || !now.Applied {
		t.Errorf("harness mutation with --now = %+v; want applied now", now)
	}
	want := pricing.CostUSD(CacheUsage{CacheCreationTokens: h.tokens, WriteTTL: h.tier})
	if now.RebuildCostUSD != want {
		t.Errorf("harness --now RebuildCostUSD = $%g; want C1 CostUSD $%g", now.RebuildCostUSD, want)
	}
}

// TestStagedMutationNowNegativeTokensClamped guards the primitive against a
// mis-measuring harness: a negative prefix size is clamped to 0, never a
// negative charge.
func TestStagedMutationNowNegativeTokensClamped(t *testing.T) {
	q := NewStagedMutationQueue(stagedMutPricing(t))
	res := q.Mutate(StagedMutation{Name: "reload memory", NewPrefixTokens: -5, WriteTTL: CacheTTL5m}, true)
	if res.CacheCreationTokens != 0 {
		t.Errorf("negative prefix tokens induced %d; want 0 (clamped)", res.CacheCreationTokens)
	}
	if res.RebuildCostUSD != 0 {
		t.Errorf("negative prefix tokens charged $%g; want $0", res.RebuildCostUSD)
	}
}

// fakeHarnessMutation is a minimal HarnessMutation for the adapter witness.
type fakeHarnessMutation struct {
	name   string
	tokens int
	tier   CacheTTL
}

func (f fakeHarnessMutation) MutationName() string      { return f.name }
func (f fakeHarnessMutation) StablePrefixTokens() int   { return f.tokens }
func (f fakeHarnessMutation) CacheTier() CacheTTL       { return f.tier }
