package model

// v41_prefill_expert_union_test.go -- the #13294 frontier witness for V4.1
// prefill routed-expert read attribution.
//
// DoD item 1 of fak#13294 requires that "the prefill first-token path's
// streamed-expert fault cost is attributed (resident hit fraction, faults per
// token)". The attribution ledger (v41ExpertFaultAttribution) computes
//
//	ResidentHitFraction = residentHits / (residentHits + faults)
//
// to answer "what share of routed-expert reads this phase served from
// residency". The #13296 layer-scoped f32 cache exists precisely to raise that
// share, but its hit path (v41ExpertTripleInto) returns the retained block and
// `continue`s BEFORE reaching v41ExpertF32Into, which is the ONLY site that
// increments ResidentHits. So a cache-served read is counted as NEITHER a
// resident hit NOR a fault: the denominator silently drops every cache hit and
// the fraction UNDERSTATES residency by exactly the cache's contribution.
//
// On the physical fed6a6a37 rung (0.1 tok/s prefill, 30 tok / 492.02 s) the
// layer cache was the one mechanism already turning repeated routed-expert
// reads into RAM hits, yet the ledger could not see any of them -- so the very
// number the issue's first DoD item asks for misdirects the frontier. This
// witness pins the corrected accounting: every routed-expert read resolved from
// the layer cache lands in ResidentHits, so the fraction reflects the residency
// the #13296 cache actually provides.

import (
	"math"
	"testing"
)

// v41PrefillAttributionModel builds the reduced tier-only model whose routed
// experts live ONLY in the checkpoint tier, so every read the layer cache does
// not serve is a real tier fault.
func v41PrefillAttributionModel(t *testing.T) *Model {
	t.Helper()
	m, _ := v41TierOnlyBudgetModel(t, 0)
	return m
}

// TestV41PrefillCacheHitsCountAsResidentHits is the #13294 DoD item-1 witness.
//
// A multi-token prefill of identical tokens re-reads the same distinct routed
// expert across the token dimension. The #13296 layer cache serves those repeats
// from RAM, so the reads that do NOT reach the tier are residency-served and must
// be counted as ResidentHits. The forward issues `tokens * topK * 3` routed-expert
// projection reads in total; each is either a tier fault or a residency hit, so
// the ledger's two counters must SUM to that total. When cache hits are dropped
// the sum equals only the tier faults and the fraction is wrong.
func TestV41PrefillCacheHitsCountAsResidentHits(t *testing.T) {
	m := v41PrefillAttributionModel(t)
	const tokens = 8

	m.v41SetExpertFaultPhase(V41PhasePrefill)
	ids := make([]int, tokens)
	for i := range ids {
		ids[i] = 1
	}
	if _, err := m.forwardV41(ids, nil); err != nil {
		t.Fatalf("%d-token prefill: %v", tokens, err)
	}
	m.v41SetExpertFaultPhase(V41PhaseUnknown)

	att := m.V41ExpertFaultAttribution().Prefill
	if att.Faults == 0 {
		t.Fatal("fixture faulted no routed projections; the attribution under test would be vacuous")
	}

	// Each token routes topK experts and each pick reads 3 projections. A correct
	// ledger accounts for EVERY read as a fault or a residency hit; a dropped
	// cache hit leaves the sum short of the total.
	wantReads := tokens * V41RouterTopK * 3
	gotReads := att.ResidentHits + att.Faults
	if gotReads != wantReads {
		t.Fatalf("attribution accounts for %d routed-expert reads (residentHits=%d + faults=%d), want %d: "+
			"layer-cache hits are dropped from the ledger", gotReads, att.ResidentHits, att.Faults, wantReads)
	}
	if att.ResidentHits == 0 {
		t.Fatal("a prefill that re-reads already-cached routed experts recorded zero resident hits")
	}
	if f := att.ResidentHitFraction; !(f > 0 && f < 1) || math.IsNaN(f) {
		t.Fatalf("resident hit fraction %v is outside (0,1) for a mixed fault/hit prefill", f)
	}
	// The fraction must equal the ledger's own ratio, never a stale/other value.
	if want := float64(att.ResidentHits) / float64(att.ResidentHits+att.Faults); math.Abs(att.ResidentHitFraction-want) > 1e-9 {
		t.Fatalf("resident hit fraction %v != residentHits/(residentHits+faults) = %v", att.ResidentHitFraction, want)
	}
}

// TestV41PrefillCacheMissStillCountsAsFault pins the complement: a single-position
// prefill cannot repeat a projection, so every read reaches the tier and none may
// be reported as a cache hit. The fix therefore cannot double-count.
func TestV41PrefillCacheMissStillCountsAsFault(t *testing.T) {
	m := v41PrefillAttributionModel(t)

	m.v41SetExpertFaultPhase(V41PhasePrefill)
	if _, err := m.forwardV41([]int{1}, nil); err != nil {
		t.Fatalf("1-token prefill: %v", err)
	}
	m.v41SetExpertFaultPhase(V41PhaseUnknown)

	att := m.V41ExpertFaultAttribution().Prefill
	if att.Faults != V41RouterTopK*3 {
		t.Fatalf("1-token prefill recorded %d faults, want %d (one position's distinct set)",
			att.Faults, V41RouterTopK*3)
	}
	if att.ResidentHits != 0 {
		t.Fatalf("a single-position prefill recorded %d resident hits; a first read cannot be a cache hit",
			att.ResidentHits)
	}
}
