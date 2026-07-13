package model

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestExpertResidencyLFUDecayBeatsLRUOnRoutingJitter is the promotion witness for #4357:
// on a stable hot set sprayed with cold one-off experts (the routing jitter the axis
// targets), the value-aware ring (hysteresis + LFU-decay) reaches the offline Belady
// optimum while pagedRing's LRU collapses to near-pure thrashing. Every cold one-off is
// bypassed instead of evicting a hot resident, so the hot set is paged in exactly once.
func TestExpertResidencyLFUDecayBeatsLRUOnRoutingJitter(t *testing.T) {
	const hot, cold = 2, 6
	weight := int64(q4kBlockBytes)
	trace := GenerateHotSetJitterTrace(hot, cold, weight, weight*int64(hot))

	report, err := ReplayExpertResidencyLFUDecay(trace, ExpertResidencyLFUOptions{})
	if err != nil {
		t.Fatalf("ReplayExpertResidencyLFUDecay: %v", err)
	}
	if !report.Oracle.Exact {
		t.Fatal("8-span jitter fixture unexpectedly used the approximate Belady fallback")
	}
	// The value-aware policy reaches the exact offline optimum: it keeps {hot} resident and
	// never re-pages, so its hit bytes equal Belady's upper bound.
	if report.LFUDecay.HitTokens != report.Oracle.HitTokens {
		t.Fatalf("LFU-decay hit bytes=%d, want Belady optimum %d", report.LFUDecay.HitTokens, report.Oracle.HitTokens)
	}
	// It strictly beats LRU on both axes the issue names: more hits, fewer page-in evictions.
	if report.HitDelta <= 0 {
		t.Fatalf("HitDelta=%d, want value-aware strictly above LRU (%d vs %d)", report.HitDelta, report.LFUDecay.HitTokens, report.LRU.HitTokens)
	}
	if report.EvictionDelta <= 0 {
		t.Fatalf("EvictionDelta=%d, want value-aware to thrash less than LRU (%d vs %d)", report.EvictionDelta, report.LFUDecay.Evictions, report.LRU.Evictions)
	}
	// Hysteresis actually engaged: every cold one-off was declined, and the hot set paged in
	// exactly once with zero evictions.
	if report.LFUDecay.Bypasses != cold {
		t.Fatalf("bypasses=%d, want one per cold one-off (%d)", report.LFUDecay.Bypasses, cold)
	}
	if report.LFUDecay.Evictions != 0 || report.LFUDecay.PageIns != hot {
		t.Fatalf("LFU-decay evictions/pageIns=%d/%d, want 0/%d", report.LFUDecay.Evictions, report.LFUDecay.PageIns, hot)
	}
	if report.LFUDecay.GoodDecisionRatio <= report.LRU.GoodDecisionRatio {
		t.Fatalf("LFU-decay good-decision ratio=%.4f, want > LRU %.4f", report.LFUDecay.GoodDecisionRatio, report.LRU.GoodDecisionRatio)
	}
	t.Logf("routing-jitter witness: LFU-decay hits=%d evict=%d bypass=%d gdr=%.3f | LRU hits=%d evict=%d gdr=%.3f | oracle=%d",
		report.LFUDecay.HitTokens, report.LFUDecay.Evictions, report.LFUDecay.Bypasses, report.LFUDecay.GoodDecisionRatio,
		report.LRU.HitTokens, report.LRU.Evictions, report.LRU.GoodDecisionRatio, report.Oracle.HitTokens)
}

// TestExpertResidencyLFUDecayReducesToLRUWhenWorkingSetFits is the compatibility evidence:
// when the whole working set fits in budget there is nothing to protect, so the value-aware
// policy pages each expert in once and then hits forever — byte-identical hit/eviction
// behaviour to LRU, and it never fires a bypass. The new policy adds no regression on the
// workload LRU already serves optimally.
func TestExpertResidencyLFUDecayReducesToLRUWhenWorkingSetFits(t *testing.T) {
	w := int64(q4kBlockBytes)
	trace := ExpertAccessTrace{
		Schema: ExpertReplayTraceSchema, Name: "fits-in-budget", Source: "deterministic-unit",
		BudgetBytes: 2 * w,
		Events: []ExpertAccessTraceEvent{
			{Layer: 0, Expert: 0, WeightBytes: w}, {Layer: 0, Expert: 1, WeightBytes: w},
			{Layer: 0, Expert: 0, WeightBytes: w}, {Layer: 0, Expert: 1, WeightBytes: w},
			{Layer: 0, Expert: 0, WeightBytes: w}, {Layer: 0, Expert: 1, WeightBytes: w},
		},
	}
	report, err := ReplayExpertResidencyLFUDecay(trace, ExpertResidencyLFUOptions{})
	if err != nil {
		t.Fatalf("ReplayExpertResidencyLFUDecay: %v", err)
	}
	if report.LFUDecay.HitTokens != report.LRU.HitTokens {
		t.Fatalf("LFU-decay hits=%d, want LRU-identical %d", report.LFUDecay.HitTokens, report.LRU.HitTokens)
	}
	if report.LFUDecay.Evictions != 0 || report.LRU.Evictions != 0 {
		t.Fatalf("evictions LFU/LRU=%d/%d, want 0/0 when the working set fits", report.LFUDecay.Evictions, report.LRU.Evictions)
	}
	if report.LFUDecay.Bypasses != 0 {
		t.Fatalf("bypasses=%d, want 0 when no eviction is ever forced", report.LFUDecay.Bypasses)
	}
}

// TestExpertResidencyLFUDecayNoBenefitWithoutReuse is the demotion criterion / invalidating
// assumption: the value-aware policy's whole premise is a reused hot set to protect. On a
// trace with NO reuse (every expert touched once), there is no hot set, so it yields exactly
// zero hit-rate improvement over LRU — HitDelta==0. This is the boundary where the policy
// should be retired rather than promoted: its bypass cost buys nothing on a workload without
// frequency-skewed reuse.
func TestExpertResidencyLFUDecayNoBenefitWithoutReuse(t *testing.T) {
	w := int64(q4kBlockBytes)
	events := make([]ExpertAccessTraceEvent, 0, 6)
	for e := 0; e < 6; e++ {
		events = append(events, ExpertAccessTraceEvent{Layer: 0, Expert: e, WeightBytes: w})
	}
	trace := ExpertAccessTrace{
		Schema: ExpertReplayTraceSchema, Name: "no-reuse", Source: "deterministic-unit",
		BudgetBytes: 2 * w, Events: events,
	}
	report, err := ReplayExpertResidencyLFUDecay(trace, ExpertResidencyLFUOptions{})
	if err != nil {
		t.Fatalf("ReplayExpertResidencyLFUDecay: %v", err)
	}
	if report.LFUDecay.HitTokens != 0 || report.LRU.HitTokens != 0 {
		t.Fatalf("hits LFU/LRU=%d/%d, want 0/0 on a no-reuse trace", report.LFUDecay.HitTokens, report.LRU.HitTokens)
	}
	if report.HitDelta != 0 {
		t.Fatalf("HitDelta=%d, want 0: the value-aware policy earns no hit advantage without reuse", report.HitDelta)
	}
}

// TestExpertResidencyLFUDecayDeterministic guards against Go's randomized map iteration
// leaking into the victim order: identical inputs must yield an identical report.
func TestExpertResidencyLFUDecayDeterministic(t *testing.T) {
	trace := GenerateHotSetJitterTrace(3, 5, int64(q4kBlockBytes), 0)
	first, err := ReplayExpertResidencyLFUDecay(trace, ExpertResidencyLFUOptions{DecayEveryAccesses: 8})
	if err != nil {
		t.Fatalf("ReplayExpertResidencyLFUDecay (first): %v", err)
	}
	second, err := ReplayExpertResidencyLFUDecay(trace, ExpertResidencyLFUOptions{DecayEveryAccesses: 8})
	if err != nil {
		t.Fatalf("ReplayExpertResidencyLFUDecay (second): %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatal("value-aware residency report changed for identical inputs")
	}
}

// TestExpertResidencyLFUDecayRejectsInvalidTrace confirms the simulation reuses the same
// trace validation the LRU harness does, so a malformed corpus cannot silently score.
func TestExpertResidencyLFUDecayRejectsInvalidTrace(t *testing.T) {
	if _, err := ReplayExpertResidencyLFUDecay(ExpertAccessTrace{Schema: "wrong"}, ExpertResidencyLFUOptions{}); err == nil {
		t.Fatal("replay accepted a trace with the wrong schema")
	}
	// A caller can also confirm the LRU baseline it is scored against is the canonical policy.
	if compute.KVEvictLRU == compute.KVEvictCostAware {
		t.Fatal("KV evict policy identities collapsed")
	}
}
