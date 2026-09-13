package deepseekv4moe

import (
	"errors"
	"reflect"
	"testing"
)

// TestWitnessExpertCacheHitRateColdAndWarm pins the cold and warm receipt for the
// V4-shaped fixture from cache_trace_test.go: 12 groups exactly afford the two
// disjoint top-6 routes cold, and replay them as hits warm.
func TestWitnessExpertCacheHitRateColdAndWarm(t *testing.T) {
	routes := []ExpertRoute{
		{Layer: 0, Experts: []int{0, 1, 2, 3, 4, 5}},
		{Layer: 60, Experts: []int{378, 379, 380, 381, 382, 383}},
	}
	trace := ExpertActivationTrace{Layers: 61, Experts: 384, TopK: 6, GroupBytes: 1, BudgetBytes: 12, Routes: routes}

	cold, err := WitnessExpertCacheHitRate(trace)
	if err != nil {
		t.Fatal(err)
	}
	if cold.Schema != ExpertCacheHitRateWitnessSchema || cold.Label != "SW-VERIFIED" || cold.Policy != "lru" {
		t.Fatalf("cold identity = schema %q label %q policy %q", cold.Schema, cold.Label, cold.Policy)
	}
	if cold.CacheGroups != 12 || cold.CacheBytes != 12 || cold.GroupBytes != 1 {
		t.Fatalf("cold cache = groups %d bytes %d group_bytes %d, want 12/12/1", cold.CacheGroups, cold.CacheBytes, cold.GroupBytes)
	}
	if cold.Hits != 0 || cold.Misses != 12 || cold.Accesses != 12 || cold.PeakResident != 12 {
		t.Fatalf("cold counts = hits %d misses %d accesses %d peak %d, want 0/12/12/12",
			cold.Hits, cold.Misses, cold.Accesses, cold.PeakResident)
	}
	if !cold.HitRateKnown || cold.HitRate != 0 {
		t.Fatalf("cold hit rate = (%v, known=%v), want (0, true)", cold.HitRate, cold.HitRateKnown)
	}
	if !cold.ByteWeightedHitKnown || cold.ByteWeightedHitRate != 0 {
		t.Fatalf("cold byte rate = (%v, known=%v), want (0, true)", cold.ByteWeightedHitRate, cold.ByteWeightedHitKnown)
	}
	if cold.BytesReadResident != 0 || cold.BytesReadStreamed != 12 {
		t.Fatalf("cold bytes = resident %d streamed %d, want 0/12", cold.BytesReadResident, cold.BytesReadStreamed)
	}
	if !cold.BeladyExact || cold.BeladyOptimalMisses != 12 || cold.BeladyOptimalHits != 0 || cold.BeladyRegretHits != 0 {
		t.Fatalf("cold belady = exact=%v optimal %d/%d regret %d, want true 0/12/0",
			cold.BeladyExact, cold.BeladyOptimalHits, cold.BeladyOptimalMisses, cold.BeladyRegretHits)
	}

	warmTrace := trace
	warmTrace.Routes = append(append([]ExpertRoute{}, routes...), routes...)
	warm, err := WitnessExpertCacheHitRate(warmTrace)
	if err != nil {
		t.Fatal(err)
	}
	if warm.Hits != 12 || warm.Misses != 12 || warm.Accesses != 24 {
		t.Fatalf("warm counts = hits %d misses %d accesses %d, want 12/12/24", warm.Hits, warm.Misses, warm.Accesses)
	}
	if !warm.HitRateKnown || warm.HitRate != 0.5 {
		t.Fatalf("warm hit rate = (%v, known=%v), want (0.5, true)", warm.HitRate, warm.HitRateKnown)
	}
	if warm.BytesReadResident != 12 || warm.BytesReadStreamed != 12 {
		t.Fatalf("warm bytes = resident %d streamed %d, want 12/12", warm.BytesReadResident, warm.BytesReadStreamed)
	}
	if !warm.ByteWeightedHitKnown || warm.ByteWeightedHitRate != 0.5 {
		t.Fatalf("warm byte rate = (%v, known=%v), want (0.5, true)", warm.ByteWeightedHitRate, warm.ByteWeightedHitKnown)
	}
	if !warm.BeladyExact || warm.BeladyOptimalHits != 12 || warm.BeladyRegretHits != 0 {
		t.Fatalf("warm belady = exact=%v optimal %d regret %d, want true 12/0",
			warm.BeladyExact, warm.BeladyOptimalHits, warm.BeladyRegretHits)
	}
}

// TestWitnessExpertCacheHitRateBeladyRegretNonNegative proves the LRU-vs-Belady
// regret is a real, non-negative gap on a thrashing trace: two layers each holding
// 6 groups cycle through a 6-group cache, so every touch is a cold page-in.
func TestWitnessExpertCacheHitRateBeladyRegretNonNegative(t *testing.T) {
	trace := ExpertActivationTrace{
		Layers: 2, Experts: 384, TopK: 6, GroupBytes: 1, BudgetBytes: 6,
		Routes: []ExpertRoute{
			{Layer: 0, Experts: []int{0, 1, 2, 3, 4, 5}},
			{Layer: 1, Experts: []int{0, 1, 2, 3, 4, 5}},
			{Layer: 0, Experts: []int{0, 1, 2, 3, 4, 5}},
		},
	}
	w, err := WitnessExpertCacheHitRate(trace)
	if err != nil {
		t.Fatal(err)
	}
	if w.Hits != 0 || w.Misses != 18 || w.Accesses != 18 || w.PeakResident != 6 {
		t.Fatalf("trace = hits %d misses %d accesses %d peak %d, want 0/18/18/6",
			w.Hits, w.Misses, w.Accesses, w.PeakResident)
	}
	if !w.BeladyExact {
		t.Fatalf("belady exact = false, want true for an exact 12-span oracle")
	}
	if w.BeladyRegretHits < 0 {
		t.Fatalf("belady regret = %d, must be non-negative", w.BeladyRegretHits)
	}
	if w.BeladyOptimalHits < w.Hits {
		t.Fatalf("belady optimal %d below LRU hits %d - oracle is not an upper bound", w.BeladyOptimalHits, w.Hits)
	}
	if w.BeladyRegretHits != w.BeladyOptimalHits-w.Hits {
		t.Fatalf("belady regret = %d, want optimal %d - hits %d = %d",
			w.BeladyRegretHits, w.BeladyOptimalHits, w.Hits, w.BeladyOptimalHits-w.Hits)
	}
	if w.BeladyOptimalHits+w.BeladyOptimalMisses != w.Accesses {
		t.Fatalf("belady optimal = %d hits + %d misses, want accesses %d",
			w.BeladyOptimalHits, w.BeladyOptimalMisses, w.Accesses)
	}
}

// TestWitnessExpertCacheHitRateBudgetCapAndMinimum covers the two budget edges:
// a budget larger than the model clamps to Layers*Experts, and a budget below one
// whole group refuses with the typed budget error.
func TestWitnessExpertCacheHitRateBudgetCapAndMinimum(t *testing.T) {
	t.Run("cap at model size", func(t *testing.T) {
		trace := ExpertActivationTrace{
			Layers: 2, Experts: 4, TopK: 2, GroupBytes: 1, BudgetBytes: 100,
			Routes: []ExpertRoute{{Layer: 0, Experts: []int{0, 1}}},
		}
		w, err := WitnessExpertCacheHitRate(trace)
		if err != nil {
			t.Fatal(err)
		}
		if w.CacheGroups != 8 || w.CacheBytes != 8 {
			t.Fatalf("cache = groups %d bytes %d, want capped at 8/8", w.CacheGroups, w.CacheBytes)
		}
	})
	t.Run("budget below one group", func(t *testing.T) {
		trace := ExpertActivationTrace{
			Layers: 2, Experts: 4, TopK: 2, GroupBytes: 10, BudgetBytes: 1,
			Routes: []ExpertRoute{{Layer: 0, Experts: []int{0, 1}}},
		}
		w, err := WitnessExpertCacheHitRate(trace)
		if !errors.Is(err, ErrInvalidHitRateWitnessBudget) {
			t.Fatalf("error = %v, want %v", err, ErrInvalidHitRateWitnessBudget)
		}
		if w != (ExpertCacheHitRateWitness{}) {
			t.Fatalf("witness = %+v, want zero value on error", w)
		}
	})
}

// TestWitnessExpertCacheHitRateFailsClosed is the fail-closed table: every
// degenerate trace is a typed error and never a partially fabricated rate.
func TestWitnessExpertCacheHitRateFailsClosed(t *testing.T) {
	tests := []struct {
		name  string
		trace ExpertActivationTrace
		want  error
	}{
		{"no routes", ExpertActivationTrace{Layers: 2, Experts: 4, TopK: 2, GroupBytes: 1, BudgetBytes: 4}, ErrNoTrace},
		{"bad shape all zero", ExpertActivationTrace{Layers: 0, Experts: 0, TopK: 0, GroupBytes: 1, BudgetBytes: 4, Routes: []ExpertRoute{{Layer: 0, Experts: []int{0}}}}, ErrInvalidTraceShape},
		{"top-k over experts", ExpertActivationTrace{Layers: 2, Experts: 2, TopK: 3, GroupBytes: 1, BudgetBytes: 4, Routes: []ExpertRoute{{Layer: 0, Experts: []int{0, 1}}}}, ErrInvalidTraceShape},
		{"zero group bytes", ExpertActivationTrace{Layers: 2, Experts: 4, TopK: 2, GroupBytes: 0, BudgetBytes: 4, Routes: []ExpertRoute{{Layer: 0, Experts: []int{0, 1}}}}, ErrInvalidTraceShape},
		{"zero budget", ExpertActivationTrace{Layers: 2, Experts: 4, TopK: 2, GroupBytes: 1, BudgetBytes: 0, Routes: []ExpertRoute{{Layer: 0, Experts: []int{0, 1}}}}, ErrInvalidTraceCapacity},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			w, err := WitnessExpertCacheHitRate(tc.trace)
			if !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want %v", err, tc.want)
			}
			if w != (ExpertCacheHitRateWitness{}) {
				t.Fatalf("witness = %+v, want zero value on error", w)
			}
		})
	}
}

// TestWitnessExpertCacheHitRateDeterministicReplay proves the witness is a pure
// function of the trace: identical input yields a byte-identical receipt.
func TestWitnessExpertCacheHitRateDeterministicReplay(t *testing.T) {
	trace := ExpertActivationTrace{
		Layers: 61, Experts: 384, TopK: 6, GroupBytes: 1, BudgetBytes: 12,
		Routes: []ExpertRoute{
			{Layer: 0, Experts: []int{0, 1, 2, 3, 4, 5}},
			{Layer: 60, Experts: []int{378, 379, 380, 381, 382, 383}},
		},
	}
	first, err := WitnessExpertCacheHitRate(trace)
	if err != nil {
		t.Fatal(err)
	}
	second, err := WitnessExpertCacheHitRate(trace)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("replay mismatch:\nfirst  %+v\nsecond %+v", first, second)
	}
}

// TestWitnessExpertCacheHitRateMonotoneInBudget proves a larger cache cannot
// reduce the hit rate on a fixed hot-set trace. The relation is asserted, not a
// frozen magic number: 4 distinct groups over a 2-group cache thrash, and the
// same trace over a 4-group cache is fully resident after the first pass.
func TestWitnessExpertCacheHitRateMonotoneInBudget(t *testing.T) {
	routes := []ExpertRoute{
		{Layer: 0, Experts: []int{0, 1}},
		{Layer: 0, Experts: []int{2, 3}},
		{Layer: 0, Experts: []int{0, 1}},
		{Layer: 0, Experts: []int{2, 3}},
	}
	small := ExpertActivationTrace{Layers: 1, Experts: 4, TopK: 2, GroupBytes: 1, BudgetBytes: 2, Routes: routes}
	large := small
	large.BudgetBytes = 4

	smallW, err := WitnessExpertCacheHitRate(small)
	if err != nil {
		t.Fatal(err)
	}
	largeW, err := WitnessExpertCacheHitRate(large)
	if err != nil {
		t.Fatal(err)
	}
	if largeW.CacheGroups <= smallW.CacheGroups {
		t.Fatalf("larger budget groups %d not above smaller %d", largeW.CacheGroups, smallW.CacheGroups)
	}
	if !smallW.HitRateKnown || !largeW.HitRateKnown {
		t.Fatalf("hit rates unknown: small known=%v large known=%v", smallW.HitRateKnown, largeW.HitRateKnown)
	}
	if largeW.HitRate < smallW.HitRate {
		t.Fatalf("hit rate regressed with budget: small %.6f (groups %d) large %.6f (groups %d)",
			smallW.HitRate, smallW.CacheGroups, largeW.HitRate, largeW.CacheGroups)
	}
}

// TestBeladyEventsMapping pins the (layer, expert) -> SpanID mapping the Belady
// oracle keys on: Layer*Experts+Expert+1, one unit per group touch.
func TestBeladyEventsMapping(t *testing.T) {
	trace := ExpertActivationTrace{
		Layers: 3, Experts: 5, TopK: 2,
		Routes: []ExpertRoute{
			{Layer: 0, Experts: []int{0, 4}},
			{Layer: 2, Experts: []int{1, 3}},
		},
	}
	events := beladyEvents(trace)
	want := []int{0*5 + 0 + 1, 0*5 + 4 + 1, 2*5 + 1 + 1, 2*5 + 3 + 1}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("beladyEvents = %v, want %v", events, want)
	}
}

// TestBeladyOptimalHitsExactMinimum is the direct correctness gate for the
// witness-owned oracle: the headline churn fixture has a true offline optimum of
// 5 hits (keep 5 of the 6 layer-0 groups across the full-cache layer-1 pass),
// NOT the 6 a buggy "keep the whole mask without admitting the miss" DP reports.
func TestBeladyOptimalHitsExactMinimum(t *testing.T) {
	trace := ExpertActivationTrace{
		Layers: 2, Experts: 384, TopK: 6, GroupBytes: 1, BudgetBytes: 6,
		Routes: []ExpertRoute{
			{Layer: 0, Experts: []int{0, 1, 2, 3, 4, 5}},
			{Layer: 1, Experts: []int{0, 1, 2, 3, 4, 5}},
			{Layer: 0, Experts: []int{0, 1, 2, 3, 4, 5}},
		},
	}
	got, exact := beladyOptimalHits(beladyEvents(trace), 6)
	if !exact {
		t.Fatalf("beladyOptimalHits reported approximate for a 12-span trace")
	}
	if got != 5 {
		t.Fatalf("beladyOptimalHits = %d, want exactly 5 (honest offline optimum)", got)
	}
}
