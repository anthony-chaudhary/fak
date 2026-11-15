package perfledger

import (
	"math"
	"testing"
)

// near reports whether a and b are equal within an absolute tolerance.
func near(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

// TestPrefixHitRate verifies the ported prefix hit-rate analysis against known
// inputs, asserting the hit-rate values (and the surrounding overlap statistics)
// match the expected outputs of the source aiperf prefix_analyzer.
func TestPrefixHitRate(t *testing.T) {
	// Main case: three requests sharing a two-block prefix, block_size = 1 so
	// the context/unique split stays in token units.
	//
	//   t1 = [1,2,3,4]  (first to arrive: nothing seen yet)
	//   t2 = [1,2,5,6]  (shares blocks 1,2 with t1)
	//   t3 = [1,2,3,4]  (identical to t1: fully cached)
	//
	// Per-request hit rate (dynamo first-unseen-block):
	//   t1 -> 0/4 = 0.0
	//   t2 -> 2/4 = 0.5   (blocks 1,2 seen; 5 is the first unseen)
	//   t3 -> 4/4 = 1.0   (all blocks already seen)
	traces := []Trace{
		{InputLength: 4, OutputLength: 1, HashIDs: []int{1, 2, 3, 4}},
		{InputLength: 4, OutputLength: 1, HashIDs: []int{1, 2, 5, 6}},
		{InputLength: 4, OutputLength: 1, HashIDs: []int{1, 2, 3, 4}},
	}
	a := NewPrefixAnalyzer(1)

	t.Run("PerRequestHitRates", func(t *testing.T) {
		got := a.PerRequestHitRates(traces)
		want := []float64{0.0, 0.5, 1.0}
		if len(got) != len(want) {
			t.Fatalf("PerRequestHitRates() len = %d, want %d (got %v)", len(got), len(want), got)
		}
		for i := range want {
			if !near(got[i], want[i]) {
				t.Errorf("PerRequestHitRates()[%d] = %v, want %v", i, got[i], want[i])
			}
		}
	})

	t.Run("AnalyzeTraces", func(t *testing.T) {
		s := a.AnalyzeTraces(traces)

		if s.TotalRequests != 3 {
			t.Errorf("TotalRequests = %d, want 3", s.TotalRequests)
		}
		// Mean of the per-request hit rates.
		if !near(s.CacheHitRate, 0.5) {
			t.Errorf("CacheHitRate = %v, want 0.5", s.CacheHitRate)
		}
		// Distinct prefix tuples: {1},{1,2},{1,2,3},{1,2,3,4},{1,2,5},{1,2,5,6}.
		if s.UniquePrefixes != 6 {
			t.Errorf("UniquePrefixes = %d, want 6", s.UniquePrefixes)
		}
		// All three requests start with block 1 -> one shared first-block group.
		if s.NumPrefixGroups != 1 {
			t.Errorf("NumPrefixGroups = %d, want 1", s.NumPrefixGroups)
		}
		// Reused prefixes: {1}:3,{1,2}:3,{1,2,3}:2,{1,2,3,4}:2 = 10 of 12 total.
		if !near(s.PrefixReuseRatio, 10.0/12.0) {
			t.Errorf("PrefixReuseRatio = %v, want %v", s.PrefixReuseRatio, 10.0/12.0)
		}

		if s.MinISL != 4 || s.MaxISL != 4 || !near(s.AvgISL, 4) {
			t.Errorf("ISL min/max/avg = %d/%d/%v, want 4/4/4", s.MinISL, s.MaxISL, s.AvgISL)
		}
		if s.MinOSL != 1 || s.MaxOSL != 1 || !near(s.AvgOSL, 1) {
			t.Errorf("OSL min/max/avg = %d/%d/%v, want 1/1/1", s.MinOSL, s.MaxOSL, s.AvgOSL)
		}

		// Context lengths: t1 all-repeated -> 4; t2 two repeated blocks -> 2;
		// t3 all-repeated -> 4. Unique prompt: 0, 2, 0.
		if s.ContextLengthStats == nil || s.UniquePromptLengthStats == nil {
			t.Fatalf("expected non-nil context/unique-prompt stats")
		}
		if !near(s.ContextLengthStats.Min, 2) || !near(s.ContextLengthStats.Max, 4) ||
			!near(s.ContextLengthStats.Median, 4) || !near(s.ContextLengthStats.Mean, 10.0/3.0) {
			t.Errorf("ContextLengthStats = %+v, want min=2 max=4 median=4 mean=10/3", s.ContextLengthStats)
		}
		if !near(s.UniquePromptLengthStats.Min, 0) || !near(s.UniquePromptLengthStats.Max, 2) ||
			!near(s.UniquePromptLengthStats.Median, 0) || !near(s.UniquePromptLengthStats.Mean, 2.0/3.0) {
			t.Errorf("UniquePromptLengthStats = %+v, want min=0 max=2 median=0 mean=2/3", s.UniquePromptLengthStats)
		}

		// Hit-rate distribution over [0, 0.5, 1].
		if s.HitRateStats == nil {
			t.Fatalf("expected non-nil HitRateStats")
		}
		if !near(s.HitRateStats.Mean, 0.5) || !near(s.HitRateStats.StdDev, math.Sqrt(1.0/6.0)) ||
			!near(s.HitRateStats.Min, 0) || !near(s.HitRateStats.P25, 0.25) ||
			!near(s.HitRateStats.Median, 0.5) || !near(s.HitRateStats.P75, 0.75) ||
			!near(s.HitRateStats.Max, 1) {
			t.Errorf("HitRateStats = %+v, want mean=0.5 std=sqrt(1/6) min=0 p25=0.25 median=0.5 p75=0.75 max=1", s.HitRateStats)
		}
	})
}

// TestPrefixHitRateEmptyHashIDs verifies a request with no hash_ids is skipped
// in the hit-rate computation and yields context 0 / unique = full ISL.
func TestPrefixHitRateEmptyHashIDs(t *testing.T) {
	a := NewPrefixAnalyzer(1)
	traces := []Trace{
		{InputLength: 8, OutputLength: 2, HashIDs: nil},
	}

	if got := a.PerRequestHitRates(traces); len(got) != 0 {
		t.Fatalf("PerRequestHitRates() = %v, want empty", got)
	}

	s := a.AnalyzeTraces(traces)
	if s.TotalRequests != 1 {
		t.Errorf("TotalRequests = %d, want 1", s.TotalRequests)
	}
	if s.CacheHitRate != 0 {
		t.Errorf("CacheHitRate = %v, want 0", s.CacheHitRate)
	}
	if s.UniquePrefixes != 0 || s.NumPrefixGroups != 0 || s.PrefixReuseRatio != 0 {
		t.Errorf("prefix stats = %d/%d/%v, want 0/0/0", s.UniquePrefixes, s.NumPrefixGroups, s.PrefixReuseRatio)
	}
	if s.ContextLengthStats == nil || !near(s.ContextLengthStats.Min, 0) || !near(s.ContextLengthStats.Max, 0) {
		t.Errorf("ContextLengthStats = %+v, want all-zero", s.ContextLengthStats)
	}
	if s.UniquePromptLengthStats == nil || !near(s.UniquePromptLengthStats.Min, 8) || !near(s.UniquePromptLengthStats.Max, 8) {
		t.Errorf("UniquePromptLengthStats = %+v, want 8/8", s.UniquePromptLengthStats)
	}
}

// TestPrefixHitRateNoOverlap verifies disjoint requests all report a 0.0 hit
// rate and no prefix reuse.
func TestPrefixHitRateNoOverlap(t *testing.T) {
	a := NewPrefixAnalyzer(1)
	traces := []Trace{
		{InputLength: 2, OutputLength: 1, HashIDs: []int{10, 11}},
		{InputLength: 2, OutputLength: 1, HashIDs: []int{20, 21}},
	}

	got := a.PerRequestHitRates(traces)
	want := []float64{0.0, 0.0}
	if len(got) != len(want) {
		t.Fatalf("PerRequestHitRates() len = %d, want %d (got %v)", len(got), len(want), got)
	}
	for i := range want {
		if !near(got[i], want[i]) {
			t.Errorf("PerRequestHitRates()[%d] = %v, want %v", i, got[i], want[i])
		}
	}

	s := a.AnalyzeTraces(traces)
	if !near(s.CacheHitRate, 0) {
		t.Errorf("CacheHitRate = %v, want 0", s.CacheHitRate)
	}
	if s.UniquePrefixes != 4 {
		t.Errorf("UniquePrefixes = %d, want 4", s.UniquePrefixes)
	}
	if s.NumPrefixGroups != 0 {
		t.Errorf("NumPrefixGroups = %d, want 0", s.NumPrefixGroups)
	}
	if s.PrefixReuseRatio != 0 {
		t.Errorf("PrefixReuseRatio = %v, want 0", s.PrefixReuseRatio)
	}
}

// TestPrefixHitRateAllRepeatedUsesFullISL verifies the source's all-repeated
// shortcut: when every (position, hash_id) pair is repeated, the context length
// is the full ISL (not repeated_count * block_size), even under the default
// 512-token block size.
func TestPrefixHitRateAllRepeatedUsesFullISL(t *testing.T) {
	a := NewPrefixAnalyzer(DefaultBlockSize)
	traces := []Trace{
		{InputLength: 4, OutputLength: 1, HashIDs: []int{1, 2, 3, 4}},
		{InputLength: 4, OutputLength: 1, HashIDs: []int{1, 2, 3, 4}},
	}

	got := a.PerRequestHitRates(traces)
	want := []float64{0.0, 1.0}
	if len(got) != len(want) {
		t.Fatalf("PerRequestHitRates() len = %d, want %d (got %v)", len(got), len(want), got)
	}
	for i := range want {
		if !near(got[i], want[i]) {
			t.Errorf("PerRequestHitRates()[%d] = %v, want %v", i, got[i], want[i])
		}
	}

	s := a.AnalyzeTraces(traces)
	// All pairs repeated for both requests -> context == full ISL (4), not 4*512.
	if s.ContextLengthStats == nil || !near(s.ContextLengthStats.Min, 4) || !near(s.ContextLengthStats.Max, 4) {
		t.Errorf("ContextLengthStats = %+v, want min=max=4 (full ISL shortcut)", s.ContextLengthStats)
	}
	if s.UniquePromptLengthStats == nil || !near(s.UniquePromptLengthStats.Min, 0) || !near(s.UniquePromptLengthStats.Max, 0) {
		t.Errorf("UniquePromptLengthStats = %+v, want all-zero", s.UniquePromptLengthStats)
	}
}

// TestPrefixHitRatePartialOverlapBlockSized verifies the partial-overlap rule
// context = repeated_count * block_size under the default 512-token block size,
// including the source's behavior of a negative unique-prompt length when the
// block size exceeds the ISL.
func TestPrefixHitRatePartialOverlapBlockSized(t *testing.T) {
	a := NewPrefixAnalyzer(DefaultBlockSize)
	traces := []Trace{
		{InputLength: 4, OutputLength: 1, HashIDs: []int{1, 2, 3, 4}},
		{InputLength: 4, OutputLength: 1, HashIDs: []int{1, 2, 5, 6}},
	}

	s := a.AnalyzeTraces(traces)
	// Two repeated (position, hash) blocks per request -> 2 * 512 = 1024.
	if s.ContextLengthStats == nil || !near(s.ContextLengthStats.Min, 1024) || !near(s.ContextLengthStats.Max, 1024) {
		t.Errorf("ContextLengthStats = %+v, want min=max=1024", s.ContextLengthStats)
	}
	// unique = isl - context = 4 - 1024 = -1020 (faithful to the source).
	if s.UniquePromptLengthStats == nil || !near(s.UniquePromptLengthStats.Min, -1020) || !near(s.UniquePromptLengthStats.Max, -1020) {
		t.Errorf("UniquePromptLengthStats = %+v, want min=max=-1020", s.UniquePromptLengthStats)
	}
}
