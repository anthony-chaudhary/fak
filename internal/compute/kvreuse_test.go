package compute

import (
	"math"
	"testing"
)

// TestKVReuseTermExactReduction verifies that zero-value and basic inputs reduce exactly to Hits + 1 (#3411).
func TestKVReuseTermExactReduction(t *testing.T) {
	cases := []struct {
		hits int
		want float64
	}{
		{0, 1.0},
		{1, 2.0},
		{5, 6.0},
		{100, 101.0},
		{-1, 1.0},
	}

	for _, tc := range cases {
		s := KVSpanStats{Hits: tc.hits}
		if got := KVReuseTerm(s); got != tc.want {
			t.Fatalf("KVReuseTerm(Hits=%d) = %v, want %v", tc.hits, got, tc.want)
		}
	}
}

// TestKVReuseSeamParityAcrossCostVariants verifies that KVEvictionCost, KVEvictionCostFanout,
// KVEvictionCostPinned, and KVEvictionCostHazard all compute byte-identical values on base inputs (#3411).
func TestKVReuseSeamParityAcrossCostVariants(t *testing.T) {
	s := KVSpanStats{
		Tokens: 100,
		Bytes:  400,
		Hits:   3,
	}

	wantBase := float64(100) * float64(4) / float64(400) // 1.0

	cost := KVEvictionCost(s)
	if cost != wantBase {
		t.Fatalf("KVEvictionCost = %v, want %v", cost, wantBase)
	}

	costFanout := KVEvictionCostFanout(s)
	if costFanout != wantBase {
		t.Fatalf("KVEvictionCostFanout = %v, want %v", costFanout, wantBase)
	}

	costPinned := KVEvictionCostPinned(s)
	if costPinned != wantBase {
		t.Fatalf("KVEvictionCostPinned = %v, want %v", costPinned, wantBase)
	}

	costHazard := KVEvictionCostHazard(s, 0)
	if costHazard != wantBase {
		t.Fatalf("KVEvictionCostHazard = %v, want %v", costHazard, wantBase)
	}
}

// TestKVEvictionCostHazardBreaksEqualHitsTie verifies that age-conditioned hazard decay breaks
// ties between spans with equal Hits, favoring the fresher span (#2669/#3411).
func TestKVEvictionCostHazardBreaksEqualHitsTie(t *testing.T) {
	clock := uint64(1000)

	// Both spans have 10 hits and same size.
	// fresh: accessed at 980 (age = 20), meanIA = 50. age <= meanIA -> no penalty, reuse = 11.
	// stale: accessed at 200 (age = 800), meanIA = 50. age >> meanIA -> decayed reuse < 11.
	fresh := KVSpanStats{
		Tokens:        100,
		Bytes:         400,
		Hits:          10,
		LastUsed:      980,
		IntervalSum:   100,
		IntervalCount: 2, // meanIA = 50
	}
	stale := KVSpanStats{
		Tokens:        100,
		Bytes:         400,
		Hits:          10,
		LastUsed:      200,
		IntervalSum:   100,
		IntervalCount: 2, // meanIA = 50
	}

	costFresh := KVEvictionCostHazard(fresh, clock)
	costStale := KVEvictionCostHazard(stale, clock)

	if costFresh <= costStale {
		t.Fatalf("expected fresh span cost (%v) > stale span cost (%v)", costFresh, costStale)
	}

	// Raw KVEvictionCost cannot separate them
	if KVEvictionCost(fresh) != KVEvictionCost(stale) {
		t.Fatalf("expected raw KVEvictionCost to be identical on equal Hits")
	}

	// PickEvictionVictimHazard must choose the stale span as the cheaper-to-lose victim
	spans := []KVSpanStats{fresh, stale}
	victim := PickEvictionVictimHazard(spans, clock)
	if victim != 1 {
		t.Fatalf("PickEvictionVictimHazard chose %d, want 1 (stale span)", victim)
	}
}

// TestKVReuseTermCustomEstimator verifies the pluggable seam with a custom estimator.
func TestKVReuseTermCustomEstimator(t *testing.T) {
	customEstimator := func(s KVSpanStats) float64 {
		return float64(s.Hits)*2.0 + 1.0
	}

	s := KVSpanStats{Hits: 5}
	got := KVReuseTermWithEstimator(s, customEstimator)
	want := 11.0
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("custom estimator got %v, want %v", got, want)
	}

	// nil falls back to default
	gotDefault := KVReuseTermWithEstimator(s, nil)
	if gotDefault != 6.0 {
		t.Fatalf("nil estimator fallback got %v, want 6.0", gotDefault)
	}
}
