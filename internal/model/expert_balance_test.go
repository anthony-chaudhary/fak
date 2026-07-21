package model

import "testing"

// expert_balance_test.go — the #3886 acceptance witness: LoadBalancedExpertBands over a
// skewed per-expert load vector has STRICTLY lower max-per-rank load (and higher
// balancedness) than ExpertParallelPlan's count-even NewTPPlan bands on the same skew,
// while staying a valid contiguous TPPlan bit-exact-shaped at ranks=1. Pure Go, no GPU,
// no network, deterministic.

// maxOf is a local test helper (the max per-rank load caps latency at the hottest rank).
func maxOf(v []int) int {
	m := 0
	for _, x := range v {
		if x > m {
			m = x
		}
	}
	return m
}

// TestLoadBalancedExpertBandsBeatsCountEvenBands is the core acceptance: on a skewed load
// vector, the load-balanced contiguous placement reduces the max-per-rank load below the
// count-even bands NewTPPlan produces. This is the "rebalances under imbalance" property the
// GLM EP note said the count plan lacked.
func TestLoadBalancedExpertBandsBeatsCountEvenBands(t *testing.T) {
	// 8 experts, 2 ranks. Four hot experts (load 10) then four cold (load 1). The count-even
	// plan bands [0,4)/[4,8) put ALL the hot experts on rank 0: loads {40, 4}, max 40.
	load := []int{10, 10, 10, 10, 1, 1, 1, 1}
	ranks := 2

	even, err := ExpertParallelPlan(len(load), ranks) // count-even NewTPPlan bands
	if err != nil {
		t.Fatalf("ExpertParallelPlan: %v", err)
	}
	evenLoads, err := RankLoads(load, even)
	if err != nil {
		t.Fatalf("RankLoads(even): %v", err)
	}

	bal, err := LoadBalancedExpertBands(load, ranks)
	if err != nil {
		t.Fatalf("LoadBalancedExpertBands: %v", err)
	}
	balLoads, err := RankLoads(load, bal)
	if err != nil {
		t.Fatalf("RankLoads(balanced): %v", err)
	}

	evenMax, balMax := maxOf(evenLoads), maxOf(balLoads)
	if balMax >= evenMax {
		t.Fatalf("balanced max-per-rank load %d not below count-even %d (loads even=%v balanced=%v)",
			balMax, evenMax, evenLoads, balLoads)
	}
	// The optimal contiguous 2-partition of this skew is [0,2)/[2,8): loads {20,24}, max 24.
	if balMax != 24 {
		t.Fatalf("balanced max-per-rank load = %d, want optimal 24 (loads=%v bands=%v)", balMax, balLoads, bal.Shards)
	}

	// Balancedness (mean/max) must move toward 1.0 — the vLLM-comparable metric.
	evenB, err := Balancedness(load, even)
	if err != nil {
		t.Fatalf("Balancedness(even): %v", err)
	}
	balB, err := Balancedness(load, bal)
	if err != nil {
		t.Fatalf("Balancedness(balanced): %v", err)
	}
	if !(balB > evenB) {
		t.Fatalf("balanced balancedness %.4f not above count-even %.4f", balB, evenB)
	}
	t.Logf("max-per-rank load: even=%d balanced=%d; balancedness: even=%.3f balanced=%.3f",
		evenMax, balMax, evenB, balB)
}

// TestLoadBalancedExpertBandsSingleHotExpert exercises an extreme skew: one dominant expert.
// The balanced plan must isolate it in its own band so the other rank carries everything else.
func TestLoadBalancedExpertBandsSingleHotExpert(t *testing.T) {
	load := []int{100, 1, 1, 1, 1, 1, 1, 1} // one hot expert, seven cold
	bal, err := LoadBalancedExpertBands(load, 2)
	if err != nil {
		t.Fatalf("LoadBalancedExpertBands: %v", err)
	}
	loads, err := RankLoads(load, bal)
	if err != nil {
		t.Fatalf("RankLoads: %v", err)
	}
	// Optimal split isolates expert 0: bands [0,1)/[1,8), loads {100,7}, max 100 — below the
	// count-even {103,4} max of 103.
	if got := maxOf(loads); got != 100 {
		t.Fatalf("max-per-rank load = %d, want 100 (loads=%v bands=%v)", got, loads, bal.Shards)
	}
	if bal.Shards[0].Lo != 0 || bal.Shards[0].Hi != 1 {
		t.Fatalf("hot expert not isolated: band0 = [%d,%d), want [0,1)", bal.Shards[0].Lo, bal.Shards[0].Hi)
	}
}

// TestLoadBalancedExpertBandsRanksOneIsWholeBand pins the ranks=1 invariant: a single band
// [0,n) — identical to NewTPPlan(n,1) — so the load-balanced plan keeps the bit-exact-vs-
// monolith property expert_parallel.go depends on at ranks=1.
func TestLoadBalancedExpertBandsRanksOneIsWholeBand(t *testing.T) {
	load := []int{5, 9, 1, 7, 3}
	bal, err := LoadBalancedExpertBands(load, 1)
	if err != nil {
		t.Fatalf("LoadBalancedExpertBands: %v", err)
	}
	if len(bal.Shards) != 1 || bal.Shards[0].Lo != 0 || bal.Shards[0].Hi != len(load) {
		t.Fatalf("ranks=1 plan = %v, want single band [0,%d)", bal.Shards, len(load))
	}
	even, err := ExpertParallelPlan(len(load), 1)
	if err != nil {
		t.Fatalf("ExpertParallelPlan: %v", err)
	}
	if bal.Shards[0] != even.Shards[0] {
		t.Fatalf("ranks=1 balanced band %v != count-even band %v", bal.Shards[0], even.Shards[0])
	}
}

// TestLoadBalancedExpertBandsFlatLoadMatchesCountEven proves the plan does NOT thrash a
// balanced workload: on a flat load vector the load-optimal partition coincides with the
// count-even bands (equal load makes count-even already max-optimal).
func TestLoadBalancedExpertBandsFlatLoadMatchesCountEven(t *testing.T) {
	load := []int{4, 4, 4, 4, 4, 4} // perfectly flat
	ranks := 3
	bal, err := LoadBalancedExpertBands(load, ranks)
	if err != nil {
		t.Fatalf("LoadBalancedExpertBands: %v", err)
	}
	even, err := ExpertParallelPlan(len(load), ranks)
	if err != nil {
		t.Fatalf("ExpertParallelPlan: %v", err)
	}
	balLoads, _ := RankLoads(load, bal)
	evenLoads, _ := RankLoads(load, even)
	if maxOf(balLoads) != maxOf(evenLoads) {
		t.Fatalf("flat load: balanced max %v != count-even max %v", maxOf(balLoads), maxOf(evenLoads))
	}
	b, err := Balancedness(load, bal)
	if err != nil {
		t.Fatalf("Balancedness: %v", err)
	}
	if b != 1.0 {
		t.Fatalf("flat load balancedness = %.4f, want 1.0", b)
	}
}

// TestLoadBalancedExpertBandsFailsClosed pins the fail-closed boundaries: empty load, non-
// positive ranks, ranks exceeding the expert count, and a negative load entry.
func TestLoadBalancedExpertBandsFailsClosed(t *testing.T) {
	cases := []struct {
		name  string
		load  []int
		ranks int
	}{
		{"empty load", nil, 1},
		{"zero ranks", []int{1, 2, 3}, 0},
		{"negative ranks", []int{1, 2, 3}, -2},
		{"ranks exceed experts", []int{1, 2}, 3},
		{"negative load", []int{1, -4, 2}, 2},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := LoadBalancedExpertBands(c.load, c.ranks); err == nil {
				t.Fatalf("LoadBalancedExpertBands(%v, %d) = nil error, want fail-closed", c.load, c.ranks)
			}
		})
	}
}

// TestBalancednessFailsClosedOnZeroTraffic proves the metric refuses to fabricate a 1.0 for a
// serve that has drawn no picks yet (undefined balancedness).
func TestBalancednessFailsClosedOnZeroTraffic(t *testing.T) {
	load := []int{0, 0, 0, 0}
	p, err := LoadBalancedExpertBands(load, 2)
	if err != nil {
		t.Fatalf("LoadBalancedExpertBands: %v", err)
	}
	if _, err := Balancedness(load, p); err == nil {
		t.Fatalf("Balancedness on zero traffic = nil error, want fail-closed")
	}
}

// TestRankLoadsRejectsDimMismatch pins that a plan built for a different expert count cannot be
// silently mis-summed against a load vector.
func TestRankLoadsRejectsDimMismatch(t *testing.T) {
	p, err := LoadBalancedExpertBands([]int{1, 2, 3, 4}, 2)
	if err != nil {
		t.Fatalf("LoadBalancedExpertBands: %v", err)
	}
	if _, err := RankLoads([]int{1, 2, 3}, p); err == nil {
		t.Fatalf("RankLoads with mismatched load len = nil error, want fail-closed")
	}
}
