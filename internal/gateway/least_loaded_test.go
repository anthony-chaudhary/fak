package gateway

import (
	"math/rand"
	"testing"
)

// TestLeastLoaded* pins the ai-dynamo/aiperf sticky-router borrow landed in
// this package (ADAPT — re-expressed in Go, no copied code; source:
// src/aiperf/credit/sticky_router.py:389-391,550-581
// @ab610add4e8876fca223197596c728e17365f381, Apache-2.0): the fleet load vector
// kept as a bucket map so the minimum load is reachable in O(1) and a
// least-loaded pick scans only the k-way tie. The closed contracts are exact
// placements and minima against a brute-force reference — no error prose,
// since the router reports placement, not errors.

// TestLeastLoadedMatchesBruteForce drives the load index through 500 seeded
// steps mixing incremental bumps (bumpLoad) and wholesale slice reassignment
// (the same-package pattern of setting r.load between Route calls), asserting
// after every step that minLoad and leastLoaded agree with the O(n)
// brute-force reference and that the buckets still partition the instance set.
func TestLeastLoadedMatchesBruteForce(t *testing.T) {
	const n = 8
	r := NewSlackTolerantFleetCacheRouter(n, 4, 2.0)
	rng := rand.New(rand.NewSource(1))
	for step := 0; step < 500; step++ {
		if rng.Intn(4) == 0 {
			// Wholesale reassignment: the index's slice stamp must detect it
			// and rebuild before the next read.
			loads := make([]int, n)
			for i := range loads {
				loads[i] = rng.Intn(6)
			}
			r.load = loads
		} else {
			r.bumpLoad(rng.Intn(n))
		}
		if got, want := r.minLoad(), bruteMin(r.load); got != want {
			t.Fatalf("step %d: minLoad = %d, want %d (loads %v)", step, got, want, r.load)
		}
		for _, except := range []int{-1, 0, n - 1} {
			if got, want := r.leastLoaded(except), bruteLeastLoaded(r.load, except); got != want {
				t.Fatalf("step %d: leastLoaded(%d) = %d, want %d (loads %v)", step, except, got, want, r.load)
			}
		}
		assertBucketPartition(t, r, step)
	}
}

// TestLeastLoadedTieBreakContract pins the closed placement contracts: the
// minimum class is found directly, ties resolve to the lower index, exclusion
// moves within a tie, and excluding the sole minimum occupant advances to the
// next load class.
func TestLeastLoadedTieBreakContract(t *testing.T) {
	r := NewSlackTolerantFleetCacheRouter(4, 8, 2.0)

	r.load = []int{5, 2, 2, 9}
	if got := r.minLoad(); got != 2 {
		t.Fatalf("minLoad = %d, want 2", got)
	}
	if got := r.leastLoaded(-1); got != 1 {
		t.Fatalf("leastLoaded(-1) = %d, want 1 (lower index of the tie)", got)
	}
	if got := r.leastLoaded(1); got != 2 {
		t.Fatalf("leastLoaded(1) = %d, want 2 (exclusion moves within the tie)", got)
	}
	if got := r.leastLoaded(2); got != 1 {
		t.Fatalf("leastLoaded(2) = %d, want 1", got)
	}

	r.load = []int{5, 9, 9, 9}
	if got := r.leastLoaded(-1); got != 0 {
		t.Fatalf("sole minimum: leastLoaded = %d, want 0", got)
	}

	r.load = []int{3, 3, 3, 3}
	if got := r.minLoad(); got != 3 {
		t.Fatalf("uniform minLoad = %d, want 3", got)
	}
	if got := r.leastLoaded(0); got != 1 {
		t.Fatalf("uniform leastLoaded(0) = %d, want 1", got)
	}

	// Excluding the SOLE minimum occupant advances to the next load class.
	r.load = []int{0, 4, 4, 4}
	if got := r.leastLoaded(0); got != 1 {
		t.Fatalf("leastLoaded(0) past sole minimum = %d, want 1 (next class, lower index)", got)
	}
}

// TestLeastLoadedIncrementalBump pins the incremental maintenance path: bumps
// flow through the index without a rebuild, and a wholesale reassignment
// mid-stream heals on the next use without corrupting subsequent bumps.
func TestLeastLoadedIncrementalBump(t *testing.T) {
	r := NewSlackTolerantFleetCacheRouter(3, 4, 2.0)

	r.bumpLoad(1)
	if r.load[0] != 0 || r.load[1] != 1 || r.load[2] != 0 {
		t.Fatalf("after bumpLoad(1): load = %v, want [0 1 0]", r.load)
	}
	if got := r.minLoad(); got != 0 {
		t.Fatalf("minLoad = %d, want 0", got)
	}
	if got := r.leastLoaded(-1); got != 0 {
		t.Fatalf("leastLoaded = %d, want 0 (tie {0,2} breaks low)", got)
	}

	// Wholesale reassignment, then a bump on top of it: the stamp detects the
	// new slice, rebuilds, and the bump lands on the fresh view.
	r.load = []int{4, 1, 4}
	r.bumpLoad(2)
	if r.load[0] != 4 || r.load[1] != 1 || r.load[2] != 5 {
		t.Fatalf("after reassign+bump: load = %v, want [4 1 5]", r.load)
	}
	if got := r.minLoad(); got != 1 {
		t.Fatalf("minLoad = %d, want 1", got)
	}
	if got := r.leastLoaded(-1); got != 1 {
		t.Fatalf("leastLoaded = %d, want 1", got)
	}
	if got := r.leastLoaded(1); got != 0 {
		t.Fatalf("leastLoaded(1) = %d, want 0 (next class {0,2} breaks low)", got)
	}
}

// bruteMin is the O(n) reference: the fleet's minimum load.
func bruteMin(loads []int) int {
	m := loads[0]
	for _, l := range loads[1:] {
		if l < m {
			m = l
		}
	}
	return m
}

// bruteLeastLoaded is the O(n) reference pick: fewest in-flight requests,
// skipping except, ties to the lower index.
func bruteLeastLoaded(loads []int, except int) int {
	best := -1
	for i := range loads {
		if i == except {
			continue
		}
		if best == -1 || loads[i] < loads[best] {
			best = i
		}
	}
	return best
}

// assertBucketPartition checks the borrowed structure's invariant: the buckets
// partition the instance set exactly — every instance appears in exactly one
// bucket, under its own load, and the key list is the sorted distinct loads.
func assertBucketPartition(t *testing.T, r *FleetCacheRouter, step int) {
	t.Helper()
	x := r.idx
	if x == nil {
		t.Fatalf("step %d: load index not built", step)
	}
	seen := make(map[int]int, len(r.load))
	for l, members := range x.buckets {
		for _, i := range members {
			seen[i]++
			if r.load[i] != l {
				t.Fatalf("step %d: instance %d in bucket %d but load is %d", step, i, l, r.load[i])
			}
		}
	}
	for i := range r.load {
		if seen[i] != 1 {
			t.Fatalf("step %d: instance %d appears %d times across buckets, want 1", step, i, seen[i])
		}
	}
	if len(x.keys) != len(x.buckets) {
		t.Fatalf("step %d: %d keys vs %d buckets", step, len(x.keys), len(x.buckets))
	}
	for k := 1; k < len(x.keys); k++ {
		if x.keys[k-1] >= x.keys[k] {
			t.Fatalf("step %d: keys not strictly ascending: %v", step, x.keys)
		}
	}
}
