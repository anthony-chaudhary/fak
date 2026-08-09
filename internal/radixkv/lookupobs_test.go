package radixkv

import "testing"

// lookupobs_test.go — the prefix-cache lifecycle counters (#5804). These tests pin the two
// properties the counters exist for: the three lookup buckets PARTITION every probe, and a
// miss is attributed to warmup (cold) vs genuine non-overlap (divergent) rather than being
// collapsed into one unfalsifiable "miss" figure.

// wantLifecycle asserts the full counter vector in one place so a drift in any single
// bucket names itself instead of surfacing as a partition-invariant failure elsewhere.
func wantLifecycle(t *testing.T, tree *Tree, lookups, hits, cold, divergent, fills int) {
	t.Helper()
	st := tree.Stats()
	if st.Lookups != lookups || st.LookupHits != hits || st.LookupMissCold != cold ||
		st.LookupMissDivergent != divergent || st.Fills != fills {
		t.Fatalf("lifecycle = {lookups:%d hits:%d cold:%d divergent:%d fills:%d}, want {%d %d %d %d %d}",
			st.Lookups, st.LookupHits, st.LookupMissCold, st.LookupMissDivergent, st.Fills,
			lookups, hits, cold, divergent, fills)
	}
}

// The headline invariant from lookupobs.go's contract: every probe lands in exactly one
// bucket, so the three always sum to Lookups. Driven over a mixed workload (cold warmup,
// exact reuse, shared-prefix reuse, and a disjoint request) so all three buckets are live.
func TestLookupLifecycleCountersPartition(t *testing.T) {
	tree := New(0)
	preamble := seq(1, 16)

	touchPure(tree, preamble)                         // cold: empty tree
	touchPure(tree, preamble)                         // hit: exact reuse
	touchPure(tree, cat(preamble, distinctReq(1, 8))) // hit: shared prefix, splits the edge
	touchPure(tree, distinctReq(9, 8))                // divergent: populated tree, no shared token

	st := tree.Stats()
	if st.Lookups != 4 {
		t.Fatalf("Lookups = %d, want 4", st.Lookups)
	}
	if sum := st.LookupHits + st.LookupMissCold + st.LookupMissDivergent; sum != st.Lookups {
		t.Fatalf("buckets sum to %d, want Lookups=%d (hits=%d cold=%d divergent=%d)",
			sum, st.Lookups, st.LookupHits, st.LookupMissCold, st.LookupMissDivergent)
	}
	// fills=3, not 4: the exact-reuse probe is fully cached, so its suffix is empty and
	// attaches nothing. Fills counts attached leaves, not served requests.
	wantLifecycle(t, tree, 4, 2, 1, 1, 3)
}

// The distinction the file exists for: an identical 0-hit outcome is COLD against an empty
// cache and DIVERGENT against a populated one. Collapsing these is the diagnosis bug.
func TestLookupMissColdVsDivergent(t *testing.T) {
	tree := New(0)

	touchPure(tree, distinctReq(0, 8)) // nothing resident → cold
	wantLifecycle(t, tree, 1, 0, 1, 0, 1)

	touchPure(tree, distinctReq(5, 8)) // resident, but shares no leading token → divergent
	wantLifecycle(t, tree, 2, 0, 1, 1, 2)
}

// An empty request cannot match anything regardless of residency, so it is warmup-shaped
// (cold), not a workload non-overlap signal.
func TestLookupEmptyRequestIsColdMiss(t *testing.T) {
	tree := New(0)
	touchPure(tree, distinctReq(0, 8)) // populate

	b, m := tree.Lookup(nil)
	tree.Done(b)
	if m != 0 {
		t.Fatalf("empty request matched %d tokens, want 0", m)
	}
	wantLifecycle(t, tree, 2, 0, 2, 0, 1) // both the populate and the empty probe are cold
}

// Cold-vs-divergent is judged per NAMESPACE (#3889): a busy namespace must not make another
// namespace's first probe look like divergence, or every namespace's warmup is misreported.
func TestLookupMissColdIsPerNamespace(t *testing.T) {
	tree := New(0)
	req := distinctReq(0, 8)

	b, m := tree.LookupNS("busy", req)
	tree.Done(tree.Insert(b, req[m:], nil)) // the namespace is carried by the boundary
	wantLifecycle(t, tree, 1, 0, 1, 0, 1)

	// "quiet" is empty even though "busy" holds nodes → this probe is COLD, not divergent.
	b2, m2 := tree.LookupNS("quiet", distinctReq(3, 8))
	tree.Done(b2)
	if m2 != 0 {
		t.Fatalf("cross-namespace probe matched %d tokens, want 0", m2)
	}
	wantLifecycle(t, tree, 2, 0, 2, 0, 1)
}

// MatchLenNS is documented as a read-only accounting probe. Counting it would double-book
// exactly the workloads that use it to measure the hit rate these counters explain.
func TestMatchLenIsNotCountedAsLookup(t *testing.T) {
	tree := New(0)
	req := distinctReq(0, 8)
	touchPure(tree, req)
	before := tree.Stats()

	if got := tree.MatchLen(req); got != len(req) {
		t.Fatalf("MatchLen = %d, want %d", got, len(req))
	}
	if got := tree.MatchLen(distinctReq(7, 8)); got != 0 {
		t.Fatalf("MatchLen on a disjoint request = %d, want 0", got)
	}

	after := tree.Stats()
	if after.Lookups != before.Lookups || after.LookupHits != before.LookupHits ||
		after.LookupMissCold != before.LookupMissCold ||
		after.LookupMissDivergent != before.LookupMissDivergent {
		t.Fatalf("MatchLen moved the lookup counters: before=%+v after=%+v", before, after)
	}
}

// Fills book EVERY attached leaf, so demand Insert and prewarm WarmInsert both count — a
// prewarmed entry occupies budget exactly like a demand-filled one.
func TestFillsCountDemandAndPrewarm(t *testing.T) {
	tree := New(0)

	touchPure(tree, distinctReq(0, 8)) // demand fill
	wantLifecycle(t, tree, 1, 0, 1, 0, 1)

	tree.WarmInsert(distinctReq(1, 8), nil) // prewarm fill: no Lookup, so no probe is booked
	wantLifecycle(t, tree, 1, 0, 1, 0, 2)

	// A fully-cached request attaches nothing (empty suffix) → a hit with no new fill.
	touchPure(tree, distinctReq(0, 8))
	wantLifecycle(t, tree, 2, 1, 1, 0, 2)
}

// The counters book EVENTS, not residency: eviction reclaims nodes but must never roll a
// lifecycle counter back, or a warm/cold rerun cannot reconstruct what the cache was asked.
func TestLifecycleCountersAreMonotonicAcrossEviction(t *testing.T) {
	tree := New(20)
	touchPure(tree, distinctReq(0, 10))
	touchPure(tree, distinctReq(1, 10))
	touchPure(tree, distinctReq(2, 10)) // 30 > 20 → evicts the first (LRU)

	st := tree.Stats()
	if st.Evictions == 0 {
		t.Fatalf("setup: expected a budget eviction, got Evictions=0")
	}
	if st.Lookups != 3 || st.Fills != 3 {
		t.Fatalf("eviction rolled back lifecycle counters: lookups=%d fills=%d, want 3/3",
			st.Lookups, st.Fills)
	}
	// The evicted prefix is gone, so re-demanding it is a DIVERGENT miss (the tree is still
	// populated) — the bucket that distinguishes "evicted too early" from "never overlapped".
	touchPure(tree, distinctReq(0, 10))
	if st2 := tree.Stats(); st2.LookupMissDivergent != 3 || st2.Lookups != 4 {
		t.Fatalf("post-eviction re-demand: divergent=%d lookups=%d, want 3/4",
			st2.LookupMissDivergent, st2.Lookups)
	}
}
