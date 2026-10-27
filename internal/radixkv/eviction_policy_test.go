package radixkv

import (
	"math"
	"testing"
)

func touchPure(t *Tree, req []int) {
	_, leaf := servePure(t, req)
	t.Done(leaf)
}

func TestCostAwareEvictionKeepsReusedSpan(t *testing.T) {
	a, b, c := distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)

	lru := New(20)
	touchPure(lru, a)
	touchPure(lru, a) // refresh recency before b arrives; LRU should still later evict a.
	touchPure(lru, b)
	touchPure(lru, c)

	if m := lru.MatchLen(a); m != 0 {
		t.Fatalf("LRU should evict the older hot span once it becomes least-recently-used, matched %d", m)
	}

	costAware := NewWithEvictionPolicy(20, EvictionCostAware)
	costAware.SetAdmissionEnabled(false) // isolate the eviction selector from #9311 admission
	touchPure(costAware, a)
	touchPure(costAware, a)
	touchPure(costAware, b)
	touchPure(costAware, c)

	if m := costAware.MatchLen(a); m != len(a) {
		t.Fatalf("cost-aware should keep reused a, matched %d/%d", m, len(a))
	}
	if m := costAware.MatchLen(b); m != 0 {
		t.Fatalf("cost-aware should evict one-shot b, matched %d", m)
	}
	if m := costAware.MatchLen(c); m != len(c) {
		t.Fatalf("newly inserted c should survive, matched %d/%d", m, len(c))
	}
}

func TestCostAwareEvictionStatsExposeVictimTelemetry(t *testing.T) {
	tree := NewWithEvictionPolicy(20, EvictionCostAware)
	tree.SetAdmissionEnabled(false) // this test exercises eviction telemetry, not admission
	a, b, c := distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)

	touchPure(tree, a)
	touchPure(tree, a)
	touchPure(tree, b)
	touchPure(tree, c)

	st := tree.Stats()
	if st.EvictionPolicy != "cost-aware" {
		t.Fatalf("EvictionPolicy=%q, want cost-aware", st.EvictionPolicy)
	}
	if st.Evictions != 1 || st.CostEvictions != 1 {
		t.Fatalf("evictions=%d costEvictions=%d, want 1/1", st.Evictions, st.CostEvictions)
	}
	if st.ReuseHits == 0 {
		t.Fatal("ReuseHits=0, want observed demand reuse to feed the victim rule")
	}
	if st.LastEvictPolicy != "cost-aware" {
		t.Fatalf("LastEvictPolicy=%q, want cost-aware", st.LastEvictPolicy)
	}
	if st.LastEvictCandidates != 3 || st.LastEvictLocked != 1 {
		t.Fatalf("last eviction candidates/locked=%d/%d, want 3/1", st.LastEvictCandidates, st.LastEvictLocked)
	}
	if st.LastEvictVictimHits != 0 {
		t.Fatalf("victim hits=%d, want one-shot victim", st.LastEvictVictimHits)
	}
	if st.LastEvictVictimTokens != len(b) || st.LastEvictVictimPrefix != len(b) {
		t.Fatalf("victim tokens/prefix=%d/%d, want %d/%d",
			st.LastEvictVictimTokens, st.LastEvictVictimPrefix, len(b), len(b))
	}
	if math.Abs(st.LastEvictVictimCost-1) > 1e-9 {
		t.Fatalf("victim cost=%v, want 1", st.LastEvictVictimCost)
	}
}

func TestCostAwareEvictionRespectsLeases(t *testing.T) {
	tree := NewWithEvictionPolicy(20, EvictionCostAware)
	tree.SetAdmissionEnabled(false) // preserve the insert-always setup needed to pressure leases
	a, b, c := distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)

	_, leasedA := servePure(tree, a)
	touchPure(tree, b)
	_, leasedC := servePure(tree, c)
	tree.Done(leasedC)

	if m := tree.MatchLen(a); m != len(a) {
		t.Fatalf("leased a should survive cost-aware eviction, matched %d/%d", m, len(a))
	}
	if m := tree.MatchLen(b); m != 0 {
		t.Fatalf("unleased b should be evicted, matched %d", m)
	}
	if m := tree.MatchLen(c); m != len(c) {
		t.Fatalf("leased c should survive its insert-time eviction pass, matched %d/%d", m, len(c))
	}
	if st := tree.Stats(); st.LastEvictLocked != 2 {
		t.Fatalf("last eviction locked=%d, want 2 (a and c)", st.LastEvictLocked)
	}

	tree.Done(leasedA)
}

func TestSetEvictionPolicyUnknownFallsBackToLRU(t *testing.T) {
	tree := NewWithEvictionPolicy(0, EvictionPolicy(99))
	if got := tree.Stats().EvictionPolicy; got != "lru" {
		t.Fatalf("unknown policy = %q, want lru fallback", got)
	}
}

func TestPageAwareEvictionDrainsCompletePageFirst(t *testing.T) {
	tree, err := NewWithEvictionStrategy(20, "page-aware")
	if err != nil {
		t.Fatalf("NewWithEvictionStrategy(page-aware): %v", err)
	}

	a, b, c := distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)

	root := tree.root
	leafA := tree.InsertWithChunkID(root, a, nil, 1) // refs=1 (pinned)

	leafB := tree.InsertWithChunkID(root, b, nil, 2)
	tree.Done(leafB) // refs=0 (drainable)

	leafC := tree.InsertWithChunkID(root, c, nil, 3)
	tree.Done(leafC)

	if m := tree.MatchLen(a); m != len(a) {
		t.Fatalf("pinned leafA on chunk 1 should survive, matched %d/%d", m, len(a))
	}
	if m := tree.MatchLen(b); m != 0 {
		t.Fatalf("drainable leafB on chunk 2 should be evicted, matched %d", m)
	}
	if m := tree.MatchLen(c); m != len(c) {
		t.Fatalf("newly inserted leafC on chunk 3 should survive, matched %d/%d", m, len(c))
	}

	tree.Done(leafA)
}

// TestPageAwareEvictionIdenticalAgeCompletesEmptyPageFirst directly verifies #10721 checkable step:
// given two candidate leaves with identical age, the leaf whose removal completes an empty
// backing page is evicted first.
func TestPageAwareEvictionIdenticalAgeCompletesEmptyPageFirst(t *testing.T) {
	tree, err := NewWithEvictionStrategy(30, "page-aware")
	if err != nil {
		t.Fatalf("NewWithEvictionStrategy(page-aware): %v", err)
	}

	a := distinctReq(0, 10)
	b := distinctReq(1, 10)
	c := distinctReq(2, 10)

	// Leaf A on Page 1 (alone on Page 1).
	leafA := tree.InsertWithChunkID(tree.root, a, nil, 1)
	tree.Done(leafA)

	// Leaf B and Leaf C on Page 2 (share Page 2).
	leafB := tree.InsertWithChunkID(tree.root, b, nil, 2)
	tree.Done(leafB)
	leafC := tree.InsertWithChunkID(tree.root, c, nil, 2)
	tree.Done(leafC)

	// Force identical age between Leaf A and Leaf B.
	leafA.lastUsed = 100
	leafB.lastUsed = 100
	leafC.lastUsed = 200

	strat := tree.evictionStrategy()
	if prep, ok := strat.(TreePreparer); ok {
		prep.PrepareTree(tree)
	}

	keyA := strat.Priority(leafA)
	keyB := strat.Priority(leafB)

	// Page 1 has 1 leaf (Leaf A) -> cost 1.0 (1 eviction completely frees Page 1).
	// Page 2 has 2 leaves (Leaf B, C) -> cost 2.0 (needs 2 evictions to free Page 2).
	if !keyA.less(keyB) {
		t.Fatalf("leafA key (%+v) should be less than leafB key (%+v): leafA completes empty page 1", keyA, keyB)
	}

	victim := tree.victimLeaf()
	if victim != leafA {
		t.Fatalf("victim = %p (chunk %d), want leafA %p (chunk 1)", victim, victim.chunkID, leafA)
	}

	// Trigger eviction by reducing retention budget to 20.
	tree.SetRetention(20)

	if m := tree.MatchLen(a); m != 0 {
		t.Fatalf("leafA on chunk 1 should have been evicted to complete empty page, matched %d", m)
	}
	if m := tree.MatchLen(b); m != len(b) {
		t.Fatalf("leafB on chunk 2 should survive, matched %d/%d", m, len(b))
	}
	if m := tree.MatchLen(c); m != len(c) {
		t.Fatalf("leafC on chunk 2 should survive, matched %d/%d", m, len(c))
	}

	// Physical page stats: Page 1 was fully freed!
	active, drainable, _ := tree.PhysicalPageStats()
	if active != 1 {
		t.Fatalf("active pages = %d, want 1 (page 2 only; page 1 was completely freed)", active)
	}
	if drainable != 1 {
		t.Fatalf("drainable pages = %d, want 1", drainable)
	}
}

func TestPageAwareEvictionCheapestPageFirst(t *testing.T) {
	tree, err := NewWithEvictionStrategy(15, "page-aware")
	if err != nil {
		t.Fatalf("NewWithEvictionStrategy(page-aware): %v", err)
	}

	// Chunk 1 has two leaves of 5 tokens each (cost = 2 leaves to drain)
	// Chunk 2 has one leaf of 5 tokens (cost = 1 leaf to drain)
	c1a, c1b := distinctReq(0, 5), distinctReq(1, 5)
	c2 := distinctReq(2, 5)
	c3 := distinctReq(3, 5)

	root := tree.root
	l1a := tree.InsertWithChunkID(root, c1a, nil, 1)
	tree.Done(l1a)
	l1b := tree.InsertWithChunkID(root, c1b, nil, 1)
	tree.Done(l1b)

	l2 := tree.InsertWithChunkID(root, c2, nil, 2)
	tree.Done(l2)

	// Now tree has 15 tokens. Inserting c3 (5 tokens) requires evicting 5 tokens.
	// Chunk 2 requires only 1 leaf eviction to fully drain (cost=1).
	// Chunk 1 requires 2 leaf evictions to fully drain (cost=2).
	// Therefore, l2 on Chunk 2 must be chosen as victim first!
	l3 := tree.InsertWithChunkID(root, c3, nil, 3)
	tree.Done(l3)

	if m := tree.MatchLen(c2); m != 0 {
		t.Fatalf("cheaper-to-drain Chunk 2 leaf should be evicted first, matched %d", m)
	}
	if m := tree.MatchLen(c1a); m != len(c1a) {
		t.Fatalf("Chunk 1 leaf c1a should survive, matched %d/%d", m, len(c1a))
	}
	if m := tree.MatchLen(c1b); m != len(c1b) {
		t.Fatalf("Chunk 1 leaf c1b should survive, matched %d/%d", m, len(c1b))
	}
}

// TestEvictionStrategyFIFOFILOVictimOrder exercises the ordinary Insert eviction
// seam: creation order is independent of touches, and in-flight leaves stay leased.
// fak-test:runtime fast est=10ms lane=default
func TestEvictionStrategyFIFOFILOVictimOrder(t *testing.T) {
	for _, tc := range []struct {
		name          string
		strategy      string
		first, second int
		origin        string
	}{
		{"fifo", "fifo", 0, 1, ""},
		{"filo", "filo", 2, 1, ""},
		{"default_lru", "", 2, 0, ""},
		{"fifo_flights", "fifo", 0, 1, "flight"},
		{"filo_flights", "filo", 2, 1, "flight"},
		{"filo_unpinned", "filo", 2, 1, "pin"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tree := New(30)
			if tc.strategy != "" {
				if err := tree.SetEvictionStrategy(tc.strategy); err != nil {
					t.Fatal(err)
				}
			}
			reqs := [][]int{distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)}
			for i, req := range reqs {
				if tc.origin == "flight" && i != 1 {
					// The first demand flight is created at access clock zero.
					leaf, leader := tree.RegisterFlight(req)
					if !leader || leaf == nil {
						t.Fatal("expected new flight")
					}
					tree.CommitFlight(leaf, nil, nil)
					tree.Done(leaf)
				} else if tc.origin == "pin" && i != 1 {
					leaf := tree.PinPrefix(req)
					tree.UnpinNode(leaf)
				} else {
					touchPure(tree, req)
				}
			}
			for _, i := range []int{2, 0, 1} {
				if tc.origin != "" {
					break // preserve zero-clock flight/pin admission timestamps
				}

				n, matched := tree.Lookup(reqs[i])
				if matched != len(reqs[i]) {
					t.Fatalf("touch %d matched %d tokens", i, matched)
				}
				tree.Done(n)
			}
			// Keep new requests leased across both insert-time eviction passes so
			// the second victim must be another original A/B/C leaf, even for FILO.
			_, d := servePure(tree, distinctReq(3, 10))
			defer tree.Done(d)
			for i, req := range reqs {
				want := len(req)
				if i == tc.first {
					want = 0
				}
				if got := tree.MatchLen(req); got != want {
					t.Fatalf("first eviction: leaf %d matched %d, want %d", i, got, want)
				}
			}
			_, e := servePure(tree, distinctReq(4, 10))
			defer tree.Done(e)
			for i, req := range reqs {
				want := len(req)
				if i == tc.first || i == tc.second {
					want = 0
				}
				if got := tree.MatchLen(req); got != want {
					t.Fatalf("second eviction: leaf %d matched %d, want %d", i, got, want)
				}
			}
			wantPolicy := tc.strategy
			if wantPolicy == "" {
				wantPolicy = "lru"
			}
			if st := tree.Stats(); st.Evictions != 2 || st.LastEvictPolicy != wantPolicy {
				t.Fatalf("evictions/policy = %d/%s, want 2/%s", st.Evictions, st.LastEvictPolicy, wantPolicy)
			}
		})
	}
	for _, strategy := range []string{"fifo", "filo"} {
		t.Run(strategy+"_leased", func(t *testing.T) {
			tree, err := NewWithEvictionStrategy(30, strategy)
			if err != nil {
				t.Fatal(err)
			}
			reqs := [][]int{distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)}
			for _, req := range reqs {
				touchPure(tree, req)
			}
			protected := 0
			if strategy == "filo" {
				protected = 2
			}
			lease, _ := tree.Lookup(reqs[protected])
			defer tree.Done(lease)
			_, added := servePure(tree, distinctReq(3, 10))
			defer tree.Done(added)
			if tree.MatchLen(reqs[protected]) != 10 || tree.MatchLen(reqs[1]) != 0 {
				t.Fatal("leased preferred victim must survive; next eligible creation-order victim must leave")
			}
		})
		t.Run(strategy+"_warm_full_pool", func(t *testing.T) {
			tree, err := NewWithEvictionStrategy(20, strategy)
			if err != nil {
				t.Fatal(err)
			}
			a, b, warm := distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)
			touchPure(tree, a)
			touchPure(tree, b)
			tree.WarmInsert(warm, nil)
			if tree.MatchLen(a) != 10 || tree.MatchLen(b) != 10 || tree.MatchLen(warm) != 0 {
				t.Fatal("opportunistic warm insertion displaced demand from a full pool")
			}
		})
		t.Run(strategy+"_warm", func(t *testing.T) {
			tree, err := NewWithEvictionStrategy(0, strategy)
			if err != nil {
				t.Fatal(err)
			}
			reqs := [][]int{distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)}
			for _, req := range reqs {
				tree.WarmInsert(req, nil)
			}
			// Warm inserts share recency zero; creation order must still be unique.
			tree.SetRetention(20)
			victim := 0
			if strategy == "filo" {
				victim = 2
			}
			for i, req := range reqs {
				want := 10
				if i == victim {
					want = 0
				}
				if got := tree.MatchLen(req); got != want {
					t.Fatalf("warm leaf %d matched %d, want %d", i, got, want)
				}
			}
		})
	}
}
