package radixkv

import (
	"testing"
)

func TestPageAwareEvictionDrainableOverPinned(t *testing.T) {
	// Chunk 1 has 1 leaf, but is pinned (refs > 0).
	// Chunk 2 has 1 leaf, unpinned (refs == 0).
	// Page-aware eviction must evict Chunk 2 first, because Chunk 1 cannot be emptied.
	tree, err := NewWithEvictionStrategy(20, "page-aware")
	if err != nil {
		t.Fatalf("NewWithEvictionStrategy: %v", err)
	}

	a, b := distinctReq(0, 10), distinctReq(1, 10)
	leafA := tree.InsertWithChunkID(tree.root, a, nil, 1) // pinned (refs=1)
	leafB := tree.InsertWithChunkID(tree.root, b, nil, 2)
	tree.Done(leafB) // unpinned

	// Make Leaf A older than Leaf B. Under pure LRU, Leaf A would be selected (if unpinned).
	leafA.lastUsed = 10
	leafB.lastUsed = 50

	victim := tree.victimLeaf()
	if victim != leafB {
		t.Fatalf("victim = %p (chunk %d), want leafB %p (chunk 2)", victim, victim.chunkID, leafB)
	}

	tree.Done(leafA)
}

func TestPageAwareEvictionCheapestPageFirstMulti(t *testing.T) {
	// Page 1 has 1 leaf (unpinned). Needs 1 eviction to drain completely.
	// Page 2 has 3 leaves (unpinned). Needs 3 evictions to drain completely.
	// Even if Page 2's leaves are older, Page 1's leaf should be evicted first.
	tree, err := NewWithEvictionStrategy(40, "page-aware")
	if err != nil {
		t.Fatalf("NewWithEvictionStrategy: %v", err)
	}

	p1_leaf := tree.InsertWithChunkID(tree.root, distinctReq(1, 10), nil, 1)
	tree.Done(p1_leaf)
	p1_leaf.lastUsed = 100 // newer

	p2_leaf1 := tree.InsertWithChunkID(tree.root, distinctReq(2, 10), nil, 2)
	tree.Done(p2_leaf1)
	p2_leaf1.lastUsed = 10 // older

	p2_leaf2 := tree.InsertWithChunkID(tree.root, distinctReq(3, 10), nil, 2)
	tree.Done(p2_leaf2)
	p2_leaf2.lastUsed = 20

	p2_leaf3 := tree.InsertWithChunkID(tree.root, distinctReq(4, 10), nil, 2)
	tree.Done(p2_leaf3)
	p2_leaf3.lastUsed = 30

	victim := tree.victimLeaf()
	if victim != p1_leaf {
		t.Fatalf("victim = %p (chunk %d), want p1_leaf %p (chunk 1)", victim, victim.chunkID, p1_leaf)
	}

	// Trigger eviction by setting budget to 30.
	tree.SetRetention(30)

	// Page 1 should be completely drained (0 nodes).
	// Page 2 should still have all 3 nodes.
	if m := tree.MatchLen(distinctReq(1, 10)); m != 0 {
		t.Fatalf("Page 1 leaf should be evicted, matched %d", m)
	}
	if m := tree.MatchLen(distinctReq(2, 10)); m != 10 {
		t.Fatalf("Page 2 leaf1 should survive, matched %d", m)
	}

	active, drainable, _ := tree.PhysicalPageStats()
	if active != 1 {
		t.Fatalf("active pages = %d, want 1 (Page 2)", active)
	}
	if drainable != 1 {
		t.Fatalf("drainable pages = %d, want 1", drainable)
	}
}

func TestPageAwareEvictionLruTieBreakerWithinPage(t *testing.T) {
	// When two leaves belong to the same page, the older lastUsed is evicted first.
	tree, err := NewWithEvictionStrategy(30, "page-aware")
	if err != nil {
		t.Fatalf("NewWithEvictionStrategy: %v", err)
	}

	l1 := tree.InsertWithChunkID(tree.root, distinctReq(1, 10), nil, 1)
	tree.Done(l1)
	l1.lastUsed = 20 // older

	l2 := tree.InsertWithChunkID(tree.root, distinctReq(2, 10), nil, 1)
	tree.Done(l2)
	l2.lastUsed = 80 // newer

	victim := tree.victimLeaf()
	if victim != l1 {
		t.Fatalf("victim = %p (age %d), want l1 %p (age %d)", victim, victim.lastUsed, l1, l1.lastUsed)
	}
}

func TestPageAwareEvictionConsecutiveDraining(t *testing.T) {
	// Chunk 1 has 2 leaves: L1a, L1b.
	// Chunk 2 has 2 leaves: L2a, L2b.
	// Chunk 3 has 2 leaves: L3a, L3b.
	// All leaves are 10 tokens each (total 60 tokens).
	// Budget is reduced from 60 to 40 (must evict 2 leaves / 20 tokens).
	// Under Page-Aware eviction:
	// Once the first leaf from a chunk is evicted, that chunk's remaining cost drops to 1.
	// The next eviction MUST pick the second leaf from that SAME chunk,
	// completely draining that physical chunk!
	tree, err := NewWithEvictionStrategy(60, "page-aware")
	if err != nil {
		t.Fatalf("NewWithEvictionStrategy: %v", err)
	}

	r1a := distinctReq(1, 10)
	r1b := distinctReq(2, 10)
	r2a := distinctReq(3, 10)
	r2b := distinctReq(4, 10)
	r3a := distinctReq(5, 10)
	r3b := distinctReq(6, 10)

	l1a := tree.InsertWithChunkID(tree.root, r1a, nil, 1)
	tree.Done(l1a)
	l1a.lastUsed = 10 // oldest overall

	l1b := tree.InsertWithChunkID(tree.root, r1b, nil, 1)
	tree.Done(l1b)
	l1b.lastUsed = 90 // newer than some other leaves!

	l2a := tree.InsertWithChunkID(tree.root, r2a, nil, 2)
	tree.Done(l2a)
	l2a.lastUsed = 30

	l2b := tree.InsertWithChunkID(tree.root, r2b, nil, 2)
	tree.Done(l2b)
	l2b.lastUsed = 40

	l3a := tree.InsertWithChunkID(tree.root, r3a, nil, 3)
	tree.Done(l3a)
	l3a.lastUsed = 50

	l3b := tree.InsertWithChunkID(tree.root, r3b, nil, 3)
	tree.Done(l3b)
	l3b.lastUsed = 60

	// Reduce budget to 40 (triggers 2 evictions).
	tree.SetRetention(40)

	st := tree.Stats()
	if st.Evictions != 2 {
		t.Fatalf("evictions = %d, want 2", st.Evictions)
	}
	if st.PageEvictions != 2 {
		t.Fatalf("pageEvictions = %d, want 2", st.PageEvictions)
	}

	// Chunk 1 should be COMPLETELY drained!
	// Both r1a and r1b must be evicted, even though r1b (age 90) was newer than r2a (age 30).
	if m := tree.MatchLen(r1a); m != 0 {
		t.Fatalf("r1a should be evicted, matched %d", m)
	}
	if m := tree.MatchLen(r1b); m != 0 {
		t.Fatalf("r1b should be evicted to complete chunk 1 drain, matched %d", m)
	}

	// Chunk 2 and Chunk 3 should be 100% intact!
	if m := tree.MatchLen(r2a); m != 10 {
		t.Fatalf("r2a should survive, matched %d", m)
	}
	if m := tree.MatchLen(r2b); m != 10 {
		t.Fatalf("r2b should survive, matched %d", m)
	}
	if m := tree.MatchLen(r3a); m != 10 {
		t.Fatalf("r3a should survive, matched %d", m)
	}
	if m := tree.MatchLen(r3b); m != 10 {
		t.Fatalf("r3b should survive, matched %d", m)
	}

	active, drainable, fragmented := tree.PhysicalPageStats()
	if active != 2 {
		t.Fatalf("active pages = %d, want 2 (chunks 2 and 3; chunk 1 was freed)", active)
	}
	if drainable != 2 {
		t.Fatalf("drainable pages = %d, want 2", drainable)
	}
	if fragmented != 0 {
		t.Fatalf("fragmented pages = %d, want 0", fragmented)
	}
}

func TestPageAwareEvictionWithPageOccupancyTracker(t *testing.T) {
	// Test setting an external PageOccupancyTracker.
	tree := New(30)

	// Tracker associates odd plen to chunk 1, even plen to chunk 2.
	tracker := PageOccupancyTrackerFunc(func(n *node) int {
		if n.Plen()%2 == 1 {
			return 1
		}
		return 2
	})
	tree.SetPageOccupancyTracker(tracker)
	if err := tree.SetEvictionStrategy("page-aware"); err != nil {
		t.Fatalf("SetEvictionStrategy: %v", err)
	}

	if tree.PageOccupancyTracker() == nil {
		t.Fatal("PageOccupancyTracker() returned nil")
	}

	// Insert nodes with different lengths.
	n1 := tree.Insert(tree.root, []int{1, 2, 3}, nil) // plen 3 -> chunk 1
	tree.Done(n1)

	n2 := tree.Insert(tree.root, []int{4, 5, 6, 7}, nil) // plen 4 -> chunk 2
	tree.Done(n2)

	occ := tree.ChunkOccupancy()
	if len(occ) != 2 {
		t.Fatalf("expected 2 chunks in occupancy map, got %d", len(occ))
	}
	if occ[1].TotalNodes != 1 || occ[2].TotalNodes != 1 {
		t.Fatalf("unexpected chunk occupancy: %+v", occ)
	}
}

func TestPageAwareEvictionPhysicalFragmentationReduction(t *testing.T) {
	// Multi-agent branched prefix simulation (kvcached fragmentation benchmark replica).
	// We simulate 10 physical pages (chunks 1..10).
	// Each page holds 4 leaf blocks (40 leaves total, 10 tokens each = 400 tokens).
	// Under memory pressure, we evict 10 leaves (100 tokens).
	//
	// In Pure LRU:
	// If the leaves' ages alternate across pages, LRU evicts 1 leaf from each of the 10 pages.
	// As a result, every single page still retains 3 leaves!
	// Active pages remaining = 10 / 10 (0% physical memory freed!).
	//
	// In Page-Aware Eviction:
	// Page-Aware focuses eviction on completing whole pages.
	// It drains Page 1 (4 leaves), Page 2 (4 leaves), and Page 3 (2 leaves).
	// At least 2 full pages are 100% freed!
	// Active pages drops to 8 / 10 (20% physical memory freed!).

	// 1. Run Pure LRU
	lru := New(400)
	for page := 1; page <= 10; page++ {
		for block := 0; block <= 3; block++ {
			req := distinctReq(page*10+block, 10)
			leaf := lru.InsertWithChunkID(lru.root, req, nil, page)
			lru.Done(leaf)
			// Stagger ages so LRU evicts 1 block from each page:
			// page 1 block 0 age 1, page 2 block 0 age 2, ..., page 10 block 0 age 10.
			leaf.lastUsed = uint64(block*100 + page)
		}
	}
	lru.SetRetention(300) // evict 100 tokens (10 leaves)

	lruActive, _, _ := lru.PhysicalPageStats()
	if lruActive != 10 {
		t.Fatalf("Pure LRU active pages = %d, expected 10 (all 10 pages remain pinned with fragmented blocks)", lruActive)
	}

	// 2. Run Page-Aware Eviction on identical workload
	pageAware, err := NewWithEvictionStrategy(400, "page-aware")
	if err != nil {
		t.Fatalf("NewWithEvictionStrategy: %v", err)
	}
	for page := 1; page <= 10; page++ {
		for block := 0; block <= 3; block++ {
			req := distinctReq(page*10+block, 10)
			leaf := pageAware.InsertWithChunkID(pageAware.root, req, nil, page)
			pageAware.Done(leaf)
			leaf.lastUsed = uint64(block*100 + page)
		}
	}
	pageAware.SetRetention(300) // evict 100 tokens (10 leaves)

	paActive, _, _ := pageAware.PhysicalPageStats()
	if paActive > 8 {
		t.Fatalf("PageAware active pages = %d, want <= 8 (at least 2 whole pages completely freed)", paActive)
	}

	// Verify that the number of active physical pages is strictly less under Page-Aware!
	if paActive >= lruActive {
		t.Fatalf("PageAware active pages (%d) must be strictly less than LRU active pages (%d)", paActive, lruActive)
	}

	t.Logf("Fragmentation resistance verified: Pure LRU active pages = %d (0 freed), Page-Aware active pages = %d (%d freed)",
		lruActive, paActive, 10-paActive)
}

func TestPageAwareBackwardCompatibility(t *testing.T) {
	// Test that all legacy and new strategies are registered and resolve correctly.
	strategies := []string{
		"lru",
		"cost-aware",
		"lowest-score-first",
		"slru",
		"page-aware",
		"page-aligned",
		"chunk-aware",
	}

	for _, name := range strategies {
		tree, err := NewWithEvictionStrategy(100, name)
		if err != nil {
			t.Fatalf("NewWithEvictionStrategy(%q) failed: %v", name, err)
		}
		policyName := tree.Stats().EvictionPolicy
		if policyName == "" {
			t.Fatalf("strategy %q returned empty EvictionPolicy name", name)
		}
	}

	// Test fail closed on unknown strategy.
	if _, err := NewWithEvictionStrategy(100, "nonexistent-policy"); err == nil {
		t.Fatal("expected error on unknown eviction strategy, got nil")
	}
}

func TestPageAwareGroupAndRankCandidates(t *testing.T) {
	tree, err := NewWithEvictionStrategy(50, "page-aware")
	if err != nil {
		t.Fatalf("NewWithEvictionStrategy: %v", err)
	}

	strat, ok := tree.strategy.(*PageAwareEvictionStrategy)
	if !ok {
		t.Fatalf("expected *PageAwareEvictionStrategy, got %T", tree.strategy)
	}

	l1 := tree.InsertWithChunkID(tree.root, distinctReq(1, 10), nil, 10)
	tree.Done(l1)
	l2 := tree.InsertWithChunkID(tree.root, distinctReq(2, 10), nil, 10)
	tree.Done(l2)
	l3 := tree.InsertWithChunkID(tree.root, distinctReq(3, 10), nil, 20)
	tree.Done(l3)

	candidates := []*node{l1, l2, l3}
	groups := strat.GroupCandidatesByChunk(candidates)
	if len(groups) != 2 {
		t.Fatalf("expected 2 chunk groups, got %d", len(groups))
	}
	if len(groups[10]) != 2 || len(groups[20]) != 1 {
		t.Fatalf("unexpected chunk grouping: %+v", groups)
	}

	strat.PrepareTree(tree)
	ranked := strat.RankChunks(candidates)
	// Chunk 20 has 1 leaf (cheapest to drain), so it should rank first before chunk 10 (2 leaves).
	if len(ranked) != 2 || ranked[0] != 20 || ranked[1] != 10 {
		t.Fatalf("expected ranked chunks [20, 10], got %+v", ranked)
	}

	victim := strat.SelectVictim(candidates)
	if victim != l3 {
		t.Fatalf("SelectVictim = %p (chunk %d), want l3 %p (chunk 20)", victim, victim.chunkID, l3)
	}
}

func BenchmarkPageAwareEviction(b *testing.B) {
	tree, err := NewWithEvictionStrategy(2000, "page-aware")
	if err != nil {
		b.Fatalf("NewWithEvictionStrategy: %v", err)
	}

	// Populate tree with 100 pages, 5 leaves each (500 leaves, 5000 tokens).
	for page := 1; page <= 100; page++ {
		for block := 0; block < 5; block++ {
			leaf := tree.InsertWithChunkID(tree.root, []int{page, block, 1, 2, 3}, nil, page)
			tree.Done(leaf)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := tree.victimLeaf()
		if v == nil {
			b.Fatal("nil victim")
		}
	}
}

func BenchmarkLRUEviction(b *testing.B) {
	tree := New(2000)

	for page := 1; page <= 100; page++ {
		for block := 0; block < 5; block++ {
			leaf := tree.InsertWithChunkID(tree.root, []int{page, block, 1, 2, 3}, nil, page)
			tree.Done(leaf)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := tree.victimLeaf()
		if v == nil {
			b.Fatal("nil victim")
		}
	}
}

func BenchmarkPageAwareChunkOccupancy(b *testing.B) {
	tree := New(5000)
	for page := 1; page <= 200; page++ {
		for block := 0; block < 5; block++ {
			leaf := tree.InsertWithChunkID(tree.root, []int{page, block, 1, 2, 3}, nil, page)
			tree.Done(leaf)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		occ := tree.ChunkOccupancy()
		if len(occ) == 0 {
			b.Fatal("empty occupancy")
		}
	}
}
