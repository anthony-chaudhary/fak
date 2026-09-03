package radixkv

import (
	"sort"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// eviction_page_aware.go implements page-aligned victim selection to reduce
// physical memory fragmentation under multi-agent tree branching (Rank 6 / Issue #10721).
//
// Background & Motivation:
// In LLM serving engines with large layer counts (e.g. 36 layers), KV cache memory
// is backed by physical 2MB pages or unified GPU slabs. A physical page cannot be
// returned to the OS or GPU allocator as long as a single block in that page is still
// retained (e.g. 1 block keeping 144MB of VRAM pinned across 36 layers).
//
// When multi-agent workloads branch and diverge, conventional scalar eviction policies
// (LRU, SLRU, CostAware) select victims based solely on age, hits, or recompute cost.
// Under scattered access patterns, this evicts blocks across disjoint pages without
// completely draining any of them, leaving underlying physical allocations pinned
// (phantom residency).
//
// Page-Aligned Victim Selection (borrowed from kvcached @60cad949, Apache-2.0):
// 1. Group eviction candidates by physical chunk/page ID (ChunkID).
// 2. Identify drainable pages: physical chunks with zero pinned nodes (refs == 0)
//    that can be completely freed in this round.
// 3. Prioritize draining whole physical chunks first (Tier 0 over Tier 1).
// 4. Sort drainable chunks cheapest-page-first: chunks requiring the fewest leaf
//    evictions to reach zero occupancy are drained first.
// 5. Break ties within pages (or between equal-cost pages) using LRU recency.

// PageOccupancyTracker resolves physical chunk or page IDs for cache nodes.
// Implementations map nodes to backing 2MB GPU pages, host DRAM slabs, or block tables.
type PageOccupancyTracker interface {
	ChunkID(n *node) int
}

// PageOccupancyTrackerFunc is a functional adapter for PageOccupancyTracker.
type PageOccupancyTrackerFunc func(n *node) int

// ChunkID calls f(n).
func (f PageOccupancyTrackerFunc) ChunkID(n *node) int {
	return f(n)
}

// PageOccupancyStats captures the residency and drainability metrics for a physical chunk/page.
type PageOccupancyStats struct {
	ChunkID         int    `json:"chunk_id"`
	TotalNodes      int    `json:"total_nodes"`
	EvictableLeaves int    `json:"evictable_leaves"`
	PinnedNodes     int    `json:"pinned_nodes"`
	Drainable       bool   `json:"drainable"`
	OldestLastUsed  uint64 `json:"oldest_last_used"`
}

type chunkSummary struct {
	totalNodes      int
	evictableLeaves int
	pinnedCount     int
	oldestLastUsed  uint64
}

// PageAwareEvictionStrategy implements page-aligned victim selection.
type PageAwareEvictionStrategy struct {
	mu        sync.RWMutex
	tracker   PageOccupancyTracker
	fallback  VictimStrategy
	tree      *Tree
	occupancy map[int]chunkSummary
}

// PageAwareStrategy is an alias for PageAwareEvictionStrategy.
type PageAwareStrategy = PageAwareEvictionStrategy

// NewPageAwareStrategy constructs a page-aware eviction strategy with the given tracker.
func NewPageAwareStrategy(tracker PageOccupancyTracker) *PageAwareEvictionStrategy {
	return &PageAwareEvictionStrategy{
		tracker:   tracker,
		occupancy: make(map[int]chunkSummary),
	}
}

// NewPageAwareStrategyWithFallback constructs a page-aware eviction strategy with a fallback strategy.
func NewPageAwareStrategyWithFallback(tracker PageOccupancyTracker, fallback VictimStrategy) *PageAwareEvictionStrategy {
	return &PageAwareEvictionStrategy{
		tracker:   tracker,
		fallback:  fallback,
		occupancy: make(map[int]chunkSummary),
	}
}

// Name reports the strategy's registry key.
func (s *PageAwareEvictionStrategy) Name() string { return "page-aware" }

// SetTracker updates the strategy's physical page occupancy tracker.
func (s *PageAwareEvictionStrategy) SetTracker(tracker PageOccupancyTracker) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tracker = tracker
	if s.tree != nil {
		s.prepareTreeLocked(s.tree)
	}
}

func (s *PageAwareEvictionStrategy) bindTree(t *Tree) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tree = t
	if s.tracker == nil && t != nil && t.pageTracker != nil {
		s.tracker = t.pageTracker
	}
}

// PrepareTree precomputes chunk occupancy across the tree for victim selection.
func (s *PageAwareEvictionStrategy) PrepareTree(t *Tree) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.prepareTreeLocked(t)
}

func (s *PageAwareEvictionStrategy) prepareTreeLocked(t *Tree) {
	s.tree = t
	if s.occupancy == nil {
		s.occupancy = make(map[int]chunkSummary)
	} else {
		clear(s.occupancy)
	}
	if t == nil {
		return
	}
	var stack []*node
	t.forEachRoot(func(r *node) {
		for _, c := range r.children {
			stack = append(stack, c)
		}
	})
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		cid := s.nodeChunkIDLocked(n)
		if cid > 0 {
			st := s.occupancy[cid]
			st.totalNodes++
			if n.refs > 0 {
				st.pinnedCount++
			}
			if len(n.children) == 0 {
				if n.refs == 0 {
					st.evictableLeaves++
					if st.oldestLastUsed == 0 || n.lastUsed < st.oldestLastUsed {
						st.oldestLastUsed = n.lastUsed
					}
				}
			}
			s.occupancy[cid] = st
		}
		for _, c := range n.children {
			stack = append(stack, c)
		}
	}
}

func (s *PageAwareEvictionStrategy) prepareFromNodeLocked(n *node) {
	if n == nil {
		return
	}
	r := n
	for r.parent != nil {
		r = r.parent
	}
	if s.occupancy == nil {
		s.occupancy = make(map[int]chunkSummary)
	} else {
		clear(s.occupancy)
	}
	var stack []*node
	for _, c := range r.children {
		stack = append(stack, c)
	}
	for len(stack) > 0 {
		curr := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		cid := s.nodeChunkIDLocked(curr)
		if cid > 0 {
			st := s.occupancy[cid]
			st.totalNodes++
			if curr.refs > 0 {
				st.pinnedCount++
			}
			if len(curr.children) == 0 {
				if curr.refs == 0 {
					st.evictableLeaves++
					if st.oldestLastUsed == 0 || curr.lastUsed < st.oldestLastUsed {
						st.oldestLastUsed = curr.lastUsed
					}
				}
			}
			s.occupancy[cid] = st
		}
		for _, c := range curr.children {
			stack = append(stack, c)
		}
	}
}

func (s *PageAwareEvictionStrategy) nodeChunkIDLocked(n *node) int {
	if n == nil {
		return 0
	}
	if s.tracker != nil {
		if id := s.tracker.ChunkID(n); id != 0 {
			return id
		}
	}
	if s.tree != nil && s.tree.pageTracker != nil {
		if id := s.tree.pageTracker.ChunkID(n); id != 0 {
			return id
		}
	}
	return n.chunkID
}

func (s *PageAwareEvictionStrategy) fallbackPriority(n *node) victimKey {
	if s.fallback != nil {
		return s.fallback.Priority(n)
	}
	return victimKey{age: n.lastUsed}
}

// Priority scores node n using page-aware metrics.
//
// Segments:
//   - seg 0: Drainable whole chunk (all resident nodes are evictable leaves, zero pinned blocks).
//     Cost is evictableLeaves (cheapest-page-first), tie-break is LRU age.
//   - seg 1: Partially pinned / non-drainable chunk (some blocks are pinned or non-leaves).
//     Penalized so that drainable chunks empty first.
//   - seg 2: Unassigned chunk (chunkID <= 0).
func (s *PageAwareEvictionStrategy) Priority(n *node) victimKey {
	if n == nil {
		return victimKey{}
	}
	s.mu.RLock()
	cid := s.nodeChunkIDLocked(n)
	st, ok := s.occupancy[cid]
	hasOccupancy := len(s.occupancy) > 0
	s.mu.RUnlock()

	if !ok || !hasOccupancy {
		s.mu.Lock()
		if s.tree != nil {
			s.prepareTreeLocked(s.tree)
		} else {
			s.prepareFromNodeLocked(n)
		}
		cid = s.nodeChunkIDLocked(n)
		st = s.occupancy[cid]
		s.mu.Unlock()
	}

	fbKey := s.fallbackPriority(n)
	if cid <= 0 {
		return victimKey{seg: 2, cost: fbKey.cost, age: fbKey.age}
	}

	// Tier 0: Drainable whole chunk.
	// All resident nodes in this physical chunk are evictable candidate leaves
	// (totalNodes == evictableLeaves and pinnedCount == 0).
	// Cost is evictableLeaves: the fewest evictions needed to completely empty the chunk
	// (cheapest-page-first). LRU tie-breaking within pages.
	if st.pinnedCount == 0 && st.totalNodes == st.evictableLeaves && st.evictableLeaves > 0 {
		return victimKey{
			seg:  0,
			cost: float64(st.evictableLeaves),
			age:  fbKey.age,
		}
	}

	// Tier 1: Partially pinned / non-drainable chunk.
	// Cannot be completely emptied in this pass because some nodes are pinned (refs > 0)
	// or non-leaves. Penalize pinned chunks so drainable chunks evict first.
	cost := float64(st.totalNodes)
	if st.pinnedCount > 0 {
		cost += float64(st.pinnedCount * 1000)
	}
	return victimKey{
		seg:  1,
		cost: cost,
		age:  fbKey.age,
	}
}

// GroupCandidatesByChunk partitions evictable candidate leaves by their backing chunk/page ID.
func (s *PageAwareEvictionStrategy) GroupCandidatesByChunk(candidates []*node) map[int][]*node {
	s.mu.RLock()
	defer s.mu.RUnlock()
	groups := make(map[int][]*node)
	for _, c := range candidates {
		cid := s.nodeChunkIDLocked(c)
		groups[cid] = append(groups[cid], c)
	}
	return groups
}

// SelectVictim selects the lowest-priority (first to evict) candidate leaf among candidates.
func (s *PageAwareEvictionStrategy) SelectVictim(candidates []*node) *node {
	if len(candidates) == 0 {
		return nil
	}
	var best *node
	var bestKey victimKey
	for _, c := range candidates {
		k := s.Priority(c)
		if best == nil || k.less(bestKey) {
			best, bestKey = c, k
		}
	}
	return best
}

// RankChunks returns the chunk IDs sorted by eviction preference (drainable cheapest-first, then LRU).
func (s *PageAwareEvictionStrategy) RankChunks(candidates []*node) []int {
	groups := s.GroupCandidatesByChunk(candidates)
	chunkIDs := make([]int, 0, len(groups))
	for cid := range groups {
		chunkIDs = append(chunkIDs, cid)
	}
	sort.Slice(chunkIDs, func(i, j int) bool {
		c1, c2 := chunkIDs[i], chunkIDs[j]
		leaves1, leaves2 := groups[c1], groups[c2]
		if len(leaves1) == 0 || len(leaves2) == 0 {
			return len(leaves1) > 0
		}
		k1 := s.Priority(leaves1[0])
		k2 := s.Priority(leaves2[0])
		return k1.less(k2)
	})
	return chunkIDs
}

// SetPageOccupancyTracker registers a tracker to resolve physical chunk/page IDs.
func (t *Tree) SetPageOccupancyTracker(tracker PageOccupancyTracker) {
	t.pageTracker = tracker
	if s, ok := t.strategy.(interface{ SetTracker(PageOccupancyTracker) }); ok {
		s.SetTracker(tracker)
	}
}

// PageOccupancyTracker returns the tree's registered page occupancy tracker, if any.
func (t *Tree) PageOccupancyTracker() PageOccupancyTracker {
	return t.pageTracker
}

// SetNodeChunkID assigns the physical backing chunk or page ID for node n.
func (t *Tree) SetNodeChunkID(n *node, chunkID int) {
	if n != nil {
		n.chunkID = chunkID
	}
}

// SetStrategy sets the tree's victim rule directly to the provided VictimStrategy.
func (t *Tree) SetStrategy(s VictimStrategy) {
	if s == nil {
		t.strategy = lruStrategy{}
		t.policy = EvictionLRU
		t.SetAdmissionEnabled(false)
		return
	}
	t.strategy = s
	if tb, ok := s.(interface{ bindTree(*Tree) }); ok {
		tb.bindTree(t)
	}
	if t.pageTracker != nil {
		if ts, ok := s.(interface{ SetTracker(PageOccupancyTracker) }); ok {
			ts.SetTracker(t.pageTracker)
		}
	}
	switch s.Name() {
	case "cost-aware", "lowest-score-first":
		t.policy = EvictionCostAware
		t.SetAdmissionEnabled(true)
	case "page-aware", "page-aligned", "chunk-aware":
		t.policy = EvictionPageAware
		t.SetAdmissionEnabled(false)
	default:
		t.policy = EvictionLRU
		t.SetAdmissionEnabled(false)
	}
}

// InsertWithChunkID attaches the request's suffix as a new child of boundary,
// tagged with the backing physical chunk/page ID.
func (t *Tree) InsertWithChunkID(boundary *node, suffix []int, kv *model.KVCache, chunkID int) *node {
	return t.InsertWithLogitsAndChunk(boundary, suffix, kv, nil, chunkID)
}

// InsertWithLogitsAndChunk is InsertWithChunkID plus an optional exact-prefix logits payload.
func (t *Tree) InsertWithLogitsAndChunk(boundary *node, suffix []int, kv *model.KVCache, logits []float32, chunkID int) *node {
	n, _ := t.insertWithLogits(boundary, suffix, kv, logits, nil)
	if n != nil && chunkID > 0 {
		n.SetChunkID(chunkID)
	}
	return n
}

func (t *Tree) nodeChunkID(n *node) int {
	if n == nil {
		return 0
	}
	if t.pageTracker != nil {
		if cid := t.pageTracker.ChunkID(n); cid > 0 {
			return cid
		}
	}
	return n.chunkID
}

// ChunkOccupancy inspects the tree and returns the occupancy stats for all physical chunks.
func (t *Tree) ChunkOccupancy() map[int]PageOccupancyStats {
	occupancy := make(map[int]PageOccupancyStats)
	var stack []*node
	t.forEachRoot(func(r *node) {
		for _, c := range r.children {
			stack = append(stack, c)
		}
	})
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		cid := t.nodeChunkID(n)
		if cid > 0 {
			st := occupancy[cid]
			st.ChunkID = cid
			st.TotalNodes++
			if n.refs > 0 {
				st.PinnedNodes++
			}
			if len(n.children) == 0 {
				if n.refs == 0 {
					st.EvictableLeaves++
					if st.OldestLastUsed == 0 || n.lastUsed < st.OldestLastUsed {
						st.OldestLastUsed = n.lastUsed
					}
				}
			}
			st.Drainable = (st.PinnedNodes == 0 && st.TotalNodes == st.EvictableLeaves && st.EvictableLeaves > 0)
			occupancy[cid] = st
		}
		for _, c := range n.children {
			stack = append(stack, c)
		}
	}
	return occupancy
}

// PhysicalPageStats returns the active allocated physical pages, fully drainable pages,
// and fragmented pages in the cache.
func (t *Tree) PhysicalPageStats() (activePages, drainablePages, fragmentedPages int) {
	occ := t.ChunkOccupancy()
	activePages = len(occ)
	for _, st := range occ {
		if st.Drainable {
			drainablePages++
		} else if st.TotalNodes > 0 {
			fragmentedPages++
		}
	}
	return activePages, drainablePages, fragmentedPages
}

// LeavesForChunk returns all current leaf nodes belonging to the given physical chunk ID.
func (t *Tree) LeavesForChunk(chunkID int) []*node {
	var leaves []*node
	var stack []*node
	t.forEachRoot(func(r *node) {
		for _, c := range r.children {
			stack = append(stack, c)
		}
	})
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if len(n.children) == 0 && t.nodeChunkID(n) == chunkID {
			leaves = append(leaves, n)
		}
		for _, c := range n.children {
			stack = append(stack, c)
		}
	}
	return leaves
}
