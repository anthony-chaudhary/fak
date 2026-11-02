// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Completion ownership adapted from vllm/v1/kv_offload/tiering/manager.py:281-358
// at bd5bbce6e45e09dbd07ac3e8c394626c059078f7; modified for Go synchronous
// snapshot acquisition: publish successful owners and release rejected owners.
//
// Budget-pressure eviction, retention protection, and policy subtree removal.
package radixkv

import (
	"container/heap"
	"errors"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// DemotionDecision indicates the chosen demotion action.
type DemotionDecision struct {
	Action string // "spill", "evict"
	Tier   string // "host", "disk"
}

// DemotionDecider is consulted by evictToBudget when demoteBeforeDrop is true (#3414).
type DemotionDecider interface {
	DecideDemotion(tokens int, hits int, bytes int64) DemotionDecision
}

// SetDemotionDecider configures a demotion decider on the tree.
func (t *Tree) SetDemotionDecider(d DemotionDecider) {
	t.demotionDecider = d
}

// EnableDemoteBeforeDrop toggles whether evictToBudget consults the demotion decider.
func (t *Tree) EnableDemoteBeforeDrop(enable bool) {
	t.demoteBeforeDrop = enable
}

// Demotions reports the number of victims successfully routed to a demotion fate.
func (t *Tree) Demotions() int {
	return t.demotions
}

func (t *Tree) SetEvictionHaltPredicate(p EvictionHaltPredicate) {
	t.haltPredicate = p
}

func (t *Tree) EvictionHaltSufficientCapacity() int64 {
	return t.evictionHaltSufficientCapacity
}

// evictToBudget evicts least-recently-used, unlocked LEAVES until the cached-token count
// is within budget — RadixAttention's eviction policy. Removing a leaf can make its parent
// a leaf, which the next iteration may then evict (the upward collapse). A node that is
// leased (refs>0) is never chosen, so a prefix being served survives memory pressure.
func (t *Tree) evictToBudget() {
	budget, evicts := t.resolveBudget()
	if !evicts {
		return // keep-all: eviction disabled
	}
	// Leave the common single-victim pass on its existing selector path, with
	// no heap allocation. Heap setup is attempted only if that eviction did not satisfy budget.
	var victims *lruVictimHeap
	heapTried := false
	for t.tokens > budget {
		if t.haltPredicate != nil && t.haltPredicate() {
			t.evictionHaltSufficientCapacity++
			break
		}
		var v *node
		if victims != nil {
			var ok bool
			v, ok = victims.victim(t)
			if !ok {
				victims = nil // exact ties retain the original rescan behavior
			}
		}
		if victims == nil {
			v = t.victimLeaf()
		}
		if v == nil {
			return // everything in budget-excess is locked; cannot evict further
		}
		if t.demoteBeforeDrop && t.demotionDecider != nil {
			footprintBytes := int64(len(v.key) * 4) // estimated footprint
			res := t.demotionDecider.DecideDemotion(len(v.key), v.hits, footprintBytes)
			if res.Action == "spill" {
				t.demotions++
			}
		}
		t.noteEviction(v) // thrash detector (#3393): remember the victim before it goes
		parent := v.parent
		t.removeLeaf(v)
		t.evictions++
		if t.policy == EvictionCostAware {
			t.costEvictions++
		}
		if t.policy == EvictionPageAware {
			t.pageEvictions++
		}
		if victims != nil {
			victims.addLeaf(t, parent) // a removal may expose a new parent victim
		} else if !heapTried && t.tokens > budget {
			heapTried = true
			// Custom strategies/preparers and user callbacks may change priorities
			// between victims. Keep their existing per-victim traversal untouched.
			if _, pureLRU := t.evictionStrategy().(lruStrategy); pureLRU && t.haltPredicate == nil && !(t.demoteBeforeDrop && t.demotionDecider != nil) {
				victims = t.buildLRUVictimHeap()
			}
		}
	}
}

// Multi-victim heap adapted from SGLang radix_cache.py evict, lines 582-607 at
// 00a9a81b67774e8374172646c092e14999472703 (Copyright 2023-2024 SGLang Team,
// Apache-2.0). Modifications: defer setup until after the first victim, restrict
// to pure LRU, preserve Fak's leaf immunity/tiers/telemetry and fall back on ties.
// https://github.com/sgl-project/sglang/blob/00a9a81b67774e8374172646c092e14999472703/python/sglang/srt/mem_cache/radix_cache.py
// SPDX-License-Identifier: Apache-2.0
// This heap lives for one serialized eviction pass; it is not a persistent index.
type lruHeapEntry struct {
	n   *node
	key victimKey
}

type lruVictimHeap struct {
	entries            []lruHeapEntry
	clock              uint64
	candidates, locked int
}

func (h lruVictimHeap) Len() int           { return len(h.entries) }
func (h lruVictimHeap) Less(i, j int) bool { return h.entries[i].key.less(h.entries[j].key) }
func (h lruVictimHeap) Swap(i, j int)      { h.entries[i], h.entries[j] = h.entries[j], h.entries[i] }
func (h *lruVictimHeap) Push(x any)        { h.entries = append(h.entries, x.(lruHeapEntry)) }
func (h *lruVictimHeap) Pop() any {
	n := len(h.entries) - 1
	x := h.entries[n]
	h.entries[n] = lruHeapEntry{}
	h.entries = h.entries[:n]
	return x
}

func (t *Tree) lruLeafKey(n *node) (victimKey, bool) {
	if n.refs > 0 || n.IsComputing() || t.isNodePinnedOrImmune(n) {
		return victimKey{}, false
	}
	k := (lruStrategy{}).Priority(n)
	if seg, ok := t.nodeTierSeg(n); ok {
		k.seg = seg
	}
	return k, k.seg < 3
}

func (t *Tree) buildLRUVictimHeap() *lruVictimHeap {
	h := &lruVictimHeap{clock: t.clock}
	var stack []*node
	t.forEachRoot(func(r *node) {
		for _, c := range r.children {
			stack = append(stack, c)
		}
	})
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if len(n.children) > 0 {
			for _, c := range n.children {
				stack = append(stack, c)
			}
			continue
		}
		h.candidates++
		if k, ok := t.lruLeafKey(n); ok {
			h.entries = append(h.entries, lruHeapEntry{n, k})
		} else {
			h.locked++
		}
	}
	heap.Init(h)
	return h
}

func (h *lruVictimHeap) victim(t *Tree) (*node, bool) {
	if t.clock != h.clock {
		return nil, false
	}
	if len(h.entries) == 0 {
		t.recordEvictChoice("lru", h.candidates, h.locked, nil)
		return nil, true
	}
	// Every equal-minimum descendant has an equal-minimum child on its path.
	// Detecting these two children is enough; do not replace the rescan's
	// first-seen tie behavior with a new arbitrary heap order (warm recency=0).
	k := h.entries[0].key
	if len(h.entries) > 1 && h.entries[1].key == k || len(h.entries) > 2 && h.entries[2].key == k {
		return nil, false
	}
	v := heap.Pop(h).(lruHeapEntry).n
	t.recordEvictChoice("lru", h.candidates, h.locked, v)
	h.candidates--
	return v, true
}

func (h *lruVictimHeap) addLeaf(t *Tree, n *node) {
	if n == nil || n.parent == nil || len(n.children) > 0 {
		return
	}
	h.candidates++
	if k, ok := t.lruLeafKey(n); ok {
		heap.Push(h, lruHeapEntry{n, k})
	} else {
		h.locked++
	}
}

// victimLeaf selects the budget-pressure victim: a single argmin over the active strategy's
// Priority key across every unlocked leaf — SGLang's heappush((strategy.get_priority(node),
// node)) collapsed to one pass. One traversal, one comparison rule; the tuple ordering in
// victimKey does the tiering, so lru / cost-aware / slru all share this loop with no
// per-policy branch (#3890). Leased leaves (refs>0) are counted as locked and skipped,
// never scored, so a prefix being served is never a victim. Returns nil when every
// budget-excess leaf is locked — the signal evictToBudget bails on.
func (t *Tree) victimLeaf() *node {
	return t.selectVictimLeaf(true)
}

func (t *Tree) selectVictimLeaf(record bool) *node {
	strat := t.evictionStrategy()
	if prep, ok := strat.(TreePreparer); ok {
		prep.PrepareTree(t)
	}
	var best *node
	var bestKey victimKey
	candidates, locked := 0, 0
	var stack []*node
	t.forEachRoot(func(r *node) { // one global victim pool spanning every namespace root
		for _, c := range r.children {
			stack = append(stack, c)
		}
	})
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if len(n.children) == 0 {
			candidates++
			if n.refs > 0 || n.IsComputing() {
				locked++
				continue
			}
			// Tier 0 (pinned root, immune from eviction: seg = 3)
			if t.isNodePinnedOrImmune(n) {
				locked++
				continue
			}
			k := strat.Priority(n)
			if seg, ok := t.nodeTierSeg(n); ok {
				k.seg = seg
			}
			if k.seg >= 3 {
				locked++
				continue
			}
			if best == nil || k.less(bestKey) {
				best, bestKey = n, k
			}
			continue
		}
		for _, c := range n.children {
			stack = append(stack, c)
		}
	}
	if record {
		t.recordEvictChoice(strat.Name(), candidates, locked, best)
	}
	return best
}

// isNodePinnedOrImmune reports whether node n is pinned as a Tier 0 coordinator prompt
// and therefore immune from eviction (seg = 3).
func (t *Tree) isNodePinnedOrImmune(n *node) bool {
	return t.isNodePinnedOrImmuneAt(n, t.clock)
}

func (t *Tree) isNodePinnedOrImmuneAt(n *node, clock uint64) bool {
	if n == nil {
		return false
	}
	if n.pinned {
		return true
	}
	if n.tierSet && n.tier == Tier0PinnedRoot {
		return true
	}
	if n.retention != nil {
		if !n.retention.expiredAtClock(clock) && n.retention.Priority >= 90 {
			return true
		}
	}
	return false
}

// nodeTierSeg returns the effective eviction segment (0..3) based on explicit retention or tier.
// Returns ok=true if an explicit retention, tier, or pin was set on n.
func (t *Tree) nodeTierSeg(n *node) (int, bool) {
	return t.nodeTierSegAt(n, t.clock)
}

func (t *Tree) nodeTierSegAt(n *node, clock uint64) (int, bool) {
	if n == nil {
		return 0, false
	}
	if t.isNodePinnedOrImmuneAt(n, clock) {
		return Tier0PinnedRoot.Seg(), true
	}
	if n.retention != nil {
		if n.retention.expiredAtClock(clock) {
			return Tier3Probationary.Seg(), true
		}
		return TierFromRetentionPriority(n.retention.Priority).Seg(), true
	}
	if n.tierSet {
		return n.tier.Seg(), true
	}
	return 0, false
}

// PinPrefix pins the prefix matching tokens (creating it if absent) as a Tier 0 coordinator
// prompt, making it immune from eviction (seg = 3) even when subagent branches collapse.
// It returns the pinned boundary node.
func (t *Tree) PinPrefix(tokens []int) *node {
	return t.PinPrefixNS("", tokens)
}

// PinPrefixNS pins the prefix matching tokens under namespace ns as a Tier 0 coordinator prompt.
func (t *Tree) PinPrefixNS(ns string, tokens []int) *node {
	if len(tokens) == 0 {
		return t.rootFor(ns)
	}
	root := t.rootFor(ns)
	boundary, matched := t.boundaryFor(root, tokens)
	if matched < len(tokens) {
		boundary = t.attachLeaf(boundary, tokens[matched:], nil, nil, t.clock)
	}
	t.pinPath(boundary)
	return boundary
}

// pinPath marks node n and every ancestor on the path up to root as pinned Tier 0.
func (t *Tree) pinPath(n *node) {
	req := RetentionRequest{
		Priority: MaxRetentionPriority,
		TTL:      RetainForever,
		Admitted: int64(t.clock),
	}
	for p := n; p != nil && p.parent != nil; p = p.parent {
		p.pinned = true
		p.opportunistic = false // explicit pinning is demand, even at access clock zero
		p.tier = Tier0PinnedRoot
		p.tierSet = true
		retCopy := req
		p.retention = &retCopy
	}
}

// UnpinPrefix unpins the node matching tokens on the default namespace.
func (t *Tree) UnpinPrefix(tokens []int) {
	t.UnpinPrefixNS("", tokens)
}

// UnpinPrefixNS unpins the node matching tokens under namespace ns.
func (t *Tree) UnpinPrefixNS(ns string, tokens []int) {
	root := t.rootForRead(ns)
	if root == nil || len(tokens) == 0 {
		return
	}
	n, nlen, _, _ := t.walk(root, tokens)
	if n != nil && nlen == len(tokens) {
		t.UnpinNode(n)
	}
}

// UnpinNode clears the pinned flag and Tier 0 retention on node n.
func (t *Tree) UnpinNode(n *node) {
	if n == nil {
		return
	}
	n.pinned = false
	n.tier = Tier2IdleSubagent
	n.tierSet = false
	n.retention = nil
}

// IsNodePinned reports whether node n is pinned (Tier 0 coordinator prompt, immune from eviction).
func (t *Tree) IsNodePinned(n *node) bool {
	return t.isNodePinnedOrImmune(n)
}

// SetNodeRetention sets a client-declared per-request KV retention descriptor on node n.
// Validates the request (fails closed on out-of-range priority or negative TTL/admitted).
// Priority >= 90 protects the node as Tier 0 only within its declared TTL window.
// For these high priorities, RetainForever also pins the ancestor path, like PinPrefix.
func (t *Tree) SetNodeRetention(n *node, req RetentionRequest) error {
	if n == nil {
		return errors.New("radixkv: nil node")
	}
	if err := req.Validate(); err != nil {
		return err
	}
	retCopy := req
	n.retention = &retCopy
	tier := TierFromRetentionPriority(req.Priority)
	n.tier = tier
	// A finite high-priority request must remain clock-governed: either a
	// permanent tier marker or pinPath would make it immune after expiry.
	// This preserves the TTL lifecycle of TensorRT-LLM's releaseBlock/refresh
	// at f4c5c935aa891b0826f73936c4831236cb6ff836 (Apache-2.0; #5259), using
	// Fak's existing logical expiry and probationary demotion semantics.
	n.tierSet = tier != Tier0PinnedRoot || req.TTL == RetainForever
	if tier == Tier0PinnedRoot && req.TTL == RetainForever {
		t.pinPath(n)
	} else {
		n.pinned = false
	}
	return nil
}

// NodeRetention returns the node's current RetentionRequest, if any.
func (t *Tree) NodeRetention(n *node) (RetentionRequest, bool) {
	if n == nil || n.retention == nil {
		return RetentionRequest{}, false
	}
	return *n.retention, true
}

// SetNodeTier explicitly assigns a priority tier to node n.
func (t *Tree) SetNodeTier(n *node, tier PriorityTier) {
	if n == nil {
		return
	}
	n.tier = tier
	n.tierSet = true
	if tier == Tier0PinnedRoot {
		t.pinPath(n)
	} else {
		n.pinned = false
	}
	req := RetentionRequestForTier(tier, int64(t.clock))
	n.retention = &req
}

// NodeTier returns the effective priority tier of node n.
func (t *Tree) NodeTier(n *node) PriorityTier {
	if n == nil {
		return Tier3Probationary
	}
	if t.isNodePinnedOrImmune(n) {
		return Tier0PinnedRoot
	}
	if n.retention != nil {
		if n.retention.expiredAtClock(t.clock) {
			return Tier3Probationary
		}
		return TierFromRetentionPriority(n.retention.Priority)
	}
	if n.tierSet {
		return n.tier
	}
	if n.hits == 0 {
		return Tier3Probationary
	}
	if n.hits >= 2 {
		return Tier1ActiveSubagent
	}
	return Tier2IdleSubagent
}

// PinnedTokens returns the total tokens owned by pinned Tier 0 nodes.
func (t *Tree) PinnedTokens() int {
	total := 0
	t.forEachNode(func(n *node) {
		if t.isNodePinnedOrImmune(n) {
			total += len(n.key)
		}
	})
	return total
}

// PinnedNodes returns the number of pinned Tier 0 nodes.
func (t *Tree) PinnedNodes() int {
	count := 0
	t.forEachNode(func(n *node) {
		if t.isNodePinnedOrImmune(n) {
			count++
		}
	})
	return count
}

func (n *node) kvSpanStats() compute.KVSpanStats {
	return compute.KVSpanStats{
		Tokens:   len(n.key),
		Bytes:    int64(max(n.plen, 1)),
		Hits:     n.hits,
		LastUsed: n.lastUsed,
		Leased:   n.refs > 0,
	}
}

// lastEvictPolicyName is the strategy name of the most recent eviction choice, defaulting
// to "lru" before any eviction has run — preserving the reading the EvictionPolicy-zero
// value gave (its String() was "lru") for every legacy Stats reader.
func (t *Tree) lastEvictPolicyName() string {
	if t.lastEvictPolicy == "" {
		return "lru"
	}
	return t.lastEvictPolicy
}

func (t *Tree) recordEvictChoice(strategyName string, candidates, locked int, victim *node) {
	t.lastEvictPolicy = strategyName
	t.lastEvictCandidates = candidates
	t.lastEvictLocked = locked
	t.lastEvictVictimCost = 0
	t.lastEvictVictimHits = 0
	t.lastEvictVictimTokens = 0
	t.lastEvictVictimPrefix = 0
	if victim == nil {
		return
	}
	st := victim.kvSpanStats()
	cost := compute.KVEvictionCost(st)
	if math.IsInf(cost, 0) {
		cost = 0
	}
	t.lastEvictVictimCost = cost
	t.lastEvictVictimHits = st.Hits
	t.lastEvictVictimTokens = st.Tokens
	t.lastEvictVictimPrefix = victim.plen
}

func (t *Tree) removeLeaf(v *node) {
	if v.parent == nil {
		return
	}
	delete(v.parent.children, v.key[0])
	t.tokens -= len(v.key)
	t.releaseSnapshotPayload(v)
	v.releaseDensePrefix()
	v.kv = nil
}

// EvictNode is POLICY eviction: it removes a node and its entire subtree from the cache
// regardless of LRU recency or lease state, returning the tokens freed. This is the seam a
// quarantine verdict drives — the radix-tree analogue of model.KVCache.Evict, and the
// capability an opportunistic LRU cache cannot provide: a poisoned prefix is removed
// because POLICY says so, not because memory ran out. (Callers serving from a leased node
// should still hold their own session; this drops the SHARED cached copy so no future
// request can reuse the poisoned prefix.)
func (t *Tree) EvictNode(n *node) int {
	if n == nil || n.parent == nil { // parent==nil is any namespace root (incl. the default)
		return 0
	}
	freed := subtreeTokens(n)
	t.closeSubtreeSnapshots(n)
	delete(n.parent.children, n.key[0])
	t.tokens -= freed
	t.policyEvictions++
	return freed
}

// EvictPrefix is verdict-driven eviction by TOKEN PATH: it walks the longest cached
// prefix of `tokens` and evicts the deepest cached node on that path (its whole
// subtree), returning the tokens freed. It is the EvictNode seam for a caller that
// holds the poisoned token sequence rather than a *node handle — exactly the in-kernel
// planner, which caches by token path and (because radixkv's node type is unexported)
// keeps no node references to feed EvictNode directly. A quarantine verdict passes the
// cached transcript THROUGH the poisoned span and the branch that cached it — and
// everything built on it — is dropped, so no later request can reuse the poisoned KV.
//
// It is a no-op (returns 0) when nothing along `tokens` is cached PAST where it would
// diverge — in particular, when the cached path ends BEFORE `tokens` is exhausted
// (`tokens` extends into a region that was never prefilled+cached), nothing is evicted.
// That guard is what keeps a clean shared prefix intact when the poison's continuation
// was never cached. Sibling branches that diverged before the matched node are always
// preserved (the benign-prefix guarantee), the same span-exact, sibling-sparing
// governance EvictNode provides.
//
// Contract: `tokens` must END at or within the poisoned span (the caller — a quarantine
// verdict — renders the transcript THROUGH the poisoned result, so a full match lands on
// a node whose cached KV genuinely attended to the poison). A coincidental mid-edge
// collision on the poison's first divergent token can conservatively evict a benign
// sibling branch; that is at worst a cache miss (it re-prefills), never a poison replay.
func (t *Tree) EvictPrefix(tokens []int) int { return t.EvictPrefixNS("", tokens) }

// EvictPrefixNS is EvictPrefix scoped to namespace ns (#3889): it walks ns's virtual root
// and evicts only within that namespace, so quarantining a poisoned prefix under one tenant
// / adapter / cache-salt never drops a benign identical-token prefix cached under another.
// An absent namespace is a no-op (returns 0).
func (t *Tree) EvictPrefixNS(ns string, tokens []int) int {
	if len(tokens) == 0 {
		return 0
	}
	n, nlen, pc, oi := t.walk(t.rootForRead(ns), tokens)
	_ = oi
	switch {
	case pc != nil:
		// `tokens` diverged MID-EDGE into child pc: the matched run tokens[nlen:nlen+oi)
		// is on pc's edge, so pc's root→pc path includes those (poison) tokens. Evict pc's
		// whole branch; pc's siblings (which diverged earlier) survive.
		return t.EvictNode(pc)
	case nlen == len(tokens) && n != nil && n.parent != nil:
		// The ENTIRE poisoned token path matched a cached node exactly: that node's KV
		// attended to the poison. Evict it and its subtree.
		return t.EvictNode(n)
	default:
		// nlen < len(tokens): the cached path ended before `tokens` was consumed — the
		// poison continuation was never cached here, so there is nothing poisoned to drop.
		return 0
	}
}

func (t *Tree) closeSubtreeSnapshots(n *node) {
	if n == nil {
		return
	}
	t.releaseSnapshotPayload(n)
	n.releaseDensePrefix()
	for _, child := range n.children {
		t.closeSubtreeSnapshots(child)
	}
}

func subtreeTokens(n *node) int {
	total := len(n.key)
	for _, c := range n.children {
		total += subtreeTokens(c)
	}
	return total
}
