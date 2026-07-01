// Package radixkv is SGLang's RadixAttention prefix cache, rebuilt over fak's
// kernel-owned KV cache — the apples-to-apples answer to "how does fak compare to
// SGLang's KV-cache radix attention?".
//
// SGLang's RadixAttention (arXiv:2312.07104) is the SOTA mechanism for *automatic*
// KV-cache reuse across requests: it stores every request's KV in a radix tree
// (compressed trie) keyed by the token-id sequence, so a new request that shares a
// prefix with any prior one DISCOVERS the longest cached prefix by a tree walk and
// reuses its K/V — recomputing only its own suffix. Under memory pressure it evicts
// the least-recently-used LEAF (collapsing parents upward), keeping hot prefixes.
//
// fak already had the two HARD halves of this — a kernel-owned KVCache that can be
// CLONED to splice a known prefix into a new session (model.SessionFromPrefix) and
// EVICTED span-exactly (model.KVCache.Evict, proven == never-saw-it). What it lacked
// was the radix tree's *automatic discovery*: the existing reuse path (NewBatchFromPrefix)
// makes you DECLARE the shared prefix. RadixAttention's contribution is that you don't —
// the tree finds it, even when two requests diverge in the MIDDLE of a run (which forces
// an edge SPLIT). This package is that tree, and nothing more: it is a pure CONSUMER of
// the proven primitives (Clone, Prefill, Evict), the same posture internal/kvmmu takes —
// so the reuse it schedules is bit-identical to a full recompute (proven in the live test),
// because the only KV operations it performs are the ones internal/model already verified.
//
// Two governance modes on ONE tree, which is the whole point of doing this in fak:
//
//   - EvictToBudget — LRU eviction under a token budget. This is RadixAttention's policy
//     verbatim: opportunistic, cache-pressure-driven, hot prefixes survive. It is what we
//     benchmark head-to-head with SGLang (cache hit rate is the metric SGLang's own paper
//     headlines, and it is hardware- and model-INDEPENDENT — a function of (workload,
//     matching algorithm) only — so fak-on-CPU vs SGLang-on-GPU is a fair comparison on
//     exactly this axis, unlike raw tok/s).
//
//   - EvictNode — POLICY eviction of a named span regardless of LRU or recency. This is
//     the capability an LRU radix cache structurally cannot offer: a poison/quarantine
//     verdict (internal/ctxmmu) evicts a specific subtree from the cache, and because each
//     node's KV is a real model.KVCache, the session that cloned that prefix can Evict the
//     span bit-identically to never-having-seen it. Same primitive (prefix-addressable KV),
//     opposite governance — measured, not asserted.
//
// Faithfulness note (honest, per the repo's "labeled, not faked" rule): each node stores
// the FULL-prefix KVCache for its root→node path (length == path length), not SGLang's
// per-segment slabs in a paged pool. That costs more memory, but the metric this package
// exists to compare — cache hit rate / prefill-tokens-saved — is layout-independent, and
// storing full-prefix caches lets every reuse and every edge-split go through the *proven*
// Clone/Evict primitives unchanged (a split truncates a child's cache to the boundary via
// Evict-of-the-tail, which has no survivor to re-RoPE, so it is an exact prefix). We trade
// memory for the guarantee that not one bit of the verified KV core is re-derived here.
package radixkv

import "github.com/anthony-chaudhary/fak/internal/model"

// node is one vertex of the compressed radix tree. The edge parent→node carries `key`
// (a run of token ids); the path root→node spells the token prefix this node caches.
// children are keyed by the FIRST token of each child's edge, so each match step is an
// O(1) map lookup followed by a run compare.
type node struct {
	key      []int
	parent   *node
	children map[int]*node

	// kv is the kernel-owned KV cache for the FULL prefix root→node (every token on the
	// path, post-prefill). nil in pure-accounting mode (no model attached); set by Insert
	// to the cache the caller built, and by split to a truncated clone of a child's cache.
	kv *model.KVCache

	plen     int    // path length in tokens (parent.plen + len(key)); == len(kv) when kv!=nil
	refs     int    // active leases; a leaf with refs>0 is never LRU-evicted
	lastUsed uint64 // logical clock of the most recent match/insert touching this node — LRU key
}

// Tree is a RadixAttention prefix cache: a radix tree of token sequences with
// longest-prefix matching, an LRU token budget, and reference counting.
type Tree struct {
	root      *node
	maxTokens int    // LRU budget in cached tokens; 0 disables eviction (unbounded)
	tokens    int    // total cached tokens = Σ len(node.key) over all nodes
	clock     uint64 // logical access clock

	evictions       int // LRU leaf evictions
	policyEvictions int // EvictNode calls (the fak differentiator)
	splits          int // edge splits performed (a structural RadixAttention event)
}

// New builds an empty prefix cache. maxTokens is the LRU budget in cached tokens; pass 0
// for an unbounded tree (no eviction — the right setting when measuring the matching
// algorithm's pure hit rate, before layering a memory bound on top).
func New(maxTokens int) *Tree {
	return &Tree{root: &node{children: map[int]*node{}}, maxTokens: maxTokens}
}

func (t *Tree) tick() uint64 { t.clock++; return t.clock }

// walk finds the longest prefix of tokens already in the tree. It returns the deepest
// node whose full path is a prefix of tokens (a node boundary, nlen tokens), plus — if
// the next edge only PARTIALLY matches — the child entered (pc) and how far into its key
// matched (oi, with 0<oi<len(pc.key)). Total matched tokens = nlen + oi.
func (t *Tree) walk(tokens []int) (n *node, nlen int, pc *node, oi int) {
	n = t.root
	i := 0
	for i < len(tokens) {
		ch := n.children[tokens[i]]
		if ch == nil {
			return n, i, nil, 0
		}
		j := 0
		for j < len(ch.key) && i+j < len(tokens) && tokens[i+j] == ch.key[j] {
			j++
		}
		if j == len(ch.key) {
			n, i = ch, i+j // whole edge consumed; descend
			continue
		}
		return n, i, ch, j // partial edge match: boundary is mid-edge ch
	}
	return n, i, nil, 0
}

// boundaryFor walks the longest cached prefix of tokens and, when the match lands mid-edge
// (oi>0), SPLITS that edge so a real node boundary with a reusable cache exists exactly at
// the matched length — returning that boundary node and the total matched token count. It is
// the shared "walk then split-to-boundary" preamble both the demand path (Lookup) and the
// prefetch path (WarmInsert) run before they attach a suffix; it performs the same structural
// split (t.splits bumps once, via t.split) and mutates nothing else, so both callers observe
// byte-identical tree state. It does NOT touch recency or leases — each caller applies its own
// residency policy afterward.
func (t *Tree) boundaryFor(tokens []int) (boundary *node, matched int) {
	n, nlen, pc, oi := t.walk(tokens)
	boundary, matched = n, nlen
	if oi > 0 {
		boundary = t.split(n, pc, oi)
		matched = nlen + oi
	}
	return boundary, matched
}

// MatchLen is the read-only accounting probe: the number of leading tokens of `tokens`
// already cached (a node boundary or mid-edge), with no mutation, no lock, no split. This
// is the hit-rate measurement seam — sum it over a workload and divide by total tokens to
// get the cache hit rate SGLang's paper reports.
func (t *Tree) MatchLen(tokens []int) int {
	_, nlen, _, oi := t.walk(tokens)
	return nlen + oi
}

// split inserts an intermediate node between parent and child at offset oi into child.key,
// so a node boundary exists exactly at the matched prefix length (parent.plen+oi). The
// intermediate node's KV is the child's cache TRUNCATED to that length when the model
// supports span eviction. This is the move that lets RadixAttention reuse a prefix that
// diverges in the middle of a run, which fak's declare-the-prefix path could not.
func (t *Tree) split(parent, child *node, oi int) *node {
	first := child.key[0]
	mid := &node{
		key:      append([]int(nil), child.key[:oi]...),
		parent:   parent,
		children: map[int]*node{},
		plen:     parent.plen + oi,
		lastUsed: child.lastUsed,
	}
	if child.kv != nil {
		mid.kv = truncatePrefix(child.kv, mid.plen)
	}
	child.key = append([]int(nil), child.key[oi:]...)
	child.parent = mid
	mid.children[child.key[0]] = child
	parent.children[first] = mid // first == mid.key[0]; overwrites child's old slot
	t.splits++
	return mid
}

// Lookup matches the longest cached prefix of tokens, SPLITTING an edge if the match
// lands mid-run so a real node boundary (with a reusable cache) exists there, bumps LRU
// recency along the matched path, and LEASES the boundary node (refs++) so an eviction
// cannot reclaim the prefix while the caller is serving from it. Returns the boundary node
// (call node.KV() for the clone-able reuse cache; nil if nothing matched) and the matched
// token count. The caller prefills tokens[matched:] and then calls Insert + Done.
//
// LEASE DISCIPLINE (memory-leak contract — TestLookupLeaseMustBeReleased guards it):
// Lookup takes a lease (refs++) on the boundary. The caller MUST release it on EVERY
// path — Insert hands the lease off to the new leaf, and Done releases it directly. If a
// caller returns early between Lookup and Insert (prefill error, context cancel, panic)
// WITHOUT calling Done, the boundary stays leased forever, and because evictToBudget never
// reclaims a leased node OR its ancestors, the whole prefix subtree is pinned against the
// budget permanently — a logical memory leak. Use `defer` so the abort path cannot forget:
//
//	b, m := tree.Lookup(req)
//	leaf := b            // what we ultimately Done()
//	defer func() { tree.Done(leaf) }()
//	... prefill (may fail) ...
//	leaf = tree.Insert(b, req[m:], kv)   // moves the lease onto the leaf
func (t *Tree) Lookup(tokens []int) (*node, int) {
	boundary, matched := t.boundaryFor(tokens)
	for p := boundary; p != nil; p = p.parent {
		p.lastUsed = t.clock + 1 // freshen the whole hot path
	}
	t.clock++
	boundary.refs++
	return boundary, matched
}

// Insert attaches the request's suffix (the tokens beyond the matched prefix) as a new
// child of the boundary node, with kv = the FULL-prefix cache the caller built for the
// whole request (matched prefix reused + suffix prefilled). It then enforces the LRU
// budget. An empty suffix means the request was already fully cached: nothing is added and
// the (still-leased) boundary node is returned. kv may be nil (pure-accounting mode).
//
// Lease handoff: Lookup leases the matched BOUNDARY so the reused prefix can't be evicted
// in the gap before this Insert; Insert moves that lease onto the new LEAF (the request's
// full path) and frees the boundary, because a non-leaf is never an eviction candidate, so
// holding the leaf protects every ancestor too. The caller therefore always Dones the node
// Insert returns, exactly when the request finishes — and the just-inserted request is
// itself protected from the eviction its own Insert may trigger.
func (t *Tree) Insert(boundary *node, suffix []int, kv *model.KVCache) *node {
	if len(suffix) == 0 {
		return boundary // already fully cached; keep the boundary lease for the caller to Done
	}
	s := append([]int(nil), suffix...)
	leaf := &node{
		key:      s,
		parent:   boundary,
		children: map[int]*node{},
		kv:       kv,
		plen:     boundary.plen + len(s),
		lastUsed: t.tick(),
	}
	boundary.children[s[0]] = leaf
	t.tokens += len(s)
	leaf.refs++ // lease the in-flight request's own leaf...
	if boundary.refs > 0 {
		boundary.refs-- // ...and release the prefix lease Lookup took (the leaf now guards the path)
	}
	t.evictToBudget()
	return leaf
}

// Done releases a lease taken by Lookup/Insert (refs--), so the node becomes LRU-evictable again.
func (t *Tree) Done(n *node) {
	if n != nil && n.refs > 0 {
		n.refs--
	}
}

// KV is the reusable kernel-owned cache for this node's full prefix (nil if none). Clone
// it (model.SessionFromPrefix does) before prefilling a suffix; never mutate it in place.
func (n *node) KV() *model.KVCache { return n.kv }

// Plen is the node's cached prefix length in tokens.
func (n *node) Plen() int { return n.plen }

// evictToBudget evicts least-recently-used, unlocked LEAVES until the cached-token count
// is within budget — RadixAttention's eviction policy. Removing a leaf can make its parent
// a leaf, which the next iteration may then evict (the upward collapse). A node that is
// leased (refs>0) is never chosen, so a prefix being served survives memory pressure.
func (t *Tree) evictToBudget() {
	for t.maxTokens > 0 && t.tokens > t.maxTokens {
		v := t.lruLeaf()
		if v == nil {
			return // everything in budget-excess is locked; cannot evict further
		}
		t.removeLeaf(v)
		t.evictions++
	}
}

// lruLeaf returns the unlocked leaf (no children, not the root) with the smallest lastUsed.
func (t *Tree) lruLeaf() *node {
	var best *node
	var stack []*node
	for _, c := range t.root.children {
		stack = append(stack, c)
	}
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if len(n.children) == 0 {
			if n.refs == 0 && (best == nil || n.lastUsed < best.lastUsed) {
				best = n
			}
			continue
		}
		for _, c := range n.children {
			stack = append(stack, c)
		}
	}
	return best
}

func (t *Tree) removeLeaf(v *node) {
	if v.parent == nil {
		return
	}
	delete(v.parent.children, v.key[0])
	t.tokens -= len(v.key)
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
	if n == nil || n == t.root || n.parent == nil {
		return 0
	}
	freed := subtreeTokens(n)
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
func (t *Tree) EvictPrefix(tokens []int) int {
	if len(tokens) == 0 {
		return 0
	}
	n, nlen, pc, oi := t.walk(tokens)
	_ = oi
	switch {
	case pc != nil:
		// `tokens` diverged MID-EDGE into child pc: the matched run tokens[nlen:nlen+oi)
		// is on pc's edge, so pc's root→pc path includes those (poison) tokens. Evict pc's
		// whole branch; pc's siblings (which diverged earlier) survive.
		return t.EvictNode(pc)
	case nlen == len(tokens) && n != t.root:
		// The ENTIRE poisoned token path matched a cached node exactly: that node's KV
		// attended to the poison. Evict it and its subtree.
		return t.EvictNode(n)
	default:
		// nlen < len(tokens): the cached path ended before `tokens` was consumed — the
		// poison continuation was never cached here, so there is nothing poisoned to drop.
		return 0
	}
}

func subtreeTokens(n *node) int {
	total := len(n.key)
	for _, c := range n.children {
		total += subtreeTokens(c)
	}
	return total
}

// truncatePrefix returns a clone of c covering only its first L positions, bit-identical
// to a prefill that stopped at L. It uses model.KVCache.TryEvict of the tail span [L, Len):
// because no position is cached after the evicted tail, Evict renumbers and re-RoPEs no
// survivor, so the result is an exact prefix. Recurrent caches that cannot be truncated
// return nil so the split remains a structural boundary and callers recompute.
func truncatePrefix(c *model.KVCache, L int) *model.KVCache {
	cp := c.Clone()
	if L < cp.Len() {
		if _, err := cp.TryEvict(L, cp.Len()-L); err != nil {
			return nil
		}
	}
	return cp
}

// Stats is a snapshot of the cache's structural state for reporting.
type Stats struct {
	Tokens          int // total cached tokens (Σ edge lengths) — the LRU-budget metric
	PrefixTokens    int // Σ node.plen over nodes holding a kv — TRUE resident KV positions
	Nodes           int // non-root nodes
	Leaves          int // leaf nodes
	MaxDepthTokens  int // longest cached prefix
	Evictions       int // LRU leaf evictions performed
	PolicyEvictions int // EvictNode calls
	Splits          int // edge splits performed
	MaxTokens       int // configured LRU budget (0 = unbounded)
}

// Stats walks the tree and returns its current shape.
//
// Note the deliberate Tokens vs PrefixTokens split. The LRU budget (MaxTokens) bounds
// Tokens = Σ edge lengths — SGLang's per-segment accounting, the apples-to-apples
// hit-rate metric this package exists to compare. But each node stores the FULL prefix
// root→node KV (length plen), so the TRUE resident KV footprint is PrefixTokens = Σ plen,
// which can exceed the configured budget by an O(prefix-depth) factor on a deep/narrow
// tree (a single N-token chain holds N·(N+1)/2 positions while Tokens reports only N).
// PrefixTokens makes that gap measurable instead of silent (see TestBudgetVsTrueKVFootprint).
func (t *Tree) Stats() Stats {
	s := Stats{Evictions: t.evictions, PolicyEvictions: t.policyEvictions, Splits: t.splits, MaxTokens: t.maxTokens}
	var visit func(n *node)
	visit = func(n *node) {
		if n != t.root {
			s.Nodes++
			s.Tokens += len(n.key)
			if n.kv != nil {
				s.PrefixTokens += n.plen
			}
			if n.plen > s.MaxDepthTokens {
				s.MaxDepthTokens = n.plen
			}
			if len(n.children) == 0 {
				s.Leaves++
			}
		}
		for _, c := range n.children {
			visit(c)
		}
	}
	visit(t.root)
	return s
}
