package radixkv

// Witness tests closing two OPEN math-proof obligations for internal/radixkv.
// Discipline: fak/docs/proofs/00-METHOD.md. Each test ASSERTS an invariant
// (conservation / retention), is deterministic (fixed seed), and non-vacuous.
//
//   (1) refcount-conservation
//   (2) lru-leaf-evicted-hot-retained
//
// These use unexported state (node.refs, node.children, t.tokens) and so must
// share the package clause `radixkv` with the package's other internal tests.

import (
	"math/rand"
	"testing"
)

// sumRefs returns Σ node.refs over EVERY node, root included — the total number
// of outstanding leases. The root is counted because Lookup of a COLD request
// leases the root boundary transiently (root.refs++), handed off to the new leaf
// by Insert; excluding it would under-count an in-flight cold request's lease.
// The whole conservation argument reduces to this scalar.
func sumRefs(t *Tree) int {
	total := 0
	var visit func(n *node)
	visit = func(n *node) {
		total += n.refs
		for _, c := range n.children {
			visit(c)
		}
	}
	visit(t.root)
	return total
}

// reachable reports whether v is reachable from the root by following the
// children maps — i.e. whether it is still a live member of the tree. A
// removed/evicted node must be UNREACHABLE (and therefore never matched).
func reachable(t *Tree, v *node) bool {
	found := false
	var visit func(n *node)
	visit = func(n *node) {
		if n == v {
			found = true
		}
		for _, c := range n.children {
			visit(c)
		}
	}
	visit(t.root)
	return found
}

// ---- OPEN (1): refcount-conservation -------------------------------------

// TestRefcountConservationCycleNetsZero proves that a full request cycle
// (Lookup → Insert → Done) takes a lease and returns it, leaving Σ refs == 0,
// and that mid-cycle Σ refs equals exactly the number of in-flight (not-yet-
// Done) requests. Lookup does refs++ on the boundary; Insert hands that lease
// onto the new leaf (leaf.refs++, boundary.refs--) or, for an already-cached
// request, keeps the boundary lease; Done does refs--. So one cycle is a
// balanced +1/-1 and the running Σ refs must equal the live-request count.
func TestRefcountConservationCycleNetsZero(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5eed))
	tree := New(0) // unbounded: isolate leasing from eviction

	// Build a workload mixing cold requests, shared-prefix reuse, and exact
	// re-requests (the empty-suffix branch of Insert), so every lease path of
	// the cycle is exercised.
	preamble := seq(1, 12)
	reqs := [][]int{
		preamble,                        // cold, becomes a node
		cat(preamble, seq(200, 5)),      // reuses preamble, splits, new leaf
		cat(preamble, seq(300, 7)),      // reuses preamble (already split), new leaf
		append([]int(nil), preamble...), // EXACT re-request: empty suffix branch
		cat(preamble, seq(200, 5)),      // EXACT re-request of a deeper node: empty suffix
		distinctReq(9, 9),               // wholly disjoint cold request
	}

	var live []*node
	for i, r := range reqs {
		b, m := tree.Lookup(r)
		// Right after Lookup, exactly one new lease exists for THIS request, on
		// top of however many are already live.
		if got, want := sumRefs(tree), len(live)+1; got != want {
			t.Fatalf("req %d: after Lookup Σrefs=%d, want %d (one fresh boundary lease)", i, got, want)
		}
		leaf := tree.Insert(b, r[m:], nil)
		// Insert is a lease HANDOFF, not a new lease: the count is unchanged.
		if got, want := sumRefs(tree), len(live)+1; got != want {
			t.Fatalf("req %d: after Insert Σrefs=%d, want %d (handoff conserves the lease)", i, got, want)
		}
		if leaf.refs <= 0 {
			t.Fatalf("req %d: leaf must hold the live lease, refs=%d", i, leaf.refs)
		}
		live = append(live, leaf)
	}

	// All requests in flight: Σ refs must equal the number of live requests.
	if got, want := sumRefs(tree), len(reqs); got != want {
		t.Fatalf("all in flight: Σrefs=%d, want %d (one lease per live request)", got, want)
	}

	// Release in a shuffled order; after each Done the live count drops by one
	// and Σ refs must track it exactly (never negative, never leaked).
	rng.Shuffle(len(live), func(i, j int) { live[i], live[j] = live[j], live[i] })
	for k, n := range live {
		tree.Done(n)
		if got, want := sumRefs(tree), len(live)-(k+1); got != want {
			t.Fatalf("after %d Dones Σrefs=%d, want %d (leases net to zero)", k+1, got, want)
		}
	}
	if got := sumRefs(tree); got != 0 {
		t.Fatalf("after all Done Σrefs=%d, want 0 (no leaked lease)", got)
	}

	// Done is idempotent at zero: an extra Done never drives refs negative.
	for _, n := range live {
		tree.Done(n)
	}
	if got := sumRefs(tree); got != 0 {
		t.Fatalf("extra Done drove Σrefs to %d, want 0 (refs floored at 0)", got)
	}
}

// TestRemovedNodeUnreachableNeverMatched proves the dangling-node half of
// conservation: a node removed by eviction or policy-eviction is detached from
// the tree (unreachable from root) AND can no longer be matched. No reference
// to a removed node survives in any reachable parent's children map.
func TestRemovedNodeUnreachableNeverMatched(t *testing.T) {
	tree := New(20) // budget forces an LRU eviction
	a, b, c := distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)

	_, la := servePure(tree, a)
	tree.Done(la)
	// Capture a's leaf node before it gets evicted.
	an := tree.root.children[a[0]]
	if an == nil || !reachable(tree, an) {
		t.Fatalf("setup: a must be in the tree before eviction")
	}

	_, lb := servePure(tree, b)
	tree.Done(lb)
	_, lc := servePure(tree, c) // tokens 30 > 20: a (oldest) is evicted
	tree.Done(lc)

	if reachable(tree, an) {
		t.Fatalf("evicted node a is still reachable from root (dangling)")
	}
	if m := tree.MatchLen(a); m != 0 {
		t.Fatalf("evicted node a still matched %d tokens, want 0 (unreachable ⇒ never matched)", m)
	}
	if _, ok := tree.root.children[a[0]]; ok {
		t.Fatalf("evicted node a's slot still present in parent.children")
	}

	// Policy eviction: remove a node + its subtree, assert detached + unmatched.
	d := distinctReq(3, 8)
	_, ld := servePure(tree, d)
	tree.Done(ld)
	dn := tree.root.children[d[0]]
	if dn == nil {
		t.Fatalf("setup: d must be in the tree before policy eviction")
	}
	freed := tree.EvictNode(dn)
	if freed != len(d) {
		t.Fatalf("policy-evicted %d tokens, want %d", freed, len(d))
	}
	if reachable(tree, dn) {
		t.Fatalf("policy-evicted node d still reachable from root (dangling)")
	}
	if m := tree.MatchLen(d); m != 0 {
		t.Fatalf("policy-evicted node d still matched %d, want 0", m)
	}

	// Conservation of the token counter: t.tokens must equal the live Σ edge
	// lengths after all removals (no phantom tokens left behind).
	st := tree.Stats()
	if tree.tokens != st.Tokens {
		t.Fatalf("token counter %d != live Σ edge lengths %d (counter leaked on removal)", tree.tokens, st.Tokens)
	}
}

// ---- OPEN (2): lru-leaf-evicted-hot-retained -----------------------------

// TestLRUEvictsOldestRetainsHotAndLeased proves the core retention property:
// under a token budget, the LEAST-recently-used unlocked leaf is the one
// evicted, while (a) a hot/recently-touched prefix and (b) a leased (refs>0)
// prefix are retained even when they are the oldest by insertion order.
func TestLRUEvictsOldestRetainsHotAndLeased(t *testing.T) {
	// Budget 20 holds two 10-token requests. Insert a, b, c in order.
	// Without intervention, a (oldest) would evict. We make a HOT by touching
	// it (a Lookup freshens its lastUsed) right before c overflows the budget,
	// so b — now the oldest UNLOCKED — must be the victim instead.
	tree := New(20)
	a, b, c := distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)

	_, la := servePure(tree, a)
	tree.Done(la)
	_, lb := servePure(tree, b)
	tree.Done(lb)

	// Touch a so it becomes the most-recently-used (hot prefix). MatchLen does
	// not freshen, but a Lookup does (it freshens the whole matched path), so
	// re-Lookup a and immediately Done its handed-off lease.
	ba, ma := tree.Lookup(a)
	tree.Done(tree.Insert(ba, a[ma:], nil)) // empty suffix: keeps a, freshens it

	_, lc := servePure(tree, c) // 30 > 20 → evict the oldest UNLOCKED leaf
	tree.Done(lc)

	if m := tree.MatchLen(a); m != len(a) {
		t.Fatalf("hot prefix a evicted (matched %d), want retained %d", m, len(a))
	}
	if m := tree.MatchLen(b); m != 0 {
		t.Fatalf("cold-oldest b should be evicted, matched %d", m)
	}
	if m := tree.MatchLen(c); m != len(c) {
		t.Fatalf("freshly inserted c should survive, matched %d/%d", m, len(c))
	}

	// Now the leased-retention half: oldest is leased ⇒ a younger one evicts.
	tree2 := New(20)
	x, y, z := distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)
	_, lx := servePure(tree2, x) // x stays LEASED (no Done) — in flight
	_, ly := servePure(tree2, y)
	tree2.Done(ly)
	_, lz := servePure(tree2, z) // 30 > 20 → must evict, but x is locked
	tree2.Done(lz)

	if m := tree2.MatchLen(x); m != len(x) {
		t.Fatalf("leased oldest x must survive, matched %d/%d", m, len(x))
	}
	if m := tree2.MatchLen(y); m != 0 {
		t.Fatalf("oldest UNLOCKED y should evict, matched %d", m)
	}
	if m := tree2.MatchLen(z); m != len(z) {
		t.Fatalf("z should survive, matched %d/%d", m, len(z))
	}
	tree2.Done(lx)
}

// TestLRUUpwardCollapse proves the repeated/recursive eviction: when removing a
// leaf turns its parent into a leaf, the parent itself becomes eviction-eligible
// and is collapsed away in the same evictToBudget pass, until within budget.
//
// Construction: one shared preamble with two short tails forms a tree
//
//	root → preamble → {tailA, tailB}.
//
// Evicting both tails makes `preamble` a leaf; if the budget still demands it,
// preamble must then be evicted too (the upward collapse). We size the budget so
// the whole chain must go to make room for a large disjoint request.
func TestLRUUpwardCollapse(t *testing.T) {
	pre := seq(1, 8)
	tailA := cat(pre, seq(100, 3)) // plen 11
	tailB := cat(pre, seq(200, 3)) // plen 11
	// After both, distinct edge tokens cached = preamble(8)+tailA-suffix(3)+tailB-suffix(3) = 14.
	tree := New(14) // exactly holds the preamble+two tails

	for _, r := range [][]int{tailA, tailB} {
		_, l := servePure(tree, r)
		tree.Done(l)
	}
	st := tree.Stats()
	if st.Tokens != 14 || st.Nodes != 3 {
		t.Fatalf("setup: tokens=%d nodes=%d, want 14 and 3 (preamble + 2 tails)", st.Tokens, st.Nodes)
	}
	// preamble exists as an internal (non-leaf) node now.
	pn := tree.root.children[pre[0]]
	if pn == nil || len(pn.children) != 2 {
		t.Fatalf("setup: preamble should be an internal node with 2 children")
	}

	// Now serve a disjoint 14-token request. Budget is 14, so 14+14=28 > 14:
	// the entire stale preamble subtree (both tails AND the now-leaf preamble)
	// must collapse away to make room.
	big := distinctReq(7, 14)
	_, lbig := servePure(tree, big)
	tree.Done(lbig)

	if m := tree.MatchLen(tailA); m != 0 {
		t.Fatalf("tailA should be evicted, matched %d", m)
	}
	if m := tree.MatchLen(tailB); m != 0 {
		t.Fatalf("tailB should be evicted, matched %d", m)
	}
	// The upward-collapse assertion: the preamble node itself is gone, not left
	// as a stranded childless internal node.
	if m := tree.MatchLen(pre); m != 0 {
		t.Fatalf("preamble should have collapsed upward (matched %d), want 0", m)
	}
	if reachable(tree, pn) {
		t.Fatalf("collapsed preamble node still reachable (upward collapse incomplete)")
	}
	if m := tree.MatchLen(big); m != len(big) {
		t.Fatalf("the new request should be fully cached, matched %d/%d", m, len(big))
	}
	final := tree.Stats()
	if final.Tokens != len(big) || final.Tokens > tree.maxTokens {
		t.Fatalf("after collapse tokens=%d, want %d and ≤ budget %d", final.Tokens, len(big), tree.maxTokens)
	}
	if tree.tokens != final.Tokens {
		t.Fatalf("token counter %d != live Σ edge lengths %d after collapse", tree.tokens, final.Tokens)
	}
}
