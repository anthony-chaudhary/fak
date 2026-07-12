package radixkv

import "testing"

// Namespace-isolation tests (#3889): identical token prefixes under different namespaces
// (tenant / LoRA adapter / cache-salt / tool-world) must never share a node or its cached
// K/V, while intra-namespace sharing, the single global token budget, and every pre-namespace
// caller behave exactly as before. These use unexported state (*node, Namespaces), so they
// share the package clause `radixkv` with the package's other internal tests.

// serveNS runs one request through the tree under namespace ns with NO model (accounting
// only), returning the matched prefix length and the leased leaf to Done — the servePure
// sibling for a namespaced request.
func serveNS(tr *Tree, ns string, req []int) (matched int, leaf *node) {
	b, m := tr.LookupNS(ns, req)
	return m, tr.Insert(b, req[m:], nil)
}

// TestNamespaceIsolatesIdenticalPrefix is the core isolation witness: the SAME token
// sequence served under two namespaces produces two disjoint resident prefixes; neither
// namespace can match, lease, or reuse the other's cached K/V even though the token ids are
// identical. This is exactly the cross-tenant/adapter/cache-salt reuse the issue names.
func TestNamespaceIsolatesIdenticalPrefix(t *testing.T) {
	tree := New(0)
	req := seq(1, 20)

	mA, la := serveNS(tree, "tenant-A", req)
	tree.Done(la)
	if mA != 0 {
		t.Fatalf("first serve under tenant-A matched %d, want 0 (cold)", mA)
	}

	// The identical tokens under tenant-B must NOT reuse tenant-A's cached prefix.
	if m := tree.MatchLenNS("tenant-B", req); m != 0 {
		t.Fatalf("tenant-B matched %d of tenant-A's identical prefix, want 0 (isolation breach)", m)
	}
	mB, lb := serveNS(tree, "tenant-B", req)
	tree.Done(lb)
	if mB != 0 {
		t.Fatalf("first serve under tenant-B matched %d, want 0 (must not cross the namespace boundary)", mB)
	}

	// Each namespace fully reuses its OWN prefix — isolation, not a global miss.
	if m := tree.MatchLenNS("tenant-A", req); m != len(req) {
		t.Fatalf("tenant-A re-probe matched %d/%d, want full intra-namespace reuse", m, len(req))
	}
	if m := tree.MatchLenNS("tenant-B", req); m != len(req) {
		t.Fatalf("tenant-B re-probe matched %d/%d, want full intra-namespace reuse", m, len(req))
	}

	// Two disjoint namespace subtrees, both prefixes resident independently (no sharing).
	if n := tree.Namespaces(); n != 2 {
		t.Fatalf("Namespaces()=%d, want 2 (tenant-A, tenant-B); the default namespace is not counted", n)
	}
	if st := tree.Stats(); st.Tokens != 2*len(req) || st.Nodes != 2 {
		t.Fatalf("Tokens=%d Nodes=%d, want %d and 2 (both namespaces resident, nothing shared)", st.Tokens, st.Nodes, 2*len(req))
	}
}

// TestNamespaceIntraSharingIntact proves the isolation does not cost intra-namespace reuse:
// two requests sharing a preamble under the SAME namespace still discover it (the RadixAttention
// few-shot split), while the same preamble under a DIFFERENT namespace is not shared.
func TestNamespaceIntraSharingIntact(t *testing.T) {
	tree := New(0)
	pre := seq(1, 16)
	q1 := cat(pre, seq(500, 4))
	q2 := cat(pre, seq(900, 4))

	_, l1 := serveNS(tree, "A", q1)
	tree.Done(l1)

	if m := tree.MatchLenNS("A", q2); m != len(pre) {
		t.Fatalf("intra-namespace preamble reuse matched %d, want %d", m, len(pre))
	}
	if m := tree.MatchLenNS("B", q2); m != 0 {
		t.Fatalf("cross-namespace preamble matched %d, want 0 (isolation)", m)
	}
}

// TestNamespaceDefaultIsEmptyString pins the seam-preserving contract: the default namespace
// IS the "" namespace and the original single tree, so every pre-namespace caller keeps its
// exact behavior; a probe for a namespace that never cached anything is a pure no-op that
// materializes no root.
func TestNamespaceDefaultIsEmptyString(t *testing.T) {
	tree := New(0)
	req := seq(1, 12)

	_, leaf := servePure(tree, req) // the pre-namespace API
	tree.Done(leaf)

	if m := tree.MatchLenNS("", req); m != len(req) {
		t.Fatalf(`MatchLenNS("") matched %d, want %d (default namespace == "")`, m, len(req))
	}
	if m := tree.MatchLen(req); m != len(req) {
		t.Fatalf("MatchLen matched %d, want %d (pre-namespace path unchanged)", m, len(req))
	}
	if m := tree.MatchLenNS("other", req); m != 0 {
		t.Fatalf("a named namespace saw the default prefix: matched %d, want 0", m)
	}
	if n := tree.Namespaces(); n != 0 {
		t.Fatalf("Namespaces()=%d, want 0 (read-only probe of an absent namespace must not create a root)", n)
	}
}

// TestNamespaceBudgetIsGlobal proves namespaces isolate IDENTITY, not memory: the LRU token
// budget is one shared pool, so the globally-oldest leaf is evicted under pressure even when
// it is the sole occupant of its namespace.
func TestNamespaceBudgetIsGlobal(t *testing.T) {
	tree := New(20) // one shared 20-token pool across all namespaces
	a := distinctReq(0, 10)
	b := distinctReq(1, 10)
	c := distinctReq(2, 10)

	_, la := serveNS(tree, "A", a)
	tree.Done(la)
	_, lb := serveNS(tree, "B", b)
	tree.Done(lb)
	_, lc := serveNS(tree, "C", c) // 30 tokens over 3 namespaces > 20 → evict the global LRU
	tree.Done(lc)

	if m := tree.MatchLenNS("A", a); m != 0 {
		t.Fatalf("A (globally oldest) should be evicted across the namespace boundary, matched %d", m)
	}
	if m := tree.MatchLenNS("B", b); m != len(b) {
		t.Fatalf("B should survive, matched %d/%d", m, len(b))
	}
	if m := tree.MatchLenNS("C", c); m != len(c) {
		t.Fatalf("C should survive, matched %d/%d", m, len(c))
	}
	if st := tree.Stats(); st.Tokens != 20 || st.Evictions != 1 {
		t.Fatalf("Tokens=%d Evictions=%d, want 20 and 1 (a single global eviction)", st.Tokens, st.Evictions)
	}
}

// TestNamespaceEvictPrefixIsolated proves verdict-driven quarantine is namespace-scoped:
// evicting a poisoned prefix under one namespace leaves an identical-token prefix cached
// under another untouched, and a quarantine for an absent namespace is a no-op.
func TestNamespaceEvictPrefixIsolated(t *testing.T) {
	tree := New(0)
	poison := seq(1, 12)

	_, la := serveNS(tree, "A", poison)
	tree.Done(la)
	_, lb := serveNS(tree, "B", poison)
	tree.Done(lb)

	if freed := tree.EvictPrefixNS("A", poison); freed != len(poison) {
		t.Fatalf("EvictPrefixNS(A) freed %d, want %d", freed, len(poison))
	}
	if m := tree.MatchLenNS("A", poison); m != 0 {
		t.Fatalf("A's poisoned prefix should be gone, matched %d", m)
	}
	if m := tree.MatchLenNS("B", poison); m != len(poison) {
		t.Fatalf("B's identical-token prefix was wrongly evicted: matched %d/%d", m, len(poison))
	}
	if got := tree.EvictPrefixNS("absent", poison); got != 0 {
		t.Fatalf("EvictPrefixNS(absent) freed %d, want 0 (no-op for an unknown namespace)", got)
	}
}

// TestNamespaceWarmInsertScoped proves opportunistic prewarm respects the namespace boundary:
// a warm lands under its namespace's root (reusable only there), and WarmTokens accounts it
// across every namespace.
func TestNamespaceWarmInsertScoped(t *testing.T) {
	tree := New(0)
	warm := seq(1, 8)

	if got := tree.WarmInsertNS("A", warm, nil); got != len(warm) {
		t.Fatalf("WarmInsertNS(A) warmed %d, want %d", got, len(warm))
	}
	if m := tree.MatchLenNS("A", warm); m != len(warm) {
		t.Fatalf("A's warm prefix not matched within its namespace: %d/%d", m, len(warm))
	}
	if m := tree.MatchLenNS("B", warm); m != 0 {
		t.Fatalf("warm prefix leaked into namespace B: matched %d, want 0", m)
	}
	if wt := tree.WarmTokens(); wt != len(warm) {
		t.Fatalf("WarmTokens=%d, want %d (opportunistic warm counted across namespaces)", wt, len(warm))
	}
}
