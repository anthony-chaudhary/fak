package radixkv

// namespace.go — (tokens, nsKey) NODE IDENTITY: the RadixKey/extra_key isolation SGLang has
// and fak's radix cache lacked (#3889). SGLang's RadixKey bundles the token ids with an
// `extra_key` — a LoRA-adapter id / cache-salt / namespace — so a tree child is keyed by
// (token, extra_key) and identical token prefixes under different extra_keys never share a
// node. fak keyed nodes PURELY by token id, so any two requests with the same token prefix
// shared the same cached K/V regardless of tenant / adapter / cache-salt / tool-world — a
// silent cross-namespace reuse that is an ISOLATION/CORRECTNESS hole, not just a perf gap.
//
// The borrow here is SGLang's "(or a virtual per-namespace root)" form: instead of widening
// every child map key to a (token, nsKey) pair, the tree keeps ONE virtual root per
// namespace (nsRoots, lazily created by rootFor). A Lookup/Insert/MatchLen/EvictPrefix/
// WarmInsert under namespace ns walks from ns's root, so its subtree is structurally
// disjoint from every other namespace's: a prefix cached under "A" is unreachable from a
// walk that starts at "B"'s root, so it can never be matched, leased, or served across the
// boundary — even when the token ids are identical. This is the plumbing #2674 (per-tenant
// weighted fairness floors) is blocked on: tenant identity now reaches the radix leaves.
//
// SEAM-PRESERVING. The default ("") namespace IS the original root, so every pre-namespace
// caller and test is byte-identical — Lookup/MatchLen/EvictPrefix/WarmInsert are thin
// wrappers that pass ns="". Insert/InsertWithLogits/Done/EvictNode/KV/Logits/Plen/Stats take
// no namespace: they operate on a *node handle (or globally), and a node already lives under
// its namespace's root via its parent chain. The proven Clone/Evict/split KV primitives are
// untouched — this changes only which ROOT a walk starts from, never how K/V is handled.
//
// ONE GLOBAL BUDGET, MANY IDENTITIES. Eviction (evictToBudget/lruLeaf/costAwareLeaf/
// leafCount) and Stats/WarmTokens span EVERY root via forEachRoot, so the token budget is a
// single shared pool the way the pre-namespace single-tree budget was — namespaces isolate
// IDENTITY, not memory. (Per-namespace budget FLOORS are #2674's job, layered on top of this
// identity plumbing, not this issue's.)
//
// SCOPE / follow-on (honest boundary). This lands the RadixKey/extra_key half. The second
// SGLang mechanism the issue cites — session_radix_cache's multi-holder `session_ids` set
// with last-holder frees and a bounded late-arrival tombstone — is deliberately NOT here: it
// would change the lease SEAM (Lookup/Insert/Done gaining a session id), rippling into the
// planner's lease discipline, so it is a separate non-seam-preserving increment. Residency
// stays the proven single-owner refs counter; only node identity gains the namespace axis.
// The evict→reuse thrash side-map (thrash.go) still keys on the token path alone, so two
// namespaces sharing a token path can at worst count one spurious thrash — that plane is
// observability-only and never steers victim choice or which K/V is served, so the isolation
// guarantee is unaffected; noted here rather than widened.

// rootFor returns the virtual root for namespace ns, lazily creating it (and the nsRoots
// map) on first write-path use. The default "" namespace always maps to t.root, so every
// pre-namespace caller keeps its exact behavior and shares the original tree.
func (t *Tree) rootFor(ns string) *node {
	if ns == "" {
		return t.root
	}
	if t.nsRoots == nil {
		t.nsRoots = map[string]*node{}
	}
	r := t.nsRoots[ns]
	if r == nil {
		r = &node{children: map[int]*node{}}
		t.nsRoots[ns] = r
	}
	return r
}

// rootForRead returns namespace ns's root WITHOUT creating one — nil for an absent
// namespace. It is the read/verdict-path sibling of rootFor (MatchLenNS/EvictPrefixNS), so a
// probe or quarantine for a namespace that never cached anything stays a pure no-op and
// materializes no empty root. The "" namespace is always the live t.root.
func (t *Tree) rootForRead(ns string) *node {
	if ns == "" {
		return t.root
	}
	return t.nsRoots[ns]
}

// forEachRoot invokes fn for the default root and every namespace root. It is how the
// budget-global passes — eviction victim selection, leaf counting, Stats, and warm
// accounting — enumerate the whole cache across all namespaces while node IDENTITY stays
// per-namespace. Namespace subtrees are disjoint, so no node is visited twice.
func (t *Tree) forEachRoot(fn func(r *node)) {
	fn(t.root)
	for _, r := range t.nsRoots {
		fn(r)
	}
}

// forEachRootNS is the identity-preserving sibling of forEachRoot. Callers
// that publish an address for a node must include its namespace so identical
// token paths in isolated trees cannot alias.
func (t *Tree) forEachRootNS(fn func(ns string, r *node)) {
	fn("", t.root)
	for ns, r := range t.nsRoots {
		fn(ns, r)
	}
}

// pruneEmptyNamespaceRoot removes a non-default virtual root once its last lease is
// released and it owns no children. Admission bypass makes this path common for scans over
// unique namespaces; without pruning, rejected candidates would consume no token budget
// yet could grow nsRoots without bound.
func (t *Tree) pruneEmptyNamespaceRoot(root *node) {
	if root == nil || root == t.root || root.parent != nil || root.refs != 0 || len(root.children) != 0 {
		return
	}
	for ns, candidate := range t.nsRoots {
		if candidate == root {
			delete(t.nsRoots, ns)
			return
		}
	}
}

// Namespaces reports how many distinct NON-default namespaces currently hold a virtual root
// (an observability/testing seam). The default ("") namespace is always present via t.root
// and is not counted.
func (t *Tree) Namespaces() int { return len(t.nsRoots) }
