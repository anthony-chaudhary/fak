package radixkv

import (
	"sort"
	"testing"
)

// admin_io_test.go - cache-admin seam proofs: snapshots are pure reads that track
// the engine's own Stats() counters, EvictAdmin is bounded and routes through the
// verdict seam, and namespace listing matches residency.

// servePureTreeNS is the namespace-scoped sibling of radixkv_test.go servePure:
// it runs one request through namespace ns with NO model (accounting only),
// returning the matched prefix length and the leased leaf for the caller to Done.
func servePureTreeNS(tree *Tree, ns string, req []int) (int, *node) {
	b, m := tree.LookupNS(ns, req)
	return m, tree.Insert(b, req[m:], nil)
}

// TestSnapshotAdminTracksStats puts several disjoint requests in, then checks the
// admin snapshot's shape fields agree with Stats() and that the snapshot is a pure
// read (calling it again changes nothing).
func TestSnapshotAdminTracksStats(t *testing.T) {
	tree := New(0)
	pre := seq(1, 32)
	for i := 0; i < 3; i++ {
		_, leaf := servePure(tree, cat(pre, seq(2000+i*100, 8)))
		tree.Done(leaf)
	}
	st := tree.Stats()
	got := SnapshotAdmin(tree)
	if got.Namespace != "" {
		t.Fatalf("default snapshot namespace = %q, want empty", got.Namespace)
	}
	if got.Nodes != st.Nodes {
		t.Fatalf("Nodes=%d, want %d (Stats agreement)", got.Nodes, st.Nodes)
	}
	if got.Bytes != int64(st.Tokens) {
		t.Fatalf("Bytes=%d, want %d (radix-token footprint)", got.Bytes, st.Tokens)
	}
	if got.Evictions != int64(st.Evictions) || got.CostEvictions != int64(st.CostEvictions) ||
		got.PageEvictions != int64(st.PageEvictions) || got.PolicyEvictions != int64(st.PolicyEvictions) ||
		got.Splits != int64(st.Splits) {
		t.Fatalf("counter mismatch vs Stats: %+v vs %+v", got, st)
	}
	if got.Timestamp.IsZero() {
		t.Fatal("snapshot Timestamp must be set")
	}
	// Pure read: a second snapshot is identical in every deterministic field.
	again := SnapshotAdmin(tree)
	if again.Nodes != got.Nodes || again.Bytes != got.Bytes {
		t.Fatalf("snapshot mutated state: %+v then %+v", got, again)
	}
}

// TestSnapshotAdminZeroValueSafety covers the nil-tree and empty-tree guards.
func TestSnapshotAdminZeroValueSafety(t *testing.T) {
	var nilTree *Tree
	got := SnapshotAdmin(nilTree)
	if got.Nodes != 0 || got.Bytes != 0 || got.Namespace != "" {
		t.Fatalf("nil-tree snapshot = %+v, want zero default snapshot", got)
	}
	empty := SnapshotAdmin(New(0))
	if empty.Nodes != 0 || empty.Bytes != 0 {
		t.Fatalf("empty-tree snapshot = %+v, want zero shape", empty)
	}
	if !empty.Empty() {
		t.Fatal("empty namespace must report Empty()")
	}
}

// TestSnapshotAdminNSEmptyNamespace proves an absent namespace reads as zero
// without materializing a root (rootForRead discipline).
func TestSnapshotAdminNSEmptyNamespace(t *testing.T) {
	tree := New(0)
	_, leaf := servePure(tree, seq(1, 16))
	tree.Done(leaf)
	got := SnapshotAdminNS(tree, "absent")
	if got.Nodes != 0 || got.Bytes != 0 {
		t.Fatalf("absent-ns snapshot = %+v, want zero", got)
	}
	if tree.Namespaces() != 0 {
		t.Fatalf("SnapshotAdminNS must not materialize roots, nsRoots=%d", tree.Namespaces())
	}
}

// TestSnapshotAdminAllPerNamespace fills the default namespace and two others,
// then checks per-namespace shape attribution, default-first ordering, and
// determinism (sorted non-default rows).
func TestSnapshotAdminAllPerNamespace(t *testing.T) {
	tree := New(0)
	_, leaf := servePure(tree, seq(1, 16))
	tree.Done(leaf)
	_, leafB := servePureTreeNS(tree, "b", seq(50, 8))
	tree.Done(leafB)
	_, leafA := servePureTreeNS(tree, "a", seq(30, 8))
	tree.Done(leafA)

	all := SnapshotAdminAll(tree)
	if len(all.Trees) != 3 {
		t.Fatalf("got %d namespace rows, want 3 (default first)", len(all.Trees))
	}
	if all.Trees[0].Namespace != DefaultNamespaceLabel || all.Trees[0].Nodes == 0 {
		t.Fatalf("first row must be non-empty default ns, got %+v", all.Trees[0])
	}
	byNS := map[string]AdminSnapshot{}
	for _, row := range all.Trees {
		byNS[row.Namespace] = row
	}
	for _, ns := range []string{"a", "b"} {
		row, ok := byNS[ns]
		if !ok || row.Nodes == 0 || row.Bytes == 0 {
			t.Fatalf("ns %q missing or empty in %+v", ns, all.Trees)
		}
		if row.Bytes != 8 {
			t.Fatalf("ns %q Bytes=%d, want 8", ns, row.Bytes)
		}
	}
	names := []string{all.Trees[1].Namespace, all.Trees[2].Namespace}
	if !sort.StringsAreSorted(names) {
		t.Fatalf("non-default rows not sorted: %v", names)
	}
}

// TestEvictAdminRespectsLimit fills one namespace with separate disjoint request
// chains and asserts EvictAdmin evicts AT MOST limit subtrees, bumps the verdict
// counter exactly once per round, and collapses the namespace when re-run to
// exhaustion.
func TestEvictAdminRespectsLimit(t *testing.T) {
	tree := New(0)
	for i := 0; i < 6; i++ {
		_, leaf := servePureTreeNS(tree, "adm", seq(1000+i*100, 8))
		tree.Done(leaf)
	}
	n, err := EvictAdmin(tree, []byte("adm"), 2)
	if err != nil {
		t.Fatalf("EvictAdmin error: %v", err)
	}
	if n != 2 {
		t.Fatalf("evicted %d rounds, want exactly limit 2", n)
	}
	if st := tree.Stats(); st.PolicyEvictions != 2 {
		t.Fatalf("PolicyEvictions=%d, want 2 (verdict-seam routing)", st.PolicyEvictions)
	}
	// Exhaust the namespace: each remaining chain must fall, but the wrapper must
	// stop (not loop forever) once the namespace is empty. 4 disjoint leaf chains
	// remain, so the collapse needs at most 4 more rounds.
	n2, err := EvictAdmin(tree, []byte("adm"), 100)
	if err != nil {
		t.Fatalf("EvictAdmin exhaust error: %v", err)
	}
	if n2 < 1 || n2 > 4 {
		t.Fatalf("exhaust evicted %d rounds, want 1..4 remaining chains", n2)
	}
	after := SnapshotAdminNS(tree, "adm")
	if !after.Empty() {
		t.Fatalf("namespace not collapsed: %+v", after)
	}
	// The virtual root may legitimately survive (it prunes only via Done on the
	// last lease path); residency is what must read empty. Assert no node in the
	// whole namespace remains, which is the invariant ListResidentNamespaces
	// consumers care about (residency, not root bookkeeping).
	if afterRes := SnapshotAdminNS(tree, "adm"); !afterRes.Empty() {
		t.Fatalf("namespace not collapsed: %+v", afterRes)
	}
}

// TestEvictAdminDefaultNamespaceAndNegatives covers ns=nil (default), limit=0
// no-op, negative-limit error, and nil-tree error.
func TestEvictAdminDefaultNamespaceAndNegatives(t *testing.T) {
	tree := New(0)
	_, leaf := servePure(tree, seq(1, 16))
	tree.Done(leaf)
	if n, err := EvictAdmin(tree, nil, 0); err != nil || n != 0 {
		t.Fatalf("limit 0 must be a no-op, got (%d, %v)", n, err)
	}
	if _, err := EvictAdmin(tree, nil, -1); err == nil {
		t.Fatal("negative limit must error")
	}
	if _, err := EvictAdmin(nil, nil, 1); err == nil {
		t.Fatal("nil tree must error")
	}
	n, err := EvictAdmin(tree, nil, 5)
	if err != nil || n == 0 {
		t.Fatalf("default-ns eviction failed: (%d, %v)", n, err)
	}
	if st := tree.Stats(); st.Nodes != 0 {
		t.Fatalf("default ns not collapsed: Nodes=%d", st.Nodes)
	}
}

// TestListResidentNamespacesReflectsUsage checks the default label is always
// present, a used namespace appears, and the nil-tree error explains the missing
// listing primitive.
func TestListResidentNamespacesReflectsUsage(t *testing.T) {
	tree := New(0)
	list, err := ListResidentNamespaces(tree)
	if err != nil {
		t.Fatalf("ListResidentNamespaces error: %v", err)
	}
	if len(list) != 1 || list[0] != DefaultNamespaceLabel {
		t.Fatalf("fresh tree namespaces = %v, want just the default label", list)
	}
	_, leaf := servePureTreeNS(tree, "tenant", seq(9, 4))
	tree.Done(leaf)
	list, err = ListResidentNamespaces(tree)
	if err != nil {
		t.Fatalf("ListResidentNamespaces error: %v", err)
	}
	if !containsNamespace(list, "tenant") {
		t.Fatalf("used namespace missing: %v", list)
	}
	var nilTree *Tree
	if _, err := ListResidentNamespaces(nilTree); err == nil {
		t.Fatal("nil tree must error (no listing primitive without a tree)")
	}
}

// TestSnapshotAdminCounterMonotonicAfterEvict asserts the eviction counters move
// forward across an EvictAdmin round and never roll back.
func TestSnapshotAdminCounterMonotonicAfterEvict(t *testing.T) {
	tree := New(0)
	_, leaf := servePureTreeNS(tree, "mono", seq(7, 16))
	tree.Done(leaf)
	before := SnapshotAdminNS(tree, "mono")
	if _, err := EvictAdmin(tree, []byte("mono"), 10); err != nil {
		t.Fatalf("EvictAdmin error: %v", err)
	}
	after := SnapshotAdminNS(tree, "mono")
	if after.PolicyEvictions <= before.PolicyEvictions {
		t.Fatalf("PolicyEvictions %d -> %d, want strict increase", before.PolicyEvictions, after.PolicyEvictions)
	}
	if after.Nodes > before.Nodes {
		t.Fatalf("Nodes %d -> %d, want decrease or equal", before.Nodes, after.Nodes)
	}
}

// containsNamespace reports whether the namespace list includes ns.
func containsNamespace(list []string, ns string) bool {
	for _, v := range list {
		if v == ns {
			return true
		}
	}
	return false
}
