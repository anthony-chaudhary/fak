package radixkv

import (
	"errors"
	"fmt"
	"sort"
	"time"
)

// admin_io.go - typed admin I/O for radixkv: read-atomic cache-state snapshots and
// a bounded namespace eviction wrapper for cache-admin tooling. Every reader here is
// a pure fold over the same traversal Stats() uses over the tree; nothing on the
// serving path changes and no counter is mutated by the snapshot readers. EvictAdmin
// mutates ONLY through the existing verdict-driven EvictPrefixNS / EvictNode seams
// and only when the caller asks, capped by limit.

// AdminSnapshot is a point-in-time view of one namespace's cache state for
// operators and admin endpoints. Field names follow Stats()'s eviction counters;
// Bytes is the radix-token footprint (sum of edge lengths) - the LRU-budget metric.
// A namespace currently holds nothing when Nodes == 0.
type AdminSnapshot struct {
	Namespace string

	// Resident shape.
	Nodes        int   // non-root nodes under this namespace
	Bytes        int64 // sum of node.key edge lengths (radix tokens) - the budget metric
	ResidentKV   int   // nodes with non-nil kv: TRUE resident KV prefixes
	SnapshotHold int   // nodes carrying a complete snapshot payload (any tier)
	Tokens       int64 // sum of node.plen over nodes holding a kv (resident KV positions)

	// Lifetime eviction counters (never roll back; mirrored from Stats()).
	Evictions       int64 // LRU leaf evictions performed
	CostEvictions   int64 // cost-aware leaf evictions performed
	PageEvictions   int64 // page-aware leaf evictions performed
	PolicyEvictions int64 // EvictNode calls (verdict-driven subtree removals)

	// Splits is the structural edge-split count (tree-lifetime, mirrored from Stats).
	Splits int64

	Timestamp time.Time // when the snapshot was taken
}

// Empty reports whether the namespace currently holds no resident nodes.
func (s AdminSnapshot) Empty() bool { return s.Nodes == 0 }

// AdminMultiSnapshot is a snapshot of every namespace's cache state at one instant.
// Trees carries one entry per namespace present at snapshot time, always including
// the default ("") namespace even when it is empty.
type AdminMultiSnapshot struct {
	Trees     []AdminSnapshot
	Timestamp time.Time
}

// SnapshotAdmin reads the default namespace's current shape WITHOUT mutating it.
// It walks the namespace's subtree - the same traversal discipline Stats() uses -
// so the shape is a consistent fold over the live tree. Tree-lifetime counters
// (Evictions / CostEvictions / PageEvictions / PolicyEvictions / Splits) are global
// tree counters mirrored unchanged, because the budget, the global eviction passes,
// and the EvictNode seam all operate tree-wide; the node-shape fields are
// namespace-local. Zero-value and nil trees yield a zero snapshot for the default
// namespace - never a panic.
func SnapshotAdmin(t *Tree) AdminSnapshot { return SnapshotAdminNS(t, "") }

// SnapshotAdminNS is SnapshotAdmin scoped to namespace ns ("" = the default
// namespace). Tree-lifetime counters are mirrored unchanged; node-shape fields
// are ns-local. An absent namespace reads as the zero snapshot.
func SnapshotAdminNS(t *Tree, ns string) AdminSnapshot {
	s := AdminSnapshot{Namespace: ns, Timestamp: time.Now()}
	if t == nil {
		return s
	}
	s.Evictions = int64(t.evictions)
	s.CostEvictions = int64(t.costEvictions)
	s.PageEvictions = int64(t.pageEvictions)
	s.PolicyEvictions = int64(t.policyEvictions)
	s.Splits = int64(t.splits)
	root := t.rootForRead(ns)
	if root == nil {
		return s
	}
	walkNS(root, &s)
	return s
}

// walkNS folds one namespace's subtree into s, skipping the virtual root (its
// parent==nil marks a namespace root, which pays no node cost - the same rule
// Stats()'s visitor applies).
func walkNS(root *node, s *AdminSnapshot) {
	var visit func(n *node)
	visit = func(n *node) {
		if n != root { // skip every namespace root; count real nodes once
			s.Nodes++
			s.Bytes += int64(len(n.key))
			if n.snapshot != nil || n.hostSnapshot != nil || n.remoteSnapshot != nil {
				s.SnapshotHold++
			}
			if n.kv != nil {
				s.ResidentKV++
				s.Tokens += int64(n.plen)
			}
		}
		for _, c := range n.children {
			visit(c)
		}
	}
	visit(root)
}

// SnapshotAdminAll snapshots EVERY namespace present on the tree: the default
// namespace first, then non-default namespaces sorted by name for deterministic
// output. Read-only, zero-mutation; a nil tree yields a snapshot with just the
// empty default entry.
func SnapshotAdminAll(t *Tree) AdminMultiSnapshot {
	now := time.Now()
	out := AdminMultiSnapshot{Timestamp: now}
	if t == nil {
		out.Trees = []AdminSnapshot{SnapshotAdminNS(nil, "")}
		return out
	}
	def := SnapshotAdminNS(t, "")
	def.Timestamp = now
	out.Trees = append(out.Trees, def)
	names := make([]string, 0, len(t.nsRoots))
	for ns := range t.nsRoots {
		names = append(names, ns)
	}
	sort.Strings(names)
	for _, ns := range names {
		row := SnapshotAdminNS(t, ns)
		row.Timestamp = now
		out.Trees = append(out.Trees, row)
	}
	return out
}

// evictAdminLimitCap bounds one EvictAdmin call's round count; every round evicts
// a whole subtree, so namespaces collapse in O(rounds) walks.
const evictAdminLimitCap = 1 << 20

// EvictAdmin evicts up to limit nodes (subtrees) resident under namespace ns,
// returning how many subtree evictions were performed. It is a thin bounded
// wrapper over the existing EvictPrefixNS / EvictNode policy seams: each round
// removes ONE deepest cached node boundary under ns (its whole subtree goes with
// it - parents only surface after children are gone), so repeated rounds collapse
// a namespace toward empty. ns == nil targets the default namespace; a
// never-populated namespace is an immediate (0, nil) no-op.
//
// Errors are precondition-only: a nil tree, or a negative limit. limit == 0 evicts
// nothing and returns (0, nil).
func EvictAdmin(t *Tree, ns []byte, limit int) (int, error) {
	if t == nil {
		return 0, errors.New("radixkv: EvictAdmin on nil tree")
	}
	if limit < 0 {
		return 0, fmt.Errorf("radixkv: EvictAdmin limit %d is negative", limit)
	}
	name := ""
	if ns != nil {
		name = string(ns)
	}
	if limit == 0 {
		return 0, nil
	}
	if limit > evictAdminLimitCap {
		limit = evictAdminLimitCap
	}
	evicted := 0
	for evicted < limit {
		root := t.rootForRead(name)
		if root == nil {
			break // namespace never populated: no-op
		}
		best, bestDepth := evictAdminBoundary(root)
		if best == nil {
			break // nothing left resident under this namespace
		}
		tokens := make([]int, 0, bestDepth)
		for p := best; p != nil && p.parent != nil; p = p.parent {
			tokens = append(tokens, p.key[0])
		}
		for i, j := 0, len(tokens)-1; i < j; i, j = i+1, j-1 {
			tokens[i], tokens[j] = tokens[j], tokens[i]
		}
		// Route through the verdict seam: EvictPrefixNS re-walks ns's root, lands on
		// this boundary (or its mid-edge split image - an equal-or-wider subtree) and
		// evicts it via EvictNode, so the policyEvictions counter and the
		// snapshot-close bookkeeping stay owned by the engine's proven path.
		t.EvictPrefixNS(name, tokens)
		evicted++
	}
	return evicted, nil
}

// evictAdminBoundary returns the deepest resident node under root, scanning
// depth-first (child-map order; the choice among equal-depth nodes is arbitrary by
// design - every candidate is a legal subtree eviction), plus its token depth.
// nil when root owns no children. Unsynchronized exactly like Stats(): reads
// without mutation or locking.
func evictAdminBoundary(root *node) (*node, int) {
	if root == nil || len(root.children) == 0 {
		return nil, 0
	}
	stack := make([]*node, 0, len(root.children))
	for _, c := range root.children {
		stack = append(stack, c)
	}
	var best *node
	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if best == nil || n.plen > best.plen {
			best = n
		}
		for _, c := range n.children {
			stack = append(stack, c)
		}
	}
	if best == nil {
		return nil, 0
	}
	return best, best.plen
}

// DefaultNamespaceLabel is the namespace label ListResidentNamespaces reports for
// the tree's always-present default ("") namespace.
const DefaultNamespaceLabel = ""

// ListResidentNamespaces enumerates the namespaces present as virtual roots
// (default first, then sorted). The default namespace is ALWAYS reported: the tree
// keeps t.root resident unconditionally, so it is trivially present. Non-default
// roots are lazily materialized and pruned when empty, so their presence tracks
// namespace residency. A nil tree errors - the engine has no global identity
// registry to derive a namespace list from without a tree (no reflection, no
// unsafe reads), so the listing is impossible rather than an empty result.
func ListResidentNamespaces(t *Tree) ([]string, error) {
	if t == nil {
		return nil, errors.New("radixkv: engine lacks a default-namespace residency listing primitive (nil tree)")
	}
	names := make([]string, 0, len(t.nsRoots)+1)
	for ns := range t.nsRoots {
		names = append(names, ns)
	}
	sort.Strings(names)
	return append([]string{DefaultNamespaceLabel}, names...), nil
}
