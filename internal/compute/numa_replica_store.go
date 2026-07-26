package compute

import (
	"bytes"
	"errors"
	"fmt"
	"sort"
)

// numa_replica_store.go — the allocation half of the per-node replica read path, the lever
// that clears the witnessed interleave ceiling. PlanNUMAReplicas says WHICH nodes can hold a
// byte-identical copy (capacity only; it allocates nothing) and ScheduleDecodeNUMA says WHICH
// worker runs on which node. This file materialises the copies: one byte-identical replica of
// the resident weight bytes per CPU-bearing node, each bound to its node, so a worker pinned by
// the schedule reads weights from LOCAL memory and never crosses the inter-die fabric.
//
// Why replicas rather than a sharded mbind of the single slab: the decode GEMV row-partition
// (parFor over output rows) is not a stable contract — the worker→row mapping is an internal
// scheduling detail and prefill/decode partition differently. A FULL replica per node makes any
// worker on node k able to read any row from replica k, so locality holds regardless of which
// rows a worker draws. That decoupling is what the committed planner is designed around.
//
// Placement discipline: the linux allocator mmaps anonymous memory and binds it to its node
// BEFORE the copy, so the pages first-touch onto the target node. Binding after a copy would
// force a migration of the whole slab instead. Off the Go heap by construction: a 27B Q4_K
// model replicated across 8 nodes is ~120 GB, which must never be GC-scanned.

// errReplicaUnsupported is returned by the portable allocator path when a caller asks for
// node-bound storage on a platform with no mbind — the copy still succeeds (bytes are correct)
// but carries no placement, so only correctness, never a locality claim, may be inferred.
var errReplicaUnsupported = errors.New("compute: node-bound replica storage unsupported on this platform")

// NUMAReplicaSet owns one byte-identical copy of a source region per CPU-bearing NUMA node.
// Nodes are ascending and index-aligned with the copies. Bound counts the copies whose
// node binding actually took effect; on a platform without mbind it is 0 and the set is a
// correctness-only fallback (identical bytes, unmanaged placement).
type NUMAReplicaSet struct {
	nodes  []int
	copies [][]byte
	frees  []func() error
	bound  int
}

// Nodes returns the ascending node ids this set holds a replica for (a fresh copy).
func (s *NUMAReplicaSet) Nodes() []int {
	if s == nil {
		return nil
	}
	return append([]int(nil), s.nodes...)
}

// Len reports how many replicas the set holds.
func (s *NUMAReplicaSet) Len() int {
	if s == nil {
		return 0
	}
	return len(s.copies)
}

// Bound reports how many replicas were successfully bound to their node. Bound < Len means
// the set is correctness-safe but not fully local — a caller must not claim node locality.
func (s *NUMAReplicaSet) Bound() int {
	if s == nil {
		return 0
	}
	return s.bound
}

// For returns the replica bytes homed on node, or nil when this set holds no copy for it.
// The returned slice aliases the replica's backing store: it is read-only by contract (the
// decode GEMV only ever reads weights) and must not be mutated or retained past Free.
func (s *NUMAReplicaSet) For(node int) []byte {
	if s == nil {
		return nil
	}
	i := sort.SearchInts(s.nodes, node)
	if i < len(s.nodes) && s.nodes[i] == node {
		return s.copies[i]
	}
	return nil
}

// Label renders the set for a decode witness line, e.g. "replicas=8(bound=8,bytes=15032385536)".
func (s *NUMAReplicaSet) Label() string {
	if s == nil || len(s.copies) == 0 {
		return "replicas=none"
	}
	return fmt.Sprintf("replicas=%d(bound=%d,bytes=%d)", len(s.copies), s.bound, len(s.copies[0]))
}

// Free releases every replica's backing store. It is safe to call once; the set is unusable
// afterwards. The first release error is returned, but every replica is still attempted.
func (s *NUMAReplicaSet) Free() error {
	if s == nil {
		return nil
	}
	var first error
	for _, f := range s.frees {
		if f == nil {
			continue
		}
		if err := f(); err != nil && first == nil {
			first = err
		}
	}
	s.copies, s.frees, s.nodes, s.bound = nil, nil, nil, 0
	return first
}

// BuildNUMAReplicas materialises one byte-identical copy of src per eligible plan target and
// binds each to its node. It refuses (nil, error) on an ineligible plan, an empty src, or a
// plan whose per-target ReplicaBytes does not match len(src) — a size disagreement means the
// caller planned for different bytes than it is now replicating, which would silently
// under-allocate. On any allocation failure every already-built replica is freed, so a partial
// replica set is never returned.
func BuildNUMAReplicas(src []byte, plan NUMAReplicaPlan) (*NUMAReplicaSet, error) {
	if len(src) == 0 {
		return nil, errors.New("compute: empty source region has no replicas")
	}
	if !plan.Eligible {
		return nil, fmt.Errorf("compute: replica plan refused (reason=%s)", plan.Reason)
	}
	if len(plan.Targets) == 0 {
		return nil, errors.New("compute: eligible replica plan carries no targets")
	}
	if plan.ReplicaBytes != int64(len(src)) {
		return nil, fmt.Errorf("compute: plan replica bytes %d != source bytes %d", plan.ReplicaBytes, len(src))
	}

	set := &NUMAReplicaSet{}
	for _, t := range plan.Targets {
		region, free, bound, err := allocNodeRegion(len(src), t.NodeID)
		if err != nil {
			_ = set.Free()
			return nil, fmt.Errorf("compute: replica alloc on node %d: %w", t.NodeID, err)
		}
		// Copy AFTER the bind so the pages first-touch onto the target node.
		copy(region, src)
		set.nodes = append(set.nodes, t.NodeID)
		set.copies = append(set.copies, region)
		set.frees = append(set.frees, free)
		if bound {
			set.bound++
		}
	}
	// Targets arrive ascending from the planner; keep the invariant explicit for For()'s search.
	if !sort.IntsAreSorted(set.nodes) {
		_ = set.Free()
		return nil, errors.New("compute: replica targets are not in ascending node order")
	}
	return set, nil
}

// VerifyNUMAReplicas re-reads every replica and reports the first node whose bytes drift from
// src. It is the bit-identity witness for the read path: a replica that is not byte-identical
// would silently change decode output, so a caller may assert this after Build (and a test
// always does). Returns nil when every replica matches.
func VerifyNUMAReplicas(src []byte, set *NUMAReplicaSet) error {
	if set == nil {
		return errors.New("compute: nil replica set")
	}
	for i, node := range set.nodes {
		if !bytes.Equal(set.copies[i], src) {
			return fmt.Errorf("compute: replica on node %d is not byte-identical to source", node)
		}
	}
	return nil
}
