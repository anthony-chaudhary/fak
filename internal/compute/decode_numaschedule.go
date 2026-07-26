package compute

import "sort"

// decode_numaschedule.go — the scheduler that DecodeNUMALocality (a witness) and
// PlanNUMAReplicas (a capacity plan) both deliberately stop short of. The locality
// model quantifies the fabric-roaming collapse ("workers placed on other nodes stream
// the same bytes across the fabric — adding latency, not bandwidth") and the replica
// planner says WHICH nodes can hold a byte-identical weight copy; neither PLACES a
// worker. This file closes that gap: given the CPU-bearing node topology and the decode
// worker count, it assigns each worker to a node and to that node's CPU set, so a worker
// pinned to CPUs(worker) and reading Replica(NodeID) streams weights from LOCAL memory.
//
// Why this is the lever past the witnessed interleave regime. Interleave (#4974) spreads
// the resident weights across all nodes so every core reaches memory at AGGREGATE
// bandwidth — the witnessed 2.61x over the default node-0 first-touch cell — but ~(N-1)/N
// of each core's reads still land on a REMOTE node and pay the inter-die fabric latency.
// A per-node replica removes that remainder: every worker reads its own node's copy, so
// the decode stream is BOTH aggregate-bandwidth AND fully local. This scheduler is the
// worker-placement half of that read path; PlanNUMAReplicas is the allocation half.
//
// The assignment is intentionally decoupled from the GEMV row-partition. Because each
// node holds a FULL byte-identical replica, any worker on node k may read any row from
// replica k — so correctness does not depend on which rows a worker draws, unlike a
// sharded placement that must mbind each worker's exact row-range. That decoupling is the
// robustness win the committed replica planner is designed around, and it lets this
// scheduler be a pure function of (topology, workers) with no coupling to parFor bounds.

// NUMANodeTopology is one CPU-bearing NUMA node's identity and its online CPU ids (the
// node's `cpulist`). It is the per-node input the scheduler places workers onto; a
// memory-only node (empty CPUs) is never a scheduling target and must be omitted upstream.
type NUMANodeTopology struct {
	NodeID int
	CPUs   []int
}

// DecodeWorkerPlacement pins one decode worker: it may run only on CPUs (its node's core
// set, for sched_setaffinity) and it reads the resident weights from the replica on NodeID.
// CPUs is a fresh copy in ascending order so a caller cannot mutate the node topology through it.
type DecodeWorkerPlacement struct {
	Worker int
	NodeID int
	CPUs   []int
}

// DecodeNUMAScheduleReason is the closed vocabulary for schedule eligibility. A reason
// other than DecodeNUMAScheduleEligible is a refusal: Placements is nil and the caller
// must fall back to its ordinary (unpinned, single-residency) decode path.
type DecodeNUMAScheduleReason string

const (
	DecodeNUMAScheduleEligible          DecodeNUMAScheduleReason = "eligible"
	DecodeNUMAScheduleInvalidWorkers    DecodeNUMAScheduleReason = "invalid_workers"
	DecodeNUMAScheduleInsufficientNodes DecodeNUMAScheduleReason = "insufficient_nodes"
	DecodeNUMAScheduleInvalidTopology   DecodeNUMAScheduleReason = "invalid_topology"
)

// DecodeNUMASchedule is a worker→node placement verdict. Placements is populated only
// when Eligible is true, in ascending Worker order. PerNodeWorkers is index-aligned to
// the ascending-NodeID node order and counts the workers homed on each node. Oversubscribed
// is set when some node received more workers than it has CPUs — the workers still place
// (the kernel time-slices them within the node) but they cannot each hold a distinct core,
// so a caller tuning for peak streaming width should reduce the worker count to fit.
type DecodeNUMASchedule struct {
	Eligible       bool
	Reason         DecodeNUMAScheduleReason
	Workers        int
	Nodes          int
	Placements     []DecodeWorkerPlacement
	PerNodeWorkers []int
	Oversubscribed bool
}

// ScheduleDecodeNUMA assigns `workers` decode workers across the CPU-bearing NUMA nodes so
// each worker is pinned to one node's CPU set and reads that node's replica. Workers are
// distributed as evenly as possible in ascending-NodeID order, block-wise (a node's workers
// occupy a contiguous worker-index span), which keeps neighbouring worker indices co-resident
// on a node. Guards: fewer than two nodes is a refusal (nothing to place across — the caller's
// single-node path already streams locally); a non-positive worker count, an empty CPU set on
// any node, or a duplicate/negative node id is a refusal with an empty plan.
func ScheduleDecodeNUMA(topology []NUMANodeTopology, workers int) DecodeNUMASchedule {
	out := DecodeNUMASchedule{Reason: DecodeNUMAScheduleInvalidTopology, Workers: workers, Nodes: len(topology)}
	refuse := func(r DecodeNUMAScheduleReason) DecodeNUMASchedule {
		out.Eligible = false
		out.Reason = r
		out.Placements = nil
		out.PerNodeWorkers = nil
		return out
	}
	if workers <= 0 {
		return refuse(DecodeNUMAScheduleInvalidWorkers)
	}
	if len(topology) < 2 {
		return refuse(DecodeNUMAScheduleInsufficientNodes)
	}

	// Copy and sort by NodeID so the schedule is a deterministic function of the node SET,
	// independent of the order the caller discovered the nodes in.
	nodes := make([]NUMANodeTopology, len(topology))
	copy(nodes, topology)
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].NodeID < nodes[j].NodeID })
	for i, n := range nodes {
		if n.NodeID < 0 || len(n.CPUs) == 0 {
			return refuse(DecodeNUMAScheduleInvalidTopology)
		}
		if i > 0 && nodes[i-1].NodeID == n.NodeID {
			return refuse(DecodeNUMAScheduleInvalidTopology)
		}
	}

	// Even block distribution: the first (workers mod nodes) nodes take one extra worker.
	nNodes := len(nodes)
	base := workers / nNodes
	extra := workers % nNodes
	out.PerNodeWorkers = make([]int, nNodes)
	out.Placements = make([]DecodeWorkerPlacement, 0, workers)
	w := 0
	for i, node := range nodes {
		count := base
		if i < extra {
			count++
		}
		out.PerNodeWorkers[i] = count
		if count > len(node.CPUs) {
			out.Oversubscribed = true
		}
		cpus := append([]int(nil), node.CPUs...)
		sort.Ints(cpus)
		for c := 0; c < count; c++ {
			out.Placements = append(out.Placements, DecodeWorkerPlacement{
				Worker: w,
				NodeID: node.NodeID,
				CPUs:   cpus,
			})
			w++
		}
	}

	out.Eligible = true
	out.Reason = DecodeNUMAScheduleEligible
	return out
}
