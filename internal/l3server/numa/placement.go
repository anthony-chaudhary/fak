package numa

import "sort"

// NodeSpec describes a NUMA node's capacity for placement decisions.
// This is the sysfs-free input to PlaceShards, making the algorithm testable.
type NodeSpec struct {
	ID      int
	MemGB   float64 // total memory in GB
	HasCPUs bool    // true for full nodes, false for memory-only (CXL/HBM)
}

// PlaceShards distributes numShards across nodes described by specs, respecting
// each node's 90%-capacity limit. Returns the assignment slice and a list of
// memory-only node IDs that received shards.
//
// Algorithm (3 phases):
//  1. Fill CPU-bearing nodes first (up to 90% capacity each, round-robin)
//  2. Overflow to memory-only nodes (same strategy)
//  3. Last resort: put remaining on the largest node (best effort — may exceed 90%;
//     the caller should detect and warn about this oversubscription)
func PlaceShards(specs []NodeSpec, numShards int, perShardBytes uint64) (assignment []int, memOnlyUsed []int) {
	type nodeInfo struct {
		id        int
		maxShards int
		memGB     float64
		hasCPUs   bool
	}

	var cpuNodes, memOnlyNodes []nodeInfo
	for _, s := range specs {
		nodeBytes := uint64(s.MemGB * float64(1<<30))
		avail90 := nodeBytes * 9 / 10
		maxS := int(avail90 / perShardBytes)
		ni := nodeInfo{id: s.ID, maxShards: maxS, memGB: s.MemGB, hasCPUs: s.HasCPUs}
		if s.HasCPUs {
			cpuNodes = append(cpuNodes, ni)
		} else {
			memOnlyNodes = append(memOnlyNodes, ni)
		}
	}

	orderedNodes := append(cpuNodes, memOnlyNodes...)

	corrected := make([]int, numShards)
	assigned := make(map[int]int)
	idx := 0
	placed := 0

	// Phase 1: fill CPU-bearing nodes
	for shard := 0; shard < numShards; shard++ {
		ok := false
		for tries := 0; tries < len(cpuNodes); tries++ {
			ni := cpuNodes[(idx+tries)%len(cpuNodes)]
			if assigned[ni.id] < ni.maxShards {
				corrected[shard] = ni.id
				assigned[ni.id]++
				idx = (idx + tries + 1) % len(cpuNodes)
				ok = true
				placed++
				break
			}
		}
		if !ok {
			break
		}
	}

	// Phase 2: overflow to memory-only nodes
	if placed < numShards {
		moIdx := 0
		for shard := placed; shard < numShards; shard++ {
			ok := false
			for tries := 0; tries < len(memOnlyNodes); tries++ {
				ni := memOnlyNodes[(moIdx+tries)%len(memOnlyNodes)]
				if assigned[ni.id] < ni.maxShards {
					corrected[shard] = ni.id
					assigned[ni.id]++
					moIdx = (moIdx + tries + 1) % len(memOnlyNodes)
					ok = true
					placed++
					break
				}
			}
			if !ok {
				break
			}
		}
	}

	// Phase 3: last resort — put overflow on largest node.
	// This is best-effort and may exceed the 90% threshold. The caller
	// (redistributeByCapacity) detects and warns about this case.
	if placed < numShards && len(orderedNodes) > 0 {
		best := orderedNodes[0]
		for _, ni := range orderedNodes[1:] {
			if ni.memGB > best.memGB {
				best = ni
			}
		}
		for shard := placed; shard < numShards; shard++ {
			corrected[shard] = best.id
			assigned[best.id]++
		}
	}

	// Collect memory-only nodes that received shards
	for _, ni := range memOnlyNodes {
		if assigned[ni.id] > 0 {
			memOnlyUsed = append(memOnlyUsed, ni.id)
		}
	}

	return corrected, memOnlyUsed
}

// NodeBudget describes a per-NUMA-node shard budget.
type NodeBudget struct {
	NodeID    int
	Shards    int
	BytesNeed uint64
}

// NodeShardBudgets computes per-node slab bytes from a shard→node assignment.
// Used to feed into per-node hugepage distribution.
func NodeShardBudgets(assignment []int, perShardBytes uint64) []NodeBudget {
	counts := make(map[int]int)
	for _, node := range assignment {
		if node >= 0 {
			counts[node]++
		}
	}
	budgets := make([]NodeBudget, 0, len(counts))
	for node, cnt := range counts {
		budgets = append(budgets, NodeBudget{
			NodeID:    node,
			Shards:    cnt,
			BytesNeed: uint64(cnt) * perShardBytes,
		})
	}
	sort.Slice(budgets, func(i, j int) bool { return budgets[i].NodeID < budgets[j].NodeID })
	return budgets
}
