package compute

import "sort"

// NUMAReplicaPlanReason is the closed vocabulary for replica eligibility. A
// reason other than NUMAReplicaPlanEligible is a refusal: callers must not
// allocate or bind any replica from that result.
type NUMAReplicaPlanReason string

const (
	NUMAReplicaPlanEligible           NUMAReplicaPlanReason = "eligible"
	NUMAReplicaPlanUnsupported        NUMAReplicaPlanReason = "unsupported"
	NUMAReplicaPlanConstrainedPolicy  NUMAReplicaPlanReason = "constrained_policy"
	NUMAReplicaPlanInvalidSize        NUMAReplicaPlanReason = "invalid_size"
	NUMAReplicaPlanInsufficientNodes  NUMAReplicaPlanReason = "insufficient_nodes"
	NUMAReplicaPlanUnknownNodeMemory  NUMAReplicaPlanReason = "unknown_node_memory"
	NUMAReplicaPlanArithmeticOverflow NUMAReplicaPlanReason = "arithmetic_overflow"
	NUMAReplicaPlanNodeBelowFloor     NUMAReplicaPlanReason = "node_below_floor"
)

// NUMAReplicaTarget is the exact metadata for one future node-local replica.
// It describes capacity only; it owns no memory and carries no placement or
// scheduling side effect.
type NUMAReplicaTarget struct {
	NodeID        int
	ReplicaBytes  int64
	ReserveBytes  int64
	RequiredBytes int64
	FreeBytes     int64
}

// NUMAReplicaPlan is a pre-allocation eligibility verdict. Targets is populated
// only when Eligible is true, in ascending NodeID order. In particular, a
// refusal never returns a partial plan that an allocator could accidentally use.
type NUMAReplicaPlan struct {
	Eligible             bool
	Reason               NUMAReplicaPlanReason
	ReplicaBytes         int64
	ReserveBytes         int64
	RequiredPerNodeBytes int64
	TotalReplicaBytes    int64
	TotalRequiredBytes   int64
	Targets              []NUMAReplicaTarget
}

// PlanNUMAReplicas reads the supported host topology and returns a pure metadata
// plan for one byte-identical replica on every CPU-bearing NUMA node. It never
// mmaps, allocates replica storage, binds memory, pins threads, or loads weights.
// Unsupported or incomplete host evidence is a refusal, not a guessed plan.
func PlanNUMAReplicas(replicaBytes, reserveBytes int64) NUMAReplicaPlan {
	return planNUMAReplicas(readNUMAReplicaSnapshot(), replicaBytes, reserveBytes)
}

type numaReplicaPolicy uint8

const (
	numaReplicaPolicyUnsupported numaReplicaPolicy = iota
	numaReplicaPolicyUnconstrained
	numaReplicaPolicyConstrained
)

type numaReplicaNode struct {
	id          int
	freeBytes   int64
	memoryKnown bool
}

type numaReplicaSnapshot struct {
	policy        numaReplicaPolicy
	topologyKnown bool
	nodes         []numaReplicaNode
}

func planNUMAReplicas(snapshot numaReplicaSnapshot, replicaBytes, reserveBytes int64) NUMAReplicaPlan {
	plan := NUMAReplicaPlan{
		Reason:       NUMAReplicaPlanUnsupported,
		ReplicaBytes: replicaBytes,
		ReserveBytes: reserveBytes,
	}
	refuse := func(reason NUMAReplicaPlanReason) NUMAReplicaPlan {
		plan.Eligible = false
		plan.Reason = reason
		plan.Targets = nil
		return plan
	}

	if replicaBytes <= 0 || reserveBytes < 0 {
		return refuse(NUMAReplicaPlanInvalidSize)
	}
	required, ok := checkedAddPositiveInt64(replicaBytes, reserveBytes)
	if !ok {
		return refuse(NUMAReplicaPlanArithmeticOverflow)
	}
	plan.RequiredPerNodeBytes = required

	if snapshot.policy == numaReplicaPolicyConstrained {
		return refuse(NUMAReplicaPlanConstrainedPolicy)
	}
	if snapshot.policy != numaReplicaPolicyUnconstrained || !snapshot.topologyKnown {
		return refuse(NUMAReplicaPlanUnsupported)
	}
	if len(snapshot.nodes) < 2 {
		return refuse(NUMAReplicaPlanInsufficientNodes)
	}

	// A malformed or duplicate node identity makes the topology non-exact. Check
	// without building Targets so every refusal remains an empty plan.
	for i, node := range snapshot.nodes {
		if node.id < 0 {
			return refuse(NUMAReplicaPlanUnsupported)
		}
		for j := 0; j < i; j++ {
			if snapshot.nodes[j].id == node.id {
				return refuse(NUMAReplicaPlanUnsupported)
			}
		}
	}
	for _, node := range snapshot.nodes {
		if !node.memoryKnown || node.freeBytes < 0 {
			return refuse(NUMAReplicaPlanUnknownNodeMemory)
		}
	}
	// Capacity is deliberately checked per target node. Summing free bytes first
	// would let a roomy peer mask one node that cannot hold its own replica.
	for _, node := range snapshot.nodes {
		if node.freeBytes < required {
			return refuse(NUMAReplicaPlanNodeBelowFloor)
		}
	}

	count := int64(len(snapshot.nodes))
	totalReplica, ok := checkedMulPositiveInt64(replicaBytes, count)
	if !ok {
		return refuse(NUMAReplicaPlanArithmeticOverflow)
	}
	totalRequired, ok := checkedMulPositiveInt64(required, count)
	if !ok {
		return refuse(NUMAReplicaPlanArithmeticOverflow)
	}

	targets := make([]NUMAReplicaTarget, len(snapshot.nodes))
	for i, node := range snapshot.nodes {
		targets[i] = NUMAReplicaTarget{
			NodeID:        node.id,
			ReplicaBytes:  replicaBytes,
			ReserveBytes:  reserveBytes,
			RequiredBytes: required,
			FreeBytes:     node.freeBytes,
		}
	}
	sort.Slice(targets, func(i, j int) bool { return targets[i].NodeID < targets[j].NodeID })

	plan.Eligible = true
	plan.Reason = NUMAReplicaPlanEligible
	plan.TotalReplicaBytes = totalReplica
	plan.TotalRequiredBytes = totalRequired
	plan.Targets = targets
	return plan
}

func checkedAddPositiveInt64(a, b int64) (int64, bool) {
	if a < 0 || b < 0 || a > int64(maxInt64Uint64)-b {
		return 0, false
	}
	return a + b, true
}

func checkedMulPositiveInt64(a, b int64) (int64, bool) {
	if a < 0 || b < 0 || (a != 0 && b > int64(maxInt64Uint64)/a) {
		return 0, false
	}
	return a * b, true
}
