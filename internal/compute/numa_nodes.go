package compute

// OnlineNUMANodes returns the host's online NUMA node ids (ascending), or nil when the
// topology is unreadable or the platform exposes none. It is the exported accessor over the
// same /sys probe the interleave planner uses, so a caller outside this package (e.g. the
// matmul pool sizing its per-node work shards) reads ONE definition of the node set rather
// than re-deriving the topology.
func OnlineNUMANodes() []int { return append([]int(nil), onlineNUMANodes()...) }

// OnlineNUMANodeCount reports how many online NUMA nodes the host exposes (0 when unknown).
// A count >= 2 is the precondition for any node-sharded scheduling decision; 0 or 1 means the
// caller must keep its single-pool behaviour.
func OnlineNUMANodeCount() int { return len(onlineNUMANodes()) }
