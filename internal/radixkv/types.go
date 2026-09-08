package radixkv

import (
	"errors"
)

var ErrSnapshotByteBudget = errors.New("radixkv: snapshot resident-byte budget exceeded")
var ErrHostSnapshotByteBudget = errors.New("radixkv: host snapshot resident-byte budget exceeded")

// SnapshotTier is the physical source that satisfied a complete-prefix lookup.
// The zero value is a miss; callers must never infer a hit without an owned
// PrefixSnapshot result.
type SnapshotTier string

const (
	SnapshotTierMiss     SnapshotTier = ""
	SnapshotTierDeviceL1 SnapshotTier = "device_l1"
	SnapshotTierHostL2   SnapshotTier = "host_dram_l2"
	SnapshotTierRemoteL3 SnapshotTier = "remote_http_l3"
)

// NodeState models the lifecycle and computation states of a radix tree node.
type NodeState uint32

const (
	// NodeWarm indicates the node has completed prefill and holds a valid KV cache.
	NodeWarm NodeState = iota
	// NodeComputingPrefill indicates the node is actively undergoing prefill by a leader subagent.
	NodeComputingPrefill
	// NodeFailed indicates prefill computation failed or was abandoned.
	NodeFailed
	// NodeEvicted indicates the node has been evicted from the tree.
	NodeEvicted
)

func (s NodeState) String() string {
	switch s {
	case NodeWarm:
		return "warm"
	case NodeComputingPrefill:
		return "computing_prefill"
	case NodeFailed:
		return "failed"
	case NodeEvicted:
		return "evicted"
	default:
		return "unknown"
	}
}
