//go:build !linux

package compute

// Per-node CPU and free-memory topology is not available through the supported
// seams on other platforms. Eligibility therefore fails closed.
func readNUMAReplicaSnapshot() numaReplicaSnapshot {
	return numaReplicaSnapshot{policy: numaReplicaPolicyUnsupported}
}
