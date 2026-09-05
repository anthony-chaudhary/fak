//go:build !linux

package numa

import "log"

// NodeType classifies a NUMA node by its capabilities.
type NodeType int

const (
	NodeTypeFull NodeType = iota
	NodeTypeMemoryOnly
	NodeTypeCPUOnly
)

func (nt NodeType) String() string {
	switch nt {
	case NodeTypeFull:
		return "full"
	case NodeTypeMemoryOnly:
		return "memory-only"
	case NodeTypeCPUOnly:
		return "cpu-only"
	default:
		return "unknown"
	}
}

// Topology holds the detected NUMA topology — nil on non-Linux.
type Topology struct {
	Nodes    []Node
	DevNode  map[string]int
	NumNodes int
}

// Node represents a single NUMA node.
type Node struct {
	ID   int
	CPUs []int
	Type NodeType
}

// Detect is a no-op on non-Linux platforms.
func Detect() *Topology {
	log.Printf("[numa] NUMA pinning not available on this platform")
	return nil
}

func (t *Topology) AddDevice(rdmaDevice string) int { return -1 }
func (t *Topology) CPUsForNode(node int) []int      { return nil }
func (t *Topology) PinCurrentThread(node int) error { return nil }
func (t *Topology) AssignShards(numShards int) []int {
	result := make([]int, numShards)
	for i := range result {
		result[i] = -1
	}
	return result
}

func SetMemPolicy(node int) error { return nil }
func ResetMemPolicy()             {}

// NodeMemoryKB is a no-op on non-Linux platforms.
func NodeMemoryKB(node int) uint64 { return 0 }

// ValidateNodeBudgets is a no-op on non-Linux platforms.
func (t *Topology) ValidateNodeBudgets(assignment []int, perShardBytes uint64) ([]int, string) {
	return assignment, ""
}

// PinThread is a no-op on non-Linux platforms.
func PinThread(node int) error { return nil }

// SetGlobal is a no-op on non-Linux platforms.
func SetGlobal(t *Topology) {}
