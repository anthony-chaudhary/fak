//go:build linux

package numa

import (
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"unsafe"
)

// Topology holds the detected NUMA topology of the system.
type Topology struct {
	Nodes    []Node         // one per NUMA node
	DevNode  map[string]int // RDMA device name → NUMA node (e.g. "mlx5_0" → 0)
	NumNodes int
}

// NodeType classifies a NUMA node by its capabilities.
type NodeType int

const (
	// NodeTypeFull has both CPUs and memory — ideal for shard placement.
	NodeTypeFull NodeType = iota
	// NodeTypeMemoryOnly has memory but no CPUs (CXL, HBM, disaggregated).
	// Shards placed here can't pin goroutines; every op crosses the interconnect.
	NodeTypeMemoryOnly
	// NodeTypeCPUOnly has CPUs but negligible memory — not useful for shards.
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

// Node represents a single NUMA node.
type Node struct {
	ID   int
	CPUs []int // OS CPU IDs belonging to this node
	Type NodeType
}

// Detect reads the NUMA topology from sysfs. Returns a nil Topology if
// NUMA is not available (single-socket, container, etc.) — callers should
// treat nil as "no pinning needed".
func Detect() *Topology {
	nodes, err := discoverNodes()
	if err != nil {
		log.Printf("[numa] could not read NUMA topology from sysfs: %v — pinning disabled", err)
		return nil
	}
	if len(nodes) <= 1 {
		if len(nodes) == 1 {
			log.Printf("[numa] single NUMA node detected (node %d, %d CPUs) — pinning not needed", nodes[0].ID, len(nodes[0].CPUs))
		} else {
			log.Printf("[numa] no NUMA nodes found in sysfs — pinning disabled")
		}
		return nil
	}

	t := &Topology{
		Nodes:    nodes,
		DevNode:  make(map[string]int),
		NumNodes: len(nodes),
	}

	// Classify nodes by capability
	for i := range nodes {
		nodeKB := NodeMemoryKB(nodes[i].ID)
		hasCPUs := len(nodes[i].CPUs) > 0
		hasMem := nodeKB > 0
		switch {
		case hasCPUs && hasMem:
			nodes[i].Type = NodeTypeFull
		case hasMem:
			nodes[i].Type = NodeTypeMemoryOnly
		default:
			nodes[i].Type = NodeTypeCPUOnly
		}
	}

	log.Printf("[numa] detected %d NUMA nodes", len(nodes))
	for _, n := range nodes {
		memKB := NodeMemoryKB(n.ID)
		log.Printf("[numa]   node %d: %s, CPUs %v, memory %.1f GB", n.ID, n.Type, n.CPUs, float64(memKB)/(1024*1024))
	}

	return t
}

// AddDevice reads the NUMA node for an RDMA device from sysfs and records it.
func (t *Topology) AddDevice(rdmaDevice string) int {
	if t == nil {
		return -1
	}

	path := fmt.Sprintf("/sys/class/infiniband/%s/device/numa_node", rdmaDevice)
	data, err := os.ReadFile(path)
	if err != nil {
		log.Printf("[numa] could not read NUMA node for %s: %v", rdmaDevice, err)
		return -1
	}

	node, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil || node < 0 {
		log.Printf("[numa] invalid NUMA node for %s: %q", rdmaDevice, strings.TrimSpace(string(data)))
		return -1
	}

	t.DevNode[rdmaDevice] = node
	log.Printf("[numa] RDMA device %s → NUMA node %d", rdmaDevice, node)
	return node
}

// CPUsForNode returns the CPU set for the given NUMA node, or nil if unknown.
func (t *Topology) CPUsForNode(node int) []int {
	if t == nil {
		return nil
	}
	for _, n := range t.Nodes {
		if n.ID == node {
			return n.CPUs
		}
	}
	return nil
}

// cpuSetSize is the size of the kernel cpu_set_t in bytes (1024 bits = 128 bytes).
const cpuSetSize = 128

// PinCurrentThread pins the calling OS thread to the CPUs of the given NUMA
// node. Must be called after runtime.LockOSThread(). Returns an error if
// pinning fails — callers should log and continue (non-fatal).
func (t *Topology) PinCurrentThread(node int) error {
	if t == nil {
		return nil
	}

	cpus := t.CPUsForNode(node)
	if len(cpus) == 0 {
		return fmt.Errorf("no CPUs for NUMA node %d", node)
	}

	// Build a CPU bitmask (1024 bits = 128 bytes, matching kernel's CPU_SETSIZE)
	var mask [cpuSetSize / 8]uint64 // 16 uint64s = 128 bytes = 1024 bits
	for _, cpu := range cpus {
		if cpu >= cpuSetSize*8 {
			continue // skip CPUs beyond mask size
		}
		mask[cpu/64] |= 1 << (uint(cpu) % 64)
	}

	// sched_setaffinity(pid=0 means current thread, cpusetsize, mask)
	_, _, errno := syscall.RawSyscall(
		syscall.SYS_SCHED_SETAFFINITY,
		0, // pid=0 → current thread
		uintptr(len(mask)*8),
		uintptr(unsafe.Pointer(&mask[0])),
	)
	if errno != 0 {
		return fmt.Errorf("sched_setaffinity(node=%d): %v", node, errno)
	}
	return nil
}

// SetMemPolicy sets the memory allocation policy for the current thread to
// prefer the given NUMA node. Uses MPOL_BIND to allocate only from the
// specified node. Must be called before large allocations (slab regions).
func SetMemPolicy(node int) error {
	if node < 0 {
		return nil
	}

	// Build nodemask — one bit per node
	const bitsPerLong = int(unsafe.Sizeof(uintptr(0))) * 8
	maxNode := node + 1
	maskLen := (maxNode + bitsPerLong - 1) / bitsPerLong
	mask := make([]uintptr, maskLen)
	mask[node/bitsPerLong] |= 1 << (uint(node) % uint(bitsPerLong))

	// SYS_SET_MEMPOLICY = 238 on amd64
	// MPOL_BIND = 2
	_, _, errno := syscall.RawSyscall(
		238, // SYS_SET_MEMPOLICY
		2,   // MPOL_BIND
		uintptr(unsafe.Pointer(&mask[0])),
		uintptr(maxNode+1),
	)
	if errno != 0 {
		return fmt.Errorf("set_mempolicy(MPOL_BIND, node=%d): %v", node, errno)
	}
	return nil
}

// ResetMemPolicy restores the default memory policy (MPOL_DEFAULT).
func ResetMemPolicy() {
	syscall.RawSyscall(238, 0, 0, 0) // SYS_SET_MEMPOLICY, MPOL_DEFAULT=0
}

// AssignShards distributes N shards across NUMA nodes, preferring the NIC's
// node. Returns a slice where result[shardID] = NUMA node.
//
// Strategy: if there's a single NIC, all shards go to that NIC's NUMA node.
// With multiple NICs on different nodes, shards are split proportionally.
func (t *Topology) AssignShards(numShards int) []int {
	if t == nil {
		result := make([]int, numShards)
		for i := range result {
			result[i] = -1
		}
		return result
	}

	// Collect unique NIC NUMA nodes
	nicNodes := make(map[int]bool)
	for _, node := range t.DevNode {
		if node >= 0 {
			nicNodes[node] = true
		}
	}

	result := make([]int, numShards)

	if len(nicNodes) == 0 {
		// No NIC info — distribute round-robin across all nodes
		for i := 0; i < numShards; i++ {
			result[i] = t.Nodes[i%len(t.Nodes)].ID
		}
	} else if len(nicNodes) == 1 {
		// Single NIC node — all shards on that node
		var node int
		for n := range nicNodes {
			node = n
		}
		for i := range result {
			result[i] = node
		}
	} else {
		// Multiple NIC nodes — round-robin across NIC nodes only
		// Sort for deterministic assignment across restarts.
		nodes := make([]int, 0, len(nicNodes))
		for n := range nicNodes {
			nodes = append(nodes, n)
		}
		sort.Ints(nodes)
		for i := 0; i < numShards; i++ {
			result[i] = nodes[i%len(nodes)]
		}
	}

	return result
}

// NodeMemoryKB reads the total memory for a NUMA node from sysfs.
// Returns 0 if the node's meminfo is unreadable.
func NodeMemoryKB(node int) uint64 {
	path := fmt.Sprintf("/sys/devices/system/node/node%d/meminfo", node)
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		// Format: "Node 0 MemTotal:       263174212 kB"
		if strings.Contains(line, "MemTotal") {
			fields := strings.Fields(line)
			if len(fields) >= 4 {
				kb, err := strconv.ParseUint(fields[3], 10, 64)
				if err == nil {
					return kb
				}
			}
		}
	}
	return 0
}

// ValidateNodeBudgets checks whether the per-node shard assignment fits within
// each NUMA node's physical memory. Returns a warning message and a corrected
// assignment (round-robin across all nodes) if any node is oversubscribed.
// perShardBytes is the slab memory per shard (totalMem / numShards).
func (t *Topology) ValidateNodeBudgets(assignment []int, perShardBytes uint64) (corrected []int, warning string) {
	if t == nil {
		return assignment, ""
	}

	// Count shards per node
	nodeCounts := make(map[int]int)
	for _, n := range assignment {
		if n >= 0 {
			nodeCounts[n]++
		}
	}

	// Check each node's budget
	for node, count := range nodeCounts {
		nodeKB := NodeMemoryKB(node)
		if nodeKB == 0 {
			continue
		}
		nodeBytes := nodeKB * 1024
		budgetBytes := uint64(count) * perShardBytes
		// 90% threshold — leave 10% for OS and other processes on this node
		if budgetBytes > nodeBytes*9/10 {
			// Oversubscribed — redistribute proportionally to node memory
			corrected, warning = t.redistributeByCapacity(len(assignment), perShardBytes, node, count, budgetBytes, nodeBytes)
			return corrected, warning
		}
	}

	return assignment, ""
}

// redistributeByCapacity assigns shards across NUMA nodes proportionally to
// each node's available memory (90% threshold), ensuring no node is oversubscribed.
// Delegates to PlaceShards (placement.go) for the actual 3-phase algorithm.
func (t *Topology) redistributeByCapacity(numShards int, perShardBytes uint64, triggerNode, triggerCount int, triggerBudget, triggerNodeBytes uint64) ([]int, string) {
	// Build NodeSpec slice from live sysfs data
	var specs []NodeSpec
	for _, n := range t.Nodes {
		nodeKB := NodeMemoryKB(n.ID)
		if nodeKB == 0 {
			continue
		}
		memGB := float64(nodeKB) / (1024 * 1024)
		hasCPUs := len(n.CPUs) > 0
		if !hasCPUs {
			log.Printf("[numa] node %d is memory-only (%.1f GB, 0 CPUs) — will use as overflow; shard goroutines cannot pin there",
				n.ID, memGB)
		}
		specs = append(specs, NodeSpec{
			ID:      n.ID,
			MemGB:   memGB,
			HasCPUs: hasCPUs,
		})
	}

	corrected, memOnlyUsed := PlaceShards(specs, numShards, perShardBytes)

	// Build warning message
	warning := fmt.Sprintf("%d shard(s) on node %d require %.1f GB slab but node has %.1f GB (90%% threshold: %.1f GB) — "+
		"redistributing by capacity across %d NUMA nodes. "+
		"Cross-node access may add ~100ns latency; add a NIC per socket for optimal placement.",
		triggerCount, triggerNode,
		float64(triggerBudget)/(1<<30),
		float64(triggerNodeBytes)/(1<<30),
		float64(triggerNodeBytes*9/10)/(1<<30),
		len(t.Nodes))

	if len(memOnlyUsed) > 0 {
		warning += fmt.Sprintf(" NOTE: node(s) %v are memory-only (0 CPUs) — shards there run unpinned with cross-node latency on every op.", memOnlyUsed)
	}

	// Check for phase 3 oversubscription (shards placed beyond 90% on a single node)
	budgets := NodeShardBudgets(corrected, perShardBytes)
	for _, b := range budgets {
		nodeKB := NodeMemoryKB(b.NodeID)
		if nodeKB == 0 {
			continue
		}
		nodeBytes := nodeKB * 1024
		if b.BytesNeed > nodeBytes*9/10 {
			warning += fmt.Sprintf(" WARNING: node %d still oversubscribed at %.1f GB / %.1f GB after redistribution — workload may not fit available NUMA memory.",
				b.NodeID, float64(b.BytesNeed)/float64(1<<30), float64(nodeBytes)/float64(1<<30))
		}
	}

	return corrected, warning
}

// discoverNodes reads NUMA node topology from sysfs.
func discoverNodes() ([]Node, error) {
	entries, err := os.ReadDir("/sys/devices/system/node")
	if err != nil {
		return nil, err
	}

	var nodes []Node
	for _, e := range entries {
		if !e.IsDir() || !strings.HasPrefix(e.Name(), "node") {
			continue
		}
		idStr := strings.TrimPrefix(e.Name(), "node")
		id, err := strconv.Atoi(idStr)
		if err != nil {
			continue
		}

		cpuList, err := os.ReadFile(fmt.Sprintf("/sys/devices/system/node/%s/cpulist", e.Name()))
		if err != nil {
			log.Printf("[numa] could not read cpulist for %s: %v — skipping", e.Name(), err)
			continue
		}

		cpus := parseCPUList(strings.TrimSpace(string(cpuList)))
		nodes = append(nodes, Node{ID: id, CPUs: cpus})
	}

	return nodes, nil
}

// parseCPUList parses a Linux CPU list string like "0-3,8-11" into individual CPU IDs.
func parseCPUList(s string) []int {
	if s == "" {
		return nil
	}
	var cpus []int
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if strings.Contains(part, "-") {
			bounds := strings.SplitN(part, "-", 2)
			lo, _ := strconv.Atoi(bounds[0])
			hi, _ := strconv.Atoi(bounds[1])
			for i := lo; i <= hi; i++ {
				cpus = append(cpus, i)
			}
		} else {
			cpu, _ := strconv.Atoi(part)
			cpus = append(cpus, cpu)
		}
	}
	return cpus
}

var globalTopology *Topology

// SetGlobal stores the detected topology for use by PinThread.
func SetGlobal(t *Topology) {
	globalTopology = t
}

// PinThread pins the calling OS thread to the CPUs of the given NUMA node
// using the global detected topology. Must be called after runtime.LockOSThread().
// If no topology was detected or node is -1, this is a no-op.
func PinThread(node int) error {
	if node < 0 || globalTopology == nil {
		return nil
	}
	return globalTopology.PinCurrentThread(node)
}
