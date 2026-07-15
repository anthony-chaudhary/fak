//go:build linux

package compute

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

func readNUMAReplicaSnapshot() numaReplicaSnapshot {
	status := ReadHostMemStatus()
	policy := numaReplicaPolicyUnsupported
	if status.PolicyLabel != "" {
		policy = numaReplicaPolicyUnconstrained
		if status.Constrained {
			policy = numaReplicaPolicyConstrained
		}
	}
	nodes, known := readNUMAReplicaNodes(sysNodeRoot)
	return numaReplicaSnapshot{policy: policy, topologyKnown: known, nodes: nodes}
}

// readNUMAReplicaNodes reads every CPU-bearing node rather than reusing the
// fail-open far-memory probe. A target with unreadable memory must survive as an
// unknown record so the planner can refuse instead of silently omitting it.
func readNUMAReplicaNodes(root string) ([]numaReplicaNode, bool) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, false
	}
	nodes := make([]numaReplicaNode, 0, len(entries))
	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasPrefix(name, "node") {
			continue
		}
		id, err := strconv.Atoi(strings.TrimPrefix(name, "node"))
		if err != nil || id < 0 || !entry.IsDir() {
			continue
		}
		dir := filepath.Join(root, name)
		cpulist, err := os.ReadFile(filepath.Join(dir, "cpulist"))
		if err != nil {
			return nil, false
		}
		cpus := strings.TrimSpace(string(cpulist))
		if cpus == "" {
			continue // memory-only nodes are not replica targets
		}
		if len(parseNodeList(cpus)) == 0 {
			return nil, false
		}
		_, free, memoryKnown := parseNodeMeminfo(filepath.Join(dir, "meminfo"))
		nodes = append(nodes, numaReplicaNode{id: id, freeBytes: free, memoryKnown: memoryKnown})
	}
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].id < nodes[j].id })
	for i := 1; i < len(nodes); i++ {
		if nodes[i-1].id == nodes[i].id {
			return nil, false
		}
	}
	return nodes, true
}
