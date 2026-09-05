//go:build linux

package compute

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// detectHostNUMATopologySysfs reads CPU-bearing NUMA nodes from sysfs root.
func detectHostNUMATopologySysfs(root string) []NUMANodeTopology {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	var nodes []NUMANodeTopology
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
			continue
		}
		cpusStr := strings.TrimSpace(string(cpulist))
		if cpusStr == "" {
			continue // memory-only nodes are not CPU scheduling targets
		}
		cpus := parseNodeList(cpusStr)
		if len(cpus) == 0 {
			continue
		}
		sort.Ints(cpus)
		nodes = append(nodes, NUMANodeTopology{
			NodeID: id,
			CPUs:   cpus,
		})
	}
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].NodeID < nodes[j].NodeID })
	for i := 1; i < len(nodes); i++ {
		if nodes[i-1].NodeID == nodes[i].NodeID {
			return nil
		}
	}
	return nodes
}

func detectHostNUMATopologyPlatform() []NUMANodeTopology {
	return detectHostNUMATopologySysfs(sysNodeRoot)
}
