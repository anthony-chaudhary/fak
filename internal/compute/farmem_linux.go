//go:build linux

package compute

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// sysNodeRoot is the kernel's NUMA topology directory the far-memory probes read.
const sysNodeRoot = "/sys/devices/system/node"

// nodeMemory is one NUMA node's probed identity: its id, whether it has CPUs (a
// memory-only node is the kernel's representation of CPU-less expansion memory,
// CXL.mem the canonical instance), and its MemTotal/MemFree in bytes.
type nodeMemory struct {
	id      int
	hasCPUs bool
	total   int64
	free    int64
}

func numaFarMemory() (total, free int64, known bool) {
	total, free, known, _, _, _ = farMemoryFromNodes(readNodeMemory(sysNodeRoot))
	if !known {
		return 0, FreeUnknown, false
	}
	return total, free, true
}

func cxlMemory() (total, free int64, known bool) {
	_, _, _, total, free, known = farMemoryFromNodes(readNodeMemory(sysNodeRoot))
	if !known {
		return 0, FreeUnknown, false
	}
	return total, free, true
}

// farMemoryFromNodes splits a box's NUMA nodes into the two far-memory classes the
// tier ladder distinguishes:
//
//   - NUMA-far: CPU-ful nodes other than the LOCAL node, approximated as the
//     lowest-numbered CPU-ful node. That is a process-agnostic simplification (a
//     thread on socket 1 sees socket 0 as far), but tier pressure is a box-level
//     signal, not a per-thread one: "the other socket's DRAM" is what the demote
//     ladder cares about, and a single-socket box correctly reports no far tier.
//   - CXL: memory-only nodes (memory but no CPUs) — how the kernel exposes CPU-less
//     expansion memory onlined as system RAM, CXL.mem being the canonical instance.
//
// Each class fails open independently: no far node (or no memory-only node, or an
// unreadable topology) makes that class unknown rather than zero-capacity.
func farMemoryFromNodes(nodes []nodeMemory) (numaTotal, numaFree int64, numaKnown bool, cxlTotal, cxlFree int64, cxlKnown bool) {
	localID := -1
	for _, n := range nodes {
		if n.hasCPUs && (localID == -1 || n.id < localID) {
			localID = n.id
		}
	}
	for _, n := range nodes {
		if n.total <= 0 {
			continue
		}
		switch {
		case n.hasCPUs && n.id != localID:
			numaTotal += n.total
			numaFree += n.free
			numaKnown = true
		case !n.hasCPUs:
			cxlTotal += n.total
			cxlFree += n.free
			cxlKnown = true
		}
	}
	if !numaKnown {
		numaTotal, numaFree = 0, FreeUnknown
	}
	if !cxlKnown {
		cxlTotal, cxlFree = 0, FreeUnknown
	}
	return numaTotal, numaFree, numaKnown, cxlTotal, cxlFree, cxlKnown
}

// readNodeMemory enumerates root's node<N> directories into nodeMemory records. The
// root is a parameter so tests can point it at a synthetic tree; production callers
// pass sysNodeRoot. Unreadable entries are skipped (fail open) — a node this cannot
// parse simply does not contribute to any far-memory class.
func readNodeMemory(root string) []nodeMemory {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	var nodes []nodeMemory
	for _, e := range entries {
		name := e.Name()
		if !strings.HasPrefix(name, "node") {
			continue
		}
		id, err := strconv.Atoi(name[len("node"):])
		if err != nil || id < 0 {
			continue
		}
		dir := filepath.Join(root, name)
		total, free, ok := parseNodeMeminfo(filepath.Join(dir, "meminfo"))
		if !ok {
			continue
		}
		nodes = append(nodes, nodeMemory{
			id:      id,
			hasCPUs: nodeHasCPUs(filepath.Join(dir, "cpulist")),
			total:   total,
			free:    free,
		})
	}
	return nodes
}

// nodeHasCPUs reports whether a node's cpulist names any CPU. The kernel writes an
// empty (or whitespace-only) cpulist for a memory-only node. An unreadable cpulist
// counts as CPU-ful — the conservative reading, since misclassifying a CPU-ful node
// as CXL would fabricate far-memory capacity that is not expansion memory.
func nodeHasCPUs(path string) bool {
	b, err := os.ReadFile(path)
	if err != nil {
		return true
	}
	return strings.TrimSpace(string(b)) != ""
}

// parseNodeMeminfo reads "Node <N> MemTotal:" and "Node <N> MemFree:" (in kB) from a
// per-node meminfo file. Per-node meminfo has no MemAvailable, so free is strictly-free
// pages — an UNDER-estimate of what is reclaimable, which errs toward HIGHER pressure:
// conservative for a probe whose job is to keep a demote out of a full tier, and never
// a fabricated number.
func parseNodeMeminfo(path string) (total, free int64, ok bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, 0, false
	}
	var haveTotal, haveFree bool
	for _, line := range strings.Split(string(b), "\n") {
		fields := strings.Fields(line)
		// Shape: "Node <N> MemTotal: <kB> kB" — the field after the label is the count.
		if len(fields) < 4 {
			continue
		}
		switch fields[2] {
		case "MemTotal:":
			if v, err := strconv.ParseInt(fields[3], 10, 64); err == nil && v >= 0 && v <= (1<<63-1)/1024 {
				total = v * 1024
				haveTotal = true
			}
		case "MemFree:":
			if v, err := strconv.ParseInt(fields[3], 10, 64); err == nil && v >= 0 && v <= (1<<63-1)/1024 {
				free = v * 1024
				haveFree = true
			}
		}
		if haveTotal && haveFree {
			break
		}
	}
	if !haveTotal || !haveFree {
		return 0, 0, false
	}
	return total, free, true
}
