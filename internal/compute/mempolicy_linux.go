//go:build linux

package compute

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"syscall"
)

// readMemPolicy detects the process's strict NUMA confinement, if any, from two
// sources: the task mempolicy as rendered in /proc/self/numa_maps (numactl --membind
// shows every VMA as "bind:<nodes>"), and the cpuset mems restriction from
// /proc/self/status Mems_allowed_list (containers). Either one being a strict subset
// of the online nodes means allocations can OOM-kill while the box has memory free.
// numa_maps is read once here and never again: the kernel walks pages to render it,
// which is far too expensive per progress tick (policy cannot change post-exec anyway).
func readMemPolicy() memPolicy {
	online := parseNodeList(readTrimmedFile("/sys/devices/system/node/online"))
	label, bindNodes := firstNumaMapsPolicy("/proc/self/numa_maps")
	cpuset := parseNodeList(memsAllowedList("/proc/self/status"))

	p := memPolicy{label: label}
	switch {
	case len(bindNodes) > 0:
		p.nodes = bindNodes
		if len(cpuset) > 0 {
			p.nodes = intersectNodes(bindNodes, cpuset)
		}
		p.constrained = true
	case len(cpuset) > 0 && len(online) > 0 && !subsetOf(online, cpuset):
		p.nodes = cpuset
		p.label = "cpuset:" + formatNodeList(cpuset)
		p.constrained = true
	}
	// A "confinement" to every online node is no confinement at all.
	if p.constrained && len(online) > 0 && subsetOf(online, p.nodes) {
		p.constrained = false
		p.nodes = nil
	}
	return p
}

// firstNumaMapsPolicy parses the policy token off the first line of numa_maps.
func firstNumaMapsPolicy(path string) (string, []int) {
	f, err := os.Open(path)
	if err != nil {
		return "", nil
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		return parseNumaMapsPolicy(line)
	}
	return "", nil
}

// memsAllowedList extracts the Mems_allowed_list value from /proc/<pid>/status.
func memsAllowedList(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if rest, ok := strings.CutPrefix(sc.Text(), "Mems_allowed_list:"); ok {
			return strings.TrimSpace(rest)
		}
	}
	return ""
}

// policyFreeBytes sums the CURRENT MemFree across the given nodes from sysfs. This is
// deliberately MemFree, not an availability estimate: under MPOL_BIND the reclaim the
// kernel is willing to do before OOM-killing is what it is — the strictly-free pages
// are the honest headroom number to show a human watching a load approach the cliff.
func policyFreeBytes(nodes []int) (int64, bool) {
	var total int64
	any := false
	for _, n := range nodes {
		data, err := os.ReadFile("/sys/devices/system/node/node" + strconv.Itoa(n) + "/meminfo")
		if err != nil {
			continue
		}
		if b, ok := parseNodeMeminfoFree(string(data)); ok {
			total += b
			any = true
		}
	}
	return total, any
}

// processRSSBytes reads the resident set from /proc/self/statm (field 2, in pages).
func processRSSBytes() int64 {
	data, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return 0
	}
	fields := strings.Fields(string(data))
	if len(fields) < 2 {
		return 0
	}
	pages, err := strconv.ParseInt(fields[1], 10, 64)
	if err != nil || pages < 0 {
		return 0
	}
	return pages * int64(syscall.Getpagesize())
}

// intersectNodes returns the ids present in both sorted slices.
func intersectNodes(a, b []int) []int {
	var out []int
	i := 0
	for _, v := range a {
		for i < len(b) && b[i] < v {
			i++
		}
		if i < len(b) && b[i] == v {
			out = append(out, v)
		}
	}
	return out
}

// readTrimmedFile returns the whitespace-trimmed content of a small sysfs file.
func readTrimmedFile(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}
