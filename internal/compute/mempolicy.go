package compute

import (
	"sort"
	"strconv"
	"strings"
	"sync"
)

// HostMemStatus is a point-in-time snapshot of the process's host-memory situation,
// including any NUMA confinement the process was launched under. It exists because a
// strict memory policy turns a routine model load into a silent death: the kernel
// OOM-kills the process (CONSTRAINT_MEMORY_POLICY) the moment the ALLOWED nodes run
// dry, even when the box as a whole has hundreds of GB free — no Go panic, no log
// line, just SIGKILL. The q36 27B loads died exactly this way under
// `numactl --membind=0` (~60 GB peak vs ~58 GB free on node 0). Surfacing the
// confinement and its CURRENT free headroom by default makes that failure legible
// before and while it happens instead of only in dmesg afterwards.
type HostMemStatus struct {
	// RSS is the process's current resident set in bytes (0 when unknown).
	RSS int64
	// HostAvail is MemAvailable in bytes (FreeUnknown when unknown).
	HostAvail int64
	// Constrained reports that allocations are STRICTLY confined to a subset of the
	// online NUMA nodes (MPOL_BIND via numactl --membind, or a cpuset/cgroup mems
	// restriction). Non-strict policies (prefer, interleave) do not set it: they
	// spill instead of killing.
	Constrained bool
	// PolicyLabel is the raw policy token when one was detected, e.g. "bind:0".
	PolicyLabel string
	// PolicyNodes is the allowed node list when Constrained, e.g. "0" or "0-1".
	PolicyNodes string
	// PolicyFree is the CURRENT total MemFree in bytes across the allowed nodes when
	// Constrained (0 when unknown). This is the number an allocation must fit under —
	// not HostAvail — and it moves as other tenants allocate.
	PolicyFree int64
}

// memPolicy is the cached process-lifetime confinement (policy is set before exec by
// numactl/cpuset and cannot change under us; free memory can, so it is NOT cached).
type memPolicy struct {
	label       string
	nodes       []int
	constrained bool
}

var (
	memPolicyOnce   sync.Once
	memPolicyCached memPolicy
)

// ReadHostMemStatus returns the current host-memory snapshot. The confinement itself
// is detected once and cached; RSS, MemAvailable, and the allowed-node free total are
// re-read on every call (they are single small file reads). On platforms without the
// needed introspection every field reports unknown/unconstrained.
func ReadHostMemStatus() HostMemStatus {
	memPolicyOnce.Do(func() { memPolicyCached = readMemPolicy() })
	st := HostMemStatus{RSS: processRSSBytes(), HostAvail: FreeUnknown}
	if _, avail, ok := hostSystemMemory(); ok {
		st.HostAvail = avail
	}
	p := memPolicyCached
	st.PolicyLabel = p.label
	if p.constrained {
		st.Constrained = true
		st.PolicyNodes = formatNodeList(p.nodes)
		if free, ok := policyFreeBytes(p.nodes); ok {
			st.PolicyFree = free
		}
	}
	return st
}

// parseNumaMapsPolicy extracts the effective NUMA policy from one /proc/self/numa_maps
// line ("<addr> <policy> <details...>"). Only MPOL_BIND ("bind:<nodes>") is strict —
// the kernel OOM-kills rather than spill outside the mask — so only bind reports
// nodes. "prefer"/"interleave"/"default" return the label alone.
func parseNumaMapsPolicy(line string) (label string, nodes []int) {
	fields := strings.Fields(line)
	if len(fields) < 2 {
		return "", nil
	}
	label = fields[1]
	if rest, ok := strings.CutPrefix(label, "bind:"); ok {
		return label, parseNodeList(rest)
	}
	return label, nil
}

// parseNodeList parses a kernel node-list string ("0", "0-3", "0-1,4,6-7") into a
// sorted slice of node ids. Malformed input yields nil.
func parseNodeList(s string) []int {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	var out []int
	for _, part := range strings.Split(s, ",") {
		lo, hi, ok := strings.Cut(part, "-")
		a, err := strconv.Atoi(strings.TrimSpace(lo))
		if err != nil || a < 0 {
			return nil
		}
		b := a
		if ok {
			b, err = strconv.Atoi(strings.TrimSpace(hi))
			if err != nil || b < a {
				return nil
			}
		}
		for n := a; n <= b; n++ {
			out = append(out, n)
		}
	}
	sort.Ints(out)
	return out
}

// formatNodeList renders sorted node ids back into compact kernel form ("0", "0-1,4").
func formatNodeList(nodes []int) string {
	if len(nodes) == 0 {
		return ""
	}
	var b strings.Builder
	for i := 0; i < len(nodes); {
		j := i
		for j+1 < len(nodes) && nodes[j+1] == nodes[j]+1 {
			j++
		}
		if b.Len() > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.Itoa(nodes[i]))
		if j > i {
			b.WriteByte('-')
			b.WriteString(strconv.Itoa(nodes[j]))
		}
		i = j + 1
	}
	return b.String()
}

// subsetOf reports whether every id in a appears in b (both sorted ascending).
func subsetOf(a, b []int) bool {
	i := 0
	for _, v := range a {
		for i < len(b) && b[i] < v {
			i++
		}
		if i >= len(b) || b[i] != v {
			return false
		}
	}
	return true
}

// parseNodeMeminfoFree extracts the MemFree byte count from a
// /sys/devices/system/node/node<N>/meminfo document ("Node N MemFree: X kB").
func parseNodeMeminfoFree(content string) (int64, bool) {
	for _, line := range strings.Split(content, "\n") {
		fields := strings.Fields(line)
		// "Node" "0" "MemFree:" "59976508" "kB"
		if len(fields) >= 4 && fields[0] == "Node" && fields[2] == "MemFree:" {
			kb, err := strconv.ParseInt(fields[3], 10, 64)
			if err != nil {
				return 0, false
			}
			return kb * 1024, true
		}
	}
	return 0, false
}
