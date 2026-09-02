//go:build linux

package modelperfobs

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

func DiscoverNUMATopology() (NUMATopology, error) {
	return discoverNUMATopology("/sys/devices/system/node", "/proc/self/status")
}

func discoverNUMATopology(nodeRoot, statusPath string) (NUMATopology, error) {
	entries, err := os.ReadDir(nodeRoot)
	if err != nil {
		return NUMATopology{}, fmt.Errorf("read NUMA sysfs root: %w", err)
	}
	onlinePath := filepath.Join(nodeRoot, "online")
	online, err := readIDListFile(onlinePath)
	if err != nil {
		return NUMATopology{}, fmt.Errorf("read NUMA online nodes: %w", err)
	}
	onlineSet := idSet(online)
	allowed, err := readAllowedCPUs(statusPath)
	if err != nil {
		return NUMATopology{}, err
	}
	allowedSet := idSet(allowed)
	nodes := make([]NUMATopologyNode, 0)
	allCPU := map[int]bool{}
	for _, e := range entries {
		if !e.IsDir() || !strings.HasPrefix(e.Name(), "node") {
			continue
		}
		id, err := strconv.Atoi(strings.TrimPrefix(e.Name(), "node"))
		if err != nil {
			continue
		}
		cpus, err := readIDListFile(filepath.Join(nodeRoot, e.Name(), "cpulist"))
		if err != nil {
			return NUMATopology{}, fmt.Errorf("read node%d cpulist: %w", id, err)
		}
		for _, cpu := range cpus {
			allCPU[cpu] = true
		}
		mem, err := readNodeMemory(filepath.Join(nodeRoot, e.Name(), "meminfo"))
		if err != nil {
			return NUMATopology{}, fmt.Errorf("read node%d meminfo: %w", id, err)
		}
		permitted := intersectIDs(cpus, allowedSet)
		n := NUMATopologyNode{ID: id, Online: onlineSet[id], MemoryBytes: mem, CPUIds: cpus, AllowedCPUIds: permitted}
		switch {
		case !n.Online:
			n.OmissionReason = "offline"
		case n.MemoryBytes == 0:
			n.OmissionReason = "memoryless"
		case len(n.CPUIds) == 0:
			n.OmissionReason = "no-cpus"
		case len(n.AllowedCPUIds) == 0:
			n.OmissionReason = "excluded-by-process-cpuset"
		default:
			n.Eligible = true
		}
		nodes = append(nodes, n)
	}
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].ID < nodes[j].ID })
	omissions := make([]NUMATopologyOmission, 0)
	for _, n := range nodes {
		if !n.Eligible {
			omissions = append(omissions, NUMATopologyOmission{NodeID: n.ID, Reason: n.OmissionReason})
		}
	}
	return NUMATopology{Provenance: "linux-sysfs", NodeRoot: nodeRoot, OnlineSource: onlinePath, CPUSetSource: statusPath + ":Cpus_allowed_list", OnlineNodeIDs: online, AllowedCPUIds: allowed, CPUSetRestricted: len(allowed) < len(allCPU), Nodes: nodes, Omissions: omissions}, nil
}

func readAllowedCPUs(path string) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("read process cpuset: %w", err)
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := s.Text()
		if strings.HasPrefix(line, "Cpus_allowed_list:") {
			return parseIDList(strings.TrimSpace(strings.TrimPrefix(line, "Cpus_allowed_list:")))
		}
	}
	if err := s.Err(); err != nil {
		return nil, fmt.Errorf("scan process cpuset: %w", err)
	}
	return nil, fmt.Errorf("process status lacks Cpus_allowed_list")
}
func readIDListFile(path string) ([]int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parseIDList(strings.TrimSpace(string(b)))
}
func readNodeMemory(path string) (uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		fields := strings.Fields(s.Text())
		for i, f := range fields {
			if f == "MemTotal:" && i+2 < len(fields) {
				kb, err := strconv.ParseUint(fields[i+1], 10, 64)
				if err != nil {
					return 0, err
				}
				return kb * 1024, nil
			}
		}
	}
	if err := s.Err(); err != nil {
		return 0, err
	}
	return 0, fmt.Errorf("MemTotal missing")
}
func idSet(ids []int) map[int]bool {
	m := make(map[int]bool, len(ids))
	for _, id := range ids {
		m[id] = true
	}
	return m
}
func intersectIDs(ids []int, set map[int]bool) []int {
	out := make([]int, 0, len(ids))
	for _, id := range ids {
		if set[id] {
			out = append(out, id)
		}
	}
	return out
}
