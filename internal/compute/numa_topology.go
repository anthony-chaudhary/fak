package compute

import (
	"fmt"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

// NUMAReplicaConfig describes the resolved NUMA replica regime.
type NUMAReplicaConfig struct {
	Enabled   bool
	Requested string
	Replicas  int
	Topology  []NUMANodeTopology
	Reason    string
}

// DetectHostNUMATopology discovers the host's online CPU-bearing NUMA nodes.
// Returns nil on non-Linux platforms or when fewer than 2 nodes are detected.
func DetectHostNUMATopology() []NUMANodeTopology {
	nodes := detectHostNUMATopologyPlatform()
	if len(nodes) < 2 {
		return nil
	}
	return nodes
}

// SynthesizeNUMATopology creates a synthetic NUMANodeTopology with `nodes` nodes,
// distributing available logical CPUs evenly across them.
func SynthesizeNUMATopology(nodes int) []NUMANodeTopology {
	if nodes < 2 {
		return nil
	}
	totalCPUs := runtime.NumCPU()
	if totalCPUs < nodes {
		totalCPUs = nodes
	}
	topo := make([]NUMANodeTopology, nodes)
	base := totalCPUs / nodes
	extra := totalCPUs % nodes
	cpuIdx := 0
	for i := 0; i < nodes; i++ {
		count := base
		if i < extra {
			count++
		}
		cpus := make([]int, count)
		for c := 0; c < count; c++ {
			cpus[c] = cpuIdx
			cpuIdx++
		}
		topo[i] = NUMANodeTopology{
			NodeID: i,
			CPUs:   cpus,
		}
	}
	return topo
}

// ResolveNUMAReplicaConfig resolves the NUMA replica configuration from a flag value
// or the FAK_NUMA_REPLICAS environment variable.
//
// Recognized formats:
//   - "off", "0", "1", "false", "none", "disabled": replicas disabled
//   - "auto", "default", "": auto-detect host NUMA topology; if >= 2 nodes, enable
//   - "all", "on", "true": enable across all host NUMA nodes (or min 2 synthesized)
//   - integer N (>= 2): exactly N replicas across N nodes (detecting or synthesizing)
func ResolveNUMAReplicaConfig(requested string) NUMAReplicaConfig {
	req := strings.TrimSpace(requested)
	if req == "" {
		req = strings.TrimSpace(os.Getenv("FAK_NUMA_REPLICAS"))
	}
	if req == "" {
		req = "auto"
	}

	norm := strings.ToLower(req)
	switch norm {
	case "off", "0", "1", "false", "none", "disabled":
		return NUMAReplicaConfig{
			Enabled:   false,
			Requested: req,
			Replicas:  1,
			Reason:    "disabled_by_request",
		}
	case "auto", "default":
		hostTopo := DetectHostNUMATopology()
		if len(hostTopo) >= 2 {
			return NUMAReplicaConfig{
				Enabled:   true,
				Requested: req,
				Replicas:  len(hostTopo),
				Topology:  hostTopo,
				Reason:    "auto_detected_host_topology",
			}
		}
		return NUMAReplicaConfig{
			Enabled:   false,
			Requested: req,
			Replicas:  len(hostTopo),
			Topology:  hostTopo,
			Reason:    "insufficient_host_nodes",
		}
	case "all", "on", "true":
		hostTopo := DetectHostNUMATopology()
		if len(hostTopo) >= 2 {
			return NUMAReplicaConfig{
				Enabled:   true,
				Requested: req,
				Replicas:  len(hostTopo),
				Topology:  hostTopo,
				Reason:    "force_all_host_nodes",
			}
		}
		synth := SynthesizeNUMATopology(2)
		return NUMAReplicaConfig{
			Enabled:   true,
			Requested: req,
			Replicas:  2,
			Topology:  synth,
			Reason:    "synthesized_min_nodes",
		}
	default:
		n, err := strconv.Atoi(norm)
		if err != nil || n < 2 {
			return NUMAReplicaConfig{
				Enabled:   false,
				Requested: req,
				Replicas:  0,
				Reason:    fmt.Sprintf("invalid_replica_spec_%q", req),
			}
		}
		hostTopo := DetectHostNUMATopology()
		if len(hostTopo) >= n {
			return NUMAReplicaConfig{
				Enabled:   true,
				Requested: req,
				Replicas:  n,
				Topology:  hostTopo[:n],
				Reason:    "exact_host_nodes",
			}
		}
		synth := SynthesizeNUMATopology(n)
		return NUMAReplicaConfig{
			Enabled:   true,
			Requested: req,
			Replicas:  n,
			Topology:  synth,
			Reason:    "synthesized_exact_nodes",
		}
	}
}

// PlanNUMAReplicasForTopology constructs an eligible NUMAReplicaPlan for the given
// topology and replica size in bytes.
func PlanNUMAReplicasForTopology(topology []NUMANodeTopology, replicaBytes int64) NUMAReplicaPlan {
	if replicaBytes <= 0 {
		return NUMAReplicaPlan{
			Eligible:     false,
			Reason:       NUMAReplicaPlanInvalidSize,
			ReplicaBytes: replicaBytes,
		}
	}
	if len(topology) < 2 {
		return NUMAReplicaPlan{
			Eligible:     false,
			Reason:       NUMAReplicaPlanInsufficientNodes,
			ReplicaBytes: replicaBytes,
		}
	}

	targets := make([]NUMAReplicaTarget, len(topology))
	nodes := make([]NUMANodeTopology, len(topology))
	copy(nodes, topology)
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].NodeID < nodes[j].NodeID })

	for i, node := range nodes {
		if node.NodeID < 0 || len(node.CPUs) == 0 {
			return NUMAReplicaPlan{
				Eligible:     false,
				Reason:       NUMAReplicaPlanUnsupported,
				ReplicaBytes: replicaBytes,
			}
		}
		if i > 0 && nodes[i-1].NodeID == node.NodeID {
			return NUMAReplicaPlan{
				Eligible:     false,
				Reason:       NUMAReplicaPlanUnsupported,
				ReplicaBytes: replicaBytes,
			}
		}
		targets[i] = NUMAReplicaTarget{
			NodeID:        node.NodeID,
			ReplicaBytes:  replicaBytes,
			RequiredBytes: replicaBytes,
		}
	}

	return NUMAReplicaPlan{
		Eligible:             true,
		Reason:               NUMAReplicaPlanEligible,
		ReplicaBytes:         replicaBytes,
		RequiredPerNodeBytes: replicaBytes,
		TotalReplicaBytes:    replicaBytes * int64(len(targets)),
		TotalRequiredBytes:   replicaBytes * int64(len(targets)),
		Targets:              targets,
	}
}

// BuildNUMAReplicasForTopology builds one byte-identical replica per node in topology.
// It verifies that each replica matches src before returning.
func BuildNUMAReplicasForTopology(src []byte, topology []NUMANodeTopology) (*NUMAReplicaSet, error) {
	plan := PlanNUMAReplicasForTopology(topology, int64(len(src)))
	if !plan.Eligible {
		return nil, fmt.Errorf("compute: plan ineligible: %s", plan.Reason)
	}
	set, err := BuildNUMAReplicas(src, plan)
	if err != nil {
		return nil, err
	}
	if err := VerifyNUMAReplicas(src, set); err != nil {
		_ = set.Free()
		return nil, err
	}
	return set, nil
}
