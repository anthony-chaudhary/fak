package modelperfobs

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"runtime"
	"sort"
	"strings"
)

const (
	NUMARooflineCaptureSchema = "fak-host-memory-roofline-numa-capture/1"
	NUMARooflineMatrixSchema  = "fak-host-memory-roofline-numa-matrix/1"
)

// NUMATopology is a sysfs-derived snapshot of the nodes visible to the capture
// process. Node IDs are identifiers, not dense slice indexes.
type NUMATopology struct {
	Provenance       string                 `json:"provenance"`
	NodeRoot         string                 `json:"node_root"`
	OnlineSource     string                 `json:"online_source"`
	CPUSetSource     string                 `json:"cpuset_source"`
	OnlineNodeIDs    []int                  `json:"online_node_ids"`
	AllowedCPUIds    []int                  `json:"allowed_cpu_ids"`
	CPUSetRestricted bool                   `json:"cpuset_restricted"`
	Nodes            []NUMATopologyNode     `json:"nodes"`
	Omissions        []NUMATopologyOmission `json:"omissions,omitempty"`
}

type NUMATopologyNode struct {
	ID             int    `json:"id"`
	Online         bool   `json:"online"`
	MemoryBytes    uint64 `json:"memory_bytes"`
	CPUIds         []int  `json:"cpu_ids"`
	AllowedCPUIds  []int  `json:"allowed_cpu_ids"`
	Eligible       bool   `json:"eligible"`
	OmissionReason string `json:"omission_reason,omitempty"`
}

type NUMATopologyOmission struct {
	NodeID int    `json:"node_id"`
	Reason string `json:"reason"`
}

// NUMAVerifiedPlacement is evidence observed independently of the launcher
// request. A capture is usable only when both node IDs exactly match the
// request and both evidence strings name the OS observation used.
type NUMAVerifiedPlacement struct {
	CPUNode        int    `json:"cpu_node"`
	MemoryNode     int    `json:"memory_node"`
	CPUVerifier    string `json:"cpu_verifier"`
	MemoryVerifier string `json:"memory_verifier"`
	CPUEvidence    string `json:"cpu_evidence"`
	MemoryEvidence string `json:"memory_evidence"`
}

type NUMARooflinePairCapture struct {
	RequestedCPUNode    int                    `json:"requested_cpu_node"`
	RequestedMemoryNode int                    `json:"requested_memory_node"`
	RequestedCommand    []string               `json:"requested_command"`
	Verified            *NUMAVerifiedPlacement `json:"verified,omitempty"`
	Trials              []RooflineTrial        `json:"trials,omitempty"`
	OmissionReason      string                 `json:"omission_reason,omitempty"`
}

// NUMARooflineCapture is intentionally import-first. The benchmark may be
// launched under numactl, but the capture must independently verify CPU and
// page placement; a first-touch claim is not verification.
type NUMARooflineCapture struct {
	Schema           string                    `json:"schema"`
	MachineClass     string                    `json:"machine_class"`
	Topology         NUMATopology              `json:"topology"`
	WorkingSetBytes  uint64                    `json:"working_set_bytes"`
	PeakBufferBytes  uint64                    `json:"peak_buffer_bytes"`
	TargetDurationMS int64                     `json:"target_duration_ms"`
	RuntimeBudgetMS  int64                     `json:"runtime_budget_ms"`
	DRAMIsolation    string                    `json:"dram_isolation"`
	Pairs            []NUMARooflinePairCapture `json:"pairs"`
	Omissions        []NUMARooflinePairCapture `json:"omissions,omitempty"`
}

type NUMARooflinePair struct {
	CPUNode          int                   `json:"cpu_node"`
	MemoryNode       int                   `json:"memory_node"`
	Local            bool                  `json:"local"`
	RequestedCommand []string              `json:"requested_command"`
	Verified         NUMAVerifiedPlacement `json:"verified"`
	Trials           []RooflineTrial       `json:"trials"`
	SustainableGBS   float64               `json:"sustainable_gb_s"`
	RatioToLocal     *float64              `json:"ratio_to_local,omitempty"`
}

type NUMARooflineMatrix struct {
	Schema            string                    `json:"schema"`
	Scope             string                    `json:"scope"`
	MachineClass      string                    `json:"machine_class"`
	Method            string                    `json:"method"`
	TrafficAccounting string                    `json:"traffic_accounting"`
	DRAMIsolation     string                    `json:"dram_isolation"`
	WorkingSetBytes   uint64                    `json:"working_set_bytes"`
	PeakBufferBytes   uint64                    `json:"peak_buffer_bytes"`
	TargetDurationMS  int64                     `json:"target_duration_ms"`
	RuntimeBudgetMS   int64                     `json:"runtime_budget_ms"`
	Topology          NUMATopology              `json:"topology"`
	Pairs             []NUMARooflinePair        `json:"pairs"`
	Omissions         []NUMARooflinePairCapture `json:"omissions,omitempty"`
}

func ImportNUMARooflineCapture(r io.Reader) (NUMARooflineMatrix, error) {
	var capture NUMARooflineCapture
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&capture); err != nil {
		return NUMARooflineMatrix{}, fmt.Errorf("decode NUMA roofline capture: %w", err)
	}
	if err := ensureJSONEOF(dec); err != nil {
		return NUMARooflineMatrix{}, err
	}
	return BuildNUMARooflineMatrix(capture)
}

func ensureJSONEOF(dec *json.Decoder) error {
	var extra any
	if err := dec.Decode(&extra); err == io.EOF {
		return nil
	} else if err != nil {
		return fmt.Errorf("decode trailing NUMA roofline data: %w", err)
	}
	return fmt.Errorf("NUMA roofline capture contains multiple JSON values")
}

func BuildNUMARooflineMatrix(c NUMARooflineCapture) (NUMARooflineMatrix, error) {
	if c.Schema != NUMARooflineCaptureSchema {
		return NUMARooflineMatrix{}, fmt.Errorf("NUMA roofline schema %q, want %q", c.Schema, NUMARooflineCaptureSchema)
	}
	if runtime.GOOS != "linux" && c.Topology.Provenance == "" {
		return NUMARooflineMatrix{}, fmt.Errorf("NUMA roofline capture requires Linux sysfs topology provenance")
	}
	if err := validateNUMATopology(c.Topology); err != nil {
		return NUMARooflineMatrix{}, err
	}
	if c.WorkingSetBytes < MinRooflineWorkingSet || c.WorkingSetBytes > MaxRooflineWorkingSet {
		return NUMARooflineMatrix{}, fmt.Errorf("NUMA roofline working set %d outside [%d,%d]", c.WorkingSetBytes, MinRooflineWorkingSet, MaxRooflineWorkingSet)
	}
	if c.PeakBufferBytes < 2*c.WorkingSetBytes || c.PeakBufferBytes > 2*MaxRooflineWorkingSet {
		return NUMARooflineMatrix{}, fmt.Errorf("NUMA roofline peak buffer bytes %d must cover two working sets and remain bounded", c.PeakBufferBytes)
	}
	if c.TargetDurationMS <= 0 || c.RuntimeBudgetMS <= 0 || c.RuntimeBudgetMS > MaxRooflineRuntimeBudget.Milliseconds() {
		return NUMARooflineMatrix{}, fmt.Errorf("NUMA roofline time bounds are invalid")
	}
	if c.DRAMIsolation != "not-proven" {
		return NUMARooflineMatrix{}, fmt.Errorf("NUMA roofline dram_isolation must be not-proven without hardware-counter proof")
	}
	if len(c.Pairs)+len(c.Omissions) == 0 {
		return NUMARooflineMatrix{}, fmt.Errorf("NUMA roofline capture has no pairs or omissions")
	}
	eligible := make(map[int]bool)
	for _, n := range c.Topology.Nodes {
		eligible[n.ID] = n.Eligible
	}
	seen := make(map[[2]int]bool)
	pairs := make([]NUMARooflinePair, 0, len(c.Pairs))
	for _, p := range c.Pairs {
		key := [2]int{p.RequestedCPUNode, p.RequestedMemoryNode}
		if !registerNUMAPair(seen, key) {
			return NUMARooflineMatrix{}, fmt.Errorf("duplicate NUMA node pair cpu=%d memory=%d", key[0], key[1])
		}
		if !eligible[key[0]] || !eligible[key[1]] {
			return NUMARooflineMatrix{}, fmt.Errorf("NUMA pair cpu=%d memory=%d uses offline, memoryless, or cpuset-ineligible node", key[0], key[1])
		}
		if strings.TrimSpace(p.OmissionReason) != "" {
			return NUMARooflineMatrix{}, fmt.Errorf("measured NUMA pair cpu=%d memory=%d also has an omission reason", key[0], key[1])
		}
		if len(p.RequestedCommand) == 0 {
			return NUMARooflineMatrix{}, fmt.Errorf("NUMA pair cpu=%d memory=%d lacks requested launch command", key[0], key[1])
		}
		if p.Verified == nil {
			return NUMARooflineMatrix{}, fmt.Errorf("NUMA pair cpu=%d memory=%d lacks independent placement verification", key[0], key[1])
		}
		v := *p.Verified
		if v.CPUNode != key[0] || v.MemoryNode != key[1] {
			return NUMARooflineMatrix{}, fmt.Errorf("NUMA pair requested cpu=%d memory=%d but verified cpu=%d memory=%d", key[0], key[1], v.CPUNode, v.MemoryNode)
		}
		if strings.TrimSpace(v.CPUVerifier) == "" || strings.TrimSpace(v.MemoryVerifier) == "" || strings.TrimSpace(v.CPUEvidence) == "" || strings.TrimSpace(v.MemoryEvidence) == "" {
			return NUMARooflineMatrix{}, fmt.Errorf("NUMA pair cpu=%d memory=%d has incomplete independent verification evidence", key[0], key[1])
		}
		if len(p.Trials) < MinRooflineTrials || len(p.Trials) > MaxRooflineTrials {
			return NUMARooflineMatrix{}, fmt.Errorf("NUMA pair cpu=%d memory=%d trial count %d outside [%d,%d]", key[0], key[1], len(p.Trials), MinRooflineTrials, MaxRooflineTrials)
		}
		values := make([]float64, len(p.Trials))
		for i, t := range p.Trials {
			if t.GBS <= 0 || math.IsNaN(t.GBS) || math.IsInf(t.GBS, 0) {
				return NUMARooflineMatrix{}, fmt.Errorf("NUMA pair cpu=%d memory=%d trial %d has invalid bandwidth", key[0], key[1], i)
			}
			values[i] = t.GBS
		}
		pairs = append(pairs, NUMARooflinePair{CPUNode: key[0], MemoryNode: key[1], Local: key[0] == key[1], RequestedCommand: append([]string(nil), p.RequestedCommand...), Verified: v, Trials: append([]RooflineTrial(nil), p.Trials...), SustainableGBS: medianFloat64(values)})
	}
	for _, o := range c.Omissions {
		key := [2]int{o.RequestedCPUNode, o.RequestedMemoryNode}
		if !registerNUMAPair(seen, key) {
			return NUMARooflineMatrix{}, fmt.Errorf("duplicate NUMA node pair cpu=%d memory=%d", key[0], key[1])
		}
		if strings.TrimSpace(o.OmissionReason) == "" {
			return NUMARooflineMatrix{}, fmt.Errorf("omitted NUMA pair cpu=%d memory=%d lacks an explicit reason", key[0], key[1])
		}
		if len(o.Trials) > 0 || o.Verified != nil {
			return NUMARooflineMatrix{}, fmt.Errorf("omitted NUMA pair cpu=%d memory=%d contains measurement data", key[0], key[1])
		}
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].CPUNode != pairs[j].CPUNode {
			return pairs[i].CPUNode < pairs[j].CPUNode
		}
		return pairs[i].MemoryNode < pairs[j].MemoryNode
	})
	local := make(map[int]float64)
	for _, p := range pairs {
		if p.Local {
			local[p.CPUNode] = p.SustainableGBS
		}
	}
	for i := range pairs {
		if base, ok := local[pairs[i].CPUNode]; ok && base > 0 {
			ratio := pairs[i].SustainableGBS / base
			pairs[i].RatioToLocal = &ratio
		}
	}
	return NUMARooflineMatrix{Schema: NUMARooflineMatrixSchema, Scope: "host-memory", MachineClass: c.MachineClass, Method: "externally-launched-numactl-copy-roofline-with-independent-cpu-and-page-placement-verification", TrafficAccounting: "copy-bytes-read-plus-bytes-written", DRAMIsolation: "not-proven", WorkingSetBytes: c.WorkingSetBytes, PeakBufferBytes: c.PeakBufferBytes, TargetDurationMS: c.TargetDurationMS, RuntimeBudgetMS: c.RuntimeBudgetMS, Topology: c.Topology, Pairs: pairs, Omissions: append([]NUMARooflinePairCapture(nil), c.Omissions...)}, nil
}

func registerNUMAPair(seen map[[2]int]bool, key [2]int) bool {
	if seen[key] {
		return false
	}
	seen[key] = true
	return true
}

func medianFloat64(v []float64) float64 {
	x := append([]float64(nil), v...)
	sort.Float64s(x)
	n := len(x)
	if n%2 == 1 {
		return x[n/2]
	}
	return (x[n/2-1] + x[n/2]) / 2
}

func validateNUMATopology(t NUMATopology) error {
	if t.Provenance != "linux-sysfs" {
		return fmt.Errorf("NUMA topology provenance %q, want linux-sysfs", t.Provenance)
	}
	if strings.TrimSpace(t.NodeRoot) == "" || strings.TrimSpace(t.OnlineSource) == "" || strings.TrimSpace(t.CPUSetSource) == "" {
		return fmt.Errorf("NUMA topology lacks exact sysfs/proc provenance paths")
	}
	if len(t.Nodes) == 0 {
		return fmt.Errorf("NUMA topology has no nodes")
	}
	seen := map[int]bool{}
	for _, n := range t.Nodes {
		if seen[n.ID] {
			return fmt.Errorf("duplicate NUMA topology node %d", n.ID)
		}
		seen[n.ID] = true
		if n.Eligible && (!n.Online || n.MemoryBytes == 0 || len(n.AllowedCPUIds) == 0) {
			return fmt.Errorf("NUMA topology node %d is incorrectly eligible", n.ID)
		}
		if !n.Eligible && strings.TrimSpace(n.OmissionReason) == "" {
			return fmt.Errorf("ineligible NUMA topology node %d lacks omission reason", n.ID)
		}
	}
	return nil
}
