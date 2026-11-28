package engine

// single_node_topology.go — Single-node 1-8 GPU (12-141 GiB/GPU) topology
// classification, fail-closed VRAM/KV/overhead/slot admission, and DP-replica vs
// intra-node TP / hybrid DPxTP placement planning.
//
// Scope fence: strictly single-node 1-8 GPU boxes (PCIe, NVLink, UMA). Multi-node
// TPxPPxEP, cross-rack RDMA/InfiniBand, DeepEP cross-node EP, GB200/GB300 NVL72,
// and large scale-out clusters are out of scope and fail closed at the boundary.

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	bytesPerMiB int64 = 1 << 20
	bytesPerGiB int64 = 1 << 30

	// MinSingleNodeGPUs and MaxSingleNodeGPUs bound the supported single-node device count.
	MinSingleNodeGPUs = 1
	MaxSingleNodeGPUs = 8

	// MinSingleNodeGPUVRAMBytes and MaxSingleNodeGPUVRAMBytes bound the supported
	// per-GPU physical memory envelope (12 GiB .. 141 GiB).
	MinSingleNodeGPUVRAMBytes = 12 * bytesPerGiB
	MaxSingleNodeGPUVRAMBytes = 141 * bytesPerGiB

	// NVLinkMinBandwidthGBs is the one-way peer link threshold above which a multi-GPU
	// discrete box without an explicit interconnect label classifies as NVLink.
	NVLinkMinBandwidthGBs = 150.0

	// DefaultPCIeLinkBandwidthGBs / DefaultPCIeLinkLatencyMicros and their NVLink
	// counterparts provide modeled link characteristics when the box does not state
	// measured link rates. GenericLinkRates records when defaults were used.
	DefaultPCIeLinkBandwidthGBs  = 32.0
	DefaultPCIeLinkLatencyMicros = 15.0
	DefaultNVLinkBandwidthGBs    = 300.0
	DefaultNVLinkLatencyMicros   = 1.5
)

// NodeInterconnect is the closed vocabulary of single-node accelerator interconnects.
type NodeInterconnect string

const (
	InterconnectPCIe   NodeInterconnect = "pcie"
	InterconnectNVLink NodeInterconnect = "nvlink"
	InterconnectUMA    NodeInterconnect = "uma"
)

// NodeInterconnects returns the closed interconnect vocabulary in stable order.
func NodeInterconnects() []NodeInterconnect {
	return []NodeInterconnect{InterconnectPCIe, InterconnectNVLink, InterconnectUMA}
}

// Valid reports whether ic is a recognized single-node interconnect.
func (ic NodeInterconnect) Valid() bool {
	switch ic {
	case InterconnectPCIe, InterconnectNVLink, InterconnectUMA:
		return true
	default:
		return false
	}
}

// PlacementStrategy names how a model is placed across a single-node box's GPUs.
type PlacementStrategy string

const (
	PlacementDataParallel   PlacementStrategy = "data_parallel"
	PlacementTensorParallel PlacementStrategy = "tensor_parallel"
	PlacementHybridDPTP     PlacementStrategy = "hybrid_dp_tp"
)

// PlacementStrategies returns the closed placement strategy vocabulary in stable order.
func PlacementStrategies() []PlacementStrategy {
	return []PlacementStrategy{PlacementDataParallel, PlacementTensorParallel, PlacementHybridDPTP}
}

// Valid reports whether s is a recognized placement strategy.
func (s PlacementStrategy) Valid() bool {
	switch s {
	case PlacementDataParallel, PlacementTensorParallel, PlacementHybridDPTP:
		return true
	default:
		return false
	}
}

// PlacementObjective selects how multiple admissible (DP, TP) layouts are ranked.
type PlacementObjective string

const (
	ObjectiveThroughput PlacementObjective = "throughput"
	ObjectiveLatency    PlacementObjective = "latency"
)

// PlacementObjectives returns the closed placement objective vocabulary in stable order.
func PlacementObjectives() []PlacementObjective {
	return []PlacementObjective{ObjectiveThroughput, ObjectiveLatency}
}

// Valid reports whether o is a recognized placement objective.
func (o PlacementObjective) Valid() bool {
	switch o {
	case ObjectiveThroughput, ObjectiveLatency:
		return true
	default:
		return false
	}
}

// PlacementStatus is the closed admission outcome for single-node placement.
type PlacementStatus string

const (
	PlacementAdmitted PlacementStatus = "admitted"
	PlacementRefused  PlacementStatus = "refused"
)

// PlacementStatuses returns the closed placement status vocabulary in stable order.
func PlacementStatuses() []PlacementStatus {
	return []PlacementStatus{PlacementAdmitted, PlacementRefused}
}

// Valid reports whether st is a recognized placement status.
func (st PlacementStatus) Valid() bool {
	switch st {
	case PlacementAdmitted, PlacementRefused:
		return true
	default:
		return false
	}
}

// PlacementRefusalReason is the closed vocabulary of fail-closed placement refusals.
// PlacementRefusalNone is the sole non-refusal value.
type PlacementRefusalReason string

const (
	PlacementRefusalNone                       PlacementRefusalReason = ""
	PlacementRefusalMultiNodeOutOfScope        PlacementRefusalReason = "multi-node-out-of-scope"
	PlacementRefusalGPUCountOutOfRange         PlacementRefusalReason = "gpu-count-out-of-range"
	PlacementRefusalVRAMUnknown                PlacementRefusalReason = "vram-unknown"
	PlacementRefusalVRAMOutOfRange             PlacementRefusalReason = "vram-out-of-range"
	PlacementRefusalVRAMInvalid                PlacementRefusalReason = "vram-invalid"
	PlacementRefusalTopologyMismatch           PlacementRefusalReason = "topology-mismatch"
	PlacementRefusalModelInvalid               PlacementRefusalReason = "model-invalid"
	PlacementRefusalKVGeometryUnknown          PlacementRefusalReason = "kv-geometry-unknown"
	PlacementRefusalOverheadUnbudgeted         PlacementRefusalReason = "overhead-unbudgeted"
	PlacementRefusalSlotsInvalid               PlacementRefusalReason = "slots-invalid"
	PlacementRefusalInsufficientVRAM           PlacementRefusalReason = "insufficient-vram"
	PlacementRefusalSlotsExceedReplicaCapacity PlacementRefusalReason = "slots-exceed-replica-capacity"
	PlacementRefusalModelNotDivisibleByTP      PlacementRefusalReason = "model-not-divisible-by-tp"
	PlacementRefusalP2PUnavailable             PlacementRefusalReason = "p2p-unavailable"
	PlacementRefusalTPRequiresNVLink           PlacementRefusalReason = "tp-requires-nvlink"
	PlacementRefusalRateUnknownForSLO          PlacementRefusalReason = "rate-unknown-for-slo"
	PlacementRefusalSLOViolated                PlacementRefusalReason = "slo-violated"
)

// PlacementRefusalReasons returns the closed refusal vocabulary in stable order.
func PlacementRefusalReasons() []PlacementRefusalReason {
	return []PlacementRefusalReason{
		PlacementRefusalNone,
		PlacementRefusalMultiNodeOutOfScope,
		PlacementRefusalGPUCountOutOfRange,
		PlacementRefusalVRAMUnknown,
		PlacementRefusalVRAMOutOfRange,
		PlacementRefusalVRAMInvalid,
		PlacementRefusalTopologyMismatch,
		PlacementRefusalModelInvalid,
		PlacementRefusalKVGeometryUnknown,
		PlacementRefusalOverheadUnbudgeted,
		PlacementRefusalSlotsInvalid,
		PlacementRefusalInsufficientVRAM,
		PlacementRefusalSlotsExceedReplicaCapacity,
		PlacementRefusalModelNotDivisibleByTP,
		PlacementRefusalP2PUnavailable,
		PlacementRefusalTPRequiresNVLink,
		PlacementRefusalRateUnknownForSLO,
		PlacementRefusalSLOViolated,
	}
}

// Valid reports whether r is a member of the closed refusal vocabulary.
func (r PlacementRefusalReason) Valid() bool {
	switch r {
	case PlacementRefusalNone,
		PlacementRefusalMultiNodeOutOfScope,
		PlacementRefusalGPUCountOutOfRange,
		PlacementRefusalVRAMUnknown,
		PlacementRefusalVRAMOutOfRange,
		PlacementRefusalVRAMInvalid,
		PlacementRefusalTopologyMismatch,
		PlacementRefusalModelInvalid,
		PlacementRefusalKVGeometryUnknown,
		PlacementRefusalOverheadUnbudgeted,
		PlacementRefusalSlotsInvalid,
		PlacementRefusalInsufficientVRAM,
		PlacementRefusalSlotsExceedReplicaCapacity,
		PlacementRefusalModelNotDivisibleByTP,
		PlacementRefusalP2PUnavailable,
		PlacementRefusalTPRequiresNVLink,
		PlacementRefusalRateUnknownForSLO,
		PlacementRefusalSLOViolated:
		return true
	default:
		return false
	}
}

// Refused reports whether r indicates a refused placement (anything other than PlacementRefusalNone).
func (r PlacementRefusalReason) Refused() bool { return r != PlacementRefusalNone }

// SingleNodeDevice describes one physical GPU (or unified APU pool) in a single-node box.
type SingleNodeDevice struct {
	Index              int     `json:"index"`
	DeviceID           string  `json:"device_id,omitempty"`
	TotalVRAMBytes     int64   `json:"total_vram_bytes"`
	FreeVRAMBytes      int64   `json:"free_vram_bytes,omitempty"`
	FreeVRAMKnown      bool    `json:"free_vram_known,omitempty"`
	MemoryBandwidthGBs float64 `json:"memory_bandwidth_gbs,omitempty"`
	ComputeTFLOPs      float64 `json:"compute_tflops,omitempty"`
}

// SingleNodeDeviceFromBackend probes live device memory via compute.DeviceMemoryInfo
// (and DeviceCeilingInfo / DeviceHBMPressure) for a compute.Backend. When the backend
// cannot report capacity, ok is false so callers fail closed rather than guessing VRAM.
func SingleNodeDeviceFromBackend(index int, deviceID string, b compute.Backend, residentBytes int64) (SingleNodeDevice, bool) {
	total, free, known := compute.DeviceMemoryInfo(b)
	if ceiling, cknown := compute.DeviceCeilingInfo(b); cknown && ceiling > 0 && total <= 0 {
		total = ceiling
		known = true
	}
	if !known || total <= 0 {
		return SingleNodeDevice{}, false
	}
	if deviceID == "" {
		deviceID = fmt.Sprintf("gpu-%d", index)
	}
	dev := SingleNodeDevice{
		Index:          index,
		DeviceID:       deviceID,
		TotalVRAMBytes: total,
	}
	if free >= 0 && free <= total {
		dev.FreeVRAMBytes = free
		dev.FreeVRAMKnown = true
		return dev, true
	}
	if pressure, capBytes, pknown := DeviceHBMPressure(b, residentBytes); pknown && capBytes > 0 {
		used := int64(math.Ceil(pressure * float64(capBytes)))
		if used < 0 {
			used = 0
		}
		if used > capBytes {
			used = capBytes
		}
		dev.FreeVRAMBytes = capBytes - used
		dev.FreeVRAMKnown = true
	}
	return dev, true
}

// SingleNodeBoxSpec describes a candidate single-node 1-8 GPU machine.
type SingleNodeBoxSpec struct {
	// NodeCount defaults to 1 when 0; values > 1 fail closed as out-of-scope multi-node.
	NodeCount int `json:"node_count,omitempty"`
	// CrossNodeFabric, when non-empty (e.g. "rdma", "infiniband", "deepep", "nvl72"),
	// fails closed as out-of-scope multi-node/rack scale-out.
	CrossNodeFabric string `json:"cross_node_fabric,omitempty"`

	Interconnect  NodeInterconnect `json:"interconnect,omitempty"`
	UnifiedMemory bool             `json:"unified_memory,omitempty"`
	NVLinkActive  bool             `json:"nvlink_active,omitempty"`
	// P2PUnavailable explicitly disables GPU-to-GPU P2P over PCIe. Zero-value (false)
	// permits PCIe P2P when present unless RequireNVLinkForTP is set.
	P2PUnavailable     bool `json:"p2p_unavailable,omitempty"`
	RequireNVLinkForTP bool `json:"require_nvlink_for_tp,omitempty"`

	// Either supply explicit GPUs (1..8) OR homogeneous GPUCount + PerGPUVRAMBytes.
	GPUs               []SingleNodeDevice `json:"gpus,omitempty"`
	GPUCount           int                `json:"gpu_count,omitempty"`
	PerGPUVRAMBytes    int64              `json:"per_gpu_vram_bytes,omitempty"`
	FreeVRAMBytes      int64              `json:"free_vram_bytes,omitempty"`
	FreeVRAMKnown      bool               `json:"free_vram_known,omitempty"`
	MemoryBandwidthGBs float64            `json:"memory_bandwidth_gbs,omitempty"`
	ComputeTFLOPs      float64            `json:"compute_tflops,omitempty"`

	HostDRAMBytes     int64   `json:"host_dram_bytes,omitempty"`
	HeadroomFraction  float64 `json:"headroom_fraction,omitempty"`
	LinkBandwidthGBs  float64 `json:"link_bandwidth_gbs,omitempty"`
	LinkLatencyMicros float64 `json:"link_latency_micros,omitempty"`
}

// ProbeSingleNodeBox populates spec.GPUs from live compute.Backend probes via
// SingleNodeDeviceFromBackend and, for UMA boxes, HostDRAMPressure. A backend that
// cannot report capacity leaves its TotalVRAMBytes 0 so placement fails closed.
func ProbeSingleNodeBox(backends []compute.Backend, residentBytes []int64, spec SingleNodeBoxSpec) SingleNodeBoxSpec {
	out := spec
	out.GPUCount = len(backends)
	out.GPUs = make([]SingleNodeDevice, len(backends))
	for i, b := range backends {
		var res int64
		if i < len(residentBytes) {
			res = residentBytes[i]
		}
		if dev, ok := SingleNodeDeviceFromBackend(i, fmt.Sprintf("gpu-%d", i), b, res); ok {
			dev.MemoryBandwidthGBs = spec.MemoryBandwidthGBs
			dev.ComputeTFLOPs = spec.ComputeTFLOPs
			out.GPUs[i] = dev
		} else {
			out.GPUs[i] = SingleNodeDevice{Index: i, DeviceID: fmt.Sprintf("gpu-%d", i)}
		}
	}
	if (out.UnifiedMemory || out.Interconnect == InterconnectUMA) && out.HostDRAMBytes <= 0 {
		var hostRes int64
		if len(residentBytes) > 0 {
			hostRes = residentBytes[0]
		}
		if pressure, capBytes, known := HostDRAMPressure(hostRes); known && capBytes > 0 {
			used := int64(math.Ceil(pressure * float64(capBytes)))
			if used < capBytes {
				out.HostDRAMBytes = capBytes - used
			}
		}
	}
	return out
}

// SingleNodeBoxProfile is the normalized, validated classification of a single-node box.
type SingleNodeBoxProfile struct {
	Interconnect      NodeInterconnect   `json:"interconnect"`
	GPUCount          int                `json:"gpu_count"`
	UnifiedMemory     bool               `json:"unified_memory"`
	P2PCapable        bool               `json:"p2p_capable"`
	LinkBandwidthGBs  float64            `json:"link_bandwidth_gbs"`
	LinkLatencyMicros float64            `json:"link_latency_micros"`
	GenericLinkRates  bool               `json:"generic_link_rates,omitempty"`
	Devices           []SingleNodeDevice `json:"devices"`
	UsableVRAMBytes   []int64            `json:"usable_vram_bytes"`
	TotalVRAMBytes    int64              `json:"total_vram_bytes"`
}

// ClassifySingleNodeBox validates the single-node 1-8 GPU (12-141 GiB/GPU) hardware
// envelope and classifies the interconnect as PCIe, NVLink, or UMA. It fails closed
// on multi-node fabrics, >8 GPUs, out-of-range/unknown VRAM, or contradictory flags.
func ClassifySingleNodeBox(spec SingleNodeBoxSpec) (SingleNodeBoxProfile, PlacementRefusalReason, string) {
	if spec.NodeCount > 1 || strings.TrimSpace(spec.CrossNodeFabric) != "" {
		return SingleNodeBoxProfile{}, PlacementRefusalMultiNodeOutOfScope, "only single-node 1-8 GPU boxes are supported (multi-node/RDMA/NVL72 out of scope)"
	}
	if spec.Interconnect != "" && !spec.Interconnect.Valid() {
		return SingleNodeBoxProfile{}, PlacementRefusalTopologyMismatch, fmt.Sprintf("unknown interconnect %q", spec.Interconnect)
	}
	if spec.HeadroomFraction < 0 || spec.HeadroomFraction >= 1 || math.IsNaN(spec.HeadroomFraction) || math.IsInf(spec.HeadroomFraction, 0) {
		return SingleNodeBoxProfile{}, PlacementRefusalVRAMInvalid, fmt.Sprintf("invalid headroom fraction %v", spec.HeadroomFraction)
	}

	devices, reason, detail := normalizeSingleNodeDevices(spec)
	if reason.Refused() {
		return SingleNodeBoxProfile{}, reason, detail
	}

	usable := make([]int64, len(devices))
	var totalVRAM int64
	for i, d := range devices {
		if d.TotalVRAMBytes <= 0 {
			return SingleNodeBoxProfile{}, PlacementRefusalVRAMUnknown, fmt.Sprintf("gpu %d total VRAM is unknown (%d)", d.Index, d.TotalVRAMBytes)
		}
		if d.TotalVRAMBytes < MinSingleNodeGPUVRAMBytes || d.TotalVRAMBytes > MaxSingleNodeGPUVRAMBytes {
			return SingleNodeBoxProfile{}, PlacementRefusalVRAMOutOfRange, fmt.Sprintf("gpu %d VRAM %d bytes outside supported [12 GiB, 141 GiB] range", d.Index, d.TotalVRAMBytes)
		}
		avail := d.TotalVRAMBytes
		if d.FreeVRAMKnown {
			if d.FreeVRAMBytes < 0 || d.FreeVRAMBytes > d.TotalVRAMBytes {
				return SingleNodeBoxProfile{}, PlacementRefusalVRAMInvalid, fmt.Sprintf("gpu %d free VRAM %d outside [0, %d]", d.Index, d.FreeVRAMBytes, d.TotalVRAMBytes)
			}
			avail = d.FreeVRAMBytes
		}
		usable[i] = compute.BudgetAfterHeadroom(avail, spec.HeadroomFraction)
		var ok bool
		totalVRAM, ok = checkedAddInt64(totalVRAM, d.TotalVRAMBytes)
		if !ok {
			return SingleNodeBoxProfile{}, PlacementRefusalVRAMInvalid, "total VRAM overflow"
		}
	}

	kind, p2pCapable, reason, detail := classifyInterconnect(spec, devices, usable)
	if reason.Refused() {
		return SingleNodeBoxProfile{}, reason, detail
	}

	linkBW, linkLat, genericRates := resolveLinkRates(kind, spec)
	return SingleNodeBoxProfile{
		Interconnect:      kind,
		GPUCount:          len(devices),
		UnifiedMemory:     kind == InterconnectUMA,
		P2PCapable:        p2pCapable,
		LinkBandwidthGBs:  linkBW,
		LinkLatencyMicros: linkLat,
		GenericLinkRates:  genericRates,
		Devices:           devices,
		UsableVRAMBytes:   usable,
		TotalVRAMBytes:    totalVRAM,
	}, PlacementRefusalNone, ""
}

func normalizeSingleNodeDevices(spec SingleNodeBoxSpec) ([]SingleNodeDevice, PlacementRefusalReason, string) {
	if len(spec.GPUs) > 0 {
		if spec.GPUCount != 0 && spec.GPUCount != len(spec.GPUs) {
			return nil, PlacementRefusalTopologyMismatch, fmt.Sprintf("gpu_count=%d disagrees with len(gpus)=%d", spec.GPUCount, len(spec.GPUs))
		}
		if len(spec.GPUs) < MinSingleNodeGPUs || len(spec.GPUs) > MaxSingleNodeGPUs {
			return nil, PlacementRefusalGPUCountOutOfRange, fmt.Sprintf("gpu count %d outside [1, 8]", len(spec.GPUs))
		}
		devices := append([]SingleNodeDevice(nil), spec.GPUs...)
		seen := make(map[int]struct{}, len(devices))
		for i := range devices {
			if devices[i].Index < 0 {
				return nil, PlacementRefusalTopologyMismatch, fmt.Sprintf("negative gpu index %d", devices[i].Index)
			}
			if _, dup := seen[devices[i].Index]; dup {
				return nil, PlacementRefusalTopologyMismatch, fmt.Sprintf("duplicate gpu index %d", devices[i].Index)
			}
			seen[devices[i].Index] = struct{}{}
			if devices[i].DeviceID == "" {
				devices[i].DeviceID = fmt.Sprintf("gpu-%d", devices[i].Index)
			}
			if devices[i].MemoryBandwidthGBs <= 0 && spec.MemoryBandwidthGBs > 0 {
				devices[i].MemoryBandwidthGBs = spec.MemoryBandwidthGBs
			}
			if devices[i].ComputeTFLOPs <= 0 && spec.ComputeTFLOPs > 0 {
				devices[i].ComputeTFLOPs = spec.ComputeTFLOPs
			}
		}
		sort.Slice(devices, func(i, j int) bool { return devices[i].Index < devices[j].Index })
		return devices, PlacementRefusalNone, ""
	}

	if spec.GPUCount < MinSingleNodeGPUs || spec.GPUCount > MaxSingleNodeGPUs {
		return nil, PlacementRefusalGPUCountOutOfRange, fmt.Sprintf("gpu count %d outside [1, 8]", spec.GPUCount)
	}
	devices := make([]SingleNodeDevice, spec.GPUCount)
	for i := 0; i < spec.GPUCount; i++ {
		devices[i] = SingleNodeDevice{
			Index:              i,
			DeviceID:           fmt.Sprintf("gpu-%d", i),
			TotalVRAMBytes:     spec.PerGPUVRAMBytes,
			FreeVRAMBytes:      spec.FreeVRAMBytes,
			FreeVRAMKnown:      spec.FreeVRAMKnown,
			MemoryBandwidthGBs: spec.MemoryBandwidthGBs,
			ComputeTFLOPs:      spec.ComputeTFLOPs,
		}
	}
	return devices, PlacementRefusalNone, ""
}

func classifyInterconnect(spec SingleNodeBoxSpec, devices []SingleNodeDevice, usable []int64) (NodeInterconnect, bool, PlacementRefusalReason, string) {
	isUMA := spec.UnifiedMemory || spec.Interconnect == InterconnectUMA
	if isUMA {
		if spec.NVLinkActive || spec.Interconnect == InterconnectNVLink {
			return "", false, PlacementRefusalTopologyMismatch, "UMA box cannot simultaneously declare NVLink interconnect"
		}
		if len(devices) != 1 {
			return "", false, PlacementRefusalTopologyMismatch, fmt.Sprintf("UMA box requires 1 unified pool device, got %d", len(devices))
		}
		if spec.HostDRAMBytes > 0 {
			hostUsable := compute.BudgetAfterHeadroom(spec.HostDRAMBytes, spec.HeadroomFraction)
			if hostUsable < usable[0] {
				usable[0] = hostUsable
			}
		}
		return InterconnectUMA, false, PlacementRefusalNone, ""
	}

	isNVLink := spec.Interconnect == InterconnectNVLink ||
		(spec.Interconnect == "" && len(devices) > 1 && (spec.NVLinkActive || spec.LinkBandwidthGBs >= NVLinkMinBandwidthGBs))
	if isNVLink {
		if len(devices) < 2 {
			return "", false, PlacementRefusalTopologyMismatch, "NVLink interconnect requires at least 2 GPUs"
		}
		if spec.P2PUnavailable {
			return "", false, PlacementRefusalTopologyMismatch, "NVLink interconnect cannot have p2p_unavailable=true"
		}
		return InterconnectNVLink, true, PlacementRefusalNone, ""
	}

	if spec.NVLinkActive {
		return "", false, PlacementRefusalTopologyMismatch, "pcie interconnect contradicts nvlink_active=true"
	}
	p2p := !spec.P2PUnavailable
	if len(devices) == 1 {
		p2p = false
	}
	return InterconnectPCIe, p2p, PlacementRefusalNone, ""
}

func resolveLinkRates(kind NodeInterconnect, spec SingleNodeBoxSpec) (bw, lat float64, generic bool) {
	switch kind {
	case InterconnectUMA:
		return 0, 0, false
	case InterconnectNVLink:
		bw, lat = spec.LinkBandwidthGBs, spec.LinkLatencyMicros
		if bw <= 0 {
			bw = DefaultNVLinkBandwidthGBs
			generic = true
		}
		if lat <= 0 {
			lat = DefaultNVLinkLatencyMicros
			generic = true
		}
		return bw, lat, generic
	case InterconnectPCIe:
		bw, lat = spec.LinkBandwidthGBs, spec.LinkLatencyMicros
		if bw <= 0 {
			bw = DefaultPCIeLinkBandwidthGBs
			generic = true
		}
		if lat <= 0 {
			lat = DefaultPCIeLinkLatencyMicros
			generic = true
		}
		return bw, lat, generic
	default:
		return 0, 0, false
	}
}

// SingleNodeModelSpec carries the weight, KV, architecture, and per-rank overhead
// requirements for planning a model onto a single-node 1-8 GPU box.
type SingleNodeModelSpec struct {
	ModelID                      string           `json:"model_id,omitempty"`
	TotalWeightBytes             int64            `json:"total_weight_bytes,omitempty"`
	ShardedWeightBytes           int64            `json:"sharded_weight_bytes,omitempty"`
	ReplicatedWeightBytes        int64            `json:"replicated_weight_bytes,omitempty"`
	ActiveParams                 int64            `json:"active_params,omitempty"`
	HiddenSize                   int              `json:"hidden_size,omitempty"`
	AttnLayers                   int              `json:"attn_layers,omitempty"`
	NumAttentionHeads            int              `json:"num_attention_heads,omitempty"`
	NumKVHeads                   int              `json:"num_kv_heads,omitempty"`
	HeadDim                      int              `json:"head_dim,omitempty"`
	KVBytesPerElem               int              `json:"kv_bytes_per_elem,omitempty"`
	KVBytesPerToken              int64            `json:"kv_bytes_per_token,omitempty"`
	CompressedKV                 bool             `json:"compressed_kv,omitempty"`
	LoopedBlock                  LoopedBlockShape `json:"looped_block,omitempty"`
	RuntimeOverheadBytes         int64            `json:"runtime_overhead_bytes,omitempty"`
	ActivationWorkspaceBytes     int64            `json:"activation_workspace_bytes,omitempty"`
	CollectiveBufferBytesPerRank int64            `json:"collective_buffer_bytes_per_rank,omitempty"`
}

// EffectiveKVBytesPerToken returns the per-token KV cache footprint, deriving it
// from explicit geometry when KVBytesPerToken is unset. Returns 0 when unknown.
func (m SingleNodeModelSpec) EffectiveKVBytesPerToken() int64 {
	if m.KVBytesPerToken > 0 {
		return m.KVBytesPerToken
	}
	if m.AttnLayers <= 0 || m.NumKVHeads <= 0 || m.HeadDim <= 0 || m.KVBytesPerElem <= 0 {
		return 0
	}
	// 2 tensors (K and V) * AttnLayers * NumKVHeads * HeadDim * bytes/elem.
	v, ok := checkedMulInt64(2, int64(m.AttnLayers))
	if !ok {
		return 0
	}
	for _, factor := range []int64{int64(m.NumKVHeads), int64(m.HeadDim), int64(m.KVBytesPerElem)} {
		v, ok = checkedMulInt64(v, factor)
		if !ok {
			return 0
		}
	}
	return v
}

func (m SingleNodeModelSpec) effectiveWeights() (sharded, replicated int64, ok bool) {
	if m.TotalWeightBytes < 0 || m.ShardedWeightBytes < 0 || m.ReplicatedWeightBytes < 0 {
		return 0, 0, false
	}
	if m.ShardedWeightBytes > 0 || m.ReplicatedWeightBytes > 0 {
		sum, addOK := checkedAddInt64(m.ShardedWeightBytes, m.ReplicatedWeightBytes)
		if !addOK || sum <= 0 {
			return 0, 0, false
		}
		if m.TotalWeightBytes > 0 && m.TotalWeightBytes != sum {
			return 0, 0, false
		}
		return m.ShardedWeightBytes, m.ReplicatedWeightBytes, true
	}
	if m.TotalWeightBytes <= 0 {
		return 0, 0, false
	}
	return m.TotalWeightBytes, 0, true
}

// SingleNodeWorkloadSpec defines concurrency, context, and optional SLO constraints.
type SingleNodeWorkloadSpec struct {
	ConcurrentSlots    int                `json:"concurrent_slots"`
	MinSlotsPerReplica int                `json:"min_slots_per_replica,omitempty"`
	MaxSlotsPerReplica int                `json:"max_slots_per_replica,omitempty"`
	ContextTokens      int                `json:"context_tokens,omitempty"`
	PromptTokens       int                `json:"prompt_tokens,omitempty"`
	OutputTokens       int                `json:"output_tokens,omitempty"`
	ExecutionDepth     int                `json:"execution_depth,omitempty"`
	Objective          PlacementObjective `json:"objective,omitempty"`
	MaxTPOTMS          float64            `json:"max_tpot_ms,omitempty"`
	MaxTTFTMS          float64            `json:"max_ttft_ms,omitempty"`
}

func (w SingleNodeWorkloadSpec) effectiveContextTokens() int {
	if w.ContextTokens > 0 {
		return w.ContextTokens
	}
	if w.PromptTokens > 0 || w.OutputTokens > 0 {
		if w.PromptTokens < 0 || w.OutputTokens < 0 {
			return 0
		}
		return w.PromptTokens + w.OutputTokens
	}
	return 0
}

func (w SingleNodeWorkloadSpec) effectiveDepth() int {
	if w.ExecutionDepth <= 0 {
		return 1
	}
	return w.ExecutionDepth
}

// SingleNodePlacementRequest bundles box, model, workload, and optional pins.
type SingleNodePlacementRequest struct {
	Box      SingleNodeBoxSpec      `json:"box"`
	Model    SingleNodeModelSpec    `json:"model"`
	Workload SingleNodeWorkloadSpec `json:"workload"`
	PinnedDP int                    `json:"pinned_dp,omitempty"`
	PinnedTP int                    `json:"pinned_tp,omitempty"`
}

// ReplicaPlacement describes one admitted DP replica (possibly intra-node TP-sharded).
type ReplicaPlacement struct {
	ReplicaIndex         int           `json:"replica_index"`
	ReplicaID            string        `json:"replica_id"`
	GPUIDs               []int         `json:"gpu_ids"`
	TPDegree             int           `json:"tp_degree"`
	KVShards             int           `json:"kv_shards"`
	WeightsPerRankBytes  int64         `json:"weights_per_rank_bytes"`
	OverheadPerRankBytes int64         `json:"overhead_per_rank_bytes"`
	KVPerRankBytes       int64         `json:"kv_per_rank_bytes"`
	RequiredPerRankBytes int64         `json:"required_per_rank_bytes"`
	UsableMinRankBytes   int64         `json:"usable_min_rank_bytes"`
	MaxKVCacheBytes      int64         `json:"max_kv_cache_bytes"`
	MaxSlots             int           `json:"max_slots"`
	TPPlan               *model.TPPlan `json:"tp_plan,omitempty"`
}

// CandidateEvaluation records the per-(DP, TP) evaluation trace for auditability.
type CandidateEvaluation struct {
	Strategy             PlacementStrategy      `json:"strategy"`
	DP                   int                    `json:"dp"`
	TP                   int                    `json:"tp"`
	KVShards             int                    `json:"kv_shards"`
	SlotsPerReplica      int                    `json:"slots_per_replica"`
	WeightsPerRankBytes  int64                  `json:"weights_per_rank_bytes"`
	OverheadPerRankBytes int64                  `json:"overhead_per_rank_bytes"`
	KVPerRankBytes       int64                  `json:"kv_per_rank_bytes"`
	RequiredPerRankBytes int64                  `json:"required_per_rank_bytes"`
	EstTPOTMS            float64                `json:"est_tpot_ms,omitempty"`
	EstTTFTMS            float64                `json:"est_ttft_ms,omitempty"`
	EstAggregateTokSec   float64                `json:"est_aggregate_tok_sec,omitempty"`
	Admitted             bool                   `json:"admitted"`
	Refusal              PlacementRefusalReason `json:"refusal,omitempty"`
	Detail               string                 `json:"detail,omitempty"`
}

// SingleNodePlacementPlan is the fail-closed result of PlanSingleNodePlacement.
// When Admitted is false, Replicas is nil so no caller can use a partial plan.
type SingleNodePlacementPlan struct {
	Status        PlacementStatus        `json:"status"`
	Admitted      bool                   `json:"admitted"`
	Refusal       PlacementRefusalReason `json:"refusal,omitempty"`
	RefusalDetail string                 `json:"refusal_detail,omitempty"`
	Box           SingleNodeBoxProfile   `json:"box"`
	Strategy      PlacementStrategy      `json:"strategy,omitempty"`
	DP            int                    `json:"dp,omitempty"`
	TP            int                    `json:"tp,omitempty"`
	Replicas      []ReplicaPlacement     `json:"replicas,omitempty"`
	UnusedGPUIDs  []int                  `json:"unused_gpu_ids,omitempty"`
	Candidates    []CandidateEvaluation  `json:"candidates,omitempty"`
	Warnings      []string               `json:"warnings,omitempty"`
}

type evaluatedCandidate struct {
	eval      CandidateEvaluation
	replicas  []ReplicaPlacement
	unused    []int
	minSpare  int64
	rateKnown bool
}

// PlanSingleNodePlacement classifies the 1-8 GPU single-node box, enforces fail-closed
// VRAM + KV/overhead + concurrent slot admission across every rank, and selects between
// DP prefix-aware replicas and intra-node TP / hybrid DPxTP layouts.
func PlanSingleNodePlacement(req SingleNodePlacementRequest) SingleNodePlacementPlan {
	refusePlan := func(box SingleNodeBoxProfile, reason PlacementRefusalReason, detail string, cands []CandidateEvaluation) SingleNodePlacementPlan {
		return SingleNodePlacementPlan{
			Status:        PlacementRefused,
			Admitted:      false,
			Refusal:       reason,
			RefusalDetail: detail,
			Box:           box,
			Candidates:    cands,
		}
	}

	box, reason, detail := ClassifySingleNodeBox(req.Box)
	if reason.Refused() {
		return refusePlan(box, reason, detail, nil)
	}

	shardedWeights, replicatedWeights, ok := req.Model.effectiveWeights()
	if !ok {
		return refusePlan(box, PlacementRefusalModelInvalid, "positive model weight bytes required", nil)
	}
	kvPerToken := req.Model.EffectiveKVBytesPerToken()
	if kvPerToken <= 0 {
		return refusePlan(box, PlacementRefusalKVGeometryUnknown, "positive KVBytesPerToken or complete attention KV geometry required", nil)
	}
	if req.Model.RuntimeOverheadBytes < 0 || req.Model.ActivationWorkspaceBytes < 0 || req.Model.CollectiveBufferBytesPerRank < 0 {
		return refusePlan(box, PlacementRefusalOverheadUnbudgeted, "negative overhead bytes are invalid", nil)
	}
	baseOverhead, addOK := checkedAddInt64(req.Model.RuntimeOverheadBytes, req.Model.ActivationWorkspaceBytes)
	if !addOK {
		return refusePlan(box, PlacementRefusalOverheadUnbudgeted, "overhead bytes overflow", nil)
	}
	if baseOverhead == 0 && req.Box.HeadroomFraction <= 0 {
		return refusePlan(box, PlacementRefusalOverheadUnbudgeted, "positive runtime/activation overhead or headroom_fraction required", nil)
	}

	w := req.Workload
	if w.ConcurrentSlots < 1 {
		return refusePlan(box, PlacementRefusalSlotsInvalid, fmt.Sprintf("concurrent_slots=%d must be >= 1", w.ConcurrentSlots), nil)
	}
	maxSlotsPerReplica := w.MaxSlotsPerReplica
	if maxSlotsPerReplica <= 0 {
		maxSlotsPerReplica = DefaultMaxSlots
	}
	minSlotsPerReplica := w.MinSlotsPerReplica
	if minSlotsPerReplica <= 0 {
		minSlotsPerReplica = 1
	}
	if minSlotsPerReplica > maxSlotsPerReplica {
		return refusePlan(box, PlacementRefusalSlotsInvalid, fmt.Sprintf("min_slots_per_replica=%d > max_slots_per_replica=%d", minSlotsPerReplica, maxSlotsPerReplica), nil)
	}
	ctxTokens := w.effectiveContextTokens()
	if ctxTokens <= 0 {
		return refusePlan(box, PlacementRefusalKVGeometryUnknown, "positive context_tokens required", nil)
	}
	obj := w.Objective
	if obj == "" {
		obj = ObjectiveThroughput
	} else if !obj.Valid() {
		return refusePlan(box, PlacementRefusalModelInvalid, fmt.Sprintf("invalid objective %q", obj), nil)
	}
	if req.PinnedDP < 0 || req.PinnedTP < 0 {
		return refusePlan(box, PlacementRefusalTopologyMismatch, "negative pin value", nil)
	}
	if w.MaxTPOTMS < 0 || w.MaxTTFTMS < 0 || math.IsNaN(w.MaxTPOTMS) || math.IsNaN(w.MaxTTFTMS) {
		return refusePlan(box, PlacementRefusalModelInvalid, "negative or NaN SLO bound", nil)
	}

	kvPerSlot := perSlotKVBytes(req.Model, kvPerToken, ctxTokens, w.effectiveDepth())
	if kvPerSlot <= 0 {
		return refusePlan(box, PlacementRefusalKVGeometryUnknown, "per-slot KV bytes overflow or non-positive", nil)
	}

	tpDegrees := []int{1, 2, 4, 8}
	var candidates []CandidateEvaluation
	var admitted []evaluatedCandidate

	for _, tp := range tpDegrees {
		if tp > box.GPUCount {
			continue
		}
		if req.PinnedTP > 0 && tp != req.PinnedTP {
			continue
		}
		ev := evaluateTPCandidate(box, req, shardedWeights, replicatedWeights, baseOverhead, kvPerSlot, minSlotsPerReplica, maxSlotsPerReplica, tp)
		candidates = append(candidates, ev.eval)
		if ev.eval.Admitted {
			admitted = append(admitted, ev)
		}
	}

	if len(admitted) == 0 {
		if len(candidates) == 0 {
			return refusePlan(box, PlacementRefusalTopologyMismatch, fmt.Sprintf("pinned_tp=%d incompatible with gpu_count=%d", req.PinnedTP, box.GPUCount), nil)
		}
		primaryReason, primaryDetail := pickPrimaryRefusal(candidates)
		return refusePlan(box, primaryReason, primaryDetail, candidates)
	}

	sortAdmittedCandidates(admitted, obj, box.Interconnect)
	winner := admitted[0]

	var warnings []string
	if box.Interconnect == InterconnectPCIe && winner.eval.TP > 1 {
		warnings = append(warnings, fmt.Sprintf("pcie_tp_degraded: running tp=%d over PCIe (%.1f GB/s, %.1fus) incurs collective latency penalty",
			winner.eval.TP, box.LinkBandwidthGBs, box.LinkLatencyMicros))
	}
	if len(winner.unused) > 0 {
		warnings = append(warnings, fmt.Sprintf("partial_node_allocation: %d GPU(s) unused %v due to VRAM or TP group divisibility",
			len(winner.unused), winner.unused))
	}

	return SingleNodePlacementPlan{
		Status:       PlacementAdmitted,
		Admitted:     true,
		Box:          box,
		Strategy:     winner.eval.Strategy,
		DP:           winner.eval.DP,
		TP:           winner.eval.TP,
		Replicas:     winner.replicas,
		UnusedGPUIDs: winner.unused,
		Candidates:   candidates,
		Warnings:     warnings,
	}
}

func evaluateTPCandidate(
	box SingleNodeBoxProfile,
	req SingleNodePlacementRequest,
	shardedWeights, replicatedWeights, baseOverhead, kvPerSlot int64,
	minSlotsPerReplica, maxSlotsPerReplica, tp int,
) evaluatedCandidate {
	eval := CandidateEvaluation{TP: tp}
	refuse := func(reason PlacementRefusalReason, detail string) evaluatedCandidate {
		eval.Admitted = false
		eval.Refusal = reason
		eval.Detail = detail
		return evaluatedCandidate{eval: eval}
	}

	if box.Interconnect == InterconnectUMA && tp > 1 {
		return refuse(PlacementRefusalTopologyMismatch, "UMA box supports tp=1 only")
	}

	kvShards, tpPlan, reason, detail := checkModelTPCompatibility(req.Model, tp)
	eval.KVShards = kvShards
	if reason.Refused() {
		return refuse(reason, detail)
	}

	weightsPerRank, ok := checkedAddInt64(ceilDivInt64(shardedWeights, int64(tp)), replicatedWeights)
	if !ok {
		return refuse(PlacementRefusalModelInvalid, "per-rank weights overflow")
	}
	overheadPerRank := baseOverhead
	if tp > 1 {
		overheadPerRank, ok = checkedAddInt64(overheadPerRank, req.Model.CollectiveBufferBytesPerRank)
		if !ok {
			return refuse(PlacementRefusalOverheadUnbudgeted, "collective overhead overflow")
		}
	}
	eval.WeightsPerRankBytes = weightsPerRank
	eval.OverheadPerRankBytes = overheadPerRank

	// Determine which GPUs can host at least weights + overhead + 1 slot of KV.
	minSlotKVPerRank := ceilDivInt64(kvPerSlot, int64(kvShards))
	minNeed, ok := checkedAdd3Int64(weightsPerRank, overheadPerRank, minSlotKVPerRank)
	if !ok {
		return refuse(PlacementRefusalInsufficientVRAM, "minimum per-rank byte demand overflow")
	}

	var eligible []int
	for i := range box.Devices {
		if box.UsableVRAMBytes[i] >= minNeed {
			eligible = append(eligible, i)
		}
	}
	maxDP := len(eligible) / tp
	if maxDP == 0 {
		return refuse(PlacementRefusalInsufficientVRAM, fmt.Sprintf("tp=%d needs >= %d bytes/rank, insufficient usable VRAM across GPUs", tp, minNeed))
	}
	if req.PinnedDP > 0 {
		if req.PinnedDP > maxDP {
			return refuse(PlacementRefusalInsufficientVRAM, fmt.Sprintf("pinned_dp=%d x tp=%d exceeds %d feasible replica group(s)", req.PinnedDP, tp, maxDP))
		}
		maxDP = req.PinnedDP
	}

	// Select the largest dp in [1..maxDP] (or exact PinnedDP) whose per-replica
	// concurrent slot demand fits both slot cap and every rank's VRAM budget.
	startDP := maxDP
	endDP := 1
	if req.PinnedDP > 0 {
		endDP = req.PinnedDP
	}

	var lastReason PlacementRefusalReason = PlacementRefusalInsufficientVRAM
	var lastDetail string
	for dp := startDP; dp >= endDP; dp-- {
		slotsPerReplica := ceilDivInt(req.Workload.ConcurrentSlots, dp)
		if slotsPerReplica < minSlotsPerReplica {
			slotsPerReplica = minSlotsPerReplica
		}
		if slotsPerReplica > maxSlotsPerReplica {
			lastReason = PlacementRefusalSlotsExceedReplicaCapacity
			lastDetail = fmt.Sprintf("dp=%d requires %d slots/replica > max %d", dp, slotsPerReplica, maxSlotsPerReplica)
			continue
		}

		replicaKV, mulOK := checkedMulInt64(int64(slotsPerReplica), kvPerSlot)
		if !mulOK {
			lastReason = PlacementRefusalInsufficientVRAM
			lastDetail = "replica KV reservation overflow"
			continue
		}
		kvPerRank := ceilDivInt64(replicaKV, int64(kvShards))
		requiredPerRank, addOK := checkedAdd3Int64(weightsPerRank, overheadPerRank, kvPerRank)
		if !addOK {
			lastReason = PlacementRefusalInsufficientVRAM
			lastDetail = "per-rank VRAM requirement overflow"
			continue
		}

		groupIndices := packFitGroups(box.UsableVRAMBytes, tp, dp, requiredPerRank)
		if len(groupIndices) < dp {
			lastReason = PlacementRefusalInsufficientVRAM
			lastDetail = fmt.Sprintf("dp=%d, tp=%d requires %d bytes/rank (%d weights + %d kv + %d overhead)",
				dp, tp, requiredPerRank, weightsPerRank, kvPerRank, overheadPerRank)
			continue
		}

		// Enforce interconnect P2P / NVLink gates only once the candidate has passed
		// VRAM and slot capacity checks, so an infeasible model still reports
		// InsufficientVRAM while a VRAM-feasible PCIe TP group reports the interconnect gate.
		if tp > 1 && box.Interconnect == InterconnectPCIe {
			if req.Box.RequireNVLinkForTP {
				lastReason = PlacementRefusalTPRequiresNVLink
				lastDetail = fmt.Sprintf("tp=%d refused on PCIe box with require_nvlink_for_tp", tp)
				continue
			}
			if !box.P2PCapable {
				lastReason = PlacementRefusalP2PUnavailable
				lastDetail = fmt.Sprintf("tp=%d refused on PCIe box without P2P", tp)
				continue
			}
		}

		eval.DP = dp
		eval.Strategy = strategyForDPTP(dp, tp)
		eval.SlotsPerReplica = slotsPerReplica
		eval.KVPerRankBytes = kvPerRank
		eval.RequiredPerRankBytes = requiredPerRank

		tpotMS, ttftMS, aggTokSec, rateKnown := estimateCandidatePerformance(box, req, dp, tp, slotsPerReplica, weightsPerRank, kvPerRank)
		eval.EstTPOTMS = tpotMS
		eval.EstTTFTMS = ttftMS
		eval.EstAggregateTokSec = aggTokSec

		if req.Workload.MaxTPOTMS > 0 || req.Workload.MaxTTFTMS > 0 {
			if !rateKnown {
				lastReason = PlacementRefusalRateUnknownForSLO
				lastDetail = "hardware memory_bandwidth_gbs (and compute_tflops / active_params for TTFT) required for SLO admission"
				continue
			}
			if req.Workload.MaxTPOTMS > 0 && tpotMS > req.Workload.MaxTPOTMS {
				lastReason = PlacementRefusalSLOViolated
				lastDetail = fmt.Sprintf("dp=%d, tp=%d modeled TPOT %.3fms > max %.3fms", dp, tp, tpotMS, req.Workload.MaxTPOTMS)
				continue
			}
			if req.Workload.MaxTTFTMS > 0 && ttftMS > req.Workload.MaxTTFTMS {
				lastReason = PlacementRefusalSLOViolated
				lastDetail = fmt.Sprintf("dp=%d, tp=%d modeled TTFT %.3fms > max %.3fms", dp, tp, ttftMS, req.Workload.MaxTTFTMS)
				continue
			}
		}

		replicas, unused, minSpare := buildReplicaPlacements(box, groupIndices, eval, tpPlan)
		eval.Admitted = true
		return evaluatedCandidate{
			eval:      eval,
			replicas:  replicas,
			unused:    unused,
			minSpare:  minSpare,
			rateKnown: rateKnown,
		}
	}

	eval.DP = maxDP
	eval.Strategy = strategyForDPTP(maxDP, tp)
	return refuse(lastReason, lastDetail)
}

func checkModelTPCompatibility(m SingleNodeModelSpec, tp int) (kvShards int, tpPlan *model.TPPlan, reason PlacementRefusalReason, detail string) {
	if m.HiddenSize > 0 && m.HiddenSize%tp != 0 {
		return 0, nil, PlacementRefusalModelNotDivisibleByTP, fmt.Sprintf("hidden_size=%d not divisible by tp=%d", m.HiddenSize, tp)
	}
	if m.NumAttentionHeads > 0 && m.NumAttentionHeads%tp != 0 {
		return 0, nil, PlacementRefusalModelNotDivisibleByTP, fmt.Sprintf("num_attention_heads=%d not divisible by tp=%d", m.NumAttentionHeads, tp)
	}
	switch {
	case m.CompressedKV:
		kvShards = 1
	case m.NumKVHeads > 0:
		if m.NumKVHeads >= tp {
			if m.NumKVHeads%tp != 0 {
				return 0, nil, PlacementRefusalModelNotDivisibleByTP, fmt.Sprintf("num_kv_heads=%d not divisible by tp=%d", m.NumKVHeads, tp)
			}
			kvShards = tp
		} else {
			if tp%m.NumKVHeads != 0 {
				return 0, nil, PlacementRefusalModelNotDivisibleByTP, fmt.Sprintf("tp=%d not divisible by num_kv_heads=%d", tp, m.NumKVHeads)
			}
			kvShards = m.NumKVHeads
		}
	default:
		kvShards = tp
	}

	if tp > 1 {
		dim := m.NumAttentionHeads
		if dim <= 0 {
			dim = m.HiddenSize
		}
		if dim > 0 {
			p, err := model.NewTPPlan(dim, tp)
			if err != nil {
				return 0, nil, PlacementRefusalModelNotDivisibleByTP, err.Error()
			}
			tpPlan = &p
		}
	}
	return kvShards, tpPlan, PlacementRefusalNone, ""
}

func packFitGroups(usable []int64, tp, maxDP int, requiredPerRank int64) [][]int {
	if tp <= 0 || maxDP <= 0 {
		return nil
	}
	var eligible []int
	for i, u := range usable {
		if u >= requiredPerRank {
			eligible = append(eligible, i)
		}
	}
	if len(eligible)/tp < maxDP {
		return nil
	}
	groups := make([][]int, 0, maxDP)
	for g := 0; g < maxDP; g++ {
		start := g * tp
		grp := append([]int(nil), eligible[start:start+tp]...)
		groups = append(groups, grp)
	}
	return groups
}

func buildReplicaPlacements(box SingleNodeBoxProfile, groups [][]int, eval CandidateEvaluation, tpPlan *model.TPPlan) ([]ReplicaPlacement, []int, int64) {
	used := make([]bool, len(box.Devices))
	replicas := make([]ReplicaPlacement, len(groups))
	minSpare := int64(math.MaxInt64)

	for r, grp := range groups {
		gpuIDs := make([]int, len(grp))
		usableMin := int64(math.MaxInt64)
		for k, devIdx := range grp {
			used[devIdx] = true
			gpuIDs[k] = box.Devices[devIdx].Index
			if box.UsableVRAMBytes[devIdx] < usableMin {
				usableMin = box.UsableVRAMBytes[devIdx]
			}
		}
		spare := usableMin - eval.RequiredPerRankBytes
		if spare < minSpare {
			minSpare = spare
		}

		kvBudgetPerRank := usableMin - eval.WeightsPerRankBytes - eval.OverheadPerRankBytes
		if kvBudgetPerRank < eval.KVPerRankBytes {
			kvBudgetPerRank = eval.KVPerRankBytes
		}
		maxKVCacheBytes, ok := checkedMulInt64(kvBudgetPerRank, int64(eval.KVShards))
		if !ok {
			maxKVCacheBytes = math.MaxInt64
		}

		var planCopy *model.TPPlan
		if tpPlan != nil {
			cp := model.TPPlan{
				Dim:    tpPlan.Dim,
				Shards: append([]model.TPShard(nil), tpPlan.Shards...),
			}
			planCopy = &cp
		}

		replicas[r] = ReplicaPlacement{
			ReplicaIndex:         r,
			ReplicaID:            fmt.Sprintf("replica-%d", r),
			GPUIDs:               gpuIDs,
			TPDegree:             eval.TP,
			KVShards:             eval.KVShards,
			WeightsPerRankBytes:  eval.WeightsPerRankBytes,
			OverheadPerRankBytes: eval.OverheadPerRankBytes,
			KVPerRankBytes:       eval.KVPerRankBytes,
			RequiredPerRankBytes: eval.RequiredPerRankBytes,
			UsableMinRankBytes:   usableMin,
			MaxKVCacheBytes:      maxKVCacheBytes,
			MaxSlots:             eval.SlotsPerReplica,
			TPPlan:               planCopy,
		}
	}

	var unused []int
	for i, isUsed := range used {
		if !isUsed {
			unused = append(unused, box.Devices[i].Index)
		}
	}
	if minSpare == int64(math.MaxInt64) {
		minSpare = 0
	}
	return replicas, unused, minSpare
}

func estimateCandidatePerformance(
	box SingleNodeBoxProfile,
	req SingleNodePlacementRequest,
	dp, tp, slotsPerReplica int,
	weightsPerRank, kvPerRank int64,
) (tpotMS, ttftMS, aggTokSec float64, rateKnown bool) {
	if len(box.Devices) == 0 {
		return 0, 0, 0, false
	}
	minMemBW := box.Devices[0].MemoryBandwidthGBs
	minCompute := box.Devices[0].ComputeTFLOPs
	for _, d := range box.Devices {
		if d.MemoryBandwidthGBs <= 0 {
			return 0, 0, 0, false
		}
		if d.MemoryBandwidthGBs < minMemBW {
			minMemBW = d.MemoryBandwidthGBs
		}
		if d.ComputeTFLOPs < minCompute {
			minCompute = d.ComputeTFLOPs
		}
	}
	if tp > 1 && box.LinkBandwidthGBs <= 0 {
		return 0, 0, 0, false
	}

	needTTFT := req.Workload.MaxTTFTMS > 0
	computeKnown := req.Model.ActiveParams > 0 && minCompute > 0
	if needTTFT && !computeKnown {
		return 0, 0, 0, false
	}

	hidden := req.Model.HiddenSize
	if hidden <= 0 {
		hidden = 4096
	}
	layers := req.Model.AttnLayers
	if layers <= 0 {
		layers = 32
	}

	weightStreamSec := float64(weightsPerRank) / (minMemBW * 1e9)
	kvReadSec := float64(kvPerRank) / (minMemBW * 1e9)
	decodeComputeSec := 0.0
	if computeKnown {
		decodeComputeSec = float64(int64(slotsPerReplica)*2*req.Model.ActiveParams/int64(tp)) / (minCompute * 1e12)
	}

	decodeLinkSec := 0.0
	if tp > 1 {
		payload := float64(slotsPerReplica) * float64(hidden) * 2.0
		ringBytes := 2.0 * float64(tp-1) / float64(tp) * payload
		decodeLinkSec = float64(layers) * 2.0 * (box.LinkLatencyMicros*1e-6 + ringBytes/(box.LinkBandwidthGBs*1e9))
	}

	stepSec := math.Max(weightStreamSec+kvReadSec, decodeComputeSec) + decodeLinkSec
	if stepSec <= 0 {
		return 0, 0, 0, false
	}
	tpotMS = stepSec * 1000.0
	aggTokSec = float64(dp*slotsPerReplica) / stepSec

	if computeKnown {
		promptLen := req.Workload.PromptTokens
		if promptLen <= 0 {
			promptLen = req.Workload.effectiveContextTokens()
		}
		prefillComputeSec := float64(2*int64(promptLen)*req.Model.ActiveParams/int64(tp)) / (minCompute * 1e12)
		prefillLinkSec := 0.0
		if tp > 1 {
			payload := float64(promptLen) * float64(hidden) * 2.0
			ringBytes := 2.0 * float64(tp-1) / float64(tp) * payload
			prefillLinkSec = float64(layers) * 2.0 * (box.LinkLatencyMicros*1e-6 + ringBytes/(box.LinkBandwidthGBs*1e9))
		}
		ttftMS = (math.Max(weightStreamSec, prefillComputeSec) + prefillLinkSec) * 1000.0
	}
	return tpotMS, ttftMS, aggTokSec, true
}

func sortAdmittedCandidates(admitted []evaluatedCandidate, obj PlacementObjective, ic NodeInterconnect) {
	sort.SliceStable(admitted, func(i, j int) bool {
		a, b := admitted[i], admitted[j]
		switch obj {
		case ObjectiveLatency:
			if a.rateKnown && b.rateKnown && a.eval.EstTPOTMS != b.eval.EstTPOTMS {
				return a.eval.EstTPOTMS < b.eval.EstTPOTMS
			}
			if a.eval.TP != b.eval.TP {
				if ic == InterconnectNVLink {
					return a.eval.TP > b.eval.TP
				}
				return a.eval.TP < b.eval.TP
			}
			if a.eval.DP != b.eval.DP {
				return a.eval.DP > b.eval.DP
			}
			return a.minSpare > b.minSpare
		case ObjectiveThroughput:
			if a.eval.DP != b.eval.DP {
				return a.eval.DP > b.eval.DP
			}
			if a.eval.TP != b.eval.TP {
				return a.eval.TP < b.eval.TP
			}
			return a.minSpare > b.minSpare
		default:
			if a.eval.DP != b.eval.DP {
				return a.eval.DP > b.eval.DP
			}
			return a.eval.TP < b.eval.TP
		}
	})
}

func pickPrimaryRefusal(candidates []CandidateEvaluation) (PlacementRefusalReason, string) {
	priority := []PlacementRefusalReason{
		PlacementRefusalSLOViolated,
		PlacementRefusalRateUnknownForSLO,
		PlacementRefusalSlotsExceedReplicaCapacity,
		PlacementRefusalTPRequiresNVLink,
		PlacementRefusalP2PUnavailable,
		PlacementRefusalInsufficientVRAM,
		PlacementRefusalModelNotDivisibleByTP,
		PlacementRefusalTopologyMismatch,
	}
	for _, target := range priority {
		for _, c := range candidates {
			if c.Refusal == target {
				return c.Refusal, c.Detail
			}
		}
	}
	return candidates[0].Refusal, candidates[0].Detail
}

func strategyForDPTP(dp, tp int) PlacementStrategy {
	switch {
	case tp <= 1:
		return PlacementDataParallel
	case dp <= 1:
		return PlacementTensorParallel
	default:
		return PlacementHybridDPTP
	}
}

func perSlotKVBytes(m SingleNodeModelSpec, kvPerToken int64, ctxTokens, depth int) int64 {
	if kvPerToken <= 0 || ctxTokens <= 0 || depth <= 0 {
		return 0
	}
	base, ok := checkedMulInt64(int64(ctxTokens), kvPerToken)
	if !ok {
		return 0
	}
	if m.LoopedBlock.Declared() {
		visits := int64(m.LoopedBlock.LayerVisits(depth))
		scaled, mulOK := checkedMulInt64(base, visits)
		if !mulOK {
			return 0
		}
		return scaled / int64(m.LoopedBlock.PhysicalLayers())
	}
	res, ok := checkedMulInt64(base, int64(depth))
	if !ok {
		return 0
	}
	return res
}

func checkedAddInt64(a, b int64) (int64, bool) {
	if b > 0 && a > math.MaxInt64-b {
		return 0, false
	}
	if b < 0 && a < math.MinInt64-b {
		return 0, false
	}
	return a + b, true
}

func checkedAdd3Int64(a, b, c int64) (int64, bool) {
	ab, ok := checkedAddInt64(a, b)
	if !ok {
		return 0, false
	}
	return checkedAddInt64(ab, c)
}

func checkedMulInt64(a, b int64) (int64, bool) {
	if a == 0 || b == 0 {
		return 0, true
	}
	if a > math.MaxInt64/b {
		return 0, false
	}
	return a * b, true
}

func ceilDivInt64(num, den int64) int64 {
	if den <= 0 || num <= 0 {
		return 0
	}
	return (num-1)/den + 1
}

func ceilDivInt(num, den int) int {
	if den <= 0 || num <= 0 {
		return 0
	}
	return (num-1)/den + 1
}
