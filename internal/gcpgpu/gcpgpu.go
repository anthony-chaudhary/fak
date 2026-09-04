// Package gcpgpu manages GCP GPU fleet compute nodes, accelerator descriptors
// (L4, A100, H100, T4), health probes, VRAM capacity calculation, quota tracking,
// and remote dispatch readiness.
//
// Pure Go, zero external dependencies, robust thread-safety (sync.RWMutex),
// error handling with typed sentinel errors.
package gcpgpu

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// Standard binary byte units for VRAM and host RAM calculations.
// Invariant: binary byte multipliers must follow IEC power-of-two definitions.
// Guard: byte scale constants are immutable positive powers of two.
const (
	// KiB represents kibibytes (1024 bytes).
	KiB = int64(1024)
	// MiB represents mebibytes (1024 KiB).
	MiB = int64(1024 * 1024)
	// GiB represents gibibytes (1024 MiB).
	GiB = int64(1024 * 1024 * 1024)
	// TiB represents tebibytes (1024 GiB).
	TiB = int64(1024 * 1024 * 1024 * 1024)
)

// AcceleratorType identifies supported GCP GPU accelerator hardware types.
// Invariant: values must correspond to supported GCP accelerator designations.
// Guard: unsupported accelerator strings fail IsValid and return errors during parse.
type AcceleratorType string

// Accelerator constants for supported GCP GPU accelerator hardware types.
// Invariant: accelerator types must map to valid GCP compute engine accelerator designations.
// Guard: unknown designations return zero-value capabilities and fail parse operations.
const (
	// L4 represents the NVIDIA L4 Ada Lovelace 24GB accelerator.
	L4 AcceleratorType = "nvidia-l4"
	// A100_40GB represents the NVIDIA A100 Ampere 40GB accelerator.
	A100_40GB AcceleratorType = "nvidia-tesla-a100"
	// A100_80GB represents the NVIDIA A100 Ampere 80GB accelerator.
	A100_80GB AcceleratorType = "nvidia-a100-80gb"
	// H100_80GB represents the NVIDIA H100 Hopper 80GB accelerator.
	H100_80GB AcceleratorType = "nvidia-h100-80gb"
	// T4 represents the NVIDIA T4 Turing 16GB accelerator.
	T4 AcceleratorType = "nvidia-tesla-t4"

	// AccelL4 is an alias for L4.
	AccelL4 = L4
	// AccelA100_40GB is an alias for A100_40GB.
	AccelA100_40GB = A100_40GB
	// AccelA100_80GB is an alias for A100_80GB.
	AccelA100_80GB = A100_80GB
	// AccelH100_80GB is an alias for H100_80GB.
	AccelH100_80GB = H100_80GB
	// AccelT4 is an alias for T4.
	AccelT4 = T4
)

// MemoryPerGPU returns standard physical VRAM capacity per GPU in bytes.
// Invariant: returns positive byte capacity for supported accelerators; 0 for invalid types.
// Guard: unsupported accelerators return 0 bytes to fail closed against unverified capacity.
func (a AcceleratorType) MemoryPerGPU() int64 {
	switch a {
	case L4:
		return 24 * GiB
	case A100_40GB:
		return 40 * GiB
	case A100_80GB:
		return 80 * GiB
	case H100_80GB:
		return 80 * GiB
	case T4:
		return 16 * GiB
	default:
		return 0
	}
}

// ComputeCapability returns the CUDA compute capability version (major.minor).
// Invariant: returns non-empty major.minor string for supported accelerators.
// Guard: unsupported accelerators return an empty string to prevent invalid dispatch.
func (a AcceleratorType) ComputeCapability() string {
	switch a {
	case L4:
		return "8.9"
	case A100_40GB, A100_80GB:
		return "8.0"
	case H100_80GB:
		return "9.0"
	case T4:
		return "7.5"
	default:
		return ""
	}
}

// Architecture returns the GPU micro-architecture family name.
// Invariant: returns known micro-architecture string for supported accelerators, or "Unknown".
// Guard: unsupported accelerators return "Unknown" to fail closed against invalid execution plans.
func (a AcceleratorType) Architecture() string {
	switch a {
	case L4:
		return "Ada Lovelace"
	case A100_40GB, A100_80GB:
		return "Ampere"
	case H100_80GB:
		return "Hopper"
	case T4:
		return "Turing"
	default:
		return "Unknown"
	}
}

// GenerationRank returns relative hardware generation rank (higher = newer silicon).
// Used for priority tie-breaking in scheduling decisions.
// Invariant: newer GPU architectures must have strictly higher rank than older ones.
// Guard: unsupported accelerators return rank 0 to rank lowest in selection.
func (a AcceleratorType) GenerationRank() int {
	switch a {
	case H100_80GB:
		return 90
	case A100_80GB:
		return 80
	case A100_40GB:
		return 75
	case L4:
		return 70
	case T4:
		return 50
	default:
		return 0
	}
}

// GPUFamily returns the GCP Quotas API family name for this accelerator.
// Invariant: maps supported accelerators to standard GCP quota metric identifiers.
// Guard: unsupported accelerators return empty string to prevent invalid quota queries.
func (a AcceleratorType) GPUFamily() string {
	switch a {
	case L4:
		return "NVIDIA_L4"
	case A100_40GB:
		return "NVIDIA_A100"
	case A100_80GB:
		return "NVIDIA_A100_80GB"
	case H100_80GB:
		return "NVIDIA_H100"
	case T4:
		return "NVIDIA_T4"
	default:
		return ""
	}
}

// IsValid reports whether the accelerator type is one of the recognized GCP GPU types.
// Invariant: returns true only for supported NVIDIA GPU architectures (L4, A100, H100, T4).
// Guard: unknown or malformed accelerator designations return false.
func (a AcceleratorType) IsValid() bool {
	switch a {
	case L4, A100_40GB, A100_80GB, H100_80GB, T4:
		return true
	default:
		return false
	}
}

// ParseAccelerator parses a string (slug, short name, or GCP type) into a recognized AcceleratorType.
// Invariant: case-insensitive and trims whitespace; returns ErrInvalidAccelerator for unrecognized input.
// Guard: unrecognized strings return ErrInvalidAccelerator to prevent unregistered device allocation.
func ParseAccelerator(s string) (AcceleratorType, error) {
	norm := strings.ToLower(strings.TrimSpace(s))
	switch norm {
	case "l4", "nvidia-l4", "nvidia_l4":
		return L4, nil
	case "a100-40gb", "a100_40gb", "nvidia-tesla-a100", "a100-40", "a100":
		return A100_40GB, nil
	case "a100-80gb", "a100_80gb", "nvidia-a100-80gb", "a100-80":
		return A100_80GB, nil
	case "h100", "h100-80gb", "h100_80gb", "nvidia-h100-80gb", "h100-80":
		return H100_80GB, nil
	case "t4", "nvidia-t4", "nvidia-tesla-t4", "nvidia_t4":
		return T4, nil
	default:
		return "", fmt.Errorf("%w: %q", ErrInvalidAccelerator, s)
	}
}

// InstanceSpec describes the hardware topology and capacity of a GCP GPU instance.
// Invariant: GPUCount and VCPUs must be positive; TotalVRAMBytes equals GPUCount * VRAMBytesPerGPU.
// Guard: specifications with non-positive GPU counts or negative memory are rejected on construction.
type InstanceSpec struct {
	MachineType     string          `json:"machine_type"`       // GCP machine type, e.g. "g2-standard-8", "a2-highgpu-1g"
	Accelerator     AcceleratorType `json:"accelerator"`        // L4, A100_40GB, A100_80GB, H100_80GB, T4
	GPUCount        int             `json:"gpu_count"`          // Number of attached GPUs (1, 2, 4, 8)
	VCPUs           int             `json:"vcpus"`              // Host vCPU count
	HostMemoryBytes int64           `json:"host_memory_bytes"`  // Host system RAM in bytes
	VRAMBytesPerGPU int64           `json:"vram_bytes_per_gpu"` // Per-GPU VRAM in bytes
	TotalVRAMBytes  int64           `json:"total_vram_bytes"`   // GPUCount * VRAMBytesPerGPU
}

// NewInstanceSpec constructs and validates an InstanceSpec, populating VRAM defaults if omitted.
// Invariant: machineType must be non-empty, accelerator must be valid, and gpuCount/vcpus must be positive.
// Guard: returns ErrInvalidNodeSpec or ErrInvalidAccelerator when constraints are violated.
func NewInstanceSpec(machineType string, accel AcceleratorType, gpuCount int, vcpus int, hostMemBytes int64) (InstanceSpec, error) {
	if strings.TrimSpace(machineType) == "" {
		return InstanceSpec{}, fmt.Errorf("%w: machine type cannot be empty", ErrInvalidNodeSpec)
	}
	if !accel.IsValid() {
		return InstanceSpec{}, fmt.Errorf("%w: %q", ErrInvalidAccelerator, accel)
	}
	if gpuCount <= 0 {
		return InstanceSpec{}, fmt.Errorf("%w: gpu count must be positive (got %d)", ErrInvalidNodeSpec, gpuCount)
	}
	if vcpus <= 0 {
		return InstanceSpec{}, fmt.Errorf("%w: vcpus must be positive (got %d)", ErrInvalidNodeSpec, vcpus)
	}
	if hostMemBytes < 0 {
		return InstanceSpec{}, fmt.Errorf("%w: host memory bytes cannot be negative", ErrInvalidNodeSpec)
	}

	vramPerGPU := accel.MemoryPerGPU()
	spec := InstanceSpec{
		MachineType:     strings.TrimSpace(machineType),
		Accelerator:     accel,
		GPUCount:        gpuCount,
		VCPUs:           vcpus,
		HostMemoryBytes: hostMemBytes,
		VRAMBytesPerGPU: vramPerGPU,
		TotalVRAMBytes:  int64(gpuCount) * vramPerGPU,
	}
	return spec, nil
}

// NodeStatus defines the operational lifecycle state of a GPU node.
// Invariant: must be one of Ready, Busy, Degraded, or Offline.
// Guard: unrecognized status values fail IsValid checks and are blocked during updates.
type NodeStatus string

// Operational status constants defining node availability states.
// Invariant: a registered node must always hold one of Ready, Busy, Degraded, or Offline.
// Guard: non-Ready statuses prevent workload dispatch.
const (
	// Ready indicates the node is operational and accepting workloads.
	Ready NodeStatus = "Ready"
	// Busy indicates the node is at full capacity with no available VRAM.
	Busy NodeStatus = "Busy"
	// Degraded indicates the node has failed health checks or exceeded thermal thresholds.
	Degraded NodeStatus = "Degraded"
	// Offline indicates the node has been manually or administratively taken offline.
	Offline NodeStatus = "Offline"

	// NodeStatusReady is an alias for Ready.
	NodeStatusReady = Ready
	// NodeStatusBusy is an alias for Busy.
	NodeStatusBusy = Busy
	// NodeStatusDegraded is an alias for Degraded.
	NodeStatusDegraded = Degraded
	// NodeStatusOffline is an alias for Offline.
	NodeStatusOffline = Offline
)

// IsValid reports whether the status is one of Ready, Busy, Degraded, or Offline.
// Invariant: returns true only for recognized lifecycle states.
// Guard: invalid status strings return false to block illegal state transitions.
func (s NodeStatus) IsValid() bool {
	switch s {
	case Ready, Busy, Degraded, Offline:
		return true
	default:
		return false
	}
}

// TelemetryMetrics holds real-time utilization, memory, and thermal telemetry for a node.
// Invariant: AllocatedVRAMBytes and UsedVRAMBytes must be non-negative; GPUUtilizationPct is in [0, 100].
// Guard: negative memory counters are clamped to zero; temperatures >= 95C trigger Degraded state.
type TelemetryMetrics struct {
	AllocatedVRAMBytes int64     `json:"allocated_vram_bytes"`
	UsedVRAMBytes      int64     `json:"used_vram_bytes"`
	GPUUtilizationPct  float64   `json:"gpu_utilization_pct"` // 0.0 - 100.0
	TemperatureCelsius float64   `json:"temperature_celsius"`
	PowerUsageWatts    float64   `json:"power_usage_watts"`
	ActiveJobs         int       `json:"active_jobs"`
	LastHeartbeat      time.Time `json:"last_heartbeat"`
}

// NodeTelemetry is an alias for TelemetryMetrics.
// Invariant: identical structure and semantics to TelemetryMetrics.
// Guard: inherits the same safety bounds and clamping rules as TelemetryMetrics.
type NodeTelemetry = TelemetryMetrics

// GPUNode represents an individual GCP GPU host registered in the fleet.
// Invariant: ID must be non-empty and unique within FleetManager; AvailableVRAM must not exceed TotalVRAMBytes.
// Guard: nodes missing endpoints, carrying health errors, or holding non-Ready status cannot accept work.
type GPUNode struct {
	ID           string            `json:"id"`
	Name         string            `json:"name"`
	Project      string            `json:"project"`
	Zone         string            `json:"zone"`
	Region       string            `json:"region"`
	Spec         InstanceSpec      `json:"spec"`
	Status       NodeStatus        `json:"status"`
	Endpoint     string            `json:"endpoint"` // Remote dispatch address (e.g. "10.128.0.5:4765")
	Labels       map[string]string `json:"labels"`
	Telemetry    TelemetryMetrics  `json:"telemetry"`
	RegisteredAt time.Time         `json:"registered_at"`
	LastProbedAt time.Time         `json:"last_probed_at"`
	HealthErrors []string          `json:"health_errors,omitempty"`
}

// AvailableVRAM returns the free VRAM in bytes on the node.
// Invariant: returns non-negative value bounded by Spec.TotalVRAMBytes.
// Guard: nil node or over-allocated VRAM returns 0 to prevent negative capacity assumptions.
func (n *GPUNode) AvailableVRAM() int64 {
	if n == nil {
		return 0
	}
	allocated := n.Telemetry.AllocatedVRAMBytes
	if n.Telemetry.UsedVRAMBytes > allocated {
		allocated = n.Telemetry.UsedVRAMBytes
	}
	avail := n.Spec.TotalVRAMBytes - allocated
	if avail < 0 {
		return 0
	}
	return avail
}

// IsDispatchReady reports whether the node is ready for immediate remote workload dispatch.
// Invariant: returns true only if Status is Ready, endpoint is set, HealthErrors is empty, and AvailableVRAM > 0.
// Guard: any health error, empty endpoint, non-Ready status, or zero VRAM returns false.
func (n *GPUNode) IsDispatchReady() bool {
	if n == nil {
		return false
	}
	if n.Status != Ready {
		return false
	}
	if strings.TrimSpace(n.Endpoint) == "" {
		return false
	}
	if len(n.HealthErrors) > 0 {
		return false
	}
	return n.AvailableVRAM() > 0
}

// Clone produces a deep copy of GPUNode to prevent cross-goroutine mutation.
// Invariant: deep-copies labels and health errors slices.
// Guard: nil receiver returns an empty GPUNode struct safely.
func (n *GPUNode) Clone() GPUNode {
	if n == nil {
		return GPUNode{}
	}
	cp := *n
	if n.Labels != nil {
		cp.Labels = make(map[string]string, len(n.Labels))
		for k, v := range n.Labels {
			cp.Labels[k] = v
		}
	}
	if n.HealthErrors != nil {
		cp.HealthErrors = make([]string, len(n.HealthErrors))
		copy(cp.HealthErrors, n.HealthErrors)
	}
	return cp
}

// ProbeResult captures the verified health state and telemetry check of a node probe.
// Invariant: Healthy is true only when Status is Ready and Errors is empty.
// Guard: probe failures or context cancellations populate Errors and mark Healthy false.
type ProbeResult struct {
	NodeID        string        `json:"node_id"`
	Timestamp     time.Time     `json:"timestamp"`
	Healthy       bool          `json:"healthy"`
	Status        NodeStatus    `json:"status"`
	Latency       time.Duration `json:"latency"`
	TotalVRAM     int64         `json:"total_vram"`
	AvailableVRAM int64         `json:"available_vram"`
	Message       string        `json:"message"`
	Errors        []string      `json:"errors,omitempty"`
}

// QuotaSpec records GPU quota limits and live consumption for an accelerator in a region.
// Invariant: Limit must be non-negative; Available returns max(0, Limit - InUse).
// Guard: Available clamps negative headroom to 0 when InUse exceeds Limit.
type QuotaSpec struct {
	Accelerator AcceleratorType `json:"accelerator"`
	Region      string          `json:"region"`
	Limit       int             `json:"limit"`  // Total GPU units allowed by GCP quota
	InUse       int             `json:"in_use"` // Total GPU units registered across active nodes
}

// Available returns the remaining unallocated GPU quota.
// Invariant: never returns negative; clamps to zero when InUse exceeds Limit.
// Guard: over-subscribed quota returns 0 to block additional node admissions.
func (q QuotaSpec) Available() int {
	avail := q.Limit - q.InUse
	if avail < 0 {
		return 0
	}
	return avail
}

// AccelCapacity summarizes capacity for a specific accelerator type across the fleet.
// Invariant: TotalNodes equals ReadyNodes + BusyNodes + DegradedNodes + OfflineNodes for this accelerator.
// Guard: metrics provide a point-in-time read-only view of accelerator resources.
type AccelCapacity struct {
	Accelerator        AcceleratorType `json:"accelerator"`
	TotalNodes         int             `json:"total_nodes"`
	ReadyNodes         int             `json:"ready_nodes"`
	TotalGPUs          int             `json:"total_gpus"`
	ReadyGPUs          int             `json:"ready_gpus"`
	TotalVRAMBytes     int64           `json:"total_vram_bytes"`
	AllocatedVRAMBytes int64           `json:"allocated_vram_bytes"`
	AvailableVRAMBytes int64           `json:"available_vram_bytes"`
	QuotaLimit         int             `json:"quota_limit"`
	QuotaInUse         int             `json:"quota_in_use"`
}

// RegionCapacity summarizes capacity metrics for a specific GCP region.
// Invariant: TotalNodes and TotalGPUs equal the sum of registered nodes and GPUs in this region.
// Guard: region capacity reports provide read-only isolation across regional boundaries.
type RegionCapacity struct {
	Region             string `json:"region"`
	TotalNodes         int    `json:"total_nodes"`
	ReadyNodes         int    `json:"ready_nodes"`
	TotalGPUs          int    `json:"total_gpus"`
	ReadyGPUs          int    `json:"ready_gpus"`
	TotalVRAMBytes     int64  `json:"total_vram_bytes"`
	AvailableVRAMBytes int64  `json:"available_vram_bytes"`
}

// CapacityReport provides a consolidated snapshot of fleet-wide GPU compute and memory readiness.
// Invariant: TotalNodes equals sum of ReadyNodes, BusyNodes, DegradedNodes, and OfflineNodes.
// Guard: unready or degraded nodes are isolated from DispatchReadyNodes.
type CapacityReport struct {
	Timestamp          time.Time                         `json:"timestamp"`
	TotalNodes         int                               `json:"total_nodes"`
	ReadyNodes         int                               `json:"ready_nodes"`
	BusyNodes          int                               `json:"busy_nodes"`
	DegradedNodes      int                               `json:"degraded_nodes"`
	OfflineNodes       int                               `json:"offline_nodes"`
	TotalGPUs          int                               `json:"total_gpus"`
	ReadyGPUs          int                               `json:"ready_gpus"`
	TotalVRAMBytes     int64                             `json:"total_vram_bytes"`
	AllocatedVRAMBytes int64                             `json:"allocated_vram_bytes"`
	AvailableVRAMBytes int64                             `json:"available_vram_bytes"`
	ByAccelerator      map[AcceleratorType]AccelCapacity `json:"by_accelerator"`
	ByRegion           map[string]RegionCapacity         `json:"by_region"`
	DispatchReadyNodes []string                          `json:"dispatch_ready_nodes"`
}

// Sentinel errors for deterministic error checking with errors.Is.
// Invariant: all sentinel errors are typed and immutable singletons.
// Guard: sentinel comparisons prevent ambiguous failure classification across subsystems.
var (
	// ErrNodeNotFound is returned when the specified node ID does not exist in the fleet.
	// Invariant: indicates lookup failure for an un-registered node ID.
	// Guard: operations targeting missing IDs fail closed immediately.
	ErrNodeNotFound = errors.New("gcpgpu: node not found")
	// ErrNodeAlreadyExists is returned when attempting to register a node with an existing ID.
	// Invariant: node registration requires unique node IDs.
	// Guard: duplicate IDs are rejected without overwriting existing registrations.
	ErrNodeAlreadyExists = errors.New("gcpgpu: node already registered")
	// ErrInvalidNodeSpec is returned when a node specification fails parameter validation.
	// Invariant: specifications must have positive GPU count, vCPUs, and valid parameters.
	// Guard: malformed hardware topologies are rejected on admission.
	ErrInvalidNodeSpec = errors.New("gcpgpu: invalid node specification")
	// ErrInvalidAccelerator is returned when an accelerator type string is not recognized.
	// Invariant: only supported GCP accelerator types are permitted.
	// Guard: unrecognized accelerator types fail closed to prevent erroneous allocation.
	ErrInvalidAccelerator = errors.New("gcpgpu: unsupported accelerator type")
	// ErrInsufficientVRAM is returned when no ready node has sufficient free VRAM for a model.
	// Invariant: dispatch requests require available VRAM >= required bytes.
	// Guard: model dispatch requests exceeding capacity are rejected without partial allocation.
	ErrInsufficientVRAM = errors.New("gcpgpu: insufficient VRAM for model")
	// ErrNoReadyNodes is returned when the fleet has no nodes in Ready status.
	// Invariant: at least one node must be Ready to service model dispatch requests.
	// Guard: dispatch halts fail-closed when zero nodes are in operational status.
	ErrNoReadyNodes = errors.New("gcpgpu: no ready nodes available")
	// ErrNodeNotReady is returned when an operation requires a Ready node but the node is not Ready.
	// Invariant: allocation requires node Status == Ready.
	// Guard: Busy, Degraded, or Offline nodes reject direct allocation attempts.
	ErrNodeNotReady = errors.New("gcpgpu: node is not ready for dispatch")
	// ErrQuotaExceeded is returned when registering a node would exceed the regional GPU quota.
	// Invariant: enforced only when FleetManager is configured with quota enforcement.
	// Guard: registrations exceeding regional limits fail closed when enforcement is active.
	ErrQuotaExceeded = errors.New("gcpgpu: accelerator quota exceeded")
	// ErrFleetEmpty is returned when an operation requires registered nodes but fleet size is zero.
	// Invariant: operations requiring nodes fail-fast when len(nodes) == 0.
	// Guard: empty fleets reject node scheduling requests immediately.
	ErrFleetEmpty = errors.New("gcpgpu: fleet has no registered nodes")
	// ErrProbeFailed is returned when a node health probe cannot execute.
	// Invariant: indicates transport or context failure during health probe execution.
	// Guard: probe execution failures mark nodes Degraded.
	ErrProbeFailed = errors.New("gcpgpu: probe execution failed")
	// ErrNodeDegraded is returned when a node is degraded due to health errors or thermal throttling.
	// Invariant: degraded nodes are excluded from dispatch scheduling.
	// Guard: degraded nodes reject new workload assignments until recovered.
	ErrNodeDegraded = errors.New("gcpgpu: node is in degraded state")
	// ErrNodeOffline is returned when an operation is attempted on an offline node.
	// Invariant: offline nodes cannot receive allocations or dispatch workloads.
	// Guard: administratively offline nodes are excluded from all active dispatch routes.
	ErrNodeOffline = errors.New("gcpgpu: node is offline")
)

// ProberFunc defines the signature for node health checking probes.
// Invariant: probers must respect context cancellation and return non-nil ProbeResult.
// Guard: unhandled errors during probing return Degraded status and record failure details.
type ProberFunc func(ctx context.Context, node *GPUNode) (ProbeResult, error)

// FleetManager coordinates GCP GPU compute nodes, capacity allocation, health probes,
// telemetry, quota constraints, and remote dispatch readiness.
// Invariant: all exported methods are concurrency-safe via sync.RWMutex synchronization.
// Guard: invalid inputs and unready node states return typed sentinel errors.
type FleetManager struct {
	mu           sync.RWMutex
	nodes        map[string]*GPUNode
	quotas       map[string]*QuotaSpec // key: region + ":" + accelerator
	enforceQuota bool
	prober       ProberFunc
	timeNow      func() time.Time
}

// Option configures a FleetManager instance.
// Invariant: options mutate FleetManager state before concurrent use.
// Guard: functional options apply only during initialization before concurrent access.
type Option func(*FleetManager)

// WithQuotaEnforcement enables rejection of node registration if regional GPU quota is exceeded.
// Invariant: enforces quota limits when enforce is true.
// Guard: registration attempts exceeding quota return ErrQuotaExceeded when enabled.
func WithQuotaEnforcement(enforce bool) Option {
	return func(m *FleetManager) {
		m.enforceQuota = enforce
	}
}

// WithProber overrides the default health check prober implementation.
// Invariant: custom prober replaces defaultProber when provided.
// Guard: nil prober falls back safely to defaultProber during probe execution.
func WithProber(p ProberFunc) Option {
	return func(m *FleetManager) {
		m.prober = p
	}
}

// WithTimeSource overrides time.Now for deterministic testing.
// Invariant: replaces wall-clock time with deterministic time provider.
// Guard: nil provider falls back safely to time.Now().UTC().
func WithTimeSource(fn func() time.Time) Option {
	return func(m *FleetManager) {
		m.timeNow = fn
	}
}

// NewFleetManager constructs an initialized FleetManager.
// Invariant: returns non-nil manager with initialized maps and default clock.
// Guard: allocations and quota maps are initialized empty to prevent nil-map panics.
func NewFleetManager(opts ...Option) *FleetManager {
	m := &FleetManager{
		nodes:   make(map[string]*GPUNode),
		quotas:  make(map[string]*QuotaSpec),
		timeNow: time.Now,
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

func (m *FleetManager) now() time.Time {
	if m.timeNow != nil {
		return m.timeNow().UTC()
	}
	return time.Now().UTC()
}

func quotaKey(region string, accel AcceleratorType) string {
	return strings.ToLower(strings.TrimSpace(region)) + ":" + string(accel)
}

// DeriveRegion extracts a region from a zone (e.g. "us-central1-a" -> "us-central1") or returns input.
// Invariant: strips single-letter zone suffix following last dash.
// Guard: inputs without single-letter zone suffix are returned unchanged.
func DeriveRegion(zoneOrRegion string) string {
	s := strings.TrimSpace(zoneOrRegion)
	if idx := strings.LastIndex(s, "-"); idx != -1 {
		// If last segment is a single letter zone suffix like "-a", strip it.
		suffix := s[idx+1:]
		if len(suffix) == 1 && suffix >= "a" && suffix <= "z" {
			return s[:idx]
		}
	}
	return s
}

// RegisterNode registers a new GPUNode in the fleet.
// It validates spec parameters, calculates VRAM totals, verifies quotas if enforced,
// and stores the node thread-safely.
// Invariant: fails with ErrNodeAlreadyExists if node ID is already registered; verifies quota if enforced.
// Guard: invalid node specs or quota exceedances return typed errors and reject registration.
func (m *FleetManager) RegisterNode(node GPUNode) error {
	nodeID := strings.TrimSpace(node.ID)
	if nodeID == "" {
		return fmt.Errorf("%w: node ID is required", ErrInvalidNodeSpec)
	}

	if !node.Spec.Accelerator.IsValid() {
		return fmt.Errorf("%w: %q", ErrInvalidAccelerator, node.Spec.Accelerator)
	}

	if node.Spec.GPUCount <= 0 {
		return fmt.Errorf("%w: gpu count must be positive (got %d)", ErrInvalidNodeSpec, node.Spec.GPUCount)
	}

	// Auto-fill VRAM per GPU if zero
	if node.Spec.VRAMBytesPerGPU <= 0 {
		node.Spec.VRAMBytesPerGPU = node.Spec.Accelerator.MemoryPerGPU()
	}
	if node.Spec.TotalVRAMBytes <= 0 {
		node.Spec.TotalVRAMBytes = int64(node.Spec.GPUCount) * node.Spec.VRAMBytesPerGPU
	}

	// Normalise status
	if node.Status == "" {
		node.Status = Ready
	} else if !node.Status.IsValid() {
		return fmt.Errorf("%w: invalid status %q", ErrInvalidNodeSpec, node.Status)
	}

	// Derive region if empty
	if strings.TrimSpace(node.Region) == "" && strings.TrimSpace(node.Zone) != "" {
		node.Region = DeriveRegion(node.Zone)
	}
	node.Region = strings.TrimSpace(node.Region)

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.nodes[nodeID]; exists {
		return fmt.Errorf("%w: node %q", ErrNodeAlreadyExists, nodeID)
	}

	// Quota check
	qKey := quotaKey(node.Region, node.Spec.Accelerator)
	if q, ok := m.quotas[qKey]; ok {
		if m.enforceQuota && (q.InUse+node.Spec.GPUCount) > q.Limit {
			return fmt.Errorf("%w: region %q accel %q limit %d would be exceeded by %d (in use: %d)",
				ErrQuotaExceeded, node.Region, node.Spec.Accelerator, q.Limit, node.Spec.GPUCount, q.InUse)
		}
		q.InUse += node.Spec.GPUCount
	}

	now := m.now()
	if node.RegisteredAt.IsZero() {
		node.RegisteredAt = now
	}
	if node.Telemetry.LastHeartbeat.IsZero() {
		node.Telemetry.LastHeartbeat = now
	}

	nodeCopy := node.Clone()
	nodeCopy.ID = nodeID
	m.nodes[nodeID] = &nodeCopy
	return nil
}

// UnregisterNode removes a node from the fleet and releases its quota allocation.
// Invariant: returns ErrNodeNotFound if ID does not exist; decrements quota InUse counter.
// Guard: missing node IDs return ErrNodeNotFound without modifying fleet state.
func (m *FleetManager) UnregisterNode(nodeID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	node, exists := m.nodes[nodeID]
	if !exists {
		return fmt.Errorf("%w: %q", ErrNodeNotFound, nodeID)
	}

	qKey := quotaKey(node.Region, node.Spec.Accelerator)
	if q, ok := m.quotas[qKey]; ok {
		q.InUse -= node.Spec.GPUCount
		if q.InUse < 0 {
			q.InUse = 0
		}
	}

	delete(m.nodes, nodeID)
	return nil
}

// GetNode retrieves a deep-copy of a node by ID.
// Invariant: returns ErrNodeNotFound if ID does not exist; returned node is isolated from internal state.
// Guard: missing node IDs return ErrNodeNotFound and an empty GPUNode value.
func (m *FleetManager) GetNode(nodeID string) (GPUNode, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	node, exists := m.nodes[nodeID]
	if !exists {
		return GPUNode{}, fmt.Errorf("%w: %q", ErrNodeNotFound, nodeID)
	}
	return node.Clone(), nil
}

// ListNodes returns a slice of all registered nodes in the fleet.
// Invariant: returned slice is sorted lexicographically by node ID.
// Guard: returns empty non-nil slice when no nodes are registered.
func (m *FleetManager) ListNodes() []GPUNode {
	m.mu.RLock()
	defer m.mu.RUnlock()

	res := make([]GPUNode, 0, len(m.nodes))
	for _, n := range m.nodes {
		res = append(res, n.Clone())
	}
	// Sort by ID for deterministic output
	sort.Slice(res, func(i, j int) bool {
		return res[i].ID < res[j].ID
	})
	return res
}

// UpdateNodeStatus explicitly updates the operational status of a node.
// Invariant: status must be valid; returns ErrNodeNotFound if node does not exist.
// Guard: invalid status values return ErrInvalidNodeSpec without altering node state.
func (m *FleetManager) UpdateNodeStatus(nodeID string, status NodeStatus) error {
	if !status.IsValid() {
		return fmt.Errorf("%w: invalid status %q", ErrInvalidNodeSpec, status)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	node, exists := m.nodes[nodeID]
	if !exists {
		return fmt.Errorf("%w: %q", ErrNodeNotFound, nodeID)
	}

	node.Status = status
	return nil
}

// UpdateTelemetry updates live telemetry metrics for a node and evaluates status thresholds.
// Invariant: thermal readings >= 95C transition node to Degraded; memory saturation toggles Busy/Ready.
// Guard: missing node IDs return ErrNodeNotFound; negative metrics are clamped to zero.
func (m *FleetManager) UpdateTelemetry(nodeID string, metrics TelemetryMetrics) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	node, exists := m.nodes[nodeID]
	if !exists {
		return fmt.Errorf("%w: %q", ErrNodeNotFound, nodeID)
	}

	if metrics.LastHeartbeat.IsZero() {
		metrics.LastHeartbeat = m.now()
	}
	if metrics.AllocatedVRAMBytes < 0 {
		metrics.AllocatedVRAMBytes = 0
	}
	if metrics.UsedVRAMBytes < 0 {
		metrics.UsedVRAMBytes = 0
	}

	node.Telemetry = metrics

	// Thermal safety guard: critical thermal levels trigger Degraded state
	const criticalTempCelsius = 95.0
	if metrics.TemperatureCelsius >= criticalTempCelsius {
		node.Status = Degraded
		errMsg := fmt.Sprintf("thermal critical: %.1fC >= %.1fC", metrics.TemperatureCelsius, criticalTempCelsius)
		hasErr := false
		for _, e := range node.HealthErrors {
			if e == errMsg {
				hasErr = true
				break
			}
		}
		if !hasErr {
			node.HealthErrors = append(node.HealthErrors, errMsg)
		}
	}

	// Dynamic Busy / Ready transition based on VRAM allocation saturation
	if node.Status == Ready && metrics.AllocatedVRAMBytes >= node.Spec.TotalVRAMBytes {
		node.Status = Busy
	} else if node.Status == Busy && metrics.AllocatedVRAMBytes < node.Spec.TotalVRAMBytes {
		node.Status = Ready
	}

	return nil
}

// BestNodeForModel finds the most optimal dispatch-ready node to host a model
// requiring vramRequiredBytes.
//
// Selection Strategy:
//  1. Filter: Candidate must be in Ready status, have an active endpoint, zero health errors,
//     and available VRAM >= vramRequiredBytes.
//  2. Best-Fit: Minimizes unused headroom (availableVRAM - vramRequiredBytes) to pack
//     models efficiently and leave higher-capacity nodes (e.g. H100) available for larger workloads.
//  3. Tie-breakers: Lowest GPU utilization pct, fewest active jobs, higher silicon generation rank,
//     and lexicographical node ID.
//
// Invariant: candidate must be Ready, endpoint non-empty, error-free, and satisfy VRAM >= required.
// Guard: returns ErrFleetEmpty, ErrNoReadyNodes, or ErrInsufficientVRAM when no node qualifies.
func (m *FleetManager) BestNodeForModel(vramRequiredBytes int64) (*GPUNode, error) {
	if vramRequiredBytes <= 0 {
		return nil, fmt.Errorf("%w: required VRAM must be positive (got %d)", ErrInvalidNodeSpec, vramRequiredBytes)
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.nodes) == 0 {
		return nil, ErrFleetEmpty
	}

	readyNodesCount := 0
	var candidates []*GPUNode

	for _, node := range m.nodes {
		if node.Status == Ready {
			readyNodesCount++
		}
		if !node.IsDispatchReady() {
			continue
		}
		if node.AvailableVRAM() >= vramRequiredBytes {
			candidates = append(candidates, node)
		}
	}

	if readyNodesCount == 0 {
		return nil, ErrNoReadyNodes
	}

	if len(candidates) == 0 {
		return nil, fmt.Errorf("%w: needed %d bytes (%.2f GiB)",
			ErrInsufficientVRAM, vramRequiredBytes, float64(vramRequiredBytes)/float64(GiB))
	}

	// Sort candidates according to the multi-tiered best-fit policy
	sort.Slice(candidates, func(i, j int) bool {
		ci, cj := candidates[i], candidates[j]
		wasteI := ci.AvailableVRAM() - vramRequiredBytes
		wasteJ := cj.AvailableVRAM() - vramRequiredBytes

		// Primary: minimize wasted headroom (best-fit)
		if wasteI != wasteJ {
			return wasteI < wasteJ
		}

		// Secondary: lower current GPU utilization
		if ci.Telemetry.GPUUtilizationPct != cj.Telemetry.GPUUtilizationPct {
			return ci.Telemetry.GPUUtilizationPct < cj.Telemetry.GPUUtilizationPct
		}

		// Tertiary: fewer active jobs
		if ci.Telemetry.ActiveJobs != cj.Telemetry.ActiveJobs {
			return ci.Telemetry.ActiveJobs < cj.Telemetry.ActiveJobs
		}

		// Quaternary: prefer newer silicon architecture
		rankI := ci.Spec.Accelerator.GenerationRank()
		rankJ := cj.Spec.Accelerator.GenerationRank()
		if rankI != rankJ {
			return rankI > rankJ
		}

		// Quinary: deterministic lexicographical ID
		return ci.ID < cj.ID
	})

	best := candidates[0].Clone()
	return &best, nil
}

// AllocateVRAM reserves a slice of VRAM on a specific node.
// Returns ErrInsufficientVRAM if the node cannot satisfy the request, or ErrNodeNotReady.
// Invariant: allocates bytes only if node is Ready and AvailableVRAM >= bytes; saturating VRAM marks Busy.
// Guard: requests exceeding available VRAM or targeting non-Ready nodes return typed errors.
func (m *FleetManager) AllocateVRAM(nodeID string, bytes int64) error {
	if bytes <= 0 {
		return fmt.Errorf("%w: bytes to allocate must be positive", ErrInvalidNodeSpec)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	node, exists := m.nodes[nodeID]
	if !exists {
		return fmt.Errorf("%w: %q", ErrNodeNotFound, nodeID)
	}

	if node.Status != Ready {
		return fmt.Errorf("%w: node %q is in status %s", ErrNodeNotReady, nodeID, node.Status)
	}

	avail := node.AvailableVRAM()
	if avail < bytes {
		return fmt.Errorf("%w: node %q has %d bytes available, requested %d",
			ErrInsufficientVRAM, nodeID, avail, bytes)
	}

	node.Telemetry.AllocatedVRAMBytes += bytes
	node.Telemetry.ActiveJobs++

	if node.Telemetry.AllocatedVRAMBytes >= node.Spec.TotalVRAMBytes {
		node.Status = Busy
	}

	return nil
}

// ReleaseVRAM releases previously allocated VRAM on a node.
// Invariant: decrements AllocatedVRAMBytes floor-clamped at 0; restores Ready if busy due to memory.
// Guard: missing node IDs return ErrNodeNotFound; allocated VRAM is prevented from underflowing below 0.
func (m *FleetManager) ReleaseVRAM(nodeID string, bytes int64) error {
	if bytes <= 0 {
		return fmt.Errorf("%w: bytes to release must be positive", ErrInvalidNodeSpec)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	node, exists := m.nodes[nodeID]
	if !exists {
		return fmt.Errorf("%w: %q", ErrNodeNotFound, nodeID)
	}

	node.Telemetry.AllocatedVRAMBytes -= bytes
	if node.Telemetry.AllocatedVRAMBytes < 0 {
		node.Telemetry.AllocatedVRAMBytes = 0
	}
	if node.Telemetry.ActiveJobs > 0 {
		node.Telemetry.ActiveJobs--
	}

	// If node was busy due to memory saturation, restore Ready
	if node.Status == Busy && node.Telemetry.AllocatedVRAMBytes < node.Spec.TotalVRAMBytes && len(node.HealthErrors) == 0 {
		node.Status = Ready
	}

	return nil
}

// defaultProber conducts a comprehensive local health and reachability evaluation.
func defaultProber(ctx context.Context, node *GPUNode) (ProbeResult, error) {
	start := time.Now()

	result := ProbeResult{
		NodeID:        node.ID,
		Timestamp:     start.UTC(),
		TotalVRAM:     node.Spec.TotalVRAMBytes,
		AvailableVRAM: node.AvailableVRAM(),
	}

	// Check context cancellation
	if err := ctx.Err(); err != nil {
		result.Healthy = false
		result.Status = Degraded
		result.Message = fmt.Sprintf("probe context cancelled: %v", err)
		result.Errors = []string{err.Error()}
		result.Latency = time.Since(start)
		return result, err
	}

	var probeErrors []string

	// Endpoint check
	if strings.TrimSpace(node.Endpoint) == "" {
		probeErrors = append(probeErrors, "dispatch endpoint is unconfigured")
	}

	// Offline status check
	if node.Status == Offline {
		probeErrors = append(probeErrors, "node is marked offline")
	}

	// Heartbeat staleness check (> 5 minutes)
	const heartbeatTimeout = 5 * time.Minute
	if !node.Telemetry.LastHeartbeat.IsZero() && time.Since(node.Telemetry.LastHeartbeat) > heartbeatTimeout {
		probeErrors = append(probeErrors, fmt.Sprintf("stale heartbeat (%s ago)", time.Since(node.Telemetry.LastHeartbeat).Round(time.Second)))
	}

	// Existing health errors
	if len(node.HealthErrors) > 0 {
		probeErrors = append(probeErrors, node.HealthErrors...)
	}

	result.Latency = time.Since(start)

	if len(probeErrors) > 0 {
		result.Healthy = false
		if node.Status == Offline {
			result.Status = Offline
		} else {
			result.Status = Degraded
		}
		result.Errors = probeErrors
		result.Message = strings.Join(probeErrors, "; ")
		return result, nil
	}

	result.Healthy = true
	result.Status = node.Status
	result.Message = "node is healthy and ready for dispatch"
	return result, nil
}

// ProbeNode evaluates the health and readiness of a specific node.
// It executes the configured ProberFunc without holding locks during probe execution,
// then applies the verified state back to the node thread-safely.
// Invariant: prober runs lock-free; results update node status and LastProbedAt under write lock.
// Guard: missing node IDs return ErrNodeNotFound; probe errors set node status to Degraded.
func (m *FleetManager) ProbeNode(ctx context.Context, nodeID string) (ProbeResult, error) {
	// Snapshot node under read lock
	m.mu.RLock()
	node, exists := m.nodes[nodeID]
	if !exists {
		m.mu.RUnlock()
		return ProbeResult{}, fmt.Errorf("%w: %q", ErrNodeNotFound, nodeID)
	}
	nodeCopy := node.Clone()
	prober := m.prober
	m.mu.RUnlock()

	if prober == nil {
		prober = defaultProber
	}

	// Execute probe without holding lock
	result, err := prober(ctx, &nodeCopy)

	// Apply results back under write lock
	m.mu.Lock()
	defer m.mu.Unlock()

	node, exists = m.nodes[nodeID]
	if !exists {
		return result, fmt.Errorf("%w: %q", ErrNodeNotFound, nodeID)
	}

	now := m.now()
	node.LastProbedAt = now

	if !result.Healthy {
		if node.Status != Offline {
			node.Status = result.Status
		}
		node.HealthErrors = result.Errors
	} else {
		if node.Status == Degraded {
			node.Status = Ready
		}
		node.HealthErrors = nil
	}

	return result, err
}

// ProbeAll concurrently probes all nodes in the fleet and returns their ProbeResults.
// Invariant: probes execute concurrently across goroutines; results sorted by node ID.
// Guard: returns nil slice and nil error immediately when the fleet has no registered nodes.
func (m *FleetManager) ProbeAll(ctx context.Context) ([]ProbeResult, error) {
	m.mu.RLock()
	nodeIDs := make([]string, 0, len(m.nodes))
	for id := range m.nodes {
		nodeIDs = append(nodeIDs, id)
	}
	m.mu.RUnlock()

	if len(nodeIDs) == 0 {
		return nil, nil
	}

	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		results = make([]ProbeResult, 0, len(nodeIDs))
	)

	for _, id := range nodeIDs {
		wg.Add(1)
		go func(nid string) {
			defer wg.Done()
			res, _ := m.ProbeNode(ctx, nid)
			mu.Lock()
			results = append(results, res)
			mu.Unlock()
		}(id)
	}

	wg.Wait()

	sort.Slice(results, func(i, j int) bool {
		return results[i].NodeID < results[j].NodeID
	})
	return results, nil
}

// SetQuota registers or updates the GCP GPU quota limit for an accelerator in a region.
// Invariant: recalculates InUse count from currently registered nodes; limit must be non-negative.
// Guard: negative limits or invalid accelerator types return ErrInvalidNodeSpec / ErrInvalidAccelerator.
func (m *FleetManager) SetQuota(accel AcceleratorType, region string, limit int) error {
	if !accel.IsValid() {
		return fmt.Errorf("%w: %q", ErrInvalidAccelerator, accel)
	}
	reg := strings.ToLower(strings.TrimSpace(region))
	if reg == "" {
		return fmt.Errorf("%w: region is required", ErrInvalidNodeSpec)
	}
	if limit < 0 {
		return fmt.Errorf("%w: quota limit cannot be negative (got %d)", ErrInvalidNodeSpec, limit)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	qKey := quotaKey(reg, accel)
	q, exists := m.quotas[qKey]
	if !exists {
		q = &QuotaSpec{
			Accelerator: accel,
			Region:      reg,
		}
		m.quotas[qKey] = q
	}
	q.Limit = limit

	// Recompute live InUse from registered nodes
	inUse := 0
	for _, n := range m.nodes {
		if strings.EqualFold(n.Region, reg) && n.Spec.Accelerator == accel {
			inUse += n.Spec.GPUCount
		}
	}
	q.InUse = inUse

	return nil
}

// GetQuota retrieves the quota spec for an accelerator and region.
// Invariant: returns copy of QuotaSpec and true if configured, or empty QuotaSpec and false if absent.
// Guard: unconfigured quotas return false and zero-value QuotaSpec to prevent false headroom assumptions.
func (m *FleetManager) GetQuota(accel AcceleratorType, region string) (QuotaSpec, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	qKey := quotaKey(region, accel)
	q, exists := m.quotas[qKey]
	if !exists {
		return QuotaSpec{}, false
	}
	return *q, true
}

// CapacityReport computes a consolidated snapshot of fleet-wide GPU compute and memory readiness.
// Invariant: consolidates counts across all nodes and provides breakdown by accelerator and region.
// Guard: dispatch ready node list filters out non-Ready or failing nodes fail-closed.
func (m *FleetManager) CapacityReport() CapacityReport {
	m.mu.RLock()
	defer m.mu.RUnlock()

	report := CapacityReport{
		Timestamp:          m.now(),
		TotalNodes:         len(m.nodes),
		ByAccelerator:      make(map[AcceleratorType]AccelCapacity),
		ByRegion:           make(map[string]RegionCapacity),
		DispatchReadyNodes: make([]string, 0),
	}

	for _, node := range m.nodes {
		accel := node.Spec.Accelerator
		reg := node.Region
		if reg == "" {
			reg = "unknown"
		}

		gpus := node.Spec.GPUCount
		totalVRAM := node.Spec.TotalVRAMBytes
		allocatedVRAM := node.Telemetry.AllocatedVRAMBytes
		availVRAM := node.AvailableVRAM()

		report.TotalGPUs += gpus
		report.TotalVRAMBytes += totalVRAM
		report.AllocatedVRAMBytes += allocatedVRAM
		report.AvailableVRAMBytes += availVRAM

		switch node.Status {
		case Ready:
			report.ReadyNodes++
			report.ReadyGPUs += gpus
		case Busy:
			report.BusyNodes++
		case Degraded:
			report.DegradedNodes++
		case Offline:
			report.OfflineNodes++
		}

		if node.IsDispatchReady() {
			report.DispatchReadyNodes = append(report.DispatchReadyNodes, node.ID)
		}

		// Rollup by Accelerator
		ac := report.ByAccelerator[accel]
		ac.Accelerator = accel
		ac.TotalNodes++
		ac.TotalGPUs += gpus
		ac.TotalVRAMBytes += totalVRAM
		ac.AllocatedVRAMBytes += allocatedVRAM
		ac.AvailableVRAMBytes += availVRAM
		if node.Status == Ready {
			ac.ReadyNodes++
			ac.ReadyGPUs += gpus
		}
		// Populate quota if configured
		if q, ok := m.quotas[quotaKey(reg, accel)]; ok {
			ac.QuotaLimit = q.Limit
			ac.QuotaInUse = q.InUse
		}
		report.ByAccelerator[accel] = ac

		// Rollup by Region
		rc := report.ByRegion[reg]
		rc.Region = reg
		rc.TotalNodes++
		rc.TotalGPUs += gpus
		rc.TotalVRAMBytes += totalVRAM
		rc.AvailableVRAMBytes += availVRAM
		if node.Status == Ready {
			rc.ReadyNodes++
			rc.ReadyGPUs += gpus
		}
		report.ByRegion[reg] = rc
	}

	sort.Strings(report.DispatchReadyNodes)
	return report
}
