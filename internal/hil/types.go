package hil

import "time"

// Schema constants for hardware-in-the-loop artifacts.
const (
	ReportSchema     = "fak.hil.report.v1"
	MicroDoseSchema  = "fak.hil.microdose.v1"
	ComparisonSchema = "fak.hil.comparison.v1"
	InventorySchema  = "fak.hil.inventory.v1"
)

// HardwareKind classifies the physical accelerator or compute architecture.
type HardwareKind string

const (
	HardwareMetal     HardwareKind = "metal"
	HardwareCUDA      HardwareKind = "cuda"
	HardwareVulkan    HardwareKind = "vulkan"
	HardwareROCm      HardwareKind = "rocm"
	HardwareStrixHalo HardwareKind = "strix_halo"
	HardwareCPUSIMD   HardwareKind = "cpu_simd"
	HardwareUnknown   HardwareKind = "unknown"
)

// DoseKind identifies the specific micro-dose probe executed.
type DoseKind string

const (
	DoseLiveness      DoseKind = "liveness"
	DoseComputeGEMV   DoseKind = "compute_gemv"
	DoseComputeGEMM   DoseKind = "compute_gemm"
	DoseBandwidth     DoseKind = "bandwidth"
	DoseNumericParity DoseKind = "numeric_parity"
	DoseLANLiveness   DoseKind = "lan_liveness"
)

// LANNodeStatus represents the operational readiness of a remote LAN compute node.
type LANNodeStatus string

const (
	LANNodeOnline       LANNodeStatus = "ONLINE"
	LANNodeOffline      LANNodeStatus = "OFFLINE"
	LANNodeUnconfigured LANNodeStatus = "UNCONFIGURED"
)

// LANNodeInfo records the connectivity, appliance identity, and transport of a LAN accelerator node.
type LANNodeInfo struct {
	Status        LANNodeStatus `json:"status"`
	Host          string        `json:"host"`
	Reachable     bool          `json:"reachable"`
	LatencyMicros int64         `json:"latency_micros"`
	Transport     string        `json:"transport"` // "http_healthz" | "tcp_port" | "ssh_port" | "offline"
	Endpoint      string        `json:"endpoint,omitempty"`
	Appliance     string        `json:"appliance,omitempty"`
	GPU           string        `json:"gpu,omitempty"`
	Error         string        `json:"error,omitempty"`
}

// ComparisonVerdict specifies the outcome of evaluating a reported comparison.
type ComparisonVerdict string

const (
	// VerdictRealHardwareVerified indicates both comparison arms are measured on physical silicon.
	VerdictRealHardwareVerified ComparisonVerdict = "VERIFIED_REAL_HARDWARE"
	// VerdictEarlyIndicatorOnly indicates at least one arm is simulated; valid only as early indicator.
	VerdictEarlyIndicatorOnly ComparisonVerdict = "EARLY_INDICATOR_ONLY"
	// VerdictRejectedSimulation indicates a simulated result was improperly claimed as an achieved win.
	VerdictRejectedSimulation ComparisonVerdict = "REJECTED_UNPROVEN_SIMULATION"
	// VerdictInvalidComparison indicates missing metrics or malformed envelope.
	VerdictInvalidComparison ComparisonVerdict = "INVALID_COMPARISON"
)

// HardwareInfo records the physical silicon identity and device properties.
type HardwareInfo struct {
	Kind              HardwareKind      `json:"kind"`
	DeviceName        string            `json:"device_name"`
	Architecture      string            `json:"architecture"`
	Platform          string            `json:"platform"`
	MemoryTotalBytes  uint64            `json:"memory_total_bytes"`
	MemoryUnified     bool              `json:"memory_unified"`
	PhysicalAvailable bool              `json:"physical_available"`
	Details           map[string]string `json:"details,omitempty"`
	LANNode           *LANNodeInfo      `json:"lan_node,omitempty"`
}

// MicroDoseResult records the execution receipt of a single sub-second hardware probe.
type MicroDoseResult struct {
	Schema            string       `json:"schema"`
	Name              string       `json:"name"`
	Kind              DoseKind     `json:"kind"`
	Hardware          HardwareInfo `json:"hardware"`
	DurationMicros    int64        `json:"duration_micros"`
	DurationFormatted string       `json:"duration_formatted"`
	Operations        int64        `json:"operations"`
	BandwidthGBs      float64      `json:"bandwidth_gb_s,omitempty"`
	ComputeGFLOPS     float64      `json:"compute_gflops,omitempty"`
	OutputVerified    bool         `json:"output_verified"`
	Passed            bool         `json:"passed"`
	Detail            string       `json:"detail"`
}

// Report folds hardware discovery, micro-dose test results, and hardware-bias status.
type Report struct {
	Schema              string            `json:"schema"`
	Timestamp           time.Time         `json:"timestamp"`
	Hardware            HardwareInfo      `json:"hardware"`
	BiasTowardHardware  bool              `json:"bias_toward_hardware"`
	MicroDoses          []MicroDoseResult `json:"micro_doses"`
	TotalDurationMicros int64             `json:"total_duration_micros"`
	AllPassed           bool              `json:"all_passed"`
	Guidance            string            `json:"guidance"`
	LANNode             *LANNodeInfo      `json:"lan_node,omitempty"`
}

// InventoryReport folds dynamic local silicon discovery and LAN node status.
type InventoryReport struct {
	Schema     string       `json:"schema"` // "fak.hil.inventory.v1"
	Timestamp  time.Time    `json:"timestamp"`
	Local      HardwareInfo `json:"local"`
	LAN        LANNodeInfo  `json:"lan"`
	Sanitized  bool         `json:"sanitized"`
	NextAction string       `json:"next_action"`
}

// ComparisonArm represents one candidate or baseline in a head-to-head comparison.
type ComparisonArm struct {
	Name              string  `json:"name"`
	IsPhysicalSilicon bool    `json:"is_physical_silicon"`
	HardwareTarget    string  `json:"hardware_target"`
	EvidenceType      string  `json:"evidence_type"` // e.g., "hardware_measurement", "analytical_bound", "trace_sim"
	Metric            string  `json:"metric"`
	Value             float64 `json:"value"`
	Unit              string  `json:"unit"`
	SampleCount       int     `json:"sample_count"`
	ReceiptRef        string  `json:"receipt_ref,omitempty"`
}

// ComparisonAudit evaluates whether a reported comparison adheres to the physical hardware requirement.
type ComparisonAudit struct {
	Schema               string            `json:"schema"`
	Timestamp            time.Time         `json:"timestamp"`
	Headline             string            `json:"headline"`
	Candidate            ComparisonArm     `json:"candidate"`
	Baseline             ComparisonArm     `json:"baseline"`
	Speedup              float64           `json:"speedup"`
	Verdict              ComparisonVerdict `json:"verdict"`
	AllowedAsAchievedWin bool              `json:"allowed_as_achieved_win"`
	IsEarlyIndicator     bool              `json:"is_early_indicator"`
	Reason               string            `json:"reason"`
	EnforcementAction    string            `json:"enforcement_action"`
}
