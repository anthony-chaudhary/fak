package macobs

import (
	"context"
	"time"
)

// CommandRunner defines the execution hook for running host commands.
type CommandRunner func(ctx context.Context, name string, args ...string) ([]byte, error)

// SchemaV1 is the canonical schema identifier for macobs envelopes.
const SchemaV1 = "fak.macobs.v1"

// ActionVerdict represents a closed set of steering action recommendations for agents.
type ActionVerdict string

const (
	VerdictHeadroomOK        ActionVerdict = "HEADROOM_OK"
	VerdictReduceConcurrency ActionVerdict = "REDUCE_CONCURRENCY"
	VerdictEvictPrefixCache  ActionVerdict = "EVICT_PREFIX_CACHE"
	VerdictPressureDegrade   ActionVerdict = "PRESSURE_DEGRADE"
	VerdictThermalThrottled  ActionVerdict = "THERMAL_THROTTLED"
	VerdictSwapCritical      ActionVerdict = "SWAP_CRITICAL"
)

// Valid returns true if the verdict is in the closed set.
func (v ActionVerdict) Valid() bool {
	switch v {
	case VerdictHeadroomOK,
		VerdictReduceConcurrency,
		VerdictEvictPrefixCache,
		VerdictPressureDegrade,
		VerdictThermalThrottled,
		VerdictSwapCritical:
		return true
	default:
		return false
	}
}

// BottleneckType represents the primary runtime bottleneck identified on Apple Silicon.
type BottleneckType string

const (
	BottleneckNone            BottleneckType = "NONE"
	BottleneckMemoryCapacity  BottleneckType = "MEMORY_CAPACITY"
	BottleneckMemoryBandwidth BottleneckType = "MEMORY_BANDWIDTH"
	BottleneckComputePrefill  BottleneckType = "COMPUTE_PREFILL"
	BottleneckComputeDecode   BottleneckType = "COMPUTE_DECODE"
	BottleneckThermal         BottleneckType = "THERMAL"
	BottleneckSwap            BottleneckType = "SWAP"
	BottleneckCacheSaturation BottleneckType = "CACHE_SATURATION"
)

// Valid returns true if the bottleneck is in the closed set.
func (b BottleneckType) Valid() bool {
	switch b {
	case BottleneckNone,
		BottleneckMemoryCapacity,
		BottleneckMemoryBandwidth,
		BottleneckComputePrefill,
		BottleneckComputeDecode,
		BottleneckThermal,
		BottleneckSwap,
		BottleneckCacheSaturation:
		return true
	default:
		return false
	}
}

// ThermalState represents the Apple Silicon thermal throttling state.
type ThermalState string

const (
	ThermalNominal  ThermalState = "NOMINAL"
	ThermalFair     ThermalState = "FAIR"
	ThermalSerious  ThermalState = "SERIOUS"
	ThermalCritical ThermalState = "CRITICAL"
	ThermalUnknown  ThermalState = "UNKNOWN"
)

// Valid returns true if the thermal state is in the closed set.
func (t ThermalState) Valid() bool {
	switch t {
	case ThermalNominal, ThermalFair, ThermalSerious, ThermalCritical, ThermalUnknown:
		return true
	default:
		return false
	}
}

// PowerSource represents whether the device is on AC power or battery.
type PowerSource string

const (
	PowerAC      PowerSource = "AC"
	PowerBattery PowerSource = "BATTERY"
	PowerUnknown PowerSource = "UNKNOWN"
)

// Valid returns true if the power source is in the closed set.
func (p PowerSource) Valid() bool {
	switch p {
	case PowerAC, PowerBattery, PowerUnknown:
		return true
	default:
		return false
	}
}

// Provenance constants indicating data origin integrity.
const (
	ProvenanceWitnessed   = "WITNESSED"
	ProvenanceModeled     = "MODELED"
	ProvenanceUnavailable = "UNAVAILABLE"
)

// HardwareTelemetry holds raw and parsed hardware performance counters from macOS.
type HardwareTelemetry struct {
	TotalSystemMemoryBytes uint64       `json:"total_system_memory_bytes"`
	WiredMemoryLimitBytes  uint64       `json:"wired_memory_limit_bytes"`
	AllocSystemMemoryBytes uint64       `json:"alloc_system_memory_bytes"`
	InUseSystemMemoryBytes uint64       `json:"in_use_system_memory_bytes"`
	DeviceUtilizationPct   float64      `json:"device_utilization_pct"`
	RendererUtilizationPct float64      `json:"renderer_utilization_pct"`
	RecoveryCount          uint64       `json:"recovery_count"`
	SwapTotalBytes         uint64       `json:"swap_total_bytes"`
	SwapUsedBytes          uint64       `json:"swap_used_bytes"`
	MemoryStatusLevel      uint64       `json:"memorystatus_level"`
	CompressedBytes        uint64       `json:"compressed_bytes"`
	WiredBytes             uint64       `json:"wired_bytes"`
	FreeBytes              uint64       `json:"free_bytes"`
	PageIns                uint64       `json:"page_ins"`
	PageOuts               uint64       `json:"page_outs"`
	ThermalState           ThermalState `json:"thermal_state"`
	CPUThermalLevel        uint64       `json:"cpu_thermal_level"`
	GPUThermalLevel        uint64       `json:"gpu_thermal_level"`
	PowerSource            PowerSource  `json:"power_source"`
	BatteryLevelPct        int          `json:"battery_level_pct,omitempty"`
	Available              bool         `json:"available"`
}

// MLXServingTelemetry holds MLX serving runtime performance metrics.
type MLXServingTelemetry struct {
	ActiveRequests      int     `json:"active_requests"`
	QueuedRequests      int     `json:"queued_requests"`
	KVCacheUsagePct     float64 `json:"kv_cache_usage_pct"`
	PromptTokensPerSec  float64 `json:"prompt_tokens_per_sec"`
	DecodeTokensPerSec  float64 `json:"decode_tokens_per_sec"`
	AvgTTFTMs           float64 `json:"avg_ttft_ms"`
	AvgITLMs            float64 `json:"avg_itl_ms"`
	PrefixCacheHitRatio float64 `json:"prefix_cache_hit_ratio"`
	ServerType          string  `json:"server_type"` // e.g., "vllm-mlx", "mlx-lm", or "unknown"
	Available           bool    `json:"available"`
}

// PrefixCacheTelemetry captures prefix caching reuse and memory efficiency.
type PrefixCacheTelemetry struct {
	Hits          uint64  `json:"hits"`
	Misses        uint64  `json:"misses"`
	HitRatio      float64 `json:"hit_ratio"`
	QueriedBlocks uint64  `json:"queried_blocks"`
	CachedBlocks  uint64  `json:"cached_blocks"`
	Available     bool    `json:"available"`
}

// HeadroomTelemetry models concurrent agent capacity in unified memory.
type HeadroomTelemetry struct {
	ModelKVBytesPerToken uint64  `json:"model_kv_bytes_per_token"`
	AvailableKVPoolBytes uint64  `json:"available_kv_pool_bytes"`
	MaxSharedAgents      int     `json:"max_shared_agents"`
	MaxIsolatedAgents    int     `json:"max_isolated_agents"`
	ConcurrencyAdvantage float64 `json:"concurrency_advantage"`
	SharedPrefixTokens   uint64  `json:"shared_prefix_tokens"`
	PrivateTailTokens    uint64  `json:"private_tail_tokens"`
	ModelWeightBytes     uint64  `json:"model_weight_bytes"`
	Available            bool    `json:"available"`
}

// AnalysisReport contains synthesized diagnostics, bottlenecks, and action verdicts.
type AnalysisReport struct {
	Verdict           ActionVerdict  `json:"verdict"`
	PrimaryBottleneck BottleneckType `json:"primary_bottleneck"`
	BottleneckReason  string         `json:"bottleneck_reason"`
	RecommendedAgents int            `json:"recommended_agents"`
	Remediation       string         `json:"remediation"`
	Confidence        float64        `json:"confidence"`
}

// Snapshot is the comprehensive, canonical envelope representing all Mac & MLX observability signals.
type Snapshot struct {
	Schema      string               `json:"schema"`
	Timestamp   time.Time            `json:"timestamp"`
	Provenance  string               `json:"provenance"`
	Hardware    HardwareTelemetry    `json:"hardware"`
	MLXServing  MLXServingTelemetry  `json:"mlx_serving"`
	Headroom    HeadroomTelemetry    `json:"headroom"`
	PrefixCache PrefixCacheTelemetry `json:"prefix_cache"`
	Analysis    AnalysisReport       `json:"analysis"`
}
