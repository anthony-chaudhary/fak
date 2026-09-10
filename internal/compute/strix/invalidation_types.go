// Package strix implements high-density KV cache packing, micro-scaling quantization,
// 32MB MALL Infinity Cache attention tiling, and RDNA 3.5 cache modifier and invalidation protocols
// for AMD Strix Halo (Ryzen AI Max+ 395 / GFX1151).
package strix

import (
	"errors"
	"time"
)

// Invalidation instruction opcodes for RDNA 3.5 (GFX1151).
const (
	// OpcodeBufferWBL2 is the RDNA 3.5 buffer writeback and invalidate L2/MALL instruction.
	OpcodeBufferWBL2 = "buffer_wbl2"

	// OpcodeBufferGL0Inv is the instruction to invalidate Vector L0 cache.
	OpcodeBufferGL0Inv = "buffer_gl0_inv"

	// OpcodeBufferGL1Inv is the instruction to invalidate Vector L1 cache.
	OpcodeBufferGL1Inv = "buffer_gl1_inv"
)

// MUBUF / Buffer instruction binary encodings for invalidation packets.
const (
	// MUBUFOpcodeBufferWBL2 is the 8-bit opcode field [25:18] for buffer_wbl2 in GFX11 MUBUF encoding.
	MUBUFOpcodeBufferWBL2 uint32 = 0x2A << 18

	// MUBUFOpcodeBufferGL0Inv is the opcode field [25:18] for buffer_gl0_inv in GFX11.
	MUBUFOpcodeBufferGL0Inv uint32 = 0x2B << 18

	// MUBUFOpcodeBufferGL1Inv is the opcode field [25:18] for buffer_gl1_inv in GFX11.
	MUBUFOpcodeBufferGL1Inv uint32 = 0x2C << 18
)

// TenantTransitionState models the state machine lifecycle during cross-tenant context switches.
type TenantTransitionState string

const (
	// StateIdle indicates the runtime is executing within a stable tenant context.
	StateIdle TenantTransitionState = "STATE_IDLE"

	// StateSwitching indicates a context switch has been initiated.
	StateSwitching TenantTransitionState = "STATE_SWITCHING"

	// StateFingerprinting indicates shared prefix and private KV segments are being classified.
	StateFingerprinting TenantTransitionState = "STATE_FINGERPRINTING"

	// StateInvalidating indicates selective or global buffer invalidation packets are in flight.
	StateInvalidating TenantTransitionState = "STATE_INVALIDATING"

	// StateReTiling indicates the 32MB MALL partition map is being updated for the incoming tenant.
	StateReTiling TenantTransitionState = "STATE_RE_TILING"

	// StateActive indicates the transition has completed and the new tenant is verified active.
	StateActive TenantTransitionState = "STATE_ACTIVE"
)

// InvalidationAblationArm identifies the experimental or operational invalidation policy arm.
type InvalidationAblationArm string

const (
	// Arm1SelectiveInvalidation executes prefix-fingerprinted selective invalidation using buffer_wbl2
	// targeting only tenant-private cache lines, preserving shared root prefix blocks in MALL.
	Arm1SelectiveInvalidation InvalidationAblationArm = "ARM1_SELECTIVE_INVALIDATION"

	// Arm2GlobalCoarseFlush executes a complete L2/MALL writeback and purge (buffer_gl0_inv + global flush),
	// wiping all resident cache lines including shared root prefixes.
	Arm2GlobalCoarseFlush InvalidationAblationArm = "ARM2_GLOBAL_COARSE_FLUSH"

	// Arm3LazyLRUBaseline executes zero explicit invalidations, relying strictly on passive hardware LRU write-over.
	Arm3LazyLRUBaseline InvalidationAblationArm = "ARM3_LAZY_LRU_BASELINE"
)

// TenantKVSegment defines the memory boundaries and metadata for a tenant's attention KV cache.
type TenantKVSegment struct {
	TenantID          string `json:"tenant_id"`
	PrefixFingerprint string `json:"prefix_fingerprint"`
	PrefixBaseAddr    uint64 `json:"prefix_base_addr"`
	PrefixSizeBytes   int64  `json:"prefix_size_bytes"`
	PrivateBaseAddr   uint64 `json:"private_base_addr"`
	PrivateSizeBytes  int64  `json:"private_size_bytes"`
	GenerationEpoch   uint64 `json:"generation_epoch"`
}

// TotalBytes returns the sum of prefix and private KV bytes.
func (t TenantKVSegment) TotalBytes() int64 {
	return t.PrefixSizeBytes + t.PrivateSizeBytes
}

// BufferWBL2Packet represents a synthesized user-space RDNA 3.5 buffer_wbl2 invalidation command.
type BufferWBL2Packet struct {
	Opcode               string `json:"opcode"`
	BaseAddress          uint64 `json:"base_address"`
	RangeBytes           int64  `json:"range_bytes"`
	CacheLineCount       int    `json:"cache_line_count"`
	GLC                  int    `json:"glc"`
	SLC                  int    `json:"slc"`
	RawInstructionDword0 uint32 `json:"raw_instruction_dword0"`
	RawInstructionDword1 uint32 `json:"raw_instruction_dword1"`
	FenceConfirmed       bool   `json:"fence_confirmed"`
}

// TransitionRequest holds inputs required to safely transition the 32MB MALL from one tenant to another.
type TransitionRequest struct {
	PreviousTenant TenantKVSegment         `json:"previous_tenant"`
	IncomingTenant TenantKVSegment         `json:"incoming_tenant"`
	AblationArm    InvalidationAblationArm `json:"ablation_arm"`
	ForceFallback  bool                    `json:"force_fallback"`
	TimeoutNs      int64                   `json:"timeout_ns"`
}

// TransitionReceipt records the authoritative outcome and latency metrics of an invalidation transition.
type TransitionReceipt struct {
	Schema                  string                  `json:"schema"`
	PreviousTenantID        string                  `json:"previous_tenant_id"`
	IncomingTenantID        string                  `json:"incoming_tenant_id"`
	PrefixFingerprint       string                  `json:"prefix_fingerprint"`
	PrefixPreserved         bool                    `json:"prefix_preserved"`
	PreservedPrefixBytes    int64                   `json:"preserved_prefix_bytes"`
	InvalidatedPrivateBytes int64                   `json:"invalidated_private_bytes"`
	InvalidatedLineCount    int                     `json:"invalidated_line_count"`
	PacketsEmitted          int                     `json:"packets_emitted"`
	DurationNs              int64                   `json:"duration_ns"`
	DurationMicroseconds    float64                 `json:"duration_us"`
	Sub50usMet              bool                    `json:"sub_50us_met"`
	AblationArm             InvalidationAblationArm `json:"ablation_arm"`
	FallbackTriggered       bool                    `json:"fallback_triggered"`
	FallbackReason          string                  `json:"fallback_reason,omitempty"`
	NewEpoch                uint64                  `json:"new_epoch"`
	Timestamp               time.Time               `json:"timestamp"`
}

// DynamicReTilerConfig sets the operational parameters for the dynamic re-tiler and invalidation engine.
type DynamicReTilerConfig struct {
	TargetArch               string                  `json:"target_arch"`
	DefaultArm               InvalidationAblationArm `json:"default_arm"`
	MaxTransitionLatencyNs   int64                   `json:"max_transition_latency_ns"`
	EnableQuarantineFallback bool                    `json:"enable_quarantine_fallback"`
	StrictFenceVerification  bool                    `json:"strict_fence_verification"`
}

// DefaultDynamicReTilerConfig returns standard configuration for AMD Strix Halo (Ryzen AI Max+ 395 / GFX1151).
func DefaultDynamicReTilerConfig() DynamicReTilerConfig {
	return DynamicReTilerConfig{
		TargetArch:               TargetArchGFX1151,
		DefaultArm:               Arm1SelectiveInvalidation,
		MaxTransitionLatencyNs:   50000, // 50 microseconds
		EnableQuarantineFallback: true,
		StrictFenceVerification:  true,
	}
}

// InvalidationTelemetry aggregates metrics across all executed multi-tenant transitions.
type InvalidationTelemetry struct {
	TotalTransitions          uint64    `json:"total_transitions"`
	SelectiveTransitions      uint64    `json:"selective_transitions"`
	GlobalFlushes             uint64    `json:"global_flushes"`
	QuarantinedFallbacks      uint64    `json:"quarantined_fallbacks"`
	LazyLRUBaselines          uint64    `json:"lazy_lru_baselines"`
	PreservedPrefixLines      uint64    `json:"preserved_prefix_lines"`
	EvictedPrivateLines       uint64    `json:"evicted_private_lines"`
	TotalDurationNs           uint64    `json:"total_duration_ns"`
	MaxDurationNs             uint64    `json:"max_duration_ns"`
	StaleHitsPrevented        uint64    `json:"stale_hits_prevented"`
	CrossTenantLeaksPrevented uint64    `json:"cross_tenant_leaks_prevented"`
	LastTransition            time.Time `json:"last_transition"`
}

// AverageDurationNs computes the average transition time in nanoseconds.
func (t InvalidationTelemetry) AverageDurationNs() uint64 {
	if t.TotalTransitions == 0 {
		return 0
	}
	return t.TotalDurationNs / t.TotalTransitions
}

// AverageDurationUs computes the average transition time in microseconds.
func (t InvalidationTelemetry) AverageDurationUs() float64 {
	return float64(t.AverageDurationNs()) / 1000.0
}

// Typed sentinel errors for the invalidation protocol and dynamic re-tiler.
var (
	// ErrInvalidTransitionRequest is returned when transition inputs are malformed or missing tenant identities.
	ErrInvalidTransitionRequest = errors.New("strix/cache: invalid transition request: tenant IDs and valid addresses required")

	// ErrFenceVerificationFailed is returned when hardware semaphore fences fail to confirm cache line purge.
	ErrFenceVerificationFailed = errors.New("strix/cache: hardware semaphore fence failed to confirm buffer_wbl2 invalidation")

	// ErrCrossTenantLeakDetected is returned when an access would read un-invalidated lines from another tenant.
	ErrCrossTenantLeakDetected = errors.New("strix/cache: cross-tenant context leak detected: attempted access to stale tenant lines")

	// ErrReTilerClosed is returned when operations are attempted on a terminated re-tiler instance.
	ErrReTilerClosed = errors.New("strix/cache: dynamic re-tiler is closed")

	// ErrTargetArchMismatch is returned when target architecture is not confirmed as GFX1151.
	ErrTargetArchMismatch = errors.New("strix/cache: invalid architecture target for MALL invalidation: expected gfx1151")
)
