// Package strix implements the hardware profile, memory bandwidth physics,
// compute HAL, and unified memory pipeline for AMD Strix Halo (Ryzen AI Max+ 395).
package strix

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// Typed errors for Strix Halo unified memory, TTM eviction, and lifecycle management.
var (
	// ErrTTMPageEvictionDetected is returned when kernel TTM evicts GTT pages, breaking zero-copy pointer identity.
	ErrTTMPageEvictionDetected = errors.New("strix: TTM page eviction detected: zero-copy pointer identity broken under memory pressure")

	// ErrZeroCopyPointerMismatch indicates that host and device virtual pointers diverge.
	ErrZeroCopyPointerMismatch = errors.New("strix: zero-copy pointer mismatch: host and device pointers diverge")

	// ErrStationaryBlockEvicted indicates that a stationary KV block was evicted during tool wait.
	ErrStationaryBlockEvicted = errors.New("strix: stationary KV block evicted during yielded I/O")

	// ErrHandoffNotFound indicates that a requested session handoff record does not exist.
	ErrHandoffNotFound = errors.New("strix/handoff: session handoff record not found")

	// ErrInvalidHandoffState indicates an illegal state transition during yield/resume handoff.
	ErrInvalidHandoffState = errors.New("strix/handoff: invalid session state transition")
)

// Memory constants for Strix Halo unified silicon (Line 28 seam).
const (
	// DefaultAIMDGTTTotalBytes is the 112 GiB GTT memory aperture boundary on Strix Halo 128GB APUs.
	DefaultAIMDGTTTotalBytes uint64 = 112 * 1024 * 1024 * 1024 // 112 GiB = 120,259,084,288 bytes
)

var (

	// ErrHandoffTTMWatchdog indicates that a tool wait exceeded the hard watchdog TTL in YIELDED_IO.
	ErrHandoffTTMWatchdog = errors.New("strix/handoff: tool wait TTL expired in YIELDED_IO")

	// ErrTTMEvictionPressure indicates that system memory pressure exceeds safe GTT allocation thresholds.
	ErrTTMEvictionPressure = errors.New("strix: kernel TTM eviction pressure threshold exceeded")

	// ErrInvalidGTTSize indicates an invalid size passed to GTT allocation.
	ErrInvalidGTTSize = errors.New("strix/gtt: allocation size must be positive and within GTT pool limit")

	// ErrGTTBufferClosed indicates an operation was attempted on a closed GTT buffer.
	ErrGTTBufferClosed = errors.New("strix/gtt: GTT buffer is closed")

	// ErrUnalignedKVMemory indicates that KV cache memory is not aligned to the 128-byte LPDDR5X burst boundary.
	ErrUnalignedKVMemory = errors.New("strix/gqa: KV memory buffer is not aligned to 128-byte LPDDR5X burst boundary")

	// ErrInvalidGQAGeometry indicates an unsupported or mismatched GQA head configuration.
	ErrInvalidGQAGeometry = errors.New("strix/gqa: invalid GQA geometry: query heads must be divisible by KV heads")

	// ErrUnalignedWeightStride indicates weight stride is not aligned to 128-byte LPDDR5X burst boundary.
	ErrUnalignedWeightStride = errors.New("strix/streamer: weight stride must be aligned to 128-byte boundary")

	// ErrPrefetchQueueOverflow indicates prefetch queue depth exceeded maximum capacity.
	ErrPrefetchQueueOverflow = errors.New("strix/prefetch: prefetch queue depth exceeded maximum capacity")

	// ErrPrefetchPipelineClosed indicates operation was attempted on a closed prefetch pipeline.
	ErrPrefetchPipelineClosed = errors.New("strix/prefetch: prefetch pipeline is closed")

	// ErrWeightStreamOutOfBounds indicates access beyond allocated layer or weight dimensions.
	ErrWeightStreamOutOfBounds = errors.New("strix/streamer: weight buffer access out of bounds")

	// ErrZeroWeightData indicates weight data is empty or nil.
	ErrZeroWeightData = errors.New("strix/streamer: weight layer data is empty or nil")

	// ErrQoSTimeout indicates CPU I/O latency target was exceeded under bus contention.
	ErrQoSTimeout = errors.New("strix/qos: CPU I/O wait timed out under memory bus contention")
)

// ContractState represents the 2D batch scheduling lifecycle state of an agent.
type ContractState string

const (
	// StateExecuting indicates the agent occupies an active GPU decode wave slot.
	StateExecuting ContractState = "STATE_EXECUTING"

	// StateYieldedIO indicates the agent yielded GPU compute while holding stationary KV memory in DRAM.
	StateYieldedIO ContractState = "STATE_YIELDED_IO"

	// StateResuming indicates the agent is validating invariants and re-entering the decode wave.
	StateResuming ContractState = "STATE_RESUMING"
)

// ActiveKVBlock represents an active KV cache block in Strix Halo unified DRAM.
type ActiveKVBlock struct {
	BlockID    int64     `json:"block_id"`
	SessionID  string    `json:"session_id"`
	HostPtr    uintptr   `json:"host_ptr"`
	DevicePtr  uintptr   `json:"device_ptr"`
	SizeBytes  int64     `json:"size_bytes"`
	Stationary bool      `json:"stationary"`
	PinCount   int32     `json:"pin_count"`
	LastCheck  time.Time `json:"last_check,omitempty"`
}

// IsZeroCopy returns true if both host and device pointers are non-zero and identical.
func (b *ActiveKVBlock) IsZeroCopy() bool {
	return b != nil && b.HostPtr != 0 && b.HostPtr == b.DevicePtr
}

// GQAPackedBlock represents a 128-byte aligned KV token block tile matching
// LPDDR5X burst alignment (8 channels * 16 bytes) for RDNA 3.5 cache line coalescing.
type GQAPackedBlock struct {
	BlockID      int64   `json:"block_id"`
	LayerIdx     int     `json:"layer_idx"`
	HeadGroupIdx int     `json:"head_group_idx"`
	TokenStart   int     `json:"token_start"`
	NumTokens    int     `json:"num_tokens"`
	Data         []byte  `json:"-"`
	HostPtr      uintptr `json:"host_ptr"`
	DevicePtr    uintptr `json:"device_ptr"`
}

// IsAligned128 returns true if HostPtr is 128-byte aligned and DevicePtr
// is also 128-byte aligned when non-zero.
func (b *GQAPackedBlock) IsAligned128() bool {
	if b == nil {
		return false
	}
	if b.HostPtr == 0 || b.HostPtr%128 != 0 {
		return false
	}
	if b.DevicePtr != 0 && b.DevicePtr%128 != 0 {
		return false
	}
	return true
}

// IsZeroCopy returns true if both host and device pointers are non-zero and identical.
func (b *GQAPackedBlock) IsZeroCopy() bool {
	return b != nil && b.HostPtr != 0 && b.HostPtr == b.DevicePtr
}

// AQLDispatchPacket represents an architected AQL packet dispatched to RDNA 3.5 compute queues.
type AQLDispatchPacket struct {
	Header           uint16    `json:"header"`
	Dimensions       uint16    `json:"dimensions"`
	WorkgroupSizeX   uint16    `json:"workgroup_size_x"`
	WorkgroupSizeY   uint16    `json:"workgroup_size_y"`
	WorkgroupSizeZ   uint16    `json:"workgroup_size_z"`
	GridSizeX        uint32    `json:"grid_size_x"`
	GridSizeY        uint32    `json:"grid_size_y"`
	GridSizeZ        uint32    `json:"grid_size_z"`
	KernelObject     uint64    `json:"kernel_object"`
	KernargAddress   uint64    `json:"kernarg_address"`
	CompletionSignal uint64    `json:"completion_signal"`
	DispatchedAt     time.Time `json:"dispatched_at"`
}

// TTMReconciliationResult captures the action taken when memory pressure is detected.
type TTMReconciliationResult struct {
	PressureDetected bool   `json:"pressure_detected"`
	Action           string `json:"action"` // "offload_moe_layers", "throttle", "nominal"
	FreedBytes       uint64 `json:"freed_bytes"`
	OffloadedLayers  int    `json:"offloaded_layers"`
	Throttled        bool   `json:"throttled"`
	EvictionsSeen    uint64 `json:"evictions_seen"`
	Message          string `json:"message"`
}

// Hardware constants for Strix Halo LPDDR5X-8533 weight streaming and prefetch pipeline.
const (
	// StrixHaloLPDDR5XChannels is the number of 32-bit physical memory channels on AMD Strix Halo.
	StrixHaloLPDDR5XChannels = 8

	// StrixHaloBurstStrideBytes is the burst transaction stride across 8 channels (8 x 16 bytes = 128 bytes).
	StrixHaloBurstStrideBytes = 128

	// StrixHaloTargetBandwidthGBps is 80% saturation threshold of LPDDR5X-8533 peak (218.44 GB/s).
	StrixHaloTargetBandwidthGBps float64 = 218.44

	// DefaultStrixDRMDevicePath is the standard Linux sysfs path to AMDGPU device metrics.
	DefaultStrixDRMDevicePath = "/sys/class/drm/card0/device"
)

// WeightStride represents a 128-byte aligned burst transaction stride striped across
// the 8 LPDDR5X-8533 memory channels.
type WeightStride struct {
	StrideIndex int   `json:"stride_index"`
	Offset      int64 `json:"offset"`
	SizeBytes   int   `json:"size_bytes"`
	ChannelIdx  int   `json:"channel_idx"` // 0..7
}

// MemoryBusQoS encapsulates Infinity Fabric and DRAM bus QoS arbitration state.
type MemoryBusQoS struct {
	Level                string  `json:"level"` // "OPTIMAL", "DEGRADED", "CRITICAL"
	BusUtilizationPct    float64 `json:"bus_utilization_pct"`
	CPUPriorityActive    bool    `json:"cpu_priority_active"`
	CPULatencyTargetMs   float64 `json:"cpu_latency_target_ms"` // Target < 5.0 ms
	ObservedCPULatencyMs float64 `json:"observed_cpu_latency_ms"`
	GPUThrottled         bool    `json:"gpu_throttled"`
	ThrottleFactor       float64 `json:"throttle_factor"` // 1.0 = normal, < 1.0 = throttled
	ContentionEvents     int64   `json:"contention_events"`
}

// WeightStreamMetrics captures sustained throughput and cache invariants for weight scanning.
type WeightStreamMetrics struct {
	LayersStreamed   int           `json:"layers_streamed"`
	BytesStreamed    int64         `json:"bytes_streamed"`
	Duration         time.Duration `json:"duration"`
	ThroughputGBs    float64       `json:"throughput_gbs"`
	DRAMReadBytes    uint64        `json:"dram_read_bytes"`
	MCBusyPercent    float64       `json:"mc_busy_percent"`
	MALLEvictions    uint64        `json:"mall_evictions"`
	NonTemporalLoads int64         `json:"non_temporal_loads"`
	StrideCount      int64         `json:"stride_count"`
	ChannelBalance   []float64     `json:"channel_balance"` // Channel utilization fraction [0..7]
	QoS              MemoryBusQoS  `json:"qos"`
	Timestamp        time.Time     `json:"timestamp"`
}

// PrefetchRequest represents an asynchronous software prefetch directive (s_prefetch_data).
type PrefetchRequest struct {
	RequestID   string          `json:"request_id"`
	LayerIdx    int             `json:"layer_idx"`
	Offset      int64           `json:"offset"`
	SizeBytes   int64           `json:"size_bytes"`
	BurstStride int             `json:"burst_stride"`
	CacheHint   CachePolicyHint `json:"cache_hint"`
	Priority    int             `json:"priority"` // 0 = normal GPU, 1 = high, 2 = CPU priority
	SubmittedAt time.Time       `json:"submitted_at"`
	CompletedAt time.Time       `json:"completed_at"`
	Done        chan struct{}   `json:"-"`
	Error       error           `json:"error,omitempty"`
	doneOnce    sync.Once       `json:"-"`
}

// Complete finishes the prefetch request safely, closing Done channel exactly once.
func (r *PrefetchRequest) Complete(err error) {
	r.doneOnce.Do(func() {
		if err != nil {
			r.Error = err
		}
		r.CompletedAt = time.Now()
		if r.Done != nil {
			close(r.Done)
		}
	})
}

// MALLCacheBudgetTracker tracks the 32 MiB MALL Infinity Cache capacity, pinned root KV prefix,
// and confirms zero cache line evictions under non-temporal weight streaming bypass.
type MALLCacheBudgetTracker struct {
	mu             sync.RWMutex
	capacityBytes  int64
	pinnedKVBytes  int64
	pinnedKVTokens int
	evictions      uint64
	weightLoads    uint64
	weightBytes    uint64
	bypassedBytes  uint64
}

// NewMALLCacheBudgetTracker creates a tracker with specified MALL cache capacity (default 32 MiB).
func NewMALLCacheBudgetTracker(capacityBytes int64) *MALLCacheBudgetTracker {
	if capacityBytes <= 0 {
		capacityBytes = StrixHaloMALLSizeBytes
	}
	return &MALLCacheBudgetTracker{
		capacityBytes: capacityBytes,
	}
}

// PinKVTokens pins KV cache tokens (e.g. 8,192 tokens = 32 MiB) into the MALL budget.
func (b *MALLCacheBudgetTracker) PinKVTokens(tokens int) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if tokens < 0 {
		return errors.New("strix/mall: tokens cannot be negative")
	}
	bytes := int64(tokens) * StrixHaloBytesPerToken
	if bytes > b.capacityBytes {
		return fmt.Errorf("strix/mall: pinned KV tokens %d (%d bytes) exceeds MALL capacity %d bytes",
			tokens, bytes, b.capacityBytes)
	}
	b.pinnedKVTokens = tokens
	b.pinnedKVBytes = bytes
	return nil
}

// PinKVBytes pins a raw byte footprint into the MALL budget.
func (b *MALLCacheBudgetTracker) PinKVBytes(bytes int64) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if bytes < 0 {
		return errors.New("strix/mall: bytes cannot be negative")
	}
	if bytes > b.capacityBytes {
		return fmt.Errorf("strix/mall: pinned KV footprint %d exceeds MALL capacity %d bytes",
			bytes, b.capacityBytes)
	}
	b.pinnedKVBytes = bytes
	b.pinnedKVTokens = int(bytes / StrixHaloBytesPerToken)
	return nil
}

// RecordWeightAccess updates MALL budget metrics when weight bytes are streamed.
// Non-temporal bypass (NT=1, SLC=1) routes weights directly to vector registers, yielding 0 evictions.
// Cached/temporal accesses that exceed remaining headroom cause MALL line evictions.
func (b *MALLCacheBudgetTracker) RecordWeightAccess(bytes int64, hint CachePolicyHint) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.weightLoads++
	b.weightBytes += uint64(bytes)

	if hint.Bypass || (hint.NT == 1 && hint.SLC == 1) {
		b.bypassedBytes += uint64(bytes)
		// Non-temporal bypass routes around MALL: zero evictions
		return
	}

	// Temporal load: weight data attempts to allocate into MALL cache.
	excess := (b.pinnedKVBytes + bytes) - b.capacityBytes
	if excess > 0 {
		// Evict blocks to make room
		blocks := (excess + StrixHaloBlockSizeBytes - 1) / StrixHaloBlockSizeBytes
		b.evictions += uint64(blocks)
	}
}

// EvictionCount returns total MALL cache line/block evictions observed.
func (b *MALLCacheBudgetTracker) EvictionCount() uint64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.evictions
}

// EvictionRate returns eviction ratio over weight load operations.
func (b *MALLCacheBudgetTracker) EvictionRate() float64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.weightLoads == 0 {
		return 0.0
	}
	return float64(b.evictions) / float64(b.weightLoads)
}

// PinnedKVIntact returns true iff zero evictions occurred, confirming hot KV prefix residency.
func (b *MALLCacheBudgetTracker) PinnedKVIntact() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.evictions == 0
}

// HeadroomBytes returns unallocated MALL cache headroom in bytes.
func (b *MALLCacheBudgetTracker) HeadroomBytes() int64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	h := b.capacityBytes - b.pinnedKVBytes
	if h < 0 {
		return 0
	}
	return h
}

// PinnedKVBytes returns total bytes of KV cache pinned in MALL.
func (b *MALLCacheBudgetTracker) PinnedKVBytes() int64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.pinnedKVBytes
}

// PinnedKVTokens returns total count of KV tokens pinned in MALL.
func (b *MALLCacheBudgetTracker) PinnedKVTokens() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.pinnedKVTokens
}

// BypassedBytes returns total weight bytes that bypassed MALL via non-temporal directives.
func (b *MALLCacheBudgetTracker) BypassedBytes() uint64 {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.bypassedBytes
}

// Reset clears recorded accesses and evictions while preserving capacity and pinned KV.
func (b *MALLCacheBudgetTracker) Reset() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.evictions = 0
	b.weightLoads = 0
	b.weightBytes = 0
	b.bypassedBytes = 0
}
