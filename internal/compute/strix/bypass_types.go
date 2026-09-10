// Package strix implements high-density KV cache packing, micro-scaling quantization,
// 32MB MALL Infinity Cache attention tiling, and RDNA 3.5 cache modifier, invalidation,
// and non-temporal weight streaming bypass pipelines for AMD Strix Halo (Ryzen AI Max+ 395 / GFX1151).
package strix

import (
	"errors"
	"fmt"
	"time"
)

// Hardware streaming constants for AMD Strix Halo (GFX1151 / RDNA 3.5).
const (
	// DefaultStreamingBurstStrideBytes is the required 128-byte burst transaction stride
	// matching the 8 physical 32-bit LPDDR5X-8533 channels (8 * 16 bytes = 128 bytes).
	DefaultStreamingBurstStrideBytes = 128

	// DefaultStreamingChunkSizeBytes is the default streaming chunk size (2 MiB = 2,097,152 bytes)
	// matching the Linux DRM GEM / KFD 2MB hugepage alignment.
	DefaultStreamingChunkSizeBytes int64 = 2 * 1024 * 1024

	// DefaultWavefrontStrideBytes is the cache line span of a 32-lane Wave32 SIMD unit
	// issuing 128-bit (16-byte) loads (32 lanes * 16 bytes = 512 bytes = 4 burst strides).
	DefaultWavefrontStrideBytes int64 = 512

	// DefaultLookaheadWavefronts is the calibrated 2-wavefront lookahead distance
	// for interleaving s_prefetch_data scalar instructions ahead of vector GEMV execution.
	DefaultLookaheadWavefronts = 2

	// DefaultMaxPrefetchQueueDepth is the hardware request queue depth ceiling (32 slots)
	// preventing memory controller request queue starvation or overflow.
	DefaultMaxPrefetchQueueDepth = 32

	// PrefetchInstructionOpcode is the AMD GFX11 assembly opcode for scalar data prefetch.
	PrefetchInstructionOpcode = "s_prefetch_data"

	// MinimalMALLResidencyThreshold is the mandatory 95.0% MALL root KV residency floor.
	MinimalMALLResidencyThreshold = 95.0
)

// Typed errors for weight streaming pipeline scheduler and descriptor validation.
var (
	// ErrUnalignedWeightAddress indicates base address is not aligned to 128-byte burst boundary.
	ErrUnalignedWeightAddress = errors.New("cache: weight buffer base address must be 128-byte aligned")

	// ErrUnalignedChunkSize indicates streaming chunk size is not a multiple of 128 bytes.
	ErrUnalignedChunkSize = errors.New("cache: streaming chunk size must be a multiple of 128 bytes")

	// ErrInvalidWeightSize indicates weight buffer size <= 0.
	ErrInvalidWeightSize = errors.New("cache: weight buffer size must be greater than zero")

	// ErrStreamingPipelineClosed indicates operation attempted on closed streaming pipeline.
	ErrStreamingPipelineClosed = errors.New("cache: weight streaming bypass pipeline is closed")

	// ErrInvalidLookahead indicates non-positive lookahead wavefront distance.
	ErrInvalidLookahead = errors.New("cache: lookahead wavefronts must be greater than zero")

	// ErrDescriptorNonBypassViolation indicates descriptor auditor found non-bypass flags on weights.
	ErrDescriptorNonBypassViolation = errors.New("cache: descriptor auditor found non-bypass SLC=0 on streaming weight operand")
)

// StreamingAblationArm designates the validation and execution regime for weight streaming.
type StreamingAblationArm string

const (
	// Arm1NonTemporalWithPrefetch represents the production target: full non-temporal weight streaming
	// (SLC=1) interleaved with s_prefetch_data 2-wavefront lookahead.
	Arm1NonTemporalWithPrefetch StreamingAblationArm = "ARM1_NON_TEMPORAL_PREFETCH"

	// Arm2NonTemporalNoPrefetch represents non-temporal weight streaming (SLC=1) relying solely
	// on demand-driven synchronous reads without s_prefetch_data lookahead instructions.
	Arm2NonTemporalNoPrefetch StreamingAblationArm = "ARM2_NON_TEMPORAL_NO_PREFETCH"

	// Arm3CachedWeightsBaseline represents the unmanaged baseline where weight loads are issued with
	// SLC=0, polluting and thrashing the 32MB on-die MALL Infinity Cache.
	Arm3CachedWeightsBaseline StreamingAblationArm = "ARM3_CACHED_WEIGHTS_BASELINE"
)

// StreamingBypassConfig defines the configuration parameters for the weight streaming scheduler.
type StreamingBypassConfig struct {
	// TargetArch identifies the GPU architecture (default: "gfx1151").
	TargetArch string `json:"target_arch"`

	// BurstStrideBytes is the burst transaction stride (default: 128).
	BurstStrideBytes int `json:"burst_stride_bytes"`

	// ChunkSizeBytes is the streaming chunk partitioning size (default: 2 MiB).
	ChunkSizeBytes int64 `json:"chunk_size_bytes"`

	// LookaheadWavefronts is the prefetch lookahead distance in wavefront units (default: 2).
	LookaheadWavefronts int `json:"lookahead_wavefronts"`

	// MaxPrefetchQueueDepth is the maximum allowable outstanding prefetch slots (default: 32).
	MaxPrefetchQueueDepth int `json:"max_prefetch_queue_depth"`

	// EnablePrefetch controls whether s_prefetch_data instructions are emitted (default: true).
	EnablePrefetch bool `json:"enable_prefetch"`

	// FallbackOnBackpressure enables automatic throttle / synchronous fallback when queue overflows.
	FallbackOnBackpressure bool `json:"fallback_on_backpressure"`

	// AblationArm selects the active validation arm (default: Arm1NonTemporalWithPrefetch).
	AblationArm StreamingAblationArm `json:"ablation_arm"`
}

// DefaultStreamingBypassConfig returns a production-calibrated configuration for AMD Strix Halo.
func DefaultStreamingBypassConfig() StreamingBypassConfig {
	return StreamingBypassConfig{
		TargetArch:             TargetArchGFX1151,
		BurstStrideBytes:       DefaultStreamingBurstStrideBytes,
		ChunkSizeBytes:         DefaultStreamingChunkSizeBytes,
		LookaheadWavefronts:    DefaultLookaheadWavefronts,
		MaxPrefetchQueueDepth:  DefaultMaxPrefetchQueueDepth,
		EnablePrefetch:         true,
		FallbackOnBackpressure: true,
		AblationArm:            Arm1NonTemporalWithPrefetch,
	}
}

// Validate ensures configuration fields conform to Strix Halo hardware constraints.
func (c *StreamingBypassConfig) Validate() error {
	if c.BurstStrideBytes <= 0 || c.BurstStrideBytes%DefaultStreamingBurstStrideBytes != 0 {
		return ErrUnalignedStride
	}
	if c.ChunkSizeBytes <= 0 || c.ChunkSizeBytes%int64(c.BurstStrideBytes) != 0 {
		return ErrUnalignedChunkSize
	}
	if c.LookaheadWavefronts <= 0 {
		return ErrInvalidLookahead
	}
	if c.MaxPrefetchQueueDepth <= 0 {
		c.MaxPrefetchQueueDepth = DefaultMaxPrefetchQueueDepth
	}
	if c.AblationArm == "" {
		c.AblationArm = Arm1NonTemporalWithPrefetch
	}
	return nil
}

// StreamingChunk represents a single scheduled weight streaming chunk with memory bounds,
// instruction modifier bitmasks, and optional prefetch lookahead.
type StreamingChunk struct {
	// ChunkIndex is the 0-based sequence number of the chunk.
	ChunkIndex int `json:"chunk_index"`

	// BaseAddress is the 128-byte aligned virtual memory address of the chunk.
	BaseAddress uint64 `json:"base_address"`

	// SizeBytes is the length of the chunk in bytes (128-byte aligned).
	SizeBytes int64 `json:"size_bytes"`

	// Modifier is the synthesized RDNA 3.5 cache modifier bitmask.
	Modifier InstructionModifierBitmask `json:"modifier"`

	// PrefetchAddress is the lookahead target address for s_prefetch_data.
	PrefetchAddress uint64 `json:"prefetch_address"`

	// PrefetchSizeBytes is the length of the prefetch target span.
	PrefetchSizeBytes int64 `json:"prefetch_size_bytes"`

	// PrefetchIssued indicates whether an s_prefetch_data instruction was emitted for this chunk.
	PrefetchIssued bool `json:"prefetch_issued"`

	// AssemblyTokens is the formatted disassembly of instructions for this chunk.
	AssemblyTokens []string `json:"assembly_tokens"`
}

// StreamingDescriptor represents an AMD GFX11 128-bit buffer resource descriptor (V#)
// for streaming weight loads.
type StreamingDescriptor struct {
	Word0 uint32 `json:"word0"` // Base address [31:0]
	Word1 uint32 `json:"word1"` // Base address [47:32] | flags
	Word2 uint32 `json:"word2"` // Number of records / size
	Word3 uint32 `json:"word3"` // Cache control flags (SLC/GLC/DLC)
}

// IsBypass returns true if Word3 has the SLC bit set (bypassing MALL Infinity Cache).
func (d StreamingDescriptor) IsBypass() bool {
	return (d.Word3 & DescriptorWord3SLCBit) != 0
}

// StreamingBypassTelemetry captures performance metrics, queue statistics,
// and cache residency during weight streaming passes.
type StreamingBypassTelemetry struct {
	TotalBytesScheduled        int64     `json:"total_bytes_scheduled"`
	TotalChunksScheduled       int64     `json:"total_chunks_scheduled"`
	TotalPrefetchIssued        int64     `json:"total_prefetch_issued"`
	PrefetchStallsDetected     int64     `json:"prefetch_stalls_detected"`
	BackpressureEvents         int64     `json:"backpressure_events"`
	FallbackSynchronousEmitted int64     `json:"fallback_synchronous_emitted"`
	SustainedDRAMBandwidthGBps float64   `json:"sustained_dram_bandwidth_gbps"`
	MALLRootResidencyPercent   float64   `json:"mall_root_residency_percent"`
	LastTimestamp              time.Time `json:"last_timestamp"`
}

// String returns a compact representation of the streaming telemetry.
func (t StreamingBypassTelemetry) String() string {
	return fmt.Sprintf(
		"Bytes: %d MB | Chunks: %d | Prefetches: %d | Stalls: %d | Backpressure: %d | MALL Residency: %.1f%% | BW: %.1f GB/s",
		t.TotalBytesScheduled/(1024*1024),
		t.TotalChunksScheduled,
		t.TotalPrefetchIssued,
		t.PrefetchStallsDetected,
		t.BackpressureEvents,
		t.MALLRootResidencyPercent,
		t.SustainedDRAMBandwidthGBps,
	)
}
