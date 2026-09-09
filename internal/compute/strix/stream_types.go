// Package strix implements the AMD Strix Halo (Ryzen AI Max+ 395 / GFX1151)
// non-temporal weight streaming pipeline, RDNA 3.5 V# buffer descriptor synthesis,
// and 32MB MALL (Memory Attached Last-Level) Infinity Cache eviction protection.
package strix

import (
	"errors"
	"fmt"
	"time"
)

// Hardware constants for AMD Strix Halo (Ryzen AI Max+ 395 / GFX1151).
const (
	// MALLSetCount is the number of sets in the 32MB MALL cache (2^15 = 32,768).
	MALLSetCount = 32768

	// MALLAssociativityWays is the 16-way set-associativity of the MALL cache.
	MALLAssociativityWays = 16

	// WeightStreamThresholdBytes is the minimum tensor size (10 MiB) for streaming bypass qualification.
	WeightStreamThresholdBytes int64 = 10 * 1024 * 1024

	// IntraMALLBandwidthGBs is the internal SRAM crossbar interconnect bandwidth (> 1.2 TB/s).
	IntraMALLBandwidthGBs float64 = 1200.0

	// TargetStreamingBWGBs is the target sustained streaming bandwidth for weight scanning (>= 220.0 GB/s).
	TargetStreamingBWGBs float64 = 220.0

	// BurstStrideBytes is the burst transaction stride across the 8 physical memory channels (8 * 16 = 128 bytes).
	BurstStrideBytes = 128

	// MemoryChannels is the number of 32-bit physical memory channels on Strix Halo.
	MemoryChannels = 8

	// Default35BWeightBytes represents the ~17.5 GB weight footprint of a 35B model quantized to 4-bit.
	Default35BWeightBytes int64 = 17500 * 1024 * 1024

	// Default35BEvictionRatio represents the mathematical eviction frequency per token forward pass
	// when 17.5 GB of weights stream through an unmanaged 32MB MALL cache (17,500 MB / 33.55 MB = 521.5x).
	Default35BEvictionRatio float64 = 521.5
)

// Typed errors for buffer descriptor synthesis, memory alignment, and stream pipeline validation.
var (
	// ErrInvalidBaseAddress indicates a virtual address exceeding the 48-bit address space limit.
	ErrInvalidBaseAddress = errors.New("cache: base address exceeds 48-bit virtual address limit (0x0000FFFFFFFFFFFF)")

	// ErrUnalignedBaseAddress indicates a base address not aligned to 16 bytes for RDNA 3.5 buffer descriptors.
	ErrUnalignedBaseAddress = errors.New("cache: base address must be 16-byte aligned for RDNA 3.5 buffer descriptors")

	// ErrInvalidTensorKind indicates an unknown or unhandled tensor kind.
	ErrInvalidTensorKind = errors.New("cache: invalid or unspecified tensor kind")

	// ErrUnalignedStride indicates a burst stride not matching the 128-byte LPDDR5X channel boundary.
	ErrUnalignedStride = errors.New("cache: burst stride must be a positive multiple of 128 bytes")

	// ErrPipelineClosed indicates an operation on a closed weight stream pipeline.
	ErrPipelineClosed = errors.New("cache: weight stream pipeline is closed")
)

// CalculateEvictionRatio computes the number of times a given weight footprint purges the MALL cache.
func CalculateEvictionRatio(weightBytes int64, mallCapacityBytes int64) float64 {
	if mallCapacityBytes <= 0 {
		mallCapacityBytes = MALLCapacityBytes
	}
	return float64(weightBytes) / float64(mallCapacityBytes)
}

// CacheModifierFlags defines the hardware cache control bits injected into RDNA 3.5
// buffer resource descriptors (V#) and AQL packets.
//
// In RDNA 3.5 (GFX1151):
//   - SLC (System Level Coherent): when set to 1, bypasses the MALL Infinity Cache,
//     streaming reads directly from DRAM into L2/vector registers without allocating in MALL.
//   - GLC (Globally Coherent): when set to 1, bypasses L1/L2 vector caches for global coherence.
//   - DLC (Device Level Coherent): when set to 1, bypasses L1 scalar cache.
type CacheModifierFlags struct {
	SLC         int    `json:"slc"`          // System Level Coherent: 0 = temporal MALL cache, 1 = bypass MALL
	GLC         int    `json:"glc"`          // Globally Coherent: 0 = normal caching, 1 = bypass L1/L2
	DLC         int    `json:"dlc"`          // Device Level Coherent: 0 = normal caching, 1 = bypass L1 scalar
	NonTemporal bool   `json:"non_temporal"` // True if non-temporal streaming access
	Temporal    bool   `json:"temporal"`     // True if temporal reuse access
	Bypass      bool   `json:"bypass"`       // True if explicitly bypassing MALL
	PolicyName  string `json:"policy_name"`  // e.g. "NON_TEMPORAL_BYPASS" or "TEMPORAL_PINNED"
}

// IsSLCSet returns true if the System Level Coherent bit is set (MALL bypass active).
func (c CacheModifierFlags) IsSLCSet() bool {
	return c.SLC == 1
}

// IsGLCSet returns true if Globally Coherent bit is set.
func (c CacheModifierFlags) IsGLCSet() bool {
	return c.GLC == 1
}

// IsDLCSet returns true if Device Level Coherent bit is set.
func (c CacheModifierFlags) IsDLCSet() bool {
	return c.DLC == 1
}

// IsNonTemporal returns true if the flags specify non-temporal streaming semantics.
func (c CacheModifierFlags) IsNonTemporal() bool {
	return c.NonTemporal
}

// IsTemporal returns true if the flags specify temporal caching semantics.
func (c CacheModifierFlags) IsTemporal() bool {
	return c.Temporal
}

// IsBypass returns true if the flags instruct the memory controller to bypass the MALL cache.
func (c CacheModifierFlags) IsBypass() bool {
	return c.Bypass || c.SLC == 1
}

// Standard predefined cache modifier flag configurations.
var (
	// FlagsStreamingBypass configures non-temporal weight streaming (SLC=1, GLC=0, DLC=0),
	// routing 17.5 GB weight loads around the 32MB MALL Infinity Cache directly into vector registers.
	FlagsStreamingBypass = CacheModifierFlags{
		SLC:         1,
		GLC:         0,
		DLC:         0,
		NonTemporal: true,
		Temporal:    false,
		Bypass:      true,
		PolicyName:  "NON_TEMPORAL_BYPASS",
	}

	// FlagsTemporalPinned configures temporal caching (SLC=0, GLC=0, DLC=0),
	// pinning hot KV cache blocks and speculative tree masks in the 32MB MALL cache.
	FlagsTemporalPinned = CacheModifierFlags{
		SLC:         0,
		GLC:         0,
		DLC:         0,
		NonTemporal: false,
		Temporal:    true,
		Bypass:      false,
		PolicyName:  "TEMPORAL_PINNED",
	}
)

// BufferDescriptor represents an architected 128-bit RDNA 3.5 V# buffer resource descriptor
// structured across four 32-bit dwords (Word0..Word3).
//
// Hardware Bitfield Layout (GFX1151 / RDNA 3.5):
//   - Word0 [31:0]: Base address low 32 bits (bits [31:0] of 48-bit virtual address).
//   - Word1 [15:0]: Base address high 16 bits (bits [47:32] of 48-bit virtual address).
//   - Word1 [31:16]: Stride in bytes (e.g. 128-byte burst stride).
//   - Word2 [31:0]: Buffer size / number of records in bytes (uint32).
//   - Word3 [31:28]: Resource type (0x8 for buffer descriptor).
//   - Word3 [22]: SLC (System Level Coherent: 1 = bypass MALL, 0 = allocate in MALL).
//   - Word3 [13]: DLC (Device Level Coherent: 1 = bypass L1 scalar cache).
//   - Word3 [12]: GLC (Globally Coherent: 1 = bypass L1/L2 vector caches).
type BufferDescriptor struct {
	Word0 uint32             `json:"word0"`
	Word1 uint32             `json:"word1"`
	Word2 uint32             `json:"word2"`
	Word3 uint32             `json:"word3"`
	Flags CacheModifierFlags `json:"flags"`
}

// BaseAddress extracts the full 48-bit virtual address from Word0 and Word1.
func (b BufferDescriptor) BaseAddress() uint64 {
	return (uint64(b.Word1&0xFFFF) << 32) | uint64(b.Word0)
}

// Size returns the buffer size in bytes from Word2.
func (b BufferDescriptor) Size() uint32 {
	return b.Word2
}

// Stride extracts the stride in bytes from Word1 bits [31:16].
func (b BufferDescriptor) Stride() uint16 {
	return uint16((b.Word1 >> 16) & 0xFFFF)
}

// IsSLCSet returns true if bit 22 (SLC) is asserted in Word3.
func (b BufferDescriptor) IsSLCSet() bool {
	return (b.Word3 & (1 << 22)) != 0
}

// IsGLCSet returns true if bit 12 (GLC) is asserted in Word3.
func (b BufferDescriptor) IsGLCSet() bool {
	return (b.Word3 & (1 << 12)) != 0
}

// IsDLCSet returns true if bit 13 (DLC) is asserted in Word3.
func (b BufferDescriptor) IsDLCSet() bool {
	return (b.Word3 & (1 << 13)) != 0
}

// ResourceType returns the resource type field from Word3 bits [31:28].
func (b BufferDescriptor) ResourceType() uint8 {
	return uint8((b.Word3 >> 28) & 0xF)
}

// Raw returns the 128-bit descriptor as a 4-element uint32 array.
func (b BufferDescriptor) Raw() [4]uint32 {
	return [4]uint32{b.Word0, b.Word1, b.Word2, b.Word3}
}

// TensorKind identifies the architectural role and access pattern of a tensor.
type TensorKind string

const (
	// TensorKindWeight represents read-once, non-recurrent model weights (QKV, MLP up/down/gate).
	TensorKindWeight TensorKind = "weight"

	// TensorKindKVCache represents recurrent, multi-step attention KV cache blocks.
	TensorKindKVCache TensorKind = "kv_cache"

	// TensorKindTreeMask represents 2D causal ancestor masks for speculative draft trees.
	TensorKindTreeMask TensorKind = "tree_mask"

	// TensorKindActivation represents transient intermediate activations.
	TensorKindActivation TensorKind = "activation"
)

// TensorMemoryAttributes describes the memory footprint, access recurrence,
// and hardware classification parameters for a tensor.
type TensorMemoryAttributes struct {
	Name         string     `json:"name"`
	SizeBytes    int64      `json:"size_bytes"`
	Kind         TensorKind `json:"kind"`
	NonRecurrent bool       `json:"non_recurrent"`
	ReadOnce     bool       `json:"read_once"`
	BaseAddr     uint64     `json:"base_addr"`
}

// WeightStreamTelemetry aggregates hardware and simulation telemetry metrics
// capturing MALL cache residency, eviction counts, and DRAM bandwidth conservation.
type WeightStreamTelemetry struct {
	MALLEvictions         uint64    `json:"mall_evictions"`
	MALLResidencyPct      float64   `json:"mall_residency_pct"`
	DRAMBandwidthSavedGBs float64   `json:"dram_bandwidth_saved_gbs"`
	ThroughputGBs         float64   `json:"throughput_gbs"`
	WeightBytesStreamed   int64     `json:"weight_bytes_streamed"`
	PinnedKVBytes         int64     `json:"pinned_kv_bytes"`
	LayersProcessed       int       `json:"layers_processed"`
	BypassedDescriptors   uint64    `json:"bypassed_descriptors"`
	TemporalDescriptors   uint64    `json:"temporal_descriptors"`
	EvictionRatio         float64   `json:"eviction_ratio"`
	DRAMKVBytesRead       int64     `json:"dram_kv_bytes_read"`
	Timestamp             time.Time `json:"timestamp"`
}

// WeightStreamConfig configures the non-temporal weight streaming pipeline.
type WeightStreamConfig struct {
	BurstStrideBytes           int     `json:"burst_stride_bytes"`            // Default: 128 bytes (8 channels * 16 bytes)
	MemoryChannels             int     `json:"memory_channels"`               // Default: 8
	PrefetchLookahead          int     `json:"prefetch_lookahead"`            // Default: 1 layer lookahead
	WeightStreamThresholdBytes int64   `json:"weight_stream_threshold_bytes"` // Default: 10 MiB
	TargetBandwidthGBs         float64 `json:"target_bandwidth_gbs"`          // Default: 220.0 GB/s
	MaxQueueDepth              int     `json:"max_queue_depth"`               // Default: 8
}

// DefaultWeightStreamConfig returns standard configuration matching AMD Strix Halo silicon.
func DefaultWeightStreamConfig() WeightStreamConfig {
	return WeightStreamConfig{
		BurstStrideBytes:           BurstStrideBytes,
		MemoryChannels:             MemoryChannels,
		PrefetchLookahead:          1,
		WeightStreamThresholdBytes: WeightStreamThresholdBytes,
		TargetBandwidthGBs:         TargetStreamingBWGBs,
		MaxQueueDepth:              8,
	}
}

// Validate verifies that the configuration conforms to physical memory and cache constraints.
func (c *WeightStreamConfig) Validate() error {
	if c.BurstStrideBytes <= 0 || c.BurstStrideBytes%BurstStrideBytes != 0 {
		return fmt.Errorf("%w: got %d, expected multiple of %d", ErrUnalignedStride, c.BurstStrideBytes, BurstStrideBytes)
	}
	if c.MemoryChannels <= 0 {
		return errors.New("cache: memory channels must be positive")
	}
	if c.WeightStreamThresholdBytes <= 0 {
		return errors.New("cache: weight stream threshold bytes must be positive")
	}
	if c.TargetBandwidthGBs <= 0 {
		c.TargetBandwidthGBs = TargetStreamingBWGBs
	}
	if c.MaxQueueDepth <= 0 {
		c.MaxQueueDepth = 8
	}
	return nil
}
