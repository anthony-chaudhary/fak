// Package strix implements high-density KV cache packing, micro-scaling quantization,
// and 32MB MALL (Memory Attached Last-Level) Infinity Cache attention tiling for AMD Strix Halo (GFX1151).
package strix

import (
	"errors"
	"fmt"
)

// MALLSizeBytes is the on-die 32MB MALL Infinity Cache size.
const MALLSizeBytes = StrixHaloMALLSizeBytes

// AttentionPrecision identifies numerical precision for attention KV cache tensors.
type AttentionPrecision string

const (
	// AttentionPrecisionFP16 represents 16-bit half-precision float (2 bytes per element).
	AttentionPrecisionFP16 AttentionPrecision = "FP16"

	// AttentionPrecisionBF16 represents 16-bit bfloat16 (2 bytes per element).
	AttentionPrecisionBF16 AttentionPrecision = "BF16"

	// AttentionPrecisionFP8 represents 8-bit float (1 byte per element).
	AttentionPrecisionFP8 AttentionPrecision = "FP8"

	// AttentionPrecisionINT8 represents 8-bit signed integer (1 byte per element).
	AttentionPrecisionINT8 AttentionPrecision = "INT8"
)

// PartitionArm identifies the allocation ablation arm for 32MB MALL partitioning.
type PartitionArm string

const (
	// ArmStatic8kPinnedRoot allocates the 32MB MALL to an 8,192-token pinned root prefix.
	ArmStatic8kPinnedRoot PartitionArm = "STATIC_8K_PINNED_ROOT"

	// ArmDynamic4kRootDraft partitions 32MB into 16MB root (4,096 tokens), 14MB draft, and 2MB reserve.
	ArmDynamic4kRootDraft PartitionArm = "DYNAMIC_4K_ROOT_DRAFT"

	// ArmUnpartitionedBaseline operates with ad-hoc unmanaged allocation and DRAM streaming fallback.
	ArmUnpartitionedBaseline PartitionArm = "UNPARTITIONED_BASELINE"
)

// Typed errors for MALL attention working set geometry and capacity partitioning.
var (
	// ErrInvalidGeometry indicates invalid attention dimensions (e.g. non-positive layers, heads, dim, or bytes).
	ErrInvalidGeometry = errors.New("strix/cache: invalid attention geometry: layers, heads, dim, and bytes must be positive")

	// ErrMALLPartitionOverflow indicates requested partition sizes exceed the physical 32MB (33,554,432 bytes) MALL capacity.
	ErrMALLPartitionOverflow = errors.New("strix/cache: MALL partition overflow: requested allocation exceeds 32MB (33,554,432 bytes)")

	// ErrInvalidPartitionConfig indicates missing or contradictory partition configuration parameters.
	ErrInvalidPartitionConfig = errors.New("strix/cache: invalid partition config: negative token count or reserve bytes")
)

// AttentionGeometry defines the architectural tensor geometry of the attention working set.
// Standard GQA on AMD Strix Halo: 8 KV heads, dim 128, FP16 = 4,096 bytes per token per layer.
type AttentionGeometry struct {
	Layers          int                `json:"layers"`            // Transformer layer count (e.g. 32 or 40)
	KVHeads         int                `json:"kv_heads"`          // Key-Value attention head count (e.g. 8 for GQA)
	HeadDim         int                `json:"head_dim"`          // Dimension per head (e.g. 128)
	BytesPerElement int                `json:"bytes_per_element"` // Element byte width (2 for FP16/BF16, 1 for FP8)
	Precision       AttentionPrecision `json:"precision"`         // Precision format tag
}

// DefaultGQAAttentionGeometry returns reference GQA geometry for Qwen/Llama models on Strix Halo:
// 32 layers, 8 KV heads, dim 128, FP16 (2 bytes).
func DefaultGQAAttentionGeometry() AttentionGeometry {
	return AttentionGeometry{
		Layers:          32,
		KVHeads:         GQAKVHeads,          // 8
		HeadDim:         GQAHeadDim,          // 128
		BytesPerElement: FP16BytesPerElement, // 2
		Precision:       AttentionPrecisionFP16,
	}
}

// Validate checks that all geometry parameters are strictly positive.
func (g AttentionGeometry) Validate() error {
	if g.Layers <= 0 || g.KVHeads <= 0 || g.HeadDim <= 0 || g.BytesPerElement <= 0 {
		return ErrInvalidGeometry
	}
	return nil
}

// KeyElementsPerToken returns Key elements per token for one layer: KVHeads * HeadDim.
func (g AttentionGeometry) KeyElementsPerToken() int64 {
	return int64(g.KVHeads) * int64(g.HeadDim)
}

// ValueElementsPerToken returns Value elements per token for one layer: KVHeads * HeadDim.
func (g AttentionGeometry) ValueElementsPerToken() int64 {
	return int64(g.KVHeads) * int64(g.HeadDim)
}

// ElementsPerTokenPerLayer returns combined Key + Value elements per token for one layer:
// 2 * KVHeads * HeadDim.
func (g AttentionGeometry) ElementsPerTokenPerLayer() int64 {
	return 2 * int64(g.KVHeads) * int64(g.HeadDim)
}

// BytesPerTokenPerLayer computes the KV footprint in bytes for a single token in one layer:
// 2 (K+V) * KVHeads * HeadDim * BytesPerElement.
// For standard GQA (8 heads, dim 128, FP16): 2 * 8 * 128 * 2 = 4,096 bytes/token.
func (g AttentionGeometry) BytesPerTokenPerLayer() int64 {
	return g.ElementsPerTokenPerLayer() * int64(g.BytesPerElement)
}

// TotalBytesPerToken computes the total KV footprint in bytes across all transformer layers:
// 2 (K+V) * Layers * KVHeads * HeadDim * BytesPerElement.
func (g AttentionGeometry) TotalBytesPerToken() int64 {
	return int64(g.Layers) * g.BytesPerTokenPerLayer()
}

// HeadStrideBytes computes the stride in bytes for a single attention head: HeadDim * BytesPerElement.
func (g AttentionGeometry) HeadStrideBytes() int64 {
	return int64(g.HeadDim) * int64(g.BytesPerElement)
}

// KVStrideBytes computes the stride in bytes across all KV heads in one state: KVHeads * HeadDim * BytesPerElement.
func (g AttentionGeometry) KVStrideBytes() int64 {
	return int64(g.KVHeads) * g.HeadStrideBytes()
}

// PartitionSegment represents an allocated or reserved byte segment within the 32MB MALL.
type PartitionSegment struct {
	Name        string `json:"name"`         // Segment identifier: "RootPrefix", "ActiveDraft", "Reserve", "Headroom"
	SizeBytes   int64  `json:"size_bytes"`   // Allocated bytes
	OffsetBytes int64  `json:"offset_bytes"` // Byte offset in 32MB MALL
	Tokens      int    `json:"tokens"`       // Token capacity accommodated
	CachePolicy string `json:"cache_policy"` // RDNA 3.5 cache policy hint ("TEMPORAL_PINNED", "TRANSIENT_PINNED", etc.)
}

// PartitionMetrics tracks real-time partition utilization, head stride geometry, and capacity margin.
type PartitionMetrics struct {
	TotalCapacityBytes     int64   `json:"total_capacity_bytes"`
	TotalAllocatedBytes    int64   `json:"total_allocated_bytes"`
	RemainingHeadroomBytes int64   `json:"remaining_headroom_bytes"`
	MALLUtilizationRatio   float64 `json:"mall_utilization_ratio"`
	RootPrefixUtilization  float64 `json:"root_prefix_utilization"`
	DraftUtilization       float64 `json:"draft_utilization"`
	ReserveRatio           float64 `json:"reserve_ratio"`
	HeadroomRatio          float64 `json:"headroom_ratio"`
	BytesPerToken          int64   `json:"bytes_per_token"`
	MaxMALLTokens          int     `json:"max_mall_tokens"`
	Fallbacks              int64   `json:"fallbacks"`
}

// MALLPartitionConfig specifies the configuration for partitioning the 32MB MALL.
type MALLPartitionConfig struct {
	Geometry               AttentionGeometry `json:"geometry"`
	TotalMALLCapacityBytes int64             `json:"total_mall_capacity_bytes"` // Defaults to MALLSizeBytes (33,554,432 bytes)
	RootPrefixTokens       int               `json:"root_prefix_tokens"`        // Target pinned root prompt prefix tokens
	DraftTokens            int               `json:"draft_tokens"`              // Target active speculative draft tokens
	ReserveBytes           int64             `json:"reserve_bytes"`             // Safety reserve in bytes (e.g. 2MB = 2,097,152)
	LayerTiled             bool              `json:"layer_tiled"`               // If true, partitions per-layer tile; if false, partitions across all layers
	Arm                    PartitionArm      `json:"arm"`                       // Target ablation arm
}

// Validate checks that the partition configuration parameters are well-formed.
func (c MALLPartitionConfig) Validate() error {
	if err := c.Geometry.Validate(); err != nil {
		return err
	}
	if c.RootPrefixTokens < 0 || c.DraftTokens < 0 || c.ReserveBytes < 0 {
		return ErrInvalidPartitionConfig
	}
	if c.TotalMALLCapacityBytes <= 0 {
		return errors.New("strix/cache: total MALL capacity must be positive")
	}
	return nil
}

// EffectiveBytesPerToken returns the token byte stride governed by the LayerTiled configuration.
func (c MALLPartitionConfig) EffectiveBytesPerToken() int64 {
	if c.LayerTiled {
		return c.Geometry.BytesPerTokenPerLayer()
	}
	return c.Geometry.TotalBytesPerToken()
}

// MALLPartitionPlan contains the deterministic capacity partitioning layout across the 32MB MALL.
type MALLPartitionPlan struct {
	Config                 MALLPartitionConfig `json:"config"`
	RootPrefix             PartitionSegment    `json:"root_prefix"`
	ActiveDraft            PartitionSegment    `json:"active_draft"`
	Reserve                PartitionSegment    `json:"reserve"`
	Headroom               PartitionSegment    `json:"headroom"`
	TotalAllocatedBytes    int64               `json:"total_allocated_bytes"`
	RemainingHeadroomBytes int64               `json:"remaining_headroom_bytes"`
	IsPartitioned          bool                `json:"is_partitioned"`
	FallbackToDRAM         bool                `json:"fallback_to_dram"`
	StatusMessage          string              `json:"status_message"`
	Metrics                PartitionMetrics    `json:"metrics"`
	PolicyDirectives       CachePolicyHint     `json:"policy_directives"`
}

// String returns a human-readable summary of the partition plan.
func (p *MALLPartitionPlan) String() string {
	if p.FallbackToDRAM {
		return fmt.Sprintf("MALLPartitionPlan[FALLBACK_DRAM]: %s", p.StatusMessage)
	}
	return fmt.Sprintf("MALLPartitionPlan[PARTITIONED]: Root=%d B (%d tok), Draft=%d B (%d tok), Reserve=%d B, Headroom=%d B (Util=%.2f%%)",
		p.RootPrefix.SizeBytes, p.RootPrefix.Tokens,
		p.ActiveDraft.SizeBytes, p.ActiveDraft.Tokens,
		p.Reserve.SizeBytes, p.RemainingHeadroomBytes,
		p.Metrics.MALLUtilizationRatio*100.0)
}
