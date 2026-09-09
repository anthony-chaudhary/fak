// Package strix implements high-density KV cache packing, micro-scaling quantization,
// and 32MB MALL (Memory Attached Last-Level) Infinity Cache attention tiling for AMD Strix Halo (GFX1151).
package strix

import (
	"errors"
	"fmt"
	"time"
)

// Operational constants for AMD Strix Halo 32MB MALL Infinity Cache attention tiling.
const (
	// DefaultStrixRootTokens is the standard 8,192-token system prompt prefix length.
	DefaultStrixRootTokens = 8192

	// DefaultMALLHitRateTarget is the minimum target hit rate for root prefix cache lines (>= 95.0%).
	DefaultMALLHitRateTarget = 0.95

	// MinRootMALLResidencyPct is the minimum required resident percentage of the pinned root KV cache (>= 95.0%).
	MinRootMALLResidencyPct = 95.0

	// DefaultDRAMBandwidthSavingsFloorGBs is the required minimum LPDDR5X DRAM bandwidth
	// reduction achieved by locking root KV in 32MB MALL (>= 32.0 GB/s).
	DefaultDRAMBandwidthSavingsFloorGBs = 32.0

	// PhysicalLPDDR5XBandwidthGBs is the theoretical peak memory bandwidth of the 256-bit
	// LPDDR5X-8533 memory crossbar on AMD Strix Halo (273.056 GB/s).
	PhysicalLPDDR5XBandwidthGBs = 273.056

	// IntraMALLBandwidthFloorGBs is the minimum internal SRAM crossbar transfer rate (> 1.2 TB/s).
	IntraMALLBandwidthFloorGBs = 1200.0

	// MALLCapacityBytes is the exact 32 MiB capacity of the on-die Infinity Cache.
	MALLCapacityBytes = StrixHaloMALLSizeBytes

	// MALLLineSizeBytes is the fundamental hardware coherency granule across Zen 5 and RDNA 3.5 (64 bytes).
	MALLLineSizeBytes = CacheLineBytes
)

// Typed errors for MALL attention working set tiling and cache hint generation.
var (
	// ErrTileOverflow indicates that the requested attention working set exceeds physical 32MB MALL capacity.
	ErrTileOverflow = errors.New("strix/cache: attention tile working set exceeds 32MB MALL capacity (33,554,432 bytes)")

	// ErrInvalidTileGeometry indicates invalid or non-positive attention geometry dimensions.
	ErrInvalidTileGeometry = errors.New("strix/cache: invalid attention geometry for tiling: layers, kv_heads, and head_dim must be positive")

	// ErrTilerClosed indicates an operation on a closed MALL tiler instance.
	ErrTilerClosed = errors.New("strix/cache: MALL attention tiler is closed")
)

// MALLCacheHint identifies cache tier intent and memory hierarchy placement for attention buffers.
type MALLCacheHint string

const (
	// HintTemporalRootKV indicates hot, multi-turn attention KV prefix blocks pinned
	// in 32MB MALL Infinity Cache with temporal reuse (SLC=0, GLC=0, DLC=0).
	HintTemporalRootKV MALLCacheHint = "TEMPORAL_ROOT_KV"

	// HintNonTemporalWeight indicates streaming model weights that bypass 32MB MALL
	// to prevent LRU cache pollution (SLC=1, GLC=0, DLC=0).
	HintNonTemporalWeight MALLCacheHint = "NONTEMPORAL_WEIGHT"

	// HintDivergentKV indicates transient subagent-specific continuation tokens exceeding
	// the root prefix capacity, streaming non-temporally to protect the root tile (SLC=1, GLC=0, DLC=0).
	HintDivergentKV MALLCacheHint = "DIVERGENT_KV"

	// HintAggressiveBypass indicates full non-temporal bypass across both Vector L1 and MALL (SLC=1, GLC=1, DLC=0).
	HintAggressiveBypass MALLCacheHint = "AGGRESSIVE_BYPASS"

	// HintDriverDefault indicates standard untagged memory loads relying on transparent driver LRU caching.
	HintDriverDefault MALLCacheHint = "DRIVER_DEFAULT"
)

// MALLTile represents a contiguous token slice mapped into the attention memory hierarchy.
type MALLTile struct {
	TileID         int                        `json:"tile_id"`
	StartToken     int                        `json:"start_token"`
	EndToken       int                        `json:"end_token"`
	TokenCount     int                        `json:"token_count"`
	BytesPerToken  int64                      `json:"bytes_per_token"`
	TotalSizeBytes int64                      `json:"total_size_bytes"`
	IsRootPinned   bool                       `json:"is_root_pinned"`
	Hint           MALLCacheHint              `json:"hint"`
	ModifierFlags  CacheModifierFlags         `json:"modifier_flags"`
	Bitmask        InstructionModifierBitmask `json:"bitmask"`
	AssemblySuffix string                     `json:"assembly_suffix"`
}

// MALLTilerConfig configures attention geometry, capacity constraints, and ISA modifier flags.
type MALLTilerConfig struct {
	Geometry               AttentionGeometry   `json:"geometry"`
	TotalMALLCapacityBytes int64               `json:"total_mall_capacity_bytes"` // Defaults to MALLCapacityBytes (33,554,432 bytes)
	RootTokenCount         int                 `json:"root_token_count"`          // Defaults to DefaultStrixRootTokens (8,192 tokens)
	TargetArch             string              `json:"target_arch"`               // Defaults to TargetArchGFX1151 ("gfx1151")
	EnableAssemblyTagging  bool                `json:"enable_assembly_tagging"`   // Synthesize explicit SLC/GLC bitmasks
	AblationArm            ModifierAblationArm `json:"ablation_arm"`              // Ablation mode (default: Arm1ExplicitBitmasks)
}

// Validate checks configuration invariants and establishes default fallback values.
func (c *MALLTilerConfig) Validate() error {
	if c.TotalMALLCapacityBytes <= 0 {
		c.TotalMALLCapacityBytes = MALLCapacityBytes
	}
	if c.RootTokenCount <= 0 {
		c.RootTokenCount = DefaultStrixRootTokens
	}
	if c.TargetArch == "" {
		c.TargetArch = TargetArchGFX1151
	}
	if c.AblationArm == "" {
		c.AblationArm = Arm1ExplicitBitmasks
	}

	if c.Geometry.Layers <= 0 || c.Geometry.KVHeads <= 0 || c.Geometry.HeadDim <= 0 {
		return ErrInvalidTileGeometry
	}
	if c.Geometry.BytesPerElement <= 0 {
		c.Geometry.BytesPerElement = 2 // FP16 default
	}

	return nil
}

// MALLTilerPlan captures the calculated working set geometry, capacity margins, and RDNA 3.5 cache policies.
type MALLTilerPlan struct {
	Geometry                 AttentionGeometry `json:"geometry"`
	BytesPerTokenPerLayer    int64             `json:"bytes_per_token_per_layer"`
	TotalMALLCapacityBytes   int64             `json:"total_mall_capacity_bytes"`
	RootTokens               int               `json:"root_tokens"`
	RootAllocatedBytes       int64             `json:"root_allocated_bytes"`
	RemainingMALLBytes       int64             `json:"remaining_mall_bytes"`
	CapacityUtilizationPct   float64           `json:"capacity_utilization_pct"`
	RootTile                 MALLTile          `json:"root_tile"`
	WeightTilePolicy         MALLTile          `json:"weight_tile_policy"`
	DivergentTilePolicy      MALLTile          `json:"divergent_tile_policy"`
	EstimatedDRAMSavingsGBps float64           `json:"estimated_dram_savings_gbps"`
	TargetHitRate            float64           `json:"target_hit_rate"`
}

// MALLTilerTelemetry records real-time hardware hit rates, bypass counters, and bandwidth savings.
type MALLTilerTelemetry struct {
	RootHits                 int64     `json:"root_hits"`
	RootMisses               int64     `json:"root_misses"`
	WeightBypasses           int64     `json:"weight_bypasses"`
	DivergentLoads           int64     `json:"divergent_loads"`
	TotalAccesses            int64     `json:"total_accesses"`
	RootHitRate              float64   `json:"root_hit_rate"`
	DRAMBytesSaved           int64     `json:"dram_bytes_saved"`
	EffectiveDRAMSavingsGBps float64   `json:"effective_dram_savings_gbps"`
	LastUpdated              time.Time `json:"last_updated"`
}

// String provides a concise summary of telemetry state.
func (t MALLTilerTelemetry) String() string {
	return fmt.Sprintf("MALLTiler[Hits=%d, Misses=%d, Rate=%.2f%%, Bypasses=%d, Saved=%dMB, BW_Saved=%.1fGB/s]",
		t.RootHits, t.RootMisses, t.RootHitRate*100.0, t.WeightBypasses, t.DRAMBytesSaved/(1024*1024), t.EffectiveDRAMSavingsGBps)
}
