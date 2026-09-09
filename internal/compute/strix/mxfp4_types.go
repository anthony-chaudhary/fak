// Package strix implements high-density KV cache packing, micro-scaling quantization,
// and 32MB MALL (Memory Attached Last-Level) Infinity Cache attention tiling for AMD Strix Halo (GFX1151).
package strix

import "errors"

// Physical silicon constants for AMD Strix Halo (Ryzen AI Max+ 395 / GFX1151).
const (
	// MALLSets is the number of sets in the 32MB MALL cache.
	MALLSets = 32768

	// MALLWays is the 16-way set associativity of the MALL cache.
	MALLWays = 16

	// MALLLineBytes is the MALL cache line size in bytes (64 bytes).
	MALLLineBytes = 64

	// GQAKVHeads is the reference number of Key-Value attention heads in GQA models (e.g. Qwen 2.5 / 3.8 Coder 35B).
	GQAKVHeads = 8

	// GQAHeadDim is the dimension per attention head (d = 128).
	GQAHeadDim = 128

	// ElementsPerHead is the number of elements per attention head (128).
	ElementsPerHead = GQAHeadDim

	// KeyElementsPerToken is the total elements in the Key state per token: 8 heads * 128 dim = 1,024 elements.
	KeyElementsPerToken = GQAKVHeads * GQAHeadDim

	// ValueElementsPerToken is the total elements in the Value state per token: 8 heads * 128 dim = 1,024 elements.
	ValueElementsPerToken = GQAKVHeads * GQAHeadDim

	// ElementsPerToken is the combined Key and Value elements per token: 1,024 + 1,024 = 2,048 elements.
	ElementsPerToken = KeyElementsPerToken + ValueElementsPerToken

	// FP16BytesPerElement is the size of standard half-precision IEEE 754 float16 elements (2 bytes).
	FP16BytesPerElement = 2

	// FP16BytesPerToken is the uncompressed KV cache footprint per token under FP16:
	// 2 (K+V) * 8 heads * 128 dim * 2 bytes = 4,096 bytes per token.
	FP16BytesPerToken = ElementsPerToken * FP16BytesPerElement

	// FP16MaxMALLTokens is the maximum number of GQA tokens that fit in 32MB MALL under FP16:
	// 33,554,432 bytes / 4,096 bytes/token = 8,192 tokens.
	FP16MaxMALLTokens = int(MALLSizeBytes / int64(FP16BytesPerToken))

	// MicroBlockElements is the OCP MXFP4 micro-vector block granularity (32 elements).
	MicroBlockElements = 32

	// MicroBlockPayloadBytes is the packed 4-bit nibbles for 32 elements (16 bytes = 128 bits).
	MicroBlockPayloadBytes = 16

	// MicroBlockScaleBytes is the 1-byte E8M0 scale factor per 32-element micro-block.
	MicroBlockScaleBytes = 1

	// MicroBlockSizeBytes is the total byte size of an MXFP4 block: 16 + 1 = 17 bytes.
	MicroBlockSizeBytes = MicroBlockPayloadBytes + MicroBlockScaleBytes

	// KeyBlocksPerToken is the number of MXFP4 micro-blocks in the Key state: 1,024 / 32 = 32 blocks.
	KeyBlocksPerToken = KeyElementsPerToken / MicroBlockElements

	// ValueBlocksPerToken is the number of MXFP4 micro-blocks in the Value state: 1,024 / 32 = 32 blocks.
	ValueBlocksPerToken = ValueElementsPerToken / MicroBlockElements

	// BlocksPerToken is the total MXFP4 blocks per token: 32 Key + 32 Value = 64 blocks.
	BlocksPerToken = KeyBlocksPerToken + ValueBlocksPerToken

	// MXFP4BytesPerToken is the packed KV cache footprint per token under OCP MXFP4 micro-scaling:
	// 64 blocks * 17 bytes = 1,088 bytes per token (3.7647x compression vs 4,096 bytes FP16, 4.25 bpw).
	MXFP4BytesPerToken = BlocksPerToken * MicroBlockSizeBytes

	// MXFP4MaxMALLTokens is the maximum number of tokens fitting in 32MB MALL under pure MXFP4:
	// floor(33,554,432 / 1,088) = exactly 30,840 tokens (33,553,920 bytes).
	MXFP4MaxMALLTokens = int(MALLSizeBytes / int64(MXFP4BytesPerToken))

	// MALLHeadroomBytes is the unused SRAM remaining in 32MB MALL when 30,840 MXFP4 tokens are stored:
	// 33,554,432 - (30,840 * 1,088) = 512 bytes.
	MALLHeadroomBytes = MALLSizeBytes - int64(MXFP4MaxMALLTokens*MXFP4BytesPerToken)

	// AttentionSinkTokens is the default count of initial attention sink tokens preserved in high precision (128).
	AttentionSinkTokens = 128

	// SustainedContigDRAMBandwidthGBs is the target sustained DRAM decode memory bandwidth (220.0 GB/s).
	SustainedContigDRAMBandwidthGBs = 220.0

	// PhysicalDRAMBandwidthGBs is the theoretical peak DRAM bandwidth of 16-channel LPDDR5X-8533 (273.056 GB/s).
	PhysicalDRAMBandwidthGBs = StrixHaloPhysicalDRAMBandwidthGBs

	// PeakMALLBandwidthGBs is the peak intra-cache transfer bandwidth (> 1.2 TB/s).
	PeakMALLBandwidthGBs = StrixHaloPeakMALLBandwidthGBs

	// E8M0ExponentBias is the IEEE 754 float32 exponent bias used by OCP MXFP4 E8M0 scales (127).
	E8M0ExponentBias = 127

	// E2M1MaxRepresentable is the maximum finite value representable in 4-bit FP4 E2M1 (6.0).
	E2M1MaxRepresentable = 6.0
)

// Typed errors for MXFP4 micro-scaling KV cache tiling and allocation.
var (
	// ErrInvalidTokenIndex indicates an invalid or negative token index was requested.
	ErrInvalidTokenIndex = errors.New("strix/cache: invalid negative token index")

	// ErrNilTokenFrame indicates a nil token frame operand was provided.
	ErrNilTokenFrame = errors.New("strix/cache: nil MXFP4TokenFrame operand")

	// ErrCapacityExceeded indicates that allocation or tiling exceeded available cache capacity.
	ErrCapacityExceeded = errors.New("strix/cache: capacity exceeded")

	// ErrInvalidFloatValue indicates a NaN or Infinite float value was encountered in activations.
	ErrInvalidFloatValue = errors.New("strix/cache: NaN or Inf float value encountered in KV activation")

	// ErrCorruptedFrameData indicates raw byte buffer does not match the 1,088-byte MXFP4 token frame format.
	ErrCorruptedFrameData = errors.New("strix/cache: raw byte slice length does not match 1,088 bytes")
)

// MXFP4Block represents an OCP MXFP4 micro-scaled block of 32 elements.
// Layout:
//   - 16 bytes: 32 packed 4-bit FP4 (E2M1) nibbles (2 elements per byte).
//   - 1 byte: E8M0 unsigned exponent scale factor (Scale in [0, 255], representing 2^(Scale-127)).
//
// Total block footprint is exactly 17 bytes (16 + 1).
type MXFP4Block struct {
	Data  [MicroBlockPayloadBytes]byte `json:"data"`  // 16 bytes packed nibbles
	Scale uint8                        `json:"scale"` // 1 byte E8M0 scale exponent
}

// MXFP4TokenFrame represents a packed KV activation frame for a single token under GQA 8 KV heads, dim 128.
// Contains 32 Key blocks + 32 Value blocks = 64 blocks, totaling 64 * 17 = 1,088 bytes.
type MXFP4TokenFrame struct {
	TokenIndex  int                             `json:"token_index"`
	KeyBlocks   [KeyBlocksPerToken]MXFP4Block   `json:"key_blocks"`   // 32 blocks (544 bytes)
	ValueBlocks [ValueBlocksPerToken]MXFP4Block `json:"value_blocks"` // 32 blocks (544 bytes)
	IsSink      bool                            `json:"is_sink"`
}

// FP16TokenFrame represents an uncompressed high-precision KV activation frame used for
// attention sink protection (first 128 tokens) or smooth fallback mode.
type FP16TokenFrame struct {
	TokenIndex int       `json:"token_index"`
	KeyData    []float32 `json:"key_data"`   // 1,024 float32 elements
	ValueData  []float32 `json:"value_data"` // 1,024 float32 elements
	IsSink     bool      `json:"is_sink"`
}

// Predefined cache policy hints for RDNA 3.5 shader dispatch.
var (
	// CacheHintTemporalPinned directs CUs to fetch tokens with temporal caching (SLC=0, GLC=0, NT=0),
	// pinning hot KV blocks in the 32MB MALL Infinity Cache.
	CacheHintTemporalPinned = CachePolicyHint{
		SLC:        0,
		GLC:        0,
		NT:         0,
		Temporal:   true,
		Bypass:     false,
		PolicyName: "TEMPORAL_PINNED",
	}
)

// MXFP4TilerConfig configures the MXFP4 KV cache tiling engine.
type MXFP4TilerConfig struct {
	MALLSizeBytes  int64 `json:"mall_size_bytes"`
	KVHeads        int   `json:"kv_heads"`
	HeadDim        int   `json:"head_dim"`
	SinkProtection bool  `json:"sink_protection"`
	SinkTokens     int   `json:"sink_tokens"`
	FallbackMode   bool  `json:"fallback_mode"`
}

// MXFP4Telemetry captures runtime residency, compression, scale distribution, and DRAM offload statistics.
type MXFP4Telemetry struct {
	ResidencyTokens       int              `json:"residency_tokens"`
	MaxMALLTokens         int              `json:"max_mall_tokens"`
	ResidencyRatio        float64          `json:"residency_ratio"`
	CompressionRatio      float64          `json:"compression_ratio"`
	EffectiveBPW          float64          `json:"effective_bpw"`
	AllocatedBytes        int64            `json:"allocated_bytes"`
	MALLCapacityBytes     int64            `json:"mall_capacity_bytes"`
	MALLHeadroomBytes     int64            `json:"mall_headroom_bytes"`
	HitRate               float64          `json:"hit_rate"`
	DRAMOffloadGBs        float64          `json:"dram_offload_gbs"`
	EffectiveBandwidthGBs float64          `json:"effective_bandwidth_gbs"`
	ScaleDistribution     map[uint8]uint64 `json:"scale_distribution"`
	AttentionSinkCount    int              `json:"attention_sink_count"`
	FallbackActive        bool             `json:"fallback_active"`
	Mode                  string           `json:"mode"`
}

// GFX1151ResidencyEvaluation encapsulates the evaluation metrics comparing MXFP4 32MB MALL on-die
// residency against uncompressed FP16 on the physical AMD Strix Halo appliance.
type GFX1151ResidencyEvaluation struct {
	ContextTokens         int     `json:"context_tokens"`
	FP16ResidentTokens    int     `json:"fp16_resident_tokens"`
	FP16SpilledTokens     int     `json:"fp16_spilled_tokens"`
	FP16ResidencyRatio    float64 `json:"fp16_residency_ratio"`
	FP16EffectiveBWGBs    float64 `json:"fp16_effective_bw_gbs"`
	MXFP4ResidentTokens   int     `json:"mxfp4_resident_tokens"`
	MXFP4SpilledTokens    int     `json:"mxfp4_spilled_tokens"`
	MXFP4ResidencyRatio   float64 `json:"mxfp4_residency_ratio"`
	MXFP4EffectiveBWGBs   float64 `json:"mxfp4_effective_bw_gbs"`
	BandwidthSpeedupRatio float64 `json:"bandwidth_speedup_ratio"`
	DRAMTrafficSavedGBs   float64 `json:"dram_traffic_saved_gbs"`
	SWVerified            bool    `json:"sw_verified"`
	HWWitnessedRequired   bool    `json:"hw_witnessed_required"`
	StatusMessage         string  `json:"status_message"`
}

// RawBytes serializes the MXFP4TokenFrame into a contiguous 1,088-byte buffer.
// Key blocks occupy bytes 0..543 (32 * 17), Value blocks occupy bytes 544..1087 (32 * 17).
func (f *MXFP4TokenFrame) RawBytes() []byte {
	raw := make([]byte, MXFP4BytesPerToken)
	offset := 0

	for i := 0; i < KeyBlocksPerToken; i++ {
		copy(raw[offset:offset+MicroBlockPayloadBytes], f.KeyBlocks[i].Data[:])
		offset += MicroBlockPayloadBytes
		raw[offset] = f.KeyBlocks[i].Scale
		offset += MicroBlockScaleBytes
	}

	for i := 0; i < ValueBlocksPerToken; i++ {
		copy(raw[offset:offset+MicroBlockPayloadBytes], f.ValueBlocks[i].Data[:])
		offset += MicroBlockPayloadBytes
		raw[offset] = f.ValueBlocks[i].Scale
		offset += MicroBlockScaleBytes
	}

	return raw
}

// FromBytes populates the MXFP4TokenFrame from a contiguous 1,088-byte buffer.
func (f *MXFP4TokenFrame) FromBytes(raw []byte) error {
	if len(raw) != MXFP4BytesPerToken {
		return ErrCorruptedFrameData
	}
	offset := 0

	for i := 0; i < KeyBlocksPerToken; i++ {
		copy(f.KeyBlocks[i].Data[:], raw[offset:offset+MicroBlockPayloadBytes])
		offset += MicroBlockPayloadBytes
		f.KeyBlocks[i].Scale = raw[offset]
		offset += MicroBlockScaleBytes
	}

	for i := 0; i < ValueBlocksPerToken; i++ {
		copy(f.ValueBlocks[i].Data[:], raw[offset:offset+MicroBlockPayloadBytes])
		offset += MicroBlockPayloadBytes
		f.ValueBlocks[i].Scale = raw[offset]
		offset += MicroBlockScaleBytes
	}

	return nil
}
