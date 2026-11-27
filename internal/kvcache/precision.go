package kvcache

import (
	"errors"
	"fmt"
	"strings"
)

const (
	// NVFP4BlockSize is the element group size for block-16 NVFP4 KV storage.
	NVFP4BlockSize = 16

	// NVFP4MaxE2M1 is the maximum representable magnitude in OCP E2M1.
	NVFP4MaxE2M1 = 6.0

	// NVFP4MaxE4M3FN is the maximum finite positive scale in OCP E4M3FN.
	NVFP4MaxE4M3FN = 448.0

	// NVFP4MaxBlockAbsmax is the maximum block absmax representable without
	// scale overflow (6.0 * 448.0 = 2688.0).
	NVFP4MaxBlockAbsmax = NVFP4MaxE2M1 * NVFP4MaxE4M3FN

	// NVFP4MinAbsmax is the sub-threshold floor below which a block encodes as zero.
	NVFP4MinAbsmax = 1e-30
)

var (
	// ErrInvalidPrecision is returned when a KV precision token is unrecognized.
	ErrInvalidPrecision = errors.New("kvcache: invalid precision")
	// ErrInvalidLayout is returned when a KV layout descriptor is malformed or unsupported.
	ErrInvalidLayout = errors.New("kvcache: invalid layout descriptor")
)

// Precision identifies the storage format for KV cache vectors.
type Precision string

const (
	PrecisionFP32  Precision = "fp32"
	PrecisionFP16  Precision = "fp16"
	PrecisionBF16  Precision = "bf16"
	PrecisionFP8   Precision = "fp8"
	PrecisionQ8_0  Precision = "q8_0"
	PrecisionQ4_0  Precision = "q4_0"
	PrecisionNVFP4 Precision = "nvfp4"
)

// ParsePrecision parses a case-insensitive KV precision selector.
func ParsePrecision(s string) (Precision, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "fp32", "f32", "32":
		return PrecisionFP32, nil
	case "fp16", "f16", "16":
		return PrecisionFP16, nil
	case "bf16", "bfloat16":
		return PrecisionBF16, nil
	case "fp8", "e4m3", "e4m3fn":
		return PrecisionFP8, nil
	case "q8_0", "q8", "8":
		return PrecisionQ8_0, nil
	case "q4_0", "q4", "4":
		return PrecisionQ4_0, nil
	case "nvfp4", "fp4", "e2m1":
		return PrecisionNVFP4, nil
	default:
		return "", fmt.Errorf("%w: %q (want fp32, fp16, bf16, fp8, q8_0, q4_0, or nvfp4)", ErrInvalidPrecision, s)
	}
}

// Valid reports whether p is a recognized Precision constant.
func (p Precision) Valid() bool {
	switch p {
	case PrecisionFP32, PrecisionFP16, PrecisionBF16, PrecisionFP8, PrecisionQ8_0, PrecisionQ4_0, PrecisionNVFP4:
		return true
	default:
		return false
	}
}

// LayoutMode describes how tokens are arranged in the KV cache backing store.
type LayoutMode string

const (
	LayoutContiguous LayoutMode = "contiguous"
	LayoutPaged      LayoutMode = "paged"
)

// LayoutDesc is a model-agnostic KV cache geometry and precision descriptor
// shared across Inkling-2, Qwen, Gemma, Llama, and DeepSeek backends.
type LayoutDesc struct {
	Arch                    string     `json:"arch,omitempty"`
	Precision               Precision  `json:"precision"`
	Layers                  int        `json:"layers"`
	KVHeads                 int        `json:"kv_heads"`
	HeadDim                 int        `json:"head_dim"`
	Capacity                int        `json:"capacity,omitempty"`
	Layout                  LayoutMode `json:"layout,omitempty"`
	PageTokens              int        `json:"page_tokens,omitempty"`
	ExtraStateFloatsPerSlot int        `json:"extra_state_floats_per_slot,omitempty"`
}

// Validate checks that the descriptor has consistent positive geometry and
// satisfies format-specific alignment invariants (e.g. HeadDim % 16 == 0 for NVFP4).
func (d LayoutDesc) Validate() error {
	if !d.Precision.Valid() {
		return fmt.Errorf("%w: unknown precision %q", ErrInvalidLayout, d.Precision)
	}
	if d.Layers <= 0 || d.KVHeads <= 0 || d.HeadDim <= 0 {
		return fmt.Errorf("%w: layers=%d kv_heads=%d head_dim=%d must all be positive", ErrInvalidLayout, d.Layers, d.KVHeads, d.HeadDim)
	}
	if d.Capacity < 0 {
		return fmt.Errorf("%w: capacity=%d must be non-negative", ErrInvalidLayout, d.Capacity)
	}
	if d.ExtraStateFloatsPerSlot < 0 {
		return fmt.Errorf("%w: extra_state_floats_per_slot=%d must be non-negative", ErrInvalidLayout, d.ExtraStateFloatsPerSlot)
	}
	mode := d.Layout
	if mode == "" {
		mode = LayoutContiguous
	}
	switch mode {
	case LayoutContiguous:
	case LayoutPaged:
		if d.PageTokens <= 0 {
			return fmt.Errorf("%w: paged layout requires page_tokens > 0", ErrInvalidLayout)
		}
	default:
		return fmt.Errorf("%w: unknown layout mode %q", ErrInvalidLayout, d.Layout)
	}
	if d.Precision == PrecisionNVFP4 && d.HeadDim%NVFP4BlockSize != 0 {
		return fmt.Errorf("%w: nvfp4 requires head_dim (%d) divisible by %d", ErrInvalidLayout, d.HeadDim, NVFP4BlockSize)
	}
	return nil
}

// VectorBytes returns the packed byte count for a single vector of length dim
// at precision p, including per-group scale/metadata bytes.
func VectorBytes(dim int, p Precision) int64 {
	if dim <= 0 {
		return 0
	}
	switch p {
	case PrecisionFP32:
		return int64(dim) * 4
	case PrecisionFP16, PrecisionBF16:
		return int64(dim) * 2
	case PrecisionFP8:
		// E4M3FN code (1B/elem) + 1 F32 scale per vector.
		return int64(dim) + 4
	case PrecisionQ8_0:
		groups := int64((dim + 31) / 32)
		return int64(dim) + groups*4
	case PrecisionQ4_0:
		groups := int64((dim + 31) / 32)
		return int64((dim+1)/2) + groups*8
	case PrecisionNVFP4:
		blocks := int64((dim + NVFP4BlockSize - 1) / NVFP4BlockSize)
		return int64((dim+1)/2) + blocks
	default:
		return int64(dim) * 4
	}
}

// NVFP4CodeBytes returns the packed E2M1 code byte count for one K or V buffer
// of tokens positions across KVHeads and HeadDim.
func (d LayoutDesc) NVFP4CodeBytes(tokens int) int64 {
	if tokens <= 0 || d.KVHeads <= 0 || d.HeadDim <= 0 {
		return 0
	}
	return int64(tokens) * int64(d.KVHeads) * int64((d.HeadDim+1)/2)
}

// NVFP4ScaleBytes returns the E4M3FN block-16 scale byte count for one K or V
// buffer of tokens positions across KVHeads and HeadDim.
func (d LayoutDesc) NVFP4ScaleBytes(tokens int) int64 {
	if tokens <= 0 || d.KVHeads <= 0 || d.HeadDim <= 0 {
		return 0
	}
	blocksPerHead := int64((d.HeadDim + NVFP4BlockSize - 1) / NVFP4BlockSize)
	return int64(tokens) * int64(d.KVHeads) * blocksPerHead
}

// NVFP4CodeFloats returns the 32-bit word count for one NVFP4 K or V code buffer.
func (d LayoutDesc) NVFP4CodeFloats(tokens int) int {
	return int((d.NVFP4CodeBytes(tokens) + 3) / 4)
}

// NVFP4ScaleFloats returns the 32-bit word count for one NVFP4 K or V scale buffer.
func (d LayoutDesc) NVFP4ScaleFloats(tokens int) int {
	return int((d.NVFP4ScaleBytes(tokens) + 3) / 4)
}

// PerTokenLayerBytes returns the K+V byte footprint for one token in one layer/slot.
func (d LayoutDesc) PerTokenLayerBytes() int64 {
	if d.KVHeads <= 0 || d.HeadDim <= 0 {
		return 0
	}
	switch d.Precision {
	case PrecisionFP8:
		// E4M3FN codes per element + one F32 scale per (token, kvHead), for both K and V.
		perVec := int64(d.KVHeads) * (int64(d.HeadDim) + 4)
		return 2 * perVec
	case PrecisionNVFP4:
		return 2 * (d.NVFP4CodeBytes(1) + d.NVFP4ScaleBytes(1))
	default:
		return 2 * VectorBytes(d.KVHeads*d.HeadDim, d.Precision)
	}
}

// PerTokenBytes returns the K+V byte footprint for one token across all layers/slots.
func (d LayoutDesc) PerTokenBytes() int64 {
	if d.Layers <= 0 {
		return 0
	}
	return int64(d.Layers) * d.PerTokenLayerBytes()
}

// TotalBytes returns the total KV byte footprint for tokens positions across all layers/slots.
func (d LayoutDesc) TotalBytes(tokens int) int64 {
	if tokens <= 0 || d.Layers <= 0 {
		return 0
	}
	extra := int64(d.Layers) * int64(d.ExtraStateFloatsPerSlot) * 4
	return int64(tokens)*d.PerTokenBytes() + extra
}

// CompressionRatioVsFP32 returns the ratio of FP32 KV bytes/token to this layout's bytes/token.
func (d LayoutDesc) CompressionRatioVsFP32() float64 {
	cur := d.PerTokenBytes()
	if cur <= 0 {
		return 0
	}
	fp32 := LayoutDesc{
		Precision: PrecisionFP32,
		Layers:    d.Layers,
		KVHeads:   d.KVHeads,
		HeadDim:   d.HeadDim,
	}.PerTokenBytes()
	return float64(fp32) / float64(cur)
}

// CompressionRatioVsFP16 returns the ratio of FP16 KV bytes/token to this layout's bytes/token.
func (d LayoutDesc) CompressionRatioVsFP16() float64 {
	cur := d.PerTokenBytes()
	if cur <= 0 {
		return 0
	}
	fp16 := LayoutDesc{
		Precision: PrecisionFP16,
		Layers:    d.Layers,
		KVHeads:   d.KVHeads,
		HeadDim:   d.HeadDim,
	}.PerTokenBytes()
	return float64(fp16) / float64(cur)
}
