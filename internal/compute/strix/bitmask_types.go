// Package strix implements high-density KV cache packing, micro-scaling quantization,
// 32MB MALL Infinity Cache attention tiling, and RDNA 3.5 cache modifier bitmask generation
// for AMD Strix Halo (Ryzen AI Max+ 395 / GFX1151).
package strix

import (
	"errors"
	"time"
)

// Hardware architecture targets for AMD RDNA APUs.
const (
	// TargetArchGFX1151 is the AMD Strix Halo RDNA 3.5 APU compute architecture (Ryzen AI Max+ 395).
	TargetArchGFX1151 = "gfx1151"

	// TargetArchGFX1150 is the AMD Strix Point RDNA 3.5 APU compute architecture.
	TargetArchGFX1150 = "gfx1150"

	// TargetArchGFX1100 is the AMD Navi 31/32 RDNA 3 discrete GPU compute architecture.
	TargetArchGFX1100 = "gfx1100"
)

// Cache control bit definitions for RDNA 3.5 (GFX1151).
const (
	// MUBUF instruction encoding bitmasks (GFX11 ISA):
	// MUBUFOpcodePrefix is the 6-bit instruction prefix for MUBUF format (bits [31:26] = 0b111000 = 0x38).
	MUBUFOpcodePrefix uint32 = 0x38 << 26

	// MUBUFGLCBit is the Globally Coherent bit in MUBUF DWord 0 (bit 14).
	// GLC=1 bypasses or invalidates Vector L1 (GL1) cache.
	MUBUFGLCBit uint32 = 1 << 14

	// MUBUFDLCBit is the Device Level Coherent bit in MUBUF DWord 0 (bit 15).
	// DLC=1 bypasses Vector L0 (GL0) cache.
	MUBUFDLCBit uint32 = 1 << 15

	// MUBUFOffenBit is the Offset Enable bit in MUBUF DWord 0 (bit 12).
	MUBUFOffenBit uint32 = 1 << 12

	// MUBUFIdxenBit is the Index Enable bit in MUBUF DWord 0 (bit 13).
	MUBUFIdxenBit uint32 = 1 << 13

	// MUBUFSLCBit is the System Level Coherent bit in MUBUF DWord 1 (bit 14).
	// SLC=1 bypasses the 32MB MALL Infinity Cache, streaming directly to/from DRAM.
	MUBUFSLCBit uint32 = 1 << 14

	// Buffer Resource Descriptor (V# 128-bit descriptor) Word3 bit definitions (AMD GFX11 specification):
	// DescriptorWord3ResourceTypeBuffer is bits [31:28] = 0x8 for buffer resources.
	DescriptorWord3ResourceTypeBuffer uint32 = 0x8 << 28

	// DescriptorWord3SLCBit is bit 22 in Word3 (System Level Coherent: 1 = bypass MALL, 0 = allocate in MALL).
	DescriptorWord3SLCBit uint32 = 1 << 22

	// DescriptorWord3DLCBit is bit 13 in Word3 (Device Level Coherent: 1 = bypass vector cache).
	DescriptorWord3DLCBit uint32 = 1 << 13

	// DescriptorWord3GLCBit is bit 12 in Word3 (Globally Coherent: 1 = bypass L1/L2 vector caches).
	DescriptorWord3GLCBit uint32 = 1 << 12

	// DescriptorWord3FormatMask isolates data format bits in Word3 [27:23].
	DescriptorWord3FormatMask uint32 = 0x1F << 23
)

// Supported buffer load opcode strings.
const (
	OpcodeBufferLoadDword    = "buffer_load_dword"
	OpcodeBufferLoadDwordx2  = "buffer_load_dwordx2"
	OpcodeBufferLoadDwordx3  = "buffer_load_dwordx3"
	OpcodeBufferLoadDwordx4  = "buffer_load_dwordx4"
	OpcodeBufferStoreDword   = "buffer_store_dword"
	OpcodeBufferStoreDwordx4 = "buffer_store_dwordx4"
)

// Opcode binary values for MUBUF buffer instructions (bits [25:18] in DWord 0).
const (
	MUBUFOpcodeBufferLoadDword    uint32 = 0x00 << 18
	MUBUFOpcodeBufferLoadDwordx2  uint32 = 0x01 << 18
	MUBUFOpcodeBufferLoadDwordx3  uint32 = 0x02 << 18
	MUBUFOpcodeBufferLoadDwordx4  uint32 = 0x03 << 18
	MUBUFOpcodeBufferStoreDword   uint32 = 0x18 << 18
	MUBUFOpcodeBufferStoreDwordx4 uint32 = 0x1B << 18
)

// CacheTierIntent identifies the operational cache tier and lifetime characteristics of the buffer.
type CacheTierIntent string

const (
	// IntentTemporalRootKV indicates hot, multi-step attention KV cache prefix blocks
	// pinned in 32MB MALL (SLC=0, GLC=0, DLC=0).
	IntentTemporalRootKV CacheTierIntent = "TEMPORAL_ROOT_KV"

	// IntentNonTemporalWeight indicates read-once streaming model weights
	// bypassing 32MB MALL (SLC=1, GLC=0, DLC=0).
	IntentNonTemporalWeight CacheTierIntent = "NONTEMPORAL_WEIGHT"

	// IntentAggressiveBypass indicates full non-temporal bypass across both
	// MALL and Vector L1 (SLC=1, GLC=1, DLC=0).
	IntentAggressiveBypass CacheTierIntent = "AGGRESSIVE_BYPASS"

	// IntentDriverDefault indicates standard untagged loads relying on transparent
	// driver LRU caching (SLC=0, GLC=0, DLC=0, untagged).
	IntentDriverDefault CacheTierIntent = "DRIVER_DEFAULT"
)

// ModifierAblationArm identifies the ablation experimental arm.
type ModifierAblationArm string

const (
	// Arm1ExplicitBitmasks dynamically emits SLC=0, GLC=0 for KV cache loads and SLC=1, GLC=0 for streaming weights.
	Arm1ExplicitBitmasks ModifierAblationArm = "ARM1_EXPLICIT_BITMASKS"

	// Arm2DriverDefault emits standard untagged instructions relying on driver transparent LRU caching.
	Arm2DriverDefault ModifierAblationArm = "ARM2_DRIVER_DEFAULT"

	// Arm3AggressiveBypass emits full non-temporal bypass with SLC=1, GLC=1 for both Vector L1 and MALL Infinity Cache.
	Arm3AggressiveBypass ModifierAblationArm = "ARM3_AGGRESSIVE_BYPASS"
)

// InstructionModifierBitmask represents the synthesized cache control fields
// for RDNA 3.5 (GFX1151) instruction and descriptor generation.
type InstructionModifierBitmask struct {
	SLC                 int                 `json:"slc"`
	GLC                 int                 `json:"glc"`
	DLC                 int                 `json:"dlc"`
	InstructionDword0   uint32              `json:"instruction_dword0"`
	InstructionDword1   uint32              `json:"instruction_dword1"`
	DescriptorWord3Mask uint32              `json:"descriptor_word3_mask"`
	AssemblySuffix      string              `json:"assembly_suffix"`
	Intent              CacheTierIntent     `json:"intent"`
	Arm                 ModifierAblationArm `json:"arm"`
	TargetArch          string              `json:"target_arch"`
	Tagged              bool                `json:"tagged"`
}

// GeneratorConfig holds configuration parameters for the cache modifier generator.
type GeneratorConfig struct {
	TargetArch            string              `json:"target_arch"`
	AblationArm           ModifierAblationArm `json:"ablation_arm"`
	EnableAssemblyTagging bool                `json:"enable_assembly_tagging"`
	StrictArchValidation  bool                `json:"strict_arch_validation"`
}

// DefaultGeneratorConfig returns standard configuration targeting AMD Strix Halo GFX1151 silicon.
func DefaultGeneratorConfig() GeneratorConfig {
	return GeneratorConfig{
		TargetArch:            TargetArchGFX1151,
		AblationArm:           Arm1ExplicitBitmasks,
		EnableAssemblyTagging: true,
		StrictArchValidation:  true,
	}
}

// ModifierGeneratorTelemetry records metrics on generated bitmasks, encodings, and validation outcomes.
type ModifierGeneratorTelemetry struct {
	TotalGenerated             uint64    `json:"total_generated"`
	TemporalRootKVEmitted      uint64    `json:"temporal_root_kv_emitted"`
	NonTemporalWeightEmitted   uint64    `json:"non_temporal_weight_emitted"`
	AggressiveBypassEmitted    uint64    `json:"aggressive_bypass_emitted"`
	DriverDefaultEmitted       uint64    `json:"driver_default_emitted"`
	FallbackUntaggedEmitted    uint64    `json:"fallback_untagged_emitted"`
	DescriptorWord3Validations uint64    `json:"descriptor_word3_validations"`
	ValidationErrors           uint64    `json:"validation_errors"`
	EmittedAssemblyLines       uint64    `json:"emitted_assembly_lines"`
	LastTimestamp              time.Time `json:"last_timestamp"`
}

// Typed sentinel errors for cache modifier bitmask generation and validation.
var (
	// ErrUnsupportedArchitecture is returned when the target architecture does not match GFX1151.
	ErrUnsupportedArchitecture = errors.New("strix/cache: unsupported GPU architecture: target must be gfx1151 for RDNA 3.5 cache modifiers")

	// ErrInvalidCacheIntent is returned when an unrecognized or empty cache intent is provided.
	ErrInvalidCacheIntent = errors.New("strix/cache: invalid or unspecified cache tier intent")

	// ErrDescriptorValidationFailed is returned when a buffer descriptor fails GFX11 architecture specification checks.
	ErrDescriptorValidationFailed = errors.New("strix/cache: buffer descriptor word3 cache control bits failed GFX11 validation")

	// ErrUnsupportedOpcode is returned when an instruction opcode is not supported by the MUBUF encoder.
	ErrUnsupportedOpcode = errors.New("strix/cache: unsupported buffer instruction opcode")

	// ErrGeneratorClosed is returned when operations are attempted on a closed generator.
	ErrGeneratorClosed = errors.New("strix/cache: cache modifier generator is closed")
)
