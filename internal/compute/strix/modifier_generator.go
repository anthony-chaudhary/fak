// Package strix implements high-density KV cache packing, micro-scaling quantization,
// 32MB MALL Infinity Cache attention tiling, and RDNA 3.5 cache modifier bitmask generation
// for AMD Strix Halo (Ryzen AI Max+ 395 / GFX1151).
package strix

import (
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ModifierGenerator synthesizes and validates RDNA 3.5 (GFX1151) cache modifier
// bitmasks for assembly instruction emission and V# buffer resource descriptors.
type ModifierGenerator struct {
	mu        sync.RWMutex
	config    GeneratorConfig
	telemetry ModifierGeneratorTelemetry
	closed    bool
}

// NewModifierGenerator creates and initializes an RDNA 3.5 cache modifier generator.
// If TargetArch is unspecified, it defaults to TargetArchGFX1151 ("gfx1151").
// If AblationArm is unspecified, it defaults to Arm1ExplicitBitmasks.
func NewModifierGenerator(cfg GeneratorConfig) (*ModifierGenerator, error) {
	if cfg.TargetArch == "" {
		cfg.TargetArch = TargetArchGFX1151
	}
	if cfg.AblationArm == "" {
		cfg.AblationArm = Arm1ExplicitBitmasks
	}

	if cfg.StrictArchValidation && !ValidateGFX1151Architecture(cfg.TargetArch) {
		return nil, fmt.Errorf("%w: %q", ErrUnsupportedArchitecture, cfg.TargetArch)
	}

	return &ModifierGenerator{
		config: cfg,
		telemetry: ModifierGeneratorTelemetry{
			LastTimestamp: time.Now(),
		},
	}, nil
}

// ValidateGFX1151Architecture checks whether an architecture string identifies
// the AMD Strix Halo GFX1151 (RDNA 3.5) APU compute target.
func ValidateGFX1151Architecture(arch string) bool {
	clean := strings.ToLower(strings.TrimSpace(arch))
	return clean == TargetArchGFX1151 || strings.Contains(clean, "gfx1151")
}

// GenerateBitmask synthesizes an InstructionModifierBitmask based on cache tier intent,
// active ablation arm, and architecture configuration.
//
// Quarantined Fallback Mechanism:
// If target architecture is not confirmed as gfx1151 or assembly tagging is disabled,
// generator returns zeroed bitmasks (driver default untagged loads) without aborting inference.
func (g *ModifierGenerator) GenerateBitmask(intent CacheTierIntent) (InstructionModifierBitmask, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.closed {
		return InstructionModifierBitmask{}, ErrGeneratorClosed
	}

	g.telemetry.TotalGenerated++
	g.telemetry.LastTimestamp = time.Now()

	// Quarantined fallback check
	if !ValidateGFX1151Architecture(g.config.TargetArch) || !g.config.EnableAssemblyTagging {
		g.telemetry.FallbackUntaggedEmitted++
		return InstructionModifierBitmask{
			SLC:                 0,
			GLC:                 0,
			DLC:                 0,
			InstructionDword0:   0,
			InstructionDword1:   0,
			DescriptorWord3Mask: 0,
			AssemblySuffix:      "",
			Intent:              intent,
			Arm:                 g.config.AblationArm,
			TargetArch:          g.config.TargetArch,
			Tagged:              false,
		}, nil
	}

	// Arm 2: Driver Default (transparent hardware LRU caching, untagged loads)
	if g.config.AblationArm == Arm2DriverDefault {
		g.telemetry.DriverDefaultEmitted++
		return InstructionModifierBitmask{
			SLC:                 0,
			GLC:                 0,
			DLC:                 0,
			InstructionDword0:   0,
			InstructionDword1:   0,
			DescriptorWord3Mask: 0,
			AssemblySuffix:      "",
			Intent:              intent,
			Arm:                 Arm2DriverDefault,
			TargetArch:          g.config.TargetArch,
			Tagged:              false,
		}, nil
	}

	// Arm 3: Aggressive Bypass (full non-temporal bypass: SLC=1, GLC=1 for both MALL and Vector L1)
	if g.config.AblationArm == Arm3AggressiveBypass {
		g.telemetry.AggressiveBypassEmitted++
		return InstructionModifierBitmask{
			SLC:                 1,
			GLC:                 1,
			DLC:                 0,
			InstructionDword0:   MUBUFGLCBit,
			InstructionDword1:   MUBUFSLCBit,
			DescriptorWord3Mask: DescriptorWord3SLCBit | DescriptorWord3GLCBit,
			AssemblySuffix:      "slc glc",
			Intent:              intent,
			Arm:                 Arm3AggressiveBypass,
			TargetArch:          g.config.TargetArch,
			Tagged:              true,
		}, nil
	}

	// Arm 1: Explicit Bitmasks (Dynamic emission based on Intent)
	switch intent {
	case IntentTemporalRootKV:
		g.telemetry.TemporalRootKVEmitted++
		return InstructionModifierBitmask{
			SLC:                 0,
			GLC:                 0,
			DLC:                 0,
			InstructionDword0:   0,
			InstructionDword1:   0,
			DescriptorWord3Mask: 0,
			AssemblySuffix:      "",
			Intent:              IntentTemporalRootKV,
			Arm:                 Arm1ExplicitBitmasks,
			TargetArch:          g.config.TargetArch,
			Tagged:              true,
		}, nil

	case IntentNonTemporalWeight:
		g.telemetry.NonTemporalWeightEmitted++
		return InstructionModifierBitmask{
			SLC:                 1,
			GLC:                 0,
			DLC:                 0,
			InstructionDword0:   0,
			InstructionDword1:   MUBUFSLCBit,
			DescriptorWord3Mask: DescriptorWord3SLCBit,
			AssemblySuffix:      "slc",
			Intent:              IntentNonTemporalWeight,
			Arm:                 Arm1ExplicitBitmasks,
			TargetArch:          g.config.TargetArch,
			Tagged:              true,
		}, nil

	case IntentAggressiveBypass:
		g.telemetry.AggressiveBypassEmitted++
		return InstructionModifierBitmask{
			SLC:                 1,
			GLC:                 1,
			DLC:                 0,
			InstructionDword0:   MUBUFGLCBit,
			InstructionDword1:   MUBUFSLCBit,
			DescriptorWord3Mask: DescriptorWord3SLCBit | DescriptorWord3GLCBit,
			AssemblySuffix:      "slc glc",
			Intent:              IntentAggressiveBypass,
			Arm:                 Arm1ExplicitBitmasks,
			TargetArch:          g.config.TargetArch,
			Tagged:              true,
		}, nil

	case IntentDriverDefault:
		g.telemetry.DriverDefaultEmitted++
		return InstructionModifierBitmask{
			SLC:                 0,
			GLC:                 0,
			DLC:                 0,
			InstructionDword0:   0,
			InstructionDword1:   0,
			DescriptorWord3Mask: 0,
			AssemblySuffix:      "",
			Intent:              IntentDriverDefault,
			Arm:                 Arm1ExplicitBitmasks,
			TargetArch:          g.config.TargetArch,
			Tagged:              false,
		}, nil

	default:
		g.telemetry.ValidationErrors++
		return InstructionModifierBitmask{}, fmt.Errorf("%w: %q", ErrInvalidCacheIntent, intent)
	}
}

// ValidateDescriptorWord3 verifies that a 32-bit Word3 from a 128-bit RDNA 3.5 V# buffer descriptor
// conforms to official AMD GFX11 architecture specifications and matches expected cache modifiers.
func (g *ModifierGenerator) ValidateDescriptorWord3(word3 uint32, expected InstructionModifierBitmask) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.closed {
		return ErrGeneratorClosed
	}

	g.telemetry.DescriptorWord3Validations++

	// 1. Validate Resource Type (bits [31:28] must be 0x8 for buffer descriptor in GFX11)
	resType := word3 & (0xF << 28)
	if resType != DescriptorWord3ResourceTypeBuffer {
		g.telemetry.ValidationErrors++
		return fmt.Errorf("%w: resource type bits [31:28]=0x%X, want 0x8 (buffer)", ErrDescriptorValidationFailed, resType>>28)
	}

	// 2. Validate SLC bit (bit 22 in Word3)
	hasSLC := (word3 & DescriptorWord3SLCBit) != 0
	if expected.SLC == 1 && !hasSLC {
		g.telemetry.ValidationErrors++
		return fmt.Errorf("%w: SLC bit 22 is 0, want 1 for non-temporal bypass", ErrDescriptorValidationFailed)
	}
	if expected.SLC == 0 && hasSLC {
		g.telemetry.ValidationErrors++
		return fmt.Errorf("%w: SLC bit 22 is 1, want 0 for temporal caching", ErrDescriptorValidationFailed)
	}

	// 3. Validate GLC bit (bit 12 in Word3)
	hasGLC := (word3 & DescriptorWord3GLCBit) != 0
	if expected.GLC == 1 && !hasGLC {
		g.telemetry.ValidationErrors++
		return fmt.Errorf("%w: GLC bit 12 is 0, want 1 for vector L1 bypass", ErrDescriptorValidationFailed)
	}
	if expected.GLC == 0 && hasGLC {
		g.telemetry.ValidationErrors++
		return fmt.Errorf("%w: GLC bit 12 is 1, want 0 for coherent vector caching", ErrDescriptorValidationFailed)
	}

	// 4. Validate DLC bit (bit 13 in Word3)
	hasDLC := (word3 & DescriptorWord3DLCBit) != 0
	if expected.DLC == 1 && !hasDLC {
		g.telemetry.ValidationErrors++
		return fmt.Errorf("%w: DLC bit 13 is 0, want 1 for device level bypass", ErrDescriptorValidationFailed)
	}
	if expected.DLC == 0 && hasDLC {
		g.telemetry.ValidationErrors++
		return fmt.Errorf("%w: DLC bit 13 is 1, want 0 for device level caching", ErrDescriptorValidationFailed)
	}

	return nil
}

// SynthesizeDescriptorWord3 constructs an architected GFX11 Word3 for a buffer resource descriptor (V#),
// injecting the resource type (0x8) and the active cache modifier bitmask.
func (g *ModifierGenerator) SynthesizeDescriptorWord3(baseWord3 uint32, bitmask InstructionModifierBitmask) uint32 {
	// Clear existing resource type [31:28] and cache control bits [22, 13, 12]
	cleaned := baseWord3 &^ ((0xF << 28) | DescriptorWord3SLCBit | DescriptorWord3DLCBit | DescriptorWord3GLCBit)
	// Inject buffer resource type (0x8) and modifier flags
	return cleaned | DescriptorWord3ResourceTypeBuffer | bitmask.DescriptorWord3Mask
}

// SynthesizeBufferLoadInstruction formats an RDNA 3.5 assembly instruction string
// and computes the corresponding 64-bit binary machine instruction encoding.
func (g *ModifierGenerator) SynthesizeBufferLoadInstruction(
	opcode string,
	dstVGPR string,
	srcVAddr string,
	srcRsrc string,
	offset int,
	bitmask InstructionModifierBitmask,
) (string, uint64, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.closed {
		return "", 0, ErrGeneratorClosed
	}

	// 1. Resolve opcode binary encoding
	var opBin uint32
	switch opcode {
	case OpcodeBufferLoadDword:
		opBin = MUBUFOpcodeBufferLoadDword
	case OpcodeBufferLoadDwordx2:
		opBin = MUBUFOpcodeBufferLoadDwordx2
	case OpcodeBufferLoadDwordx3:
		opBin = MUBUFOpcodeBufferLoadDwordx3
	case OpcodeBufferLoadDwordx4:
		opBin = MUBUFOpcodeBufferLoadDwordx4
	case OpcodeBufferStoreDword:
		opBin = MUBUFOpcodeBufferStoreDword
	case OpcodeBufferStoreDwordx4:
		opBin = MUBUFOpcodeBufferStoreDwordx4
	default:
		g.telemetry.ValidationErrors++
		return "", 0, fmt.Errorf("%w: %q", ErrUnsupportedOpcode, opcode)
	}

	// 2. Synthesize assembly text
	asmLine := fmt.Sprintf("%s %s, %s, %s, %d offen", opcode, dstVGPR, srcVAddr, srcRsrc, offset)
	if bitmask.AssemblySuffix != "" {
		asmLine += " " + bitmask.AssemblySuffix
	}

	// 3. Compute 64-bit machine instruction encoding
	// DWord 0: MUBUF prefix [31:26] | Opcode [25:18] | OFFEN [12] | 12-bit Offset [11:0] | InstructionDword0 (GLC/DLC)
	dword0 := MUBUFOpcodePrefix | opBin | MUBUFOffenBit | (uint32(offset) & 0xFFF) | bitmask.InstructionDword0

	// DWord 1: VDATA [31:24] | VADDR [23:16] | InstructionDword1 (SLC [14]) | SRSRC [7:0]
	dstReg := parseRegisterIndex(dstVGPR)
	vaddrReg := parseRegisterIndex(srcVAddr)
	srsrcReg := parseRegisterIndex(srcRsrc)

	dword1 := ((uint32(dstReg) & 0xFF) << 24) |
		((uint32(vaddrReg) & 0xFF) << 16) |
		bitmask.InstructionDword1 |
		(uint32(srsrcReg/4) & 0x7F) // SRSRC is encoded as SGPR quad index (sgpr_base / 4)

	rawEncoding := (uint64(dword1) << 32) | uint64(dword0)
	g.telemetry.EmittedAssemblyLines++

	return asmLine, rawEncoding, nil
}

// parseRegisterIndex extracts the base register numeric index from string tokens
// like "v0", "v[0:3]", "s[0:3]", "s4".
func parseRegisterIndex(reg string) int {
	s := strings.TrimSpace(reg)
	s = strings.TrimPrefix(s, "v")
	s = strings.TrimPrefix(s, "s")
	s = strings.TrimPrefix(s, "[")
	colonIdx := strings.Index(s, ":")
	if colonIdx != -1 {
		s = s[:colonIdx]
	}
	s = strings.TrimSuffix(s, "]")
	val, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return val
}

// Telemetry returns a snapshot of runtime metrics captured by the generator.
func (g *ModifierGenerator) Telemetry() ModifierGeneratorTelemetry {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.telemetry
}

// ResetTelemetry clears all captured telemetry metrics.
func (g *ModifierGenerator) ResetTelemetry() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.telemetry = ModifierGeneratorTelemetry{
		LastTimestamp: time.Now(),
	}
}

// Close marks the generator closed and prevents further emissions.
func (g *ModifierGenerator) Close() error {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.closed = true
	return nil
}
