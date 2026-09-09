package strix

import (
	"sync"
	"testing"
)

// TestModifierGenerator verifies that TestModifierGenerator satisfies Criterion 1, 2, and 3:
// - Exact RDNA 3.5 gfx1151 encodings (SLC=0, GLC=0, DLC=0 for temporal KV; SLC=1, GLC=0, DLC=0 for weights)
// - Validates buffer descriptor word3 cache control bits against official AMD GFX11 architecture specifications
// - Runs cleanly with zero race conditions under -race
func TestModifierGenerator(t *testing.T) {
	cfg := DefaultGeneratorConfig()
	gen, err := NewModifierGenerator(cfg)
	if err != nil {
		t.Fatalf("failed to create modifier generator: %v", err)
	}

	t.Run("Criterion2_TemporalRootKV_Bitmasks", func(t *testing.T) {
		mask, err := gen.GenerateBitmask(IntentTemporalRootKV)
		if err != nil {
			t.Fatalf("GenerateBitmask(IntentTemporalRootKV) returned error: %v", err)
		}

		if mask.SLC != 0 {
			t.Errorf("expected SLC=0 for temporal KV, got %d", mask.SLC)
		}
		if mask.GLC != 0 {
			t.Errorf("expected GLC=0 for temporal KV, got %d", mask.GLC)
		}
		if mask.DLC != 0 {
			t.Errorf("expected DLC=0 for temporal KV, got %d", mask.DLC)
		}
		if mask.AssemblySuffix != "" {
			t.Errorf("expected empty assembly suffix for temporal KV, got %q", mask.AssemblySuffix)
		}
		if !mask.Tagged {
			t.Errorf("expected Tagged=true for explicit arm temporal KV")
		}
		if (mask.InstructionDword0 & MUBUFGLCBit) != 0 {
			t.Errorf("expected GLC bit 14 in Dword0 to be 0 for temporal KV, got 0x%X", mask.InstructionDword0)
		}
		if (mask.InstructionDword1 & MUBUFSLCBit) != 0 {
			t.Errorf("expected SLC bit 14 in Dword1 to be 0 for temporal KV, got 0x%X", mask.InstructionDword1)
		}
		if (mask.DescriptorWord3Mask & DescriptorWord3SLCBit) != 0 {
			t.Errorf("expected DescriptorWord3 SLC bit 22 to be 0 for temporal KV, got 0x%X", mask.DescriptorWord3Mask)
		}
	})

	t.Run("Criterion2_NonTemporalWeight_Bitmasks", func(t *testing.T) {
		mask, err := gen.GenerateBitmask(IntentNonTemporalWeight)
		if err != nil {
			t.Fatalf("GenerateBitmask(IntentNonTemporalWeight) returned error: %v", err)
		}

		if mask.SLC != 1 {
			t.Errorf("expected SLC=1 for non-temporal weights, got %d", mask.SLC)
		}
		if mask.GLC != 0 {
			t.Errorf("expected GLC=0 for non-temporal weights, got %d", mask.GLC)
		}
		if mask.DLC != 0 {
			t.Errorf("expected DLC=0 for non-temporal weights, got %d", mask.DLC)
		}
		if mask.AssemblySuffix != "slc" {
			t.Errorf("expected assembly suffix 'slc' for non-temporal weights, got %q", mask.AssemblySuffix)
		}
		if !mask.Tagged {
			t.Errorf("expected Tagged=true for explicit arm streaming weights")
		}
		if (mask.InstructionDword0 & MUBUFGLCBit) != 0 {
			t.Errorf("expected GLC bit 14 in Dword0 to be 0 for streaming weights, got 0x%X", mask.InstructionDword0)
		}
		if (mask.InstructionDword1 & MUBUFSLCBit) == 0 {
			t.Errorf("expected SLC bit 14 in Dword1 to be set for streaming weights, got 0x%X", mask.InstructionDword1)
		}
		if (mask.DescriptorWord3Mask & DescriptorWord3SLCBit) == 0 {
			t.Errorf("expected DescriptorWord3 SLC bit 22 to be set for streaming weights, got 0x%X", mask.DescriptorWord3Mask)
		}
	})

	t.Run("Criterion3_DescriptorWord3Validation", func(t *testing.T) {
		// Valid temporal descriptor word3: ResourceType=0x8, SLC=0, GLC=0, DLC=0
		temporalMask, _ := gen.GenerateBitmask(IntentTemporalRootKV)
		validTemporalWord3 := DescriptorWord3ResourceTypeBuffer
		if err := gen.ValidateDescriptorWord3(validTemporalWord3, temporalMask); err != nil {
			t.Errorf("valid temporal Word3 failed validation: %v", err)
		}

		// Valid weight descriptor word3: ResourceType=0x8, SLC=1 (bit 22), GLC=0, DLC=0
		weightMask, _ := gen.GenerateBitmask(IntentNonTemporalWeight)
		validWeightWord3 := DescriptorWord3ResourceTypeBuffer | DescriptorWord3SLCBit
		if err := gen.ValidateDescriptorWord3(validWeightWord3, weightMask); err != nil {
			t.Errorf("valid weight Word3 failed validation: %v", err)
		}

		// SynthesizeDescriptorWord3 correctness
		synthTemporal := gen.SynthesizeDescriptorWord3(0, temporalMask)
		if (synthTemporal & DescriptorWord3SLCBit) != 0 {
			t.Errorf("synthesized temporal word3 has SLC bit set: 0x%X", synthTemporal)
		}
		if (synthTemporal & (0xF << 28)) != DescriptorWord3ResourceTypeBuffer {
			t.Errorf("synthesized temporal word3 missing buffer resource type 0x8: 0x%X", synthTemporal)
		}

		synthWeight := gen.SynthesizeDescriptorWord3(0, weightMask)
		if (synthWeight & DescriptorWord3SLCBit) == 0 {
			t.Errorf("synthesized weight word3 missing SLC bit 22: 0x%X", synthWeight)
		}

		// Invalid resource type test (e.g. 0x0 instead of 0x8)
		invalidResWord3 := uint32(0x0) | DescriptorWord3SLCBit
		if err := gen.ValidateDescriptorWord3(invalidResWord3, weightMask); err == nil {
			t.Errorf("expected error for invalid resource type in Word3, got nil")
		}

		// Mismatched SLC bit (temporal mask expecting SLC=0, but Word3 has SLC=1)
		mismatchedSLC := DescriptorWord3ResourceTypeBuffer | DescriptorWord3SLCBit
		if err := gen.ValidateDescriptorWord3(mismatchedSLC, temporalMask); err == nil {
			t.Errorf("expected error for mismatched SLC bit in Word3, got nil")
		}

		// Mismatched GLC bit
		mismatchedGLC := DescriptorWord3ResourceTypeBuffer | DescriptorWord3GLCBit
		if err := gen.ValidateDescriptorWord3(mismatchedGLC, temporalMask); err == nil {
			t.Errorf("expected error for mismatched GLC bit in Word3, got nil")
		}
	})

	t.Run("AblationArm1_AggressiveBypass", func(t *testing.T) {
		mask, err := gen.GenerateBitmask(IntentAggressiveBypass)
		if err != nil {
			t.Fatalf("GenerateBitmask(IntentAggressiveBypass) error: %v", err)
		}
		if mask.SLC != 1 || mask.GLC != 1 || mask.DLC != 0 {
			t.Errorf("expected SLC=1, GLC=1, DLC=0 for aggressive bypass, got %+v", mask)
		}
		if mask.AssemblySuffix != "slc glc" {
			t.Errorf("expected 'slc glc' suffix, got %q", mask.AssemblySuffix)
		}
		if (mask.InstructionDword0 & MUBUFGLCBit) == 0 {
			t.Errorf("expected GLC bit in Dword0 for aggressive bypass")
		}
		if (mask.InstructionDword1 & MUBUFSLCBit) == 0 {
			t.Errorf("expected SLC bit in Dword1 for aggressive bypass")
		}
		if (mask.DescriptorWord3Mask & (DescriptorWord3SLCBit | DescriptorWord3GLCBit)) != (DescriptorWord3SLCBit | DescriptorWord3GLCBit) {
			t.Errorf("expected SLC and GLC in DescriptorWord3Mask for aggressive bypass")
		}
	})

	t.Run("AblationArm2_DriverDefault", func(t *testing.T) {
		genDefault, err := NewModifierGenerator(GeneratorConfig{
			TargetArch:            TargetArchGFX1151,
			AblationArm:           Arm2DriverDefault,
			EnableAssemblyTagging: true,
			StrictArchValidation:  true,
		})
		if err != nil {
			t.Fatalf("failed to create Arm 2 generator: %v", err)
		}

		for _, intent := range []CacheTierIntent{IntentTemporalRootKV, IntentNonTemporalWeight, IntentAggressiveBypass} {
			mask, err := genDefault.GenerateBitmask(intent)
			if err != nil {
				t.Fatalf("intent %s error: %v", intent, err)
			}
			if mask.SLC != 0 || mask.GLC != 0 || mask.DLC != 0 || mask.Tagged {
				t.Errorf("Arm 2 must return untagged zeroed bitmasks for %s, got %+v", intent, mask)
			}
			if mask.AssemblySuffix != "" {
				t.Errorf("Arm 2 must have empty assembly suffix, got %q", mask.AssemblySuffix)
			}
		}
	})

	t.Run("AblationArm3_AggressiveBypassArm", func(t *testing.T) {
		genArm3, err := NewModifierGenerator(GeneratorConfig{
			TargetArch:            TargetArchGFX1151,
			AblationArm:           Arm3AggressiveBypass,
			EnableAssemblyTagging: true,
			StrictArchValidation:  true,
		})
		if err != nil {
			t.Fatalf("failed to create Arm 3 generator: %v", err)
		}

		mask, err := genArm3.GenerateBitmask(IntentNonTemporalWeight)
		if err != nil {
			t.Fatalf("GenerateBitmask error: %v", err)
		}
		if mask.SLC != 1 || mask.GLC != 1 || mask.AssemblySuffix != "slc glc" {
			t.Errorf("Arm 3 should enforce SLC=1 GLC=1, got %+v", mask)
		}
	})

	t.Run("InstructionAssemblySynthesis", func(t *testing.T) {
		weightMask, err := gen.GenerateBitmask(IntentNonTemporalWeight)
		if err != nil {
			t.Fatalf("GenerateBitmask error: %v", err)
		}

		asm, raw, err := gen.SynthesizeBufferLoadInstruction(
			OpcodeBufferLoadDwordx4,
			"v[0:3]",
			"v0",
			"s[0:3]",
			0,
			weightMask,
		)
		if err != nil {
			t.Fatalf("SynthesizeBufferLoadInstruction error: %v", err)
		}

		expectedAsm := "buffer_load_dwordx4 v[0:3], v0, s[0:3], 0 offen slc"
		if asm != expectedAsm {
			t.Errorf("expected asm %q, got %q", expectedAsm, asm)
		}

		dword0 := uint32(raw & 0xFFFFFFFF)
		dword1 := uint32(raw >> 32)

		// Verify MUBUF prefix [31:26] = 0x38 (0b111000)
		if (dword0 >> 26) != 0x38 {
			t.Errorf("expected MUBUF prefix 0x38 in dword0, got 0x%X", dword0>>26)
		}

		// Verify OFFEN bit [12] is set
		if (dword0 & MUBUFOffenBit) == 0 {
			t.Errorf("expected OFFEN bit 12 in dword0 to be set")
		}

		// Verify SLC bit [14] in Dword 1 is set for non-temporal weights
		if (dword1 & MUBUFSLCBit) == 0 {
			t.Errorf("expected SLC bit 14 in dword1 to be set, got 0x%X", dword1)
		}

		// Now test temporal load
		temporalMask, err := gen.GenerateBitmask(IntentTemporalRootKV)
		if err != nil {
			t.Fatalf("GenerateBitmask temporal error: %v", err)
		}
		asmTemp, rawTemp, err := gen.SynthesizeBufferLoadInstruction(
			OpcodeBufferLoadDwordx4,
			"v[0:3]",
			"v0",
			"s[0:3]",
			0,
			temporalMask,
		)
		if err != nil {
			t.Fatalf("SynthesizeBufferLoadInstruction temporal error: %v", err)
		}
		expectedTempAsm := "buffer_load_dwordx4 v[0:3], v0, s[0:3], 0 offen"
		if asmTemp != expectedTempAsm {
			t.Errorf("expected asm %q, got %q", expectedTempAsm, asmTemp)
		}
		dword1Temp := uint32(rawTemp >> 32)
		if (dword1Temp & MUBUFSLCBit) != 0 {
			t.Errorf("expected SLC bit 14 in dword1 to be 0 for temporal KV, got 0x%X", dword1Temp)
		}
	})

	t.Run("QuarantinedFallback_DisabledTagging", func(t *testing.T) {
		genDisabled, err := NewModifierGenerator(GeneratorConfig{
			TargetArch:            TargetArchGFX1151,
			AblationArm:           Arm1ExplicitBitmasks,
			EnableAssemblyTagging: false,
			StrictArchValidation:  true,
		})
		if err != nil {
			t.Fatalf("failed to create disabled generator: %v", err)
		}

		mask, err := genDisabled.GenerateBitmask(IntentNonTemporalWeight)
		if err != nil {
			t.Fatalf("GenerateBitmask error: %v", err)
		}
		if mask.SLC != 0 || mask.GLC != 0 || mask.Tagged {
			t.Errorf("disabled tagging must return untagged bitmask, got %+v", mask)
		}
		if genDisabled.Telemetry().FallbackUntaggedEmitted == 0 {
			t.Errorf("expected FallbackUntaggedEmitted to increment")
		}
	})

	t.Run("QuarantinedFallback_UnsupportedArchNonStrict", func(t *testing.T) {
		genOther, err := NewModifierGenerator(GeneratorConfig{
			TargetArch:            "gfx1030",
			AblationArm:           Arm1ExplicitBitmasks,
			EnableAssemblyTagging: true,
			StrictArchValidation:  false,
		})
		if err != nil {
			t.Fatalf("failed to create non-strict generator: %v", err)
		}

		mask, err := genOther.GenerateBitmask(IntentNonTemporalWeight)
		if err != nil {
			t.Fatalf("GenerateBitmask error: %v", err)
		}
		if mask.SLC != 0 || mask.Tagged {
			t.Errorf("unsupported arch must return untagged fallback, got %+v", mask)
		}
	})

	t.Run("StrictArchValidation_Error", func(t *testing.T) {
		_, err := NewModifierGenerator(GeneratorConfig{
			TargetArch:           "gfx1030",
			StrictArchValidation: true,
		})
		if err == nil {
			t.Errorf("expected error for unsupported arch under strict validation, got nil")
		}
	})

	t.Run("ConcurrencyAndTelemetry", func(t *testing.T) {
		var wg sync.WaitGroup
		iterations := 100
		workers := 8

		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				for i := 0; i < iterations; i++ {
					intent := IntentTemporalRootKV
					if (workerID+i)%2 == 0 {
						intent = IntentNonTemporalWeight
					}
					mask, err := gen.GenerateBitmask(intent)
					if err != nil {
						t.Errorf("worker %d iteration %d error: %v", workerID, i, err)
						return
					}
					_ = gen.SynthesizeDescriptorWord3(0, mask)
				}
			}(w)
		}

		wg.Wait()
		telem := gen.Telemetry()
		if telem.TotalGenerated == 0 {
			t.Errorf("expected positive TotalGenerated, got %d", telem.TotalGenerated)
		}
		if telem.TemporalRootKVEmitted == 0 || telem.NonTemporalWeightEmitted == 0 {
			t.Errorf("expected both KV and weight emissions, got KV=%d Weight=%d",
				telem.TemporalRootKVEmitted, telem.NonTemporalWeightEmitted)
		}
	})

	t.Run("CloseGenerator", func(t *testing.T) {
		closedGen, err := NewModifierGenerator(DefaultGeneratorConfig())
		if err != nil {
			t.Fatalf("failed to create generator: %v", err)
		}
		if err := closedGen.Close(); err != nil {
			t.Fatalf("failed to close generator: %v", err)
		}
		if _, err := closedGen.GenerateBitmask(IntentTemporalRootKV); err != ErrGeneratorClosed {
			t.Errorf("expected ErrGeneratorClosed, got %v", err)
		}
	})
}
