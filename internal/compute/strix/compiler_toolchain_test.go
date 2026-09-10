package strix

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestToolchainGFX1151Wave32Flags serves as the verifiable witness for Ticket #632,
// asserting that compiler flags strictly enforce Wave32 execution mode (-mwavefrontsize64=0),
// the GFX1151 target microarchitecture, AMDGCN triple, and 64-bit flat scratch addressing.
func TestToolchainGFX1151Wave32Flags(t *testing.T) {
	cfg := DefaultToolchainConfig()
	tc, err := NewToolchain(cfg)
	if err != nil {
		t.Fatalf("failed to initialize toolchain: %v", err)
	}

	opts := CompileOptions{
		TargetArch:        TargetArchGFX1151,
		TargetTriple:      TargetTripleAMDGCN,
		WavefrontSize:     NativeWavefrontSizeWave32,
		EnableFlatScratch: true,
		OptimizationLevel: "-O3",
	}

	flags := tc.GenerateFlags(opts)

	// Invariant 1: Must strictly include -target amdgcn-amd-amdhsa
	hasTarget := false
	for i, flag := range flags {
		if flag == FlagTarget && i+1 < len(flags) && flags[i+1] == TargetTripleAMDGCN {
			hasTarget = true
			break
		}
	}
	if !hasTarget {
		t.Errorf("expected %s %s in flags, got: %v", FlagTarget, TargetTripleAMDGCN, flags)
	}

	// Invariant 2: Must strictly include -mcpu=gfx1151
	hasMCPU := false
	for _, flag := range flags {
		if flag == FlagMCPU {
			hasMCPU = true
			break
		}
	}
	if !hasMCPU {
		t.Errorf("expected %s in flags, got: %v", FlagMCPU, flags)
	}

	// Invariant 3: Must strictly include -mwavefrontsize64=0 (Wave32 native)
	hasWave32 := false
	for _, flag := range flags {
		if flag == FlagWavefrontSize32 {
			hasWave32 = true
			break
		}
	}
	if !hasWave32 {
		t.Errorf("expected %s (Wave32) in flags, got: %v", FlagWavefrontSize32, flags)
	}

	// Invariant 4: Must strictly include -mllvm -amdgpu-enable-flat-scratch=1
	hasFlatScratch := false
	for i, flag := range flags {
		if flag == "-mllvm" && i+1 < len(flags) && flags[i+1] == FlagEnableFlatScratchLLVM {
			hasFlatScratch = true
			break
		}
	}
	if !hasFlatScratch {
		t.Errorf("expected -mllvm %s in flags, got: %v", FlagEnableFlatScratchLLVM, flags)
	}

	// Execute compilation pass (dry-run fallback verification)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	res, err := tc.Compile(ctx, CompileRequest{
		SourceCode: "// Strix Halo Wave32 WMMA kernel",
		Options:    opts,
		DryRun:     true,
	})
	if err != nil {
		t.Fatalf("compilation request failed: %v", err)
	}

	if !res.Success {
		t.Errorf("expected compilation success, got false")
	}
	if !res.Wave32Verified {
		t.Errorf("expected Wave32 verified, got false")
	}
	if !res.ZeroSpillVerified {
		t.Errorf("expected zero spill verified, got false")
	}
	if res.Metadata.VGPRCount > MaxVGPRsPerWave32Target {
		t.Errorf("expected VGPR count <= %d, got %d", MaxVGPRsPerWave32Target, res.Metadata.VGPRCount)
	}
	if res.Metadata.ScratchSpillBytes != 0 {
		t.Errorf("expected 0 scratch spill bytes, got %d", res.Metadata.ScratchSpillBytes)
	}
	if res.Occupancy.ActiveComputeUnits < 38.0 {
		t.Errorf("expected active compute units >= 38.0 (targeting 40 CUs), got %.1f", res.Occupancy.ActiveComputeUnits)
	}

	t.Logf("Compiled Wave32 kernel report:\n%s", res.AuditReport)
}

func TestToolchainELFMetadataParser(t *testing.T) {
	tc, err := NewToolchain(DefaultToolchainConfig())
	if err != nil {
		t.Fatalf("failed to initialize toolchain: %v", err)
	}

	// Valid Wave32 assembly without scratch spills
	cleanASM := `
	.amdhsa_kernel clean_kernel
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 42
		.amdhsa_next_free_sgpr 32
		.amdhsa_reserve_flat_scratch 1
		.amdhsa_private_segment_fixed_size 0
	.end_amdhsa_kernel
	v_wmma_f32_16x16x16_bf16 v[16:23], v[4:7], v[8:11], v[16:23]
	s_endpgm
`
	meta, err := tc.ParseISAAssembly(cleanASM)
	if err != nil {
		t.Fatalf("unexpected error parsing clean Wave32 assembly: %v", err)
	}
	if meta.WavefrontSize != 32 {
		t.Errorf("expected WavefrontSize 32, got %d", meta.WavefrontSize)
	}
	if meta.VGPRCount != 42 {
		t.Errorf("expected VGPRCount 42, got %d", meta.VGPRCount)
	}
	if meta.ScratchSpillBytes != 0 {
		t.Errorf("expected 0 spill bytes, got %d", meta.ScratchSpillBytes)
	}

	// Spilling assembly (should fail validation)
	spillingASM := `
	.amdhsa_kernel spill_kernel
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 46
		.amdhsa_private_segment_fixed_size 64
	.end_amdhsa_kernel
	s_scratch_load_dwordx4 s[0:3], s[4:5], 0x0
	s_endpgm
`
	_, err = tc.ParseISAAssembly(spillingASM)
	if err == nil {
		t.Fatalf("expected error on scratch memory spill, got nil")
	}
	if !strings.Contains(err.Error(), "scratch memory spilling detected") {
		t.Errorf("expected scratch memory spilling error, got: %v", err)
	}

	// Excessive VGPR assembly (> 48)
	highVGPRASM := `
	.amdhsa_kernel high_vgpr_kernel
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 64
		.amdhsa_private_segment_fixed_size 0
	.end_amdhsa_kernel
	s_endpgm
`
	_, err = tc.ParseISAAssembly(highVGPRASM)
	if err == nil {
		t.Fatalf("expected error on excessive VGPR count (> 48), got nil")
	}
	if !strings.Contains(err.Error(), "VGPR allocation exceeds 48") {
		t.Errorf("expected VGPR limit error, got: %v", err)
	}

	// Wave64 assembly
	wave64ASM := `
	.amdhsa_kernel wave64_kernel
		.amdhsa_wavefront_size32 0
		.amdhsa_next_free_vgpr 38
		.amdhsa_private_segment_fixed_size 0
	.end_amdhsa_kernel
	s_endpgm
`
	_, err = tc.ParseISAAssembly(wave64ASM)
	if err == nil {
		t.Fatalf("expected error on Wave64 size mismatch, got nil")
	}
	if !strings.Contains(err.Error(), "wavefront size must be 32") {
		t.Errorf("expected wavefront size mismatch error, got: %v", err)
	}
}

func TestToolchainOccupancyCalculation(t *testing.T) {
	// Case 1: Ideal Wave32 with 38 VGPRs and 0 spills -> full 40 CUs
	occWave32 := CalculateOccupancy(38, 32, 0, NativeWavefrontSizeWave32)
	if occWave32.ActiveComputeUnits != float64(StrixHaloTotalCUs) {
		t.Errorf("expected 40.0 active CUs for Wave32 with 38 VGPRs, got %.1f", occWave32.ActiveComputeUnits)
	}
	if occWave32.WavefrontsPerSIMD != 8 {
		t.Errorf("expected 8 wavefronts per SIMD, got %d", occWave32.WavefrontsPerSIMD)
	}
	if occWave32.OccupancyPercentage != 100.0 {
		t.Errorf("expected 100%% occupancy, got %.1f%%", occWave32.OccupancyPercentage)
	}

	// Case 2: Legacy Wave64 with 72 VGPRs and scratch spilling -> collapses to ~18 CUs or lower
	occWave64 := CalculateOccupancy(72, 32, 64, LegacyWavefrontSizeWave64)
	if occWave64.ActiveComputeUnits > 18.0 {
		t.Errorf("expected collapsed active CUs (<= 18.0) for Wave64 with scratch spills, got %.1f", occWave64.ActiveComputeUnits)
	}
	if !occWave64.SpillPenaltyDetected {
		t.Errorf("expected spill penalty detected, got false")
	}

	t.Logf("Wave32 vs Wave64 Occupancy: Wave32=%.1f CUs (%.1f%%) vs Wave64=%.1f CUs (%.1f%%)",
		occWave32.ActiveComputeUnits, occWave32.OccupancyPercentage,
		occWave64.ActiveComputeUnits, occWave64.OccupancyPercentage)
}

func TestToolchainConcurrentCompile(t *testing.T) {
	tc, err := NewToolchain(DefaultToolchainConfig())
	if err != nil {
		t.Fatalf("failed to initialize toolchain: %v", err)
	}

	const goroutines = 10
	var wg sync.WaitGroup
	wg.Add(goroutines)

	for i := 0; i < goroutines; i++ {
		go func(id int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()

			res, compileErr := tc.Compile(ctx, CompileRequest{
				SourceCode: "// Concurrent compilation test",
				Options: CompileOptions{
					TargetArch:        TargetArchGFX1151,
					TargetTriple:      TargetTripleAMDGCN,
					WavefrontSize:     NativeWavefrontSizeWave32,
					EnableFlatScratch: true,
				},
				DryRun: true,
			})
			if compileErr != nil {
				t.Errorf("goroutine %d compile failed: %v", id, compileErr)
				return
			}
			if !res.Success || !res.Wave32Verified {
				t.Errorf("goroutine %d unexpected result: %+v", id, res)
			}
		}(i)
	}

	wg.Wait()
}
