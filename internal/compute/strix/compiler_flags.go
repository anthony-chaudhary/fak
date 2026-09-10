// Package strix provides custom Clang/LLVM GFX1151 target configuration,
// compiler argument synthesis, ELF/ISA assembly metadata parsing, and active
// CU occupancy analysis for AMD Strix Halo (Ryzen AI Max+ 395).
package strix

import (
	"fmt"
	"strings"
)

const (
	// TargetTripleAMDGCN is the standard LLVM target triple for AMD HSA code objects.
	TargetTripleAMDGCN = "amdgcn-amd-amdhsa"

	// FlagTarget is the Clang flag specifying target triple.
	FlagTarget = "-target"

	// FlagMCPU is the Clang flag setting the target microarchitecture to gfx1151.
	FlagMCPU = "-mcpu=gfx1151"

	// FlagWavefrontSize32 forces Wave32 execution mode on GFX1151 (-mwavefrontsize64=0).
	FlagWavefrontSize32 = "-mwavefrontsize64=0"

	// FlagWavefrontSize64 specifies legacy Wave64 execution mode (-mwavefrontsize64=1).
	FlagWavefrontSize64 = "-mwavefrontsize64=1"

	// FlagEnableFlatScratchLLVM enables 64-bit flat scratch addressing in LLVM backend.
	FlagEnableFlatScratchLLVM = "-amdgpu-enable-flat-scratch=1"

	// StrixHaloTotalCUs is the physical count of Compute Units on Radeon 8060S (40 CUs).
	StrixHaloTotalCUs = 40

	// StrixHaloSIMDsPerCU is the count of SIMD32 execution units per Compute Unit (2).
	StrixHaloSIMDsPerCU = 2

	// StrixHaloTotalSIMD32Units is the total count of SIMD32 engines across all 40 CUs (80).
	StrixHaloTotalSIMD32Units = StrixHaloTotalCUs * StrixHaloSIMDsPerCU

	// MaxVGPRsPerWave32Target is the target VGPR allocation limit per thread (<= 48)
	// to completely eliminate scratch memory spilling and maximize wavefront occupancy.
	MaxVGPRsPerWave32Target = 48

	// MaxVGPRsPerSIMD is the physical register file capacity per SIMD unit on GFX1151 (512 VGPRs).
	MaxVGPRsPerSIMD = 512

	// MaxWavefrontsPerSIMD is the architectural wavefront capacity per SIMD unit on GFX1151 (8 waves).
	MaxWavefrontsPerSIMD = 8

	// NativeWavefrontSizeWave32 is the thread width of a native Wave32 wavefront.
	NativeWavefrontSizeWave32 = 32

	// LegacyWavefrontSizeWave64 is the thread width of a legacy Wave64 wavefront.
	LegacyWavefrontSizeWave64 = 64
)

// TargetProfile specifies the microarchitecture and toolchain profile for GFX1151.
type TargetProfile struct {
	TargetArch          string `json:"target_arch"`
	TargetTriple        string `json:"target_triple"`
	WavefrontSize       int    `json:"wavefront_size"`
	EnableFlatScratch   bool   `json:"enable_flat_scratch"`
	TotalComputeUnits   int    `json:"total_compute_units"`
	MaxVGPRPerThread    int    `json:"max_vgpr_per_thread"`
	CodeObjectVersion   int    `json:"code_object_version"`
	DefaultOptimization string `json:"default_optimization"`
}

// DefaultGFX1151TargetProfile returns the canonical target configuration profile
// enforcing native Wave32 execution and 64-bit flat scratch addressing.
func DefaultGFX1151TargetProfile() TargetProfile {
	return TargetProfile{
		TargetArch:          TargetArchGFX1151,
		TargetTriple:        TargetTripleAMDGCN,
		WavefrontSize:       NativeWavefrontSizeWave32,
		EnableFlatScratch:   true,
		TotalComputeUnits:   StrixHaloTotalCUs,
		MaxVGPRPerThread:    MaxVGPRsPerWave32Target,
		CodeObjectVersion:   5,
		DefaultOptimization: "-O3",
	}
}

// CompileOptions defines configurable options passed into compiler flag generation.
type CompileOptions struct {
	TargetArch        string            `json:"target_arch,omitempty"`
	TargetTriple      string            `json:"target_triple,omitempty"`
	WavefrontSize     int               `json:"wavefront_size,omitempty"`
	EnableFlatScratch bool              `json:"enable_flat_scratch"`
	OptimizationLevel string            `json:"optimization_level,omitempty"`
	CodeObjectVersion int               `json:"code_object_version,omitempty"`
	IncludeDirs       []string          `json:"include_dirs,omitempty"`
	Defines           map[string]string `json:"defines,omitempty"`
	ExtraFlags        []string          `json:"extra_flags,omitempty"`
	EmitLLVMIR        bool              `json:"emit_llvm_ir,omitempty"`
	EmitAssembly      bool              `json:"emit_assembly,omitempty"`
}

// BuildCompilerFlags synthesizes the exact Clang/LLVM invocation argument slice
// adhering to the GFX1151 target configuration requirements.
func BuildCompilerFlags(opts CompileOptions) []string {
	flags := make([]string, 0, 16)

	// Target triple: -target amdgcn-amd-amdhsa
	triple := opts.TargetTriple
	if triple == "" {
		triple = TargetTripleAMDGCN
	}
	flags = append(flags, FlagTarget, triple)

	// Architecture: -mcpu=gfx1151
	arch := opts.TargetArch
	if arch == "" {
		arch = TargetArchGFX1151
	}
	flags = append(flags, fmt.Sprintf("-mcpu=%s", arch))

	// Wavefront size: Wave32 (-mwavefrontsize64=0) vs Wave64 (-mwavefrontsize64=1)
	if opts.WavefrontSize == LegacyWavefrontSizeWave64 {
		flags = append(flags, FlagWavefrontSize64)
	} else {
		// Default to Wave32 for GFX1151
		flags = append(flags, FlagWavefrontSize32)
	}

	// 64-bit Flat scratch addressing: -mllvm -amdgpu-enable-flat-scratch=1
	if opts.EnableFlatScratch {
		flags = append(flags, "-mllvm", FlagEnableFlatScratchLLVM)
	}

	// Code object version
	cov := opts.CodeObjectVersion
	if cov <= 0 {
		cov = 5
	}
	flags = append(flags, fmt.Sprintf("-mcode-object-version=%d", cov))

	// Optimization level
	opt := opts.OptimizationLevel
	if opt == "" {
		opt = "-O3"
	}
	if !strings.HasPrefix(opt, "-O") {
		opt = "-O" + opt
	}
	flags = append(flags, opt)

	// Emit options
	if opts.EmitAssembly {
		flags = append(flags, "-S")
	} else if opts.EmitLLVMIR {
		flags = append(flags, "-emit-llvm", "-S")
	}

	// Include directories
	for _, inc := range opts.IncludeDirs {
		if inc != "" {
			flags = append(flags, "-I"+inc)
		}
	}

	// Macro definitions
	for k, v := range opts.Defines {
		if v == "" {
			flags = append(flags, "-D"+k)
		} else {
			flags = append(flags, fmt.Sprintf("-D%s=%s", k, v))
		}
	}

	// Extra user flags
	flags = append(flags, opts.ExtraFlags...)

	return flags
}

// OccupancyResult captures the simulated or measured hardware occupancy across
// the 40 Compute Units on Strix Halo for given register and wavefront parameters.
type OccupancyResult struct {
	TargetArch             string  `json:"target_arch"`
	TotalComputeUnits      int     `json:"total_compute_units"`
	WavefrontSize          int     `json:"wavefront_size"`
	VGPRPerThread          int     `json:"vgpr_per_thread"`
	SGPRPerWave            int     `json:"sgpr_per_wave"`
	ScratchSpillBytes      int     `json:"scratch_spill_bytes"`
	WavefrontsPerSIMD      int     `json:"wavefronts_per_simd"`
	MaxWavefrontsPerSIMD   int     `json:"max_wavefronts_per_simd"`
	ActiveWavefrontsTotal  int     `json:"active_wavefronts_total"`
	ActiveComputeUnits     float64 `json:"active_compute_units"`
	OccupancyPercentage    float64 `json:"occupancy_percentage"`
	SpillPenaltyDetected   bool    `json:"spill_penalty_detected"`
	OccupancyLimiterReason string  `json:"occupancy_limiter_reason"`
}

// CalculateOccupancy computes theoretical wavefront occupancy across all 40 CUs
// of Strix Halo given register allocations and wavefront mode.
func CalculateOccupancy(vgprPerThread, sgprPerWave, scratchSpillBytes, waveSize int) OccupancyResult {
	if waveSize != LegacyWavefrontSizeWave64 {
		waveSize = NativeWavefrontSizeWave32
	}

	res := OccupancyResult{
		TargetArch:           TargetArchGFX1151,
		TotalComputeUnits:    StrixHaloTotalCUs,
		WavefrontSize:        waveSize,
		VGPRPerThread:        vgprPerThread,
		SGPRPerWave:          sgprPerWave,
		ScratchSpillBytes:    scratchSpillBytes,
		MaxWavefrontsPerSIMD: MaxWavefrontsPerSIMD,
	}

	// In GFX1151:
	// Each SIMD has 512 VGPR slots.
	// In Wave32 mode, 1 VGPR allocation = 32 registers.
	// In Wave64 mode, 1 VGPR allocation = 64 registers (doubles register consumption).
	// When scratch spilling occurs (scratchSpillBytes > 0), off-chip latency and
	// register pressure stall waves, heavily penalizing active occupancy.
	var wavesPerSIMD int
	if vgprPerThread <= 0 {
		vgprPerThread = 32
	}

	if waveSize == NativeWavefrontSizeWave32 {
		// Wave32: registers are granular to 8 or 4 VGPRs per wave allocation.
		// Maximum 8 waves per SIMD.
		// Limit based on VGPR file: MaxWavefronts = min(8, 512 / vgprPerThread)
		wavesByVGPR := MaxVGPRsPerSIMD / vgprPerThread
		if wavesByVGPR > MaxWavefrontsPerSIMD {
			wavesByVGPR = MaxWavefrontsPerSIMD
		}
		if wavesByVGPR < 1 {
			wavesByVGPR = 1
		}
		wavesPerSIMD = wavesByVGPR

		if scratchSpillBytes > 0 {
			res.SpillPenaltyDetected = true
			res.OccupancyLimiterReason = "scratch memory spilling active"
			// Scratch spilling reduces effective occupancy by ~50%
			if wavesPerSIMD > 2 {
				wavesPerSIMD = 2
			}
		} else if vgprPerThread <= MaxVGPRsPerWave32Target {
			res.OccupancyLimiterReason = "unconstrained Wave32 occupancy"
		} else {
			res.OccupancyLimiterReason = fmt.Sprintf("VGPR budget constrained (%d > %d)", vgprPerThread, MaxVGPRsPerWave32Target)
		}
	} else {
		// Wave64 mode:
		// Double register footprint per wavefront. Each thread still consumes vgprPerThread,
		// but wave has 64 threads, requiring 2 SIMD physical slots or double register allocations.
		// On GFX1151, Wave64 cuts waves per SIMD to at most 4.
		wavesByVGPR := (MaxVGPRsPerSIMD / 2) / vgprPerThread
		if wavesByVGPR > 4 {
			wavesByVGPR = 4
		}
		if wavesByVGPR < 1 {
			wavesByVGPR = 1
		}
		wavesPerSIMD = wavesByVGPR

		if scratchSpillBytes > 0 || vgprPerThread > MaxVGPRsPerWave32Target {
			res.SpillPenaltyDetected = true
			res.OccupancyLimiterReason = "Wave64 register pressure & scratch spills"
			// Wave64 with register pressure collapses to 1-2 waves per SIMD
			wavesPerSIMD = 1
		} else {
			res.OccupancyLimiterReason = "Wave64 mode wavefront limit"
		}
	}

	res.WavefrontsPerSIMD = wavesPerSIMD
	res.ActiveWavefrontsTotal = wavesPerSIMD * StrixHaloTotalSIMD32Units
	maxTotalWavefronts := MaxWavefrontsPerSIMD * StrixHaloTotalSIMD32Units
	res.OccupancyPercentage = float64(res.ActiveWavefrontsTotal) / float64(maxTotalWavefronts) * 100.0

	// Calculate equivalent active compute units:
	// Full theoretical occupancy (8 waves/SIMD in Wave32) = 40.0 active CUs.
	// When throttled, active CUs scale down proportionally to 18 CUs or lower.
	res.ActiveComputeUnits = float64(StrixHaloTotalCUs) * (float64(wavesPerSIMD) / float64(MaxWavefrontsPerSIMD))
	if res.ActiveComputeUnits < 5.0 {
		res.ActiveComputeUnits = 5.0
	}

	return res
}
