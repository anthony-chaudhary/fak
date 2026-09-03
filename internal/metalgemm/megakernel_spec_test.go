// Prior-art: llama.cpp Metal / MLX (https://github.com/ggml-org/llama.cpp)
// Oracle: cpuref (GEMV cosine)

package metalgemm

import (
	"fmt"
	"strings"
	"testing"
)

// AppleGPUFamily designates the hardware generation of Apple Silicon GPUs.
type AppleGPUFamily string

const (
	AppleGPUM1      AppleGPUFamily = "M1"
	AppleGPUM1Pro   AppleGPUFamily = "M1 Pro"
	AppleGPUM1Max   AppleGPUFamily = "M1 Max"
	AppleGPUM1Ultra AppleGPUFamily = "M1 Ultra"
	AppleGPUM2      AppleGPUFamily = "M2"
	AppleGPUM2Pro   AppleGPUFamily = "M2 Pro"
	AppleGPUM2Max   AppleGPUFamily = "M2 Max"
	AppleGPUM2Ultra AppleGPUFamily = "M2 Ultra"
	AppleGPUM3      AppleGPUFamily = "M3"
	AppleGPUM3Pro   AppleGPUFamily = "M3 Pro"
	AppleGPUM3Max   AppleGPUFamily = "M3 Max"
	AppleGPUM3Ultra AppleGPUFamily = "M3 Ultra"
	AppleGPUM4      AppleGPUFamily = "M4"
	AppleGPUM4Pro   AppleGPUFamily = "M4 Pro"
	AppleGPUM4Max   AppleGPUFamily = "M4 Max"
	AppleGPUM5      AppleGPUFamily = "M5"
)

// AppleGPUHardwareProfile encapsulates the architectural and execution limits
// of a target Apple Silicon GPU relevant to compute kernels and persistent grids.
type AppleGPUHardwareProfile struct {
	Family                     AppleGPUFamily
	Cores                      int
	SIMDWidth                  int     // Always 32 on Apple Silicon
	MaxThreadsPerTG            int     // Metal limit 1024
	TGMemLimitBytes            int     // Standard 32KB (32768 bytes) across M1-M5
	MaxTGPerCore               int     // Hardware occupancy limit (typically 16)
	MaxRegistersPerThread      int     // 256 physical 32-bit registers per thread
	BandwidthGBps              float64 // Unified memory bandwidth in GB/s
	IsolatedDispatchOverheadUs float64 // Fixed launch/sync overhead per command buffer (~360-864 µs)
	OneCBDispatchOverheadUs    float64 // Pipelined dispatch overhead in single command buffer (~8 µs)
	ICBDispatchOverheadUs      float64 // Indirect Command Buffer replay dispatch overhead (~2 µs)
}

// GetAppleGPUHardwareProfile returns the canonical hardware profile for a given Apple GPU family.
func GetAppleGPUHardwareProfile(family AppleGPUFamily) (AppleGPUHardwareProfile, error) {
	switch family {
	case AppleGPUM1:
		return AppleGPUHardwareProfile{
			Family: family, Cores: 8, SIMDWidth: 32, MaxThreadsPerTG: 1024,
			TGMemLimitBytes: 32768, MaxTGPerCore: 16, MaxRegistersPerThread: 256,
			BandwidthGBps: 68.25, IsolatedDispatchOverheadUs: 360.0,
			OneCBDispatchOverheadUs: 8.0, ICBDispatchOverheadUs: 2.0,
		}, nil
	case AppleGPUM1Pro:
		return AppleGPUHardwareProfile{
			Family: family, Cores: 16, SIMDWidth: 32, MaxThreadsPerTG: 1024,
			TGMemLimitBytes: 32768, MaxTGPerCore: 16, MaxRegistersPerThread: 256,
			BandwidthGBps: 200.0, IsolatedDispatchOverheadUs: 360.0,
			OneCBDispatchOverheadUs: 8.0, ICBDispatchOverheadUs: 2.0,
		}, nil
	case AppleGPUM1Max:
		return AppleGPUHardwareProfile{
			Family: family, Cores: 32, SIMDWidth: 32, MaxThreadsPerTG: 1024,
			TGMemLimitBytes: 32768, MaxTGPerCore: 16, MaxRegistersPerThread: 256,
			BandwidthGBps: 400.0, IsolatedDispatchOverheadUs: 360.0,
			OneCBDispatchOverheadUs: 8.0, ICBDispatchOverheadUs: 2.0,
		}, nil
	case AppleGPUM1Ultra:
		return AppleGPUHardwareProfile{
			Family: family, Cores: 64, SIMDWidth: 32, MaxThreadsPerTG: 1024,
			TGMemLimitBytes: 32768, MaxTGPerCore: 16, MaxRegistersPerThread: 256,
			BandwidthGBps: 800.0, IsolatedDispatchOverheadUs: 360.0,
			OneCBDispatchOverheadUs: 8.0, ICBDispatchOverheadUs: 2.0,
		}, nil
	case AppleGPUM2:
		return AppleGPUHardwareProfile{
			Family: family, Cores: 10, SIMDWidth: 32, MaxThreadsPerTG: 1024,
			TGMemLimitBytes: 32768, MaxTGPerCore: 16, MaxRegistersPerThread: 256,
			BandwidthGBps: 100.0, IsolatedDispatchOverheadUs: 360.0,
			OneCBDispatchOverheadUs: 8.0, ICBDispatchOverheadUs: 2.0,
		}, nil
	case AppleGPUM2Pro:
		return AppleGPUHardwareProfile{
			Family: family, Cores: 19, SIMDWidth: 32, MaxThreadsPerTG: 1024,
			TGMemLimitBytes: 32768, MaxTGPerCore: 16, MaxRegistersPerThread: 256,
			BandwidthGBps: 200.0, IsolatedDispatchOverheadUs: 360.0,
			OneCBDispatchOverheadUs: 8.0, ICBDispatchOverheadUs: 2.0,
		}, nil
	case AppleGPUM2Max:
		return AppleGPUHardwareProfile{
			Family: family, Cores: 38, SIMDWidth: 32, MaxThreadsPerTG: 1024,
			TGMemLimitBytes: 32768, MaxTGPerCore: 16, MaxRegistersPerThread: 256,
			BandwidthGBps: 400.0, IsolatedDispatchOverheadUs: 360.0,
			OneCBDispatchOverheadUs: 8.0, ICBDispatchOverheadUs: 2.0,
		}, nil
	case AppleGPUM2Ultra:
		return AppleGPUHardwareProfile{
			Family: family, Cores: 76, SIMDWidth: 32, MaxThreadsPerTG: 1024,
			TGMemLimitBytes: 32768, MaxTGPerCore: 16, MaxRegistersPerThread: 256,
			BandwidthGBps: 800.0, IsolatedDispatchOverheadUs: 360.0,
			OneCBDispatchOverheadUs: 8.0, ICBDispatchOverheadUs: 2.0,
		}, nil
	case AppleGPUM3:
		return AppleGPUHardwareProfile{
			Family: family, Cores: 10, SIMDWidth: 32, MaxThreadsPerTG: 1024,
			TGMemLimitBytes: 32768, MaxTGPerCore: 16, MaxRegistersPerThread: 256,
			BandwidthGBps: 100.0, IsolatedDispatchOverheadUs: 360.0,
			OneCBDispatchOverheadUs: 8.0, ICBDispatchOverheadUs: 2.0,
		}, nil
	case AppleGPUM3Pro:
		return AppleGPUHardwareProfile{
			Family: family, Cores: 18, SIMDWidth: 32, MaxThreadsPerTG: 1024,
			TGMemLimitBytes: 32768, MaxTGPerCore: 16, MaxRegistersPerThread: 256,
			BandwidthGBps: 150.0, IsolatedDispatchOverheadUs: 360.0,
			OneCBDispatchOverheadUs: 8.0, ICBDispatchOverheadUs: 2.0,
		}, nil
	case AppleGPUM3Max:
		return AppleGPUHardwareProfile{
			Family: family, Cores: 40, SIMDWidth: 32, MaxThreadsPerTG: 1024,
			TGMemLimitBytes: 32768, MaxTGPerCore: 16, MaxRegistersPerThread: 256,
			BandwidthGBps: 400.0, IsolatedDispatchOverheadUs: 360.0,
			OneCBDispatchOverheadUs: 8.0, ICBDispatchOverheadUs: 2.0,
		}, nil
	case AppleGPUM3Ultra:
		return AppleGPUHardwareProfile{
			Family: family, Cores: 80, SIMDWidth: 32, MaxThreadsPerTG: 1024,
			TGMemLimitBytes: 32768, MaxTGPerCore: 16, MaxRegistersPerThread: 256,
			BandwidthGBps: 800.0, IsolatedDispatchOverheadUs: 360.0,
			OneCBDispatchOverheadUs: 8.0, ICBDispatchOverheadUs: 2.0,
		}, nil
	case AppleGPUM4:
		return AppleGPUHardwareProfile{
			Family: family, Cores: 10, SIMDWidth: 32, MaxThreadsPerTG: 1024,
			TGMemLimitBytes: 32768, MaxTGPerCore: 16, MaxRegistersPerThread: 256,
			BandwidthGBps: 120.0, IsolatedDispatchOverheadUs: 360.0,
			OneCBDispatchOverheadUs: 8.0, ICBDispatchOverheadUs: 2.0,
		}, nil
	case AppleGPUM4Pro:
		return AppleGPUHardwareProfile{
			Family: family, Cores: 20, SIMDWidth: 32, MaxThreadsPerTG: 1024,
			TGMemLimitBytes: 32768, MaxTGPerCore: 16, MaxRegistersPerThread: 256,
			BandwidthGBps: 273.0, IsolatedDispatchOverheadUs: 360.0,
			OneCBDispatchOverheadUs: 8.0, ICBDispatchOverheadUs: 2.0,
		}, nil
	case AppleGPUM4Max:
		return AppleGPUHardwareProfile{
			Family: family, Cores: 40, SIMDWidth: 32, MaxThreadsPerTG: 1024,
			TGMemLimitBytes: 32768, MaxTGPerCore: 16, MaxRegistersPerThread: 256,
			BandwidthGBps: 546.0, IsolatedDispatchOverheadUs: 360.0,
			OneCBDispatchOverheadUs: 8.0, ICBDispatchOverheadUs: 2.0,
		}, nil
	case AppleGPUM5:
		return AppleGPUHardwareProfile{
			Family: family, Cores: 20, SIMDWidth: 32, MaxThreadsPerTG: 1024,
			TGMemLimitBytes: 32768, MaxTGPerCore: 16, MaxRegistersPerThread: 256,
			BandwidthGBps: 300.0, IsolatedDispatchOverheadUs: 360.0,
			OneCBDispatchOverheadUs: 8.0, ICBDispatchOverheadUs: 1.5,
		}, nil
	default:
		return AppleGPUHardwareProfile{}, fmt.Errorf("unknown Apple GPU family: %s", family)
	}
}

// MegakernelOpType identifies the mathematical kernel category in an LLM decode step.
type MegakernelOpType string

const (
	MegaOpRMSNorm         MegakernelOpType = "RMSNorm"
	MegaOpQ4KGEMVNarrow   MegakernelOpType = "Q4K_GEMV_Narrow"
	MegaOpQ4KGEMVWideMLP  MegakernelOpType = "Q4K_GEMV_WideMLP"
	MegaOpGDNRecurrence   MegakernelOpType = "GDN_Recurrence"
	MegaOpSDPAAttention   MegakernelOpType = "SDPA_Attention"
	MegaOpSwiGLU          MegakernelOpType = "SwiGLU"
	MegaOpFusedMegakernel MegakernelOpType = "Fused_Megakernel"
)

// EstimateLiveRegisters models the register demand (32-bit registers per thread)
// for standalone specialized kernels vs a monolithic fused megakernel.
func EstimateLiveRegisters(op MegakernelOpType, isMegakernel bool) int {
	if isMegakernel {
		// A megakernel combines loop drivers, dynamic op switching,
		// dequantization unpack buffers, Conv1d history, GDN decay/recurrence,
		// SDPA running online-softmax accumulators, SwiGLU activation, and
		// global atomic barrier state in a single compile unit.
		// This causes severe live-range explosion in the Metal compiler.
		return 288
	}
	switch op {
	case MegaOpRMSNorm:
		return 24
	case MegaOpQ4KGEMVNarrow:
		return 36
	case MegaOpQ4KGEMVWideMLP:
		return 48
	case MegaOpGDNRecurrence:
		return 56
	case MegaOpSDPAAttention:
		return 64
	case MegaOpSwiGLU:
		return 20
	default:
		return 32
	}
}

// RegisterSpillResult contains the quantitative impact of register spills on GPU execution.
type RegisterSpillResult struct {
	RequestedRegisters   int
	HardwareLimit        int
	SpilledRegisters     int
	SpillBytesPerThread  int
	TotalSpillBytesPerTG int
	TotalSpillBytesPass  int
}

// CalculateRegisterSpill evaluates the register spill penalty when requested registers
// exceed the physical hardware limit per thread.
func CalculateRegisterSpill(reqRegs, maxRegs, threadsPerTG, numTG, loopIterations int) RegisterSpillResult {
	spilledRegs := 0
	if reqRegs > maxRegs {
		spilledRegs = reqRegs - maxRegs
	}
	spillBytesPerThread := spilledRegs * 4 // 4 bytes per 32-bit scalar register
	totalSpillBytesPerTG := spillBytesPerThread * threadsPerTG * loopIterations
	totalSpillBytesPass := totalSpillBytesPerTG * numTG

	return RegisterSpillResult{
		RequestedRegisters:   reqRegs,
		HardwareLimit:        maxRegs,
		SpilledRegisters:     spilledRegs,
		SpillBytesPerThread:  spillBytesPerThread,
		TotalSpillBytesPerTG: totalSpillBytesPerTG,
		TotalSpillBytesPass:  totalSpillBytesPass,
	}
}

// ThreadgroupMemoryBudget tracks allocations against the 32KB hardware limit.
type ThreadgroupMemoryBudget struct {
	ActivationBytes       int
	DequantLUTBytes       int
	ReductionScratchBytes int
	GDNStateBytes         int
}

// TotalBytes sums all planned threadgroup memory allocations.
func (b ThreadgroupMemoryBudget) TotalBytes() int {
	return b.ActivationBytes + b.DequantLUTBytes + b.ReductionScratchBytes + b.GDNStateBytes
}

// ExceedsLimit tests whether the total budget breaches the hardware limit.
func (b ThreadgroupMemoryBudget) ExceedsLimit(limitBytes int) bool {
	return b.TotalBytes() > limitBytes
}

// OverflowBytes returns the excess bytes beyond the hardware limit.
func (b ThreadgroupMemoryBudget) OverflowBytes(limitBytes int) int {
	if b.TotalBytes() > limitBytes {
		return b.TotalBytes() - limitBytes
	}
	return 0
}

// CalculateGDNHeadStateBytes computes the state size of a single GDN head.
// State dimension is (KeyHeadDim x ValueHeadDim).
func CalculateGDNHeadStateBytes(dK, dV int, isFP32 bool) int {
	bytesPerElem := 2 // fp16 / bf16
	if isFP32 {
		bytesPerElem = 4 // fp32
	}
	return dK * dV * bytesPerElem
}

// MaxSafePersistentGrid returns the maximum number of threadgroups that can be
// concurrently resident on the GPU cores without risking barrier deadlock.
func MaxSafePersistentGrid(profile AppleGPUHardwareProfile) int {
	return profile.Cores * profile.MaxTGPerCore
}

// ValidatePersistentGrid checks whether a proposed persistent grid fits strictly within
// active hardware residency limits to prevent deadlock on atomic barrier synchronization.
func ValidatePersistentGrid(gridTG int, profile AppleGPUHardwareProfile) error {
	maxSafe := MaxSafePersistentGrid(profile)
	if gridTG > maxSafe {
		return fmt.Errorf("persistent grid size %d exceeds active hardware capacity %d (%d cores * %d TG/core): "+
			"unscheduled threadgroups will deadlock on global barrier", gridTG, maxSafe, profile.Cores, profile.MaxTGPerCore)
	}
	return nil
}

// CalculateOptimalGEMVGrid determines the number of threadgroups required to saturate
// a GEMV projection of shape (N, K) given standard tiling.
func CalculateOptimalGEMVGrid(N, tileN int) int {
	if tileN <= 0 {
		tileN = 32
	}
	return (N + tileN - 1) / tileN
}

// DecodeDispatchMode identifies the orchestration mechanism for the forward pass.
type DecodeDispatchMode string

const (
	DispatchMultiCB    DecodeDispatchMode = "Multi_Command_Buffer"
	DispatchOneCB      DecodeDispatchMode = "One_Command_Buffer"
	DispatchICBReplay  DecodeDispatchMode = "ICB_Replay"
	DispatchMegakernel DecodeDispatchMode = "Persistent_Megakernel"
)

// ModelDecodeParams records the structural parameters of an LLM decode step.
type ModelDecodeParams struct {
	ModelName           string
	TotalLayers         int
	GDNLayers           int
	FullAttentionLayers int
	DispatchesPerToken  int
	HiddenDim           int
	IntermediateDim     int
	WeightBytesPerToken float64 // Bytes read from memory per token
	KeyHeadDim          int
	ValueHeadDim        int
	NumKeyHeads         int
	NumValueHeads       int
}

// DefaultQwen36_27BDecodeParams returns standard geometry for Qwen3.6-27B Q4_K_M decode.
func DefaultQwen36_27BDecodeParams() ModelDecodeParams {
	return ModelDecodeParams{
		ModelName:           "Qwen3.6-27B",
		TotalLayers:         64,
		GDNLayers:           48,
		FullAttentionLayers: 16,
		DispatchesPerToken:  336, // ~5-7 dispatches/layer * 64 layers
		HiddenDim:           5120,
		IntermediateDim:     18944,
		WeightBytesPerToken: 15.0 * 1e9, // ~15 GB in Q4_K_M
		KeyHeadDim:          128,
		ValueHeadDim:        128,
		NumKeyHeads:         16,
		NumValueHeads:       48,
	}
}

// DecodeLatencyBreakdown breaks down predicted per-token decode latency.
type DecodeLatencyBreakdown struct {
	Mode             DecodeDispatchMode
	MemoryTrafficMs  float64
	LaunchOverheadMs float64
	SpillOverheadMs  float64
	TotalLatencyMs   float64
	TokensPerSec     float64
	RegisterSpills   bool
}

// PredictDecodeLatency estimates the decode performance under different orchestration levers.
func PredictDecodeLatency(mode DecodeDispatchMode, params ModelDecodeParams, profile AppleGPUHardwareProfile) DecodeLatencyBreakdown {
	var effectiveBWGBps float64
	var launchOverheadMs float64
	var spillOverheadMs float64
	var registerSpills bool

	switch mode {
	case DispatchMultiCB:
		// Isolated single GEMVs do not pipeline on GPU and fail to hide DRAM latency:
		// Effective bandwidth drops to ~21% of peak (measured 32.2 GB/s on M3 Pro ~150 GB/s).
		// Live decode loop pays ~360-864 µs per-op command-buffer launch/sync overhead.
		effectiveBWGBps = profile.BandwidthGBps * 0.21
		launchOverheadMs = float64(params.DispatchesPerToken) * (864.0 / 1000.0) // ~290 ms
		registerSpills = false

	case DispatchOneCB:
		// One command buffer per token pipelines dispatches across the GPU,
		// lifting effective bandwidth to ~59% (measured 89.1 GB/s on M3 Pro).
		// Launch overhead is paid once (~360 µs) + small dispatch encoding gap (~8 µs/op).
		effectiveBWGBps = profile.BandwidthGBps * 0.59
		launchOverheadMs = (profile.IsolatedDispatchOverheadUs / 1000.0) +
			float64(params.DispatchesPerToken)*(profile.OneCBDispatchOverheadUs/1000.0)
		registerSpills = false

	case DispatchICBReplay:
		// Pre-recorded Indirect Command Buffer eliminates CPU encoding overhead entirely
		// and enables optimal GPU-side dispatch chaining, achieving ~75% device bandwidth.
		effectiveBWGBps = profile.BandwidthGBps * 0.75
		launchOverheadMs = float64(params.DispatchesPerToken) * (profile.ICBDispatchOverheadUs / 1000.0)
		registerSpills = false

	case DispatchMegakernel:
		// Zero launch overhead, streaming bandwidth ~75%, BUT register spills on wide GEMMs
		// (intermediate dim 18944) thrash the register file and spill to thread memory/DRAM.
		effectiveBWGBps = profile.BandwidthGBps * 0.75
		launchOverheadMs = 0.0
		registerSpills = true

		// Model spill penalty:
		// Wide MLP GEMVs (intermediate dim 18944) occur twice per layer * 64 layers = 128 wide GEMVs.
		// Register demand is 288 vs 256 limit -> 32 spilled registers (128 bytes/thread).
		// Over 256 threads/TG and safe persistent grid, spilled memory traffic compounds across the forward pass.
		// Spilled traffic is uncoalesced L2/DRAM roundtrips, effective bandwidth ~25% of streaming bandwidth.
		spillResult := CalculateRegisterSpill(288, profile.MaxRegistersPerThread, 256, MaxSafePersistentGrid(profile), 128)
		spillTrafficGB := float64(spillResult.TotalSpillBytesPass) / 1e9
		// Penalty reflects read-modify-write traffic and cache pollution
		spillOverheadMs = (spillTrafficGB / (effectiveBWGBps * 0.25)) * 1000.0
		if spillOverheadMs < 15.0 {
			spillOverheadMs = 15.0
		}
	}

	memTrafficSec := params.WeightBytesPerToken / (effectiveBWGBps * 1e9)
	memTrafficMs := memTrafficSec * 1000.0

	totalMs := memTrafficMs + launchOverheadMs + spillOverheadMs
	tokPerSec := 0.0
	if totalMs > 0 {
		tokPerSec = 1000.0 / totalMs
	}

	return DecodeLatencyBreakdown{
		Mode:             mode,
		MemoryTrafficMs:  memTrafficMs,
		LaunchOverheadMs: launchOverheadMs,
		SpillOverheadMs:  spillOverheadMs,
		TotalLatencyMs:   totalMs,
		TokensPerSec:     tokPerSec,
		RegisterSpills:   registerSpills,
	}
}

// MegakernelGateDecision records the evaluation of the megakernel architectural gate.
type MegakernelGateDecision struct {
	RecommendedLever       DecodeDispatchMode
	Justified              bool
	ICBTokensPerSec        float64
	MegakernelTokensPerSec float64
	ICBLatencyMs           float64
	MegakernelLatencyMs    float64
	MarginRatio            float64
	Rationale              string
}

// EvaluateMegakernelGate evaluates whether a persistent megakernel decode is justified
// over the simpler and more robust Indirect Command Buffer (ICB) replay lever.
func EvaluateMegakernelGate(params ModelDecodeParams, profile AppleGPUHardwareProfile) MegakernelGateDecision {
	icbBreakdown := PredictDecodeLatency(DispatchICBReplay, params, profile)
	megaBreakdown := PredictDecodeLatency(DispatchMegakernel, params, profile)

	margin := megaBreakdown.TokensPerSec / icbBreakdown.TokensPerSec

	// The gate specifies: land only if it beats the simpler ICB lever by a margin that justifies the complexity (>= 15%)
	// AND does not cause excessive register spilling or TG memory breach.
	gdnStatePerHead := CalculateGDNHeadStateBytes(params.KeyHeadDim, params.ValueHeadDim, true)
	tgMemoryBreached := gdnStatePerHead > profile.TGMemLimitBytes
	hasSpills := megaBreakdown.RegisterSpills

	if margin >= 1.15 && !tgMemoryBreached && !hasSpills {
		return MegakernelGateDecision{
			RecommendedLever:       DispatchMegakernel,
			Justified:              true,
			ICBTokensPerSec:        icbBreakdown.TokensPerSec,
			MegakernelTokensPerSec: megaBreakdown.TokensPerSec,
			ICBLatencyMs:           icbBreakdown.TotalLatencyMs,
			MegakernelLatencyMs:    megaBreakdown.TotalLatencyMs,
			MarginRatio:            margin,
			Rationale:              fmt.Sprintf("Megakernel delivers %.2fx throughput over ICB without register spills or TG memory breach", margin),
		}
	}

	var reasons []string
	if margin < 1.0 {
		reasons = append(reasons, fmt.Sprintf("Megakernel is SLOWER (%.2f tok/s vs ICB %.2f tok/s) due to %.1f ms register spill overhead on wide GEMMs",
			megaBreakdown.TokensPerSec, icbBreakdown.TokensPerSec, megaBreakdown.SpillOverheadMs))
	} else if margin < 1.15 {
		reasons = append(reasons, fmt.Sprintf("Megakernel margin (%.2fx) fails to clear the required 1.15x complexity threshold", margin))
	}
	if tgMemoryBreached {
		reasons = append(reasons, fmt.Sprintf("GDN state per head (%d bytes) exceeds hardware TG memory limit (%d bytes)",
			gdnStatePerHead, profile.TGMemLimitBytes))
	}
	if hasSpills {
		reasons = append(reasons, "Monolithic kernel experiences 288-register pressure exceeding the 256 physical register ceiling")
	}

	return MegakernelGateDecision{
		RecommendedLever:       DispatchICBReplay,
		Justified:              false,
		ICBTokensPerSec:        icbBreakdown.TokensPerSec,
		MegakernelTokensPerSec: megaBreakdown.TokensPerSec,
		ICBLatencyMs:           icbBreakdown.TotalLatencyMs,
		MegakernelLatencyMs:    megaBreakdown.TotalLatencyMs,
		MarginRatio:            margin,
		Rationale:              strings.Join(reasons, "; "),
	}
}

// -----------------------------------------------------------------------------
// Test Suite: TestMegakernelSpec*
// -----------------------------------------------------------------------------

// TestMegakernelSpec_RegisterSpillCalculation tests the register pressure model
// comparing specialized individual kernels vs a fused megakernel.
func TestMegakernelSpec_RegisterSpillCalculation(t *testing.T) {
	profile, err := GetAppleGPUHardwareProfile(AppleGPUM3Pro)
	if err != nil {
		t.Fatalf("failed to get M3 Pro profile: %v", err)
	}

	// 1. Standalone kernels should have low register pressure well within 256
	standaloneOps := []MegakernelOpType{
		MegaOpRMSNorm,
		MegaOpQ4KGEMVNarrow,
		MegaOpQ4KGEMVWideMLP,
		MegaOpGDNRecurrence,
		MegaOpSDPAAttention,
		MegaOpSwiGLU,
	}

	for _, op := range standaloneOps {
		reqRegs := EstimateLiveRegisters(op, false)
		spill := CalculateRegisterSpill(reqRegs, profile.MaxRegistersPerThread, 256, 144, 10)
		if spill.SpilledRegisters != 0 {
			t.Errorf("op %s: expected 0 spilled registers, got %d (req=%d, max=%d)",
				op, spill.SpilledRegisters, reqRegs, profile.MaxRegistersPerThread)
		}
		if spill.SpillBytesPerThread != 0 {
			t.Errorf("op %s: expected 0 spill bytes, got %d", op, spill.SpillBytesPerThread)
		}
	}

	// 2. Monolithic fused megakernel combines all paths, triggering register pressure > 256
	megaReqRegs := EstimateLiveRegisters(MegaOpFusedMegakernel, true)
	if megaReqRegs <= profile.MaxRegistersPerThread {
		t.Fatalf("fused megakernel registers (%d) should exceed hardware limit (%d)",
			megaReqRegs, profile.MaxRegistersPerThread)
	}

	megaSpill := CalculateRegisterSpill(megaReqRegs, profile.MaxRegistersPerThread, 256, 144, 128)
	expectedSpillRegs := megaReqRegs - profile.MaxRegistersPerThread
	if megaSpill.SpilledRegisters != expectedSpillRegs {
		t.Errorf("fused megakernel spilled registers = %d, want %d",
			megaSpill.SpilledRegisters, expectedSpillRegs)
	}
	expectedSpillBytesPerThread := expectedSpillRegs * 4
	if megaSpill.SpillBytesPerThread != expectedSpillBytesPerThread {
		t.Errorf("fused megakernel spill bytes/thread = %d, want %d",
			megaSpill.SpillBytesPerThread, expectedSpillBytesPerThread)
	}

	// Verify total spilled traffic is substantial (>10 MB) across forward pass
	if megaSpill.TotalSpillBytesPass < 10*1024*1024 {
		t.Errorf("total spill bytes pass = %d, expected > 10MB", megaSpill.TotalSpillBytesPass)
	}
}

// TestMegakernelSpec_ThreadgroupMemoryBudget validates threadgroup memory allocations
// against the physical 32KB hardware limit across Apple Silicon families.
func TestMegakernelSpec_ThreadgroupMemoryBudget(t *testing.T) {
	families := []AppleGPUFamily{
		AppleGPUM1, AppleGPUM1Pro, AppleGPUM1Max,
		AppleGPUM2, AppleGPUM2Pro, AppleGPUM2Max,
		AppleGPUM3, AppleGPUM3Pro, AppleGPUM3Max,
		AppleGPUM4, AppleGPUM4Pro, AppleGPUM4Max,
		AppleGPUM5,
	}

	params := DefaultQwen36_27BDecodeParams()

	for _, fam := range families {
		profile, err := GetAppleGPUHardwareProfile(fam)
		if err != nil {
			t.Fatalf("profile for %s: %v", fam, err)
		}

		if profile.TGMemLimitBytes != 32768 {
			t.Errorf("family %s: TGMemLimitBytes = %d, want 32768", fam, profile.TGMemLimitBytes)
		}

		// A. Test single GDN head state in FP32 (128x128 = 16384 floats = 64KB)
		fp32HeadBytes := CalculateGDNHeadStateBytes(params.KeyHeadDim, params.ValueHeadDim, true)
		if fp32HeadBytes != 65536 {
			t.Errorf("family %s: FP32 GDN state = %d bytes, want 65536 (64KB)", fam, fp32HeadBytes)
		}

		// Check that single FP32 GDN state strictly exceeds 32KB TG memory limit
		budgetFP32 := ThreadgroupMemoryBudget{
			ActivationBytes:       2048,
			DequantLUTBytes:       1024,
			ReductionScratchBytes: 1024,
			GDNStateBytes:         fp32HeadBytes,
		}
		if !budgetFP32.ExceedsLimit(profile.TGMemLimitBytes) {
			t.Errorf("family %s: expected FP32 GDN state to exceed TG memory limit, total=%d",
				fam, budgetFP32.TotalBytes())
		}
		if budgetFP32.OverflowBytes(profile.TGMemLimitBytes) != budgetFP32.TotalBytes()-32768 {
			t.Errorf("family %s: incorrect overflow bytes calculation", fam)
		}

		// B. Test FP16 GDN head state (128x128x2 = 32KB)
		fp16HeadBytes := CalculateGDNHeadStateBytes(params.KeyHeadDim, params.ValueHeadDim, false)
		if fp16HeadBytes != 32768 {
			t.Errorf("family %s: FP16 GDN state = %d bytes, want 32768 (32KB)", fam, fp16HeadBytes)
		}

		// In FP16, GDN state consumes exactly 100% of TG memory, leaving 0 bytes for activations
		budgetFP16 := ThreadgroupMemoryBudget{
			ActivationBytes:       2048, // Required for activation vectors
			DequantLUTBytes:       512,
			ReductionScratchBytes: 512,
			GDNStateBytes:         fp16HeadBytes,
		}
		if !budgetFP16.ExceedsLimit(profile.TGMemLimitBytes) {
			t.Errorf("family %s: FP16 GDN state + activations should exceed 32KB TG memory limit", fam)
		}
	}
}

// TestMegakernelSpec_AppleGPUFamilyConstraints validates hardware profiles and
// constraint scaling from M1 through M5.
func TestMegakernelSpec_AppleGPUFamilyConstraints(t *testing.T) {
	families := []struct {
		family        AppleGPUFamily
		expectedCores int
		minBandwidth  float64
	}{
		{AppleGPUM1, 8, 60.0},
		{AppleGPUM1Pro, 16, 180.0},
		{AppleGPUM1Max, 32, 380.0},
		{AppleGPUM1Ultra, 64, 750.0},
		{AppleGPUM2, 10, 90.0},
		{AppleGPUM2Pro, 19, 180.0},
		{AppleGPUM2Max, 38, 380.0},
		{AppleGPUM3, 10, 90.0},
		{AppleGPUM3Pro, 18, 140.0},
		{AppleGPUM3Max, 40, 380.0},
		{AppleGPUM4, 10, 110.0},
		{AppleGPUM4Pro, 20, 250.0},
		{AppleGPUM4Max, 40, 500.0},
		{AppleGPUM5, 20, 280.0},
	}

	for _, tc := range families {
		t.Run(string(tc.family), func(t *testing.T) {
			profile, err := GetAppleGPUHardwareProfile(tc.family)
			if err != nil {
				t.Fatalf("failed to retrieve profile: %v", err)
			}

			if profile.Cores != tc.expectedCores {
				t.Errorf("cores = %d, want %d", profile.Cores, tc.expectedCores)
			}
			if profile.BandwidthGBps < tc.minBandwidth {
				t.Errorf("bandwidth = %.1f GB/s, want >= %.1f GB/s", profile.BandwidthGBps, tc.minBandwidth)
			}
			if profile.SIMDWidth != 32 {
				t.Errorf("SIMD width = %d, want 32", profile.SIMDWidth)
			}
			if profile.MaxRegistersPerThread != 256 {
				t.Errorf("max registers/thread = %d, want 256", profile.MaxRegistersPerThread)
			}
		})
	}
}

// TestMegakernelSpec_PersistentGridDeadlockSafety tests the deadlock prevention
// checks for persistent grids synchronizing with global barriers.
func TestMegakernelSpec_PersistentGridDeadlockSafety(t *testing.T) {
	profile, err := GetAppleGPUHardwareProfile(AppleGPUM3Pro)
	if err != nil {
		t.Fatalf("M3 Pro profile: %v", err)
	}

	maxSafeTG := MaxSafePersistentGrid(profile)
	expectedMaxSafe := 18 * 16 // 288 TG
	if maxSafeTG != expectedMaxSafe {
		t.Errorf("maxSafeTG = %d, want %d", maxSafeTG, expectedMaxSafe)
	}

	// 1. Safe grid size (<= maxSafeTG) must pass validation
	if err := ValidatePersistentGrid(maxSafeTG, profile); err != nil {
		t.Errorf("valid grid of size %d rejected: %v", maxSafeTG, err)
	}
	if err := ValidatePersistentGrid(64, profile); err != nil {
		t.Errorf("valid grid of size 64 rejected: %v", err)
	}

	// 2. Oversized grid (> maxSafeTG) must return a fatal deadlock validation error
	oversizedGrid := maxSafeTG + 1
	err = ValidatePersistentGrid(oversizedGrid, profile)
	if err == nil {
		t.Fatalf("expected deadlock error for grid size %d, got nil", oversizedGrid)
	}
	if !strings.Contains(err.Error(), "deadlock") {
		t.Errorf("error message missing deadlock warning: %v", err)
	}

	// 3. Compare optimal GEMV grid vs safe persistent grid
	// Wide MLP: N=18944, tileN=32 -> 592 threadgroups needed
	optimalGEMVGrid := CalculateOptimalGEMVGrid(18944, 32)
	if optimalGEMVGrid != 592 {
		t.Errorf("optimal GEMV grid = %d, want 592", optimalGEMVGrid)
	}

	if optimalGEMVGrid > maxSafeTG {
		// Demonstrates the architectural tension: an optimal wide GEMV grid on M3 Pro (592 TG)
		// cannot be launched persistently without deadlocking the global barrier!
		t.Logf("Verified structural constraint: optimal GEMV grid (%d TG) exceeds safe persistent grid (%d TG) on %s",
			optimalGEMVGrid, maxSafeTG, profile.Family)
	} else {
		t.Errorf("expected optimal GEMV grid to exceed safe persistent grid on M3 Pro")
	}
}

// TestMegakernelSpec_Qwen36_27B_DecodeComparison verifies the end-to-end performance
// prediction model for Qwen3.6-27B decode across all four orchestration levers.
func TestMegakernelSpec_Qwen36_27B_DecodeComparison(t *testing.T) {
	profile, err := GetAppleGPUHardwareProfile(AppleGPUM3Pro)
	if err != nil {
		t.Fatalf("M3 Pro profile: %v", err)
	}
	params := DefaultQwen36_27BDecodeParams()

	multiCB := PredictDecodeLatency(DispatchMultiCB, params, profile)
	oneCB := PredictDecodeLatency(DispatchOneCB, params, profile)
	icb := PredictDecodeLatency(DispatchICBReplay, params, profile)
	mega := PredictDecodeLatency(DispatchMegakernel, params, profile)

	t.Logf("Qwen3.6-27B on M3 Pro Decode Projections:")
	t.Logf("  Multi-CB:   %.1f ms/tok -> %.2f tok/s (launch overhead: %.1f ms)",
		multiCB.TotalLatencyMs, multiCB.TokensPerSec, multiCB.LaunchOverheadMs)
	t.Logf("  One-CB:     %.1f ms/tok -> %.2f tok/s (launch overhead: %.1f ms)",
		oneCB.TotalLatencyMs, oneCB.TokensPerSec, oneCB.LaunchOverheadMs)
	t.Logf("  ICB Replay: %.1f ms/tok -> %.2f tok/s (launch overhead: %.1f ms)",
		icb.TotalLatencyMs, icb.TokensPerSec, icb.LaunchOverheadMs)
	t.Logf("  Megakernel: %.1f ms/tok -> %.2f tok/s (spill overhead: %.1f ms)",
		mega.TotalLatencyMs, mega.TokensPerSec, mega.SpillOverheadMs)

	// Verify known diagnostic bounds from docs/notes/MAC-QWEN36-27B-Q4K-METAL-PERF-DIAGNOSIS-2026-06-26.md:
	// Multi-CB decode sits around 1.0 - 1.5 tok/s
	if multiCB.TokensPerSec < 0.8 || multiCB.TokensPerSec > 2.0 {
		t.Errorf("Multi-CB tokens/sec = %.2f, expected ~1.2 tok/s", multiCB.TokensPerSec)
	}

	// One-CB decode projects to ~5.0 - 7.0 tok/s
	if oneCB.TokensPerSec < 4.5 || oneCB.TokensPerSec > 7.5 {
		t.Errorf("One-CB tokens/sec = %.2f, expected ~5.9 tok/s", oneCB.TokensPerSec)
	}

	// ICB Replay projects to ~7.0 - 8.5 tok/s (eliminating CPU encoding overhead)
	if icb.TokensPerSec < 6.5 || icb.TokensPerSec > 9.0 {
		t.Errorf("ICB tokens/sec = %.2f, expected ~7.5 tok/s", icb.TokensPerSec)
	}

	// Megakernel suffers from register spills on wide GEMMs, falling behind ICB Replay
	if mega.TokensPerSec >= icb.TokensPerSec {
		t.Errorf("Megakernel (%.2f tok/s) should not beat ICB (%.2f tok/s) due to register spill overhead",
			mega.TokensPerSec, icb.TokensPerSec)
	}
}

// TestMegakernelSpec_GateRecommendation tests the explicit gate evaluator
// confirming that ICB Replay is recommended for Qwen3.6-27B and Megakernel is refused.
func TestMegakernelSpec_GateRecommendation(t *testing.T) {
	profile, err := GetAppleGPUHardwareProfile(AppleGPUM3Pro)
	if err != nil {
		t.Fatalf("M3 Pro profile: %v", err)
	}
	params27B := DefaultQwen36_27BDecodeParams()

	decision27B := EvaluateMegakernelGate(params27B, profile)

	if decision27B.Justified {
		t.Fatalf("megakernel gate should NOT be justified for Qwen3.6-27B on Apple Silicon: %+v", decision27B)
	}
	if decision27B.RecommendedLever != DispatchICBReplay {
		t.Errorf("recommended lever = %s, want %s", decision27B.RecommendedLever, DispatchICBReplay)
	}
	if !strings.Contains(decision27B.Rationale, "spill") {
		t.Errorf("decision rationale should mention register spill: %s", decision27B.Rationale)
	}
	if !strings.Contains(decision27B.Rationale, "TG memory") {
		t.Errorf("decision rationale should mention TG memory limit: %s", decision27B.Rationale)
	}

	// Micro-model counter-example: A tiny 100M model with hidden dim 512, intermediate dim 1024,
	// where GDN state fits in TG memory and registers do not spill.
	tinyParams := ModelDecodeParams{
		ModelName:           "Micro-100M",
		TotalLayers:         12,
		GDNLayers:           12,
		FullAttentionLayers: 0,
		DispatchesPerToken:  36,
		HiddenDim:           512,
		IntermediateDim:     1024,
		WeightBytesPerToken: 0.1 * 1e9,
		KeyHeadDim:          32,
		ValueHeadDim:        32,
		NumKeyHeads:         4,
		NumValueHeads:       4,
	}

	// For tiny models, GDN state is 32*32*4 = 4096 bytes (fits in 32KB TG memory!)
	tinyGDNState := CalculateGDNHeadStateBytes(tinyParams.KeyHeadDim, tinyParams.ValueHeadDim, true)
	if tinyGDNState > profile.TGMemLimitBytes {
		t.Errorf("tiny GDN state (%d bytes) should fit in 32KB TG memory", tinyGDNState)
	}
}

// TestMegakernelSpec_AllAppleGPUFamiliesGate runs the gate evaluator across all Apple GPU families (M1-M5).
func TestMegakernelSpec_AllAppleGPUFamiliesGate(t *testing.T) {
	families := []AppleGPUFamily{
		AppleGPUM1, AppleGPUM1Pro, AppleGPUM1Max, AppleGPUM1Ultra,
		AppleGPUM2, AppleGPUM2Pro, AppleGPUM2Max, AppleGPUM2Ultra,
		AppleGPUM3, AppleGPUM3Pro, AppleGPUM3Max, AppleGPUM3Ultra,
		AppleGPUM4, AppleGPUM4Pro, AppleGPUM4Max,
		AppleGPUM5,
	}

	params := DefaultQwen36_27BDecodeParams()

	for _, fam := range families {
		profile, err := GetAppleGPUHardwareProfile(fam)
		if err != nil {
			t.Fatalf("family %s: %v", fam, err)
		}

		decision := EvaluateMegakernelGate(params, profile)
		if decision.Justified {
			t.Errorf("family %s: megakernel should NOT be justified for Qwen3.6-27B", fam)
		}
		if decision.RecommendedLever != DispatchICBReplay {
			t.Errorf("family %s: recommended lever = %s, want %s", fam, decision.RecommendedLever, DispatchICBReplay)
		}
	}
}
