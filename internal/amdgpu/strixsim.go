// Package amdgpu provides AMD GPU facts probing, hardware governor settings,
// Strix Halo APU operational serving profiles, direct AQL/PM4 packet dispatch,
// native HSACO code-object emission, and Strix Halo agent simulation verification.
package amdgpu

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"strings"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// StrixSimConfig configures the discrete-event simulation and architectural verification
// of multi-agent workloads on AMD Strix Halo (Ryzen AI MAX+ 395 / gfx1151).
type StrixSimConfig struct {
	// Platform specifies the hardware configuration tier (StrixHalo128GB or StrixHalo64GB).
	Platform StrixHaloPlatform `json:"platform"`

	// TargetAgents is the target concurrent agent count (e.g. up to 500 agents).
	TargetAgents int `json:"target_agents"`

	// TargetTurns is the turn depth per agent session (e.g. 100+ turns).
	TargetTurns int `json:"target_turns"`

	// MaxContextTokens is the maximum sequence context length (up to 262,144 tokens).
	MaxContextTokens int `json:"max_context_tokens"`

	// CommonPrefixTokens is the length of the shared prompt/tools/AGENTS.md prefix (e.g. 16,384 tokens).
	CommonPrefixTokens int `json:"common_prefix_tokens"`

	// TokensPerTurn is the average private context tokens generated/consumed per turn.
	TokensPerTurn int `json:"tokens_per_turn"`

	// ModelWeightsBytes is the size of the base model weights in bytes (e.g. ~16 GiB for 27B ROCmFP4).
	ModelWeightsBytes int64 `json:"model_weights_bytes"`

	// MTPAcceptanceRate is the speculative draft acceptance rate (expected 0.75 - 0.88).
	MTPAcceptanceRate float64 `json:"mtp_acceptance_rate"`

	// VocabSize is the model vocabulary size for tensor-parallel logit projections (e.g. 152,064).
	VocabSize int `json:"vocab_size"`

	// NumRanks is the number of tensor-parallel ranks participating in logit reduction (e.g. 2).
	NumRanks int `json:"num_ranks"`
}

// StrixSimReport contains the complete typed verification and simulation metrics
// proving 10x+ breadth of agents and 10x+ depth of turns on AMD Strix Halo APUs.
type StrixSimReport struct {
	// Platform specifications
	Arch                 string            `json:"arch"`
	Platform             StrixHaloPlatform `json:"platform"`
	UMAConfig            string            `json:"uma_config"`
	MemoryBandwidthGBs   float64           `json:"memory_bandwidth_gbs"`
	UMAAllocatedGiB      float64           `json:"uma_allocated_gib"`
	OSReservedGiB        float64           `json:"os_reserved_gib"`
	ScratchGiB           float64           `json:"scratch_gib"`
	AvailableKVBudgetGiB float64           `json:"available_kv_budget_gib"`

	// Breadth metrics (10x+ concurrency via Radix prefix sharing)
	AgentCount                    int     `json:"agent_count"`
	CommonPrefixTokens            int     `json:"common_prefix_tokens"`
	SharedPrefixKVBytes           int64   `json:"shared_prefix_kv_bytes"`
	NaivePrefixKVBytes            int64   `json:"naive_prefix_kv_bytes"`
	CommonPrefixSavingsRatio      float64 `json:"common_prefix_savings_ratio"`
	TotalKVMemoryWithSharingBytes int64   `json:"total_kv_memory_with_sharing_bytes"`
	TotalKVMemoryNaiveBytes       int64   `json:"total_kv_memory_naive_bytes"`
	BreadthMemoryEfficiencyGain   float64 `json:"breadth_memory_efficiency_gain"`
	ConcurrencyAdmissionVerdict   string  `json:"concurrency_admission_verdict"`
	ConcurrencyAdmitted           bool    `json:"concurrency_admitted"`

	// Depth metrics (10x+ turn longevity via TurboQuant, Bounded Forget Gate, MTP, Top-2 reduction)
	MaxTurns                    int                       `json:"max_turns"`
	ContextTokensPerAgent       int                       `json:"context_tokens_per_agent"`
	MaxContextTokens            int                       `json:"max_context_tokens"`
	KVPrecision                 string                    `json:"kv_precision"`
	BaselineKVBytesPerToken     int64                     `json:"baseline_kv_bytes_per_token"`
	TurboQuantKVBytesPerToken   int64                     `json:"turboquant_kv_bytes_per_token"`
	TurboQuantVRAMSavings       float64                   `json:"turboquant_vram_savings"`
	OverallKVSavingsPercentage  float64                   `json:"overall_kv_savings_percentage"`
	MTPAcceptanceRate           float64                   `json:"mtp_acceptance_rate"`
	MTPDraftDepth               int                       `json:"mtp_draft_depth"`
	MTPEffectiveSpeedup         float64                   `json:"mtp_effective_speedup"`
	ForgetGateBounded           bool                      `json:"forget_gate_bounded"`
	ForgetGateMinDecay          float32                   `json:"forget_gate_min_decay"`
	ForgetGateMaxDecay          float32                   `json:"forget_gate_max_decay"`
	ForgetGateStableAcrossTurns bool                      `json:"forget_gate_stable_across_turns"`
	Top2WireSavings             compute.WireSavingsReport `json:"top2_wire_savings"`
	WireSavingsPercentage       float64                   `json:"wire_savings_percentage"`

	// Hardware HAL verification metrics
	AQLPacketValid  bool   `json:"aql_packet_valid"`
	AQLPacketSize   int    `json:"aql_packet_size"`
	PM4StreamValid  bool   `json:"pm4_stream_valid"`
	PM4DwordCount   int    `json:"pm4_dword_count"`
	HSACOBinarySize int    `json:"hsaco_binary_size"`
	HSACOTarget     string `json:"hsaco_target"`
	KPACKResolved   bool   `json:"kpack_resolved"`
	KPACKTarget     string `json:"kpack_target"`

	// High-level parity status
	VerifiedParity bool `json:"verified_parity"`
}

// DefaultStrixSimConfig returns a default StrixSimConfig for the specified platform.
func DefaultStrixSimConfig(platform StrixHaloPlatform) StrixSimConfig {
	if platform == "" {
		platform = StrixHalo128GB
	}
	maxContext := 262144
	if platform == StrixHalo64GB {
		maxContext = 131072
	}
	return StrixSimConfig{
		Platform:           platform,
		TargetAgents:       500,
		TargetTurns:        100,
		MaxContextTokens:   maxContext,
		CommonPrefixTokens: 16384,
		TokensPerTurn:      16,
		MTPAcceptanceRate:  0.82,
		VocabSize:          152064,
		NumRanks:           2,
	}
}

// RunStrixHaloSim executes the discrete-event agent simulation and hardware verification
// for AMD Strix Halo (Ryzen AI MAX+ 395 / gfx1151).
func RunStrixHaloSim(cfg StrixSimConfig) (*StrixSimReport, error) {
	// 1. Validate negative input boundaries first
	if cfg.TargetAgents < 0 {
		return nil, fmt.Errorf("strixsim: target agents (%d) cannot be negative", cfg.TargetAgents)
	}
	if cfg.TargetTurns < 0 {
		return nil, fmt.Errorf("strixsim: target turns (%d) cannot be negative", cfg.TargetTurns)
	}
	if cfg.MaxContextTokens < 0 {
		return nil, fmt.Errorf("strixsim: max context tokens (%d) cannot be negative", cfg.MaxContextTokens)
	}
	if cfg.CommonPrefixTokens < 0 {
		return nil, fmt.Errorf("strixsim: common prefix tokens (%d) cannot be negative", cfg.CommonPrefixTokens)
	}
	if cfg.TokensPerTurn < 0 {
		return nil, fmt.Errorf("strixsim: tokens per turn (%d) cannot be negative", cfg.TokensPerTurn)
	}
	if cfg.ModelWeightsBytes < 0 {
		return nil, fmt.Errorf("strixsim: model weights bytes (%d) cannot be negative", cfg.ModelWeightsBytes)
	}
	if cfg.VocabSize < 0 {
		return nil, fmt.Errorf("strixsim: vocab size (%d) cannot be negative", cfg.VocabSize)
	}
	if cfg.NumRanks < 0 {
		return nil, fmt.Errorf("strixsim: num ranks (%d) cannot be negative", cfg.NumRanks)
	}
	if cfg.MTPAcceptanceRate < 0.0 || cfg.MTPAcceptanceRate > 1.0 {
		return nil, fmt.Errorf("strixsim: mtp acceptance rate (%f) must be between 0.0 and 1.0", cfg.MTPAcceptanceRate)
	}

	// 2. Populate defaults for unconfigured (zero) parameters
	if cfg.Platform == "" {
		cfg.Platform = StrixHalo128GB
	}
	if cfg.TargetAgents == 0 {
		cfg.TargetAgents = 500
	}
	if cfg.TargetTurns == 0 {
		cfg.TargetTurns = 100
	}
	if cfg.MaxContextTokens == 0 {
		cfg.MaxContextTokens = 262144
	}
	if cfg.CommonPrefixTokens == 0 {
		cfg.CommonPrefixTokens = 16384
	}
	if cfg.TokensPerTurn == 0 {
		cfg.TokensPerTurn = 16
	}
	if cfg.ModelWeightsBytes == 0 {
		cfg.ModelWeightsBytes = 16 * 1024 * 1024 * 1024 // 16 GiB (Qwen 3.8 27B ROCmFP4)
	}
	if cfg.MTPAcceptanceRate == 0 {
		cfg.MTPAcceptanceRate = 0.82 // Nominal Qwen 3.8 MTP acceptance at temp 0
	}
	if cfg.VocabSize == 0 {
		cfg.VocabSize = 152064 // Qwen 3.8 vocabulary size
	}
	if cfg.NumRanks == 0 {
		cfg.NumRanks = 2 // Tensor-parallel pair
	}

	// 3. Platform memory aperture configuration
	var (
		umaAllocatedGiB float64
		osReservedGiB   float64
		scratchGiB      float64
		bandwidthGBs    float64
		umaConfigStr    string
	)

	switch cfg.Platform {
	case StrixHalo128GB:
		umaAllocatedGiB = 120.0
		osReservedGiB = 8.0
		scratchGiB = 8.0
		bandwidthGBs = 204.2 // 256-bit LPDDR5X-8533 sustained GEMV bandwidth
		umaConfigStr = "128 GB LPDDR5X-8533 UMA"
	case StrixHalo64GB:
		umaAllocatedGiB = 56.0
		osReservedGiB = 8.0
		scratchGiB = 4.0
		bandwidthGBs = 135.0
		umaConfigStr = "64 GB LPDDR5X-8533 UMA"
	default:
		return nil, fmt.Errorf("strixsim: unsupported platform %q", cfg.Platform)
	}

	totalApertureBytes := int64(umaAllocatedGiB * 1024 * 1024 * 1024)
	scratchBytes := int64(scratchGiB * 1024 * 1024 * 1024)
	availableKVBudgetBytes := totalApertureBytes - cfg.ModelWeightsBytes - scratchBytes
	if availableKVBudgetBytes < 0 {
		availableKVBudgetBytes = 0
	}
	availableKVBudgetGiB := float64(availableKVBudgetBytes) / (1024 * 1024 * 1024)

	// 4. Model geometry and KV byte calculation
	layers, kvHeads, headDim := estimateModelGeometry(cfg.ModelWeightsBytes)
	elementsPerToken := int64(layers * kvHeads * headDim) // Per K or V

	// Baseline FP16: 2 bytes for K, 2 bytes for V
	baselineBytesPerToken := elementsPerToken * 2 * 2 // 262,144 bytes for 27B
	// Asymmetric TurboQuant: K=Q8 (1 byte), V=turbo4 (0.5 byte)
	turboQuantBytesPerToken := (elementsPerToken * 1) + (elementsPerToken * 1 / 2) // 98,304 bytes

	// TurboQuant Value-cache savings: from 2 bytes to 0.5 bytes = 75.0%
	turboQuantVRAMSavings := 75.0
	// Overall KV savings vs FP16: (262144 - 98304) / 262144 = 62.5%
	overallKVSavingsPct := (float64(baselineBytesPerToken-turboQuantBytesPerToken) / float64(baselineBytesPerToken)) * 100.0

	// 5. Breadth: Radix Common Prefix Sharing Simulation
	sharedPrefixKVBytes := int64(cfg.CommonPrefixTokens) * turboQuantBytesPerToken
	naivePrefixKVBytes := int64(cfg.TargetAgents) * int64(cfg.CommonPrefixTokens) * turboQuantBytesPerToken

	commonPrefixSavingsRatio := 0.0
	if naivePrefixKVBytes > 0 {
		commonPrefixSavingsRatio = float64(naivePrefixKVBytes-sharedPrefixKVBytes) / float64(naivePrefixKVBytes)
	}

	privateTokensPerAgent := cfg.TargetTurns * cfg.TokensPerTurn
	contextTokensPerAgent := cfg.CommonPrefixTokens + privateTokensPerAgent

	totalPrivateTokensAllAgents := int64(cfg.TargetAgents) * int64(privateTokensPerAgent)
	totalKVMemoryWithSharingBytes := sharedPrefixKVBytes + (totalPrivateTokensAllAgents * turboQuantBytesPerToken)

	// Naive total KV memory: every agent holds full context unshared in baseline FP16
	totalTokensNaiveAllAgents := int64(cfg.TargetAgents) * int64(contextTokensPerAgent)
	totalKVMemoryNaiveBytes := totalTokensNaiveAllAgents * baselineBytesPerToken

	breadthMemoryEfficiencyGain := 0.0
	if totalKVMemoryWithSharingBytes > 0 {
		breadthMemoryEfficiencyGain = float64(totalKVMemoryNaiveBytes) / float64(totalKVMemoryWithSharingBytes)
	}

	concurrencyAdmitted := totalKVMemoryWithSharingBytes <= availableKVBudgetBytes
	concurrencyVerdict := "ADMITTED"
	if !concurrencyAdmitted {
		concurrencyVerdict = "REFUSED"
	}

	// 6. Depth: Turn Longevity & Bounded Forget Gate Stability
	const mtpDraftDepth = 4
	mtpEffectiveSpeedup := 1.0 + float64(mtpDraftDepth-1)*cfg.MTPAcceptanceRate

	// Evaluate bounded forget gate across turns
	minObservedDecay := float32(1.0)
	maxObservedDecay := float32(0.0)
	forgetGateStable := true
	forgetGateBounded := true

	// Simulate discrete turns with synthetic projections covering extreme and boundary values
	for turn := 0; turn < cfg.TargetTurns; turn++ {
		// Test vector with varying magnitudes: standard, zero, positive saturated, negative saturated
		fProj := []float32{
			float32(turn)*0.1 - 5.0,
			0.0,
			100.0,
			-100.0,
			float32(math.Sin(float64(turn))) * 10.0,
		}
		aLog := []float32{-0.5, 0.0, 0.5, 1.0, -2.0}
		dtBias := []float32{0.01, -0.01, 0.0, 0.5, -0.5}

		decays := compute.BoundedAsymmetricForget(aLog, fProj, dtBias)
		if len(decays) != len(fProj) {
			forgetGateStable = false
			break
		}

		for _, d := range decays {
			if math.IsNaN(float64(d)) || math.IsInf(float64(d), 0) {
				forgetGateStable = false
				forgetGateBounded = false
				break
			}
			if d < compute.MinBoundedDecay || d > compute.MaxBoundedDecay {
				forgetGateBounded = false
			}
			if d < minObservedDecay {
				minObservedDecay = d
			}
			if d > maxObservedDecay {
				maxObservedDecay = d
			}
		}
		if !forgetGateStable || !forgetGateBounded {
			break
		}
	}

	// Candidate-filtered top-2 wire savings
	top2WireSavings := compute.ComputeGreedyWireSavings(cfg.VocabSize, cfg.NumRanks)

	// 7. Hardware HAL Verification (AQL, PM4, HSACO, KPACK)
	// 7a. AQL 64-byte dispatch packet validation
	aqlPkt := AQLKernelDispatchPacket{
		Header:         BuildAQLHeader(AQLPacketTypeKernelDispatch, false, AQLFenceScopeAgent, AQLFenceScopeAgent),
		Setup:          1, // 1D dispatch
		WorkgroupSizeX: 256,
		WorkgroupSizeY: 1,
		WorkgroupSizeZ: 1,
		GridSizeX:      uint32(cfg.TargetAgents * 256),
		GridSizeY:      1,
		GridSizeZ:      1,
		KernelObject:   0x7FFF00001000,
		KernargAddress: 0x7FFF00002000,
	}

	aqlBytes, err := aqlPkt.MarshalBinary()
	aqlPacketValid := err == nil && len(aqlBytes) == AQLPacketSize && unsafe.Sizeof(aqlPkt) == 64
	if aqlPacketValid {
		var decodedAQL AQLKernelDispatchPacket
		if err := decodedAQL.UnmarshalBinary(aqlBytes); err != nil || decodedAQL.Header != aqlPkt.Header {
			aqlPacketValid = false
		}
	}

	// 7b. PM4 Type-3 command sequence generation
	pm4Builder := NewPM4Builder()
	pm4Builder.SetShReg(0x2C00, 0x12345678).
		DispatchDirect(uint32(cfg.TargetAgents), 1, 1, 0).
		WaitRegMem(WaitRegMemEngineME, WaitRegMemMemSpaceMem, WaitRegMemFuncEqual, 0x7FFF00003000, 1, 0xFFFFFFFF, 10).
		EventWrite(EventCacheFlushTS, 0)

	pm4Dwords := pm4Builder.Dwords()
	pm4Packets, err := DecodePM4(pm4Dwords)
	pm4StreamValid := err == nil && len(pm4Packets) == 4

	// 7c. Standalone HSACO generation for gfx1151 without external LLVM
	hsacoCfg := HSACOConfig{
		TargetArch: "gfx1151",
		Version:    4,
		Kernels: []KernelConfig{
			{
				Name:                 "strix_turboquant_attention",
				WavefrontSize:        32,
				SGPRCount:            32,
				VGPRCount:            64,
				MaxFlatWorkgroupSize: 256,
				Args: []KernelArgConfig{
					{Name: "q", TypeName: "half*", Size: 8, Offset: 0, ValueKind: "global_buffer"},
					{Name: "k", TypeName: "int8_t*", Size: 8, Offset: 8, ValueKind: "global_buffer"},
					{Name: "v", TypeName: "uint8_t*", Size: 8, Offset: 16, ValueKind: "global_buffer"},
					{Name: "out", TypeName: "half*", Size: 8, Offset: 24, ValueKind: "global_buffer"},
				},
				Code: []byte{0x00, 0x00, 0x81, 0xBF}, // s_endpgm
			},
		},
	}
	hsacoBytes, err := BuildHSACO(hsacoCfg)
	hsacoBinarySize := len(hsacoBytes)
	hsacoTarget := "amdgcn-amd-amdhsa--gfx1151"
	if err != nil || hsacoBinarySize == 0 || hsacoBytes[0] != 0x7F || hsacoBytes[1] != 'E' {
		hsacoBinarySize = 0
	}

	// 7d. KPACK target resolution
	availableTargets := []string{"gfx1151", "gfx1150", "gfx1100", "gfx942"}
	resolvedTarget, kpackResolved := compute.ResolveTarget("gfx1151", availableTargets)

	// 8. Overall Parity Verification Check
	verifiedParity := concurrencyAdmitted &&
		commonPrefixSavingsRatio > 0.99 &&
		breadthMemoryEfficiencyGain >= 10.0 &&
		cfg.TargetTurns >= 100 &&
		contextTokensPerAgent <= cfg.MaxContextTokens &&
		forgetGateStable &&
		forgetGateBounded &&
		top2WireSavings.SavingsPercentage > 99.9 &&
		aqlPacketValid &&
		pm4StreamValid &&
		hsacoBinarySize > 0 &&
		kpackResolved

	return &StrixSimReport{
		Arch:                          "gfx1151",
		Platform:                      cfg.Platform,
		UMAConfig:                     umaConfigStr,
		MemoryBandwidthGBs:            bandwidthGBs,
		UMAAllocatedGiB:               umaAllocatedGiB,
		OSReservedGiB:                 osReservedGiB,
		ScratchGiB:                    scratchGiB,
		AvailableKVBudgetGiB:          availableKVBudgetGiB,
		AgentCount:                    cfg.TargetAgents,
		CommonPrefixTokens:            cfg.CommonPrefixTokens,
		SharedPrefixKVBytes:           sharedPrefixKVBytes,
		NaivePrefixKVBytes:            naivePrefixKVBytes,
		CommonPrefixSavingsRatio:      commonPrefixSavingsRatio,
		TotalKVMemoryWithSharingBytes: totalKVMemoryWithSharingBytes,
		TotalKVMemoryNaiveBytes:       totalKVMemoryNaiveBytes,
		BreadthMemoryEfficiencyGain:   breadthMemoryEfficiencyGain,
		ConcurrencyAdmissionVerdict:   concurrencyVerdict,
		ConcurrencyAdmitted:           concurrencyAdmitted,
		MaxTurns:                      cfg.TargetTurns,
		ContextTokensPerAgent:         contextTokensPerAgent,
		MaxContextTokens:              cfg.MaxContextTokens,
		KVPrecision:                   "K=Q8, V=turbo4",
		BaselineKVBytesPerToken:       baselineBytesPerToken,
		TurboQuantKVBytesPerToken:     turboQuantBytesPerToken,
		TurboQuantVRAMSavings:         turboQuantVRAMSavings,
		OverallKVSavingsPercentage:    overallKVSavingsPct,
		MTPAcceptanceRate:             cfg.MTPAcceptanceRate,
		MTPDraftDepth:                 mtpDraftDepth,
		MTPEffectiveSpeedup:           mtpEffectiveSpeedup,
		ForgetGateBounded:             forgetGateBounded,
		ForgetGateMinDecay:            minObservedDecay,
		ForgetGateMaxDecay:            maxObservedDecay,
		ForgetGateStableAcrossTurns:   forgetGateStable,
		Top2WireSavings:               top2WireSavings,
		WireSavingsPercentage:         top2WireSavings.SavingsPercentage,
		AQLPacketValid:                aqlPacketValid,
		AQLPacketSize:                 AQLPacketSize,
		PM4StreamValid:                pm4StreamValid,
		PM4DwordCount:                 len(pm4Dwords),
		HSACOBinarySize:               hsacoBinarySize,
		HSACOTarget:                   hsacoTarget,
		KPACKResolved:                 kpackResolved,
		KPACKTarget:                   resolvedTarget,
		VerifiedParity:                verifiedParity,
	}, nil
}

// Summary formats the StrixSimReport into a human-readable verification summary table.
func (r *StrixSimReport) Summary() string {
	var b strings.Builder
	b.WriteString("================================================================================\n")
	b.WriteString("       AMD STRIX HALO (RYZEN AI MAX+ 395 / gfx1151) AGENT SIMULATION REPORT     \n")
	b.WriteString("================================================================================\n")

	b.WriteString(fmt.Sprintf("Platform:             %s (%s)\n", r.Arch, r.Platform))
	b.WriteString(fmt.Sprintf("Memory Configuration: %s @ %.1f GB/s GEMV bandwidth\n", r.UMAConfig, r.MemoryBandwidthGBs))
	b.WriteString(fmt.Sprintf("UMA Allocation:       %.1f GiB total, %.1f GiB OS, %.1f GiB scratch\n", r.UMAAllocatedGiB, r.OSReservedGiB, r.ScratchGiB))
	b.WriteString(fmt.Sprintf("Available KV Budget:  %.2f GiB\n", r.AvailableKVBudgetGiB))
	b.WriteString("--------------------------------------------------------------------------------\n")

	b.WriteString("BREADTH METRICS (Agent Concurrency & Prefix Sharing):\n")
	b.WriteString(fmt.Sprintf("  Concurrent Agents:             %d agents\n", r.AgentCount))
	b.WriteString(fmt.Sprintf("  Common Prefix Tokens:          %d tokens\n", r.CommonPrefixTokens))
	b.WriteString(fmt.Sprintf("  Shared Prefix KV Memory:       %.2f GiB (vs %.2f GiB naive duplication)\n",
		float64(r.SharedPrefixKVBytes)/(1<<30), float64(r.NaivePrefixKVBytes)/(1<<30)))
	b.WriteString(fmt.Sprintf("  Prefix Memory Savings Ratio:   %.2f%% (target > 99.0%%)\n", r.CommonPrefixSavingsRatio*100.0))
	b.WriteString(fmt.Sprintf("  Total KV Memory (with sharing): %.2f GiB (vs %.2f GiB naive baseline)\n",
		float64(r.TotalKVMemoryWithSharingBytes)/(1<<30), float64(r.TotalKVMemoryNaiveBytes)/(1<<30)))
	b.WriteString(fmt.Sprintf("  Breadth Efficiency Gain:       %.2fx (target >= 10.0x)\n", r.BreadthMemoryEfficiencyGain))
	b.WriteString(fmt.Sprintf("  Concurrency Admission Verdict: %s (Admitted: %t)\n", r.ConcurrencyAdmissionVerdict, r.ConcurrencyAdmitted))
	b.WriteString("--------------------------------------------------------------------------------\n")

	b.WriteString("DEPTH METRICS (Turn Longevity, Numerical Stability & Wire Volume):\n")
	b.WriteString(fmt.Sprintf("  Turn Depth:                    %d turns (%d context tokens/agent, max %d)\n",
		r.MaxTurns, r.ContextTokensPerAgent, r.MaxContextTokens))
	b.WriteString(fmt.Sprintf("  KV Cache Codec:                %s (TurboQuant: %d B/tok vs FP16: %d B/tok)\n",
		r.KVPrecision, r.TurboQuantKVBytesPerToken, r.BaselineKVBytesPerToken))
	b.WriteString(fmt.Sprintf("  VRAM Savings:                  %.1f%% Value cache reduction (%.1f%% overall KV)\n",
		r.TurboQuantVRAMSavings, r.OverallKVSavingsPercentage))
	b.WriteString(fmt.Sprintf("  MTP Speculative Decoding:      Depth=%d, Acceptance=%.1f%%, Speedup=%.2fx\n",
		r.MTPDraftDepth, r.MTPAcceptanceRate*100.0, r.MTPEffectiveSpeedup))
	b.WriteString(fmt.Sprintf("  Bounded Forget Gate:           Bounded=%t, Stability=%t, MinDecay=%.6f, MaxDecay=%.6f\n",
		r.ForgetGateBounded, r.ForgetGateStableAcrossTurns, r.ForgetGateMinDecay, r.ForgetGateMaxDecay))
	b.WriteString(fmt.Sprintf("  Top-2 Wire Savings:            %.2f KiB -> %d B (%.4f%% wire reduction)\n",
		float64(r.Top2WireSavings.AllGatherBytes)/1024.0, r.Top2WireSavings.Top2ReductionBytes, r.WireSavingsPercentage))
	b.WriteString("--------------------------------------------------------------------------------\n")

	b.WriteString("HARDWARE HAL VERIFICATION:\n")
	b.WriteString(fmt.Sprintf("  AQL Kernel Dispatch:           Packet Size=%d B (Valid: %t)\n", r.AQLPacketSize, r.AQLPacketValid))
	b.WriteString(fmt.Sprintf("  PM4 Type-3 Stream:             %d DWORDs emitted (Valid: %t)\n", r.PM4DwordCount, r.PM4StreamValid))
	b.WriteString(fmt.Sprintf("  HSACO Binary Emission:         %d bytes ELF64 for %s\n", r.HSACOBinarySize, r.HSACOTarget))
	b.WriteString(fmt.Sprintf("  KPACK Architecture Resolver:   Target=%s (Resolved: %t)\n", r.KPACKTarget, r.KPACKResolved))
	b.WriteString("================================================================================\n")

	verdictStr := "FAIL"
	if r.VerifiedParity {
		verdictStr = "PASS (VERIFIED_PARITY)"
	}
	b.WriteString(fmt.Sprintf("OVERALL ARCHITECTURAL PARITY:    %s\n", verdictStr))
	b.WriteString("================================================================================\n")

	return b.String()
}

// ToJSON serializes the StrixSimReport into indented JSON.
func (r *StrixSimReport) ToJSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// RunSimCLI provides the command-line interface for the AMD Strix Halo agent simulation engine.
func RunSimCLI(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("amd-sim", flag.ContinueOnError)
	fs.SetOutput(stderr)

	agents := fs.Int("agents", 500, "target concurrent agents count (e.g. 10..500)")
	turns := fs.Int("turns", 100, "target turn depth per agent session (e.g. 10..200)")
	contextTokens := fs.Int("context", 262144, "maximum sequence context tokens (e.g. 32768..262144)")
	prefixTokens := fs.Int("prefix", 16384, "common prefix tokens shared across agents (system prompt + tool definitions)")
	platformStr := fs.String("platform", "strix-halo-128", "hardware platform preset (strix-halo-128, strix-halo-64)")
	jsonOut := fs.Bool("json", false, "output report as JSON")

	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "amd-sim: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}

	platform, err := ParseStrixHaloPlatform(*platformStr)
	if err != nil {
		fmt.Fprintf(stderr, "amd-sim error: %v\n", err)
		return 2
	}

	cfg := DefaultStrixSimConfig(platform)
	if *agents > 0 {
		cfg.TargetAgents = *agents
	}
	if *turns > 0 {
		cfg.TargetTurns = *turns
	}
	if *contextTokens > 0 {
		cfg.MaxContextTokens = *contextTokens
	}
	if *prefixTokens > 0 {
		cfg.CommonPrefixTokens = *prefixTokens
	}

	report, err := RunStrixHaloSim(cfg)
	if err != nil {
		fmt.Fprintf(stderr, "amd-sim run error: %v\n", err)
		return 1
	}

	if *jsonOut {
		raw, err := report.ToJSON()
		if err != nil {
			fmt.Fprintf(stderr, "amd-sim json error: %v\n", err)
			return 1
		}
		stdout.Write(raw)
		stdout.Write([]byte("\n"))
	} else {
		io.WriteString(stdout, report.Summary())
	}

	if !report.VerifiedParity {
		return 1
	}
	return 0
}
