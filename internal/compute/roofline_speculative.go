package compute

import (
	"math"
	"strings"
)

// roofline_speculative.go — analytical roofline and speculative speedup projection engine
// for issue #10840.
//
// Models hardware compute and memory bandwidth trade-offs for speculative decoding architectures.
// Evaluates operational latency T = max(T_arithmetic, T_memory) + T_overhead across memory-bound
// (batch=1) and compute-bound (batch >= 16) regimes, accounting for weight streaming, KV cache
// state, and candidate tree verification.

// HardwareProfile captures accelerator compute throughput, memory hierarchy bandwidth, and
// runtime kernel dispatch overheads.
type HardwareProfile struct {
	Name                   string  `json:"name"`
	Arch                   string  `json:"arch"`
	PeakMemoryBandwidthGBs float64 `json:"peak_memory_bandwidth_gbs"`
	EffectiveBandwidthGBs  float64 `json:"effective_bandwidth_gbs"`
	PeakComputeTFLOPs      float64 `json:"peak_compute_tflops"`
	MemoryCapacityBytes    int64   `json:"memory_capacity_bytes"`
	OverheadLatencySec     float64 `json:"overhead_latency_sec"`
}

// BaselineHardwareProfile returns a standard hardware configuration by known canonical name.
// Supported keys: "l4", "a100", "m3_max", "h100".
// Name matching is case-insensitive and tolerant of hyphens/underscores.
func BaselineHardwareProfile(name string) HardwareProfile {
	p, _ := LookupHardwareProfile(name)
	return p
}

// LookupHardwareProfile returns the hardware profile and reports whether the name was recognized.
func LookupHardwareProfile(name string) (HardwareProfile, bool) {
	norm := strings.ToLower(strings.TrimSpace(name))
	norm = strings.ReplaceAll(norm, "-", "_")
	norm = strings.ReplaceAll(norm, " ", "_")

	switch norm {
	case "l4", "nvidia_l4":
		return HardwareProfile{
			Name:                   "NVIDIA L4",
			Arch:                   "ada",
			PeakMemoryBandwidthGBs: 300.0,
			EffectiveBandwidthGBs:  240.0, // ~80% sustained Model Bandwidth Utilization (MBU)
			PeakComputeTFLOPs:      242.0, // FP16/BF16 dense tensor cores
			MemoryCapacityBytes:    24 * (1 << 30),
			OverheadLatencySec:     15e-6, // 15 microseconds launch overhead
		}, true

	case "a100", "nvidia_a100", "a100_80gb", "a100_sxm4_80gb":
		return HardwareProfile{
			Name:                   "NVIDIA A100-SXM4-80GB",
			Arch:                   "ampere",
			PeakMemoryBandwidthGBs: 2000.0, // 2.0 TB/s HBM2e
			EffectiveBandwidthGBs:  1600.0, // ~80% sustained MBU
			PeakComputeTFLOPs:      312.0,  // FP16/BF16 dense tensor cores
			MemoryCapacityBytes:    80 * (1 << 30),
			OverheadLatencySec:     10e-6, // 10 microseconds launch overhead
		}, true

	case "m3_max", "apple_m3_max", "m3max":
		return HardwareProfile{
			Name:                   "Apple M3 Max",
			Arch:                   "metal",
			PeakMemoryBandwidthGBs: 400.0, // 400 GB/s unified memory
			EffectiveBandwidthGBs:  320.0, // ~80% sustained MBU
			PeakComputeTFLOPs:      32.8,  // 40-core GPU FP16 compute
			MemoryCapacityBytes:    64 * (1 << 30),
			OverheadLatencySec:     25e-6, // 25 microseconds command buffer overhead
		}, true

	case "h100", "nvidia_h100", "h100_80gb", "h100_sxm5_80gb":
		return HardwareProfile{
			Name:                   "NVIDIA H100-SXM5-80GB",
			Arch:                   "hopper",
			PeakMemoryBandwidthGBs: 3350.0, // 3.35 TB/s HBM3
			EffectiveBandwidthGBs:  2680.0, // ~80% sustained MBU
			PeakComputeTFLOPs:      989.0,  // FP16/BF16 dense tensor cores
			MemoryCapacityBytes:    80 * (1 << 30),
			OverheadLatencySec:     8e-6, // 8 microseconds launch overhead
		}, true

	default:
		return HardwareProfile{Name: name}, false
	}
}

// ModelCostProfile captures the structural parameters, precision, and state footprint of an LLM.
type ModelCostProfile struct {
	Name           string  `json:"name,omitempty"`
	ActiveParams   int64   `json:"active_params"`
	NLayers        int     `json:"n_layers"`
	HiddenDim      int     `json:"hidden_dim"`
	NHeads         int     `json:"n_heads"`
	NKVHeads       int     `json:"n_kv_heads"`
	HeadDim        int     `json:"head_dim"`
	BytesPerWeight float64 `json:"bytes_per_weight"`
	BytesPerKVElem float64 `json:"bytes_per_kv_elem"`
}

// BaselineModelCostProfile returns a standard model profile by known canonical name.
// Supported keys: "qwen_7b", "qwen_0_5b", "qwen_72b".
// Name matching is case-insensitive and tolerant of hyphens/underscores.
func BaselineModelCostProfile(name string) ModelCostProfile {
	p, _ := LookupModelCostProfile(name)
	return p
}

// LookupModelCostProfile returns the model cost profile and reports whether the name was recognized.
func LookupModelCostProfile(name string) (ModelCostProfile, bool) {
	norm := strings.ToLower(strings.TrimSpace(name))
	norm = strings.ReplaceAll(norm, "-", "_")
	norm = strings.ReplaceAll(norm, ".", "_")
	norm = strings.ReplaceAll(norm, " ", "_")

	switch norm {
	case "qwen_7b", "qwen2_5_7b", "qwen7b", "7b":
		return ModelCostProfile{
			Name:           "Qwen2.5-7B",
			ActiveParams:   7_615_616_512,
			NLayers:        28,
			HiddenDim:      3584,
			NHeads:         28,
			NKVHeads:       4,
			HeadDim:        128,
			BytesPerWeight: 2.0, // FP16/BF16 baseline
			BytesPerKVElem: 2.0, // FP16/BF16 KV cache
		}, true

	case "qwen_0_5b", "qwen2_5_0_5b", "qwen_05b", "0_5b", "05b":
		return ModelCostProfile{
			Name:           "Qwen2.5-0.5B",
			ActiveParams:   494_032_768,
			NLayers:        24,
			HiddenDim:      896,
			NHeads:         14,
			NKVHeads:       2,
			HeadDim:        64,
			BytesPerWeight: 2.0,
			BytesPerKVElem: 2.0,
		}, true

	case "qwen_72b", "qwen2_5_72b", "qwen72b", "72b":
		return ModelCostProfile{
			Name:           "Qwen2.5-72B",
			ActiveParams:   72_711_000_000,
			NLayers:        80,
			HiddenDim:      8192,
			NHeads:         64,
			NKVHeads:       8,
			HeadDim:        128,
			BytesPerWeight: 2.0,
			BytesPerKVElem: 2.0,
		}, true

	default:
		return ModelCostProfile{Name: name}, false
	}
}

// SpecAcceptanceProfile defines empirical or statistical acceptance characteristics of speculative proposals.
type SpecAcceptanceProfile struct {
	Method          string    `json:"method"`
	PositionalAlpha []float64 `json:"positional_alpha"`
	MeanAlpha       float64   `json:"mean_alpha"`
	TreeYield       float64   `json:"tree_yield"`
}

// StepLatencyBreakdown details operational execution time across compute, memory transfer, and overhead.
type StepLatencyBreakdown struct {
	TArithmeticSec      float64 `json:"t_arithmetic_sec"`
	TMemorySec          float64 `json:"t_memory_sec"`
	TOverheadSec        float64 `json:"t_overhead_sec"`
	TTotalSec           float64 `json:"t_total_sec"`
	Regime              string  `json:"regime"`
	ArithmeticIntensity float64 `json:"arithmetic_intensity"`
}

// SpeculativeProjectionResult aggregates the comparative performance projections of speculative
// decoding against the standard autoregressive baseline.
type SpeculativeProjectionResult struct {
	BatchSize              int                  `json:"batch_size"`
	SeqLenTokens           int                  `json:"seq_len_tokens"`
	DraftLength            int                  `json:"draft_length"`
	TargetAutoregressive   StepLatencyBreakdown `json:"target_autoregressive"`
	DraftStep              StepLatencyBreakdown `json:"draft_step"`
	VerifyStep             StepLatencyBreakdown `json:"verify_step"`
	ExpectedAcceptedTokens float64              `json:"expected_accepted_tokens"`
	AutoregressiveTPOTSec  float64              `json:"autoregressive_tpot_sec"`
	SpeculativeTPOTSec     float64              `json:"speculative_tpot_sec"`
	Speedup                float64              `json:"speedup"`
	OptimalK               int                  `json:"optimal_k"`
	OptimalSpeedup         float64              `json:"optimal_speedup"`
	Viable                 bool                 `json:"viable"`
}

// CalculateStepLatency calculates forward-pass operational latency under the analytical roofline:
//
//	T = max(T_arithmetic, T_memory) + T_overhead
//
// Memory transfer incorporates both model weight streaming and KV cache reading for seqLen tokens:
//
//	T_memory = (W_model + KV_Cache(L, B)) / Bandwidth_effective
//
// Arithmetic compute accounts for linear projections (2 FLOPs/param/token) and self-attention dot products:
//
//	T_arithmetic = TotalFLOPs / PeakCompute
func CalculateStepLatency(hw HardwareProfile, model ModelCostProfile, batchSize, seqLen, tokensPerStep int) StepLatencyBreakdown {
	if batchSize <= 0 || tokensPerStep <= 0 || model.ActiveParams <= 0 {
		return StepLatencyBreakdown{
			TOverheadSec: math.Max(0, hw.OverheadLatencySec),
			TTotalSec:    math.Max(0, hw.OverheadLatencySec),
			Regime:       "unspecified",
		}
	}
	if seqLen < 0 {
		seqLen = 0
	}

	bytesPerWeight := model.BytesPerWeight
	if bytesPerWeight <= 0 {
		bytesPerWeight = 2.0 // default FP16
	}

	bytesPerKVElem := model.BytesPerKVElem
	if bytesPerKVElem <= 0 {
		bytesPerKVElem = 2.0 // default FP16
	}

	// 1. Memory Traffic Accounting
	// Model weights are streamed from high-bandwidth memory once per step.
	weightBytes := float64(model.ActiveParams) * bytesPerWeight

	// KV cache traffic: reading existing KV cache entries across all layers for active sequences.
	kvHeads := model.NKVHeads
	if kvHeads <= 0 {
		kvHeads = model.NHeads
	}
	headDim := model.HeadDim
	if headDim <= 0 && model.NHeads > 0 {
		headDim = model.HiddenDim / model.NHeads
	}
	if headDim <= 0 {
		headDim = 128
	}
	nLayers := model.NLayers
	if nLayers <= 0 {
		nLayers = 1
	}

	// 2 vectors (K and V) per layer per head per token
	kvBytesPerTokenPerLayer := 2.0 * float64(kvHeads) * float64(headDim) * bytesPerKVElem
	kvBytesPerTokenAllLayers := float64(nLayers) * kvBytesPerTokenPerLayer
	kvMemoryBytes := float64(batchSize) * float64(seqLen) * kvBytesPerTokenAllLayers

	totalBytes := weightBytes + kvMemoryBytes

	// Effective Bandwidth (GB/s -> Bytes/s)
	effBWGBs := hw.EffectiveBandwidthGBs
	if effBWGBs <= 0 && hw.PeakMemoryBandwidthGBs > 0 {
		effBWGBs = hw.PeakMemoryBandwidthGBs * 0.8
	}
	var tMemory float64
	if effBWGBs > 0 {
		tMemory = totalBytes / (effBWGBs * 1e9)
	}

	// 2. Arithmetic Work Accounting
	totalTokens := float64(batchSize * tokensPerStep)

	// Linear layer weight GEMMs: 2 FLOPs per parameter per token (multiply + accumulate)
	weightFLOPs := 2.0 * float64(model.ActiveParams) * totalTokens

	// Attention FLOPs: QK^T dot products (2 FLOPs/elem) and Score*V (2 FLOPs/elem) over context
	attnDim := float64(model.NHeads * headDim)
	if attnDim <= 0 {
		attnDim = float64(model.HiddenDim)
	}
	if attnDim <= 0 {
		attnDim = 4096
	}
	// 4 FLOPs * NLayers * TotalHeadDim * SeqLen * TotalTokens
	qkFlops := 4.0 * float64(nLayers) * attnDim * float64(seqLen) * totalTokens

	totalFLOPs := weightFLOPs + qkFlops

	// Peak Compute Throughput (TFLOPs -> FLOP/s)
	peakComputeFLOPs := hw.PeakComputeTFLOPs * 1e12
	var tArithmetic float64
	if peakComputeFLOPs > 0 {
		tArithmetic = totalFLOPs / peakComputeFLOPs
	}

	// 3. Overhead and Total Time
	tOverhead := math.Max(0, hw.OverheadLatencySec)
	tTotal := math.Max(tArithmetic, tMemory) + tOverhead

	// 4. Arithmetic Intensity and Regime
	var intensity float64
	if totalBytes > 0 {
		intensity = totalFLOPs / totalBytes
	}

	var regime string
	switch {
	case tArithmetic > tMemory:
		regime = "compute_bound"
	case tMemory > tArithmetic:
		regime = "memory_bound"
	default:
		regime = "balanced"
	}

	return StepLatencyBreakdown{
		TArithmeticSec:      tArithmetic,
		TMemorySec:          tMemory,
		TOverheadSec:        tOverhead,
		TTotalSec:           tTotal,
		Regime:              regime,
		ArithmeticIntensity: intensity,
	}
}

// computeExpectedAcceptedTokens evaluates the expected accepted tokens E[N_acc] per step.
func computeExpectedAcceptedTokens(acceptance SpecAcceptanceProfile, draftLength int, isTree bool) float64 {
	if draftLength <= 0 {
		return 1.0
	}

	if isTree {
		if acceptance.TreeYield > 0 {
			yield := acceptance.TreeYield
			if yield < 1.0 {
				yield = 1.0
			}
			maxYield := 1.0 + float64(draftLength)
			if yield > maxYield {
				yield = maxYield
			}
			return yield
		}

		// Analytical tree acceptance modeling when TreeYield is not explicitly provided.
		// A candidate tree of K nodes explores multi-branch hypotheses.
		alpha := acceptance.MeanAlpha
		if len(acceptance.PositionalAlpha) > 0 {
			alpha = acceptance.PositionalAlpha[0]
		}
		if alpha < 0 {
			alpha = 0
		} else if alpha > 1 {
			alpha = 1
		}

		// Tree depth d ~= ceil(log2(K+1))
		depth := int(math.Ceil(math.Log2(float64(draftLength + 1))))
		if depth < 1 {
			depth = 1
		}
		// Effective branching per level improves acceptance probability: alpha_eff = 1 - (1-alpha)^2
		alphaEff := 1.0 - math.Pow(1.0-alpha, 2.0)
		yield := 1.0
		cumProb := 1.0
		for d := 0; d < depth; d++ {
			cumProb *= alphaEff
			yield += cumProb
		}
		if yield > 1.0+float64(draftLength) {
			yield = 1.0 + float64(draftLength)
		}
		return yield
	}

	// Linear speculative decoding
	if len(acceptance.PositionalAlpha) > 0 {
		cumProb := 1.0
		acceptedSum := 0.0
		for i := 0; i < draftLength; i++ {
			var alpha float64
			if i < len(acceptance.PositionalAlpha) {
				alpha = acceptance.PositionalAlpha[i]
			} else if acceptance.MeanAlpha > 0 {
				alpha = acceptance.MeanAlpha
			} else {
				alpha = acceptance.PositionalAlpha[len(acceptance.PositionalAlpha)-1]
			}
			if alpha < 0 {
				alpha = 0
			} else if alpha > 1 {
				alpha = 1
			}
			cumProb *= alpha
			acceptedSum += cumProb
		}
		return 1.0 + acceptedSum
	}

	if acceptance.MeanAlpha > 0 {
		alpha := acceptance.MeanAlpha
		if alpha < 0 {
			alpha = 0
		} else if alpha > 1 {
			alpha = 1
		}
		if alpha == 1.0 {
			return 1.0 + float64(draftLength)
		}
		if alpha == 0.0 {
			return 1.0
		}
		// 1 + sum_{i=1}^K alpha^i = 1 + alpha*(1 - alpha^K) / (1 - alpha)
		return 1.0 + alpha*(1.0-math.Pow(alpha, float64(draftLength)))/(1.0-alpha)
	}

	return 1.0
}

// projectSpeculativeSpeedupInternal computes projection results, optionally performing optimization search.
func projectSpeculativeSpeedupInternal(hw HardwareProfile, target, draft ModelCostProfile, acceptance SpecAcceptanceProfile, batchSize, seqLen, draftLength int, isTree bool, findOptimal bool) SpeculativeProjectionResult {
	if batchSize <= 0 {
		batchSize = 1
	}
	if seqLen < 0 {
		seqLen = 0
	}
	if draftLength < 0 {
		draftLength = 0
	}

	// 1. Target Autoregressive baseline: 1 token generated per sequence
	targetAuto := CalculateStepLatency(hw, target, batchSize, seqLen, 1)
	autoTPOT := targetAuto.TTotalSec

	// 2. Draft Step latency: 1 token generated by draft model
	var draftStep StepLatencyBreakdown
	if draft.ActiveParams > 0 {
		draftStep = CalculateStepLatency(hw, draft, batchSize, seqLen, 1)
	}

	// 3. Verify Step latency: target verifies draftLength candidate tokens in parallel
	verifyTokens := draftLength
	if verifyTokens <= 0 {
		verifyTokens = 1
	}
	verifyStep := CalculateStepLatency(hw, target, batchSize, seqLen, verifyTokens)

	// 4. Speculative Iteration Latency
	// For linear speculative decoding: K sequential draft passes + 1 target verification pass.
	// For tree-structured speculation: candidate tree is emitted in a single draft step (or parallel heads),
	// followed by 1 target verification pass over the candidate tree.
	var draftTotalTime float64
	if draft.ActiveParams > 0 {
		if isTree {
			draftTotalTime = draftStep.TTotalSec
		} else {
			draftTotalTime = float64(draftLength) * draftStep.TTotalSec
		}
	}

	var iterTime float64
	if draftLength == 0 {
		iterTime = autoTPOT
	} else {
		iterTime = draftTotalTime + verifyStep.TTotalSec
	}

	// 5. Expected Accepted Tokens and TPOT
	expectedAccepted := computeExpectedAcceptedTokens(acceptance, draftLength, isTree)

	var specTPOT float64
	if expectedAccepted > 0 {
		specTPOT = iterTime / expectedAccepted
	} else {
		specTPOT = iterTime
	}

	var speedup float64
	if specTPOT > 0 && autoTPOT > 0 {
		speedup = autoTPOT / specTPOT
	}

	res := SpeculativeProjectionResult{
		BatchSize:              batchSize,
		SeqLenTokens:           seqLen,
		DraftLength:            draftLength,
		TargetAutoregressive:   targetAuto,
		DraftStep:              draftStep,
		VerifyStep:             verifyStep,
		ExpectedAcceptedTokens: expectedAccepted,
		AutoregressiveTPOTSec:  autoTPOT,
		SpeculativeTPOTSec:     specTPOT,
		Speedup:                speedup,
		Viable:                 speedup > 1.0,
	}

	if findOptimal {
		searchMaxK := 16
		if draftLength > 8 {
			searchMaxK = draftLength * 2
		}
		if searchMaxK > 64 {
			searchMaxK = 64
		}
		bestK, bestSpeedup := FindOptimalDraftLength(hw, target, draft, acceptance, batchSize, seqLen, searchMaxK)
		res.OptimalK = bestK
		res.OptimalSpeedup = bestSpeedup
	} else {
		res.OptimalK = draftLength
		res.OptimalSpeedup = speedup
	}

	return res
}

// ProjectSpeculativeSpeedup evaluates the analytical speedup of speculative decoding given hardware limits,
// target and draft model profiles, and acceptance statistics:
//
//	S = (E[N_acc] * T_target_autoregressive) / (K * T_draft + T_verify)
//
// For tree speculation (isTree = true), verifies K candidate tokens in a single parallel verification step
// while amortizing draft proposal generation.
func ProjectSpeculativeSpeedup(hw HardwareProfile, target, draft ModelCostProfile, acceptance SpecAcceptanceProfile, batchSize, seqLen, draftLength int, isTree bool) SpeculativeProjectionResult {
	return projectSpeculativeSpeedupInternal(hw, target, draft, acceptance, batchSize, seqLen, draftLength, isTree, true)
}

// FindOptimalDraftLength searches draft length horizon k in [1, maxK] to locate the optimal
// draft length K* that maximizes projected speedup.
func FindOptimalDraftLength(hw HardwareProfile, target, draft ModelCostProfile, acceptance SpecAcceptanceProfile, batchSize, seqLen, maxK int) (int, float64) {
	if maxK <= 0 {
		maxK = 10
	}
	if maxK > 64 {
		maxK = 64
	}
	if batchSize <= 0 {
		batchSize = 1
	}
	if seqLen < 0 {
		seqLen = 0
	}

	isTree := acceptance.Method == "tree" || acceptance.TreeYield > 0

	bestK := 1
	bestSpeedup := 0.0

	for k := 1; k <= maxK; k++ {
		res := projectSpeculativeSpeedupInternal(hw, target, draft, acceptance, batchSize, seqLen, k, isTree, false)
		if k == 1 || res.Speedup > bestSpeedup {
			bestK = k
			bestSpeedup = res.Speedup
		}
	}

	return bestK, bestSpeedup
}
