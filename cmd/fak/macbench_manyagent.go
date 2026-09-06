package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
)

const (
	ManyAgentSchema             = "fak.macbench.manyagent.v1"
	ManyAgentComparisonSchema   = "fak.macbench.manyagent-compare.v1"
	DefaultSharedPrefixTokens   = 4096
	DefaultTurnDeltaTokens      = 128
	DefaultTurnOutputTokens     = 64
	DefaultManyAgentConcurrency = 4
	DefaultManyAgentHorizon     = 20
	DefaultManyAgentModel       = "Qwen3.8-27B"
	DefaultManyAgentProvenance  = "MODELED"
)

var defaultManyAgentUnmodeledEffects = []string{
	"thermal_dvfs_throttling",
	"memory_bus_contention",
	"queue_sync_jitter",
}

func init() {
	if len(os.Args) > 2 && os.Args[1] == "macbench" && os.Args[2] == "many-agent" {
		os.Exit(runMacBenchManyAgent(os.Stdout, os.Stderr, os.Args[3:]))
	}
}

// ManyAgentOptions configures the many-agent spine simulation.
type ManyAgentOptions struct {
	Concurrency        int    `json:"concurrency"`
	Model              string `json:"model"`
	Horizon            int    `json:"horizon"`
	Cache              bool   `json:"cache"`
	Output             string `json:"output"`
	SharedPrefixTokens int    `json:"shared_prefix_tokens,omitempty"`
	CompareLlama       bool   `json:"compare_llama,omitempty"`
}

// ManyAgentReport contains the computed cache value metrics and verification status.
type ManyAgentReport struct {
	Schema             string   `json:"schema"`
	Provenance         string   `json:"provenance"`
	IsPhysicalSilicon  bool     `json:"is_physical_silicon"`
	UnmodeledEffects   []string `json:"unmodeled_effects,omitempty"`
	Model              string   `json:"model"`
	Concurrency        int      `json:"concurrency"`
	Horizon            int      `json:"horizon"`
	Cache              bool     `json:"cache"`
	SharedPrefixTokens int      `json:"shared_prefix_tokens"`
	PromptTokens       uint64   `json:"prompt_tokens"`
	ReusedTokens       uint64   `json:"reused_tokens"`
	ReuseRatio         float64  `json:"reuse_ratio"`
	AgentsPerGB        float64  `json:"agents_per_gb"`
	P50TTFTMS          float64  `json:"p50_ttft_ms"`
	P95TTFTMS          float64  `json:"p95_ttft_ms"`
	PeakMemoryMB       float64  `json:"peak_memory_mb"`
	PrefixEvalCount    int      `json:"prefix_eval_count"`
	TotalWallMS        float64  `json:"total_wall_ms,omitempty"`
	EffectiveTokS      float64  `json:"effective_tok_s,omitempty"`
	TTFTFlat           bool     `json:"ttft_flat"`
	Verified           bool     `json:"verified"`
}

type manyAgentReportAlias ManyAgentReport

func (r *ManyAgentReport) UnmarshalJSON(data []byte) error {
	var aux manyAgentReportAlias
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	*r = ManyAgentReport(aux)
	if r.Provenance == "" {
		r.Provenance = DefaultManyAgentProvenance
	}
	if len(r.UnmodeledEffects) == 0 {
		r.UnmodeledEffects = append([]string(nil), defaultManyAgentUnmodeledEffects...)
	}
	return nil
}

func (r ManyAgentReport) MarshalJSON() ([]byte, error) {
	type alias ManyAgentReport
	copy := alias(r)
	if copy.Provenance == "" {
		copy.Provenance = DefaultManyAgentProvenance
	}
	if len(copy.UnmodeledEffects) == 0 {
		copy.UnmodeledEffects = append([]string(nil), defaultManyAgentUnmodeledEffects...)
	}
	return json.Marshal(copy)
}

// ManyAgentComparisonReport records head-to-head comparison between fak-native and llama.cpp.
type ManyAgentComparisonReport struct {
	Schema                 string          `json:"schema"`
	Provenance             string          `json:"provenance"`
	IsPhysicalSilicon      bool            `json:"is_physical_silicon"`
	UnmodeledEffects       []string        `json:"unmodeled_effects,omitempty"`
	Model                  string          `json:"model"`
	Concurrency            int             `json:"concurrency"`
	Horizon                int             `json:"horizon"`
	SharedPrefixTokens     int             `json:"shared_prefix_tokens"`
	FakNative              ManyAgentReport `json:"fak_native"`
	LlamaCPP               ManyAgentReport `json:"llama_cpp"`
	SpeedupRatio           float64         `json:"speedup_ratio"`
	MemorySavedMB          float64         `json:"memory_saved_mb"`
	TTFTSpeedupP50         float64         `json:"ttft_speedup_p50"`
	Turn1ColdTTFTFakMS     float64         `json:"turn1_cold_ttft_fak_ms"`
	Turn1ColdTTFTLlamaMS   float64         `json:"turn1_cold_ttft_llama_ms"`
	SteadyStateTTFTFakMS   float64         `json:"steady_state_ttft_fak_ms"`
	SteadyStateTTFTLlamaMS float64         `json:"steady_state_ttft_llama_ms"`
	Modeled4xProjected     bool            `json:"modeled_4x_projected"`
	True4xAchieved         bool            `json:"true_4x_achieved"`
	Verified               bool            `json:"verified"`
}

type manyAgentComparisonReportAlias ManyAgentComparisonReport

func (r *ManyAgentComparisonReport) UnmarshalJSON(data []byte) error {
	var aux struct {
		manyAgentComparisonReportAlias
		LegacyTrue4x *bool `json:"true_4x_achieved"`
		Modern4x     *bool `json:"modeled_4x_projected"`
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	*r = ManyAgentComparisonReport(aux.manyAgentComparisonReportAlias)
	if r.Provenance == "" {
		r.Provenance = DefaultManyAgentProvenance
	}
	if len(r.UnmodeledEffects) == 0 {
		r.UnmodeledEffects = append([]string(nil), defaultManyAgentUnmodeledEffects...)
	}
	if aux.Modern4x != nil {
		r.Modeled4xProjected = *aux.Modern4x
		r.True4xAchieved = *aux.Modern4x
	} else if aux.LegacyTrue4x != nil {
		r.Modeled4xProjected = *aux.LegacyTrue4x
		r.True4xAchieved = *aux.LegacyTrue4x
	} else {
		val := r.Modeled4xProjected || r.True4xAchieved
		r.Modeled4xProjected = val
		r.True4xAchieved = val
	}
	return nil
}

func (r ManyAgentComparisonReport) MarshalJSON() ([]byte, error) {
	type alias ManyAgentComparisonReport
	copy := alias(r)
	if copy.Provenance == "" {
		copy.Provenance = DefaultManyAgentProvenance
	}
	if len(copy.UnmodeledEffects) == 0 {
		copy.UnmodeledEffects = append([]string(nil), defaultManyAgentUnmodeledEffects...)
	}
	val := copy.Modeled4xProjected || copy.True4xAchieved
	copy.Modeled4xProjected = val
	copy.True4xAchieved = val
	return json.Marshal(copy)
}

// IsModeled4xProjected returns whether 4x speedup was projected under the model.
// Also supports legacy True4xAchieved.
func (r ManyAgentComparisonReport) IsModeled4xProjected() bool {
	return r.Modeled4xProjected || r.True4xAchieved
}

// SetModeled4xProjected sets both Modeled4xProjected and True4xAchieved.
func (r *ManyAgentComparisonReport) SetModeled4xProjected(v bool) {
	r.Modeled4xProjected = v
	r.True4xAchieved = v
}

type manyAgentModelSpec struct {
	Name                 string
	WeightMB             float64
	Layers               uint64
	FullAttnLayers       uint64
	RecurrentLayers      uint64
	RecurrentStateBytes  uint64
	KVHeads              uint64
	HeadDim              uint64
	BytesPerElement      uint64
	BasePrefillTokPerSec float64
	DeltaBaseMS          float64
	OverheadMB           float64
}

// EffectiveKVLayers returns the number of layers that allocate token-indexed KV cache rows.
// For hybrid models (such as Qwen3.8 3:1 GDN), this returns FullAttnLayers.
func (s manyAgentModelSpec) EffectiveKVLayers() uint64 {
	if s.FullAttnLayers > 0 {
		return s.FullAttnLayers
	}
	return s.Layers
}

// HybridRatio returns the ratio of full-attention layers to total layers (e.g. 0.25 for 3:1 GDN).
func (s manyAgentModelSpec) HybridRatio() float64 {
	if s.Layers == 0 {
		return 1.0
	}
	return float64(s.EffectiveKVLayers()) / float64(s.Layers)
}

// KVBytesPerToken returns the token-indexed KV cache bytes required per token.
// For hybrid architectures (like Qwen3.8 3:1 GDN), only full-attention layers maintain
// token-indexed KV cache rows; linear-attention layers maintain fixed O(1) recurrent state.
func (s manyAgentModelSpec) KVBytesPerToken() uint64 {
	return 2 * s.EffectiveKVLayers() * s.KVHeads * s.HeadDim * s.BytesPerElement
}

// KVBytes returns the token-indexed KV cache bytes for a given token count.
func (s manyAgentModelSpec) KVBytes(tokens uint64) uint64 {
	return tokens * s.KVBytesPerToken()
}

// RecurrentStateMB returns the per-agent fixed recurrent state memory in MB.
func (s manyAgentModelSpec) RecurrentStateMB() float64 {
	return float64(s.RecurrentStateBytes) / (1024.0 * 1024.0)
}

// EstimateKVMemoryBytes returns the total KV cache bytes for a given token count and agent concurrency,
// including token-indexed KV cache rows and O(1) recurrent state.
func (s manyAgentModelSpec) EstimateKVMemoryBytes(tokens uint64, concurrency int) uint64 {
	if concurrency <= 0 {
		concurrency = 1
	}
	return tokens*s.KVBytesPerToken() + uint64(concurrency)*s.RecurrentStateBytes
}

// EstimateKVMemoryMB returns the KV cache memory in MB for a given token count and agent concurrency.
func (s manyAgentModelSpec) EstimateKVMemoryMB(tokens uint64, concurrency int) float64 {
	return float64(s.EstimateKVMemoryBytes(tokens, concurrency)) / (1024.0 * 1024.0)
}

const (
	// DefaultQwen38RecurrentStateBytes is the fixed O(1) recurrent matrix and short-convolution
	// state per agent for Qwen3.8-27B (48 linear-attention layers, 156,893,184 bytes ~= 149.6 MB,
	// matching internal/model/contextsize.go).
	DefaultQwen38RecurrentStateBytes uint64 = 156893184
)

func resolveManyAgentModelSpec(modelName string) manyAgentModelSpec {
	lower := strings.ToLower(strings.TrimSpace(modelName))
	switch {
	case strings.Contains(lower, "27b"):
		weightMB := 16384.0 // ~16.0 GB for Q4_K_M weights
		prefillTokS := 65.0
		if strings.Contains(lower, "q2") || strings.Contains(lower, "2bit") || strings.Contains(lower, "2-bit") || strings.Contains(lower, "ud-q2") {
			weightMB = 9830.0 // ~9.83 GB for UD-Q2_K_XL weights
			prefillTokS = 85.0
		}
		return manyAgentModelSpec{
			Name:                 modelName,
			WeightMB:             weightMB,
			Layers:               64,
			FullAttnLayers:       16,
			RecurrentLayers:      48,
			RecurrentStateBytes:  DefaultQwen38RecurrentStateBytes,
			KVHeads:              8,
			HeadDim:              128,
			BytesPerElement:      2,
			BasePrefillTokPerSec: prefillTokS,
			DeltaBaseMS:          12.0,
			OverheadMB:           1024.0,
		}
	case strings.Contains(lower, "3b"):
		return manyAgentModelSpec{
			Name:                 modelName,
			WeightMB:             2048.0,
			Layers:               28,
			FullAttnLayers:       28,
			RecurrentLayers:      0,
			RecurrentStateBytes:  0,
			KVHeads:              8,
			HeadDim:              128,
			BytesPerElement:      2,
			BasePrefillTokPerSec: 380.0,
			DeltaBaseMS:          2.5,
			OverheadMB:           384.0,
		}
	case strings.Contains(lower, "4b"):
		return manyAgentModelSpec{
			Name:                 modelName,
			WeightMB:             2800.0,
			Layers:               32,
			FullAttnLayers:       32,
			RecurrentLayers:      0,
			RecurrentStateBytes:  0,
			KVHeads:              8,
			HeadDim:              128,
			BytesPerElement:      2,
			BasePrefillTokPerSec: 320.0,
			DeltaBaseMS:          3.2,
			OverheadMB:           448.0,
		}
	case strings.Contains(lower, "7b"):
		return manyAgentModelSpec{
			Name:                 modelName,
			WeightMB:             4608.0,
			Layers:               28,
			FullAttnLayers:       28,
			RecurrentLayers:      0,
			RecurrentStateBytes:  0,
			KVHeads:              4,
			HeadDim:              128,
			BytesPerElement:      2,
			BasePrefillTokPerSec: 220.0,
			DeltaBaseMS:          4.5,
			OverheadMB:           512.0,
		}
	default: // Qwen3.8-27B or other 27B / general default
		return manyAgentModelSpec{
			Name:                 modelName,
			WeightMB:             16384.0, // ~16.0 GB for Q4_K_M weights
			Layers:               64,
			FullAttnLayers:       16,
			RecurrentLayers:      48,
			RecurrentStateBytes:  DefaultQwen38RecurrentStateBytes,
			KVHeads:              8,
			HeadDim:              128,
			BytesPerElement:      2,
			BasePrefillTokPerSec: 65.0,
			DeltaBaseMS:          12.0,
			OverheadMB:           1024.0,
		}
	}
}

func turnPromptLen(prefix, t, deltaIn, deltaOut int) int {
	return prefix + t*deltaIn + (t-1)*deltaOut
}

// ValidateManyAgentOptions validates user-supplied options.
func ValidateManyAgentOptions(opts ManyAgentOptions) error {
	if opts.Concurrency <= 0 {
		return fmt.Errorf("--concurrency must be positive (got %d)", opts.Concurrency)
	}
	if opts.Horizon <= 0 {
		return fmt.Errorf("--horizon must be positive (got %d)", opts.Horizon)
	}
	if opts.SharedPrefixTokens < 0 {
		return fmt.Errorf("--prefix-tokens must be non-negative")
	}
	if strings.TrimSpace(opts.Model) == "" {
		return fmt.Errorf("--model must not be empty")
	}
	out := strings.ToLower(strings.TrimSpace(opts.Output))
	if out != "" && out != "summary" && out != "json" {
		return fmt.Errorf("invalid --output %q (must be summary or json)", opts.Output)
	}
	return nil
}

// RunManyAgentSpine executes the modeled multi-agent benchmark on Apple Silicon Metal.
func RunManyAgentSpine(opts ManyAgentOptions) (ManyAgentReport, error) {
	if err := ValidateManyAgentOptions(opts); err != nil {
		return ManyAgentReport{}, err
	}

	prefix := opts.SharedPrefixTokens
	if prefix < 0 {
		prefix = DefaultSharedPrefixTokens
	}

	spec := resolveManyAgentModelSpec(opts.Model)

	// 1. Calculate prompt tokens across all K agents and H turns.
	var promptTokens uint64
	for t := 1; t <= opts.Horizon; t++ {
		promptTokens += uint64(turnPromptLen(prefix, t, DefaultTurnDeltaTokens, DefaultTurnOutputTokens))
	}
	promptTokens *= uint64(opts.Concurrency)

	// 2. Calculate reused tokens.
	var reusedTokens uint64
	if opts.Cache {
		// Turn 1: Agent 1 performs cold prefill. Agents 2..K reuse the shared prefix.
		if opts.Concurrency > 1 {
			reusedTokens += uint64(opts.Concurrency-1) * uint64(prefix)
		}
		// Turns 2..H: all K agents reuse prior context (all tokens except the new deltaIn).
		for t := 2; t <= opts.Horizon; t++ {
			prevTokens := turnPromptLen(prefix, t, DefaultTurnDeltaTokens, DefaultTurnOutputTokens) - DefaultTurnDeltaTokens
			reusedTokens += uint64(opts.Concurrency) * uint64(prevTokens)
		}
	} else {
		reusedTokens = 0
	}

	reuseRatio := 0.0
	if promptTokens > 0 {
		reuseRatio = float64(reusedTokens) / float64(promptTokens)
	}

	// 3. Memory calculation.
	// KV bytes per token = 2 * effective_layers * kv_heads * head_dim * bytes_per_element.
	kvBytesPerToken := spec.KVBytesPerToken()
	sharedPrefixKVBytes := uint64(prefix) * kvBytesPerToken
	tailTokensPerAgent := opts.Horizon*DefaultTurnDeltaTokens + (opts.Horizon-1)*DefaultTurnOutputTokens

	var totalKVBytes uint64
	if opts.Cache {
		// Shared prefix stored ONCE across all K agents + private tail per agent.
		totalKVBytes = sharedPrefixKVBytes + uint64(opts.Concurrency)*uint64(tailTokensPerAgent)*kvBytesPerToken
	} else {
		// Shared prefix duplicated per agent (K separate full contexts).
		fullContextTokens := uint64(prefix + tailTokensPerAgent)
		totalKVBytes = uint64(opts.Concurrency) * fullContextTokens * kvBytesPerToken
	}

	// Fixed O(1) recurrent state per concurrent agent.
	totalKVBytes += uint64(opts.Concurrency) * spec.RecurrentStateBytes

	kvMemoryMB := float64(totalKVBytes) / (1024.0 * 1024.0)
	peakMemoryMB := spec.WeightMB + kvMemoryMB + spec.OverheadMB
	peakMemoryGB := peakMemoryMB / 1024.0

	agentsPerGB := 0.0
	if peakMemoryGB > 0 {
		agentsPerGB = float64(opts.Concurrency) / peakMemoryGB
	}

	// 4. TTFT distribution modeling over K * H turns.
	latencies := make([]float64, 0, opts.Concurrency*opts.Horizon)
	prefixEvalCount := 0

	if opts.Cache {
		prefixEvalCount = 1
		// Turn 1, Agent 1: cold prefill of shared prefix + initial delta
		coldMS := (float64(prefix+DefaultTurnDeltaTokens) / spec.BasePrefillTokPerSec) * 1000.0
		latencies = append(latencies, coldMS)

		// Turn 1, Agents 2..K: shared prefix cache hit; only evaluate delta
		for k := 2; k <= opts.Concurrency; k++ {
			warmMS := spec.DeltaBaseMS + 1.2 + 0.1*float64(k%3)
			latencies = append(latencies, warmMS)
		}

		// Turns 2..H for all K agents: prefix + history cache hit; only evaluate delta
		for t := 2; t <= opts.Horizon; t++ {
			for k := 1; k <= opts.Concurrency; k++ {
				warmMS := spec.DeltaBaseMS + 0.3*float64((k+t)%4)
				latencies = append(latencies, warmMS)
			}
		}
	} else {
		prefixEvalCount = opts.Concurrency * opts.Horizon
		contention := 1.0 + 0.25*float64(opts.Concurrency-1)
		for t := 1; t <= opts.Horizon; t++ {
			ctxLen := turnPromptLen(prefix, t, DefaultTurnDeltaTokens, DefaultTurnOutputTokens)
			baseMS := (float64(ctxLen) / spec.BasePrefillTokPerSec) * 1000.0
			for k := 1; k <= opts.Concurrency; k++ {
				ttft := baseMS * contention * (1.0 + 0.05*float64((k+t)%5))
				latencies = append(latencies, ttft)
			}
		}
	}

	sort.Float64s(latencies)
	p50 := percentile(latencies, 0.50)
	p95 := percentile(latencies, 0.95)

	// 5. Verification checks:
	// - Prefix evaluated once globally across the fleet
	// - TTFT flat: median TTFT remains bounded at delta prefill speed (< 25 ms)
	ttftFlat := opts.Cache && p50 <= 25.0
	verified := opts.Cache && prefixEvalCount == 1 && ttftFlat

	// 6. Total wall-clock time and throughput modeling.
	totalOutputTokens := float64(opts.Concurrency * opts.Horizon * DefaultTurnOutputTokens)
	var totalWallMS float64
	if opts.Cache {
		prefixMS := (float64(prefix) / spec.BasePrefillTokPerSec) * 1000.0
		deltaPrefillMS := float64(opts.Concurrency*opts.Horizon) * spec.DeltaBaseMS
		batchedTokPerSec := 7.61 * (1.0 + 0.35*float64(opts.Concurrency-1))
		decodeMS := (totalOutputTokens / batchedTokPerSec) * 1000.0
		totalWallMS = prefixMS + deltaPrefillMS + decodeMS + 1600.0
	} else {
		fullPrefillMS := float64(opts.Concurrency*opts.Horizon) * ((float64(prefix) / spec.BasePrefillTokPerSec) * 1000.0) * 0.4
		serialTokPerSec := 7.38
		decodeMS := (totalOutputTokens / serialTokPerSec) * 1000.0
		totalWallMS = fullPrefillMS + decodeMS + 25000.0
	}
	effectiveTokS := 0.0
	if totalWallMS > 0 {
		effectiveTokS = math.Round((totalOutputTokens/(totalWallMS/1000.0))*100) / 100
	}

	rep := ManyAgentReport{
		Schema:             ManyAgentSchema,
		Provenance:         DefaultManyAgentProvenance,
		IsPhysicalSilicon:  false,
		UnmodeledEffects:   append([]string(nil), defaultManyAgentUnmodeledEffects...),
		Model:              opts.Model,
		Concurrency:        opts.Concurrency,
		Horizon:            opts.Horizon,
		Cache:              opts.Cache,
		SharedPrefixTokens: prefix,
		PromptTokens:       promptTokens,
		ReusedTokens:       reusedTokens,
		ReuseRatio:         math.Round(reuseRatio*10000) / 10000,
		AgentsPerGB:        math.Round(agentsPerGB*100) / 100,
		P50TTFTMS:          math.Round(p50*10) / 10,
		P95TTFTMS:          math.Round(p95*10) / 10,
		PeakMemoryMB:       math.Round(peakMemoryMB*10) / 10,
		PrefixEvalCount:    prefixEvalCount,
		TotalWallMS:        math.Round(totalWallMS*10) / 10,
		EffectiveTokS:      effectiveTokS,
		TTFTFlat:           ttftFlat,
		Verified:           verified,
	}

	return rep, nil
}

// RunManyAgentComparison executes the head-to-head comparison between fak-native (inkernel)
// and llama.cpp (reference) on agentic shared cache workloads.
func RunManyAgentComparison(opts ManyAgentOptions) (ManyAgentComparisonReport, error) {
	if err := ValidateManyAgentOptions(opts); err != nil {
		return ManyAgentComparisonReport{}, err
	}
	prefix := opts.SharedPrefixTokens
	if prefix < 0 {
		prefix = DefaultSharedPrefixTokens
	}
	spec := resolveManyAgentModelSpec(opts.Model)

	// 1. fak-native arm: RadixAttention prefix caching ON
	fakOpts := opts
	fakOpts.Cache = true
	fakRep, err := RunManyAgentSpine(fakOpts)
	if err != nil {
		return ManyAgentComparisonReport{}, fmt.Errorf("fak-native run: %w", err)
	}

	// 2. llama.cpp reference arm: slot-isolated multi-slot serving
	totalOutputTokens := float64(opts.Concurrency * opts.Horizon * DefaultTurnOutputTokens)
	singlePrefixMS := (float64(prefix) / spec.BasePrefillTokPerSec) * 1000.0
	llamaPrefixPrefillMS := float64(opts.Concurrency) * singlePrefixMS
	// Client queue contention is reported in client-side TTFT/latency metrics (turn1TTFTs below);
	// it must not be added to server wall-clock processing time to avoid double-counting.
	llamaQueueContentionMS := float64((opts.Concurrency-1)*opts.Concurrency/2) * singlePrefixMS
	_ = llamaQueueContentionMS
	llamaDeltaPrefillMS := float64(opts.Concurrency*opts.Horizon) * spec.DeltaBaseMS * (1.0 + 0.25*float64(opts.Concurrency-1))
	// Slot context fragmentation penalty is zeroed out unless physically evidenced to avoid
	// inflating modeled server wall-clock duration without physical measurement.
	llamaSlotContentionMS := 0.0
	_ = llamaSlotContentionMS
	llamaPrefillMS := llamaPrefixPrefillMS + llamaDeltaPrefillMS
	llamaDecodeTokPerSec := 7.38 * (1.0 + 0.15*float64(opts.Concurrency-1))
	llamaDecodeMS := (totalOutputTokens / llamaDecodeTokPerSec) * 1000.0
	llamaTotalWallMS := llamaPrefillMS + llamaDecodeMS

	// Latency distribution for llama.cpp across all K * H turns:
	// Turn 1 suffers serialized prefill wait across slots, while Turns 2..H operate at delta prefill latency.
	singleTurn1PrefillMS := singlePrefixMS
	if singleTurn1PrefillMS == 0 {
		singleTurn1PrefillMS = spec.DeltaBaseMS
	}
	llamaTurn1TTFTs := make([]float64, 0, opts.Concurrency)
	for k := 1; k <= opts.Concurrency; k++ {
		llamaTurn1TTFTs = append(llamaTurn1TTFTs, float64(k)*singleTurn1PrefillMS)
	}

	llamaSteadyDeltaMS := spec.DeltaBaseMS * (1.0 + 0.25*float64(opts.Concurrency-1))
	steadyCount := 0
	if opts.Horizon > 1 {
		steadyCount = opts.Concurrency * (opts.Horizon - 1)
	}
	llamaSteadyTTFTs := make([]float64, 0, steadyCount)
	for t := 2; t <= opts.Horizon; t++ {
		for k := 1; k <= opts.Concurrency; k++ {
			llamaSteadyTTFTs = append(llamaSteadyTTFTs, llamaSteadyDeltaMS)
		}
	}

	llamaTTFTs := make([]float64, 0, opts.Concurrency*opts.Horizon)
	llamaTTFTs = append(llamaTTFTs, llamaTurn1TTFTs...)
	llamaTTFTs = append(llamaTTFTs, llamaSteadyTTFTs...)

	llamaTurn1Sorted := append([]float64(nil), llamaTurn1TTFTs...)
	sort.Float64s(llamaTurn1Sorted)
	turn1ColdTTFTLlamaMS := percentile(llamaTurn1Sorted, 0.50)

	steadyStateTTFTLlamaMS := 0.0
	if len(llamaSteadyTTFTs) > 0 {
		llamaSteadySorted := append([]float64(nil), llamaSteadyTTFTs...)
		sort.Float64s(llamaSteadySorted)
		steadyStateTTFTLlamaMS = percentile(llamaSteadySorted, 0.50)
	}

	sort.Float64s(llamaTTFTs)
	llamaP50 := percentile(llamaTTFTs, 0.50)
	llamaP95 := percentile(llamaTTFTs, 0.95)

	// Turn 1 and steady-state TTFT distributions for fak-native
	fakTurn1TTFTs := make([]float64, 0, opts.Concurrency)
	coldMS := (float64(prefix+DefaultTurnDeltaTokens) / spec.BasePrefillTokPerSec) * 1000.0
	fakTurn1TTFTs = append(fakTurn1TTFTs, coldMS)
	for k := 2; k <= opts.Concurrency; k++ {
		fakTurn1TTFTs = append(fakTurn1TTFTs, spec.DeltaBaseMS+1.2+0.1*float64(k%3))
	}
	sort.Float64s(fakTurn1TTFTs)
	turn1ColdTTFTFakMS := percentile(fakTurn1TTFTs, 0.50)

	fakSteadyTTFTs := make([]float64, 0, steadyCount)
	for t := 2; t <= opts.Horizon; t++ {
		for k := 1; k <= opts.Concurrency; k++ {
			fakSteadyTTFTs = append(fakSteadyTTFTs, spec.DeltaBaseMS+0.3*float64((k+t)%4))
		}
	}
	steadyStateTTFTFakMS := 0.0
	if len(fakSteadyTTFTs) > 0 {
		sort.Float64s(fakSteadyTTFTs)
		steadyStateTTFTFakMS = percentile(fakSteadyTTFTs, 0.50)
	}

	// Memory footprint for llama.cpp: K independent full contexts
	kvBytesPerToken := spec.KVBytesPerToken()
	tailTokensPerAgent := opts.Horizon*DefaultTurnDeltaTokens + (opts.Horizon-1)*DefaultTurnOutputTokens
	llamaTotalKVBytes := uint64(opts.Concurrency) * uint64(prefix+tailTokensPerAgent) * kvBytesPerToken
	llamaTotalKVBytes += uint64(opts.Concurrency) * spec.RecurrentStateBytes
	llamaKVMB := float64(llamaTotalKVBytes) / (1024.0 * 1024.0)
	llamaPeakMemoryMB := spec.WeightMB + llamaKVMB + spec.OverheadMB
	llamaAgentsPerGB := 0.0
	if llamaPeakMemoryMB > 0 {
		llamaAgentsPerGB = float64(opts.Concurrency) / (llamaPeakMemoryMB / 1024.0)
	}

	llamaEffectiveTokS := 0.0
	if llamaTotalWallMS > 0 {
		llamaEffectiveTokS = math.Round((totalOutputTokens/(llamaTotalWallMS/1000.0))*100) / 100
	}

	llamaRep := ManyAgentReport{
		Schema:             ManyAgentSchema,
		Provenance:         DefaultManyAgentProvenance,
		IsPhysicalSilicon:  false,
		UnmodeledEffects:   append([]string(nil), defaultManyAgentUnmodeledEffects...),
		Model:              opts.Model,
		Concurrency:        opts.Concurrency,
		Horizon:            opts.Horizon,
		Cache:              false,
		SharedPrefixTokens: prefix,
		PromptTokens:       fakRep.PromptTokens,
		ReusedTokens:       0,
		ReuseRatio:         0.0,
		AgentsPerGB:        math.Round(llamaAgentsPerGB*100) / 100,
		P50TTFTMS:          math.Round(llamaP50*10) / 10,
		P95TTFTMS:          math.Round(llamaP95*10) / 10,
		PeakMemoryMB:       math.Round(llamaPeakMemoryMB*10) / 10,
		PrefixEvalCount:    opts.Concurrency,
		TotalWallMS:        math.Round(llamaTotalWallMS*10) / 10,
		EffectiveTokS:      llamaEffectiveTokS,
		TTFTFlat:           false,
		Verified:           false,
	}

	speedupRatio := 0.0
	if fakRep.TotalWallMS > 0 {
		speedupRatio = math.Round((llamaRep.TotalWallMS/fakRep.TotalWallMS)*100) / 100
	}
	memorySavedMB := math.Round((llamaRep.PeakMemoryMB-fakRep.PeakMemoryMB)*10) / 10
	ttftSpeedupP50 := 0.0
	if fakRep.P50TTFTMS > 0 {
		ttftSpeedupP50 = math.Round((llamaRep.P50TTFTMS/fakRep.P50TTFTMS)*100) / 100
	}
	modeled4xProjected := speedupRatio >= 4.0
	verified := fakRep.Verified && speedupRatio > 1.0 && (memorySavedMB > 0 || prefix == 0)

	return ManyAgentComparisonReport{
		Schema:                 ManyAgentComparisonSchema,
		Provenance:             DefaultManyAgentProvenance,
		IsPhysicalSilicon:      false,
		UnmodeledEffects:       append([]string(nil), defaultManyAgentUnmodeledEffects...),
		Model:                  opts.Model,
		Concurrency:            opts.Concurrency,
		Horizon:                opts.Horizon,
		SharedPrefixTokens:     prefix,
		FakNative:              fakRep,
		LlamaCPP:               llamaRep,
		SpeedupRatio:           speedupRatio,
		MemorySavedMB:          memorySavedMB,
		TTFTSpeedupP50:         ttftSpeedupP50,
		Turn1ColdTTFTFakMS:     math.Round(turn1ColdTTFTFakMS*10) / 10,
		Turn1ColdTTFTLlamaMS:   math.Round(turn1ColdTTFTLlamaMS*10) / 10,
		SteadyStateTTFTFakMS:   math.Round(steadyStateTTFTFakMS*10) / 10,
		SteadyStateTTFTLlamaMS: math.Round(steadyStateTTFTLlamaMS*10) / 10,
		Modeled4xProjected:     modeled4xProjected,
		True4xAchieved:         modeled4xProjected,
		Verified:               verified,
	}, nil
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(p*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func runMacBenchManyAgent(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("macbench many-agent", flag.ContinueOnError)
	fs.SetOutput(stderr)

	concurrency := fs.Int("concurrency", DefaultManyAgentConcurrency, "number of concurrent agents K")
	fs.IntVar(concurrency, "c", DefaultManyAgentConcurrency, "number of concurrent agents K (shorthand)")
	prefixTokens := fs.Int("prefix-tokens", DefaultSharedPrefixTokens, "shared prefix length in tokens")
	fs.IntVar(prefixTokens, "p", DefaultSharedPrefixTokens, "shared prefix length in tokens (shorthand)")
	model := fs.String("model", DefaultManyAgentModel, "model identifier")
	horizon := fs.Int("horizon", DefaultManyAgentHorizon, "turns per agent")
	cache := fs.Bool("cache", true, "enable fak KV prefix caching")
	compareLlama := fs.Bool("compare-llama", false, "run head-to-head comparison against llama.cpp reference")
	output := fs.String("output", "summary", "output format: summary or json")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON (shorthand for --output json)")

	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	outFmt := strings.ToLower(strings.TrimSpace(*output))
	if *asJSON {
		outFmt = "json"
	}

	opts := ManyAgentOptions{
		Concurrency:        *concurrency,
		Model:              *model,
		Horizon:            *horizon,
		Cache:              *cache,
		Output:             outFmt,
		SharedPrefixTokens: *prefixTokens,
		CompareLlama:       *compareLlama,
	}

	if *compareLlama {
		compRep, err := RunManyAgentComparison(opts)
		if err != nil {
			fmt.Fprintf(stderr, "fak macbench many-agent: %v\n", err)
			return 2
		}
		if outFmt == "json" {
			_ = writeIndentedJSONNoEscape(stdout, compRep)
		} else {
			printManyAgentComparisonSummary(stdout, compRep)
		}
		if !compRep.Verified {
			return 1
		}
		return 0
	}

	rep, err := RunManyAgentSpine(opts)
	if err != nil {
		fmt.Fprintf(stderr, "fak macbench many-agent: %v\n", err)
		return 2
	}

	if outFmt == "json" {
		_ = writeIndentedJSONNoEscape(stdout, rep)
	} else {
		printManyAgentSummary(stdout, rep)
	}

	return 0
}

func printManyAgentComparisonSummary(w io.Writer, rep ManyAgentComparisonReport) {
	fmt.Fprintf(w, "======================================================================\n")
	fmt.Fprintf(w, "[MODELED PROJECTION] Grounded in measured single-stream rates (#2723/#9513)\n")
	fmt.Fprintf(w, "Hardware target: Apple Silicon Metal (Apple M3 Pro 36GB, node-macos-a)\n")
	fmt.Fprintf(w, "Unmodeled effects: Thermal DVFS, memory bus contention, queue sync jitter\n")
	fmt.Fprintf(w, "======================================================================\n")
	fmt.Fprintf(w, "fak macbench many-agent head-to-head: model=%s concurrency=%d horizon=%d\n",
		rep.Model, rep.Concurrency, rep.Horizon)
	fmt.Fprintf(w, "shared_prefix         : %d tokens (system + tools)\n", rep.SharedPrefixTokens)
	fmt.Fprintf(w, "%-22s %-24s %-24s %s\n", "metric", "fak-native (inkernel)", "llama.cpp (reference)", "gain")
	fmt.Fprintf(w, "%-22s %-24d %-24d %s\n", "prefix_evals",
		rep.FakNative.PrefixEvalCount, rep.LlamaCPP.PrefixEvalCount,
		fmt.Sprintf("%.1fx less prefill", float64(rep.LlamaCPP.PrefixEvalCount)/float64(rep.FakNative.PrefixEvalCount)))
	fmt.Fprintf(w, "%-22s %-24s %-24s %s\n", "reused_tokens",
		fmt.Sprintf("%d (%.1f%%)", rep.FakNative.ReusedTokens, rep.FakNative.ReuseRatio*100),
		fmt.Sprintf("%d (%.1f%%)", rep.LlamaCPP.ReusedTokens, rep.LlamaCPP.ReuseRatio*100),
		fmt.Sprintf("+%.1f%% reuse", (rep.FakNative.ReuseRatio-rep.LlamaCPP.ReuseRatio)*100))
	fmt.Fprintf(w, "%-22s %-24s %-24s %s\n", "peak_memory_mb",
		fmt.Sprintf("%.1f MB (%.2f GB)", rep.FakNative.PeakMemoryMB, rep.FakNative.PeakMemoryMB/1024.0),
		fmt.Sprintf("%.1f MB (%.2f GB)", rep.LlamaCPP.PeakMemoryMB, rep.LlamaCPP.PeakMemoryMB/1024.0),
		fmt.Sprintf("%.1f MB saved", rep.MemorySavedMB))
	fmt.Fprintf(w, "%-22s %-24s %-24s %s\n", "p50_ttft_ms",
		fmt.Sprintf("%.1f ms", rep.FakNative.P50TTFTMS),
		fmt.Sprintf("%.1f ms", rep.LlamaCPP.P50TTFTMS),
		fmt.Sprintf("%.1fx faster", rep.TTFTSpeedupP50))
	if rep.Turn1ColdTTFTFakMS > 0 && rep.Turn1ColdTTFTLlamaMS > 0 {
		fmt.Fprintf(w, "%-22s %-24s %-24s %s\n", "turn1_cold_ttft_ms",
			fmt.Sprintf("%.1f ms", rep.Turn1ColdTTFTFakMS),
			fmt.Sprintf("%.1f ms", rep.Turn1ColdTTFTLlamaMS),
			fmt.Sprintf("%.1fx faster", rep.Turn1ColdTTFTLlamaMS/rep.Turn1ColdTTFTFakMS))
	}
	if rep.SteadyStateTTFTFakMS > 0 && rep.SteadyStateTTFTLlamaMS > 0 {
		fmt.Fprintf(w, "%-22s %-24s %-24s %s\n", "steady_ttft_ms",
			fmt.Sprintf("%.1f ms", rep.SteadyStateTTFTFakMS),
			fmt.Sprintf("%.1f ms", rep.SteadyStateTTFTLlamaMS),
			fmt.Sprintf("%.1fx faster", rep.SteadyStateTTFTLlamaMS/rep.SteadyStateTTFTFakMS))
	}
	fmt.Fprintf(w, "%-22s %-24s %-24s %s\n", "total_wall_clock",
		fmt.Sprintf("%.1f s", rep.FakNative.TotalWallMS/1000.0),
		fmt.Sprintf("%.1f s", rep.LlamaCPP.TotalWallMS/1000.0),
		fmt.Sprintf("%.2fx speedup", rep.SpeedupRatio))
	fmt.Fprintf(w, "%-22s %-24s %-24s %s\n", "effective_tok_s",
		fmt.Sprintf("%.2f tok/s", rep.FakNative.EffectiveTokS),
		fmt.Sprintf("%.2f tok/s", rep.LlamaCPP.EffectiveTokS),
		fmt.Sprintf("%.2fx throughput", rep.SpeedupRatio))
	if rep.Verified {
		if rep.Modeled4xProjected || rep.True4xAchieved {
			fmt.Fprintf(w, "verification          : PROJECTED (MODELED %.2fx wall-clock speedup >= 4.0x projected)\n", rep.SpeedupRatio)
		} else {
			fmt.Fprintf(w, "verification          : PROJECTED (MODELED %.2fx wall-clock speedup projected)\n", rep.SpeedupRatio)
		}
	} else {
		fmt.Fprintf(w, "verification          : FAIL (speedup %.2fx <= 1.0x or unverified)\n", rep.SpeedupRatio)
	}
}

func printManyAgentSummary(w io.Writer, rep ManyAgentReport) {
	fmt.Fprintf(w, "======================================================================\n")
	fmt.Fprintf(w, "[MODELED PROJECTION] Grounded in measured single-stream rates (#2723/#9513)\n")
	fmt.Fprintf(w, "Hardware target: Apple Silicon Metal (Apple M3 Pro 36GB, node-macos-a)\n")
	fmt.Fprintf(w, "Unmodeled effects: Thermal DVFS, memory bus contention, queue sync jitter\n")
	fmt.Fprintf(w, "======================================================================\n")
	fmt.Fprintf(w, "fak macbench many-agent: model=%s concurrency=%d horizon=%d cache=%v\n",
		rep.Model, rep.Concurrency, rep.Horizon, rep.Cache)
	fmt.Fprintf(w, "prefix        : %d tokens (system + tools)\n", rep.SharedPrefixTokens)
	fmt.Fprintf(w, "prompt_tokens : %d\n", rep.PromptTokens)
	fmt.Fprintf(w, "reused_tokens : %d (%.1f%% reuse)\n", rep.ReusedTokens, rep.ReuseRatio*100)
	fmt.Fprintf(w, "peak_memory_mb: %.1f MB (%.2f GB)\n", rep.PeakMemoryMB, rep.PeakMemoryMB/1024.0)
	fmt.Fprintf(w, "agents_per_gb : %.2f agents/GB\n", rep.AgentsPerGB)
	fmt.Fprintf(w, "p50_ttft_ms   : %.1f ms\n", rep.P50TTFTMS)
	fmt.Fprintf(w, "p95_ttft_ms   : %.1f ms\n", rep.P95TTFTMS)
	fmt.Fprintf(w, "prefix_evals  : %d\n", rep.PrefixEvalCount)
	fmt.Fprintf(w, "ttft_flat     : %v\n", rep.TTFTFlat)
	if rep.Verified {
		fmt.Fprintln(w, "verification  : PASS (prefix evaluated once, TTFT flat under concurrency)")
	} else {
		fmt.Fprintln(w, "verification  : FAIL (caching disabled or prefix re-evaluated)")
	}
}
