package modelperfobs

import (
	"fmt"
	"math"
)

// LongContextEnvelopeSchema identifies the serialized estimator result schema.
const LongContextEnvelopeSchema = "fak-long-context-envelope/2"

// ClosedRange is an inclusive uncertainty interval. All estimator ranges use
// the same unit as the field containing them.
type ClosedRange struct {
	Min float64 `json:"min"`
	Max float64 `json:"max"`
}

// LongContextEstimatorInput describes one bounded serving-demand envelope.
// PrefillTokens plus DecodeTokens is the requested work per agent.
// ResidentContextTokens is the KV capacity reserved per resident agent; it must
// cover that request. It is not the model's maximum supported context, which
// callers must check separately against model metadata.
type LongContextEstimatorInput struct {
	TotalParameters       float64     `json:"total_parameters"`
	ActiveParameters      float64     `json:"active_parameters"`
	WeightBits            ClosedRange `json:"weight_bits"`
	MetadataOverhead      ClosedRange `json:"metadata_overhead_fraction"`
	KVBytesPerToken       ClosedRange `json:"kv_bytes_per_token"`
	ResidentContextTokens uint64      `json:"resident_context_tokens"`
	ResidentAgents        uint64      `json:"resident_agents"`
	Concurrency           uint64      `json:"concurrency"`
	PrefillTokens         uint64      `json:"prefill_tokens"`
	DecodeTokens          uint64      `json:"decode_tokens"`
	UsableMemoryBytes     float64     `json:"usable_memory_bytes"`
	BandwidthBytesPerSec  ClosedRange `json:"bandwidth_bytes_per_second"`
	ComputeFLOPS          ClosedRange `json:"compute_flops"`
	Efficiency            ClosedRange `json:"efficiency_fraction"`
	PrefillCacheHit       ClosedRange `json:"prefill_cache_hit_fraction"`
}

// RooflineTimeBounds contains inclusive compute, memory, and roofline time
// intervals for one serving phase.
type RooflineTimeBounds struct {
	ComputeSeconds ClosedRange `json:"compute_seconds"`
	MemorySeconds  ClosedRange `json:"memory_seconds"`
	TimeSeconds    ClosedRange `json:"time_seconds"`
	Bottleneck     string      `json:"bottleneck"`
}

// LongContextEnvelope is the deterministic memory and service-time envelope
// derived from a LongContextEstimatorInput.
type LongContextEnvelope struct {
	Schema                   string      `json:"schema"`
	ModelMemoryBytes         ClosedRange `json:"model_memory_bytes"`
	KVMemoryPerAgentBytes    ClosedRange `json:"kv_memory_per_agent_bytes"`
	KVMemoryBytes            ClosedRange `json:"kv_memory_bytes"`
	TotalResidentMemoryBytes ClosedRange `json:"total_resident_memory_bytes"`
	Fits                     bool        `json:"fits"`
	FitUncertain             bool        `json:"fit_uncertain"`
	HeadroomBytes            ClosedRange `json:"headroom_bytes"`
	// MaxResidentAgents contains integer-valued lower and upper counts even
	// though ClosedRange is shared with continuous estimator quantities.
	MaxResidentAgents ClosedRange        `json:"max_resident_agents"`
	Prefill           RooflineTimeBounds `json:"prefill"`
	Decode            RooflineTimeBounds `json:"decode"`
	// ServiceTimeSeconds is the completion-time envelope for the whole active
	// batch. It is not per-job latency and excludes scheduler queueing.
	ServiceTimeSeconds ClosedRange `json:"service_time_seconds"`
	// ProcessedTokensPerSec aggregates requested prefill and decode tokens
	// across the active batch.
	ProcessedTokensPerSec ClosedRange `json:"processed_tokens_per_second"`
	// DecodeTokensPerSec aggregates generated decode tokens across the active
	// batch over the decode-phase completion-time envelope. Prefill work does
	// not affect this rate.
	DecodeTokensPerSec ClosedRange `json:"decode_tokens_per_second"`
	Bottleneck         string      `json:"bottleneck"`
	SensitivityNotes   []string    `json:"sensitivity_notes"`
}

// EstimateLongContextEnvelope validates in and returns inclusive bounds for
// memory residency, roofline service time, and aggregate token throughput.
func EstimateLongContextEnvelope(in LongContextEstimatorInput) (LongContextEnvelope, error) {
	if err := validateLongContextInput(in); err != nil {
		return LongContextEnvelope{}, err
	}

	model := mulRange(mulRange(ClosedRange{in.TotalParameters, in.TotalParameters}, scaleRange(in.WeightBits, 1.0/8.0)), addScalar(in.MetadataOverhead, 1))
	kvPerAgent := scaleRange(in.KVBytesPerToken, float64(in.ResidentContextTokens))
	kv := scaleRange(kvPerAgent, float64(in.ResidentAgents))
	total := addRange(model, kv)
	headroom := ClosedRange{Min: in.UsableMemoryBytes - total.Max, Max: in.UsableMemoryBytes - total.Min}
	fits := total.Max <= in.UsableMemoryBytes
	fitUncertain := total.Min <= in.UsableMemoryBytes && !fits

	maxAgents := ClosedRange{}
	availableForKV := ClosedRange{Min: in.UsableMemoryBytes - model.Max, Max: in.UsableMemoryBytes - model.Min}
	if availableForKV.Min > 0 {
		maxAgents.Min = math.Floor(availableForKV.Min / kvPerAgent.Max)
	}
	if availableForKV.Max > 0 {
		maxAgents.Max = math.Floor(availableForKV.Max / kvPerAgent.Min)
	}

	uncachedPrefill := ClosedRange{
		Min: float64(in.PrefillTokens) * (1 - in.PrefillCacheHit.Max),
		Max: float64(in.PrefillTokens) * (1 - in.PrefillCacheHit.Min),
	}
	decodeTokens := ClosedRange{float64(in.DecodeTokens), float64(in.DecodeTokens)}
	prefill := estimatePhase(uncachedPrefill, in, model, kvPerAgent)
	decode := estimatePhase(decodeTokens, in, model, kvPerAgent)
	service := addRange(prefill.TimeSeconds, decode.TimeSeconds)
	processedTokens := float64(in.Concurrency) * float64(in.PrefillTokens+in.DecodeTokens)
	decodeTokensAcrossBatch := float64(in.Concurrency) * float64(in.DecodeTokens)
	processedThroughput := ClosedRange{Min: processedTokens / service.Max, Max: processedTokens / service.Min}
	decodeThroughput := ClosedRange{Min: decodeTokensAcrossBatch / decode.TimeSeconds.Max, Max: decodeTokensAcrossBatch / decode.TimeSeconds.Min}

	return LongContextEnvelope{
		Schema: LongContextEnvelopeSchema, ModelMemoryBytes: model,
		KVMemoryPerAgentBytes: kvPerAgent, KVMemoryBytes: kv,
		TotalResidentMemoryBytes: total, Fits: fits, FitUncertain: fitUncertain,
		HeadroomBytes: headroom, MaxResidentAgents: maxAgents,
		Prefill: prefill, Decode: decode, ServiceTimeSeconds: service,
		ProcessedTokensPerSec: processedThroughput, DecodeTokensPerSec: decodeThroughput,
		Bottleneck:       combineBottlenecks(prefill.Bottleneck, decode.Bottleneck),
		SensitivityNotes: longContextSensitivity(in),
	}, nil
}

func estimatePhase(tokens ClosedRange, in LongContextEstimatorInput, model, kvPerAgent ClosedRange) RooflineTimeBounds {
	if tokens.Max == 0 {
		zero := ClosedRange{}
		return RooflineTimeBounds{ComputeSeconds: zero, MemorySeconds: zero, TimeSeconds: zero, Bottleneck: "none"}
	}

	parallelTokens := scaleRange(tokens, float64(in.Concurrency))
	flops := scaleRange(parallelTokens, 2*in.ActiveParameters)
	computeRate := mulRange(in.ComputeFLOPS, in.Efficiency)
	computeTime := divPositiveRange(flops, computeRate)

	// Each token streams the active weight fraction plus a conservative full scan
	// of the reserved resident KV capacity. The KV term is an analytical upper
	// bound, not architecture-exact attention traffic: hybrid linear, sparse, and
	// full-attention layers can read materially less or differently. This models
	// an active-batch completion envelope, not scheduler behavior.
	activeFraction := in.ActiveParameters / in.TotalParameters
	bytesPerToken := addRange(scaleRange(model, activeFraction), kvPerAgent)
	traffic := mulRange(parallelTokens, bytesPerToken)
	bandwidthRate := mulRange(in.BandwidthBytesPerSec, in.Efficiency)
	memoryTime := divPositiveRange(traffic, bandwidthRate)
	timeBounds := ClosedRange{Min: math.Max(computeTime.Min, memoryTime.Min), Max: math.Max(computeTime.Max, memoryTime.Max)}

	bottleneck := "mixed"
	if computeTime.Min >= memoryTime.Max {
		bottleneck = "compute"
	} else if memoryTime.Min >= computeTime.Max {
		bottleneck = "bandwidth"
	}
	return RooflineTimeBounds{ComputeSeconds: computeTime, MemorySeconds: memoryTime, TimeSeconds: timeBounds, Bottleneck: bottleneck}
}

func validateLongContextInput(in LongContextEstimatorInput) error {
	positive := []struct {
		name string
		v    float64
	}{{"total_parameters", in.TotalParameters}, {"active_parameters", in.ActiveParameters}, {"usable_memory_bytes", in.UsableMemoryBytes}}
	for _, field := range positive {
		if !finite(field.v) || field.v <= 0 {
			return fmt.Errorf("%s must be finite and positive", field.name)
		}
	}
	if in.ActiveParameters > in.TotalParameters {
		return fmt.Errorf("active_parameters cannot exceed total_parameters")
	}
	if in.ResidentContextTokens == 0 || in.ResidentAgents == 0 || in.Concurrency == 0 {
		return fmt.Errorf("resident_context_tokens, resident_agents, and concurrency must be positive")
	}
	if in.Concurrency > in.ResidentAgents {
		return fmt.Errorf("concurrency cannot exceed resident_agents")
	}
	if in.PrefillTokens == 0 && in.DecodeTokens == 0 {
		return fmt.Errorf("prefill_tokens and decode_tokens cannot both be zero")
	}
	if in.DecodeTokens == 0 && in.PrefillCacheHit.Max == 1 {
		return fmt.Errorf("full prefill cache hit with zero decode tokens produces zero service time")
	}
	if in.PrefillTokens > in.ResidentContextTokens || in.DecodeTokens > in.ResidentContextTokens-in.PrefillTokens {
		return fmt.Errorf("prefill_tokens plus decode_tokens cannot exceed resident_context_tokens")
	}
	for _, r := range []struct {
		name string
		r    ClosedRange
	}{{"weight_bits", in.WeightBits}, {"kv_bytes_per_token", in.KVBytesPerToken}, {"bandwidth_bytes_per_second", in.BandwidthBytesPerSec}, {"compute_flops", in.ComputeFLOPS}} {
		if err := validateRange(r.name, r.r, true, false); err != nil {
			return err
		}
	}
	if err := validateRange("efficiency_fraction", in.Efficiency, true, true); err != nil {
		return err
	}
	for _, r := range []struct {
		name string
		r    ClosedRange
	}{{"metadata_overhead_fraction", in.MetadataOverhead}, {"prefill_cache_hit_fraction", in.PrefillCacheHit}} {
		if err := validateRange(r.name, r.r, false, true); err != nil {
			return err
		}
	}
	return nil
}

func validateRange(name string, r ClosedRange, positive, fraction bool) error {
	if !finite(r.Min) || !finite(r.Max) {
		return fmt.Errorf("%s bounds must be finite", name)
	}
	if r.Min > r.Max {
		return fmt.Errorf("%s minimum cannot exceed maximum", name)
	}
	if positive && r.Min <= 0 {
		return fmt.Errorf("%s bounds must be positive", name)
	}
	if fraction && (r.Min < 0 || r.Max > 1) {
		return fmt.Errorf("%s bounds must be fractions in [0,1]", name)
	}
	return nil
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
func scaleRange(r ClosedRange, factor float64) ClosedRange {
	return ClosedRange{r.Min * factor, r.Max * factor}
}
func addScalar(r ClosedRange, scalar float64) ClosedRange {
	return ClosedRange{r.Min + scalar, r.Max + scalar}
}
func addRange(a, b ClosedRange) ClosedRange         { return ClosedRange{a.Min + b.Min, a.Max + b.Max} }
func mulRange(a, b ClosedRange) ClosedRange         { return ClosedRange{a.Min * b.Min, a.Max * b.Max} }
func divPositiveRange(a, b ClosedRange) ClosedRange { return ClosedRange{a.Min / b.Max, a.Max / b.Min} }

func combineBottlenecks(a, b string) string {
	if a == "none" {
		return b
	}
	if b == "none" || a == b {
		return a
	}
	return "mixed"
}

func longContextSensitivity(in LongContextEstimatorInput) []string {
	notes := []string{
		"Reserved KV memory scales linearly with resident context capacity and resident agents; requested tokens may use less than the reservation.",
		"Memory traffic conservatively assumes a full scan of resident KV capacity per processed token; it is not architecture-exact attention accounting.",
		"Hybrid linear, sparse, and full-attention layer mixes can materially reduce or reshape KV traffic relative to the full-resident-scan bound.",
		"Weight precision and metadata overhead change shared model memory and memory-bound phase time.",
		"Compute, bandwidth, and efficiency bounds widen roofline service-time uncertainty.",
	}
	if in.PrefillCacheHit.Max > 0 {
		notes = append(notes, "Prefill cache hits reduce prefill work only; decode work and resident KV memory are unchanged.")
	}
	return notes
}
