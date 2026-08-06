package tokenprofile

import (
	"errors"
	"fmt"
	"math"
)

const Schema = "fak-token-profile/1"

type Kind string

const (
	InputUncached  Kind = "input.uncached"
	InputCached    Kind = "input.cached"
	OutputReserved Kind = "output.reserved"
)

type Prices struct {
	InputUncachedPerMillion float64 `json:"input_uncached_per_million"`
	InputCachedPerMillion   float64 `json:"input_cached_per_million"`
	OutputPerMillion        float64 `json:"output_per_million"`
}

type Weights struct {
	InputUncached float64 `json:"input_uncached"`
	InputCached   float64 `json:"input_cached"`
	Output        float64 `json:"output"`
}

func DefaultWeights() Weights {
	return Weights{InputUncached: 1, InputCached: 0.25, Output: 4}
}

type Forecast struct {
	InputTokens       int64   `json:"input_tokens"`
	CachedInputTokens int64   `json:"cached_input_tokens"`
	MaxOutputTokens   int64   `json:"max_output_tokens"`
	Prices            Prices  `json:"prices_usd_per_million"`
	Weights           Weights `json:"scheduler_weights"`
}

type Class struct {
	Kind           Kind    `json:"kind"`
	Tokens         int64   `json:"tokens"`
	UnitPrice      float64 `json:"usd_per_million"`
	WorstCaseUSD   float64 `json:"worst_case_usd"`
	SchedulerUnits float64 `json:"scheduler_units"`
	Certainty      string  `json:"certainty"`
}

type Report struct {
	Schema            string  `json:"schema"`
	Phase             string  `json:"phase"`
	Classes           []Class `json:"classes"`
	TotalTokens       int64   `json:"total_tokens"`
	WorstCaseUSD      float64 `json:"worst_case_usd"`
	SchedulerUnits    float64 `json:"scheduler_units"`
	DominantCostClass Kind    `json:"dominant_cost_class"`
	DominantLoadClass Kind    `json:"dominant_load_class"`
	ShiftLeftAction   string  `json:"shift_left_action"`
	Boundary          string  `json:"boundary"`
}

func Price(f Forecast) (Report, error) {
	if f.InputTokens < 0 || f.CachedInputTokens < 0 || f.MaxOutputTokens < 0 {
		return Report{}, errors.New("token counts must be non-negative")
	}
	if f.CachedInputTokens > f.InputTokens {
		return Report{}, errors.New("cached input tokens cannot exceed input tokens")
	}
	vals := []float64{f.Prices.InputUncachedPerMillion, f.Prices.InputCachedPerMillion, f.Prices.OutputPerMillion, f.Weights.InputUncached, f.Weights.InputCached, f.Weights.Output}
	for _, v := range vals {
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return Report{}, errors.New("prices and scheduler weights must be finite and non-negative")
		}
	}
	uncached := f.InputTokens - f.CachedInputTokens
	classes := []Class{
		makeClass(InputUncached, uncached, f.Prices.InputUncachedPerMillion, f.Weights.InputUncached, "forecast"),
		makeClass(InputCached, f.CachedInputTokens, f.Prices.InputCachedPerMillion, f.Weights.InputCached, "forecast-cache-expectation"),
		makeClass(OutputReserved, f.MaxOutputTokens, f.Prices.OutputPerMillion, f.Weights.Output, "reservation-upper-bound"),
	}
	r := Report{Schema: Schema, Phase: "preflight", Classes: classes, TotalTokens: f.InputTokens + f.MaxOutputTokens, Boundary: "forecast only; reconcile provider-observed usage after completion"}
	for _, c := range classes {
		r.WorstCaseUSD += c.WorstCaseUSD
		r.SchedulerUnits += c.SchedulerUnits
		if dominantCost(classes, c) {
			r.DominantCostClass = c.Kind
		}
		if dominantLoad(classes, c) {
			r.DominantLoadClass = c.Kind
		}
	}
	r.ShiftLeftAction = action(r)
	return r, nil
}

func makeClass(kind Kind, tokens int64, price, weight float64, certainty string) Class {
	return Class{Kind: kind, Tokens: tokens, UnitPrice: price, WorstCaseUSD: float64(tokens) * price / 1_000_000, SchedulerUnits: float64(tokens) * weight, Certainty: certainty}
}

func dominantCost(all []Class, candidate Class) bool {
	for _, c := range all {
		if c.WorstCaseUSD > candidate.WorstCaseUSD {
			return false
		}
	}
	return true
}
func dominantLoad(all []Class, candidate Class) bool {
	for _, c := range all {
		if c.SchedulerUnits > candidate.SchedulerUnits {
			return false
		}
	}
	return true
}

func action(r Report) string {
	switch r.DominantLoadClass {
	case OutputReserved:
		return "cap or route output before admission; decode reservation dominates weighted load"
	case InputUncached:
		return "stabilize/cache the reusable prefix or route prefill capacity before admission"
	case InputCached:
		return "preserve cache affinity and verify the expected cache hit before admission"
	default:
		return fmt.Sprintf("admit with %s as the dominant weighted class", r.DominantLoadClass)
	}
}
