package modelops

import (
	"fmt"
	"sort"
	"strings"
)

const Schema = "fak.modelops.canary-decision/1"

type Action string

const (
	Promote  Action = "PROMOTE"
	Hold     Action = "HOLD"
	Rollback Action = "ROLLBACK"
)

// Policy is declared before a canary runs. Lower tier numbers are more capable:
// tier 0 may serve every workload, while tier 2 may serve only tier-2 work.
type AlertContract struct {
	Owner         string `json:"owner"`
	Route         string `json:"route"`
	AckSLAMinutes int    `json:"ack_sla_minutes"`
	Runbook       string `json:"runbook"`
}

type Policy struct {
	Model                string        `json:"model"`
	CapabilityTier       int           `json:"capability_tier"`
	Fallbacks            []string      `json:"fallbacks,omitempty"`
	Alert                AlertContract `json:"alert"`
	WindowMinutes        int           `json:"window_minutes"`
	MinSamples           int           `json:"min_samples"`
	MinSuccessRate       float64       `json:"min_success_rate"`
	MaxProviderErrorRate float64       `json:"max_provider_error_rate"`
	MaxInvalidToolRate   float64       `json:"max_invalid_tool_rate"`
	MaxP95LatencyMS      int64         `json:"max_p95_latency_ms"`
	MaxThrottleRate      float64       `json:"max_throttle_rate"`
	MaxFallbackRate      float64       `json:"max_fallback_rate"`
}

type Observation struct {
	Model             string  `json:"model"`
	Samples           int     `json:"samples"`
	SuccessRate       float64 `json:"success_rate"`
	ProviderErrorRate float64 `json:"provider_error_rate"`
	InvalidToolRate   float64 `json:"invalid_tool_rate"`
	P95LatencyMS      int64   `json:"p95_latency_ms"`
	ThrottleRate      float64 `json:"throttle_rate"`
	FallbackRate      float64 `json:"fallback_rate"`
}

type Input struct {
	RequiredTier int                 `json:"required_tier"`
	Candidate    string              `json:"candidate"`
	Policies     []Policy            `json:"policies"`
	Observations []Observation       `json:"observations"`
	Outcomes     []InvocationOutcome `json:"outcomes,omitempty"`
}

type InvocationOutcome struct {
	InvocationID string `json:"invocation_id"`
	Model        string `json:"model"`
	Action       Action `json:"action"`
}

type OutcomeCount struct {
	Model    string `json:"model"`
	Promote  int    `json:"promote"`
	Rollback int    `json:"rollback"`
	Hold     int    `json:"hold"`
	Total    int    `json:"total"`
}

type Decision struct {
	Schema        string         `json:"schema"`
	Action        Action         `json:"action"`
	Candidate     string         `json:"candidate"`
	Selected      string         `json:"selected,omitempty"`
	RequiredTier  int            `json:"required_tier"`
	Reasons       []string       `json:"reasons"`
	OutcomeCounts []OutcomeCount `json:"outcome_counts,omitempty"`
	Alert         AlertContract  `json:"alert"`
	WindowMinutes int            `json:"window_minutes"`
}

func Evaluate(in Input) Decision {
	out := Decision{Schema: Schema, Action: Hold, Candidate: in.Candidate, RequiredTier: in.RequiredTier}
	outcomeCounts, outcomeReasons := FoldOutcomes(in.Outcomes)
	out.OutcomeCounts = outcomeCounts
	if len(outcomeReasons) > 0 {
		out.Reasons = outcomeReasons
		return out
	}
	policies, observations, reasons := index(in)
	if len(reasons) > 0 {
		out.Reasons = reasons
		return out
	}
	candidate, ok := policies[in.Candidate]
	if ok {
		out.Alert = candidate.Alert
		out.WindowMinutes = candidate.WindowMinutes
	}
	if !ok {
		out.Reasons = []string{"candidate policy is missing"}
		return out
	}
	if candidate.CapabilityTier > in.RequiredTier {
		out.Reasons = []string{fmt.Sprintf("candidate tier %d cannot serve required tier %d", candidate.CapabilityTier, in.RequiredTier)}
		return out
	}
	candidateBreaches := breaches(candidate, observations[in.Candidate])
	if len(candidateBreaches) == 0 {
		out.Action = Promote
		out.Selected = candidate.Model
		out.Reasons = []string{"candidate meets every declared canary threshold"}
		return out
	}
	out.Reasons = append(out.Reasons, candidateBreaches...)
	for _, name := range candidate.Fallbacks {
		fallback, ok := policies[name]
		if !ok {
			out.Reasons = append(out.Reasons, "fallback policy missing: "+name)
			continue
		}
		if fallback.CapabilityTier > in.RequiredTier {
			out.Reasons = append(out.Reasons, fmt.Sprintf("fallback %s tier %d cannot serve required tier %d", name, fallback.CapabilityTier, in.RequiredTier))
			continue
		}
		fallbackBreaches := breaches(fallback, observations[name])
		if len(fallbackBreaches) != 0 {
			out.Reasons = append(out.Reasons, fmt.Sprintf("fallback %s unhealthy: %s", name, strings.Join(fallbackBreaches, "; ")))
			continue
		}
		out.Action = Rollback
		out.Selected = name
		out.Alert = fallback.Alert
		out.WindowMinutes = fallback.WindowMinutes
		out.Reasons = append(out.Reasons, "selected first healthy capability-safe fallback")
		return out
	}
	out.Reasons = append(out.Reasons, "no healthy capability-safe fallback; hold for operator escalation")
	return out
}

func index(in Input) (map[string]Policy, map[string]Observation, []string) {
	policies := make(map[string]Policy, len(in.Policies))
	observations := make(map[string]Observation, len(in.Observations))
	var reasons []string
	if strings.TrimSpace(in.Candidate) == "" {
		reasons = append(reasons, "candidate is required")
	}
	if in.RequiredTier < 0 {
		reasons = append(reasons, "required_tier must be non-negative")
	}
	for _, p := range in.Policies {
		if p.Model == "" {
			reasons = append(reasons, "policy model is required")
			continue
		}
		if _, exists := policies[p.Model]; exists {
			reasons = append(reasons, "duplicate policy: "+p.Model)
		}
		if p.CapabilityTier < 0 {
			reasons = append(reasons, "policy "+p.Model+": capability_tier must be non-negative")
		}
		if p.WindowMinutes <= 0 {
			reasons = append(reasons, "policy "+p.Model+": window_minutes must be positive")
		}
		if p.MinSamples <= 0 {
			reasons = append(reasons, "policy "+p.Model+": min_samples must be positive")
		}
		if p.MaxP95LatencyMS <= 0 {
			reasons = append(reasons, "policy "+p.Model+": max_p95_latency_ms must be positive")
		}
		if err := validateAlert(p.Alert); err != nil {
			reasons = append(reasons, "policy "+p.Model+": alert "+err.Error())
		}
		for name, rate := range map[string]float64{
			"min_success_rate": p.MinSuccessRate, "max_provider_error_rate": p.MaxProviderErrorRate,
			"max_invalid_tool_rate": p.MaxInvalidToolRate, "max_throttle_rate": p.MaxThrottleRate,
			"max_fallback_rate": p.MaxFallbackRate,
		} {
			if !validRate(rate) {
				reasons = append(reasons, fmt.Sprintf("policy %s: %s must be within [0,1]", p.Model, name))
			}
		}
		policies[p.Model] = p
	}
	for _, o := range in.Observations {
		if o.Model == "" {
			reasons = append(reasons, "observation model is required")
			continue
		}
		if _, exists := observations[o.Model]; exists {
			reasons = append(reasons, "duplicate observation: "+o.Model)
		}
		if o.Samples < 0 {
			reasons = append(reasons, "observation "+o.Model+": samples must be non-negative")
		}
		if o.P95LatencyMS < 0 {
			reasons = append(reasons, "observation "+o.Model+": p95_latency_ms must be non-negative")
		}
		for name, rate := range map[string]float64{
			"success_rate": o.SuccessRate, "provider_error_rate": o.ProviderErrorRate,
			"invalid_tool_rate": o.InvalidToolRate, "throttle_rate": o.ThrottleRate,
			"fallback_rate": o.FallbackRate,
		} {
			if !validRate(rate) {
				reasons = append(reasons, fmt.Sprintf("observation %s: %s must be within [0,1]", o.Model, name))
			}
		}
		observations[o.Model] = o
	}
	sort.Strings(reasons)
	return policies, observations, reasons
}

func FoldOutcomes(outcomes []InvocationOutcome) ([]OutcomeCount, []string) {
	byModel := make(map[string]*OutcomeCount)
	seen := make(map[string]InvocationOutcome)
	var reasons []string
	for _, outcome := range outcomes {
		if outcome.InvocationID == "" {
			reasons = append(reasons, "outcome invocation_id is required")
			continue
		}
		if prior, ok := seen[outcome.InvocationID]; ok {
			if prior != outcome {
				reasons = append(reasons, "conflicting outcome invocation_id: "+outcome.InvocationID)
			}
			continue
		}
		seen[outcome.InvocationID] = outcome
		if outcome.Model == "" {
			reasons = append(reasons, "outcome "+outcome.InvocationID+": model is required")
			continue
		}
		count := byModel[outcome.Model]
		if count == nil {
			count = &OutcomeCount{Model: outcome.Model}
			byModel[outcome.Model] = count
		}
		switch outcome.Action {
		case Promote:
			count.Promote++
		case Rollback:
			count.Rollback++
		case Hold:
			count.Hold++
		}
		count.Total++
	}
	models := make([]string, 0, len(byModel))
	for model := range byModel {
		models = append(models, model)
	}
	sort.Strings(models)
	counts := make([]OutcomeCount, 0, len(models))
	for _, model := range models {
		counts = append(counts, *byModel[model])
	}
	sort.Strings(reasons)
	return counts, reasons
}

func validateAlert(a AlertContract) error {
	if strings.TrimSpace(a.Owner) == "" {
		return fmt.Errorf("owner is required")
	}
	if strings.TrimSpace(a.Route) == "" {
		return fmt.Errorf("route is required")
	}
	if a.AckSLAMinutes <= 0 {
		return fmt.Errorf("ack_sla_minutes must be positive")
	}
	if strings.TrimSpace(a.Runbook) == "" {
		return fmt.Errorf("runbook is required")
	}
	return nil
}

func validRate(v float64) bool { return v >= 0 && v <= 1 }

func breaches(p Policy, o Observation) []string {
	if o.Model == "" {
		return []string{"observation is missing"}
	}
	var out []string
	if o.Samples < p.MinSamples {
		out = append(out, fmt.Sprintf("samples %d < %d", o.Samples, p.MinSamples))
	}
	if o.SuccessRate < p.MinSuccessRate {
		out = append(out, fmt.Sprintf("success_rate %.4f < %.4f", o.SuccessRate, p.MinSuccessRate))
	}
	if o.ProviderErrorRate > p.MaxProviderErrorRate {
		out = append(out, fmt.Sprintf("provider_error_rate %.4f > %.4f", o.ProviderErrorRate, p.MaxProviderErrorRate))
	}
	if o.InvalidToolRate > p.MaxInvalidToolRate {
		out = append(out, fmt.Sprintf("invalid_tool_rate %.4f > %.4f", o.InvalidToolRate, p.MaxInvalidToolRate))
	}
	if o.P95LatencyMS > p.MaxP95LatencyMS {
		out = append(out, fmt.Sprintf("p95_latency_ms %d > %d", o.P95LatencyMS, p.MaxP95LatencyMS))
	}
	if o.ThrottleRate > p.MaxThrottleRate {
		out = append(out, fmt.Sprintf("throttle_rate %.4f > %.4f", o.ThrottleRate, p.MaxThrottleRate))
	}
	if o.FallbackRate > p.MaxFallbackRate {
		out = append(out, fmt.Sprintf("fallback_rate %.4f > %.4f", o.FallbackRate, p.MaxFallbackRate))
	}
	return out
}
