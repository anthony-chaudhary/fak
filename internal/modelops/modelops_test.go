package modelops

import (
	"reflect"
	"slices"
	"strings"
	"testing"
)

const (
	opus   = "claude-opus-4-8"
	sonnet = "claude-sonnet-4-6"
	haiku  = "claude-haiku-4-5-20251001"
)

func topThreeInput(candidate string, requiredTier int) Input {
	policy := func(model string, tier int, fallbacks ...string) Policy {
		return Policy{Model: model, CapabilityTier: tier, Fallbacks: fallbacks, MinSamples: 20,
			MinSuccessRate: .95, MaxProviderErrorRate: .02, MaxInvalidToolRate: .01,
			MaxP95LatencyMS: 5000, MaxThrottleRate: .03, MaxFallbackRate: .05}
	}
	healthy := func(model string) Observation {
		return Observation{Model: model, Samples: 30, SuccessRate: .98, ProviderErrorRate: .01,
			InvalidToolRate: 0, P95LatencyMS: 4000, ThrottleRate: .01, FallbackRate: .02}
	}
	return Input{Candidate: candidate, RequiredTier: requiredTier,
		Policies:     []Policy{policy(opus, 0, sonnet), policy(sonnet, 1, haiku), policy(haiku, 2)},
		Observations: []Observation{healthy(opus), healthy(sonnet), healthy(haiku)}}
}

func observation(in *Input, model string) *Observation {
	for i := range in.Observations {
		if in.Observations[i].Model == model {
			return &in.Observations[i]
		}
	}
	return nil
}

func TestEvaluatePromotesHealthyExactModel(t *testing.T) {
	got := Evaluate(topThreeInput(sonnet, 1))
	if got.Action != Promote || got.Selected != sonnet || got.Schema != Schema {
		t.Fatalf("decision = %+v, want Sonnet promotion", got)
	}
}

func TestEvaluateEverySLOBreachRollsBack(t *testing.T) {
	tests := []struct {
		name    string
		breakIt func(*Observation)
		want    string
	}{
		{"samples", func(o *Observation) { o.Samples = 1 }, "samples"},
		{"success", func(o *Observation) { o.SuccessRate = .5 }, "success_rate"},
		{"provider", func(o *Observation) { o.ProviderErrorRate = .5 }, "provider_error_rate"},
		{"tool", func(o *Observation) { o.InvalidToolRate = .5 }, "invalid_tool_rate"},
		{"latency", func(o *Observation) { o.P95LatencyMS = 9000 }, "p95_latency_ms"},
		{"throttle", func(o *Observation) { o.ThrottleRate = .5 }, "throttle_rate"},
		{"fallback", func(o *Observation) { o.FallbackRate = .5 }, "fallback_rate"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := topThreeInput(opus, 1)
			tt.breakIt(observation(&in, opus))
			got := Evaluate(in)
			if got.Action != Rollback || got.Selected != sonnet || !strings.Contains(strings.Join(got.Reasons, " "), tt.want) {
				t.Fatalf("decision = %+v, want Opus rollback to Sonnet for %s", got, tt.want)
			}
		})
	}
}

func TestEvaluateSonnetRollsBackToHaikuOnlyForTierTwo(t *testing.T) {
	in := topThreeInput(sonnet, 2)
	observation(&in, sonnet).SuccessRate = .5
	got := Evaluate(in)
	if got.Action != Rollback || got.Selected != haiku {
		t.Fatalf("tier-2 decision = %+v, want Haiku rollback", got)
	}

	in.RequiredTier = 1
	got = Evaluate(in)
	if got.Action != Hold || got.Selected != "" || !strings.Contains(strings.Join(got.Reasons, " "), "cannot serve required tier 1") {
		t.Fatalf("tier-1 decision = %+v, want capability-safe HOLD", got)
	}
}

func TestEvaluateHaikuFailureHolds(t *testing.T) {
	in := topThreeInput(haiku, 2)
	observation(&in, haiku).ProviderErrorRate = .5
	got := Evaluate(in)
	if got.Action != Hold || got.Selected != "" || !strings.Contains(strings.Join(got.Reasons, " "), "no healthy capability-safe fallback") {
		t.Fatalf("decision = %+v, want terminal HOLD", got)
	}
}

func TestEvaluateRejectsInvalidThresholdsAndObservations(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Input)
		want string
	}{
		{"zero samples", func(in *Input) { in.Policies[0].MinSamples = 0 }, "min_samples must be positive"},
		{"zero latency", func(in *Input) { in.Policies[0].MaxP95LatencyMS = 0 }, "max_p95_latency_ms must be positive"},
		{"policy rate", func(in *Input) { in.Policies[0].MinSuccessRate = 1.1 }, "min_success_rate must be within [0,1]"},
		{"observation rate", func(in *Input) { in.Observations[0].ThrottleRate = -0.1 }, "throttle_rate must be within [0,1]"},
		{"negative samples", func(in *Input) { in.Observations[0].Samples = -1 }, "samples must be non-negative"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := topThreeInput(opus, 1)
			tt.edit(&in)
			got := Evaluate(in)
			if got.Action != Hold || got.Selected != "" || !strings.Contains(strings.Join(got.Reasons, " "), tt.want) {
				t.Fatalf("decision = %+v, want validation HOLD containing %q", got, tt.want)
			}
		})
	}
}

func TestEvaluateFoldsExactModelInvocationOutcomes(t *testing.T) {
	in := topThreeInput(opus, 1)
	in.Outcomes = []InvocationOutcome{
		{InvocationID: "run-001", Model: "claude-opus-4-8", Action: Rollback},
		{InvocationID: "run-002", Model: "claude-opus-4-8", Action: Hold},
		{InvocationID: "run-003", Model: "claude-sonnet-4-6", Action: Promote},
		{InvocationID: "run-003", Model: "claude-sonnet-4-6", Action: Promote}, // idempotent replay
	}
	got := Evaluate(in)
	want := []OutcomeCount{
		{Model: "claude-opus-4-8", Rollback: 1, Hold: 1, Total: 2},
		{Model: "claude-sonnet-4-6", Promote: 1, Total: 1},
	}
	if !reflect.DeepEqual(got.OutcomeCounts, want) {
		t.Fatalf("outcome counts = %#v, want %#v", got.OutcomeCounts, want)
	}
}

func TestEvaluateHoldsOnConflictingOutcomeReplay(t *testing.T) {
	in := topThreeInput(opus, 1)
	in.Outcomes = []InvocationOutcome{
		{InvocationID: "run-001", Model: "claude-opus-4-8", Action: Promote},
		{InvocationID: "run-001", Model: "claude-opus-4-8", Action: Rollback},
	}
	got := Evaluate(in)
	if got.Action != Hold || !slices.Contains(got.Reasons, "conflicting outcome invocation_id: run-001") {
		t.Fatalf("decision = %+v, want fail-closed conflicting replay", got)
	}
}
