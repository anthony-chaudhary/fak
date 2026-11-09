package gateway

import (
	"maps"
	"slices"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// resolveHTTPSampleOpts binds reservation, hedge admission and every HTTP attempt
// to one ordered evaluation of caller code. HTTPPlanner folds options into zero
// SampleParams before applying its configured defaults and caps; the generic
// Planner interface does not promise that baseline, so custom fleets stay untouched.
func (r *ReplicaDispatch) resolveHTTPSampleOpts(opts []agent.SampleOpt) []agent.SampleOpt {
	if r == nil || len(r.replicas) == 0 || len(opts) == 0 {
		return opts
	}
	for _, replica := range r.replicas {
		if planner, ok := replica.Planner.(*agent.HTTPPlanner); !ok || planner == nil {
			return opts
		}
	}
	var resolved agent.SampleParams
	for _, opt := range opts {
		if opt != nil {
			opt(&resolved)
		}
	}
	// Detach from caller-owned referents before retaining the canonical snapshot.
	resolved = cloneReplicaSampleParams(resolved)
	return []agent.SampleOpt{func(sp *agent.SampleParams) {
		// Each physical attempt, including concurrent hedges, gets its own view.
		*sp = cloneReplicaSampleParams(resolved)
	}}
}

func cloneReplicaSampleParams(in agent.SampleParams) agent.SampleParams {
	// Preserve scalar/private state and opaque callback identity without invoking
	// caller code or serializing planner-internal capabilities through a wire format.
	out := in
	out.NativeBeamSearch = cloneReplicaOptionValue(in.NativeBeamSearch)
	if out.NativeBeamSearch != nil {
		out.NativeBeamSearch.LengthPenalty = cloneReplicaOptionValue(in.NativeBeamSearch.LengthPenalty)
	}
	out.MaxTokens = cloneReplicaOptionValue(in.MaxTokens)
	out.Temperature = cloneReplicaOptionValue(in.Temperature)
	out.TopP = cloneReplicaOptionValue(in.TopP)
	out.TopK = cloneReplicaOptionValue(in.TopK)
	out.Seed = cloneReplicaOptionValue(in.Seed)
	out.FrequencyPenalty = cloneReplicaOptionValue(in.FrequencyPenalty)
	out.PresencePenalty = cloneReplicaOptionValue(in.PresencePenalty)
	out.MinP = cloneReplicaOptionValue(in.MinP)
	out.SpeculativeNMax = cloneReplicaOptionValue(in.SpeculativeNMax)
	out.CompactHistoryBudget = cloneReplicaOptionValue(in.CompactHistoryBudget)
	out.ElideStaleReads = cloneReplicaOptionValue(in.ElideStaleReads)
	out.DeferColdTools = cloneReplicaOptionValue(in.DeferColdTools)
	out.PerTokenStream = cloneReplicaOptionValue(in.PerTokenStream)
	out.ThinkingBudget = cloneReplicaOptionValue(in.ThinkingBudget)
	out.Stop = slices.Clone(in.Stop)
	out.ResponseFormat = slices.Clone(in.ResponseFormat)
	out.ToolChoice = slices.Clone(in.ToolChoice)
	out.ChatTemplateKwargs = slices.Clone(in.ChatTemplateKwargs)
	out.RawRequestBody = slices.Clone(in.RawRequestBody)
	out.LogitBias = maps.Clone(in.LogitBias)
	out.GuidedDecode = maps.Clone(in.GuidedDecode)
	for key, value := range out.GuidedDecode {
		out.GuidedDecode[key] = slices.Clone(value)
	}
	return out
}

func cloneReplicaOptionValue[T any](in *T) *T {
	if in == nil {
		return nil
	}
	out := *in
	return &out
}
