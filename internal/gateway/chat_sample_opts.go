package gateway

import "github.com/anthony-chaudhary/fak/internal/agent"

// chatWireSampleOpts is the per-request sampling contract both chat-completions
// emitters (buffered and live stream) forward upstream. Keeping it in one place
// stops a field wired into one path from silently missing on the other; callers
// append only their path-specific options. Every option is a no-op when its field
// is absent, so an omitted parameter keeps the pre-seam upstream body.
func chatWireSampleOpts(req ChatRequest, sessionTurn servedSessionTurn, loopDepth *LoopDepthReceipt) []agent.SampleOpt {
	return []agent.SampleOpt{
		agent.WithModel(req.Model),
		agent.WithMaxTokens(sessionTurn.maxTokensFor(req.MaxTokens)),
		agent.WithClientTemperature(req.Temperature),
		agent.WithTopP(req.TopP),
		agent.WithSeed(req.Seed),
		agent.WithStop(normalizeStop(req.Stop)),
		// Structured-output passthrough (#907): the ride engine enforces
		// response_format / logit_bias during generation; the resulting tool
		// candidate is still adjudicated whole.
		agent.WithResponseFormat(req.ResponseFormat),
		agent.WithToolChoice(req.ToolChoice),
		agent.WithLogitBias(req.LogitBias),
		agent.WithGuidedDecode(req.GuidedDecodeFields()),
		// Repetition-penalty passthrough (#1705) lets a reasoning model break a
		// non-terminating repetition loop in the in-kernel sampler too.
		agent.WithFrequencyPenalty(req.FrequencyPenalty),
		agent.WithPresencePenalty(req.PresencePenalty),
		agent.WithChatWireTopK(req.TopK),
		agent.WithMinP(req.MinP),
		agent.WithChatTemplateKwargs(req.ChatTemplateKwargs),
		agent.WithSpeculativeNMax(req.SpeculativeNMax),
		agent.WithMultimodalMode(req.DeclaredMultimodalMode()),
		loopDepthSampleOpt(loopDepth),
	}
}
