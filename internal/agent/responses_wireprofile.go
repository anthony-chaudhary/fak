package agent

// ResponsesWireStreamsLive reports whether provider's wire can deliver a live token
// stream for a Responses-api turn served through this planner family. The OpenAI-
// compatible chat wires (OpenAI, xAI) stream natively regardless of streamGate; the
// openai-responses wire streams only when the planner asked the ChatGPT-subscription
// upstream for SSE (streamGate, i.e. HTTPPlanner.ForceResponsesStream); every other
// wire (Anthropic, Gemini, unknown) is buffered-only. The chat-side switch
// (StreamingSupported) stays byte-for-byte historical: this predicate answers only the
// gateway's /v1/responses live-passthrough question (#12765).
func ResponsesWireStreamsLive(provider Provider, streamGate bool) bool {
	switch provider {
	case ProviderOpenAI, ProviderXAI:
		return true
	case ProviderOpenAIResponses:
		return streamGate
	default:
		return false
	}
}
