package agent

import "testing"

// TestResponsesWireStreamsLive pins the /v1/responses live-passthrough predicate
// (#12765) across the closed Provider vocabulary plus an unknown wire: the
// OpenAI-compatible chat wires stream live regardless of the streamGate; the
// openai-responses wire streams only when the planner asked the upstream for SSE
// (streamGate); every other wire — and any unregistered wire — fails closed to
// buffered-only.
func TestResponsesWireStreamsLive(t *testing.T) {
	cases := []struct {
		provider   Provider
		streamGate bool
		want       bool
	}{
		{ProviderOpenAI, false, true},
		{ProviderOpenAI, true, true},
		{ProviderXAI, false, true},
		{ProviderXAI, true, true},
		{ProviderOpenAIResponses, false, false},
		{ProviderOpenAIResponses, true, true},
		{ProviderAnthropic, false, false},
		{ProviderAnthropic, true, false},
		{ProviderGemini, false, false},
		{ProviderGemini, true, false},
		{Provider("nosuchprovider"), false, false},
		{Provider("nosuchprovider"), true, false},
	}
	for _, tc := range cases {
		if got := ResponsesWireStreamsLive(tc.provider, tc.streamGate); got != tc.want {
			t.Errorf("ResponsesWireStreamsLive(%q, streamGate=%v) = %v, want %v", tc.provider, tc.streamGate, got, tc.want)
		}
	}
}

// TestResponsesWireChatSideProfileUnchanged is the behavior-preserving guard for
// #12765: the new live-passthrough predicate answers only the gateway's question —
// the chat-side capability table rows it reads alongside stay byte-for-byte
// historical, so the Responses wire keeps reporting HonorsStreaming=false (its
// buffered default) while the OpenAI chat wire keeps true.
func TestResponsesWireChatSideProfileUnchanged(t *testing.T) {
	if got := mustWireProfile(ProviderOpenAIResponses).HonorsStreaming; got != false {
		t.Errorf("wireProfiles[ProviderOpenAIResponses].HonorsStreaming = %v, want false (chat-side table row deliberately unchanged by #12765)", got)
	}
	if got := mustWireProfile(ProviderOpenAI).HonorsStreaming; got != true {
		t.Errorf("wireProfiles[ProviderOpenAI].HonorsStreaming = %v, want true (chat-side table row deliberately unchanged by #12765)", got)
	}
}
