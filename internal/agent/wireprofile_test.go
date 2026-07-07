package agent

import "testing"

// allProviders is the closed Provider vocabulary the descriptor table must cover.
// It mirrors NewTranscriptAdapter's dispatch set; a new Provider constant added
// without a profile trips TestWireProfileCoversEveryProvider below.
var allProviders = []Provider{
	ProviderOpenAI,
	ProviderOpenAIResponses,
	ProviderAnthropic,
	ProviderGemini,
	ProviderXAI,
}

// TestWireProfileCoversEveryProvider pins that every provider with a shipped adapter
// also has a capability descriptor — the closed-set invariant that keeps the two
// planes (adapter dispatch, WireProfile table) in lockstep. A provider that builds an
// adapter but has no profile would read zero-value capabilities silently; this fails
// loud instead. It also confirms an empty provider resolves (defaults to OpenAI) and
// an unknown provider does not.
func TestWireProfileCoversEveryProvider(t *testing.T) {
	for _, p := range allProviders {
		// Every provider that yields an adapter must yield a profile.
		if _, err := NewTranscriptAdapter(p); err != nil {
			t.Fatalf("NewTranscriptAdapter(%q) unexpectedly failed: %v", p, err)
		}
		if _, ok := WireProfileFor(p); !ok {
			t.Errorf("provider %q builds an adapter but has no WireProfile", p)
		}
	}
	if _, ok := WireProfileFor(""); !ok {
		t.Error("empty provider should resolve to the default (OpenAI) profile")
	}
	if _, ok := WireProfileFor(Provider("nope")); ok {
		t.Error("unregistered provider should not resolve to a profile")
	}
}

// TestStreamingSupportedMatchesProfile is the behavior-preserving witness for the
// StreamingSupported refactor: for every provider (plus the empty default and an
// unknown wire) the planner's answer must equal the descriptor's HonorsStreaming bit,
// AND must equal the original hand-written switch (OpenAI/xAI/"" stream; all else do
// not). If the two ever diverge, this is the test that catches it.
func TestStreamingSupportedMatchesProfile(t *testing.T) {
	// originalStreamingSwitch reproduces the pre-descriptor StreamingSupported logic
	// verbatim, so the refactor is checked against the behavior it replaced.
	originalStreamingSwitch := func(p Provider) bool {
		switch p {
		case ProviderOpenAI, ProviderXAI, "":
			return true
		default:
			return false
		}
	}
	cases := append([]Provider{"", Provider("nope")}, allProviders...)
	for _, p := range cases {
		planner := &HTTPPlanner{Provider: p}
		got := planner.StreamingSupported()
		if want := originalStreamingSwitch(p); got != want {
			t.Errorf("StreamingSupported(%q) = %v, original switch = %v", p, got, want)
		}
		prof, ok := WireProfileFor(p)
		wantProfile := ok && prof.HonorsStreaming
		if got != wantProfile {
			t.Errorf("StreamingSupported(%q) = %v, profile HonorsStreaming = %v", p, got, wantProfile)
		}
	}
}

// TestWireProfileNativeTopK pins which wires carry a native positive-only top_k
// field, mirroring positiveTopK's provider knowledge: Anthropic and Gemini do; the
// OpenAI surfaces and xAI do not (they have no top_k field, so a k is dropped).
func TestWireProfileNativeTopK(t *testing.T) {
	wantTopK := map[Provider]bool{
		ProviderAnthropic:       true,
		ProviderGemini:          true,
		ProviderOpenAI:          false,
		ProviderXAI:             false,
		ProviderOpenAIResponses: false,
	}
	for p, want := range wantTopK {
		prof := mustWireProfile(p)
		if prof.NativeTopK != want {
			t.Errorf("provider %q NativeTopK = %v, want %v", p, prof.NativeTopK, want)
		}
	}
}

// TestWireProfileNativeStructuredDecode pins which wires forward the OpenAI
// response_format / logit_bias carriers as native fields (#560). Only the OpenAI-
// compatible chat wire (OpenAI + xAI) does; the Responses, Anthropic, and Gemini
// wires route structured output through their own shapes, so the carriers are omitted.
func TestWireProfileNativeStructuredDecode(t *testing.T) {
	wantNative := map[Provider]bool{
		ProviderOpenAI:          true,
		ProviderXAI:             true,
		ProviderOpenAIResponses: false,
		ProviderAnthropic:       false,
		ProviderGemini:          false,
	}
	for p, want := range wantNative {
		prof := mustWireProfile(p)
		if prof.NativeStructuredDecode != want {
			t.Errorf("provider %q NativeStructuredDecode = %v, want %v", p, prof.NativeStructuredDecode, want)
		}
	}
}
