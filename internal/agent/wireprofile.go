package agent

import "fmt"

// wireprofile.go is the one place a provider's STATIC wire capabilities live as
// DATA instead of as parallel `switch provider {}` arms scattered across the
// adapter layer. Adding a provider — or answering "does this wire stream / carry
// a native top_k / forward the OpenAI structured-decode carriers" — becomes a
// table entry, not an edit to every switch that re-decides the same fact.
//
// This is the wire-plane twin of internal/harnessprofile (the harness-as-process
// descriptor, docs/notes/UNIVERSAL-HARNESS-PROFILES-2026-07-01.md). That package
// declares "which coding harness → which wire / repoint / credential"; this one
// declares "which wire → which capabilities" so the two planes meet at the same
// closed Provider vocabulary. It is PURE: data + a pure lookup, no I/O.
//
// SCOPE (honest fence). A WireProfile carries the provider's STATIC capability
// bits — the facts that do not depend on a single request's contents. It does NOT
// carry per-request quirk flags (OmitTemperature, OpenAIToolMessagesAsText), which
// are decided at the call site from live signals (the Codex subscription route, an
// operator env opt-in) and ride on adapterRequest. Nor does it reimplement a wire:
// the adapter internals (schema case-folding, header schemes, endpoint suffixes)
// stay in adapters.go. This descriptor is the CAPABILITY index over them, and the
// single source of truth the two provider `switch`es (adapter dispatch and
// StreamingSupported) now read instead of re-deciding independently.

// WireProfile is the declarative capability descriptor for one upstream provider
// wire. Every field is a static fact about the wire, not a per-request choice:
//
//   - HonorsStreaming: the wire delivers an incremental SSE token stream when a
//     request sets Stream. Only the OpenAI-compatible chat wire (OpenAI + the
//     xAI/vLLM/SGLang servers that share its `chat.completion.chunk` delta format)
//     does today; every other adapter ignores Stream and returns a buffered body
//     byte-identical to the non-streamed one.
//   - NativeTopK: the wire has a native top_k field that REQUIRES a positive integer
//     (Anthropic, Gemini). The OpenAI surfaces have no field at all, so a top_k is
//     dropped rather than clamped. positiveTopK() enforces the positive-only rule for
//     the wires that carry it.
//   - NativeStructuredDecode: the wire forwards the OpenAI structured/guided-decode
//     carriers (response_format / logit_bias, #560) as first-class request fields.
//     The OpenAI/xAI chat wire does; the other wires route structured output through
//     their own shape instead (ExtraBody, or the Responses API's text.format via
//     responsesText), so the carriers are omitted from their bodies.
type WireProfile struct {
	// Provider is the wire this profile describes; it keys the registry.
	Provider Provider
	// HonorsStreaming reports whether a Stream request yields an SSE token stream
	// rather than a buffered completion.
	HonorsStreaming bool
	// NativeTopK reports whether the wire has a native, positive-only top_k field.
	NativeTopK bool
	// NativeStructuredDecode reports whether the wire forwards the OpenAI
	// response_format / logit_bias carriers as native request fields.
	NativeStructuredDecode bool
}

// wireProfiles is the capability table over the closed Provider vocabulary. It is
// the single source of truth the adapter dispatch and StreamingSupported both read;
// a new provider is one entry here plus its adapter, never a new arm in each switch.
// The bits mirror the pre-descriptor behavior exactly:
//
//   - OpenAI / xAI share the openAIAdapter (chat-completions): they stream, carry the
//     native structured-decode carriers, and have no top_k field.
//   - OpenAI Responses: buffered-only here, no native carriers (structured output maps
//     to text.format), no top_k field.
//   - Anthropic / Gemini: buffered-only, native positive-only top_k, structured output
//     via their own shapes (no OpenAI carriers).
var wireProfiles = map[Provider]WireProfile{
	ProviderOpenAI:          {Provider: ProviderOpenAI, HonorsStreaming: true, NativeTopK: false, NativeStructuredDecode: true},
	ProviderXAI:             {Provider: ProviderXAI, HonorsStreaming: true, NativeTopK: false, NativeStructuredDecode: true},
	ProviderOpenAIResponses: {Provider: ProviderOpenAIResponses, HonorsStreaming: false, NativeTopK: false, NativeStructuredDecode: false},
	ProviderAnthropic:       {Provider: ProviderAnthropic, HonorsStreaming: false, NativeTopK: true, NativeStructuredDecode: false},
	ProviderGemini:          {Provider: ProviderGemini, HonorsStreaming: false, NativeTopK: true, NativeStructuredDecode: false},
}

// WireProfileFor returns the capability descriptor for a provider. An empty
// provider defaults to OpenAI, matching NewTranscriptAdapter's default wire. The
// bool is false for a provider with no registered profile — the same closed-set
// discipline NewTranscriptAdapter enforces, so an unknown wire fails loud at its
// call site rather than silently reading zero-value capabilities.
func WireProfileFor(provider Provider) (WireProfile, bool) {
	if provider == "" {
		provider = ProviderOpenAI
	}
	p, ok := wireProfiles[provider]
	return p, ok
}

// mustWireProfile returns the descriptor for a provider that is already known to be
// in the closed set (the caller resolved it through ParseProvider or holds a shipped
// adapter). It panics on an unregistered provider, which can only happen if a new
// Provider constant is added without its profile — a build-time-adjacent invariant a
// test pins, not a runtime path.
func mustWireProfile(provider Provider) WireProfile {
	p, ok := WireProfileFor(provider)
	if !ok {
		panic(fmt.Sprintf("agent: no wire profile registered for provider %q", provider))
	}
	return p
}
