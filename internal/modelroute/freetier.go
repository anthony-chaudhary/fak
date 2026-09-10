package modelroute

// freetier.go — the quality-gated FREE-tier pool: route across every free plan,
// serve only high-quality models.
//
// THE GAP IT FILLS. Every provider ships a free tier (Groq's request-capped
// directing, OpenRouter's `:free` / auto route, Gemini's AI Studio key, an
// OpenCode Go trial seat), but a naive "cheapest free" router fills the pool
// with whatever is free — routing/proxy slugs (groq/compound), 1B toys,
// non-chat endpoints (whisper/tts/embed) — and agentic work collapses: no tool
// calling, no reasoning, tiny contexts. Nothing in the roster validated that a
// free upstream is worth dispatching to.
//
// THIS FILE makes the free pool first-class and quality-gated: a curated
// allowlist of free upstreams (every entry's wire id already witnessed
// somewhere in this repo — never an invented slug), a deny-token gate that
// fails loud on the known-bad classes, fallback CHAINS per routed id so the
// dispatch wiring tries free seats in order instead of failing on the first
// 429, and constructors for a ready-to-use Roster + Manifest pair. It is the
// free-tier analogue of DefaultRoster/DefaultManifest: data the operator can
// dump, edit, and --check, not compiled-in dispatch.
//
// SCOPE (load-bearing). This file is PURE catalog + gate (stdlib only, like the
// rest of the leaf): it names the seats, the quality bar, and the try-order. It
// does NOT probe providers, track quotas, or spend credentials — quota/RPM
// bookkeeping and live failover are the dispatch wiring's job above this seam.
// Capability attestation per upstream (exact context windows, tool support) is
// the operator's `openrouter/models`-style verification at bind time; the gate
// enforces the structural floor (chat model, tool-capable, not a known toy)
// and the allowlist carries the provenance note per entry.

import (
	"fmt"
	"strings"
)

// Routed ids served by the free-tier pool. These are ABSTRACT tier labels (the
// same indirection the account switcher uses): bindings map each to a concrete
// free seat, so rotating a free upstream is a roster edit, never a policy edit.
const (
	// FreeCoding is the default free workhorse: the strongest free coding model.
	FreeCoding = "free-coding"
	// FreeReasoning is the hardest free work: high-complexity and destructive
	// aspects that must never drop to a fast/cheap seat.
	FreeReasoning = "free-reasoning"
	// FreeFlash is short interactive turns and read/search tool calls: fast AND
	// still quality-gated (a strong flash model, never a 1B toy).
	FreeFlash = "free-flash"
	// FreeAuto is the marketplace fallback: OpenRouter's auto route over its
	// free pool, used when the pinned free seats are exhausted.
	FreeAuto = "free-auto"
	// FreeOx is the trial-seat fallback: the OpenCode Go free model.
	FreeOx = "free-ox"
)

// Free-tier seat ids (the accounts side of the bindings).
const (
	FreeSeatGroq       = "free-groq"
	FreeSeatOpenRouter = "free-openrouter"
	FreeSeatGemini     = "free-gemini"
	FreeSeatGeminiGCP  = "free-gemini-vertex"
	FreeSeatOpenCode   = "free-opencode"
)

// FreeModel is one curated free upstream: the routed id, the wire model, the
// seat that serves it, and the quality basis. ContextTokens is 0 when the
// window is operator-verified at bind time rather than pinned here (windows
// drift; pinning a stale number is worse than deferring to the roster's
// context_tokens + a /models check).
type FreeModel struct {
	RoutedID      string `json:"routed_id"`
	Upstream      string `json:"upstream"`
	Account       string `json:"account"`
	ContextTokens int    `json:"context_tokens,omitempty"`
	Tools         bool   `json:"tools"`
	Reasoning     bool   `json:"reasoning,omitempty"`
	// Basis is the provenance note: where this wire id was witnessed.
	Basis string `json:"basis"`
}

// freeQualityContextFloor is the minimum pinned context window a free model may
// declare. Entries with ContextTokens == 0 (operator-verified) skip this check;
// entries pinning a window below it are rejected — a small-context free toy
// thrashes on any real agentic turn.
const freeQualityContextFloor = 32768

// freeDenyTokens is the closed deny set: upstream path-tokens that mark a model
// as NOT high-quality no matter who serves it. Matching is on whole path tokens
// (split on "/:-_ ."), so "11b" never matches the "1b" toy rule and
// "qwen3.6-27b" never trips anything.
var freeDenyTokens = map[string]bool{
	"compound": true, // routing/proxy slug (witnessed low-quality tier)
	"whisper":  true, // speech-to-text, not a chat model
	"tts":      true, // text-to-speech, not a chat model
	"embed":    true, // embedding endpoint, not a chat model
	"1b":       true, // too small for agentic tool/reasoning work
	"tiny":     true,
	"nano":     true,
}

// freeUpstreamTokens splits an upstream model id into matchable path tokens.
func freeUpstreamTokens(upstream string) []string {
	return strings.FieldsFunc(strings.ToLower(upstream), func(r rune) bool {
		return r == '/' || r == ':' || r == '-' || r == '_' || r == ' ' || r == '.'
	})
}

// FreeQualityReason returns "" when the upstream passes the structural quality
// gate, else the human-readable exclusion reason. The gate is three rungs:
// must look like a chat model (no deny token), must be tool-capable (the
// agentic floor — every free-tier route may carry tool calls), and a pinned
// context window must clear the floor.
func FreeQualityReason(upstream string, tools bool, contextTokens int) string {
	for _, tok := range freeUpstreamTokens(upstream) {
		if freeDenyTokens[tok] {
			return fmt.Sprintf("free-tier quality gate: upstream %q carries deny token %q (not a high-quality chat model)", upstream, tok)
		}
	}
	if !tools {
		return fmt.Sprintf("free-tier quality gate: upstream %q is not tool-capable (agentic routes need function calling)", upstream)
	}
	if contextTokens != 0 && contextTokens < freeQualityContextFloor {
		return fmt.Sprintf("free-tier quality gate: upstream %q pins context %d < floor %d", upstream, contextTokens, freeQualityContextFloor)
	}
	return ""
}

// IsHighQualityFree reports whether the upstream clears the quality gate.
func IsHighQualityFree(upstream string, tools bool, contextTokens int) bool {
	return FreeQualityReason(upstream, tools, contextTokens) == ""
}

// DefaultFreeTierModels is the curated allowlist: every wire id below is
// byte-exact from somewhere in this repo (see Basis), never an invented slug.
// groq/compound is DELIBERATELY absent (deny token "compound"); adding it back
// fails ValidateFreeTierRoster loud.
func DefaultFreeTierModels() []FreeModel {
	return []FreeModel{
		{
			RoutedID: FreeCoding, Account: FreeSeatGroq, Upstream: GroqQwen36Model,
			Tools: true, Reasoning: true,
			Basis: "witnessed: account.go GroqQwen36Model + DefaultRoster qwen36-groq binding",
		},
		{
			RoutedID: FreeReasoning, Account: FreeSeatGemini, Upstream: "gemini-3-pro",
			Tools: true, Reasoning: true,
			Basis: "witnessed: model-accounts.example.json drafter-b binding",
		},
		{
			RoutedID: FreeFlash, Account: FreeSeatGeminiGCP, Upstream: "google/gemini-3.5-flash",
			Tools: true,
			Basis: "witnessed: model-accounts.example.json gemini-flash binding",
		},
		{
			RoutedID: FreeAuto, Account: FreeSeatOpenRouter, Upstream: "openrouter/auto",
			Tools: true,
			Basis: "witnessed: account.go openrouter-free binding",
		},
		{
			RoutedID: FreeOx, Account: FreeSeatOpenCode, Upstream: OpenCodeGoOxAlphaModel,
			Tools: true,
			Basis: "witnessed: account.go OpenCodeGoOxAlphaModel + DefaultRoster binding",
		},
	}
}

// DefaultFreeTierRoster builds the ready-to-use free-tier account roster: the
// five free seats (quotas carried from the witnessed Groq profile; the rest are
// keyed seats the operator enables by exporting the cred env var) with one
// binding per curated model. There is deliberately NO Default account: an
// unbound id is a fail-loud error, never a silent pass-through that puts an
// unvetted upstream id on the wire under a free seat's credential. The manifest
// only ever emits vetted ids, so the pair is closed.
func DefaultFreeTierRoster() Roster {
	return Roster{
		Version: RosterVersion,
		Accounts: []Account{
			{
				ID: FreeSeatGroq, Kind: KindOpenAI, BaseURL: GroqOpenAIBaseURL, CredEnv: GroqAPIKeyEnv,
				RequestsPerMinute: GroqQwen36RequestsPerMinute, RequestsPerDay: GroqQwen36RequestsPerDay,
				TokensPerMinute: GroqQwen36TokensPerMinute, TokensPerDay: GroqQwen36TokensPerDay,
				Label: "Groq free tier for Qwen3.6 27B (request/token capped; key in FAK_GROQ_API_KEY)",
			},
			{
				ID: FreeSeatOpenRouter, Kind: KindOpenRouter, CredEnv: OpenRouterAPIKeyEnv,
				Label: "OpenRouter free pool via auto route (key in OPENROUTER_API_KEY; pin :free ids at bind time)",
			},
			{
				ID: FreeSeatGemini, Kind: KindGemini, CredEnv: "GEMINI_API_KEY",
				Label: "Google AI Studio free tier, native Gemini wire (key in GEMINI_API_KEY)",
			},
			{
				ID: FreeSeatGeminiGCP, Kind: KindOpenAI,
				BaseURL: "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/YOUR_GCP_PROJECT/locations/us-central1/endpoints/openapi",
				CredEnv: "FAK_GEMINI_GCP_KEY",
				Label:   "Gemini free/burst via Vertex OpenAI-compatible endpoint (GCP access token in FAK_GEMINI_GCP_KEY)",
			},
			{
				ID: FreeSeatOpenCode, Kind: KindOpenAI, BaseURL: OpenCodeGoOpenAIBaseURL, CredEnv: OpenCodeGoAPIKeyEnv,
				Label: "OpenCode Go trial seat (key in OPENCODE_GO_API_KEY)",
			},
		},
		Bindings: []Binding{
			{Model: FreeCoding, Account: FreeSeatGroq, UpstreamModel: GroqQwen36Model},
			{Model: FreeReasoning, Account: FreeSeatGemini, UpstreamModel: "gemini-3-pro"},
			{Model: FreeFlash, Account: FreeSeatGeminiGCP, UpstreamModel: "google/gemini-3.5-flash"},
			{Model: FreeAuto, Account: FreeSeatOpenRouter, UpstreamModel: "openrouter/auto"},
			{Model: FreeOx, Account: FreeSeatOpenCode, UpstreamModel: OpenCodeGoOxAlphaModel},
		},
	}
}

// freeTierFallbacks is the try-order per routed id for the dispatch wiring:
// on a 429/exhaustion the wiring walks the chain instead of failing the turn.
// Every chain stays inside the quality-gated pool — exhaustion degrades to
// another HIGH-QUALITY free seat, never to an unvetted model.
var freeTierFallbacks = map[string][]string{
	FreeCoding:    {FreeCoding, FreeAuto, FreeOx},
	FreeReasoning: {FreeReasoning, FreeCoding, FreeAuto},
	FreeFlash:     {FreeFlash, FreeCoding, FreeAuto},
	FreeAuto:      {FreeAuto, FreeCoding},
	FreeOx:        {FreeOx, FreeAuto},
}

// FreeFallbackChain returns the ordered fallback chain for a free-tier routed
// id (a copy; unknown ids yield nil so the caller fails loud, never silently
// picks a seat).
func FreeFallbackChain(routedID string) []string {
	chain, ok := freeTierFallbacks[routedID]
	if !ok {
		return nil
	}
	out := make([]string, len(chain))
	copy(out, chain)
	return out
}

// DefaultFreeTierManifest is the ready-to-use routing policy over the free-tier
// ids: hard and destructive work pins to free-reasoning (the floor that never
// drops, mirroring the dev-models security-release floor), ordinary
// medium-and-up work goes to free-coding, and cheap/read/search plus short
// interactive turns go to free-flash. The fail-closed default is free-coding.
// Rules are first-match-wins, most specific first: note MinComplexity is a
// FLOOR (a "low" rule also matches medium), so the medium band needs its own
// rule ahead of the cheap one — the same ordering the dev-models preset uses.
func DefaultFreeTierManifest() Manifest {
	return Manifest{
		Version: Version,
		Default: Plan{
			Members: []Member{{Model: FreeCoding, Role: "primary"}},
			Reason:  "free-tier default: unmatched work lands on the strongest free coding model",
		},
		Rules: []Rule{
			{
				Name:  "free-ultrahard-reasoning",
				Match: Match{MinComplexity: ComplexityHigh},
				Plan:  Plan{Members: []Member{{Model: FreeReasoning}}, Reason: "high-complexity work -> strongest free reasoning model"},
			},
			{
				Name:  "free-destructive-delete",
				Match: Match{Aspect: AspectToolCall, Tool: "delete_*"},
				Plan:  Plan{Members: []Member{{Model: FreeReasoning}}, Reason: "destructive tool call -> strongest free model, never a fast seat"},
			},
			{
				Name:  "free-destructive-drop",
				Match: Match{Aspect: AspectToolCall, Tool: "drop_*"},
				Plan:  Plan{Members: []Member{{Model: FreeReasoning}}, Reason: "destructive tool call -> strongest free model, never a fast seat"},
			},
			{
				Name:  "free-normal-coding",
				Match: Match{MinComplexity: ComplexityMedium},
				Plan:  Plan{Members: []Member{{Model: FreeCoding}}, Reason: "ordinary medium-and-up work -> strongest free coding model (ahead of the cheap rule: the floor matches up)"},
			},
			{
				Name:  "free-cheap-flash",
				Match: Match{MinComplexity: ComplexityLow},
				Plan:  Plan{Members: []Member{{Model: FreeFlash}}, Reason: "low-complexity aspect -> high-quality free flash model"},
			},
			{
				Name:  "free-read-flash",
				Match: Match{Aspect: AspectToolCall, Tool: "read_*"},
				Plan:  Plan{Members: []Member{{Model: FreeFlash}}, Reason: "read-shaped tool call -> high-quality free flash model"},
			},
			{
				Name:  "free-search-flash",
				Match: Match{Aspect: AspectToolCall, Tool: "search_*"},
				Plan:  Plan{Members: []Member{{Model: FreeFlash}}, Reason: "search tool call -> high-quality free flash model"},
			},
			{
				Name:  "free-short-interactive",
				Match: Match{Aspect: AspectRequest, Latency: LatencyInteractive, MaxPromptTokens: 4096},
				Plan:  Plan{Members: []Member{{Model: FreeFlash}}, Reason: "short interactive turn -> high-quality free flash model"},
			},
		},
	}
}

// ValidateFreeTierRoster checks a roster is BOTH structurally valid AND
// quality-gated: every binding's upstream must clear the gate (so an operator
// edit adding groq/compound or a 1b toy fails here, never at dispatch). A
// Default (when set) must name a free seat; empty is the preferred closed
// form — unbound ids fail loud instead of passing an unvetted upstream to a
// seat. A ManualOnly default is already refused by Roster.Validate.
func ValidateFreeTierRoster(r Roster) error {
	if err := r.Validate(); err != nil {
		return err
	}
	byAccount := make(map[string]Account, len(r.Accounts))
	for _, a := range r.Accounts {
		byAccount[a.ID] = a
	}
	for _, b := range r.Bindings {
		upstream := b.UpstreamModel
		if upstream == "" {
			upstream = b.Model
		}
		// Tool-capability is attested by the curated allowlist; an operator-added
		// binding is assumed tool-capable unless its id says otherwise — the gate
		// still rejects deny-token and small-context pins loud.
		tools := true
		for _, tok := range freeUpstreamTokens(upstream) {
			if tok == "notools" {
				tools = false
			}
		}
		var ctxTokens int
		if a, ok := byAccount[b.Account]; ok {
			ctxTokens = a.ContextTokens
		}
		if reason := FreeQualityReason(upstream, tools, ctxTokens); reason != "" {
			return fmt.Errorf("modelroute: free-tier binding %q -> %q rejected: %s", b.Model, upstream, reason)
		}
	}
	if r.Default != "" {
		if _, ok := byAccount[r.Default]; !ok {
			return fmt.Errorf("modelroute: free-tier default account %q is not a free seat", r.Default)
		}
	}
	return nil
}
