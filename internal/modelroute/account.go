package modelroute

// account.go — the generic ACCOUNT SWITCHER, the second half of "route any aspect
// to any model": once Route(Subject) has chosen a Plan of abstract model ids, the
// switcher BINDS each id to a concrete dispatch target — WHICH provider, WHICH of
// the user's accounts, and the upstream model name to put on the wire.
//
// THE GAP IT FILLS. modelroute.go decides the abstract model id ("small", "large",
// "guard-a"); agent.HTTPPlanner can talk to ONE upstream/account configured by
// flags. Nothing in between let a USER bring their OWN accounts and mix providers
// at any level — route the cheap aspect to a LOCAL ollama, the hard reasoning step
// to their OpenAI account, a guard ensemble half to OpenAI and half to their
// Anthropic subscription. That binding is what this file makes first-class: a
// declarative, version-tagged Roster of Accounts + Bindings, resolved by a pure,
// deterministic Resolve. It is the generic, in-product form of the private fleet's
// account switcher (tools/fleet_accounts.py) — provider-neutral, credential-safe,
// and composable with the routing spine.
//
// THE SWITCHER UNIT is an Account: a NAMED credential set for a provider. Two
// accounts can target the SAME provider kind ("openai-personal" vs "openai-work"),
// which is exactly the switch — pick WHICH credential serves a model. A credential
// is held as an ENV-VAR REFERENCE (CredEnv), never the secret itself, so a Roster
// is safe to commit and diff; Validate rejects a value that is not a valid env-var
// name (a pasted "sk-…" key fails loud instead of leaking into the manifest), and
// the secret is dereferenced (os.Getenv) ONLY at planner-build time in the deferred
// dispatch layer — it never enters a Target, the manifest, EngineRoute, or any dump.
//
// MIX AND MATCH AT ANY LEVEL falls out of the model: a Binding maps ONE routed
// model id (a Plan member OR the Plan's scout) to one account, and Plan members are
// distinct ids, so an ensemble's members can each bind to a different account /
// provider, and the cheap scout-classify probe can switch accounts independently.
// There is no per-aspect special case — the same id→account table serves a request,
// a tool call, a reasoning step, a scout, or any ensemble member alike.
//
// RESIDENCY IS A DECLARED PROPERTY, NOT A GUESS (load-bearing). internal/engine's
// residency PDP denies a tenant/sensitive payload bound for a REMOTE engine, and it
// reads the route string written to abi.ToolCall.Engine. Target.EngineRoute() stamps
// that string with a STRUCTURAL "local:" / "<kind>:" prefix derived from the
// account's Kind — so the floor's local/remote classification is the account's
// DECLARED kind, never a substring guess about the model name. Locality has ONE
// source of truth (Kind == KindLocal); there is no second bool that could disagree.
// Validate forbids a local account from carrying a REMOTE base URL, which would
// otherwise let a "local:" route egress off-box (a residency-floor bypass). The route
// MUST still be written BEFORE Kernel.Submit (route-before-adjudicate), the same
// contract the routing spine pins; ResolvePlan is PURE resolution — it never sets
// Engine or Submits.
//
// The package stays pure (stdlib only): Resolve produces a Target VALUE; building an
// agent.HTTPPlanner from it (and running an ensemble's members) is the additive
// dispatch wiring above this seam, tracked with the rest of the live-dispatch epic.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"regexp"
	"strings"
)

// RosterVersion is the account-switcher manifest schema tag. It is DISTINCT from the
// routing Manifest's Version: the two are separate files (a routing policy and an
// account roster), versioned independently. A roster MAY omit it (treated as
// current); a roster naming a different major is refused.
const RosterVersion = "fak-accounts/v1"

const (
	// OpenCodeGoProviderKey is the account-roster key for an OpenCode Go subscription.
	OpenCodeGoProviderKey = "opencode-go"
	// OpenCodeGoAPIKeyEnv names the environment variable holding the subscription key.
	OpenCodeGoAPIKeyEnv = "OPENCODE_GO_API_KEY"
	// OpenCodeGoOpenAIBaseURL is OpenCode Go's OpenAI-compatible API root.
	OpenCodeGoOpenAIBaseURL = "https://opencode.ai/zen/go/v1"
	// OpenCodeGoOxAlphaModel is the limited-time free Ox Alpha model id.
	OpenCodeGoOxAlphaModel = "ox-alpha-free"
	// DeepSeekProviderKey is the account-roster provider key for DeepSeek's
	// OpenAI-compatible wire.
	DeepSeekProviderKey = "deepseek"
	// DeepSeekAPIKeyEnv is the credential env-var name DeepSeek examples use.
	DeepSeekAPIKeyEnv = "DEEPSEEK_API_KEY"
	// DeepSeekOpenAIBaseURL is DeepSeek's OpenAI-compatible API root.
	DeepSeekOpenAIBaseURL = "https://api.deepseek.com"
	// DeepSeekAnthropicBaseURL is DeepSeek's Anthropic-compatible API root.
	DeepSeekAnthropicBaseURL = "https://api.deepseek.com/anthropic"
	// GPT6AstraModel is the OpenAI GPT-6 Astra flagship model id.
	GPT6AstraModel = "gpt-6-astra"
	// GPT6AstraWindowTokens is the documented GPT-6 Astra context window.
	GPT6AstraWindowTokens = 1_000_000
	// GPT6AstraMaxOutputTokens is the documented GPT-6 Astra maximum output tokens.
	GPT6AstraMaxOutputTokens = 131_072
	// DeepSeekV4ProModel is the DeepSeek V4 Pro API model id.
	DeepSeekV4ProModel = "deepseek-v4-pro"
	// DeepSeekV4FlashModel is the DeepSeek V4 Flash API model id.
	DeepSeekV4FlashModel = "deepseek-v4-flash"
	// DeepSeekV4ContextTokens is the documented DeepSeek V4 context window.
	DeepSeekV4ContextTokens = 1_000_000
	// DeepSeekV4MaxOutputTokens is the documented DeepSeek V4 maximum output.
	DeepSeekV4MaxOutputTokens = 384_000
	// DeepSeekLegacyAliasRetiresUTC is the documented retirement time for the
	// legacy deepseek-chat / deepseek-reasoner aliases.
	DeepSeekLegacyAliasRetiresUTC = "2026-07-24 15:59 UTC"
	// GroqAPIKeyEnv is the credential env-var name the Groq examples use.
	GroqAPIKeyEnv = "FAK_GROQ_API_KEY"
	// GroqOpenAIBaseURL is Groq's OpenAI-compatible API root.
	GroqOpenAIBaseURL = "https://api.groq.com/openai/v1"
	// GroqQwen36Model is the Groq model slug for Alibaba Cloud Qwen3.6 27B.
	GroqQwen36Model = "qwen/qwen3.6-27b"
	// GroqQwen36* limits record the published per-account ceiling for this model.
	GroqQwen36RequestsPerMinute = 30
	GroqQwen36RequestsPerDay    = 1_000
	GroqQwen36TokensPerMinute   = 8_000
	GroqQwen36TokensPerDay      = 200_000
	// GroqCompoundModel is Groq's lower-quality Compound routing model slug.
	GroqCompoundModel = "groq/compound"
	// GroqCompound* limits are request-count ceilings; this profile has no token
	// minute/day limit, so the token fields stay zero in the account metadata.
	GroqCompoundRequestsPerMinute = 30
	GroqCompoundRequestsPerDay    = 250
	// OpenRouterProviderKey is the account-roster provider key for OpenRouter's
	// OpenAI-compatible wire.
	OpenRouterProviderKey = "openrouter"
	// OpenRouterAPIKeyEnv is the credential env-var name OpenRouter examples use.
	OpenRouterAPIKeyEnv = "OPENROUTER_API_KEY"
	// OpenRouterOpenAIBaseURL is OpenRouter's OpenAI-compatible API root.
	OpenRouterOpenAIBaseURL = "https://openrouter.ai/api/v1"
)

// ---------------------------------------------------------------------------
// PROVIDER KIND — the wire protocol an account speaks (a CLOSED additive set).
// ---------------------------------------------------------------------------

// ProviderKind is the transcript wire an account speaks. It is RE-DECLARED here
// rather than imported from internal/agent ON PURPOSE: modelroute is pure stdlib by
// contract (the property that lets it compose with the frozen ABI seam without
// pulling the agent loop into the routing spine), and internal/agent is not stdlib.
// Most remote kinds mirror agent.Provider 1:1; "local" is the modelroute addition
// (an on-box, OpenAI-compatible server), and DeepSeek is an OpenAI-compatible remote
// profile with its own default host. The named cost of the boundary: a new native
// provider must be added in BOTH places. It is a CLOSED set — a new kind is an added
// constant + validation, never manifest free text.
type ProviderKind string

const (
	// KindOpenAI is the OpenAI-compatible /chat/completions wire (OpenAI, and any
	// remote server that speaks it: Together, Groq, Fireworks, …).
	KindOpenAI ProviderKind = "openai"
	// KindOpenAIResponses is the OpenAI Responses-API item wire (what `codex` speaks).
	KindOpenAIResponses ProviderKind = "openai-responses"
	// KindAnthropic is the Anthropic Claude Messages API (an API key OR a Pro/Max
	// subscription OAuth token — the adapter picks the header scheme by token shape).
	KindAnthropic ProviderKind = "anthropic"
	// KindGemini is the Google Gemini generateContent API.
	KindGemini ProviderKind = "gemini"
	// KindXAI is the xAI Grok chat-completions wire (OpenAI-compatible).
	KindXAI ProviderKind = "xai"
	// KindDeepSeek is DeepSeek's OpenAI-compatible Chat Completions wire. The
	// Anthropic-compatible DeepSeek endpoint is represented as KindAnthropic with
	// DeepSeekAnthropicBaseURL, so generic /models readiness does not mis-probe it.
	KindDeepSeek ProviderKind = DeepSeekProviderKey
	// KindOpenRouter is OpenRouter's OpenAI-compatible Chat Completions wire.
	// It is a provider marketplace that routes to 300+ upstream models.
	KindOpenRouter ProviderKind = OpenRouterProviderKey
	// KindLocal is an on-box, OpenAI-compatible server (ollama / vLLM / llama.cpp /
	// the in-kernel model). It is the ONLY local kind, so locality is exactly
	// Kind == KindLocal — there is no separate flag. A call routed to it is
	// residency-EXEMPT (the bytes never leave the box), which is why Validate forbids
	// it from carrying a non-loopback base URL.
	KindLocal ProviderKind = "local"
	// KindFleet is an OpenAI-compatible server the ORGANIZATION operates — off-box
	// but inside the org's own trust boundary (a GLM/Kimi-class open model on
	// company GPUs). It is the middle rung of the placement ladder, and the one
	// the token economics rest on: see PlacementZone in zone.go.
	//
	// It is REMOTE to the residency floor exactly as a vendor kind is. Declaring
	// hardware org-owned does not by itself make a sensitive payload safe to send
	// off the box; the zone makes the deployment EXPRESSIBLE and ATTRIBUTABLE
	// (Target.Zone().SelfHosted()), and any widening of the floor is a separate,
	// operator-declared enforcement change.
	KindFleet ProviderKind = "fleet"
)

// knownKind reports whether k is one of the closed ProviderKind set.
func knownKind(k ProviderKind) bool {
	switch k {
	case KindOpenAI, KindOpenAIResponses, KindAnthropic, KindGemini, KindXAI, KindDeepSeek, KindOpenRouter, KindLocal, KindFleet:
		return true
	}
	return false
}

// remoteKind reports whether dispatching to kind k leaves the box. KindLocal is the
// only local kind.
func remoteKind(k ProviderKind) bool { return k != KindLocal }

// KindBaseURL is the public default base URL for a REMOTE provider kind, used when an
// Account omits base_url. It matches the defaults `fak guard`/`fak serve` use
// (Anthropic WITHOUT a /v1 suffix — its adapter appends the Messages path; a parity
// test pins this against cmd/fak's guardDefaultBaseURL). KindLocal and KindFleet have
// no public default (returns ""), so a local account MUST set an explicit loopback
// base_url and a fleet account an explicit org-reachable one.
func KindBaseURL(k ProviderKind) string {
	switch k {
	case KindOpenAI, KindOpenAIResponses:
		return "https://api.openai.com/v1"
	case KindAnthropic:
		return "https://api.anthropic.com"
	case KindGemini:
		return "https://generativelanguage.googleapis.com/v1beta"
	case KindXAI:
		return "https://api.x.ai/v1"
	case KindDeepSeek:
		return DeepSeekOpenAIBaseURL
	case KindOpenRouter:
		return OpenRouterOpenAIBaseURL
	}
	return ""
}

// ---------------------------------------------------------------------------
// THE ROSTER — accounts the user brings + the id→account bindings.
// ---------------------------------------------------------------------------

// envNameRE matches a POSIX-ish environment variable NAME. A credential reference
// must look like a name (e.g. OPENAI_API_KEY), so a pasted secret ("sk-ant-…", a
// "Bearer …" string, an "X=Y" pair — all carry '-'/'.'/' '/'=') fails Validate
// instead of silently landing in the manifest.
var envNameRE = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// The auth schemes an Account may declare. These MIRROR internal/agent's
// AnthropicAuthScheme values as plain strings rather than importing them, the same
// deliberate stdlib-purity trade this file already makes for provider kinds (see the
// note above KindOpenAI): internal/agent imports modelroute, so importing back would
// cycle. TestAuthSchemeMirrorsAgent (in internal/agent, which can see both) pins the
// two lists together so a rename cannot drift them apart silently.
const (
	// AuthSchemeDefault leaves the provider adapter's own credential handling alone.
	AuthSchemeDefault = ""
	// AuthSchemeBearer presents the credential as `Authorization: Bearer <token>`.
	AuthSchemeBearer = "bearer"
	// AuthSchemeAPIKey presents the credential as the Anthropic-wire `x-api-key` header.
	AuthSchemeAPIKey = "x-api-key"
)

// validAuthScheme reports whether s is a scheme some adapter understands. Kept as a
// closed set so a typo ("Bearer", "token") fails Validate loudly instead of silently
// falling back to the shape sniff whose 401 the operator was trying to fix.
func validAuthScheme(s string) bool {
	switch s {
	case AuthSchemeDefault, AuthSchemeBearer, AuthSchemeAPIKey:
		return true
	default:
		return false
	}
}

// Account is the switcher unit: a named credential set for one provider. ID is the
// handle a Binding references. Kind is the wire (and the SOLE locality signal). BaseURL
// overrides the kind's public default (REQUIRED, and loopback-only, for a local
// server). CredEnv NAMES the env var holding the API key or subscription token — never
// the secret itself; it is required for a remote account and forbidden-to-be-a-secret
// for all.
type Account struct {
	ID                string       `json:"id"`
	Kind              ProviderKind `json:"kind"`
	BaseURL           string       `json:"base_url,omitempty"`
	CredEnv           string       `json:"cred_env,omitempty"`
	Label             string       `json:"label,omitempty"`
	ContextTokens     int          `json:"context_tokens,omitempty"`
	MaxOutputTokens   int          `json:"max_output_tokens,omitempty"`
	RequestsPerMinute int          `json:"requests_per_minute,omitempty"`
	RequestsPerDay    int          `json:"requests_per_day,omitempty"`
	TokensPerMinute   int          `json:"tokens_per_minute,omitempty"`
	TokensPerDay      int          `json:"tokens_per_day,omitempty"`
	// Principals is the OPTIONAL closed set of tenant ISOLATION principals — the
	// org/project identity a gateway keyset binds an inbound api key to (#5332) —
	// admitted to dispatch through this account. EMPTY means UNRESTRICTED: every
	// principal may reach the account, which is byte-for-byte the behavior of every
	// roster written before this field existed. A NON-EMPTY list is a fail-CLOSED
	// allowlist enforced at the routing boundary (before Submit), so one tenant can
	// never dispatch through another tenant's credential — the residency arm of the
	// keyset. It scopes WHICH accounts a tenant may reach; it is NOT an authority
	// grant and says nothing about whether a turn may consume user consent.
	Principals []string `json:"principals,omitempty"`
	// ManualOnly withholds this account from every AUTOMATIC selection pool while
	// leaving it fully reachable by an EXPLICIT, named request.
	//
	// WHY IT EXISTS. Registering an account is currently the same act as volunteering
	// it: once a binding names it, the placement/escalation walk (cmd/fak's
	// placementCandidates → Roster.Place) may pick it on its own, and Roster.Default
	// makes it the silent fallback for any unbound id. That is right for a pool of
	// interchangeable seats and WRONG for an account a human wants held in reserve — a
	// separately-billed vendor endpoint, a low-quota tenant credential, a metered
	// gateway — where every call should be one somebody asked for. Without this flag the
	// only way to keep such an account out of the automatic pool is to leave it out of
	// the roster, which also makes it unreachable: the registry cannot express
	// "available, never volunteered".
	//
	// It is a supply property of the CREDENTIAL, so it lives on the account rather than
	// on each binding: marking it once covers every model bound to it, and a new binding
	// cannot forget it. It is NOT an authorization control — it says nothing about WHO
	// may dispatch (that is Principals) and it does not stop a caller who names the
	// account. It only removes the account from pools that choose on their own.
	ManualOnly bool `json:"manual_only,omitempty"`
	// AuthScheme overrides how this account's credential is presented on the wire, for
	// the case where the credential's SHAPE is not a reliable discriminator. It is the
	// roster-level spelling of agent.AnthropicAuthScheme ("", "bearer", "x-api-key"):
	// empty keeps the provider adapter's own default, which for the Anthropic wire
	// sniffs sk-ant-oat => bearer / else x-api-key. A THIRD-PARTY Anthropic-COMPATIBLE
	// endpoint authenticates its own tenant token — a prefix fak cannot know — and
	// generally accepts it only as a bearer, so the sniff sends x-api-key and the call
	// 401s with the base URL, model and body all correct. Declaring the scheme here
	// keeps that a CONFIG fact (the CONFIG_NOT_ENV ratchet, #2863) rather than a new
	// environment variable. Validate refuses a value no adapter understands.
	AuthScheme string `json:"auth_scheme,omitempty"`
}

// Binding maps ONE routed model id (a Plan member's Model, or a Plan's Scout) to the
// account that serves it and the UpstreamModel wire name. The routed id is an
// ABSTRACT tier label; UpstreamModel is the provider-specific id sent on the wire
// (empty => the routed id is used verbatim). Distinct members bind independently, so
// an ensemble can span accounts/providers — the "mix and match at any level" the goal
// asks for.
type Binding struct {
	Model              string `json:"model"`
	Account            string `json:"account"`
	UpstreamModel      string `json:"upstream_model,omitempty"`
	CompatibilityOnly  bool   `json:"compatibility_only,omitempty"`
	DeprecatedAfterUTC string `json:"deprecated_after_utc,omitempty"`
	DeprecatedAliasFor string `json:"deprecated_alias_for,omitempty"`
}

// Roster is the on-disk account-switcher manifest: the accounts a user brings and the
// id→account bindings, plus a Default account for a routed id with no explicit
// binding. Default may be empty, in which case an unbound id is a fail-loud error
// (never a silent fallback to an arbitrary account).
type Roster struct {
	Version  string    `json:"version,omitempty"`
	Accounts []Account `json:"accounts"`
	Bindings []Binding `json:"bindings,omitempty"`
	Default  string    `json:"default,omitempty"`
	// SpawnClasses declares which work class each of the fleet's sub-agent TYPES does
	// (epic #5416, track E). Optional and omitempty: a roster without it is unchanged,
	// and an undeclared type stays undeclared rather than defaulting to a class. See
	// spawnclass.go for why the declaration lives here rather than being inferred.
	SpawnClasses []SpawnClass `json:"spawn_classes,omitempty"`
}

// Target is the resolved dispatch destination for one routed model id: which account
// serves it, the provider kind + concrete base URL + credential env var NAME, and the
// upstream wire model name. It is a VALUE — the dispatch wiring turns it into an
// agent.HTTPPlanner; this package never does I/O. CredEnv is a NAME, never the secret.
type Target struct {
	Model             string       `json:"model"`          // the routed id (the Plan member / scout)
	Account           string       `json:"account"`        // the resolved Account.ID
	Kind              ProviderKind `json:"kind"`           // the provider wire
	BaseURL           string       `json:"base_url"`       // concrete (account override or kind default)
	CredEnv           string       `json:"cred_env"`       // env var NAME for the credential ("" = local)
	UpstreamModel     string       `json:"upstream_model"` // the wire model name
	ContextTokens     int          `json:"context_tokens,omitempty"`
	MaxOutputTokens   int          `json:"max_output_tokens,omitempty"`
	RequestsPerMinute int          `json:"requests_per_minute,omitempty"`
	RequestsPerDay    int          `json:"requests_per_day,omitempty"`
	TokensPerMinute   int          `json:"tokens_per_minute,omitempty"`
	TokensPerDay      int          `json:"tokens_per_day,omitempty"`
	// Principals carries the resolving Account's tenant-isolation allowlist through to
	// the dispatch boundary so Admits can adjudicate it there (#5332). Empty =>
	// unrestricted, exactly as on the Account.
	Principals []string `json:"principals,omitempty"`
	// ManualOnly reports that the resolving account is held in reserve: reachable only
	// because this resolution NAMED it, never volunteered by an automatic pool. It rides
	// on the Target so the dispatch layer and the ledger can record that a reserved
	// credential was spent by explicit request — the audit question a reserved account
	// exists to answer.
	ManualOnly bool `json:"manual_only,omitempty"`
	// AuthScheme carries the account's credential-presentation override to the planner
	// build, where it becomes agent.AnthropicAuthScheme. Empty => the adapter default.
	AuthScheme string `json:"auth_scheme,omitempty"`
}

// Admits reports whether the given tenant ISOLATION principal may dispatch through
// this target's account (#5332, the residency arm of the gateway keyset). An account
// naming NO principals is unrestricted and admits everyone — the pre-#5332 roster is
// unchanged. An account naming ANY is a fail-CLOSED allowlist:
//
//   - only an exact, whitespace-trimmed member is admitted. The compare is never a
//     prefix or a glob, so "acme" never admits "acme-evil";
//   - the EMPTY principal — what a caller presents when no keyset key bound it to a
//     tenant, including the single --require-key-env bearer — is NEVER admitted to a
//     restricted account. An unattributable caller cannot inherit a tenant's credential;
//   - a list of only blank entries admits NOBODY rather than degrading to unrestricted,
//     so a typo'd roster fails closed instead of opening the account to everyone.
func (t Target) Admits(principal string) bool {
	if len(t.Principals) == 0 {
		return true
	}
	want := strings.TrimSpace(principal)
	if want == "" {
		return false
	}
	for _, p := range t.Principals {
		if strings.TrimSpace(p) == want {
			return true
		}
	}
	return false
}

// Local reports whether this target dispatches to an on-box server. It is DERIVED
// from Kind (the single source of truth), so it can never disagree with a separate
// flag the residency floor might trust.
func (t Target) Local() bool { return t.Kind == KindLocal }

// Remote reports whether dispatching to this target leaves the box (the inverse of
// Local). The residency floor denies a sensitive payload only on a Remote target.
func (t Target) Remote() bool { return !t.Local() }

// EngineRoute returns the value the host writes to abi.ToolCall.Engine for this
// target. It is STRUCTURALLY honest about locality: a local target is prefixed
// "local:" (which internal/engine's residency PDP reads as on-box, residency-exempt,
// via a first-checked early-return) and a remote target "<kind>:" where <kind> is one
// of the floor-recognized keywords (openai / openai-responses⊃openai / anthropic /
// gemini / xai / deepseek / fleet) — so the floor's local/remote decision is the account's DECLARED kind,
// never a guess from whether the model name contains "openai". A KindFleet target
// therefore leads with "fleet:", which the floor reads as REMOTE (org-owned is still
// off-box) while ZoneOfRoute reads it as ZoneFleet — one string, two honest answers:
// the floor gets locality, the usage ledger gets self-hosted attribution. The account/upstream
// follow for legibility; upstream model ids may contain provider namespace slashes
// (for example qwen/qwen3.6-27b), so consumers that parse this shape must split only
// on the first slash after the account id. The
// invariant `engine.remoteRoute(EngineRoute()) == Remote()` is pinned by a
// cross-package test, not left coincidental.
func (t Target) EngineRoute() string {
	up := t.UpstreamModel
	if up == "" {
		up = t.Model
	}
	prefix := string(t.Kind)
	if t.Local() {
		prefix = "local"
	}
	return prefix + ":" + t.Account + "/" + up
}

// ResolvedPlan binds a whole routed Plan to concrete targets: the optional Scout
// target (the cheap classify-first probe is its OWN routed aspect, so it can switch
// accounts) and the Member targets in Plan.Members order (the ensemble fold inputs,
// in the order Combine requires). It is pure resolution — no Submit, no Engine
// mutation; the host writes each EngineRoute() to abi.ToolCall.Engine BEFORE
// Kernel.Submit, one independently-adjudicated call per member.
type ResolvedPlan struct {
	Scout   *Target  `json:"scout,omitempty"`
	Members []Target `json:"members"`
}

// account returns the Account with the given id.
func (r Roster) account(id string) (Account, bool) {
	for _, a := range r.Accounts {
		if a.ID == id {
			return a, true
		}
	}
	return Account{}, false
}

// Resolve binds a routed model id to its concrete Target: the id's explicit Binding
// if any, else the Default account (the id is used as the upstream name); an id with
// no binding and no default is a fail-loud error. Pure and deterministic — same
// roster, same id, same target. Resolve assumes a validated roster (the dangling-ref
// and locality invariants are enforced in Validate); it still returns an error for an
// unbound id with no default and a binding to an account that does not exist.
func (r Roster) Resolve(modelID string) (Target, error) {
	acctID, upstream := "", modelID
	bound := false
	for _, b := range r.Bindings {
		if b.Model == modelID {
			acctID = b.Account
			if b.UpstreamModel != "" {
				upstream = b.UpstreamModel
			}
			bound = true
			break
		}
	}
	if !bound {
		if r.Default == "" {
			return Target{}, fmt.Errorf("modelroute: no binding for model %q and no default account", modelID)
		}
		acctID = r.Default
	}
	a, ok := r.account(acctID)
	if !ok {
		return Target{}, fmt.Errorf("modelroute: model %q binds to unknown account %q", modelID, acctID)
	}
	baseURL := a.BaseURL
	if baseURL == "" {
		baseURL = KindBaseURL(a.Kind)
	}
	return Target{
		Model:             modelID,
		Account:           a.ID,
		Kind:              a.Kind,
		BaseURL:           baseURL,
		CredEnv:           a.CredEnv,
		UpstreamModel:     upstream,
		ContextTokens:     a.ContextTokens,
		MaxOutputTokens:   a.MaxOutputTokens,
		RequestsPerMinute: a.RequestsPerMinute,
		RequestsPerDay:    a.RequestsPerDay,
		TokensPerMinute:   a.TokensPerMinute,
		TokensPerDay:      a.TokensPerDay,
		Principals:        a.Principals,
		ManualOnly:        a.ManualOnly,
		AuthScheme:        a.AuthScheme,
	}, nil
}

// ResolvePlan binds a whole Plan: the Scout (when set) and every Member, IN MEMBER
// ORDER (the same determinism contract Combine relies on). The first id that cannot
// resolve is a fail-loud error, so a misconfigured roster never silently drops an
// ensemble member or the scout.
func (r Roster) ResolvePlan(p Plan) (ResolvedPlan, error) {
	var rp ResolvedPlan
	if p.Scout != "" {
		t, err := r.Resolve(p.Scout)
		if err != nil {
			return ResolvedPlan{}, fmt.Errorf("scout: %w", err)
		}
		rp.Scout = &t
	}
	rp.Members = make([]Target, 0, len(p.Members))
	for _, m := range p.Members {
		t, err := r.Resolve(m.Model)
		if err != nil {
			return ResolvedPlan{}, err
		}
		rp.Members = append(rp.Members, t)
	}
	return rp, nil
}

// ResolveDecision binds a routed Decision's Plan — the single entry point the CLI and
// the future dispatch wiring share, so member order and the scout case live in one
// place.
func (r Roster) ResolveDecision(d Decision) (ResolvedPlan, error) { return r.ResolvePlan(d.Plan) }

// ---------------------------------------------------------------------------
// VALIDATION — fail-loud, so a misconfigured switch never mis-dispatches or leaks.
// ---------------------------------------------------------------------------

// Validate checks a Roster is well-formed and SAFE. The invariants (each a fail-loud
// boundary, never a runtime surprise):
//   - a known major version; >= 1 account;
//   - each account: a non-empty, unique, delimiter-free id not beginning with a
//     reserved local token; a known kind; a credential reference that is an env-var
//     NAME (not a pasted secret); a remote account carries a credential, a LOCAL
//     account carries a loopback base_url and no remote host (the residency invariant);
//     each rate-limit field non-negative, and no per-minute ceiling above its per-day
//     counterpart (a 0 per-day = unlimited exempts the request-gated-only profile);
//   - each binding: a non-empty, delimiter-free model bound to a real account, unique
//     per model id, with an upstream that may carry provider namespace slashes but
//     not the route's kind delimiter or whitespace;
//   - a Default (when set) naming a real account, and NOT a manual_only one (that
//     would make a reserved account the silent fallback for every unbound id);
//   - each account's auth_scheme, when set, one an adapter understands.
//
// A misconfigured switch must fail here, never fall through to an arbitrary account,
// egress a "local" route off-box, or leak a secret.
func (r Roster) Validate() error {
	if r.Version != "" && !strings.HasPrefix(r.Version, RosterVersion) {
		return fmt.Errorf("modelroute: roster version %q is not %s.x", r.Version, RosterVersion)
	}
	if len(r.Accounts) == 0 {
		return fmt.Errorf("modelroute: roster has no accounts")
	}
	seen := make(map[string]bool, len(r.Accounts))
	for i, a := range r.Accounts {
		if a.ID == "" {
			return fmt.Errorf("modelroute: account %d has an empty id", i)
		}
		if err := safeRouteToken("account id", a.ID); err != nil {
			return err
		}
		if seen[a.ID] {
			return fmt.Errorf("modelroute: duplicate account id %q", a.ID)
		}
		seen[a.ID] = true
		if !knownKind(a.Kind) {
			return fmt.Errorf("modelroute: account %q has unknown kind %q", a.ID, a.Kind)
		}
		if !validAuthScheme(a.AuthScheme) {
			return fmt.Errorf("modelroute: account %q has unknown auth_scheme %q (want %q, %q, or omitted)",
				a.ID, a.AuthScheme, AuthSchemeBearer, AuthSchemeAPIKey)
		}
		if a.ContextTokens < 0 {
			return fmt.Errorf("modelroute: account %q context_tokens must be non-negative", a.ID)
		}
		if a.MaxOutputTokens < 0 {
			return fmt.Errorf("modelroute: account %q max_output_tokens must be non-negative", a.ID)
		}
		if a.RequestsPerMinute < 0 {
			return fmt.Errorf("modelroute: account %q requests_per_minute must be non-negative", a.ID)
		}
		if a.RequestsPerDay < 0 {
			return fmt.Errorf("modelroute: account %q requests_per_day must be non-negative", a.ID)
		}
		if a.TokensPerMinute < 0 {
			return fmt.Errorf("modelroute: account %q tokens_per_minute must be non-negative", a.ID)
		}
		if a.TokensPerDay < 0 {
			return fmt.Errorf("modelroute: account %q tokens_per_day must be non-negative", a.ID)
		}
		// A per-minute ceiling above its per-day ceiling is internally contradictory —
		// the minute cap could never be reached without breaching the day cap first, so
		// the pair describes no real budget. A zero per-day field is "no daily cap"
		// (unlimited), under which ANY minute cap is consistent: that is the
		// request-gated-only / tokens-unlimited profile, so it stays exempt.
		if a.RequestsPerDay > 0 && a.RequestsPerMinute > a.RequestsPerDay {
			return fmt.Errorf("modelroute: account %q requests_per_minute (%d) exceeds requests_per_day (%d) — the minute ceiling cannot be higher than the whole-day ceiling",
				a.ID, a.RequestsPerMinute, a.RequestsPerDay)
		}
		if a.TokensPerDay > 0 && a.TokensPerMinute > a.TokensPerDay {
			return fmt.Errorf("modelroute: account %q tokens_per_minute (%d) exceeds tokens_per_day (%d) — the minute ceiling cannot be higher than the whole-day ceiling",
				a.ID, a.TokensPerMinute, a.TokensPerDay)
		}
		if a.CredEnv != "" && !envNameRE.MatchString(a.CredEnv) {
			return fmt.Errorf("modelroute: account %q cred_env %q is not an env-var name "+
				"(it must NAME the variable holding the key, e.g. OPENAI_API_KEY — never the secret itself)", a.ID, a.CredEnv)
		}
		if a.Kind == KindFleet {
			// An ORG-OPERATED server: off-box like a vendor account, but reached on
			// the org's own network, so it carries its own invariants (an explicit
			// non-loopback endpoint) and NOT the vendor credential requirement.
			if err := validateFleetAccount(a); err != nil {
				return err
			}
		} else if remoteKind(a.Kind) {
			if a.CredEnv == "" {
				return fmt.Errorf("modelroute: remote account %q needs a cred_env (the env var NAME holding its key/token)", a.ID)
			}
		} else {
			// Local account: an explicit loopback base_url, never a remote host — a
			// remote base_url under a local kind would emit a "local:" route the
			// residency floor trusts while the bytes egress off-box.
			if a.BaseURL == "" {
				return fmt.Errorf("modelroute: local account %q needs a base_url (no public default for a local server, e.g. http://127.0.0.1:11434/v1)", a.ID)
			}
			if !isLoopbackBaseURL(a.BaseURL) {
				return fmt.Errorf("modelroute: local account %q base_url %q is not a loopback host "+
					"(a local/on-box account must point at localhost/127.0.0.1/::1 — a remote host here would bypass the residency floor)", a.ID, a.BaseURL)
			}
		}
	}
	boundModels := make(map[string]bool, len(r.Bindings))
	for i, b := range r.Bindings {
		if b.Model == "" {
			return fmt.Errorf("modelroute: binding %d has an empty model", i)
		}
		if err := safeRouteToken("binding model", b.Model); err != nil {
			return err
		}
		if b.UpstreamModel == "" && strings.ContainsAny(b.Model, " \t") {
			return fmt.Errorf("modelroute: binding for model %q has no upstream_model and contains whitespace", b.Model)
		}
		if b.UpstreamModel != "" && strings.ContainsAny(b.UpstreamModel, ": \t") {
			return fmt.Errorf("modelroute: binding for model %q has an upstream_model %q containing a route delimiter (: or space)", b.Model, b.UpstreamModel)
		}
		if boundModels[b.Model] {
			return fmt.Errorf("modelroute: duplicate binding for model %q", b.Model)
		}
		boundModels[b.Model] = true
		if !seen[b.Account] {
			return fmt.Errorf("modelroute: binding for model %q names unknown account %q", b.Model, b.Account)
		}
		if err := validateDeprecatedAliasBinding(b); err != nil {
			return err
		}
	}
	if r.Default != "" && !seen[r.Default] {
		return fmt.Errorf("modelroute: default account %q is not a defined account", r.Default)
	}
	// A ManualOnly account as the Default is a contradiction that would defeat the flag
	// entirely: Resolve sends EVERY unbound model id to the Default, so a reserved
	// credential named there would serve the widest automatic path in the roster —
	// silently, and for ids nobody wrote down. Refuse it here rather than let the
	// reservation read as honored while it is being spent.
	if r.Default != "" {
		if a, ok := r.account(r.Default); ok && a.ManualOnly {
			return fmt.Errorf("modelroute: default account %q is manual_only; a reserved account cannot be the fallback for unbound model ids", r.Default)
		}
	}
	if err := validateSpawnClasses(r.SpawnClasses); err != nil {
		return err
	}
	return nil
}

// safeRouteToken rejects a token that would deform the EngineRoute string the
// residency floor parses. The load-bearing rule is the route DELIMITER: an id or
// upstream containing ':', '/', or whitespace could corrupt the route's parse-back. A
// reserved-local word (mock/local/inkernel/cassette) in the ACCOUNT id is harmless and
// allowed: a local account legitimately named "local" is fine, and a REMOTE account
// named "local" still routes remote because EngineRoute always LEADS with the
// validated <kind>: prefix (never the account id), so no id can flip the floor's
// local/remote decision. Locality is the leading prefix, full stop.
func safeRouteToken(what, tok string) error {
	delims := ":/ \t"
	if what == "binding model" {
		delims = ":/\r\n\t"
	}
	if strings.ContainsAny(tok, delims) {
		return fmt.Errorf("modelroute: %s %q contains a route delimiter (:, /, space) — it must be a plain token", what, tok)
	}
	return nil
}

func validateDeprecatedAliasBinding(b Binding) error {
	upstream := b.UpstreamModel
	if upstream == "" {
		upstream = b.Model
	}
	aliasFor, deprecated := deepSeekDeprecatedAliasFor(upstream)
	if !deprecated {
		if b.CompatibilityOnly {
			return fmt.Errorf("modelroute: binding for model %q is marked compatibility_only but upstream %q is not a known deprecated alias", b.Model, upstream)
		}
		return nil
	}
	if !b.CompatibilityOnly {
		return fmt.Errorf("modelroute: binding for model %q uses deprecated DeepSeek alias %q; bind %q/%q instead, or mark compatibility_only with deprecated_after_utc=%q",
			b.Model, upstream, DeepSeekV4ProModel, DeepSeekV4FlashModel, DeepSeekLegacyAliasRetiresUTC)
	}
	if b.DeprecatedAfterUTC != DeepSeekLegacyAliasRetiresUTC {
		return fmt.Errorf("modelroute: compatibility binding for deprecated DeepSeek alias %q must carry deprecated_after_utc=%q", upstream, DeepSeekLegacyAliasRetiresUTC)
	}
	if strings.TrimSpace(b.DeprecatedAliasFor) == "" {
		return fmt.Errorf("modelroute: compatibility binding for deprecated DeepSeek alias %q must name deprecated_alias_for=%q", upstream, aliasFor)
	}
	return nil
}

func deepSeekDeprecatedAliasFor(model string) (string, bool) {
	switch strings.TrimSpace(model) {
	case "deepseek-chat":
		return DeepSeekV4FlashModel + " non-thinking mode", true
	case "deepseek-reasoner":
		return DeepSeekV4FlashModel + " thinking mode", true
	default:
		return "", false
	}
}

// isLoopbackBaseURL reports whether a base URL points at the local box (so a KindLocal
// account cannot smuggle a remote host past the residency floor). It accepts an empty
// host with a unix scheme/socket path, and the loopback hosts localhost / 127.0.0.0-8
// / ::1.
func isLoopbackBaseURL(raw string) bool {
	if strings.HasPrefix(strings.ToLower(raw), "unix:") || strings.Contains(raw, ".sock") {
		return true
	}
	u, err := url.Parse(raw)
	if err != nil {
		return false
	}
	host := strings.ToLower(u.Hostname())
	switch host {
	case "localhost", "127.0.0.1", "0.0.0.0", "::1":
		return true
	}
	return strings.HasPrefix(host, "127.")
}

// ---------------------------------------------------------------------------
// LOAD / DUMP — the JSON roster round-trip (mirrors the routing Manifest).
// ---------------------------------------------------------------------------

// JSON renders the Roster as the canonical indented manifest (stamping the current
// RosterVersion when absent), newline-terminated so `--accounts-dump > file` is clean.
// It carries only env-var NAMES (CredEnv), never a secret value.
func (r Roster) JSON() []byte {
	out := r
	if out.Version == "" {
		out.Version = RosterVersion
	}
	b, _ := json.MarshalIndent(out, "", "  ")
	return append(b, '\n')
}

// ParseRoster decodes and validates a roster. Unknown JSON fields are REJECTED
// (DisallowUnknownFields) so a typo — or a stray "api_key" field someone hoped would
// carry a secret — fails loudly instead of silently changing which account serves a
// model.
func ParseRoster(b []byte) (Roster, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var r Roster
	if err := dec.Decode(&r); err != nil {
		return Roster{}, fmt.Errorf("modelroute: parse roster: %w", err)
	}
	if err := r.Validate(); err != nil {
		return Roster{}, err
	}
	return r, nil
}

// LoadRoster reads and validates a roster from a file path.
func LoadRoster(path string) (Roster, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Roster{}, fmt.Errorf("modelroute: read roster %s: %w", path, err)
	}
	return ParseRoster(b)
}

// DefaultRoster is the illustrative starter `fak route --accounts-dump` emits for a
// user to edit. It shows the switch in three ways: the SAME provider kind under TWO
// accounts (openai-personal vs openai-work — the literal account switch), a local
// on-box server for the cheap aspect (residency-exempt), and a guard ensemble whose
// two members hit DIFFERENT accounts/providers (openai-work + an Anthropic
// subscription). codex is bound to the Responses wire (its native shape), not plain
// openai. Credentials are env-var references, so the file is safe to commit. The model
// ids match the built-in routing DefaultManifest + the example routing manifest.
func DefaultRoster() Roster {
	return Roster{
		Version: RosterVersion,
		Accounts: []Account{
			{ID: "local", Kind: KindLocal, BaseURL: "http://127.0.0.1:11434/v1", Label: "on-box ollama / vLLM (OpenAI-compatible) — no key, residency-exempt"},
			{ID: "openai-personal", Kind: KindOpenAI, CredEnv: "OPENAI_API_KEY", Label: "your personal OpenAI account"},
			{ID: "openai-work", Kind: KindOpenAI, CredEnv: "OPENAI_WORK_API_KEY", Label: "a SECOND OpenAI account — the switch: same kind, different credential"},
			{ID: "codex", Kind: KindOpenAIResponses, CredEnv: "OPENAI_API_KEY", Label: "OpenAI Responses API (codex's native wire)"},
			{ID: "claude-sub", Kind: KindAnthropic, CredEnv: "CLAUDE_CODE_OAUTH_TOKEN", Label: "your Anthropic Pro/Max subscription (sk-ant-oat token; Bearer+oauth-beta scheme applied by the dispatch adapter)"},
			{ID: OpenCodeGoProviderKey, Kind: KindOpenAI, BaseURL: OpenCodeGoOpenAIBaseURL, CredEnv: OpenCodeGoAPIKeyEnv, Label: "OpenCode Go subscription (OpenAI-compatible; credential stays in OPENCODE_GO_API_KEY)"},
			{
				ID:                "july6netra_groq",
				Kind:              KindOpenAI,
				BaseURL:           GroqOpenAIBaseURL,
				CredEnv:           GroqAPIKeyEnv,
				RequestsPerMinute: GroqQwen36RequestsPerMinute,
				RequestsPerDay:    GroqQwen36RequestsPerDay,
				TokensPerMinute:   GroqQwen36TokensPerMinute,
				TokensPerDay:      GroqQwen36TokensPerDay,
				Label:             "Groq OpenAI-compatible account for Alibaba Cloud Qwen3.6 27B; credential stays in FAK_GROQ_API_KEY",
			},
			{
				ID:                "july6netra_groq_compound",
				Kind:              KindOpenAI,
				BaseURL:           GroqOpenAIBaseURL,
				CredEnv:           GroqAPIKeyEnv,
				RequestsPerMinute: GroqCompoundRequestsPerMinute,
				RequestsPerDay:    GroqCompoundRequestsPerDay,
				Label:             "Groq OpenAI-compatible account for groq/compound lower-quality tier; request-count limited, no token cap recorded",
			},
			{ID: "deepseek", Kind: KindDeepSeek, CredEnv: DeepSeekAPIKeyEnv, ContextTokens: DeepSeekV4ContextTokens, MaxOutputTokens: DeepSeekV4MaxOutputTokens, Label: "DeepSeek V4 OpenAI-compatible API: 1M context, 384K max output"},
			{ID: "deepseek-anthropic", Kind: KindAnthropic, BaseURL: DeepSeekAnthropicBaseURL, CredEnv: DeepSeekAPIKeyEnv, ContextTokens: DeepSeekV4ContextTokens, MaxOutputTokens: DeepSeekV4MaxOutputTokens, Label: "DeepSeek V4 Anthropic-compatible API; visible to acceptance, not probed as OpenAI /models"},
			{ID: "openrouter", Kind: KindOpenRouter, CredEnv: OpenRouterAPIKeyEnv, Label: "OpenRouter marketplace (300+ models via OpenAI-compatible wire; credential stays in OPENROUTER_API_KEY)"},
		},
		Default: "openai-personal",
		Bindings: []Binding{
			{Model: "small", Account: "local", UpstreamModel: "llama3.2"},
			{Model: "medium", Account: "openai-personal", UpstreamModel: "gpt-5.5"},
			{Model: "large", Account: "claude-sub", UpstreamModel: "claude-opus-4-6"},
			{Model: "gpt-6-astra", Account: "openai-personal", UpstreamModel: GPT6AstraModel},
			{Model: "gpt-6", Account: "openai-personal", UpstreamModel: GPT6AstraModel},
			{Model: "astra", Account: "openai-personal", UpstreamModel: GPT6AstraModel},
			{Model: "astra-gpt-6", Account: "openai-personal", UpstreamModel: GPT6AstraModel},
			{Model: "astra gpt 6", Account: "openai-personal", UpstreamModel: GPT6AstraModel},
			{Model: "gpt-6 astra", Account: "openai-personal", UpstreamModel: GPT6AstraModel},
			{Model: OpenCodeGoOxAlphaModel, Account: OpenCodeGoProviderKey, UpstreamModel: OpenCodeGoOxAlphaModel},
			{Model: "guard-a", Account: "openai-work", UpstreamModel: "gpt-5.5"},
			{Model: "guard-b", Account: "claude-sub", UpstreamModel: "claude-opus-4-6"},
			{Model: "qwen36-groq", Account: "july6netra_groq", UpstreamModel: GroqQwen36Model},
			{Model: "groq-compound", Account: "july6netra_groq_compound", UpstreamModel: GroqCompoundModel},
			{Model: "deepseek-pro", Account: "deepseek", UpstreamModel: DeepSeekV4ProModel},
			{Model: "deepseek-flash", Account: "deepseek", UpstreamModel: DeepSeekV4FlashModel},
			{Model: "deepseek-pro-anthropic", Account: "deepseek-anthropic", UpstreamModel: DeepSeekV4ProModel},
			{Model: "deepseek-chat-compat", Account: "deepseek", UpstreamModel: "deepseek-chat", CompatibilityOnly: true, DeprecatedAfterUTC: DeepSeekLegacyAliasRetiresUTC, DeprecatedAliasFor: DeepSeekV4FlashModel + " non-thinking mode"},
			{Model: "deepseek-reasoner-compat", Account: "deepseek", UpstreamModel: "deepseek-reasoner", CompatibilityOnly: true, DeprecatedAfterUTC: DeepSeekLegacyAliasRetiresUTC, DeprecatedAliasFor: DeepSeekV4FlashModel + " thinking mode"},
			{Model: "openrouter-free", Account: "openrouter", UpstreamModel: "openrouter/auto"},
			{Model: "openrouter-best", Account: "openrouter", UpstreamModel: "openrouter/best"},
		},
	}
}
