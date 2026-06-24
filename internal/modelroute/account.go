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

// ---------------------------------------------------------------------------
// PROVIDER KIND — the wire protocol an account speaks (a CLOSED additive set).
// ---------------------------------------------------------------------------

// ProviderKind is the transcript wire an account speaks. It is RE-DECLARED here
// rather than imported from internal/agent ON PURPOSE: modelroute is pure stdlib by
// contract (the property that lets it compose with the frozen ABI seam without
// pulling the agent loop into the routing spine), and internal/agent is not stdlib.
// The five remote kinds mirror agent.Provider 1:1; "local" is the modelroute
// addition (an on-box, OpenAI-compatible server). The named cost of the boundary: a
// new provider must be added in BOTH places. It is a CLOSED set — a new kind is an
// added constant + validation, never manifest free text.
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
	// KindLocal is an on-box, OpenAI-compatible server (ollama / vLLM / llama.cpp /
	// the in-kernel model). It is the ONLY local kind, so locality is exactly
	// Kind == KindLocal — there is no separate flag. A call routed to it is
	// residency-EXEMPT (the bytes never leave the box), which is why Validate forbids
	// it from carrying a non-loopback base URL.
	KindLocal ProviderKind = "local"
)

// knownKind reports whether k is one of the closed ProviderKind set.
func knownKind(k ProviderKind) bool {
	switch k {
	case KindOpenAI, KindOpenAIResponses, KindAnthropic, KindGemini, KindXAI, KindLocal:
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
// test pins this against cmd/fak's guardDefaultBaseURL). KindLocal has no public
// default (returns ""), so a local account MUST set an explicit loopback base_url.
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

// reservedLocalTokens are the route prefixes internal/engine's residency PDP reads as
// ON-BOX (and short-circuits to local). An Account.ID may not begin with one, so a
// remote target's route can never be deformed to read local. (Belt-and-suspenders:
// a remote route already starts with its <kind>: prefix, never the account id.)
var reservedLocalTokens = []string{"mock", "local", "inkernel", "cassette"}

// Account is the switcher unit: a named credential set for one provider. ID is the
// handle a Binding references. Kind is the wire (and the SOLE locality signal). BaseURL
// overrides the kind's public default (REQUIRED, and loopback-only, for a local
// server). CredEnv NAMES the env var holding the API key or subscription token — never
// the secret itself; it is required for a remote account and forbidden-to-be-a-secret
// for all.
type Account struct {
	ID      string       `json:"id"`
	Kind    ProviderKind `json:"kind"`
	BaseURL string       `json:"base_url,omitempty"`
	CredEnv string       `json:"cred_env,omitempty"`
	Label   string       `json:"label,omitempty"`
}

// Binding maps ONE routed model id (a Plan member's Model, or a Plan's Scout) to the
// account that serves it and the UpstreamModel wire name. The routed id is an
// ABSTRACT tier label; UpstreamModel is the provider-specific id sent on the wire
// (empty => the routed id is used verbatim). Distinct members bind independently, so
// an ensemble can span accounts/providers — the "mix and match at any level" the goal
// asks for.
type Binding struct {
	Model         string `json:"model"`
	Account       string `json:"account"`
	UpstreamModel string `json:"upstream_model,omitempty"`
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
}

// Target is the resolved dispatch destination for one routed model id: which account
// serves it, the provider kind + concrete base URL + credential env var NAME, and the
// upstream wire model name. It is a VALUE — the dispatch wiring turns it into an
// agent.HTTPPlanner; this package never does I/O. CredEnv is a NAME, never the secret.
type Target struct {
	Model         string       `json:"model"`          // the routed id (the Plan member / scout)
	Account       string       `json:"account"`        // the resolved Account.ID
	Kind          ProviderKind `json:"kind"`           // the provider wire
	BaseURL       string       `json:"base_url"`       // concrete (account override or kind default)
	CredEnv       string       `json:"cred_env"`       // env var NAME for the credential ("" = local)
	UpstreamModel string       `json:"upstream_model"` // the wire model name
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
// gemini / xai) — so the floor's local/remote decision is the account's DECLARED kind,
// never a guess from whether the model name contains "openai". The account/upstream
// follow for legibility and so a downstream dispatcher can recover the binding. The
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
		Model:         modelID,
		Account:       a.ID,
		Kind:          a.Kind,
		BaseURL:       baseURL,
		CredEnv:       a.CredEnv,
		UpstreamModel: upstream,
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
//   - each binding: a non-empty, delimiter-free model bound to a real account, unique
//     per model id, with a delimiter-free upstream;
//   - a Default (when set) naming a real account.
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
		if a.CredEnv != "" && !envNameRE.MatchString(a.CredEnv) {
			return fmt.Errorf("modelroute: account %q cred_env %q is not an env-var name "+
				"(it must NAME the variable holding the key, e.g. OPENAI_API_KEY — never the secret itself)", a.ID, a.CredEnv)
		}
		if remoteKind(a.Kind) {
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
		if b.UpstreamModel != "" && strings.ContainsAny(b.UpstreamModel, ":/ \t") {
			return fmt.Errorf("modelroute: binding for model %q has an upstream_model %q containing a route delimiter (:, /, space)", b.Model, b.UpstreamModel)
		}
		if boundModels[b.Model] {
			return fmt.Errorf("modelroute: duplicate binding for model %q", b.Model)
		}
		boundModels[b.Model] = true
		if !seen[b.Account] {
			return fmt.Errorf("modelroute: binding for model %q names unknown account %q", b.Model, b.Account)
		}
	}
	if r.Default != "" && !seen[r.Default] {
		return fmt.Errorf("modelroute: default account %q is not a defined account", r.Default)
	}
	return nil
}

// safeRouteToken rejects a token that would deform the EngineRoute string the
// residency floor parses: a route delimiter (':', '/', whitespace) or a leading
// reserved-local token (which could read as on-box if it ever led the route).
func safeRouteToken(what, tok string) error {
	if strings.ContainsAny(tok, ":/ \t") {
		return fmt.Errorf("modelroute: %s %q contains a route delimiter (:, /, space) — it must be a plain token", what, tok)
	}
	low := strings.ToLower(tok)
	for _, r := range reservedLocalTokens {
		if low == r || strings.HasPrefix(low, r+"-") || strings.HasPrefix(low, r+":") {
			return fmt.Errorf("modelroute: %s %q begins with the reserved local token %q (it could read as an on-box route)", what, tok, r)
		}
	}
	return nil
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
		},
		Default: "openai-personal",
		Bindings: []Binding{
			{Model: "small", Account: "local", UpstreamModel: "llama3.2"},
			{Model: "medium", Account: "openai-personal", UpstreamModel: "gpt-5.5"},
			{Model: "large", Account: "claude-sub", UpstreamModel: "claude-opus-4-6"},
			{Model: "guard-a", Account: "openai-work", UpstreamModel: "gpt-5.5"},
			{Model: "guard-b", Account: "claude-sub", UpstreamModel: "claude-opus-4-6"},
		},
	}
}
