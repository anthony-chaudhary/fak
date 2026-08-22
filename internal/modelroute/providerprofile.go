package modelroute

// ---------------------------------------------------------------------------
// THE PROVIDER PROFILE — a declarative provider descriptor (auth / endpoint /
// cache quirks), the fak analogue of Hermes' ProviderProfile dataclass (#2861,
// part of #2834 Track G — provider + config discipline).
// ---------------------------------------------------------------------------
//
// THE PATTERN WE STEAL. Hermes ships each provider as a DATA-ONLY plugin: a
// ProviderProfile (auth, endpoints, quirks) that the runtime reads while the
// kernel owns clients / rotation / streaming. Adding a provider is adding a
// profile, not writing an adapter. This file gives fak the same shape: a
// ProviderProfile schema + a ProviderRegistry keyed by provider, so provider
// coverage grows as DATA, not code.
//
// WHAT fak ADDS OVER HERMES. Hermes' profiles feed a client factory only. fak's
// profiles ALSO carry the cache-economics quirks the rest of the kernel already
// reasons about — does this provider support prompt caching, and what is its
// cache TTL — so the SAME declarative entry that adds a provider also feeds
// fak's cache accounting. Hermes structurally cannot do this: it has no
// kernel-side cache-value surface for a profile to feed. The wire is ProfileFor:
// a routing decision resolves an alias to a ResolvedModel carrying a Provider
// (registry.go), and ProfileFor maps that Provider to its declarative cache
// quirks — so the routing OUTPUT reads the profile with no per-provider code.
//
// PURITY (lane rule). Like the rest of this package the profile layer is pure
// and stdlib-only: data-in / decision-out, no I/O beyond the JSON round-trip, no
// client construction here. Owning the live client/rotation/streaming is the
// wiring layer's job (a separate lane); this file is only the declarative
// SCHEMA + REGISTRY those layers read. That is the deliberate seam — the schema
// lands now (gen/next foundation); the live client wiring consumes it later.
//
// NOT A PLACEHOLDER. The cache quirks are REAL per-provider values, not one
// default silently stamped on every provider: Anthropic declares a 5-minute
// ephemeral TTL with a 4-breakpoint cap; Gemini a 1-hour operator-set TTL;
// OpenAI and xAI provider-managed automatic caching (TTL 0 == no operator knob,
// a meaningful state distinct from "caching unsupported"). TestProviderCache
// QuirksDiffer pins that they are not uniform.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"time"
)

// ProviderProfile is the declarative descriptor of ONE model provider: how to
// authenticate, where to reach it, and the cache quirks fak's cost/cache-value
// surfaces account for. It is data-only (JSON-tagged, no methods that do I/O) so
// a deployment adds a provider by declaring a profile — the Hermes "data-only
// plugin" shape — never by writing an adapter.
type ProviderProfile struct {
	// Provider is the registry key: the canonical provider id a ResolvedModel
	// carries in its Provider field (`openai`, `anthropic`, `gemini`, `xai`).
	Provider string `json:"provider"`
	// AuthEnv names the environment variable the kernel's client reads the API
	// key from. Declaring auth as an env-var NAME (not the secret) keeps the
	// profile a reviewable, checked-in data file with no secret in it.
	AuthEnv string `json:"auth_env,omitempty"`
	// Endpoint is the provider's base URL. "" leaves the client's built-in
	// default (an in-kernel/local provider may name none).
	Endpoint string `json:"endpoint,omitempty"`

	// --- cache-economics quirks (the half Hermes has no surface for) ---

	// SupportsPromptCaching reports whether the provider serves prompt-cache
	// hits at all. The cache-value surface must not credit a cache saving to a
	// provider that cannot cache; this bit gates that.
	SupportsPromptCaching bool `json:"supports_prompt_caching"`
	// CacheTTLSeconds is the provider's prompt-cache time-to-live in seconds for
	// an OPERATOR-SET TTL (Anthropic 5-minute ephemeral == 300; Gemini default
	// 1-hour == 3600). 0 means the provider manages the TTL automatically with no
	// operator knob (OpenAI, xAI) — distinct from "caching unsupported", which
	// SupportsPromptCaching carries. Read CacheTTL() for a time.Duration.
	CacheTTLSeconds int `json:"cache_ttl_seconds,omitempty"`
	// MaxCacheBreakpoints is the most explicit cache breakpoints the provider
	// honors (Anthropic caps at 4). 0 means automatic caching with no explicit
	// breakpoint control (OpenAI, Gemini, xAI).
	MaxCacheBreakpoints int `json:"max_cache_breakpoints,omitempty"`
}

// CacheTTL returns the provider's operator-set prompt-cache TTL as a Duration
// (0 for a provider-managed/automatic TTL). It is the convenience the cache-value
// surface reads instead of re-deriving seconds -> Duration everywhere.
func (p ProviderProfile) CacheTTL() time.Duration {
	return time.Duration(p.CacheTTLSeconds) * time.Second
}

// ProviderRegistry is the provider-keyed profile map an operator configures
// alongside a routing Manifest: it turns a ResolvedModel's Provider into its
// declarative auth/endpoint/cache-quirk profile. Data-in / decision-out and
// pure — the model-routing analogue of the alias Registry in registry.go, but
// keyed by PROVIDER rather than by model alias.
type ProviderRegistry struct {
	profiles map[string]ProviderProfile
}

// NewProviderRegistry builds a registry from a set of profiles, keyed by
// Provider. It fails LOUD (mirroring NewRegistry) on a misconfiguration a silent
// default would hide: an empty provider id, a duplicate provider, a negative TTL
// or breakpoint count, or cache quirks declared on a provider that does not
// support prompt caching (a TTL/breakpoint on a non-caching provider is a config
// mistake the cache-value surface must never silently honor — the confusion risk
// the issue calls out).
func NewProviderRegistry(profiles []ProviderProfile) (*ProviderRegistry, error) {
	r := &ProviderRegistry{profiles: make(map[string]ProviderProfile, len(profiles))}
	for i, p := range profiles {
		if p.Provider == "" {
			return nil, fmt.Errorf("modelroute: provider profile %d has an empty provider id", i)
		}
		if _, dup := r.profiles[p.Provider]; dup {
			return nil, fmt.Errorf("modelroute: duplicate provider profile %q", p.Provider)
		}
		if p.CacheTTLSeconds < 0 {
			return nil, fmt.Errorf("modelroute: provider %q has a negative cache_ttl_seconds", p.Provider)
		}
		if p.MaxCacheBreakpoints < 0 {
			return nil, fmt.Errorf("modelroute: provider %q has a negative max_cache_breakpoints", p.Provider)
		}
		if !p.SupportsPromptCaching && (p.CacheTTLSeconds > 0 || p.MaxCacheBreakpoints > 0) {
			return nil, fmt.Errorf("modelroute: provider %q declares cache quirks but not prompt caching", p.Provider)
		}
		r.profiles[p.Provider] = p
	}
	return r, nil
}

// Providers returns the registered provider ids in sorted order (determinism
// helper, mirrors Registry.Aliases).
func (r *ProviderRegistry) Providers() []string {
	out := make([]string, 0, len(r.profiles))
	for p := range r.profiles {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

// Lookup returns the ProviderProfile for a provider id, or false if the provider
// has no declared profile.
func (r *ProviderRegistry) Lookup(provider string) (ProviderProfile, bool) {
	p, ok := r.profiles[provider]
	return p, ok
}

// ProfileFor is THE WIRE from the routing surface to the profile: given a
// ResolvedModel (the output of the alias Registry's Resolve), it returns the
// declarative profile of the model's serving Provider. A LOCAL/in-kernel model
// (empty Provider) has no profile and returns false. This is how a routing
// decision's cache quirks are read with no per-provider code — the caller reads
// m := reg.Resolve(...); prof, ok := providers.ProfileFor(m); prof.CacheTTL().
func (r *ProviderRegistry) ProfileFor(m ResolvedModel) (ProviderProfile, bool) {
	if m.Provider == "" {
		return ProviderProfile{}, false
	}
	return r.Lookup(m.Provider)
}

// ParseProviderProfiles decodes (and builds a validated registry from) a JSON
// array of ProviderProfile entries — the "add a provider by editing a data file"
// path. Unknown JSON fields are REJECTED (DisallowUnknownFields) so a typo fails
// loud instead of silently changing the provider surface, matching ParseManifest.
func ParseProviderProfiles(b []byte) (*ProviderRegistry, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var profiles []ProviderProfile
	if err := dec.Decode(&profiles); err != nil {
		return nil, fmt.Errorf("modelroute: parse provider profiles: %w", err)
	}
	return NewProviderRegistry(profiles)
}

// JSON renders the registry's profiles as a canonical indented JSON array
// (sorted by provider for a stable diff), terminated by a newline. It round-trips
// with ParseProviderProfiles so `--dump > file` then edit-and-reload is clean.
func (r *ProviderRegistry) JSON() []byte {
	out := make([]ProviderProfile, 0, len(r.profiles))
	for _, p := range r.Providers() {
		out = append(out, r.profiles[p])
	}
	b, _ := json.MarshalIndent(out, "", "  ")
	return append(b, '\n')
}

// DefaultProviderProfiles is the built-in seed set: declarative profiles for the
// remote providers fak already names in its model registry (openai, anthropic,
// gemini, xai — see registry.go's ResolvedModel.Provider examples). It is the
// starting data an operator edits or extends — adding the ~30th provider is
// appending one more entry here (or in a --profiles file), never new Go code.
//
// The cache quirks are REAL, per-provider public values, deliberately NOT
// uniform (see the file header): a placeholder default stamped on every provider
// would silently mis-credit cache savings, which the issue flags as a risk.
func DefaultProviderProfiles() []ProviderProfile {
	contracts := DefaultProviderContracts()
	profiles := make([]ProviderProfile, 0, len(contracts)+2)
	for _, contract := range contracts {
		profile, err := providerProfileFromContract(contract)
		if err != nil {
			panic(err)
		}
		profiles = append(profiles, profile)
	}
	// Gemini and xAI remain legacy projections until their canonical contracts land.
	profiles = append(profiles,
		ProviderProfile{Provider: "gemini", AuthEnv: "GEMINI_API_KEY", Endpoint: "https://generativelanguage.googleapis.com", SupportsPromptCaching: true, CacheTTLSeconds: 3600},
		ProviderProfile{Provider: "xai", AuthEnv: "XAI_API_KEY", Endpoint: "https://api.x.ai/v1", SupportsPromptCaching: true},
	)
	return profiles
}
