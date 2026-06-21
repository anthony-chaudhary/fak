package cachemeta

import (
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ProviderCache is the field-only shape a provider-usage adapter lowers into. A
// provider prompt-cache event is COST/LATENCY telemetry about a prefix the remote
// engine kept resident — it is never authority that a local result may be re-served
// (refusal rule 6: "provider prompt-cache hit treated as trust evidence"). This
// record exists so fleet benchmarks cannot accidentally count provider savings as
// local wins; the cachemeta plane for it is plane=provider, residency=provider.
type ProviderCache struct {
	Provider       string // "openai" | "anthropic" | "gemini" | "bedrock"
	ModelID        string
	CachedTokens   int64  // cache-read tokens reported by the provider
	WriteTokens    int64  // cache-write/create tokens
	PromptTokens   int64  // total prompt tokens the prefix covers
	SerializerID   string // prompt serializer hash (deterministic serialization)
	BreakpointMode string // "auto" | "explicit" | "implicit" | ""
	Retention      string // "5m" | "1h" | ttl/retention mode where known
	FirstDivergeAt int64  // offline first-divergence token offset (<=0 = unknown)
	Owner          string

	// Endpoint and ReasoningMode are GLM-5.2 (Z.AI) Vary axes per
	// GLM52-HOSTED-CACHE-COHERENCE-2026-06-19.md §A2: the Coding-Plan vs general
	// endpoint and the reasoning_effort/thinking toggle are SILENT cache-breakers.
	// Folding them into the entry identity records a mode/endpoint switch as a
	// distinct provider-prefix cache-write rather than an invisible miss, so a
	// metrics sink does not blend two different request shapes into one hit rate.
	// They are additive and provider-agnostic (empty = no axis contributed).
	Endpoint      string // "general" | "coding" | upstream endpoint label
	ReasoningMode string // "max" | "enabled" | "disabled" | reasoning_effort/thinking label
}

// FromProviderCache folds provider prompt-cache telemetry into a cachemeta entry.
// The entry lives on the provider plane and marks residency=provider; its metrics
// carry cached/write tokens so a metrics sink can separate provider savings from
// local hits. It is intentionally NOT admission-Allow for re-serving: a provider
// residency is observational, and ProviderCacheVerdict makes the non-serveability
// mechanical.
func FromProviderCache(p ProviderCache, opts ...Option) Entry {
	owner := p.Owner
	if owner == "" {
		owner = firstNonEmpty(p.Provider, "provider")
	}
	// Endpoint and ReasoningMode join the identity (§A2): switching the Z.AI
	// Coding-Plan vs general endpoint, or the reasoning/thinking mode, silently
	// breaks the provider's implicit cache, so they must shape a distinct digest.
	digest := DigestBytes([]byte(p.Provider + "\x00" + p.ModelID + "\x00" + p.SerializerID +
		"\x00" + p.Endpoint + "\x00" + p.ReasoningMode))
	length := p.PromptTokens
	if length == 0 {
		length = p.CachedTokens
	}
	e := Entry{
		ID: EntryID{
			Digest:    digest,
			MediaType: MediaPromptPrefix,
			Length:    length,
			Unit:      UnitTokens,
		},
		Plane: PlaneProvider,
		Derivation: Derivation{
			Producer:     owner,
			ModelID:      p.ModelID,
			SerializerID: p.SerializerID,
		},
		Validity: Validity{
			TTLMillis: providerTTLMillis(p.Retention),
		},
		Security: Security{
			// A provider residency is fleet-visible cost telemetry, never a
			// trust verdict: defer admission so no consumer mistakes a remote
			// prefix cache for a locally-admitted, re-serveable result.
			Taint:            abi.TaintTrusted,
			Scope:            abi.ScopeFleet,
			AdmissionVerdict: AdmissionDefer,
			AdmittedBy:       owner,
			Reason:           "provider_cache_telemetry",
		},
		Residency: Residency{Tier: TierProvider, Owner: owner},
		Coherence: Coherence{InvalidationMode: InvalidationPolicy},
		Metrics: Metrics{
			Hits:               0,
			PrefillTokensSaved: p.CachedTokens,
			BytesTransferred:   0,
		},
		Labels: providerLabels(p),
	}
	apply(&e, opts)
	return e
}

// ProviderCacheVerdict is the typed answer to "may this provider cache entry be
// re-served as a local hit?": never. It returns a Transform verdict (non-serveable
// by CanServe) whose Meta marks the entry as cost/latency evidence only, so the
// refusal rule is enforced in code rather than relied upon in prose.
func ProviderCacheVerdict(e Entry) LookupVerdict {
	return LookupVerdict{
		Kind:   LookupTransform,
		Reason: ReasonPolicyMismatch,
		Entry:  e,
		Handle: e.ID,
		Meta:   map[string]string{"provider_cache": "cost_latency_only"},
	}
}

func providerLabels(p ProviderCache) map[string]string {
	m := map[string]string{
		"provider": p.Provider,
		"plane":    string(PlaneProvider),
	}
	if p.BreakpointMode != "" {
		m["breakpoint_mode"] = p.BreakpointMode
	}
	if p.Endpoint != "" {
		m["endpoint"] = p.Endpoint
	}
	if p.ReasoningMode != "" {
		m["reasoning_mode"] = p.ReasoningMode
	}
	if p.Retention != "" {
		m["retention"] = p.Retention
	}
	if p.WriteTokens != 0 {
		m["cache_write_tokens"] = strconv.FormatInt(p.WriteTokens, 10)
	}
	if p.FirstDivergeAt > 0 {
		m["first_diverge_at"] = strconv.FormatInt(p.FirstDivergeAt, 10)
	}
	return m
}

// providerTTLMillis maps a provider retention hint to an advisory TTL so a metrics
// sink can age out stale provider telemetry without inventing precision.
func providerTTLMillis(retention string) int64 {
	switch retention {
	case "5m", "5min":
		return 5 * 60 * 1000
	case "1h", "60m":
		return 60 * 60 * 1000
	default:
		return 0
	}
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
