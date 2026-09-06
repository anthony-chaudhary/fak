package engine

// This file adds the wire-neutral engine.CacheCapability contract — cache-frontier
// "next 50" item 32 (docs/cache-frontier/DEFAULT-ENABLEMENT-NEXT-50.md), under epic
// #1490 (turn the vCache gates ON by default with honest per-mechanism attribution).
//
// The problem it solves: engine-specific cache knowledge is scattered —
// internal/enginecache hard-codes the SGLang/vLLM control endpoints and the gateway
// imports it directly — so every new engine drags a coupling into core. This
// interface is the SEAM: a small value type plus a producer interface that lets the
// gateway report what an upstream engine can expose about its cache WITHOUT importing
// any engine-specific adapter package. The #1551–#1553 per-engine adapters implement
// the interface behind the seam; core depends on the interface, not on them.

// CacheVerdict is the CLOSED vocabulary of engine cache capabilities — the only
// wire-neutral labels a report may use. It matches the capability-inventory terms
// (item 31 / #1549) so the contract and the inventory never drift, and a new engine
// cannot smuggle in a class of its own. The zero value is the empty string, which is NOT
// a member of the vocabulary (Valid() == false): an unset verdict is fail-closed to
// CacheUnknown by a reporter (see gateway.ReportEngineCacheCapability), never read as
// a fabricated positive. CacheUnknown is the explicit "capability not established"
// label a producer sets on purpose.
type CacheVerdict string

const (
	// CacheUnknown is the zero value: the engine's cache capability is not
	// established. The safe default — never inferred as a positive.
	CacheUnknown CacheVerdict = "unknown"
	// CachePassiveObserve: the engine exposes cache SYMPTOMS the gateway can observe
	// (usage counters, cached-token telemetry) but no control surface to act on.
	CachePassiveObserve CacheVerdict = "passive-observe"
	// CacheActiveWarm: the engine can be actively warmed — prefill a named prefix on
	// demand so a later request hits it.
	CacheActiveWarm CacheVerdict = "active-warm"
	// CacheExactEvict: the engine can evict an EXACT named span, not merely flush the
	// whole prefix/radix cache.
	CacheExactEvict CacheVerdict = "exact-evict"
	// CachePrefixClone: the engine can clone/fork a cached prefix for reuse across
	// sessions.
	CachePrefixClone CacheVerdict = "prefix-clone"
	// CachePagedKV: the engine exposes paged-KV behavior — block-addressable KV the
	// gateway can reason about.
	CachePagedKV CacheVerdict = "paged-kv"
)

// CacheVerdicts returns the closed verdict vocabulary in a stable order. A report or
// conformance test iterates it to prove the enum admits ONLY these members.
func CacheVerdicts() []CacheVerdict {
	return []CacheVerdict{
		CacheUnknown,
		CachePassiveObserve,
		CacheActiveWarm,
		CacheExactEvict,
		CachePrefixClone,
		CachePagedKV,
	}
}

// Valid reports whether v is a member of the closed vocabulary.
func (v CacheVerdict) Valid() bool {
	switch v {
	case CacheUnknown, CachePassiveObserve, CacheActiveWarm,
		CacheExactEvict, CachePrefixClone, CachePagedKV:
		return true
	default:
		return false
	}
}

// Active reports whether the verdict describes ACTIVE cache behavior (warm / exact
// evict / prefix clone) — the class for which cold-path correctness must be witnessed
// (ColdPathCorrect). Passive observation, paged-KV structure, and unknown are not
// active behaviors: they change nothing a request can come to depend on.
func (v CacheVerdict) Active() bool {
	switch v {
	case CacheActiveWarm, CacheExactEvict, CachePrefixClone:
		return true
	default:
		return false
	}
}

// CacheProvenance names the plane whose evidence backs a CacheCapability verdict.
// Epic #1490 requires these stay SEPARATE in any reported value: an OBSERVED provider
// counter is not a WITNESSED kernel event, and a FORECAST is neither. Carrying
// provenance in the contract stops a report from collapsing different trust levels
// into one label.
type CacheProvenance string

const (
	// ProvenanceKernel: a fak in-kernel WITNESSED event backs the verdict. Also the
	// gateway's own default when it reports from its own knowledge.
	ProvenanceKernel CacheProvenance = "kernel"
	// ProvenanceProvider: an OBSERVED provider / engine counter backs the verdict.
	ProvenanceProvider CacheProvenance = "provider"
	// ProvenanceContext: O(1) context / session-memory evidence backs the verdict.
	ProvenanceContext CacheProvenance = "context"
	// ProvenanceForecast: a FORECAST / synthetic estimate backs the verdict — never a
	// measured fact.
	ProvenanceForecast CacheProvenance = "forecast"
)

// CacheProvenances returns the closed provenance vocabulary in a stable order.
func CacheProvenances() []CacheProvenance {
	return []CacheProvenance{
		ProvenanceKernel,
		ProvenanceProvider,
		ProvenanceContext,
		ProvenanceForecast,
	}
}

// Valid reports whether p is a member of the closed provenance vocabulary.
func (p CacheProvenance) Valid() bool {
	switch p {
	case ProvenanceKernel, ProvenanceProvider, ProvenanceContext, ProvenanceForecast:
		return true
	default:
		return false
	}
}

// CacheCapability is the wire-neutral contract for what an upstream inference engine
// can expose about its cache. It is a value type with NO engine-specific fields and
// no imports beyond the package's own stdlib usage: "vllm" / "sglang" / "llama.cpp"
// appear only as an opaque Engine label, never as an imported adapter type. That is
// what lets the gateway report it without importing engine-specific packages into
// core (item 32's whole point).
type CacheCapability struct {
	// Engine is the opaque id of the engine the capability was observed for (e.g.
	// "vllm", "sglang", "llama.cpp"). A label for the report, never an imported type.
	Engine string
	// Verdict is the wire-neutral capability class, a member of the closed
	// CacheVerdict vocabulary. An unrecognized value is not trusted.
	Verdict CacheVerdict
	// Provenance names which plane's evidence backs the verdict, kept separate so a
	// provider rebate counter is never read as a kernel witness (#1490).
	Provenance CacheProvenance
	// Evidence is a short, human-readable anchor for the verdict — a doc reference,
	// control endpoint, or witness id — never the cached bytes.
	Evidence string
	// ColdPathCorrect records that the request's cold path stays correct whether or
	// not this capability's cache behavior fires. It must be true for any ACTIVE
	// verdict (warm / evict / clone); a false value flags an active behavior whose
	// cold-path correctness is not yet witnessed, and a reporter should refuse to
	// trust it (see gateway.ReportEngineCacheCapability).
	ColdPathCorrect bool
}

// Valid reports whether the capability is well-formed: both its verdict and its
// provenance are members of their closed vocabularies. It intentionally does NOT
// fold in ColdPathCorrect — that bit is an explicit provenance signal the reporter
// acts on (fail-closed) rather than a well-formedness error.
func (c CacheCapability) Valid() bool {
	return c.Verdict.Valid() && c.Provenance.Valid()
}

// CacheCapabilityProducer is the seam an engine-specific adapter satisfies to report
// its cache capability. A reader (the gateway) consumes the capability through THIS
// interface, so it never imports the adapter's concrete package — the wire-neutral
// boundary item 32 exists to establish. The #1551–#1553 vLLM / SGLang / llama.cpp
// adapters implement this behind the seam; core imports the interface, not them.
type CacheCapabilityProducer interface {
	// CacheCapability returns the adapter's wire-neutral cache capability.
	CacheCapability() CacheCapability
}
