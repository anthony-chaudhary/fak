package engine

// This file adds the FAIL-CLOSED admission guard for ACTIVE cache operations —
// cache-frontier "next 50" item 49 (docs/cache-frontier/DEFAULT-ENABLEMENT-NEXT-50.md,
// "Remove legacy" lane), issue #1567, under epic #1490 (turn the vCache gates ON by
// default with honest per-mechanism attribution).
//
// The problem it solves: engine.CacheCapability (item 32 / #1550) tells a caller WHAT an
// engine can expose about its cache, and gateway.ReportEngineCacheCapability already
// demotes an untrustworthy capability to CacheUnknown. But that demotion is SILENT — all
// of its causes (a nil producer, an out-of-vocabulary verdict, an active verdict whose
// cold path was never witnessed) collapse into the same bare "unknown", and nothing stops
// a caller from then acting on an unknown/unsupported capability as though the active
// primitive were available. An unknown capability that flows into an active cache path
// unchallenged IS the optimistic claim epic #1490 exists to remove.
//
// AdmitActiveCache is the single admission point: an active cache operation (warm /
// exact-evict / prefix-clone) is admitted ONLY against a capability that is well-formed,
// established, supports that exact operation, has a witnessed cold path, and is backed by
// a MEASURED plane. Every other outcome returns a NAMED reason from the closed
// CacheRefusal vocabulary — never an optimistic admit, and never a panic.
//
// Fail closed means: name the reason and let the caller take the correct COLD path (send
// the full required context). It does not mean crash, and it does not mean "cache hit".
//
// Why a new vocabulary rather than cachemeta.LookupReason: that vocabulary names why a
// cache LOOKUP missed (absent / cold / stale / residency_fault …). None of its members
// names an engine CAPABILITY refusal. Importing cachemeta here would also break the
// wire-neutrality this contract exists to establish — cachecapability.go deliberately
// carries no imports. So the capability vocabulary is extended in place, additively; the
// existing cachemeta.Reason* consumers are undisturbed.

// CacheRefusal is the CLOSED vocabulary of named reasons an active cache operation was
// refused. The zero value (RefusalNone) is the ONLY admitted value — so a caller that
// forgets to check, or a future member added to this enum, fails closed by construction
// rather than reading as an admit.
type CacheRefusal string

const (
	// RefusalNone is the zero value: the active cache operation was ADMITTED. It is the
	// only non-refusal member of the vocabulary.
	RefusalNone CacheRefusal = ""
	// RefusalOperationNotActive: the requested operation is not an ACTIVE cache verdict
	// (warm / exact-evict / prefix-clone). Passive observation, paged-KV structure and
	// unknown are not operations that can be admitted here — this names caller misuse
	// rather than an engine shortcoming.
	RefusalOperationNotActive CacheRefusal = "operation-not-active"
	// RefusalCapabilityMalformed: the capability is not well-formed — its verdict or its
	// provenance falls outside the closed vocabulary, so nothing about it is trusted.
	RefusalCapabilityMalformed CacheRefusal = "capability-malformed"
	// RefusalCapabilityUnknown: the engine/provider capability is not established
	// (CacheUnknown). This is the issue's headline refusal — an UNKNOWN capability must
	// never fall through to an optimistic claim that the active primitive is available.
	RefusalCapabilityUnknown CacheRefusal = "capability-unknown"
	// RefusalOperationUnsupported: the capability is established, but for a DIFFERENT
	// class than the requested operation (e.g. an exact-evict engine asked to
	// active-warm). The active path is unsupported on this engine.
	RefusalOperationUnsupported CacheRefusal = "operation-unsupported"
	// RefusalColdPathUnwitnessed: the capability claims an active behavior whose cold
	// path has not been witnessed correct (ColdPathCorrect == false). Cold-path
	// correctness stays explicit for active behavior; an unwitnessed one is refused.
	RefusalColdPathUnwitnessed CacheRefusal = "cold-path-unwitnessed"
	// RefusalForecastProvenance: the capability is backed only by a FORECAST / synthetic
	// estimate, which is never a measured fact. Acting on it would be exactly the
	// optimistic claim #1490 removes, so an active operation demands a measured plane
	// (kernel / provider / context). Provenance stays separate from the verdict: this
	// refuses the ACT, it does not rewrite the reported capability.
	RefusalForecastProvenance CacheRefusal = "forecast-provenance"
)

// CacheRefusals returns the closed refusal vocabulary in a stable order. A report or
// conformance test iterates it to prove the enum admits ONLY these members.
func CacheRefusals() []CacheRefusal {
	return []CacheRefusal{
		RefusalNone,
		RefusalOperationNotActive,
		RefusalCapabilityMalformed,
		RefusalCapabilityUnknown,
		RefusalOperationUnsupported,
		RefusalColdPathUnwitnessed,
		RefusalForecastProvenance,
	}
}

// Valid reports whether r is a member of the closed refusal vocabulary.
func (r CacheRefusal) Valid() bool {
	switch r {
	case RefusalNone, RefusalOperationNotActive, RefusalCapabilityMalformed,
		RefusalCapabilityUnknown, RefusalOperationUnsupported,
		RefusalColdPathUnwitnessed, RefusalForecastProvenance:
		return true
	default:
		return false
	}
}

// Refused reports whether r names a refusal (i.e. the operation was NOT admitted).
// Anything other than RefusalNone is a refusal, so an unrecognized value read off a wire
// is treated as refused rather than admitted.
func (r CacheRefusal) Refused() bool { return r != RefusalNone }

// AdmitActiveCache is the single fail-closed admission point for an ACTIVE cache
// operation (op) against an engine's reported capability (cc).
//
// It returns RefusalNone ONLY when every honesty condition holds:
//   - op is an ACTIVE verdict (warm / exact-evict / prefix-clone);
//   - cc is well-formed (verdict and provenance both in their closed vocabularies);
//   - cc's capability is established (not CacheUnknown);
//   - cc's capability is for exactly the requested operation — a CacheCapability carries
//     a single verdict, so an engine that supports two active classes reports two
//     capabilities;
//   - cc's cold path is witnessed correct;
//   - cc is backed by a MEASURED plane, not a forecast.
//
// Otherwise it returns a NAMED reason from the closed CacheRefusal vocabulary. It never
// panics and never returns an optimistic admit for an unknown or unsupported capability:
// the caller's correct response to any refusal is the COLD path (send the full required
// context), which stays correct whether or not any cache behavior fires.
func AdmitActiveCache(cc CacheCapability, op CacheVerdict) CacheRefusal {
	// The guard admits only ACTIVE operations; asking it to admit a passive/unknown
	// "operation" is caller misuse, named as such rather than silently admitted.
	if !op.Active() {
		return RefusalOperationNotActive
	}
	// An ill-formed capability is trusted for nothing at all.
	if !cc.Valid() {
		return RefusalCapabilityMalformed
	}
	// The headline: an unestablished capability never implies the primitive exists.
	if cc.Verdict == CacheUnknown {
		return RefusalCapabilityUnknown
	}
	// Established, but for some other class than the one requested.
	if cc.Verdict != op {
		return RefusalOperationUnsupported
	}
	// Active behavior without a witnessed cold path is refused, not reported.
	if !cc.ColdPathCorrect {
		return RefusalColdPathUnwitnessed
	}
	// A forecast is never a measured fact; an active act demands a measured plane.
	if cc.Provenance == ProvenanceForecast {
		return RefusalForecastProvenance
	}
	return RefusalNone
}
