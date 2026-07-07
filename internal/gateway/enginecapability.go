package gateway

// This file is the gateway's WIRE-NEUTRAL read path for an upstream engine's cache
// capability — cache-frontier "next 50" item 32, under epic #1490. It reports what an
// engine can expose about its cache WITHOUT importing any engine-specific adapter
// package: it depends only on the wire-neutral engine.CacheCapability contract and
// reads the value through the engine.CacheCapabilityProducer interface. The
// engine-specific packages (internal/enginecache's SGLang/vLLM control client, and
// the #1551–#1553 per-engine adapters) stay behind that seam.
//
// Import hygiene here is load-bearing and is pinned by a test
// (TestReportEngineCacheCapabilityIsWireNeutral): this file must import
// internal/engine and must NOT import internal/enginecache or any engine-specific
// adapter package. The legacy engine-cache RESET path (newEngineCacheClient in
// gateway.go) still couples to internal/enginecache; this new reporting seam is the
// wire-neutral replacement the External-engines lane builds against.

import "github.com/anthony-chaudhary/fak/internal/engine"

// ReportEngineCacheCapability reports what an upstream engine can expose about its
// cache, read through the engine.CacheCapabilityProducer seam. The producer is the
// ONLY coupling, and it is an interface — so the gateway learns the capability
// without importing the adapter that produced it.
//
// The reported value is trustworthy by construction:
//   - the Verdict is always a member of the closed engine.CacheVerdict vocabulary; an
//     adapter that returns an unrecognized verdict is reported as CacheUnknown, never
//     as a fabricated positive;
//   - an ACTIVE verdict (warm / evict / clone) whose cold path is not witnessed
//     correct (ColdPathCorrect == false) is DEMOTED to CacheUnknown — the gateway
//     will not report active cache behavior it cannot prove is cold-path-safe (the
//     always-cold-correct rule);
//   - a nil producer reports CacheUnknown with kernel provenance (the gateway's own
//     knowledge), never a phantom capability.
func ReportEngineCacheCapability(p engine.CacheCapabilityProducer) engine.CacheCapability {
	if p == nil {
		return engine.CacheCapability{Verdict: engine.CacheUnknown, Provenance: engine.ProvenanceKernel}
	}
	cc := p.CacheCapability()
	// Fail closed to the closed vocabulary: an unrecognized verdict or provenance is
	// reported as the safe default rather than trusted verbatim.
	if !cc.Verdict.Valid() {
		cc.Verdict = engine.CacheUnknown
	}
	if !cc.Provenance.Valid() {
		cc.Provenance = engine.ProvenanceKernel
	}
	// Cold-path correctness stays explicit for any active behavior: an active verdict
	// that has not witnessed its cold path is demoted to unknown, not reported.
	if cc.Verdict.Active() && !cc.ColdPathCorrect {
		cc.Verdict = engine.CacheUnknown
	}
	return cc
}
