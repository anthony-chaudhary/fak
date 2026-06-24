package contextq

import (
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// skillmemory.go (#513) — a PROCEDURAL-memory view over a skill invocation.
//
// The snippet/summary/kv views this package emits are projections of an immutable
// recorded CDB page: their identity is a source page step, and the ViewCache keys
// them by (step, view type, producer). A skill invocation has no recorded source
// page — it is a procedure (a skill at a version, run against a digested set of
// inputs) whose reusable artifact is the context it assembled. So a procedural-
// memory view's identity is its INVOCATION DIGEST, not a page step: two runs of
// the same skill+version over the same digested inputs must resolve to the same
// view and be served as a HIT, paging/re-rendering nothing.
//
// This file does NOT reimplement the ViewCache (#558 shipped it at contextq.go).
// It binds the existing cache to invocation identity. The cache keys on
// (step, view type, producer); a procedural view carries no source page (the
// sentinel step skillProcedureStep, which firstSourcePage already returns for an
// empty SourcePageIDs), a constant view type (ViewProcedure), and an EXACT
// composite producer key that embeds the invocation digest. Identity is the exact
// digest bytes in the map key — never a hash that could alias — so the package's
// fail-closed rule (never serve one invocation's procedural memory as another's)
// holds by construction.

// ViewProcedure is the procedural-memory view type: the rendered, reusable surface
// of a skill invocation's assembled context. Unlike the read-only views over
// recorded pages, it is keyed by the skill's invocation digest, so a re-invocation
// with an identical digest serves the prior view as a HIT.
const ViewProcedure ViewType = "procedure"

// skillProcedureStep is the sentinel source-page step for a procedural-memory view.
// A skill invocation has no recorded CDB page, so SourcePageIDs is left empty and
// firstSourcePage returns this value; ViewCache.Put and the Resolve Get must agree
// on it for the key to round-trip.
const skillProcedureStep = -1

// SkillContextRecord is the procedural-memory descriptor for one skill invocation.
// It names the skill and its version, carries the invocation digest that
// identifies an identical re-invocation, the producer, the share scope, and the
// lowered cache entry for the assembled context. Its procedural-memory view is
// stored in (and served from) the shared ViewCache keyed by the invocation digest,
// so a re-invocation with the same digest is a HIT that re-renders nothing.
type SkillContextRecord struct {
	SkillName        string          `json:"skill_name"`
	Version          string          `json:"version"`
	InvocationDigest string          `json:"invocation_digest"`
	Producer         string          `json:"producer"`
	Scope            abi.ShareScope  `json:"scope"`
	CacheEntry       cachemeta.Entry `json:"cache_entry"`
}

// SkillProcedureResult is the outcome of resolving a skill's procedural-memory view
// against the shared ViewCache. Verdict speaks the package's closed
// MaterializationKind vocabulary (HIT on a warm re-invocation, FAULT on a cold
// build), View is the served/built record, and Payload is the rendered procedural-
// memory body. Built reports whether the cold-build closure ran — false on a HIT,
// the economic proof that an identical re-invocation redid no work.
type SkillProcedureResult struct {
	Verdict MaterializationVerdict `json:"verdict"`
	View    MemoryViewRecord       `json:"view"`
	Payload []byte                 `json:"-"`
	Built   bool                   `json:"built"`
}

// producerOrDefault is the human-facing producer (who assembled the context),
// defaulting when unset. It is kept distinct from the composite cache key.
func (r SkillContextRecord) producerOrDefault() string {
	if r.Producer == "" {
		return "skill-context"
	}
	return r.Producer
}

// viewCacheProducer composes the ViewCache producer-key for this record. The shared
// cache keys on (step, view type, producer); a procedural-memory view has no source
// page (a constant sentinel step) and a constant view type, so the invocation
// identity must live in the producer component. Folding producer, skill name,
// version, and the full invocation digest into it makes the key EXACT: two records
// that share all four collide on the same cache slot (a HIT), and any difference —
// a new skill version, a different producer, a changed digest — is a distinct slot
// (a cold build). The separators are unit-separator bytes so a value containing a
// delimiter char cannot forge a collision across fields.
func (r SkillContextRecord) viewCacheProducer() string {
	const us = "\x1f"
	return r.producerOrDefault() + us + "skill=" + r.SkillName + us + "version=" + r.Version + us + "inv=" + r.InvocationDigest
}

// viewID is a stable, legible id for the procedural-memory view.
func (r SkillContextRecord) viewID() string {
	return "view-skill-" + r.SkillName + "-" + short(r.InvocationDigest)
}

// view lowers the record into a procedural-memory MemoryViewRecord ready for the
// ViewCache. The record's Producer field carries the composite cache key (so
// ViewCache.Put keys it by the invocation digest); the human-facing producer, the
// skill, version, and digest are preserved in Labels for legibility. When no
// CacheEntry was supplied the view synthesizes a well-formed memory-view entry
// whose identity digest is the invocation digest, so a stored procedural view
// always lowers into valid memory-view cache metadata.
func (r SkillContextRecord) view(payloadLen int64) MemoryViewRecord {
	entry := r.CacheEntry
	if entry.ID.Digest == "" {
		mv := cachemeta.MemoryView{
			ViewID:            r.viewID(),
			ViewType:          string(ViewProcedure),
			Digest:            r.InvocationDigest,
			Length:            payloadLen,
			Producer:          r.producerOrDefault(),
			Scope:             r.Scope,
			Coverage:          1.0,
			FaithfulnessProbe: 1.0,
		}
		entry = cachemeta.FromMemoryView(mv)
	}
	return MemoryViewRecord{
		ViewID:            r.viewID(),
		ViewType:          ViewProcedure,
		Producer:          r.viewCacheProducer(),
		SourceLen:         payloadLen,
		Scope:             r.Scope,
		Taint:             entry.Security.Taint,
		Coverage:          1.0,
		FaithfulnessProbe: 1.0,
		CacheEntry:        entry,
		Labels: map[string]string{
			"skill":             r.SkillName,
			"version":           r.Version,
			"invocation_digest": r.InvocationDigest,
			"producer":          r.producerOrDefault(),
			"step":              strconv.Itoa(skillProcedureStep),
		},
	}
}

// Resolve is the procedural-memory hot path. It consults the shared ViewCache for a
// procedural-memory view keyed by this record's invocation digest.
//
//   - HIT (warm): a prior invocation with an identical (producer, skill, version,
//     digest) already cached its view. It is served as a HIT, build is NEVER called,
//     and nothing is re-rendered — the economic point of the view-as-cache-artifact.
//   - FAULT (cold): no view exists for this invocation digest yet. The build closure
//     renders the procedural-memory body once, the view is stored under the digest
//     key, and the NEXT identical invocation becomes a HIT.
//
// A nil cache forces the cold path on every call (build runs, nothing is stored); a
// nil build closure yields an empty body.
func (r SkillContextRecord) Resolve(cache *ViewCache, build func() []byte) SkillProcedureResult {
	producerKey := r.viewCacheProducer()

	if cache != nil {
		if cached, payload, ok := cache.Get(skillProcedureStep, ViewProcedure, producerKey); ok {
			return SkillProcedureResult{
				Verdict: MaterializationVerdict{
					Kind:   MaterializationHit,
					Reason: "skill_procedure_cache_hit",
					ViewID: cached.ViewID,
					Entry:  cached.CacheEntry.ID,
				},
				View:    cached,
				Payload: payload,
				Built:   false,
			}
		}
	}

	var payload []byte
	if build != nil {
		payload = build()
	}
	view := r.view(int64(len(payload)))
	cache.Put(view, payload) // nil-safe: a nil cache simply skips storage
	return SkillProcedureResult{
		Verdict: MaterializationVerdict{
			Kind:   MaterializationFault,
			Reason: "skill_procedure_cold_build",
			ViewID: view.ViewID,
			Entry:  view.CacheEntry.ID,
		},
		View:    view,
		Payload: append([]byte(nil), payload...),
		Built:   true,
	}
}
