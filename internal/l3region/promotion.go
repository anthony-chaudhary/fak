package l3region

import (
	"context"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ---------------------------------------------------------------------------
// G6 — durability-tiered L3 promotion (child C of the L3 epic; issue #76).
//
// Re-conceive L3 admission as a TRUTH-DURATION policy, not a cache-hit-rate one. A
// CAMA-style L3 admits by recency/frequency (W-TinyLFU / SIEVE): a page earns the
// shared pool by being HOT. Re-imagined, a page reaches the shared L3 pool ONLY if the
// write-time durability gate (#498 — ctxmmu.classifyDurability stamping
// Verdict.Meta["durability"]) classified it `bounded` or `durable`. A `turn` / `session`
// page stays local (L1/L2) and never reaches the multi-tenant tier — NO MATTER HOW HOT.
// This converts a cache policy into a truth-duration policy: "this prefix is durably
// true across sessions, so it earns the shared tier" vs "this is scratch, keep it local."
//
// SAME GATE, NEW DECISION POINT. This is the default-expire gate logic the recall store
// already ships (#82 / #499 — internal/recall.PromotionMode), re-pointed at the L3 `set`
// admission decision instead of the recall core image. It is a CONTROL-PATH decision (on
// set admission) and never touches the data path — the study's single sharpest constraint
// (docs/notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md §3 G6, §4 child C).
//
// LAYERING. l3region is tier 1 (foundation): it may import only abi + stdlib, NOT ctxmmu
// (tier 2). So the durability vocabulary is mirrored here as local string constants —
// byte-identical to ctxmmu.Durability{Turn,Session,Durable} so a class the write gate
// stamped on Meta["durability"] reads through unchanged. The drift guard
// TestL3DurabilityVocabularyMatchesCtxmmu (a _test.go, which architest's layering scan
// excludes) pins these to the source so a divergence reds the build.
// ---------------------------------------------------------------------------

// Durability classes, mirrored from ctxmmu's write-time vocabulary (see LAYERING above).
// `bounded` is the rung-2 reserved class ctxmmu documents but does not yet emit; the L3
// floor admits it by default so the tier is ready when the classifier starts emitting it.
const (
	DurabilityKey     = "durability"
	DurabilityTurn    = "turn"    // true only this turn; the fail-closed default
	DurabilitySession = "session" // true for this session
	DurabilityBounded = "bounded" // true for a bounded validity interval (rung-2 reserved)
	DurabilityDurable = "durable" // true across sessions until revised
)

// L3PromotionMode is the warn-then-enforce rollout posture, mirroring
// recall.PromotionMode (#499) so a misclassification cannot silently strand pages before
// the boundary actually bites.
type L3PromotionMode uint8

const (
	// L3PromotionWarn is the audit-only, NON-BEHAVIOR-CHANGING default: classify the
	// page, count a would-deny for each below-floor/unknown class, but still admit every
	// page to L3 exactly as the ungated path. The safe posture while the write-time
	// classifier and the callers are audited for the eventual enforce flip.
	L3PromotionWarn L3PromotionMode = iota
	// L3PromotionEnforce makes the inversion bite: a page is admitted to the shared L3
	// pool ONLY if its class is at/above the floor. A turn/session/unknown page is never
	// mset into the tier, so it cannot pollute the multi-tenant pool just because it is hot.
	L3PromotionEnforce
)

// L3Reason is the typed, closed-vocabulary outcome of an L3 promotion decision — the
// "typed reason" the gate returns alongside admit/deny (acceptance criterion 1).
type L3Reason uint8

const (
	// L3ReasonAdmitted: the class is at/above the floor (bounded/durable by default) —
	// it earns the durable shared tier.
	L3ReasonAdmitted L3Reason = iota
	// L3ReasonDeniedBelowFloor: a RECOGNIZED class below the floor (turn/session by
	// default) — scratch that stays local. Frequency is irrelevant; a hot turn page is
	// still denied.
	L3ReasonDeniedBelowFloor
	// L3ReasonDeniedUnknown: a missing/unknown/unrecognized class — fail-closed
	// (default-expire), matching #499's posture and abi.FallbackDeny.
	L3ReasonDeniedUnknown
)

// String renders the typed reason for audit/metric surfaces.
func (r L3Reason) String() string {
	switch r {
	case L3ReasonAdmitted:
		return "admitted"
	case L3ReasonDeniedBelowFloor:
		return "denied_below_floor"
	case L3ReasonDeniedUnknown:
		return "denied_unknown"
	default:
		return "denied_unknown"
	}
}

// L3Decision is the gate's verdict on one page's L3 admission.
type L3Decision struct {
	// Admit is whether the page should actually be promoted to L3 under the current
	// posture: true for an at/above-floor class always; for a below-floor/unknown class
	// it is true under WARN (non-behavior-changing) and false under ENFORCE.
	Admit bool
	// Class is the class the decision was made on (the raw value, or DurabilityTurn for a
	// missing/unknown value the gate failed closed on).
	Class string
	// Reason is the typed why.
	Reason L3Reason
	// Promotable is the posture-independent classification: true iff Class is at/above the
	// floor. It differs from Admit only under WARN, where a non-promotable page is still
	// admitted — so a caller can audit "what WOULD enforce do" without changing posture.
	Promotable bool
}

// L3PromotionGate decides admit/deny-to-L3 from a page's write-time durability class. The
// floor is configurable (default: bounded+durable admitted, turn+session denied); an
// unknown/missing class fails closed to deny. It is concurrency-safe: the audit counter is
// atomic and the floor is read-only after construction.
type L3PromotionGate struct {
	mode  L3PromotionMode
	floor map[string]bool // the admitted class set (read-only after the With* builders)

	deniedPromotions int64 // would-deny (WARN) / actual-deny (ENFORCE) audit count
}

// NewL3PromotionGate builds a gate at the default floor (bounded+durable admitted) in the
// audit-only WARN posture. Opt into the enforced default-expire boundary with
// WithMode(L3PromotionEnforce); change the floor with WithFloor.
func NewL3PromotionGate() *L3PromotionGate {
	return &L3PromotionGate{
		mode:  L3PromotionWarn,
		floor: map[string]bool{DurabilityBounded: true, DurabilityDurable: true},
	}
}

// WithMode sets the rollout posture and returns the gate for chaining.
func (g *L3PromotionGate) WithMode(m L3PromotionMode) *L3PromotionGate { g.mode = m; return g }

// WithFloor replaces the admitted class set (e.g. admit only durable, or also admit
// session). An unrecognized class name is ignored so a typo cannot silently widen the
// floor; pass recognized vocabulary values. Returns the gate for chaining.
func (g *L3PromotionGate) WithFloor(classes ...string) *L3PromotionGate {
	floor := map[string]bool{}
	for _, c := range classes {
		if recognizedClass(c) {
			floor[c] = true
		}
	}
	g.floor = floor
	return g
}

// recognizedClass reports whether class is a known durability value (vs missing/garbage).
func recognizedClass(class string) bool {
	switch class {
	case DurabilityTurn, DurabilitySession, DurabilityBounded, DurabilityDurable:
		return true
	default:
		return false
	}
}

// Admit decides whether a page of the given write-time durability class may be promoted
// into the shared L3 pool, returning a typed decision. It records the decision into the
// gate's audit counter (a below-floor/unknown class increments DeniedPromotions), so one
// Admit call corresponds to one page on the admission stream.
func (g *L3PromotionGate) Admit(class string) L3Decision {
	switch {
	case g.floor[class]:
		return L3Decision{Admit: true, Class: class, Reason: L3ReasonAdmitted, Promotable: true}
	case recognizedClass(class):
		// Recognized but below the floor (turn/session by default): scratch — denied.
		atomic.AddInt64(&g.deniedPromotions, 1)
		return L3Decision{Admit: g.mode == L3PromotionWarn, Class: class, Reason: L3ReasonDeniedBelowFloor}
	default:
		// Missing/unknown/unrecognized: fail closed to turn (default-expire), matching
		// #499 and abi.FallbackDeny — remembering-when-wrong is the expensive direction.
		atomic.AddInt64(&g.deniedPromotions, 1)
		return L3Decision{Admit: g.mode == L3PromotionWarn, Class: DurabilityTurn, Reason: L3ReasonDeniedUnknown}
	}
}

// DeniedPromotions reports how many pages the gate denied (under ENFORCE) or would have
// denied (the would-deny audit count under WARN) — the auditable signal of the
// default-expire inversion, so a warn-mode rollout can size the eventual enforce impact
// before flipping it.
func (g *L3PromotionGate) DeniedPromotions() int64 { return atomic.LoadInt64(&g.deniedPromotions) }

// PutGated is the G6 durability-tiered admission path (#76): it consults the gate on the
// page's write-time durability class BEFORE admitting the region to the shared L3 pool.
//
//   - An at/above-floor (bounded/durable) page is Put exactly as the ungated path and the
//     real RefRegion handle is returned with admitted=true.
//   - Under ENFORCE a denied (turn/session/unknown) page is NOT mset — its bytes never
//     reach the shared tier no matter how hot it is — and the zero Ref is returned with
//     admitted=false.
//   - Under WARN a denied page is STILL Put (non-behavior-changing) but the would-deny is
//     counted on the gate.
//
// This is the control-path decision only: it gates whether the existing Put/mset runs; it
// adds nothing to the byte data path. The decision carries the typed reason for the caller.
func (b *L3RegionBackend) PutGated(ctx context.Context, payload []byte, class string, gate *L3PromotionGate) (abi.Ref, L3Decision, error) {
	d := gate.Admit(class)
	if !d.Admit {
		return abi.Ref{}, d, nil
	}
	ref, err := b.Put(ctx, payload)
	return ref, d, err
}
