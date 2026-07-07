package promptmmu

import "github.com/anthony-chaudhary/fak/internal/cachemeta"

// stableedit.go — issue #2201 / autoctx rung R3 (placement dual-write). Design
// contract: docs/notes/CONCEPT-R3-PLACEMENT-DUAL-WRITE-2026-07-06.md.
//
// PlanBreakpoints (breakpoint.go, #1603) already derives the residency plan: the
// ProtectedPrefix / MutableTail spans and the UnsafeToCompact hazards over one turn's
// segments. R3's contract is that cache-breakpoint placement and stable-segment
// residency read that ONE structure instead of each re-deriving span boundaries — so
// prefix stability is a contract in code, not discipline (law L4 at the prefix
// boundary). CheckStableEdit is that read: given the plan and the half-open segment
// range a transform proposes to rewrite or drop, it reports whether the edit is safe.
//
// This is the ADVISORY first increment (L7: advisory first, gate later). CheckStableEdit
// itself is a pure read — no I/O, mutates nothing, splices no bytes. A caller runs it in
// SHADOW MODE: log a non-ok result, attributed, and proceed unchanged. The gate (refuse,
// or re-price the breakpoint) is a later rung once the shadow log reads clean.

// Closed vocabulary of reasons CheckStableEdit reports, so a caller (or a shadow-mode log
// line) branches on WHY an edit is unsafe without parsing free text.
const (
	// StableEditSealed marks a proposed edit that intersects a SegSealed span
	// (refusal rule 3): rewriting or dropping it would re-serve or lose
	// fak-quarantined content. Reported wherever the sealed span sits, independent
	// of the protected-prefix boundary — the sealed rule is unconditional.
	StableEditSealed = "sealed-segment-edit"
	// StableEditStable marks a proposed edit that reaches into the protected prefix:
	// the warm, provider-cached leading run PlanBreakpoints named. Editing inside it
	// busts the cached prefix from that point on (law L4) — the hazard R3 exists to
	// make visible.
	StableEditStable = "stable-segment-edit"
)

// CheckStableEdit is the R3 advisory shadow-mode check: given a BreakpointPlan and the
// half-open segment range `edit` a transform proposes to rewrite/drop, it reports
// whether the edit is safe (ok) and, when not, a closed-vocabulary reason.
//
// It is a pure READ of the plan — the single derived structure both projections consume
// (residency plan ⇄ cache breakpoints). Placement is a read of the plan, not a second
// derivation, so this check and PlanBreakpoints can never disagree about where the
// stable boundary sits.
//
// The decision, given the plan's spans:
//   - An empty edit (Start >= End) touches nothing: ok.
//   - An edit intersecting any SegSealed span named in UnsafeToCompact: not ok,
//     StableEditSealed. Checked first because refusal rule 3 is unconditional — a sealed
//     span is a hazard wherever it sits, including inside a PrefixUnknown/PrefixStable
//     turn whose whole span is nominally "protected".
//   - Otherwise, an edit intersecting the ProtectedPrefix span: not ok, StableEditStable
//     (the L4 warm-prefix hazard).
//   - An edit lying wholly within MutableTail and touching no stable/sealed segment: ok.
//
// The only legal cache_control breakpoint position is the ProtectedPrefix.End boundary:
// a breakpoint sits BETWEEN the last protected segment and the first mutable one, never
// inside the protected run — which is exactly the boundary an ok edit stays clear of.
func CheckStableEdit(plan BreakpointPlan, edit Span) (ok bool, reason string) {
	if edit.Empty() {
		return true, ""
	}
	// Sealed spans first: refusal rule 3 is unconditional and independent of where the
	// protected-prefix boundary falls, so a sealed intersection is the more specific
	// hazard even when the sealed segment also sits inside a nominally-protected prefix.
	for _, u := range plan.UnsafeToCompact {
		if u.Kind == cachemeta.SegSealed && spansIntersect(edit, u.Span) {
			return false, StableEditSealed
		}
	}
	if spansIntersect(edit, plan.ProtectedPrefix) {
		return false, StableEditStable
	}
	return true, ""
}

// spansIntersect reports whether two half-open, end-exclusive segment ranges share at
// least one segment. Two empty spans, or a span disjoint from the other, do not
// intersect.
func spansIntersect(a, b Span) bool {
	if a.Empty() || b.Empty() {
		return false
	}
	return a.Start < b.End && b.Start < a.End
}
