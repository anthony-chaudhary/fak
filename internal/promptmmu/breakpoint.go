package promptmmu

import (
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// breakpoint.go — issue #1603 (managed-context: plan cache breakpoints as part of
// context planning).
//
// fak already preserves provider cache prefixes (CompactInboundTools/CompactInboundSystem
// splice past the last cache_control breakpoint) and separately scores whether that
// prefix survived a turn (cachemeta.PrefixStabilityTracker, #1602). Neither one is a
// PLAN: the splice acts on a byte-level breakpoint index and the score reports a
// stable/mutated verdict, but nothing today NAMES the three spans a managed-context loop
// actually needs before it decides whether a reset is safe:
//
//   - the PROTECTED (stable) prefix — the span the provider cache is warm for and that
//     must be carried byte-identical into the next turn;
//   - the MUTABLE tail — the span that is expected/safe to change every turn (the
//     provider re-bills it regardless, so compacting it costs nothing extra);
//   - the UNSAFE-TO-COMPACT spans — spans that would break provider cache affinity (a
//     mutation strictly inside the protected prefix) or lose semantic content (a
//     SegSealed span, refusal rule 3) if a compactor touched them.
//
// PlanBreakpoints is the pure planner that closes that gap: given the current turn's
// segments and the cachemeta signal for that turn (a PrefixStabilityScore, typically
// from PrefixStabilityTracker.Observe), it returns a BreakpointPlan naming all three
// spans explicitly. It is a planning-only read: it never mutates segments and never
// splices bytes — CompactInboundTools/CompactInboundSystem still own the actual splice,
// and a caller wires this plan's ProtectedPrefix boundary into those calls' breakpoint
// discipline rather than re-deriving it.
//
// This keeps the vocabulary this file introduces anchored to the two packages it draws
// on: "protected span" / "TTL" / "break-even" / "share scope" are cachemeta's
// (prefix_score.go, #1602); "segment" / "sealed" are cachemeta's SegmentKind
// (prefix_stability.go); "breakpoint" / "prefix" are promptmmu's own (promptmmu.go). No
// parallel vocabulary is invented.

// Span is a half-open, end-exclusive index range over a turn's []cachemeta.PromptSegment
// slice: segments [Start, End) belong to the span. An empty span (Start == End) reports
// no segments in that role for this turn — e.g. MutableTail is empty when the whole
// prompt is protected prefix. Indices are always valid against the segment slice the
// plan was built from; a caller must not reuse a Span against a different slice.
type Span struct {
	Start int
	End   int
}

// Len reports the number of segments the span covers.
func (s Span) Len() int { return s.End - s.Start }

// Empty reports whether the span covers no segments.
func (s Span) Empty() bool { return s.End <= s.Start }

// BreakpointPlan is the explicit, inspectable output #1603 asks the managed-context
// planner to produce: the three named spans over one turn's segments, plus the
// cachemeta working-spine facts (TTL, break-even, share scope) that turn a bare span
// list into an actionable compaction decision. A caller (a managed-context loop, or a
// future compactor) reads this INSTEAD of re-deriving span boundaries from a raw
// PrefixStabilityScore or re-scanning segments for SegSealed itself.
type BreakpointPlan struct {
	// ProtectedPrefix is the stable span: segments [0, ProtectedPrefix.End) that the
	// provider prefix cache is warm for and that MUST be carried byte-identical into
	// the next turn. It always starts at 0 — the protected prefix is, by construction,
	// the turn's leading run (CACHE GEOMETRY, promptmmu.go).
	ProtectedPrefix Span
	// MutableTail is the span strictly after ProtectedPrefix: segments that are
	// expected to change every turn (already re-billed regardless) and are therefore
	// safe to compact/drop without any additional cache cost. Empty when the whole
	// turn is protected (e.g. State == PrefixUnknown, first turn of a run).
	MutableTail Span
	// UnsafeToCompact names every span within the turn a compactor must NOT touch,
	// even though it may overlap MutableTail. Two distinct hazards populate this list,
	// each carrying its own Reason:
	//   - a SegSealed segment (refusal rule 3): compacting it would re-serve or drop
	//     fak-quarantined content, a semantic-content loss.
	//   - the first-divergent segment reported by a PrefixMutated score: the exact
	//     segment where the provider cache affinity broke this turn (the boundary
	//     segment, i.e. the first segment of MutableTail). Silently compacting or
	//     dropping it would erase the one segment a caller needs to diagnose WHY the
	//     prefix broke, so it is named even though it already sits in MutableTail.
	UnsafeToCompact []UnsafeSpan

	// State is the cachemeta.PrefixStabilityState the plan was built from, carried
	// through so a caller does not have to re-fetch the score to know WHY the prefix
	// is shaped this way.
	State cachemeta.PrefixStabilityState
	// TTLMillis, BreakEvenTokens, and ShareScope are copied verbatim from the
	// PrefixStabilityScore this plan was built from (see PrefixStabilityScore for their
	// meaning) — the same working-spine facts #1602 already computes, surfaced here so
	// a managed-context decision reads them off the plan rather than a second source.
	TTLMillis       int64
	BreakEvenTokens int
	ShareScope      abi.ShareScope
}

// UnsafeSpan names one span a compactor must not touch, plus why.
type UnsafeSpan struct {
	Span   Span
	Reason string
	Kind   cachemeta.SegmentKind
}

// Closed set of UnsafeSpan reasons, so a caller can branch on WHY without parsing text.
const (
	// UnsafeSealed marks a SegSealed segment: fak-quarantined content that must never
	// be re-served via the provider cache regardless of byte equality (refusal rule 3).
	UnsafeSealed = "sealed-span"
	// UnsafeDivergence marks the boundary segment where the protected prefix broke
	// this turn (the first segment of MutableTail): dropping or compacting it would
	// destroy the exact evidence of the cache-affinity break.
	UnsafeDivergence = "protected-prefix-diverged"
)

// PlanBreakpoints is the pure #1603 planner: given one turn's ordered segments and the
// cachemeta.PrefixStabilityScore already computed for that turn (typically from
// (*cachemeta.PrefixStabilityTracker).Observe, #1602), it names the protected prefix,
// the mutable tail, and every unsafe-to-compact span. It performs no I/O, mutates
// neither input, and never splices bytes.
//
// The protected-prefix boundary is taken from the score's own StableTokens/state
// rather than re-deriving it from segment content, so this planner and the tracker
// that produced the score can never disagree about where the boundary sits:
//
//   - PrefixUnknown (no baseline yet): the whole turn is the protected prefix (it
//     BECOMES the baseline, mirroring Observe's own seeding behavior) and MutableTail
//     is empty — there is nothing yet known to be safely mutable.
//   - PrefixStable: the whole turn is still the protected prefix (byte-identical, or a
//     pure tail-append per Observe's own definition of "stable"); MutableTail is empty.
//     A ProtectedSpanBroken score still reports the sealed segment via UnsafeToCompact
//     even though the state is otherwise stable-shaped, because sealed spans are a
//     stronger hazard than an ordinary content mutation (see Observe's own doc comment).
//   - PrefixMutated: the protected prefix ends at the first segment BEFORE the
//     divergence point that is still provider-cacheable (derived from StableTokens by
//     walking cumulative segment Tokens); everything from there on is MutableTail. The
//     diverging segment itself is additionally named in UnsafeToCompact so a caller can
//     see exactly what broke, not just where the safe boundary now sits.
//
// A SegSealed segment anywhere in the turn is always named in UnsafeToCompact,
// independent of state, because refusal rule 3 applies unconditionally.
func PlanBreakpoints(turn []cachemeta.PromptSegment, score cachemeta.PrefixStabilityScore) BreakpointPlan {
	plan := BreakpointPlan{
		State:           score.State,
		TTLMillis:       score.TTLMillis,
		BreakEvenTokens: score.BreakEvenTokens,
		ShareScope:      score.ShareScope,
	}

	protectedEnd := len(turn)
	switch score.State {
	case cachemeta.PrefixMutated:
		protectedEnd = protectedPrefixEnd(turn, score.StableTokens)
	case cachemeta.PrefixUnknown, cachemeta.PrefixStable:
		protectedEnd = len(turn)
	}

	plan.ProtectedPrefix = Span{Start: 0, End: protectedEnd}
	plan.MutableTail = Span{Start: protectedEnd, End: len(turn)}

	for i, seg := range turn {
		if seg.Kind == cachemeta.SegSealed {
			plan.UnsafeToCompact = append(plan.UnsafeToCompact, UnsafeSpan{
				Span:   Span{Start: i, End: i + 1},
				Reason: UnsafeSealed,
				Kind:   cachemeta.SegSealed,
			})
		}
	}

	if score.State == cachemeta.PrefixMutated && score.FirstDivergentSegment >= 0 &&
		score.FirstDivergentSegment < len(turn) {
		plan.UnsafeToCompact = append(plan.UnsafeToCompact, UnsafeSpan{
			Span:   Span{Start: score.FirstDivergentSegment, End: score.FirstDivergentSegment + 1},
			Reason: UnsafeDivergence,
			Kind:   score.FirstDivergentKind,
		})
	}

	return plan
}

// protectedPrefixEnd converts a token-denominated StableTokens count into a segment
// index by walking cumulative segment tokens — the same coordinate space
// TurnDivergence.FirstDivergeTokenOffset uses (prefix_stability.go), so this planner
// never invents a second offset unit. It returns the index of the first segment whose
// cumulative token count would exceed stableTokens, i.e. the first segment NOT covered
// by the provider-cacheable prefix.
func protectedPrefixEnd(turn []cachemeta.PromptSegment, stableTokens int64) int {
	var cum int64
	for i, seg := range turn {
		if cum >= stableTokens {
			return i
		}
		cum += seg.Tokens
	}
	return len(turn)
}
