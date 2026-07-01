package cachemeta

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// prefix_score.go — issue #1602 (managed-context: surface a prefix-stability score).
//
// prefix_stability.go (§A3) and prefix_coherence.go (§A4) answer OFFLINE questions
// over a whole recorded session: "how much of this transcript was cacheable" and
// "should the next turn force a break". Neither one answers the LIVE question a
// managed-context loop asks every turn/reset boundary: "did the stable, cacheable
// prefix I am carrying forward survive THIS turn unchanged, or did it just get
// mutated out from under me?" That is the gap this file closes.
//
// A continuous run that survives many context resets is only as valuable as the
// cache value each reset preserves (see docs/cache-value-rollup.md). A managed-
// context plan needs three, and only three, answers about its prefix at any turn
// boundary:
//
//   - prefix-stable   — the protected span (system + tool schema + any sealed span)
//     is still byte-identical to the last-observed baseline. The provider prefix
//     cache is still warm for that whole span.
//   - prefix-mutated  — some byte in the protected span changed. The FIRST divergent
//     segment/offset is reported so the caller can see exactly what moved (a
//     reordered tool schema, an edited system prompt, an un-sealed span).
//   - prefix-unknown  — there is no baseline yet (first turn of the run, or the
//     tracker was explicitly reset after an acknowledged intentional rebuild).
//
// PrefixStabilityTracker is the live, STATEFUL counterpart to the pure functions in
// prefix_stability.go: it remembers the last-observed protected span so a caller
// (fak info, a managed-context loop) can call Observe once per turn/reset and get a
// score back, without re-deriving or persisting the baseline itself.

// PrefixStabilityState is the closed three-state verdict a managed-context loop
// reports for the run's cacheable prefix.
type PrefixStabilityState string

const (
	// PrefixUnknown means no baseline has been observed yet: the current turn BECOMES
	// the baseline and every future turn is compared against it.
	PrefixUnknown PrefixStabilityState = "prefix-unknown"
	// PrefixStable means the protected span stayed byte-identical to the baseline.
	PrefixStable PrefixStabilityState = "prefix-stable"
	// PrefixMutated means some byte in the protected span changed; FirstDivergentSpan
	// names where.
	PrefixMutated PrefixStabilityState = "prefix-mutated"
)

// PrefixStabilityScore is the full answer #1602 asks a live session to report: the
// state, the first divergent span when mutated, and the working-spine context
// (provider TTL, break-even, and cross-agent share scope) that turns a bare
// stable/mutated bit into an actionable managed-context decision.
type PrefixStabilityScore struct {
	State PrefixStabilityState

	// FirstDivergentSegment is the index into the observed turn of the first segment
	// that broke from the baseline (content mismatch, length overrun, or a sealed
	// span). -1 when State is not PrefixMutated.
	FirstDivergentSegment int
	// FirstDivergentTokenOffset is the token offset of that same first divergence —
	// the same coordinate prefix_stability.go's TurnDivergence.FirstDivergeTokenOffset
	// and ProviderCache.FirstDivergeAt use, so this score composes with the existing
	// §A2/§A3 machinery rather than inventing a second offset space.
	FirstDivergentTokenOffset int64
	// FirstDivergentKind is the SegmentKind of the first divergent segment ("" when
	// State is not PrefixMutated), so a caller can tell a volatile-metadata drift
	// (expected, benign) from a stable/tool-schema edit (the expensive kind).
	FirstDivergentKind SegmentKind

	// StableTokens/LostTokens mirror TurnDivergence: the cacheable prefix and the
	// re-billed remainder, measured against the baseline.
	StableTokens int64
	LostTokens   int64

	// ProtectedSpanBroken is true when the divergence (or the baseline itself) hit a
	// SegSealed segment — a fak-quarantined span that must never be re-served via the
	// provider cache regardless of byte equality (refusal rule 3). A caller should
	// treat this as a stronger signal than an ordinary content mutation: the prefix
	// is not just cold, it is required to be cold.
	ProtectedSpanBroken bool

	// BaselineTurn is the turn number (1-indexed, counting Observe calls) that seeded
	// the current baseline — the turn a caller would need to inspect to see what the
	// "stable" prefix actually contains.
	BaselineTurn int
	// ObservedTurn is the turn number of the just-scored Observe call.
	ObservedTurn int

	// TTLMillis is the provider's advisory cache retention for this prefix, when
	// known (0 = unknown/not provided). It lets a managed-context loop say not just
	// "stable" but "stable, and the provider will still have it warm for another
	// N ms" — see providerTTLMillis for the retention-string mapping this reuses.
	TTLMillis int64
	// BreakEvenTokens is 1 when retaining (not evicting) this prefix is cheaper than
	// recomputing it, 0 when recompute wins, and -1 when there is not enough cost
	// data to judge (SizeBytes/PerTokenPrefillNanos unset). It reuses
	// RetainCheaperThanRecompute so the managed-context break-even question and the
	// tiered-placement break-even question stay the same formula.
	BreakEvenTokens int
	// ShareScope is the cross-agent share scope the baseline was minted under
	// (abi.ScopeAgent by default — private, fail-closed). A managed-context plan
	// that wants to share a stable prefix across agents must see this scope widen
	// explicitly; it is never inferred from stability alone.
	ShareScope abi.ShareScope

	// Reason is a short human-readable explanation, mirroring the Reason field
	// convention used across cachemeta (e.g. PlacementDecision.Reason).
	Reason string
}

// PrefixStabilityTracker is a live, per-run tracker: it remembers the last-accepted
// baseline turn and scores each new turn against it. It holds no payload beyond the
// baseline's own segments (already in memory as part of building the turn), and it
// is not safe for concurrent use without external synchronization — exactly the same
// contract as any other single-session, turn-at-a-time state machine in this
// package (e.g. session.Table's per-trace state is synchronized by its caller).
type PrefixStabilityTracker struct {
	baseline     []PromptSegment
	baselineTurn int
	turns        int

	// TTLMillis, ShareScope, SizeBytes, Tokens, and PerTokenPrefillNanos configure the
	// working-spine fields every score carries. They are exported so a caller can set
	// them once at construction (from a ProviderCache.Retention string via
	// NewPrefixStabilityTracker, or directly) without a larger options struct.
	TTLMillis            int64
	ShareScope           abi.ShareScope
	SizeBytes            int64
	Tokens               int64
	PerTokenPrefillNanos int64
}

// NewPrefixStabilityTracker builds a tracker with no baseline yet (the first Observe
// call reports PrefixUnknown and seeds the baseline). retention is a provider
// retention hint ("5m" | "1h" | "" ) mapped through the same providerTTLMillis table
// prefix_coherence.go's provider-cache lowering uses, so a caller wiring this to a
// live ProviderCache event gets a consistent TTL without re-deriving the mapping.
func NewPrefixStabilityTracker(retention string, scope abi.ShareScope) *PrefixStabilityTracker {
	return &PrefixStabilityTracker{
		TTLMillis:  providerTTLMillis(retention),
		ShareScope: scope,
	}
}

// Reset clears the baseline: the NEXT Observe call reports PrefixUnknown and reseeds
// it. Use this when a reset is intentional and acknowledged (e.g. an operator-forced
// context rebuild) so the following mutation is not misreported as a surprise break.
func (t *PrefixStabilityTracker) Reset() {
	t.baseline = nil
	t.baselineTurn = 0
}

// Observe scores one turn's protected span (the caller passes only the segments that
// are meant to be stable/cacheable — typically system + tool schema + any sealed
// span, i.e. everything AnalyzeStability would call the "front cacheable run" plus
// its sealed cap; a caller wanting the whole prompt scored can also pass the whole
// turn) against the tracker's current baseline, then folds the observed turn in as
// the new baseline for the next call. This mirrors Diverge's turn-over-turn
// semantics (prefix_stability.go) but adds the missing STATE machine: unknown until
// a baseline exists, stable/mutated afterward.
func (t *PrefixStabilityTracker) Observe(turn []PromptSegment) PrefixStabilityScore {
	t.turns++
	score := PrefixStabilityScore{
		FirstDivergentSegment: -1,
		ObservedTurn:          t.turns,
		TTLMillis:             t.TTLMillis,
		ShareScope:            t.ShareScope,
		BreakEvenTokens:       t.breakEven(),
	}

	if t.baseline == nil {
		score.State = PrefixUnknown
		score.BaselineTurn = t.turns
		score.Reason = "no baseline observed yet; this turn becomes the baseline"
		t.baseline = append([]PromptSegment(nil), turn...)
		t.baselineTurn = t.turns
		return score
	}

	score.BaselineTurn = t.baselineTurn
	d := Diverge(t.baseline, turn)
	score.StableTokens = d.StableTokens
	score.LostTokens = d.LostTokens
	score.ProtectedSpanBroken = d.SealedStop

	// A break is a divergence WITHIN the old baseline's span that strands tokens — the
	// same predicate AnalyzeStability uses to find BrokeAtTurn (prefix_stability.go).
	// A pure tail-append (the whole baseline matched; only NEW trailing segments were
	// added) is not a mutation of the protected span: the provider still hits the
	// entire old prefix, so it is reported PrefixStable even though d.Identical is
	// false (Identical also requires equal length). A sealed stop is a mutation of a
	// different, stronger kind (refusal rule 3) even when every byte matched.
	brokeWithinBaseline := d.LostTokens > 0 && d.FirstDivergeSeg < len(t.baseline)
	switch {
	case d.Identical, !d.SealedStop && !brokeWithinBaseline:
		score.State = PrefixStable
		if d.Identical {
			score.Reason = "protected span byte-identical to the baseline"
		} else {
			score.Reason = "protected span byte-identical to the baseline (turn extended it, no divergence within the baseline span)"
		}
	default:
		score.State = PrefixMutated
		score.FirstDivergentSegment = d.FirstDivergeSeg
		score.FirstDivergentTokenOffset = d.FirstDivergeTokenOffset()
		if d.FirstDivergeSeg < len(turn) {
			score.FirstDivergentKind = turn[d.FirstDivergeSeg].Kind
		}
		if d.SealedStop {
			score.Reason = fmt.Sprintf("protected span sealed at segment %d (token offset %d); must not be re-served", d.FirstDivergeSeg, score.FirstDivergentTokenOffset)
		} else {
			score.Reason = fmt.Sprintf("protected span diverged at segment %d (token offset %d)", d.FirstDivergeSeg, score.FirstDivergentTokenOffset)
		}
	}

	t.baseline = append([]PromptSegment(nil), turn...)
	t.baselineTurn = t.turns
	return score
}

// breakEven answers the working-spine break-even question with RetainCheaperThanRecompute
// when the tracker has enough cost data, else abstains (-1).
func (t *PrefixStabilityTracker) breakEven() int {
	if t.SizeBytes <= 0 || t.Tokens <= 0 || t.PerTokenPrefillNanos <= 0 {
		return -1
	}
	profile := DefaultTierProfiles()[TierDRAM]
	if RetainCheaperThanRecompute(t.SizeBytes, t.Tokens, t.PerTokenPrefillNanos, profile) {
		return 1
	}
	return 0
}
