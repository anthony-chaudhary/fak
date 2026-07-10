package rsiloop

// forksaving.go — MEASURE the per-fork prefix-reuse SAVING (#2875, part of the
// Hermes-inspiration epic #2871). Its siblings reviewgate.go (#2837) PRICE whether a
// review fork is worth spawning, and cacheparity.go (#2838) WITNESS the FRACTION of the
// fork's prefix actually served from cache. This file turns that witnessed split into a
// measured per-fork SAVING — the honest, continuously-measured answer to Hermes' one-time
// "~26% cost reduction" assertion.
//
// THE GAP (the Hermes mechanism this improves on). The Hermes background-review fork
// (agent/background_review.py) inherits the parent's cached system prompt for byte-identical
// tools[] prefix-cache parity, claimed at "~26% cost reduction (issue #25322)". That number
// is ASSERTED ONCE and never re-measured: nothing continuously prices what a given fork
// actually saved on live usage, and the single figure hides whether a fork silently
// cold-wrote its prefix (paying MORE than a fresh turn, not less). fak already sees the real
// per-turn cache_read vs cache_creation split (the gateway usage seam; the same
// ForkTurnUsage cacheparity.go reads); this file prices that split into a per-fork saving.
//
// THE ECONOMICS — GROSS vs NET, the #2797 honesty split. All figures are in input-token-
// EQUIVALENTS on the ONE canonical cacheprice basis (#2798), so this primitive and the
// compaction report value an identical cached token identically BY CONSTRUCTION:
//
//	gross          = read     × (1 − ReadMultiplier)   // the read rebate: what the reused
//	                                                    //   prefix saved vs fresh input.
//	                                                    //   The OPTIMISTIC upper bound —
//	                                                    //   Hermes' "~26%" is this shape,
//	                                                    //   the rebate with the cost of the
//	                                                    //   cache ignored.
//	writePremium   = creation × (WriteMultiplier − 1)   // the EXCESS-over-fresh the fork paid
//	                                                    //   for the prefix it COLD-WROTE (a
//	                                                    //   cache miss it should have hit) —
//	                                                    //   a real cost, not a saving.
//	net            = gross − writePremium               // the HEADLINE (#2797: net first,
//	                                                    //   gross as a labelled upper bound).
//	                                                    //   It KEEPS ITS SIGN: a cold-write-
//	                                                    //   dominant fork reads NEGATIVE — it
//	                                                    //   LOST money vs a plain fresh turn,
//	                                                    //   the exact dishonesty a one-time
//	                                                    //   asserted % cannot express (#1303:
//	                                                    //   do not floor a net at zero).
//	counterfactual = (read + creation) × 1.0            // the no-sharing baseline: the whole
//	                                                    //   cacheable prefix billed FRESH.
//
// The write premium defaults to the 5-minute tier (cacheprice.Write5mMultiplier), matching
// the Track-2 report's convention of pricing unattributed cache_creation at 5m (#2179); a
// caller with a 1h-attributed creation split passes cacheprice.Write1hMultiplier.
//
// PURE, DETERMINISTIC, and TESTABLE. The same usage split + basis prices identically every
// time, importing nothing but fmt + cacheprice, so the honesty witness below is a fixed test
// (the #2819 discipline). The live wiring — feeding this per forked turn's realized usage off
// the gateway usage seam, keyed by fork lineage, and appending NewForkSavingRow — is the
// named follow-on, exactly as cacheparity.go's fence and reviewgate.go's spawn decision
// landed pure ahead of their own live seams.
//
// GENERATION FRAME (gen/next). Promotion evidence (→ now): a corpus of real per-fork saving
// rows off live guard sessions whose fleet-median net saving is positive and stable — that
// replaces Hermes' asserted % with a measured fleet number and earns the live-seam wiring.
// Demotion/retirement evidence: if measured per-fork net savings are consistently ~zero or
// negative (forks that don't actually reuse, or whose cold-write premium eats the rebate),
// the "cheap fork" primitive isn't paying for itself — retire it or fold it into reviewgate's
// spawn gate. Invalidating assumption: that the provider's cache_read/cache_creation split is
// attributable to an INDIVIDUAL fork on the live seam; if fork usage cannot be isolated from
// the parent turn's, the per-fork saving is unmeasurable and this stays a modeled trailer,
// not a witnessed one.

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/cacheprice"
)

// ForkSavingSchema tags a durable per-fork prefix-reuse saving witness row so a reader can
// never confuse it for another rsiloop journal (the review-fire ledger, the fork-parity
// witness, the curator ledger). The "/1" is the row-shape version. The row is built pure
// (NewForkSavingRow) and persisted by the caller that wires this onto the live usage seam.
const ForkSavingSchema = "fak-fork-prefix-saving/1"

// ForkSavingBasis is the price basis the per-fork saving is booked at. Only the write tier
// is a choice: the read rebate and the fresh-input baseline are fixed cacheprice anchors, so
// the sole tunable is which cache-write premium the cold-written portion paid.
type ForkSavingBasis struct {
	// WriteMultiplier is the cache-write price relative to base input for the fork's
	// cold-written prefix — cacheprice.Write5mMultiplier (1.25×, the default, matching the
	// Track-2 report's unattributed-creation convention) or cacheprice.Write1hMultiplier
	// (2.0×) when the creation split is 1h-attributed.
	WriteMultiplier float64
}

// DefaultForkSavingBasis prices the cold-written prefix at the 5-minute cache-write tier —
// the same default the Track-2 report uses for unattributed cache_creation (#2179).
func DefaultForkSavingBasis() ForkSavingBasis {
	return ForkSavingBasis{WriteMultiplier: cacheprice.Write5mMultiplier}
}

func (b ForkSavingBasis) withDefaults() ForkSavingBasis {
	if b.WriteMultiplier <= 0 {
		b.WriteMultiplier = cacheprice.Write5mMultiplier
	}
	return b
}

// ForkPrefixSaving is the measured per-fork prefix-reuse saving — the priced form of a
// forked turn's ForkTurnUsage (the same cache_read/cache_creation split cacheparity.go
// witnesses). It carries the gross upper bound, the write premium the fork paid, and the
// NET headline (signed), each in input-token-equivalents, plus their percentages against the
// no-sharing counterfactual, so a caller can journal a re-measurable saving instead of citing
// a one-time number.
type ForkPrefixSaving struct {
	// HadParentPrefix echoes the fork-lineage bit: false means there was no parent prefix to
	// reuse (a genuine first turn), so its cold-write is EXPECTED and there is no reuse saving
	// to attribute — Attributable is false and the token figures are zero.
	HadParentPrefix bool
	// Attributable is true iff a parent prefix existed to reuse (HadParentPrefix) — the gate
	// on whether the saving figures below mean anything. A first-turn fork is not a saving of
	// zero; it is a saving that does not apply.
	Attributable bool

	// CacheReadTokens / CacheCreationTokens echo the measured split the saving was priced from.
	CacheReadTokens     uint64
	CacheCreationTokens uint64

	// GrossSavedTokenEquiv is the read rebate — read × (1 − ReadMultiplier). The optimistic
	// upper bound (Hermes' one-time % is this shape), the rebate with the cache's cost ignored.
	GrossSavedTokenEquiv float64
	// WritePremiumTokenEquiv is the excess-over-fresh the fork paid to COLD-WRITE part of its
	// prefix — creation × (WriteMultiplier − 1). A real cost that nets against the rebate.
	WritePremiumTokenEquiv float64
	// NetSavedTokenEquiv is the HEADLINE: gross − writePremium, SIGNED. Negative when a fork
	// cold-wrote enough that the write premium ate the rebate — it lost money vs a fresh turn.
	NetSavedTokenEquiv float64
	// CounterfactualTokenEquiv is the no-sharing baseline: the whole cacheable prefix billed
	// fresh, (read + creation) × 1.0 — the denominator the percentages are taken against.
	CounterfactualTokenEquiv float64

	// GrossSavedPct / NetSavedPct are the saving as a percentage of the counterfactual — the
	// directly-comparable-to-Hermes'-"~26%" figures, but MEASURED and net-honest. NetSavedPct
	// keeps its sign. Both are zero when there is no cacheable prefix (nothing to save).
	GrossSavedPct float64
	NetSavedPct   float64

	// WriteMultiplier is the cache-write tier the premium was booked at, echoed for audit.
	WriteMultiplier float64
}

// MeasureForkSaving is the PURE fold: it prices a forked turn's realized cache_read/
// cache_creation split into the gross/net prefix-reuse saving. A fork with no parent prefix
// (an expected first-turn cold-write) is NOT attributable — its figures are zero and the
// caller reports it as "no reuse saving to measure", never as a saving of zero. It does no
// I/O, so it stays deterministic and unit-testable (the #2819 discipline).
func MeasureForkSaving(basis ForkSavingBasis, usage ForkTurnUsage) ForkPrefixSaving {
	b := basis.withDefaults()
	s := ForkPrefixSaving{
		HadParentPrefix:     usage.HadParentPrefix,
		Attributable:        usage.HadParentPrefix,
		CacheReadTokens:     usage.CacheReadTokens,
		CacheCreationTokens: usage.CacheCreationTokens,
		WriteMultiplier:     b.WriteMultiplier,
	}
	if !usage.HadParentPrefix {
		// No parent prefix to reuse — the cold-write is expected, not a saving. Leave the
		// figures zero; a first-turn fork's saving does not apply (it is not zero saving).
		return s
	}
	read := float64(usage.CacheReadTokens)
	creation := float64(usage.CacheCreationTokens)
	s.GrossSavedTokenEquiv = read * (1 - cacheprice.ReadMultiplier)
	s.WritePremiumTokenEquiv = creation * (b.WriteMultiplier - 1)
	s.NetSavedTokenEquiv = s.GrossSavedTokenEquiv - s.WritePremiumTokenEquiv
	s.CounterfactualTokenEquiv = read + creation
	if s.CounterfactualTokenEquiv > 0 {
		s.GrossSavedPct = 100 * s.GrossSavedTokenEquiv / s.CounterfactualTokenEquiv
		s.NetSavedPct = 100 * s.NetSavedTokenEquiv / s.CounterfactualTokenEquiv
	}
	return s
}

// Trailer renders the measured per-fork saving as a one-line compaction-economics-style
// trailer, NET FIRST with the gross as a labelled upper bound (the #2797 honesty order) and
// the raw split + write premium so the number is auditable, not asserted. A non-attributable
// (first-turn) fork renders the honest "no reuse saving to measure" note instead of a bogus
// zero saving.
func (s ForkPrefixSaving) Trailer() string {
	if !s.Attributable {
		return "fork prefix-reuse saving: no parent prefix to reuse (first-turn prefix write is expected — no reuse saving to measure)"
	}
	return fmt.Sprintf(
		"fork prefix-reuse saving (measured): net %+.0f tok-eq (%+.1f%% of a %.0f tok-eq all-fresh prefix); gross upper bound %+.0f tok-eq (%+.1f%%); read %d tok / cold-write %d tok @%.2f× write premium %.0f tok-eq",
		s.NetSavedTokenEquiv, s.NetSavedPct, s.CounterfactualTokenEquiv,
		s.GrossSavedTokenEquiv, s.GrossSavedPct,
		s.CacheReadTokens, s.CacheCreationTokens, s.WriteMultiplier, s.WritePremiumTokenEquiv,
	)
}

// MeasureForkSavingTrailer is the convenience one-call form: measure then render. It is what
// a caller emits per forked turn once this primitive is wired onto the live usage seam.
func MeasureForkSavingTrailer(basis ForkSavingBasis, usage ForkTurnUsage) string {
	return MeasureForkSaving(basis, usage).Trailer()
}

// ForkSavingRow is one durable per-fork prefix-reuse saving witness record — the row a
// caller appends onto the gateway usage seam when this primitive is wired live, tagged
// ForkSavingSchema so it never shares a row shape with the fork-parity witness or the
// review-fire ledger. It carries the measured split and the gross/net saving so the "~26%"
// claim becomes a re-measurable ledger instead of a one-time citation.
type ForkSavingRow struct {
	Schema                 string  `json:"schema"`
	Seq                    int     `json:"seq"`
	HadParentPrefix        bool    `json:"had_parent_prefix"`
	Attributable           bool    `json:"attributable"`
	CacheReadTokens        uint64  `json:"cache_read_tokens"`
	CacheCreationTokens    uint64  `json:"cache_creation_tokens"`
	GrossSavedTokenEquiv   float64 `json:"gross_saved_token_equiv"`
	WritePremiumTokenEquiv float64 `json:"write_premium_token_equiv"`
	NetSavedTokenEquiv     float64 `json:"net_saved_token_equiv"`
	NetSavedPct            float64 `json:"net_saved_pct"`
	WriteMultiplier        float64 `json:"write_multiplier"`
}

// NewForkSavingRow builds the durable witness row for one forked turn from its usage and the
// basis. It is PURE (returns the row; the caller persists it), so wiring the live witness
// onto the gateway usage seam is a trivial append and this primitive stays unit-testable.
func NewForkSavingRow(seq int, basis ForkSavingBasis, usage ForkTurnUsage) ForkSavingRow {
	s := MeasureForkSaving(basis, usage)
	return ForkSavingRow{
		Schema:                 ForkSavingSchema,
		Seq:                    seq,
		HadParentPrefix:        s.HadParentPrefix,
		Attributable:           s.Attributable,
		CacheReadTokens:        s.CacheReadTokens,
		CacheCreationTokens:    s.CacheCreationTokens,
		GrossSavedTokenEquiv:   s.GrossSavedTokenEquiv,
		WritePremiumTokenEquiv: s.WritePremiumTokenEquiv,
		NetSavedTokenEquiv:     s.NetSavedTokenEquiv,
		NetSavedPct:            s.NetSavedPct,
		WriteMultiplier:        s.WriteMultiplier,
	}
}
