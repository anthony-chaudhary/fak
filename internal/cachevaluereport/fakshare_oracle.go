package cachevaluereport

import (
	"fmt"
	"math"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cacheprice"
	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
)

// The fak_share differential oracle (#3651): a SECOND, independent computation of the
// fleet fak_share that must AGREE with the one FoldFleetBenefit publishes as
// FleetBenefitReport.FakSharePct, and REDS when it does not.
//
// Why a second implementation at all. fak_share is a fak assertion about how much of the
// realized cache saving fak caused — the single number most exposed to motivated
// reasoning in this repo, because fak both authors the saving and grades it. A unit test
// of the primary can only prove the primary computes what its author intended; it cannot
// falsify the number. Only a second derivation that arrives at the same figure by a
// different route can. That is what this is: a differential oracle, not a replacement.
// FoldFleetBenefit stays the primary and publishes the headline (#3651 explicitly does
// NOT redefine the attribution model — that is owned by #2783/#1301).
//
// WHAT IS INDEPENDENT (the axes the diff actually covers):
//
//   - INPUTS. The oracle re-derives every per-row token-equivalent from the RAW provider
//     counters (cache_read / cache_creation / compaction shed / warm witness) through the
//     canonical cacheprice multipliers. It never reads the PERSISTED derived fields —
//     SavedTokenEquiv, NetSavedTokenEquiv — that the primary trusts via
//     providerTokenEqFromRow / fakTokenEqFromRow. A row whose persisted equiv disagrees
//     with what today's pricer computes from its own raw counters therefore reds here,
//     which is the point: the persisted field is an assertion too.
//   - ACCUMULATION. Its own traversal and its own accumulators, sharing no state with the
//     fold. It does not call FoldFleetBenefit, fakShareAtBasis, or any helper on the
//     primary's arithmetic path.
//   - FINAL ARITHMETIC. The share is expressed in the RECIPROCAL form,
//     100 / (1 + provider/fak), rather than the primary's 100 * fak / (provider + fak).
//     The two are algebraically identical and differ only in float rounding (far inside
//     the tolerance), so a slip in either division cannot be mirrored by the other.
//
// WHAT IS DELIBERATELY NOT INDEPENDENT (and why a diff here would be meaningless):
//
//   - The ATTRIBUTION DEFINITION — which rows count as provider-caused vs fak-authored,
//     and the turns>=1 Track-1 corpus gate. That definition IS the contract under test's
//     ownership elsewhere (#2783/#1301); two deliberately-different definitions would red
//     on healthy data forever and prove nothing. The oracle re-derives the VALUE of each
//     row, not the QUESTION of whose row it is.
//   - normalizeSavingsDimensions. Filling an empty provider/mechanism is input
//     conditioning that happens before either computation begins, not arithmetic under
//     test; both sides must see the same conditioned row or every legacy row reds.
//
// So a red means one of exactly three things, in falling order of likelihood: the fold's
// attribution arithmetic drifted from the definition; a persisted per-row equiv was
// written under a price basis the current cacheprice constants no longer produce; or the
// price constants themselves forked (track2 still prices cache WRITES off vcachegov while
// this reads cacheprice — equal today, and this oracle is what notices if they diverge).

// FakShareOracleTolerancePP is the agreement band, in PERCENTAGE POINTS of fak_share.
// It is sized to absorb float64 rounding across two algebraically-equivalent formulations
// (which differ in the last ulp or two, ~1e-13 pp on realistic corpora) while still
// catching any divergence an operator could act on. It is NOT a fudge factor for a real
// disagreement: a hundredth of a percentage point on a share reported to four decimals is
// already two orders of magnitude below the display precision.
const FakShareOracleTolerancePP = 0.01

// FakShareOracleReport is the differential verdict: both numbers, their gap, and whether
// the gap reds. Both pointers mirror FleetBenefitReport.FakSharePct's nil convention — nil
// is UNDEFINED (no positive saved token-equiv to divide by), never a measured zero — and a
// definedness MISMATCH is itself a divergence, since "the primary published a share where
// the oracle finds none" is exactly the failure a bare numeric compare would silently pass.
type FakShareOracleReport struct {
	PrimaryPct *float64 `json:"primary_fak_share_pct,omitempty"`
	OraclePct  *float64 `json:"oracle_fak_share_pct,omitempty"`

	// DeltaPP is |oracle − primary| in percentage points, meaningful only when both are
	// defined (zero otherwise — read Diverged and DefinednessMismatch, not this).
	DeltaPP     float64 `json:"delta_pp"`
	TolerancePP float64 `json:"tolerance_pp"`

	Diverged            bool `json:"diverged"`
	DefinednessMismatch bool `json:"definedness_mismatch,omitempty"`

	// The oracle's own independently-recomputed inputs, published so a red is
	// diagnosable against the primary's ProviderPromptCacheTokenEq / FakAuthoredTokenEq
	// without re-running the fold.
	ProviderTokenEq    float64 `json:"oracle_provider_token_equiv"`
	FakAuthoredTokenEq float64 `json:"oracle_fak_authored_token_equiv"`
	TotalTokenEq       float64 `json:"oracle_total_token_equiv"`
	KVReusedTokens     uint64  `json:"oracle_fak_kv_prefix_reused_tokens"`
	Track1Rows         int     `json:"oracle_track1_rows"`
	SavingsRows        int     `json:"oracle_savings_rows"`

	Finding string `json:"finding"`
}

// RecomputeFakSharePct is the independent second implementation: it folds the Track-1
// KV-reuse ledger and the Track-2 savings rows into a fak_share percentage WITHOUT
// consulting FoldFleetBenefit or any persisted derived token-equiv, pricing every axis
// from raw counters via cacheprice.
//
// It returns ok=false when the recomputed total saved token-equiv is not positive —
// the same UNDEFINED (not zero) convention FakSharePct uses — so an empty or upside-down
// corpus can never be compared against, or rendered as, a 0%/100% claim.
func RecomputeFakSharePct(track1 []cachevalueledger.Row, track2 []SavingsRow) (float64, bool) {
	provider, fak, _, _ := recomputeFakShareAxes(track1, track2)
	return fakShareFromAxes(provider, fak)
}

// recomputeFakShareAxes is the oracle's traversal. It returns the provider-caused and
// fak-authored token-equivalents plus the raw KV-reuse total and the count of Track-1 rows
// admitted, all derived from raw counters only.
func recomputeFakShareAxes(track1 []cachevalueledger.Row, track2 []SavingsRow) (providerTeq, fakTeq float64, kvReused uint64, track1Rows int) {
	// Track-1: WITNESSED KV-prefix reuse. A zero-turn row is not a cold session, it is a
	// session that never ran a turn — the same corpus gate the fold applies.
	for _, row := range track1 {
		if row.Turns == 0 {
			continue
		}
		track1Rows++
		kvReused += row.ReusedTokens
	}

	// Track-2: reprice each row's own raw counters. Note the classification is a
	// two-arm switch, not two independent sums: a row that satisfies BOTH arms books as
	// provider exactly once, matching the definition and making double-counting
	// structurally impossible rather than merely absent.
	for _, row := range track2 {
		normalizeSavingsDimensions(&row)
		switch {
		case row.Mechanism == "provider_prompt_cache":
			providerTeq += oracleProviderTokenEq(row)
		case row.Provider == "fak" || strings.HasPrefix(row.Mechanism, "compaction"):
			// Always the honest warm/cold blend off the raw shed and its OBSERVED warm
			// witness — never the row's persisted NetSavedTokenEquiv, which is the
			// assertion this oracle exists to check.
			fakTeq += cacheprice.ShedTokenEquiv(row.CompactionShedTokens, row.CompactionCacheReadTokens)
		}
	}
	fakTeq += float64(kvReused)
	return providerTeq, fakTeq, kvReused, track1Rows
}

// oracleProviderTokenEq reprices a provider prompt-cache row's saved token-equivalent from
// its raw counters: the read rebate (each cached read avoided 1−0.1 of a base input token)
// plus the write axis (what the tokens would have cost uncached, minus what the 1h/5m
// write tiers actually billed). The 1h/5m split reads the canonical cacheprice
// multipliers; the primary's path reaches the same numbers through vcachegov, so a fork
// between those two sources shows up here as a divergence instead of silently.
//
// The result keeps its SIGN (#1303): a session that only wrote cache and never read it
// back is genuinely negative until reads repay the write premium, and flooring it at zero
// here would hide exactly that.
func oracleProviderTokenEq(row SavingsRow) float64 {
	upgraded := row.CacheCreationTokensUpgraded
	if upgraded > row.CacheCreationTokens {
		upgraded = row.CacheCreationTokens
	}
	remainder := row.CacheCreationTokens - upgraded
	billedWrite := float64(upgraded)*cacheprice.Write1hMultiplier + float64(remainder)*cacheprice.Write5mMultiplier
	readRebate := float64(row.CacheReadTokens) * (1 - cacheprice.ReadMultiplier)
	return readRebate + float64(row.CacheCreationTokens) - billedWrite
}

// fakShareFromAxes divides in the RECIPROCAL form — 100 / (1 + provider/fak) instead of
// 100 * fak / (provider + fak) — so the oracle's final arithmetic cannot mirror a slip in
// the primary's. The two agree to float rounding.
//
// The guards cover the cases the reciprocal form alone would fumble: a non-positive total
// is UNDEFINED (matching FakSharePct), and a zero fak axis against a positive total is an
// exact 0% rather than a division by zero.
func fakShareFromAxes(providerTeq, fakTeq float64) (float64, bool) {
	total := providerTeq + fakTeq
	if total <= 0 {
		return 0, false
	}
	if fakTeq == 0 {
		return 0, true
	}
	return 100 / (1 + providerTeq/fakTeq), true
}

// VerifyFakShare is the differential compare: it recomputes fak_share independently from
// the same raw ledgers the report was folded from, diffs it against the report's published
// FakSharePct, and reds when they disagree beyond FakShareOracleTolerancePP.
//
// The caller passes the SAME track1/track2 slices it handed FoldFleetBenefit. Passing
// different corpora makes the verdict meaningless — this oracle falsifies the computation,
// not the collection.
func VerifyFakShare(rep FleetBenefitReport, track1 []cachevalueledger.Row, track2 []SavingsRow) FakShareOracleReport {
	providerTeq, fakTeq, kvReused, track1Rows := recomputeFakShareAxes(track1, track2)
	out := FakShareOracleReport{
		PrimaryPct:         rep.FakSharePct,
		TolerancePP:        FakShareOracleTolerancePP,
		ProviderTokenEq:    providerTeq,
		FakAuthoredTokenEq: fakTeq,
		TotalTokenEq:       providerTeq + fakTeq,
		KVReusedTokens:     kvReused,
		Track1Rows:         track1Rows,
		SavingsRows:        len(track2),
	}
	if pct, ok := fakShareFromAxes(providerTeq, fakTeq); ok {
		out.OraclePct = &pct
	}

	switch {
	case out.PrimaryPct == nil && out.OraclePct == nil:
		out.Finding = "AGREES: both the primary fold and the independent oracle report fak_share as UNDEFINED (no positive saved token-equiv to divide by)"
	case out.PrimaryPct == nil || out.OraclePct == nil:
		out.Diverged = true
		out.DefinednessMismatch = true
		out.Finding = fmt.Sprintf("DIVERGED (definedness): primary=%s oracle=%s — one side found a divisible corpus and the other did not; the published share and the raw ledgers disagree on whether fak_share exists at all",
			fmtOraclePct(out.PrimaryPct), fmtOraclePct(out.OraclePct))
	default:
		out.DeltaPP = math.Abs(*out.OraclePct - *out.PrimaryPct)
		if out.DeltaPP > FakShareOracleTolerancePP {
			out.Diverged = true
			out.Finding = fmt.Sprintf("DIVERGED: primary fak_share=%.6f%% vs independent oracle=%.6f%% — gap %.6f pp exceeds the %.4f pp tolerance; the published attribution is not reproducible from the raw ledgers (oracle axes: provider=%.4f fak=%.4f token-equiv over %d savings row(s) and %d track-1 row(s))",
				*out.PrimaryPct, *out.OraclePct, out.DeltaPP, FakShareOracleTolerancePP, providerTeq, fakTeq, out.SavingsRows, track1Rows)
		} else {
			out.Finding = fmt.Sprintf("AGREES: primary fak_share=%.6f%% and independent oracle=%.6f%% within %.4f pp (gap %.2e pp)",
				*out.PrimaryPct, *out.OraclePct, FakShareOracleTolerancePP, out.DeltaPP)
		}
	}
	return out
}

func fmtOraclePct(p *float64) string {
	if p == nil {
		return "UNDEFINED"
	}
	return fmt.Sprintf("%.6f%%", *p)
}
