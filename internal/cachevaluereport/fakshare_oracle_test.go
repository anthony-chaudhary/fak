package cachevaluereport

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
)

// healthyOracleCorpus builds the "healthy data" corpus the oracle must agree with: Track-2
// rows produced by the REAL emitter (NewSavingsRows) rather than hand-set persisted
// equivs, so the raw counters and the derived fields are consistent by construction — the
// shape the durable ledger actually holds. The observation deliberately mixes a warm-witness
// compaction shed (a BLENDED lump: 3000 shed against a 1200-token warm witness) with a 1h
// TTL-upgraded slice of the cache writes, so every axis the oracle reprices independently —
// read rebate, 5m/1h write blend, warm/cold shed blend — is actually exercised.
func healthyOracleCorpus() ([]cachevalueledger.Row, []SavingsRow) {
	track1 := []cachevalueledger.Row{
		{Date: "2026-06-22", SessionType: "run", Turns: 10, PromptTokens: 12000, ReusedTokens: 8000},
		{Date: "2026-06-23", SessionType: "run", Turns: 4, PromptTokens: 5000, ReusedTokens: 1500},
		// Zero-turn row: never ran a turn, so neither implementation admits it.
		{Date: "2026-06-23", SessionType: "run", Turns: 0, PromptTokens: 999, ReusedTokens: 999},
	}
	track2 := NewSavingsRows(SavingsObservation{
		SessionType:                 "run",
		Provider:                    "anthropic",
		Context:                     "oracle",
		InputTokens:                 4000,
		CacheReadTokens:             20000,
		CacheCreationTokens:         6000,
		CacheCreationTokensUpgraded: 2500,
		OutputTokens:                800,
		CompactionShedTokens:        3000,
		CompactionCacheReadTokens:   1200,
		CompactionFired:             2,
		Pricing:                     SavingsPricing{InputPerMTokUSD: 3, OutputPerMTokUSD: 15, Source: "test"},
	}, time.Date(2026, 6, 22, 10, 0, 0, 0, time.UTC))
	return track1, track2
}

// TestFakShareOracleAgreesWithPrimaryOnHealthyData is the base of the differential: on a
// corpus the real emitter produced, the independent recompute must land on the primary's
// published FakSharePct within tolerance. If this ever reds without the injection tests
// also reding, the oracle itself (not the primary) is what drifted.
func TestFakShareOracleAgreesWithPrimaryOnHealthyData(t *testing.T) {
	track1, track2 := healthyOracleCorpus()
	rep := FoldFleetBenefit(track1, track2, nil, FleetBenefitOptions{})
	if rep.FakSharePct == nil {
		t.Fatalf("precondition: the primary must publish a fak_share on this corpus, got nil (report: %+v)", rep)
	}

	got := VerifyFakShare(rep, track1, track2)
	if got.Diverged {
		t.Fatalf("healthy corpus must AGREE, got a red: %s", got.Finding)
	}
	if got.OraclePct == nil {
		t.Fatalf("oracle share = UNDEFINED on a corpus the primary found divisible: %+v", got)
	}
	if got.DeltaPP > FakShareOracleTolerancePP {
		t.Fatalf("gap %.9f pp exceeds tolerance %.4f pp (primary %.9f%% vs oracle %.9f%%)",
			got.DeltaPP, FakShareOracleTolerancePP, *rep.FakSharePct, *got.OraclePct)
	}
	// Agreement must be to float rounding, not merely inside the band — a gap anywhere
	// near the tolerance means the two paths really are computing different quantities
	// and the band is only masking it.
	if got.DeltaPP > 1e-9 {
		t.Fatalf("independent recompute should agree to float rounding, gap = %.3e pp (primary %.12f%% vs oracle %.12f%%)",
			got.DeltaPP, *rep.FakSharePct, *got.OraclePct)
	}
	if !strings.HasPrefix(got.Finding, "AGREES:") {
		t.Fatalf("finding should lead with AGREES, got %q", got.Finding)
	}
	// The oracle's own axes are published for diagnosis; they must be the same quantities
	// the primary aggregated, or a future red would be unreadable.
	if got.KVReusedTokens != rep.FakKVPrefixReusedTokens || got.Track1Rows != rep.Track1Sessions {
		t.Fatalf("oracle track-1 axes = %d reused over %d row(s), want %d over %d",
			got.KVReusedTokens, got.Track1Rows, rep.FakKVPrefixReusedTokens, rep.Track1Sessions)
	}

	// RecomputeFakSharePct is the standalone entry point; it must return the same number
	// the verdict carries, so a caller that only wants the second opinion gets it.
	pct, ok := RecomputeFakSharePct(track1, track2)
	if !ok || pct != *got.OraclePct {
		t.Fatalf("RecomputeFakSharePct = %v (ok=%v), want %v", pct, ok, *got.OraclePct)
	}
}

// TestFakShareOracleRedsOnInjectedPrimaryDiscrepancy is the falsification the issue asks
// for: with the raw ledgers held fixed, a primary that publishes a DIFFERENT fak_share must
// make the oracle red. This injects at the report boundary (the number the primary
// publishes) rather than by editing the fold, so it covers ANY arithmetic defect upstream
// of FakSharePct without pinning the test to one line of the implementation.
func TestFakShareOracleRedsOnInjectedPrimaryDiscrepancy(t *testing.T) {
	track1, track2 := healthyOracleCorpus()
	rep := FoldFleetBenefit(track1, track2, nil, FleetBenefitOptions{})
	if rep.FakSharePct == nil {
		t.Fatalf("precondition: the primary must publish a fak_share, got nil")
	}
	honest := *rep.FakSharePct

	for _, tc := range []struct {
		name  string
		share float64
	}{
		// A gross overstatement — the failure mode that actually matters, fak claiming
		// credit it did not earn.
		{"overstated", honest + 12},
		{"understated", honest - 12},
		// Just outside the band: the oracle must not need a large error to notice.
		{"just_outside_tolerance", honest + 2*FakShareOracleTolerancePP},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bugged := rep
			bugged.FakSharePct = &tc.share

			got := VerifyFakShare(bugged, track1, track2)
			if !got.Diverged {
				t.Fatalf("injected primary share %.6f%% (honest %.6f%%) must RED, got: %s", tc.share, honest, got.Finding)
			}
			if !strings.HasPrefix(got.Finding, "DIVERGED") {
				t.Fatalf("finding should lead with DIVERGED, got %q", got.Finding)
			}
			// The red must name both numbers, or an operator cannot act on it.
			if !strings.Contains(got.Finding, "primary") || !strings.Contains(got.Finding, "oracle") {
				t.Fatalf("red must name both sides, got %q", got.Finding)
			}
		})
	}

	// Inside the band the oracle must stay quiet — a tolerance that reds on float noise
	// would be a broken alarm, and a broken alarm gets silenced.
	nearly := honest + FakShareOracleTolerancePP/2
	quiet := rep
	quiet.FakSharePct = &nearly
	if got := VerifyFakShare(quiet, track1, track2); got.Diverged {
		t.Fatalf("a %.4f pp gap is inside tolerance and must not red: %s", FakShareOracleTolerancePP/2, got.Finding)
	}
}

// TestFakShareOracleRedsOnDriftedPersistedEquiv covers the second injection axis, and the
// reason the oracle reprices from RAW counters instead of reading the persisted derived
// fields: a row whose stored NetSavedTokenEquiv no longer matches what its own counters
// price to. The primary faithfully propagates the stored assertion (that is its documented
// contract), so a bare "recompute the same way from the same fields" oracle would agree
// with the drift and prove nothing. This one reds.
func TestFakShareOracleRedsOnDriftedPersistedEquiv(t *testing.T) {
	track1, track2 := healthyOracleCorpus()

	drifted := append([]SavingsRow(nil), track2...)
	var patched bool
	for i := range drifted {
		if drifted[i].Mechanism == "provider_prompt_cache" {
			// Double the stored provider saving while leaving cache_read /
			// cache_creation — the counters it was derived from — untouched.
			drifted[i].SavedTokenEquiv *= 2
			drifted[i].NetSavedTokenEquiv *= 2
			patched = true
		}
	}
	if !patched {
		t.Fatalf("precondition: the corpus must contain a provider_prompt_cache row to drift")
	}

	rep := FoldFleetBenefit(track1, drifted, nil, FleetBenefitOptions{})
	got := VerifyFakShare(rep, track1, drifted)
	if !got.Diverged {
		t.Fatalf("a persisted equiv that disagrees with its own raw counters must RED, got: %s", got.Finding)
	}
}

// TestFakShareOracleAgreesOnUndefinedCorpus pins the nil convention on both sides: an
// empty corpus has no positive saved token-equiv to divide by, so fak_share is UNDEFINED,
// and BOTH implementations must say so. A 0% here would be a measured-looking claim about
// a corpus that measured nothing.
func TestFakShareOracleAgreesOnUndefinedCorpus(t *testing.T) {
	rep := FoldFleetBenefit(nil, nil, nil, FleetBenefitOptions{})
	if rep.FakSharePct != nil {
		t.Fatalf("precondition: the primary must report UNDEFINED on an empty corpus, got %v", *rep.FakSharePct)
	}
	got := VerifyFakShare(rep, nil, nil)
	if got.Diverged || got.OraclePct != nil {
		t.Fatalf("empty corpus must agree on UNDEFINED, got diverged=%v oracle=%v (%s)", got.Diverged, got.OraclePct, got.Finding)
	}
	if _, ok := RecomputeFakSharePct(nil, nil); ok {
		t.Fatalf("RecomputeFakSharePct must report UNDEFINED on an empty corpus")
	}
}

// TestFakShareOracleRedsOnDefinednessMismatch closes the hole a bare numeric compare would
// leave: a primary that publishes a share where the raw ledgers support none (or the
// reverse) never reaches a subtraction, so it must be caught as a DEFINEDNESS divergence
// rather than silently passing on a nil.
func TestFakShareOracleRedsOnDefinednessMismatch(t *testing.T) {
	phantom := 42.0
	rep := FoldFleetBenefit(nil, nil, nil, FleetBenefitOptions{})
	rep.FakSharePct = &phantom

	got := VerifyFakShare(rep, nil, nil)
	if !got.Diverged || !got.DefinednessMismatch {
		t.Fatalf("a share published against an empty corpus must RED as a definedness mismatch, got %+v", got)
	}
	if !strings.Contains(got.Finding, "UNDEFINED") {
		t.Fatalf("the red must say which side was UNDEFINED, got %q", got.Finding)
	}
}
