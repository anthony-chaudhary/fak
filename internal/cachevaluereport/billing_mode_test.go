package cachevaluereport

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"
)

// w28ProviderRow is the shape of the headline W28 provider prompt-cache row: the reduction
// the fleet report prints is avoided/(spend+avoided), where avoided = rebate − write
// premium. These figures are chosen so that ratio is EXACTLY the 68.29% headline
// (6829/10000), which is what makes this a regression fixture rather than a re-derivation
// of whatever the live ledger happens to hold today.
func w28ProviderRow(billing string) SavingsRow {
	return SavingsRow{
		Date: "2026-07-11", Provider: "anthropic", Mechanism: "provider_prompt_cache",
		CacheReadTokens: 118867088, CacheCreationTokens: 8773566,
		SavedTokenEquiv: 104786987.7, NetSavedTokenEquiv: 104786987.7,
		RebateUSD: 7000, WritePremiumUSD: 171, SpendUSD: 3171,
		InputPerMTokUSD: 5, OutputPerMTokUSD: 25,
		PricingSource: "default:anthropic/claude-opus-4-8",
		BillingMode:   billing,
	}
}

func approxBilling(a, b float64) bool { return math.Abs(a-b) <= 1e-9 }

// TestBillingSplitAPIKeyReproducesW28Reduction pins the headline: an API-key-stamped W28
// provider row reproduces 68.29% in the blended lens AND in the real-$ column, and the
// notional column stays absent rather than claiming a 0% reduction it did not measure.
// Splitting the corpus must not move the number it splits.
func TestBillingSplitAPIKeyReproducesW28Reduction(t *testing.T) {
	rep := FoldFleetBenefit(nil, []SavingsRow{w28ProviderRow(BillingModeAPIKey)}, nil, FleetBenefitOptions{})

	if rep.ObservedAPICostReductionPct == nil || !approxBilling(*rep.ObservedAPICostReductionPct, 68.29) {
		t.Fatalf("blended W28 reduction = %v, want 68.29", pctStr(rep.ObservedAPICostReductionPct))
	}
	if rep.APIKeyReductionPct == nil || !approxBilling(*rep.APIKeyReductionPct, 68.29) {
		t.Fatalf("API-key column reduction = %v, want the same 68.29 headline", pctStr(rep.APIKeyReductionPct))
	}
	if rep.APIKeyRows != 1 || !approxBilling(rep.APIKeyAvoidedUSD, 6829) || !approxBilling(rep.APIKeyActualSpendUSD, 3171) {
		t.Fatalf("API-key column = %d row(s) avoided $%.4f spend $%.4f, want 1/6829/3171",
			rep.APIKeyRows, rep.APIKeyAvoidedUSD, rep.APIKeyActualSpendUSD)
	}
	// An empty column has no honest percentage: nil, not zero.
	if rep.NotionalRows != 0 || rep.NotionalReductionPct != nil {
		t.Fatalf("notional column = %d row(s) pct %v, want 0 rows and no percentage",
			rep.NotionalRows, pctStr(rep.NotionalReductionPct))
	}

	out := RenderFleetBenefit(rep)
	for _, want := range []string{
		"billing seat (OBSERVED both columns — list-priced projection, never a reconciled invoice; #3664):",
		"API-key (REAL-$; 1 priced row(s)): spend=$3171.0000 counterfactual=$10000.0000 avoided=$6829.0000 reduction=68.29%",
		"OAuth/unknown (NOTIONAL — flat-rate seat, no per-token invoice; 0 priced row(s)): spend=$0.0000 counterfactual=$0.0000 avoided=$0.0000 reduction=-",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("RenderFleetBenefit missing %q in:\n%s", want, out)
		}
	}
}

// TestBillingSplitOAuthContributesNotionalOnly is the attribution witness: an OAuth-billed
// row's dollars land in the notional column and NOWHERE in the real-$ one, while the two
// columns still sum exactly to the blended total they partition.
func TestBillingSplitOAuthContributesNotionalOnly(t *testing.T) {
	apiRow := w28ProviderRow(BillingModeAPIKey)
	oauthRow := w28ProviderRow(BillingModeOAuth)
	oauthRow.RebateUSD, oauthRow.WritePremiumUSD, oauthRow.SpendUSD = 500, 100, 200 // avoided 400

	rep := FoldFleetBenefit(nil, []SavingsRow{apiRow, oauthRow}, nil, FleetBenefitOptions{})

	if rep.APIKeyRows != 1 || !approxBilling(rep.APIKeyAvoidedUSD, 6829) {
		t.Fatalf("OAuth row leaked into the real-$ column: %d row(s) avoided $%.4f, want 1/6829",
			rep.APIKeyRows, rep.APIKeyAvoidedUSD)
	}
	if rep.NotionalRows != 1 || !approxBilling(rep.NotionalAvoidedUSD, 400) || !approxBilling(rep.NotionalActualSpendUSD, 200) {
		t.Fatalf("notional column = %d row(s) avoided $%.4f spend $%.4f, want 1/400/200",
			rep.NotionalRows, rep.NotionalAvoidedUSD, rep.NotionalActualSpendUSD)
	}
	// The split re-attributes; it never invents or drops dollars.
	if !approxBilling(rep.APIKeyAvoidedUSD+rep.NotionalAvoidedUSD, rep.ObservedAPICostAvoidedUSD) {
		t.Fatalf("columns %.4f + %.4f != blended avoided %.4f",
			rep.APIKeyAvoidedUSD, rep.NotionalAvoidedUSD, rep.ObservedAPICostAvoidedUSD)
	}
	if !approxBilling(rep.APIKeyActualSpendUSD+rep.NotionalActualSpendUSD, rep.ObservedActualSpendUSD) {
		t.Fatalf("columns %.4f + %.4f != blended spend %.4f",
			rep.APIKeyActualSpendUSD, rep.NotionalActualSpendUSD, rep.ObservedActualSpendUSD)
	}
	// The API-key column keeps its own denominator, so the blended figure moving does
	// not drag the real-$ headline with it.
	if rep.APIKeyReductionPct == nil || !approxBilling(*rep.APIKeyReductionPct, 68.29) {
		t.Fatalf("API-key reduction = %v, want 68.29 regardless of the OAuth row", pctStr(rep.APIKeyReductionPct))
	}
}

// TestBillingSplitUnstampedRowsFoldNotional is the back-compat fence: every row written
// before #3664 carries no billing mode at all, and must read as notional — never as
// real-dollar API-key money the ledger cannot prove.
func TestBillingSplitUnstampedRowsFoldNotional(t *testing.T) {
	rep := FoldFleetBenefit(nil, []SavingsRow{w28ProviderRow("")}, nil, FleetBenefitOptions{})
	if rep.APIKeyRows != 0 || rep.APIKeyReductionPct != nil || rep.APIKeyAvoidedUSD != 0 {
		t.Fatalf("unstamped legacy row counted as real-$: %d row(s) avoided $%.4f pct %v",
			rep.APIKeyRows, rep.APIKeyAvoidedUSD, pctStr(rep.APIKeyReductionPct))
	}
	if rep.NotionalRows != 1 || rep.NotionalReductionPct == nil || !approxBilling(*rep.NotionalReductionPct, 68.29) {
		t.Fatalf("unstamped legacy row = %d notional row(s) pct %v, want 1 at 68.29",
			rep.NotionalRows, pctStr(rep.NotionalReductionPct))
	}
}

func TestNormalizeBillingModeFailsSafeToUnknown(t *testing.T) {
	for _, tc := range []struct {
		raw        string
		want       string
		realDollar bool
	}{
		{"api_key", BillingModeAPIKey, true},
		{"  API_KEY  ", BillingModeAPIKey, true},
		{"oauth", BillingModeOAuth, false},
		{"OAuth", BillingModeOAuth, false},
		{"", BillingModeUnknown, false},
		{"unknown", BillingModeUnknown, false},
		{"subscription", BillingModeUnknown, false},
		{"api-key", BillingModeUnknown, false},
	} {
		if got := NormalizeBillingMode(tc.raw); got != tc.want {
			t.Fatalf("NormalizeBillingMode(%q) = %q, want %q", tc.raw, got, tc.want)
		}
		if got := RealDollarBillingMode(tc.raw); got != tc.realDollar {
			t.Fatalf("RealDollarBillingMode(%q) = %v, want %v", tc.raw, got, tc.realDollar)
		}
	}
}

// TestNewSavingsRowsStampsBillingMode covers the write side: a producer that names its seat
// stamps every emitted row, and one that does not leaves the key OFF the wire entirely —
// byte-identical to the ledger written before the field existed.
func TestNewSavingsRowsStampsBillingMode(t *testing.T) {
	obs := SavingsObservation{
		SessionType: "guard", Provider: "anthropic", Context: "claude",
		InputTokens: 100, CacheReadTokens: 2000, CacheCreationTokens: 100, OutputTokens: 10,
		CompactionShedTokens: 3000, CompactionFired: 1,
		BillingMode: " OAuth ",
		Pricing:     SavingsPricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25, Source: "test"},
	}
	now := time.Date(2026, 7, 11, 12, 0, 0, 0, time.UTC)

	rows := NewSavingsRows(obs, now)
	if len(rows) < 2 {
		t.Fatalf("want both mechanism rows, got %d", len(rows))
	}
	for _, row := range rows {
		if row.BillingMode != BillingModeOAuth {
			t.Fatalf("row %q billing mode = %q, want normalized %q", row.Mechanism, row.BillingMode, BillingModeOAuth)
		}
	}

	obs.BillingMode = ""
	for _, row := range NewSavingsRows(obs, now) {
		if row.BillingMode != "" {
			t.Fatalf("unnamed seat stamped %q, want blank", row.BillingMode)
		}
		line, err := AppendSavingsLine(row)
		if err != nil {
			t.Fatalf("marshal row: %v", err)
		}
		var raw map[string]any
		if err := json.Unmarshal([]byte(line), &raw); err != nil {
			t.Fatalf("unmarshal row: %v", err)
		}
		if _, present := raw["billing_mode"]; present {
			t.Fatalf("blank seat must not emit billing_mode at all: %s", line)
		}
	}
}
