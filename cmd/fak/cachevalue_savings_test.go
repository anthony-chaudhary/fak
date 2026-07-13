package main

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func TestAppendObservedCacheSavingsReportsRowsAndErrors(t *testing.T) {
	clearCachevaluePriceEnv(t)
	now := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)
	path := filepath.Join(t.TempDir(), "cache-savings.jsonl")
	sum := gateway.AdjudicationSummary{
		InputTokens:          20,
		CachedPromptTokens:   60,
		CacheCreationTokens:  20,
		OutputTokens:         3,
		CompactionShedTokens: 9,
	}

	res := appendObservedCacheSavingsTo(path, "guard", "anthropic", "claude", sum, now)
	if res.Err != nil {
		t.Fatalf("append savings returned error: %v", res.Err)
	}
	if res.RowsPlanned != 2 || res.RowsWritten != 2 {
		t.Fatalf("rows planned/written = %d/%d, want 2/2", res.RowsPlanned, res.RowsWritten)
	}
	if res.ProviderTokenEquiv != 49 || res.FakCompactionTokenEquiv != 9 || res.TotalTokenEquiv != 58 {
		t.Fatalf("token-equiv split = provider %.1f fak %.1f total %.1f, want 49/9/58",
			res.ProviderTokenEquiv, res.FakCompactionTokenEquiv, res.TotalTokenEquiv)
	}
	rows := cachevaluereport.ReadSavingsLedgerFile(path)
	if len(rows) != 2 {
		t.Fatalf("ledger rows = %d, want 2", len(rows))
	}
	if rows[0].Mechanism != "provider_prompt_cache" || rows[1].Mechanism != "compaction_shed" {
		t.Fatalf("mechanisms = %q/%q, want provider_prompt_cache/compaction_shed", rows[0].Mechanism, rows[1].Mechanism)
	}
	for i, row := range rows {
		if row.InputPerMTokUSD != gateway.ClaudeOpus48InputPerMTokUSD || row.OutputPerMTokUSD != gateway.ClaudeOpus48OutputPerMTokUSD {
			t.Fatalf("row %d default pricing = %.2f/%.2f, want %.2f/%.2f",
				i, row.InputPerMTokUSD, row.OutputPerMTokUSD,
				gateway.ClaudeOpus48InputPerMTokUSD, gateway.ClaudeOpus48OutputPerMTokUSD)
		}
		if row.PricingSource != gateway.CachePricingSourceAnthropicClaudeOpus48 {
			t.Fatalf("row %d pricing source = %q, want %q", i, row.PricingSource, gateway.CachePricingSourceAnthropicClaudeOpus48)
		}
		if row.DollarStatus == cachevaluereport.SavingsDollarStatusBlind {
			t.Fatalf("row %d should be priced by default, got dollar-blind: %+v", i, row)
		}
		if row.NetUSD == 0 {
			t.Fatalf("row %d default-pricing net_usd must not be silently zero: %+v", i, row)
		}
	}

	badPath := filepath.Join(t.TempDir(), "missing", "cache-savings.jsonl")
	failed := appendObservedCacheSavingsTo(badPath, "guard", "anthropic", "claude", sum, now)
	if failed.Err == nil {
		t.Fatal("append to a missing parent must report an error")
	}
	if failed.RowsPlanned != 2 || failed.RowsWritten != 0 {
		t.Fatalf("failed rows planned/written = %d/%d, want 2/0", failed.RowsPlanned, failed.RowsWritten)
	}
}

// TestAppendObservedCacheSavingsThreadsUpgradedCreationTokens is the #2179 closing
// witness: gateway.AdjudicationSummary.CacheCreationTokensUpgraded (the gateway's
// per-turn managed-cache 1h TTL-upgrade attribution) must reach the durable Track-2
// ledger row, not just live inside the gateway's own in-memory summary. Before this
// wire, appendObservedCacheSavingsTo dropped the field on the floor and every
// session's cache-write was priced at the flat 5m tier regardless of the upgrade.
func TestAppendObservedCacheSavingsThreadsUpgradedCreationTokens(t *testing.T) {
	clearCachevaluePriceEnv(t)
	now := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)
	path := filepath.Join(t.TempDir(), "cache-savings.jsonl")
	sum := gateway.AdjudicationSummary{
		InputTokens:                 20,
		CachedPromptTokens:          60,
		CacheCreationTokens:         2000,
		CacheCreationTokensUpgraded: 1200, // gateway-attributed: 1200 of 2000 written at the 1h tier
		OutputTokens:                3,
	}

	res := appendObservedCacheSavingsTo(path, "guard", "anthropic", "claude", sum, now)
	if res.Err != nil {
		t.Fatalf("append savings returned error: %v", res.Err)
	}
	rows := cachevaluereport.ReadSavingsLedgerFile(path)
	if len(rows) != 1 || rows[0].Mechanism != "provider_prompt_cache" {
		t.Fatalf("want one provider_prompt_cache row, got %+v", rows)
	}
	row := rows[0]
	if row.CacheCreationTokensUpgraded != 1200 {
		t.Fatalf("ledger row CacheCreationTokensUpgraded = %d, want 1200 (the gateway summary's attribution)", row.CacheCreationTokensUpgraded)
	}
	if row.CacheCreationTierProvenance != cachevaluereport.CacheCreationTierProvenanceGatewayAttributed {
		t.Fatalf("ledger row CacheCreationTierProvenance = %q, want %q", row.CacheCreationTierProvenance, cachevaluereport.CacheCreationTierProvenanceGatewayAttributed)
	}
	// write premium = 1200*(2.0-1) + 800*(1.25-1) = 1400 token-equiv, priced at
	// $5/MTok = 0.007. A dropped-on-the-floor field would instead price the whole
	// 2000 at flat 1.25x = 500 token-equiv (0.0025) — the #2179 under-count.
	if !approxSavingsTest(row.WritePremiumUSD, 0.007) {
		t.Fatalf("WritePremiumUSD = %.6f, want 0.007 (the ledger must see the gateway's 1h attribution)", row.WritePremiumUSD)
	}
}

func approxSavingsTest(a, b float64) bool { return math.Abs(a-b) <= 1e-9 }

// TestAppendObservedCacheSavingsThreadsCompactionCacheReadWitness is the closing
// witness for the durable-vs-live shed-valuation divergence: the warm witness
// gateway.AdjudicationSummary.CompactionCacheReadTokens must reach the Track-2 ledger
// row so cacheprice.ShedTokenEquiv prices a WARM shed at the cache-read marginal (0.1x),
// exactly as the live split (cache_pricing.go) and guard banner already do. Before this
// wire, appendObservedCacheSavingsTo dropped the field, so the durable row always saw
// warmWitness=0 and booked every shed at FULL_INPUT (1.0x) — the ~10x over-valuation
// #2794/#2798 corrected everywhere EXCEPT this producer, so the same `fak guard` exit
// printed the shed two ways (blended banner vs full-input persistence line) that disagreed
// ~10x on a warm session. The parallel of the #2179 CacheCreationTokensUpgraded wire above.
func TestAppendObservedCacheSavingsThreadsCompactionCacheReadWitness(t *testing.T) {
	clearCachevaluePriceEnv(t)
	now := time.Date(2026, 7, 13, 12, 0, 0, 0, time.UTC)
	path := filepath.Join(t.TempDir(), "cache-savings.jsonl")
	// Fully-warm session: the observed provider cache_read at the fires (4000) exceeds the
	// shed (1000), so all 1000 shed tokens were being served cheaply — worth the 0.1x
	// read marginal, not full input.
	sum := gateway.AdjudicationSummary{
		InputTokens:               20,
		CachedPromptTokens:        60,
		CacheCreationTokens:       20,
		OutputTokens:              3,
		CompactionShedTokens:      1000,
		CompactionCacheReadTokens: 4000,
		CompactionFired:           9,
	}

	res := appendObservedCacheSavingsTo(path, "guard", "anthropic", "claude", sum, now)
	if res.Err != nil {
		t.Fatalf("append savings returned error: %v", res.Err)
	}
	// 1000 warm shed tokens * 0.1x = 100 token-equiv. A dropped witness would instead book
	// the shed at full input = 1000 token-equiv — the ~10x over-count this wire closes.
	if !approxSavingsTest(res.FakCompactionTokenEquiv, 100) {
		t.Fatalf("FakCompactionTokenEquiv = %.4f, want 100 (warm shed at the 0.1x cache-read marginal); a value near 1000 means the warm witness was dropped and the shed was booked at FULL_INPUT",
			res.FakCompactionTokenEquiv)
	}

	rows := cachevaluereport.ReadSavingsLedgerFile(path)
	var compaction *cachevaluereport.SavingsRow
	for i := range rows {
		if rows[i].Mechanism == "compaction_shed" {
			compaction = &rows[i]
			break
		}
	}
	if compaction == nil {
		t.Fatalf("warm session must write a compaction_shed row; got %d rows: %+v", len(rows), rows)
	}
	if compaction.CompactionCacheReadTokens != 4000 {
		t.Fatalf("ledger row CompactionCacheReadTokens = %d, want 4000 (the gateway summary's warm witness must be threaded, not dropped)", compaction.CompactionCacheReadTokens)
	}
	if compaction.ValuationBasis != cachevaluereport.ValuationBasisCacheReadMarginal {
		t.Fatalf("warm shed valuation_basis = %q, want %q; FULL_INPUT here means the witness was dropped and the shed over-valued ~10x",
			compaction.ValuationBasis, cachevaluereport.ValuationBasisCacheReadMarginal)
	}
}

func TestAppendObservedCacheSavingsEnvOverridesDefaultPricing(t *testing.T) {
	t.Setenv(cachevalueInputPriceEnv, "7")
	t.Setenv(cachevalueOutputPriceEnv, "11")
	now := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)
	path := filepath.Join(t.TempDir(), "cache-savings.jsonl")
	sum := gateway.AdjudicationSummary{
		InputTokens:          20,
		CachedPromptTokens:   60,
		CacheCreationTokens:  20,
		OutputTokens:         3,
		CompactionShedTokens: 9,
	}

	res := appendObservedCacheSavingsTo(path, "guard", "anthropic", "claude", sum, now)
	if res.Err != nil {
		t.Fatalf("append savings returned error: %v", res.Err)
	}
	rows := cachevaluereport.ReadSavingsLedgerFile(path)
	if len(rows) != 2 {
		t.Fatalf("ledger rows = %d, want 2", len(rows))
	}
	for i, row := range rows {
		if row.InputPerMTokUSD != 7 || row.OutputPerMTokUSD != 11 {
			t.Fatalf("row %d env pricing = %.2f/%.2f, want 7/11", i, row.InputPerMTokUSD, row.OutputPerMTokUSD)
		}
		if row.PricingSource != cachevalueEnvPricingSource {
			t.Fatalf("row %d pricing source = %q, want %q", i, row.PricingSource, cachevalueEnvPricingSource)
		}
	}
	if got, want := rows[0].RebateUSD, 60*0.9*7/1_000_000.0; math.Abs(got-want) > 1e-12 {
		t.Fatalf("env-priced rebate_usd = %.12f, want %.12f", got, want)
	}
	if rows[0].DollarStatus == cachevaluereport.SavingsDollarStatusBlind {
		t.Fatalf("env-priced row should not be dollar-blind: %+v", rows[0])
	}
}

func TestFormatCacheValuePersistenceSummaryNamesEvidenceAndNextCommand(t *testing.T) {
	rep := cacheValuePersistenceReport{
		Since: "2026-06-30",
		Track1: cacheValueTrack1Result{
			Path:         "docs/nightrun/cache-value.jsonl",
			RowsPlanned:  1,
			RowsWritten:  1,
			Turns:        12,
			PromptTokens: 2000,
			ReusedTokens: 1500,
		},
		Track2: cacheValueTrack2Result{
			Path:                    "docs/nightrun/cache-savings.jsonl",
			RowsPlanned:             2,
			RowsWritten:             2,
			ProviderTokenEquiv:      1200,
			FakCompactionTokenEquiv: 300,
			TotalTokenEquiv:         1500,
			CacheReadTokens:         1400,
			CompactionShedTokens:    300,
		},
	}
	out := formatCacheValuePersistenceSummary("fak guard", rep)
	for _, want := range []string{
		"fak guard: cache-value evidence",
		"Track 1 WITNESSED kernel row",
		"Track 2 OBSERVED-$ rows",
		"provider +1,200 tok-eq",
		"fak compaction +300 tok-eq",
		"total +1,500 tok-eq",
		"docs/nightrun/cache-value.jsonl",
		"docs/nightrun/cache-savings.jsonl",
		"fak cachevalue report --since 2026-06-30",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("summary missing %q:\n%s", want, out)
		}
	}
}

// TestCacheValueSummaryWiredIntoGuardExit is the anti-regression witness for the
// detection-without-enforcement gap this wire closed. formatCacheValuePersistenceSummary
// was built and unit-tested but referenced by nothing on a live path, so a finished
// guard session persisted its savings silently and never showed the operator the "usage
// savings moment". This asserts the guard exit both BUILDS the report and FORMATS it —
// and fails the moment either reference is removed, re-orphaning the formatter. Matches
// the source-assertion idiom of servewiring_test.go (which guards the same class of
// dead-wiring regression on the serve path).
func TestCacheValueSummaryWiredIntoGuardExit(t *testing.T) {
	root := repoRootFromTest(t)
	body, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "guard_child.go"))
	if err != nil {
		t.Fatalf("read guard_child.go: %v", err)
	}
	src := string(body)
	for _, want := range []string{
		"buildCacheValuePersistenceReport(",
		`formatCacheValuePersistenceSummary("fak guard"`,
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("guard_child.go no longer surfaces the cache-value savings summary at session exit (missing %q); the dollar-aware formatter is orphaned again", want)
		}
	}
}

func clearCachevaluePriceEnv(t *testing.T) {
	t.Helper()
	t.Setenv(cachevalueInputPriceEnv, "")
	t.Setenv(cachevalueOutputPriceEnv, "")
}

func TestFormatCacheValuePersistenceSummaryMakesNoEvidenceExplicit(t *testing.T) {
	out := formatCacheValuePersistenceSummary("fak guard", cacheValuePersistenceReport{
		Since: "2026-06-30",
		Track1: cacheValueTrack1Result{
			Path: "docs/nightrun/cache-value.jsonl",
		},
		Track2: cacheValueTrack2Result{
			Path: "docs/nightrun/cache-savings.jsonl",
		},
	})
	for _, want := range []string{
		"cache-value evidence",
		"Track 1 WITNESSED kernel row not written",
		"no KV-prefix reuse turns",
		"Track 2 OBSERVED-$ row not written",
		"no provider-cache or compaction tokens",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("no-evidence summary missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "fak cachevalue report --since") {
		t.Fatalf("no-evidence summary should not point at an empty report command:\n%s", out)
	}
}

// TestAppendObservedCacheSavingsPersistsCompactionHealthOnFiredButZeroShed is the
// end-to-end #2039 witness: a guard session where the compaction lever FIRED but
// shed NOTHING (the anchor-starved case #1407) must still write a durable
// compaction_shed row carrying the health fields (fired>0, starved>0, shed=0).
func TestAppendObservedCacheSavingsPersistsCompactionHealthOnFiredButZeroShed(t *testing.T) {
	clearCachevaluePriceEnv(t)
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	path := filepath.Join(t.TempDir(), "cache-savings.jsonl")
	sum := gateway.AdjudicationSummary{
		InputTokens:             164_000,
		CachedPromptTokens:      12_500_000,
		CacheCreationTokens:     500_000,
		OutputTokens:            3_000,
		CompactionShedTokens:    0,
		CompactionFired:         0,
		CompactionBailed:        5,
		CompactionAnchorStarved: 5,
		CompactionBudget:        48000,
	}

	res := appendObservedCacheSavingsTo(path, "guard", "anthropic", "claude", sum, now)
	if res.Err != nil {
		t.Fatalf("append savings returned error: %v", res.Err)
	}

	rows := cachevaluereport.ReadSavingsLedgerFile(path)
	var compaction *cachevaluereport.SavingsRow
	for i := range rows {
		if rows[i].Mechanism == "compaction_shed" {
			compaction = &rows[i]
			break
		}
	}
	if compaction == nil {
		t.Fatalf("fired-but-zero-shed session must write a compaction_shed row; got %d rows: %+v", len(rows), rows)
	}
	if compaction.CompactionShedTokens != 0 {
		t.Fatalf("shed must be 0: %d", compaction.CompactionShedTokens)
	}
	if compaction.CompactionAnchorStarved != 5 {
		t.Fatalf("anchor_starved must be persisted as 5: %d", compaction.CompactionAnchorStarved)
	}
	if compaction.CompactionBudget != 48000 {
		t.Fatalf("budget must be persisted: %d", compaction.CompactionBudget)
	}
}
