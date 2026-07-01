package cachevaluereport

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
)

// now is a fixed clock so GeneratedAt is deterministic across the recompute runs.
var twoTrackNow = time.Date(2026, 6, 29, 12, 0, 0, 0, time.UTC)

// track1Fixture is a two-week WITNESSED kernel ledger trending up (60% -> 80%
// realized reuse), the same shape the CLI dry-run test uses.
func track1Fixture() []cachevalueledger.Row {
	return []cachevalueledger.Row{
		{Date: "2026-06-15", SessionType: "guard", Turns: 10, PromptTokens: 1000, ReusedTokens: 600},
		{Date: "2026-06-22", SessionType: "guard", Turns: 10, PromptTokens: 1000, ReusedTokens: 800},
	}
}

// track2Fixture is a two-week OBSERVED-$ ledger that starts NEGATIVE (a cold
// write-heavy first week: write premium + spend exceed the rebate) and crosses
// break-even in the second week once reads repay the writes. The honest sign is
// preserved per #1303 (no floor at zero).
func track2Fixture() []SavingsRow {
	return []SavingsRow{
		// Week 1: cold writes dominate -> NET negative.
		{
			Date: "2026-06-15", SessionType: "guard",
			InputTokens: 2000, CacheCreationTokens: 8000, OutputTokens: 500,
			SavedTokenEquiv: 1000, NetSavedTokenEquiv: 1000,
			RebateUSD: 0.50, WritePremiumUSD: 2.00, SpendUSD: 1.00, CompactionSavedUSD: 0.25,
		},
		// Week 2: reads repay -> NET positive, cumulative crosses break-even.
		{
			Date: "2026-06-22", SessionType: "guard",
			InputTokens: 1000, CacheReadTokens: 9000, OutputTokens: 500,
			SavedTokenEquiv: 8100, NetSavedTokenEquiv: 8100,
			RebateUSD: 5.00, WritePremiumUSD: 0.10, SpendUSD: 0.50, CompactionSavedUSD: 0.40,
		},
	}
}

// TestFoldTwoTrackRecomputeIsByteForByte is the #1304 witness: the report
// reproduces byte-for-byte from folding the two ledgers. Re-folding the SAME
// rows must yield JSON-identical reports (the fold is pure: no clock beyond the
// supplied `now`, no map-iteration nondeterminism leaking into output order).
func TestFoldTwoTrackRecomputeIsByteForByte(t *testing.T) {
	t1, t2 := track1Fixture(), track2Fixture()

	a := FoldTwoTrack(t1, t2, twoTrackNow)
	b := FoldTwoTrack(t1, t2, twoTrackNow)

	ja, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	jb, err := json.Marshal(b)
	if err != nil {
		t.Fatal(err)
	}
	if string(ja) != string(jb) {
		t.Fatalf("two-track fold is not deterministic:\n a=%s\n b=%s", ja, jb)
	}

	// The rendered table must likewise reproduce from the same fold.
	if RenderTwoTrack(a) != RenderTwoTrack(b) {
		t.Fatal("RenderTwoTrack is not reproducible from an identical fold")
	}
}

// TestFoldTwoTrackNetReDerivesFromComponents asserts every NET is exactly its
// component accounts (rebate + compaction − write premium − spend), per row and
// per folded bucket — the P&L identity the honesty fence rests on.
func TestFoldTwoTrackNetReDerivesFromComponents(t *testing.T) {
	rows := track2Fixture()
	for i, r := range rows {
		// A row with stored NetUSD must equal its computed parts; the fixture leaves
		// NetUSD zero, so NetUSDComputed is the source of truth the bucket folds.
		want := r.RebateUSD + r.CompactionSavedUSD - r.WritePremiumUSD - r.SpendUSD
		if got := r.NetUSDComputed(); math.Abs(got-want) > 1e-9 {
			t.Fatalf("row %d NetUSDComputed=%.6f want %.6f", i, got, want)
		}
	}

	rep := FoldTwoTrack(track1Fixture(), rows, twoTrackNow)
	if len(rep.Track2) != 2 {
		t.Fatalf("want 2 Track-2 buckets, got %d", len(rep.Track2))
	}

	var cumulative float64
	for i, b := range rep.Track2 {
		wantNet := b.RebateUSD + b.CompactionSavedUSD - b.WritePremiumUSD - b.SpendUSD
		if math.Abs(b.NetUSD-wantNet) > 1e-9 {
			t.Fatalf("bucket %d NetUSD=%.6f want %.6f (re-derived from components)", i, b.NetUSD, wantNet)
		}
		cumulative += b.NetUSD
		if math.Abs(b.CumulativeNetUSD-cumulative) > 1e-9 {
			t.Fatalf("bucket %d CumulativeNetUSD=%.6f want %.6f", i, b.CumulativeNetUSD, cumulative)
		}
	}
}

// TestFoldTwoTrackBreakEvenCrossing asserts the running cumulative starts below
// break-even (week 1 net is negative) and crosses it in week 2 — and that the
// crossing is shown explicitly, not hidden behind a gross headline.
func TestFoldTwoTrackBreakEvenCrossing(t *testing.T) {
	rep := FoldTwoTrack(track1Fixture(), track2Fixture(), twoTrackNow)

	w1 := rep.Track2[0]
	if w1.Provider != "unknown_provider" || w1.Mechanism != "provider_prompt_cache" {
		t.Fatalf("legacy Track-2 dimensions = %s/%s, want unknown_provider/provider_prompt_cache", w1.Provider, w1.Mechanism)
	}
	if w1.NetUSD >= 0 {
		t.Fatalf("week 1 should be NET negative (cold writes), got %.4f", w1.NetUSD)
	}
	if w1.BrokeEven {
		t.Fatalf("week 1 must not have broken even, cumulative=%.4f", w1.CumulativeNetUSD)
	}
	w2 := rep.Track2[1]
	if !w2.BrokeEven {
		t.Fatalf("week 2 should cross break-even, cumulative=%.4f", w2.CumulativeNetUSD)
	}
	if !rep.BrokeEven {
		t.Fatal("report headline should report broke-even once the running total crosses zero")
	}

	out := RenderTwoTrack(rep)
	if !strings.Contains(out, "break-even") {
		t.Fatalf("rendered P&L must show the break-even crossing explicitly:\n%s", out)
	}
	if !strings.Contains(out, "net$") {
		t.Fatalf("rendered P&L must carry an explicit NET line:\n%s", out)
	}
	if !strings.Contains(out, "provider") || !strings.Contains(out, "mechanism") {
		t.Fatalf("rendered P&L must carry provider/mechanism columns:\n%s", out)
	}
	if !strings.Contains(out, "fak_teq") {
		t.Fatalf("rendered P&L must expose fak-authored token-equivalent attribution:\n%s", out)
	}
}

func TestFoldSavingsSplitsByProviderAndMechanism(t *testing.T) {
	rows := []SavingsRow{
		{Date: "2026-06-22", Provider: "anthropic", Mechanism: "provider_prompt_cache", CacheReadTokens: 100, RebateUSD: 1.00},
		{Date: "2026-06-22", Provider: "openai", Mechanism: "provider_prompt_cache", CacheReadTokens: 200, RebateUSD: 2.00},
		{Date: "2026-06-22", Provider: "anthropic", Mechanism: "compaction_shed", CompactionShedTokens: 300, CompactionSavedUSD: 3.00},
	}
	buckets := foldSavings(rows)
	if len(buckets) != 3 {
		t.Fatalf("want 3 provider/mechanism buckets, got %d: %+v", len(buckets), buckets)
	}
	got := map[string]SavingsBucket{}
	for _, b := range buckets {
		got[b.Provider+"/"+b.Mechanism] = b
	}
	for _, key := range []string{"anthropic/provider_prompt_cache", "openai/provider_prompt_cache", "anthropic/compaction_shed"} {
		if _, ok := got[key]; !ok {
			t.Fatalf("missing bucket %s in %+v", key, buckets)
		}
	}
	if got["anthropic/compaction_shed"].CompactionShedTokens != 300 {
		t.Fatalf("compaction bucket did not retain shed tokens: %+v", got["anthropic/compaction_shed"])
	}
}

func TestFoldTwoTrackOwnerAttributionSeparatesProviderAndFakTokens(t *testing.T) {
	track1 := []cachevalueledger.Row{
		{Date: "2026-06-22", SessionType: "guard", Turns: 10, PromptTokens: 1000, ReusedTokens: 800},
	}
	track2 := []SavingsRow{
		{
			Date: "2026-06-22", Provider: "anthropic", Mechanism: "provider_prompt_cache",
			CacheReadTokens: 1000, SavedTokenEquiv: 900, NetSavedTokenEquiv: 900, RebateUSD: 1,
		},
		{
			Date: "2026-06-22", Provider: "fak", Mechanism: "compaction_shed",
			CompactionShedTokens: 300, SavedTokenEquiv: 300, NetSavedTokenEquiv: 300, CompactionSavedUSD: 0.3,
		},
	}
	rep := FoldTwoTrack(track1, track2, twoTrackNow)
	if len(rep.OwnerAttribution) != 1 {
		t.Fatalf("want one owner-attribution bucket, got %d: %+v", len(rep.OwnerAttribution), rep.OwnerAttribution)
	}
	got := rep.OwnerAttribution[0]
	if got.ProviderPromptCacheTokenEquiv != 900 {
		t.Fatalf("provider prompt-cache token-equiv = %.1f, want 900", got.ProviderPromptCacheTokenEquiv)
	}
	if got.FakKVPrefixReusedTokens != 800 || got.FakCompactionShedTokens != 300 {
		t.Fatalf("fak mechanism tokens not decomposed: %+v", got)
	}
	if got.FakAuthoredTokenEquiv != 1100 {
		t.Fatalf("fak-authored token-equiv = %.1f, want 1100", got.FakAuthoredTokenEquiv)
	}
	out := RenderTwoTrack(rep)
	for _, want := range []string{"Owner attribution", "provider_teq", "fak_teq", "kv_tok", "compact_tok", "900", "1100"} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q:\n%s", want, out)
		}
	}
}

func TestNewSavingsRowsSplitsProviderAndCompaction(t *testing.T) {
	rows := NewSavingsRows(SavingsObservation{
		SessionType:          "guard",
		Provider:             "anthropic",
		Context:              "claude",
		InputTokens:          1000,
		CacheReadTokens:      10_000,
		CacheCreationTokens:  2000,
		OutputTokens:         500,
		CompactionShedTokens: 3000,
		Pricing:              SavingsPricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25},
	}, twoTrackNow)

	if len(rows) != 2 {
		t.Fatalf("want provider + compaction rows, got %d: %+v", len(rows), rows)
	}
	provider, compaction := rows[0], rows[1]
	if provider.Provider != "anthropic" || provider.Mechanism != "provider_prompt_cache" {
		t.Fatalf("provider row dimensions = %s/%s", provider.Provider, provider.Mechanism)
	}
	if compaction.Provider != "fak" || compaction.Mechanism != "compaction_shed" {
		t.Fatalf("compaction row dimensions = %s/%s", compaction.Provider, compaction.Mechanism)
	}
	if !approxTrack2(provider.SavedTokenEquiv, 8500) {
		t.Fatalf("provider saved token-equiv = %.4f, want 8500", provider.SavedTokenEquiv)
	}
	if !approxTrack2(provider.RebateUSD, 0.045) || !approxTrack2(provider.WritePremiumUSD, 0.0025) {
		t.Fatalf("provider dollars not priced from observed axes: %+v", provider)
	}
	if !approxTrack2(provider.SpendUSD, 0.035) {
		t.Fatalf("provider spend = %.6f, want 0.035", provider.SpendUSD)
	}
	if compaction.CompactionShedTokens != 3000 || !approxTrack2(compaction.CompactionSavedUSD, 0.015) {
		t.Fatalf("compaction row did not price shed tokens: %+v", compaction)
	}
}

func TestNewSavingsRowsMarksDollarBlindWithoutPricing(t *testing.T) {
	rows := NewSavingsRows(SavingsObservation{
		SessionType:         "guard",
		Provider:            "openai",
		Context:             "codex",
		CacheReadTokens:     10_000,
		CacheCreationTokens: 1000,
		Pricing:             SavingsPricing{DollarBlind: true, Source: "none"},
	}, twoTrackNow)
	if len(rows) != 1 {
		t.Fatalf("want one provider row, got %d: %+v", len(rows), rows)
	}
	if rows[0].DollarStatus != SavingsDollarStatusBlind {
		t.Fatalf("missing dollar-blind marker on unpriced row: %+v", rows[0])
	}
	if rows[0].PricingSource != "none" {
		t.Fatalf("pricing source = %q, want none", rows[0].PricingSource)
	}
	if rows[0].RebateUSD != 0 || rows[0].WritePremiumUSD != 0 || rows[0].SpendUSD != 0 || rows[0].NetUSD != 0 {
		t.Fatalf("unpriced row dollar fields should stay placeholders at zero: %+v", rows[0])
	}

	line, err := AppendSavingsLine(rows[0])
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(line, `"dollar_status":"dollar_blind"`) {
		t.Fatalf("ledger line must carry the dollar-blind marker: %s", line)
	}

	rep := FoldTwoTrack(nil, rows, twoTrackNow)
	if rep.BrokeEven {
		t.Fatalf("dollar-blind zero dollars must not be reported as break-even: %+v", rep)
	}
	if rep.DollarBlindRows != 1 || len(rep.Track2) != 1 || rep.Track2[0].DollarStatus != SavingsDollarStatusBlind {
		t.Fatalf("fold did not carry dollar-blind status: %+v", rep)
	}
	out := RenderTwoTrack(rep)
	if !strings.Contains(out, "dollar-blind") || !strings.Contains(out, "zero dollar fields are placeholders") {
		t.Fatalf("render should make dollar-blind rows explicit:\n%s", out)
	}
}

func TestNewSavingsRowsSkipsProviderWithoutCacheCounters(t *testing.T) {
	rows := NewSavingsRows(SavingsObservation{
		SessionType:  "serve",
		Provider:     "openai",
		Context:      "http",
		InputTokens:  1000,
		OutputTokens: 200,
		Pricing:      SavingsPricing{InputPerMTokUSD: 3, OutputPerMTokUSD: 15},
	}, twoTrackNow)
	if len(rows) != 0 {
		t.Fatalf("pure input/output spend is not provider-cache evidence; got rows: %+v", rows)
	}
}

func TestAppendSavingsRoundTripsRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache-savings.jsonl")
	rows := NewSavingsRows(SavingsObservation{
		SessionType:     "serve",
		Provider:        "openai",
		Context:         "http",
		CacheReadTokens: 100,
		Pricing:         SavingsPricing{InputPerMTokUSD: 3, OutputPerMTokUSD: 15},
	}, twoTrackNow)
	if len(rows) != 1 {
		t.Fatalf("want one provider row, got %d", len(rows))
	}
	if err := AppendSavings(path, rows[0]); err != nil {
		t.Fatalf("AppendSavings: %v", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read append file: %v", err)
	}
	got := ParseSavingsLedger(string(raw))
	if len(got) != 1 {
		t.Fatalf("want one parsed row, got %d from %q", len(got), string(raw))
	}
	if got[0].Provider != "openai" || got[0].Mechanism != "provider_prompt_cache" || got[0].CacheReadTokens != 100 {
		t.Fatalf("parsed row lost dimensions/tokens: %+v", got[0])
	}
}

func approxTrack2(a, b float64) bool { return math.Abs(a-b) <= 1e-9 }

// TestFoldTwoTrackProvenanceNeverBlended is the honesty-fence witness: Track 1
// keeps its WITNESSED self-labels and Track 2 carries the OBSERVED projection
// fence; the two are separate fields, never a blended number.
func TestFoldTwoTrackProvenanceNeverBlended(t *testing.T) {
	rep := FoldTwoTrack(track1Fixture(), track2Fixture(), twoTrackNow)

	if !rep.Track1.VsNaiveMultipleExcluded {
		t.Fatal("Track 1 must keep the #1066 vs-naive-excluded fence")
	}
	if rep.Track1.PublishableValueFamily != PublishableValueFamily {
		t.Fatalf("Track 1 lost its WITNESSED value family: %q", rep.Track1.PublishableValueFamily)
	}
	if !strings.Contains(rep.ProjectionFence, "cost projection over labelled sources") ||
		!strings.Contains(rep.ProjectionFence, "never blended") {
		t.Fatalf("report missing the OBSERVED projection / never-blended fence: %q", rep.ProjectionFence)
	}
	// The verdict is MEASURED only when both tracks carry evidence.
	if rep.Verdict != "MEASURED" {
		t.Fatalf("both tracks have evidence; verdict should be MEASURED, got %q", rep.Verdict)
	}
}

// TestFoldTwoTrackEmptyTrack2IsHonest checks that with no OBSERVED-$ rows the
// report still folds Track 1 and says Track 2 is empty (rung B not appending yet),
// rather than fabricating a $ number.
func TestFoldTwoTrackEmptyTrack2IsHonest(t *testing.T) {
	rep := FoldTwoTrack(track1Fixture(), nil, twoTrackNow)
	if len(rep.Track2) != 0 {
		t.Fatalf("empty Track-2 ledger should fold to no buckets, got %d", len(rep.Track2))
	}
	if rep.BrokeEven {
		t.Fatal("no OBSERVED-$ rows cannot have broken even")
	}
	if !strings.Contains(rep.NextAction, "#1303") {
		t.Fatalf("empty Track 2 should point at rung B (#1303): %q", rep.NextAction)
	}
	out := RenderTwoTrack(rep)
	if !strings.Contains(out, "no OBSERVED-$ rows yet") {
		t.Fatalf("render should say Track 2 is empty:\n%s", out)
	}
}

// TestParseSavingsLedgerRoundTrips checks the durable JSONL shape: a row appended
// with AppendSavingsLine parses back via ParseSavingsLedger to the same values.
func TestParseSavingsLedgerRoundTrips(t *testing.T) {
	row := SavingsRow{
		Date: "2026-06-22", SessionType: "guard",
		InputTokens: 1000, CacheReadTokens: 9000, CacheCreationTokens: 100, OutputTokens: 500,
		CompactionShedTokens: 1200,
		RebateUSD:            5.0, WritePremiumUSD: 0.1, SpendUSD: 0.5, CompactionSavedUSD: 0.4,
		NetUSD: 4.8,
	}
	line, err := AppendSavingsLine(row)
	if err != nil {
		t.Fatal(err)
	}
	got := ParseSavingsLedger(line + "\n")
	if len(got) != 1 {
		t.Fatalf("want 1 parsed row, got %d", len(got))
	}
	if got[0].Schema != SavingsLedgerSchema {
		t.Fatalf("AppendSavingsLine should stamp the schema, got %q", got[0].Schema)
	}
	if got[0].CacheReadTokens != 9000 || got[0].CompactionShedTokens != 1200 {
		t.Fatalf("round-trip dropped token axes: %+v", got[0])
	}
	if got[0].Provider != "unknown_provider" || got[0].Mechanism != "provider_prompt_cache" {
		t.Fatalf("round-trip dimensions = %s/%s, want unknown_provider/provider_prompt_cache", got[0].Provider, got[0].Mechanism)
	}
	if math.Abs(got[0].NetUSDComputed()-(5.0+0.4-0.1-0.5)) > 1e-9 {
		t.Fatalf("round-trip NET mismatch: %.6f", got[0].NetUSDComputed())
	}
}
