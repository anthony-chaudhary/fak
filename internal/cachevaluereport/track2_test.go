package cachevaluereport

import (
	"encoding/json"
	"math"
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
			RebateUSD: 0.50, WritePremiumUSD: 2.00, SpendUSD: 1.00, CompactionSavedUSD: 0.25,
		},
		// Week 2: reads repay -> NET positive, cumulative crosses break-even.
		{
			Date: "2026-06-22", SessionType: "guard",
			InputTokens: 1000, CacheReadTokens: 9000, OutputTokens: 500,
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
}

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
	if !strings.Contains(rep.ProjectionFence, "OBSERVED cost projection") ||
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
	if math.Abs(got[0].NetUSDComputed()-(5.0+0.4-0.1-0.5)) > 1e-9 {
		t.Fatalf("round-trip NET mismatch: %.6f", got[0].NetUSDComputed())
	}
}
