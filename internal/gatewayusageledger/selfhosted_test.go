package gatewayusageledger

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func selfHostedRow(key string, c Counters) Row {
	return Row{Schema: Schema, RowKey: key, Kind: "exit", SessionType: "serve", Counters: c}
}

// The finding this whole split exists to prevent: a corpus written before anything
// classified its serving side must report NOT INSTRUMENTED, never 0.0%. The two read
// identically in a float and mean opposite things about a fleet.
func TestAnUnclassifiedCorpusRefusesToReportZeroSelfHosted(t *testing.T) {
	// Plenty of served volume, not one classified turn — every pre-split row.
	rows := []Row{
		selfHostedRow("a", Counters{ObservedTurns: 40, InputTokens: 900_000, OutputTokens: 120_000}),
		selfHostedRow("b", Counters{ObservedTurns: 11, InputTokens: 80_000, OutputTokens: 9_000}),
	}
	got := FoldSelfHostedShare(rows)
	if got.OutputShare != nil {
		t.Fatalf("OutputShare = %v, want nil (nothing classified) — a 0%% here is a claim the corpus never earned", *got.OutputShare)
	}
	if got.Reason != ShareNotInstrumented {
		t.Errorf("Reason = %q, want %q", got.Reason, ShareNotInstrumented)
	}
	if got.OutputTokens != 129_000 {
		t.Errorf("OutputTokens = %d, want 129000 (the unsplit total still folds)", got.OutputTokens)
	}
	if cov := got.ClassifiedOutputFraction(); cov != 0 {
		t.Errorf("ClassifiedOutputFraction = %v, want 0", cov)
	}
}

// The mirror case, and the reason the refusal above is not just "return zero when in
// doubt": a deployment that measured its turns and bought every one of them from a
// vendor has EARNED a 0.0%, and must print it as a number rather than a shrug.
func TestAnAllVendorCorpusReportsAnEarnedZero(t *testing.T) {
	rows := []Row{selfHostedRow("a", Counters{
		ObservedTurns: 12, OutputTokens: 50_000,
		VendorTurns: 12, VendorInputTokens: 400_000, VendorOutputTokens: 50_000,
	})}
	got := FoldSelfHostedShare(rows)
	if got.OutputShare == nil {
		t.Fatalf("OutputShare = nil (reason %q), want an earned 0.0 — 12 turns were classified", got.Reason)
	}
	if *got.OutputShare != 0 {
		t.Errorf("OutputShare = %v, want 0", *got.OutputShare)
	}
	if got.Reason != "" {
		t.Errorf("Reason = %q, want empty when the share is answerable", got.Reason)
	}
	if cov := got.ClassifiedOutputFraction(); cov != 1 {
		t.Errorf("ClassifiedOutputFraction = %v, want 1 (every output token classified)", cov)
	}
}

func TestFoldSelfHostedShareSplitsAndCovers(t *testing.T) {
	rows := []Row{
		selfHostedRow("a", Counters{
			ObservedTurns: 10, OutputTokens: 1_000,
			SelfHostedTurns: 6, SelfHostedInputTokens: 30_000, SelfHostedOutputTokens: 700,
			VendorTurns: 4, VendorInputTokens: 20_000, VendorOutputTokens: 300,
		}),
		selfHostedRow("b", Counters{
			ObservedTurns: 5, OutputTokens: 1_000,
			SelfHostedTurns: 1, SelfHostedInputTokens: 5_000, SelfHostedOutputTokens: 100,
			VendorTurns: 2, VendorInputTokens: 9_000, VendorOutputTokens: 400,
			// 500 output tokens on this row are UNCLASSIFIED — they must land in the
			// coverage denominator and in neither side of the share.
		}),
	}
	got := FoldSelfHostedShare(rows)
	if got.OutputShare == nil {
		t.Fatalf("OutputShare = nil (reason %q), want a share", got.Reason)
	}
	// 800 self-hosted / 1500 classified — NOT 800/2000. Dividing by the unsplit total
	// would quietly count unclassified volume as vendor.
	if want := 800.0 / 1500.0; math.Abs(*got.OutputShare-want) > 1e-9 {
		t.Errorf("OutputShare = %v, want %v (self / CLASSIFIED, not self / total)", *got.OutputShare, want)
	}
	if got.ClassifiedTurns() != 13 {
		t.Errorf("ClassifiedTurns = %d, want 13", got.ClassifiedTurns())
	}
	if got.SelfHostedInputTokens != 35_000 || got.VendorInputTokens != 29_000 {
		t.Errorf("input split = %d/%d, want 35000/29000", got.SelfHostedInputTokens, got.VendorInputTokens)
	}
	if want := 1500.0 / 2000.0; math.Abs(got.ClassifiedOutputFraction()-want) > 1e-9 {
		t.Errorf("ClassifiedOutputFraction = %v, want %v", got.ClassifiedOutputFraction(), want)
	}
}

func TestFoldSelfHostedShareRefusesAnEmptyDenominator(t *testing.T) {
	// Turns were classified; they generated nothing (an all-refusal window). The
	// fraction is undefined, and undefined is not zero.
	rows := []Row{selfHostedRow("a", Counters{ObservedTurns: 3, SelfHostedTurns: 2, VendorTurns: 1})}
	got := FoldSelfHostedShare(rows)
	if got.OutputShare != nil {
		t.Fatalf("OutputShare = %v, want nil — no classified output token exists to divide", *got.OutputShare)
	}
	if got.Reason != ShareNoClassifiedOutput {
		t.Errorf("Reason = %q, want %q", got.Reason, ShareNoClassifiedOutput)
	}
}

func TestFoldSelfHostedShareDedupesAndKeepsCarryforward(t *testing.T) {
	live := Counters{
		ObservedTurns: 4, OutputTokens: 100,
		SelfHostedTurns: 4, SelfHostedOutputTokens: 100,
	}
	rows := []Row{
		selfHostedRow("dup", live),
		selfHostedRow("dup", live), // a re-read of the same appended row
		{Schema: Schema, Kind: KindCarryforward, SessionType: "serve",
			Carryforward: &Carryforward{FoldedKind: "exit", FoldedRows: 9},
			// Cut folded these away; the file no longer holds the rows behind them.
			Counters: Counters{ObservedTurns: 30, OutputTokens: 300, VendorTurns: 30, VendorOutputTokens: 300}},
	}
	got := FoldSelfHostedShare(rows)
	if got.RowsDedupedAtFold != 1 {
		t.Errorf("RowsDedupedAtFold = %d, want 1", got.RowsDedupedAtFold)
	}
	if got.OutputShare == nil {
		t.Fatalf("OutputShare = nil (reason %q)", got.Reason)
	}
	// 100 self / 400 classified: the carryforward's era counts. Skipping it (as
	// FoldTrend does, for a different reason) would report 100% self-hosted.
	if want := 0.25; math.Abs(*got.OutputShare-want) > 1e-9 {
		t.Errorf("OutputShare = %v, want %v — a carryforward era must not be shed from a SUM", *got.OutputShare, want)
	}
}

// The split is omitempty so that a row which never measured stays byte-identical to
// the pre-split schema — the property that lets an absent field mean "not
// instrumented" instead of "zero".
func TestUnmeasuredSplitStaysOffTheWire(t *testing.T) {
	b, err := json.Marshal(Counters{ObservedTurns: 2, OutputTokens: 7})
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"self_hosted_turns", "self_hosted_input_tokens", "self_hosted_output_tokens",
		"vendor_turns", "vendor_input_tokens", "vendor_output_tokens"} {
		if strings.Contains(string(b), f) {
			t.Errorf("unmeasured Counters serialized %q; an absent field is what carries 'not instrumented'", f)
		}
	}
	measured, err := json.Marshal(Counters{VendorTurns: 1, VendorOutputTokens: 7})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(measured), "vendor_turns") {
		t.Errorf("measured Counters dropped vendor_turns: %s", measured)
	}
}

// sumCounters is reflective and REFUSES a field kind it cannot add, so a cut cannot
// silently shed the new split. Pin that the six fields survive a fold.
func TestCutSumsTheSelfHostedSplit(t *testing.T) {
	var dst Counters
	add := Counters{
		SelfHostedTurns: 2, SelfHostedInputTokens: 3, SelfHostedOutputTokens: 4,
		VendorTurns: 5, VendorInputTokens: 6, VendorOutputTokens: 7,
	}
	if err := sumCounters(&dst, add); err != nil {
		t.Fatal(err)
	}
	if err := sumCounters(&dst, add); err != nil {
		t.Fatal(err)
	}
	// Field-wise: Counters carries maps, so it is not comparable with ==.
	for _, c := range []struct {
		name string
		got  uint64
		want uint64
	}{
		{"SelfHostedTurns", dst.SelfHostedTurns, 4},
		{"SelfHostedInputTokens", dst.SelfHostedInputTokens, 6},
		{"SelfHostedOutputTokens", dst.SelfHostedOutputTokens, 8},
		{"VendorTurns", dst.VendorTurns, 10},
		{"VendorInputTokens", dst.VendorInputTokens, 12},
		{"VendorOutputTokens", dst.VendorOutputTokens, 14},
	} {
		if c.got != c.want {
			t.Errorf("sumCounters %s = %d, want %d", c.name, c.got, c.want)
		}
	}
}
