package bench

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRelayVsCompactionBench is the re-runnable witness for issue #1906: it
// produces the comparison report and asserts the acceptance criteria — flat peak
// context for the relay across the duration sweep, plus the cost/accuracy
// advantage over auto-compaction. Regenerate the golden with UPDATE_GOLDEN=1.
func TestRelayVsCompactionBench(t *testing.T) {
	r := BuildRelayVsCompactionReport()

	if r.Schema != "relayvscompaction.v1" {
		t.Fatalf("schema = %q; want relayvscompaction.v1", r.Schema)
	}
	if r.Provenance.Kind != ProvenanceSimulated {
		t.Fatalf("provenance = %q; want %q (hermetic model)", r.Provenance.Kind, ProvenanceSimulated)
	}
	if len(r.Sweep) != 4 {
		t.Fatalf("sweep points = %d; want 4", len(r.Sweep))
	}

	// Done condition (headline): the relay's peak context is FLAT — identical at
	// every swept duration, independent of goal duration (the O(1) invariant).
	if !r.FlatPeakContext {
		t.Fatalf("relay peak context is NOT flat across the sweep: %+v", peaks(r))
	}
	for _, p := range r.Sweep {
		if p.Relay.PeakContext != r.RelayPeakContext {
			t.Fatalf("relay peak drifted at %d turns: %d != %d", p.GoalTurns, p.Relay.PeakContext, r.RelayPeakContext)
		}
	}

	// The relay must peak BELOW the context-rot zone; compaction is pinned at the
	// wall. With the default model that is 60% vs 95% of the window.
	deepest := r.Sweep[len(r.Sweep)-1]
	if deepest.Relay.PeakContextFrac >= deepest.Compaction.PeakContextFrac {
		t.Fatalf("relay peak frac %.2f not below compaction %.2f",
			deepest.Relay.PeakContextFrac, deepest.Compaction.PeakContextFrac)
	}
	if deepest.Compaction.PeakContextFrac < 0.90 {
		t.Errorf("compaction should sit near the wall, got %.2f", deepest.Compaction.PeakContextFrac)
	}

	// Both-lenses acceptance: at EVERY duration the relay is lower-peak, cheaper,
	// busts less cache, and is at least as faithful. Raw cache-hit RATE is reported
	// but NOT an acceptance gate — it flatters compaction's larger prefix; the
	// spine's "compaction busts the cache" claim is the cache-bust COST below.
	for _, p := range r.Sweep {
		if !(p.Relay.PeakContext < p.Compaction.PeakContext) {
			t.Errorf("%d turns: relay peak %d not < compaction %d", p.GoalTurns, p.Relay.PeakContext, p.Compaction.PeakContext)
		}
		if !(p.Relay.BilledTokens < p.Compaction.BilledTokens) {
			t.Errorf("%d turns: relay billed %d not < compaction %d", p.GoalTurns, p.Relay.BilledTokens, p.Compaction.BilledTokens)
		}
		if !(p.Relay.CacheBustTokens < p.Compaction.CacheBustTokens) {
			t.Errorf("%d turns: relay cache-bust %d not < compaction %d", p.GoalTurns, p.Relay.CacheBustTokens, p.Compaction.CacheBustTokens)
		}
		if p.Relay.CacheHitRate <= 0 || p.Compaction.CacheHitRate <= 0 {
			t.Errorf("%d turns: cache-hit rate must be reported for both arms, got relay %.3f / compaction %.3f", p.GoalTurns, p.Relay.CacheHitRate, p.Compaction.CacheHitRate)
		}
		if !(p.Relay.Accuracy >= p.Compaction.Accuracy) {
			t.Errorf("%d turns: relay accuracy %.3f not >= compaction %.3f", p.GoalTurns, p.Relay.Accuracy, p.Compaction.Accuracy)
		}
	}
	if r.Verdict != VerdictRelayWins {
		t.Fatalf("verdict = %q; want %q", r.Verdict, VerdictRelayWins)
	}

	// Accuracy delta: the relay is lossless (fail-closed externalize gate); the
	// deepest-duration compaction has lost load-bearing facts, and that loss GROWS
	// with duration (more compaction events erode more).
	if deepest.Relay.Accuracy != 1.0 {
		t.Errorf("relay accuracy = %.3f; want 1.0 (lossless)", deepest.Relay.Accuracy)
	}
	if deepest.Compaction.Accuracy >= 1.0 {
		t.Errorf("compaction accuracy = %.3f; want < 1.0 (lossy)", deepest.Compaction.Accuracy)
	}
	if !(deepest.Compaction.Accuracy < r.Sweep[0].Compaction.Accuracy) {
		t.Errorf("compaction accuracy should degrade with duration: deepest %.3f not < shortest %.3f",
			deepest.Compaction.Accuracy, r.Sweep[0].Compaction.Accuracy)
	}
	if r.Delta.AccuracyGain <= 0 {
		t.Errorf("accuracy_gain = %.3f; want > 0", r.Delta.AccuracyGain)
	}

	// Net-true hygiene: the report names its invalidating assumptions and its
	// promotion/demotion evidence (the generation frame's closing requirement).
	if len(r.Assumptions) == 0 || r.Promotion == "" || r.DemotionRetirement == "" || r.InvalidatingUnknown == "" {
		t.Errorf("report must name assumptions + promotion + demotion + invalidating unknown")
	}

	// The report marshals to stable JSON (the re-derivable artifact).
	got, err := r.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("empty report JSON")
	}

	// Golden: the committed benchmark result artifact (the issue's "checked-in
	// result"). Regenerate with UPDATE_GOLDEN=1.
	golden := filepath.Join("testdata", "relayvscompaction_report.json")
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(golden, append(got, '\n'), 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		t.Logf("updated golden %s", golden)
		return
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	if !bytes.Equal(bytes.TrimRight(want, "\n"), bytes.TrimRight(got, "\n")) {
		t.Errorf("report drifted from golden %s; re-run with UPDATE_GOLDEN=1 if intended", golden)
	}
}

// TestRelayVsCompactionFlatInvariant isolates the O(1) claim: doubling the goal
// duration leaves the relay's peak context UNCHANGED while compaction's total
// work (rotations, billed tokens) scales up. This is the property that makes a
// week-long relay peak no higher than an hour-long one.
func TestRelayVsCompactionFlatInvariant(t *testing.T) {
	m := DefaultRVCModel()
	short := simulateRelay(m, 50)
	long := simulateRelay(m, 400)
	if short.PeakContext != long.PeakContext {
		t.Fatalf("relay peak not invariant to duration: 50-turn %d != 400-turn %d", short.PeakContext, long.PeakContext)
	}
	if !(long.Rotations > short.Rotations) {
		t.Fatalf("longer goal should rotate more legs: %d not > %d", long.Rotations, short.Rotations)
	}
	// Compaction, by contrast, also caps peak (at the wall) but does strictly more
	// summarization work as the goal grows.
	cShort := simulateCompaction(m, 50)
	cLong := simulateCompaction(m, 400)
	if !(cLong.Rotations > cShort.Rotations) {
		t.Fatalf("longer goal should compact more: %d not > %d", cLong.Rotations, cShort.Rotations)
	}
	if !(cLong.PeakContext > long.PeakContext) {
		t.Fatalf("compaction peak %d should exceed relay peak %d", cLong.PeakContext, long.PeakContext)
	}
}

// TestRelayVsCompactionCustomModelSeam checks the observed-run seam: a live leg
// record can feed different constants/durations and still fold into the same
// report shape (the promotion path).
func TestRelayVsCompactionCustomModelSeam(t *testing.T) {
	m := DefaultRVCModel()
	m.WindowCeiling = 100_000
	m.GrowthPerTurn = 4_000
	r := BuildRelayVsCompactionReportFor(m, []int{30, 60})
	if len(r.Sweep) != 2 {
		t.Fatalf("custom sweep points = %d; want 2", len(r.Sweep))
	}
	if r.Model.WindowCeiling != 100_000 {
		t.Fatalf("custom model not threaded through: window = %d", r.Model.WindowCeiling)
	}
	if !r.FlatPeakContext {
		t.Errorf("relay peak should still be flat under the custom model")
	}
}

// TestRelayVsCompactionObservedLegLedger is the promotion witness for issue
// #2659: OBSERVED per-leg records (a replayed capture, provenance != SIMULATED)
// fed through the loader reproduce the SIGN — the relay is lower-peak, cheaper,
// busts less cache, and is at least as faithful at every swept duration — and the
// relay's peak context stays FLAT across the sweep. The checked-in report is the
// issue's "checked-in OBSERVED result artifact". Regenerate with UPDATE_GOLDEN=1.
func TestRelayVsCompactionObservedLegLedger(t *testing.T) {
	ledger := filepath.Join("testdata", "relayvscompaction_observed_legs.json")
	r, err := BuildRelayVsCompactionReportFromLegLedger(DefaultRVCModel(), ledger)
	if err != nil {
		t.Fatalf("BuildRelayVsCompactionReportFromLegLedger: %v", err)
	}

	// Provenance is OBSERVED (record-derived), not the analytic model — the whole
	// point of the promotion seam.
	if r.Provenance.Kind != ProvenanceObserved {
		t.Fatalf("provenance = %q; want %q (record-derived)", r.Provenance.Kind, ProvenanceObserved)
	}
	if r.Schema != "relayvscompaction.v1" {
		t.Fatalf("schema = %q; want relayvscompaction.v1", r.Schema)
	}
	if len(r.Sweep) != 2 {
		t.Fatalf("sweep points = %d; want 2 (durations 6, 12)", len(r.Sweep))
	}

	// The sign the analytic model asserts must survive on OBSERVED records.
	for _, p := range r.Sweep {
		if !(p.Relay.PeakContext < p.Compaction.PeakContext) {
			t.Errorf("%d turns: relay peak %d not < compaction %d", p.GoalTurns, p.Relay.PeakContext, p.Compaction.PeakContext)
		}
		if !(p.Relay.BilledTokens < p.Compaction.BilledTokens) {
			t.Errorf("%d turns: relay billed %d not < compaction %d", p.GoalTurns, p.Relay.BilledTokens, p.Compaction.BilledTokens)
		}
		if !(p.Relay.CacheBustTokens < p.Compaction.CacheBustTokens) {
			t.Errorf("%d turns: relay cache-bust %d not < compaction %d", p.GoalTurns, p.Relay.CacheBustTokens, p.Compaction.CacheBustTokens)
		}
		if !(p.Relay.Accuracy >= p.Compaction.Accuracy) {
			t.Errorf("%d turns: relay accuracy %.3f not >= compaction %.3f", p.GoalTurns, p.Relay.Accuracy, p.Compaction.Accuracy)
		}
	}
	if r.Verdict != VerdictRelayWins {
		t.Fatalf("observed verdict = %q; want %q (sign reproduced)", r.Verdict, VerdictRelayWins)
	}
	// The O(1) invariant holds in the OBSERVED capture too: the relay's peak is
	// identical across the sweep even though the 12-turn run does strictly more
	// rotations than the 6-turn run.
	if !r.FlatPeakContext {
		t.Fatalf("observed relay peak is NOT flat across the sweep: %v", peaks(r))
	}
	if !(r.Sweep[1].Relay.Rotations > r.Sweep[0].Relay.Rotations) {
		t.Fatalf("longer observed goal should rotate more legs: %d not > %d",
			r.Sweep[1].Relay.Rotations, r.Sweep[0].Relay.Rotations)
	}

	got, err := r.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	golden := filepath.Join("testdata", "relayvscompaction_observed_report.json")
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(golden, append(got, '\n'), 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		t.Logf("updated golden %s", golden)
		return
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	if !bytes.Equal(bytes.TrimRight(want, "\n"), bytes.TrimRight(got, "\n")) {
		t.Errorf("observed report drifted from golden %s; re-run with UPDATE_GOLDEN=1 if intended", golden)
	}
}

// TestRelayVsCompactionObservedRefusals pins the fail-closed contract: an OBSERVED
// report is only emitted when the records actually back it. Empty input, a run
// with no turns, an unknown strategy label, a duplicate arm, and a duration
// missing one of the two arms each refuse rather than fabricate an OBSERVED sign.
func TestRelayVsCompactionObservedRefusals(t *testing.T) {
	m := DefaultRVCModel()
	relay6 := RVCRunRecord{Strategy: StrategyRelay, GoalTurns: 6, Turns: []RVCTurnRecord{
		{CachedInputTokens: 6000, UncachedInputTokens: 6000, ResidentTokens: 12000, FactsProduced: 3, FactsRetained: 3},
	}}
	comp6 := RVCRunRecord{Strategy: StrategyCompaction, GoalTurns: 6, Turns: []RVCTurnRecord{
		{CachedInputTokens: 40000, UncachedInputTokens: 20000, ResidentTokens: 60000, FactsProduced: 3, FactsRetained: 3},
	}}

	cases := []struct {
		name    string
		runs    []RVCRunRecord
		wantSub string
	}{
		{"empty", nil, "no observed leg records"},
		{"no-turns", []RVCRunRecord{{Strategy: StrategyRelay, GoalTurns: 6}, comp6}, "has no turn records"},
		{"unknown-strategy", []RVCRunRecord{{Strategy: "guess", GoalTurns: 6, Turns: relay6.Turns}}, "unknown strategy"},
		{"duplicate-arm", []RVCRunRecord{relay6, relay6, comp6}, "duplicate relay run"},
		{"missing-compaction", []RVCRunRecord{relay6}, "missing the compaction arm"},
		{"missing-relay", []RVCRunRecord{comp6}, "missing the relay arm"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := BuildRelayVsCompactionReportFromRecords(m, tc.runs, "fixture")
			if err == nil {
				t.Fatalf("want refusal containing %q, got nil error", tc.wantSub)
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Fatalf("error = %q; want substring %q", err.Error(), tc.wantSub)
			}
		})
	}
}

// TestRelayVsCompactionEmptySweepNoFalseWin pins the fail-closed contract on the
// ANALYTIC seam. BuildRelayVsCompactionReportFor is the exported seam a live
// leg-record run feeds constants into; over an empty duration sweep it has ZERO
// comparisons, so it must not claim the relay wins or that its peak context is flat
// — the same honesty BuildRelayVsCompactionReportFromRecords already enforces on
// the OBSERVED path (TestRelayVsCompactionObservedRefusals). Before the fix the
// analytic seam defaulted the verdict to VerdictRelayWins and flat_peak_context to
// true, fabricating a "relay wins + flat peak" claim over no data points.
func TestRelayVsCompactionEmptySweepNoFalseWin(t *testing.T) {
	for _, durations := range [][]int{nil, {}} {
		r := BuildRelayVsCompactionReportFor(DefaultRVCModel(), durations)
		if len(r.Sweep) != 0 {
			t.Fatalf("empty durations %v should yield an empty sweep, got %d points", durations, len(r.Sweep))
		}
		if r.Verdict != VerdictNoAdvantage {
			t.Errorf("empty sweep verdict = %q; want %q (zero comparisons cannot witness a win)", r.Verdict, VerdictNoAdvantage)
		}
		if r.FlatPeakContext {
			t.Errorf("empty sweep claims flat_peak_context=true with no data points to witness the invariant")
		}
	}
}

func peaks(r RelayVsCompactionReport) []int {
	out := make([]int, 0, len(r.Sweep))
	for _, p := range r.Sweep {
		out = append(out, p.Relay.PeakContext)
	}
	return out
}
