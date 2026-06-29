package bench

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// TestLoopVerifyBench is the re-runnable witness for issue #1190: it produces the
// JSON report with the four metrics and asserts the acceptance criterion — the
// gated loop's false-done rate is strictly lower than the naive loop's on the
// corpus.
func TestLoopVerifyBench(t *testing.T) {
	r := BuildLoopVerifyReport()
	if r.Provenance.Kind != ProvenanceSimulated {
		t.Fatalf("fixture provenance = %q, want %q", r.Provenance.Kind, ProvenanceSimulated)
	}

	// Corpus shape.
	if r.CorpusEpisodes != 4 || r.CorpusTurns != 11 {
		t.Fatalf("corpus = %d episodes / %d turns; want 4 / 11", r.CorpusEpisodes, r.CorpusTurns)
	}

	// Metric 1 — false-done rate: the headline acceptance criterion.
	if got := r.Naive.FalseDoneRate; got != 0.5 {
		t.Errorf("naive false_done_rate = %v; want 0.5", got)
	}
	if got := r.Gated.FalseDoneRate; got != 0.0 {
		t.Errorf("gated false_done_rate = %v; want 0.0", got)
	}
	if !(r.Gated.FalseDoneRate < r.Naive.FalseDoneRate) {
		t.Fatalf("acceptance VIOLATED: gated false-done %.3f not lower than naive %.3f",
			r.Gated.FalseDoneRate, r.Naive.FalseDoneRate)
	}
	if r.Verdict != VerdictGatedLower {
		t.Errorf("verdict = %q; want %q", r.Verdict, VerdictGatedLower)
	}

	// Metric 2 — slop delta: the gate ends at lower net slop because the rework
	// turns clean up what the false done shipped.
	if r.Naive.SlopShippedTotal != 5 || r.Gated.SlopShippedTotal != 1 {
		t.Errorf("slop shipped naive/gated = %d/%d; want 5/1", r.Naive.SlopShippedTotal, r.Gated.SlopShippedTotal)
	}
	if r.Delta.SlopAvoided != 4 {
		t.Errorf("slop_avoided = %d; want 4", r.Delta.SlopAvoided)
	}

	// Metric 3 — iterations-to-witnessed-done and wasted iterations.
	if r.Gated.MeanIterations != 2.75 {
		t.Errorf("gated mean_iterations_to_witnessed_done = %v; want 2.75", r.Gated.MeanIterations)
	}
	if r.Naive.WastedIterations != 5 || r.Gated.WastedIterations != 0 {
		t.Errorf("wasted iterations naive/gated = %d/%d; want 5/0", r.Naive.WastedIterations, r.Gated.WastedIterations)
	}
	if r.Delta.WastedIterationsAvoided != 5 {
		t.Errorf("wasted_iterations_avoided = %d; want 5", r.Delta.WastedIterationsAvoided)
	}

	// Metric 4 — gate cost reported NET (net-true): the naive loop pays none, the
	// gated loop pays one adjudication per turn it ran (11), and reaching a real
	// done costs 5 extra iterations over the naive loop's premature stop.
	if r.Naive.GateCostUnitsTotal != 0 {
		t.Errorf("naive gate_cost = %v; want 0", r.Naive.GateCostUnitsTotal)
	}
	if r.Gated.GateCostUnitsTotal != 11 || r.Delta.GateCostUnits != 11 {
		t.Errorf("gated gate_cost = %v (delta %v); want 11", r.Gated.GateCostUnitsTotal, r.Delta.GateCostUnits)
	}
	if r.Delta.ExtraIterationsToWitness != 5 {
		t.Errorf("extra_iterations_to_witness = %d; want 5", r.Delta.ExtraIterationsToWitness)
	}

	// The report marshals to stable JSON (the re-derivable artifact).
	got, err := r.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("empty report JSON")
	}

	// Golden: the committed report artifact. Regenerate with UPDATE_GOLDEN=1.
	golden := filepath.Join("testdata", "loopverify_report.json")
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

// TestLoopVerifyCorpusTooEasy checks the honest-finding path: a corpus the naive
// loop never trips on yields no measurable gate win, reported as such rather than
// as a failure.
func TestLoopVerifyCorpusTooEasy(t *testing.T) {
	easy := []Episode{
		{Name: "trivially-true", Turns: []Turn{
			{SelfReportedDone: true, DosVerdict: VerdictWitnessed, GateCostUnits: 1},
		}},
	}
	r := BuildLoopVerifyReportFor(easy)
	if r.Verdict != VerdictCorpusTooEasy {
		t.Errorf("verdict = %q; want %q", r.Verdict, VerdictCorpusTooEasy)
	}
	if r.Delta.FalseDoneRateReduction != 0 {
		t.Errorf("false_done_rate_reduction = %v; want 0", r.Delta.FalseDoneRateReduction)
	}
}

func TestLoopVerifyFromLoopLedgerObserved(t *testing.T) {
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	appendLoopVerifyEvent(t, path, loopmgr.Event{
		LoopID:  "goal/live",
		RunID:   "goal-live-turn-1",
		Kind:    loopmgr.EventEnd,
		Status:  loopmgr.StatusClaimedDone,
		Reason:  "EXIT_0",
		Summary: "child self-reported done",
		Metrics: map[string]int64{"slop_introduced": 3},
	})
	appendLoopVerifyEvent(t, path, loopmgr.Event{
		LoopID:  "goal/live",
		RunID:   "goal-live-turn-1",
		Kind:    loopmgr.EventWitness,
		Status:  loopmgr.StatusWitnessRefused,
		Reason:  "LOOP_DONE_UNWITNESSED",
		Summary: "commit-audit refuted the done claim",
	})
	appendLoopVerifyEvent(t, path, loopmgr.Event{
		LoopID:  "goal/live",
		RunID:   "goal-live-turn-2",
		Kind:    loopmgr.EventEnd,
		Status:  loopmgr.StatusClaimedDone,
		Reason:  "EXIT_0",
		Summary: "child self-reported done after rework",
		Metrics: map[string]int64{"slop_introduced": -2},
	})
	appendLoopVerifyEvent(t, path, loopmgr.Event{
		LoopID:  "goal/live",
		RunID:   "goal-live-turn-2",
		Kind:    loopmgr.EventWitness,
		Status:  loopmgr.StatusWitnessedDone,
		Reason:  "WITNESS_OK",
		Summary: "commit-audit witnessed the fix",
	})

	r, err := BuildLoopVerifyReportFromLoopLedger(path)
	if err != nil {
		t.Fatalf("BuildLoopVerifyReportFromLoopLedger: %v", err)
	}
	if r.Provenance.Kind != ProvenanceObserved {
		t.Fatalf("provenance = %q, want %q", r.Provenance.Kind, ProvenanceObserved)
	}
	if r.Schema != "loopverify.v1" || r.CorpusEpisodes != 1 || r.CorpusTurns != 2 {
		t.Fatalf("report shape schema=%q episodes=%d turns=%d", r.Schema, r.CorpusEpisodes, r.CorpusTurns)
	}
	if r.Naive.FalseDoneRate != 1 || r.Gated.FalseDoneRate != 0 {
		t.Fatalf("false-done rates naive/gated = %.3f/%.3f, want 1/0", r.Naive.FalseDoneRate, r.Gated.FalseDoneRate)
	}
	if r.Delta.SlopAvoided != 2 || r.Delta.WastedIterationsAvoided != 1 {
		t.Fatalf("delta slop/wasted = %d/%d, want 2/1", r.Delta.SlopAvoided, r.Delta.WastedIterationsAvoided)
	}
	if r.Delta.GateCostUnits != 2 || r.Delta.ExtraIterationsToWitness != 1 {
		t.Fatalf("delta gate/extra-iters = %.1f/%d, want 2/1", r.Delta.GateCostUnits, r.Delta.ExtraIterationsToWitness)
	}
	if r.Episodes[0].Naive.FalseDone != true || r.Episodes[0].Gated.FalseDone != false {
		t.Fatalf("episode false-done detail = %+v", r.Episodes[0])
	}
}

func TestLoopVerifyFromLoopEventsRefusesMissingWitness(t *testing.T) {
	_, err := BuildLoopVerifyReportFromLoopEvents([]loopmgr.Event{{
		LoopID: "goal/live",
		RunID:  "goal-live-turn-1",
		Kind:   loopmgr.EventEnd,
		Status: loopmgr.StatusClaimedDone,
	}}, "fixture")
	if err == nil || !strings.Contains(err.Error(), "no witness verdict") {
		t.Fatalf("missing witness error = %v, want no witness verdict", err)
	}
}

func TestLoopVerifyFromLoopEventsRefusesUnavailableWitness(t *testing.T) {
	_, err := BuildLoopVerifyReportFromLoopEvents([]loopmgr.Event{
		{LoopID: "goal/live", RunID: "goal-live-turn-1", Kind: loopmgr.EventEnd, Status: loopmgr.StatusClaimedDone},
		{LoopID: "goal/live", RunID: "goal-live-turn-1", Kind: loopmgr.EventWitness, Status: loopmgr.StatusWitnessUnavailable},
	}, "fixture")
	if err == nil || !strings.Contains(err.Error(), "non-observed witness status") {
		t.Fatalf("unavailable witness error = %v, want non-observed witness status", err)
	}
}

func appendLoopVerifyEvent(t *testing.T, path string, ev loopmgr.Event) {
	t.Helper()
	if _, err := loopmgr.Append(path, ev); err != nil {
		t.Fatalf("append loop event: %v", err)
	}
}
