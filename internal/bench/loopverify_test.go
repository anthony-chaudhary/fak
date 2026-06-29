package bench

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// TestLoopVerifyBench is the re-runnable witness for issue #1190: it produces the
// JSON report with the four metrics and asserts the acceptance criterion — the
// gated loop's false-done rate is strictly lower than the naive loop's on the
// corpus.
func TestLoopVerifyBench(t *testing.T) {
	r := BuildLoopVerifyReport()

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
