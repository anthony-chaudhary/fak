package armbench

import "testing"

func TestPairedReportStratifiesSimpsonsParadox(t *testing.T) {
	in := fixtureReceipts()
	rep, err := BuildPairedReport(in)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Comparisons) != 2 {
		t.Fatalf("comparisons=%d, want task-stratified 2", len(rep.Comparisons))
	}
	for _, c := range rep.Comparisons {
		if c.Task == "easy" && c.Metrics[0].Delta.Estimate != -20 {
			t.Fatalf("easy input delta=%v", c.Metrics[0].Delta.Estimate)
		}
		if c.Task == "hard" && c.Metrics[0].Delta.Estimate != -20 {
			t.Fatalf("hard input delta=%v", c.Metrics[0].Delta.Estimate)
		}
	}
}

func TestPairedReportWithholdsTokenClaimOnCorrectnessLoss(t *testing.T) {
	in := fixtureReceipts()
	for i := range in.Trials {
		if in.Trials[i].Task == "easy" && in.Trials[i].Arm == "treatment" {
			in.Trials[i].Success = false
		}
	}
	rep, err := BuildPairedReport(in)
	if err != nil {
		t.Fatal(err)
	}
	var easy PairedComparison
	for _, c := range rep.Comparisons {
		if c.Task == "easy" {
			easy = c
		}
	}
	if easy.EfficiencyClaimsAllowed || easy.CorrectnessGate != "fail" {
		t.Fatalf("correctness-loss treatment was claimable: %+v", easy)
	}
	if easy.Metrics[0].Delta.Upper >= 0 {
		t.Fatalf("fixture must really save input tokens: %+v", easy.Metrics[0])
	}
	for _, cc := range rep.ClaimCheck {
		if cc.Scope[:9] == "task=easy" && cc.Verdict != "not-yet" {
			t.Fatalf("claim-check verdict=%s", cc.Verdict)
		}
	}
}

func TestPairedReportSeparatesTokenKindsColdWarmAndCosts(t *testing.T) {
	rep, err := BuildPairedReport(fixtureReceipts())
	if err != nil {
		t.Fatal(err)
	}
	c := rep.Comparisons[0]
	want := []string{"input_tokens", "output_tokens", "total_tokens", "wall_latency", "ttft", "retry_rate", "failure_rate"}
	for i, n := range want {
		if c.Metrics[i].Metric != n {
			t.Fatalf("metric[%d]=%s", i, c.Metrics[i].Metric)
		}
	}
	if c.ColdPairs != 1 || c.WarmPairs != 3 || len(c.ColdMetrics) == 0 || len(c.WarmMetrics) == 0 {
		t.Fatalf("cold/warm not split: %+v", c)
	}
	if len(c.Costs) != 2 || c.Costs[0].SetupDeltaUSD != 1 {
		t.Fatalf("cost sensitivity/setup missing: %+v", c.Costs)
	}
	if rep.Correction == "" || rep.Confidence <= .95 {
		t.Fatalf("multiple correction absent: %+v", rep)
	}
}

func fixtureReceipts() *PairedReceipts {
	in := &PairedReceipts{Schema: PairedReceiptsSchema, Benchmark: "golden", TunedBaseline: "tuned", CorrectnessMargin: .1, SafetyMargin: 0, BootstrapSamples: 2000, Provenance: "deterministic golden receipts", Witness: "internal/armbench/paired_report_test.go", Setup: []ArmSetupCost{{"tuned", 0, 0}, {"treatment", 50, 1}}, Prices: []PriceScenario{{"current", 2, 8, 1}, {"local-2x", 2, 8, 2}}}
	add := func(task, id string, cold bool, bIn, tIn float64) {
		for _, x := range []struct {
			arm   string
			input float64
		}{{"tuned", bIn}, {"treatment", tIn}} {
			in.Trials = append(in.Trials, PairedTrial{PairID: id, Task: task, Model: "model", Temperature: "0", Arm: x.arm, Success: true, Safe: true, InputTokens: x.input, OutputTokens: 20, WallMS: x.input, TTFTMS: x.input / 2, Cold: cold})
		}
	}
	// Pooled arm means reverse because treatment has more hard-task observations; task-paired deltas remain -20.
	for i := 0; i < 4; i++ {
		add("easy", string(rune('a'+i)), i == 0, 100, 80)
	}
	for i := 0; i < 4; i++ {
		add("hard", string(rune('e'+i)), i == 0, 1000, 980)
	}
	return in
}
