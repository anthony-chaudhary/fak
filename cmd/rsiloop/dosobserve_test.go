package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

func TestScaleWorkPair_HigherBetterIsDirectRound(t *testing.T) {
	work, base := scaleWorkPair(4, 0, false)
	if work != 4_000_000_000 || base != 0 {
		t.Fatalf("higher-better scale: work=%d base=%d, want 4000000000/0", work, base)
	}
	if !(work > base) {
		t.Error("a measured improvement must scale to work > baseline-work")
	}
}

func TestScaleWorkPair_LowerBetterIsConstantMinusNotNegation(t *testing.T) {
	// Lower is better: candidate 2 beats baseline 5. The recorded values must stay
	// NON-negative (constant-minus, not negation) and keep work > baseline-work.
	work, base := scaleWorkPair(2, 5, true)
	if work < 0 || base < 0 {
		t.Fatalf("constant-minus must stay non-negative: work=%d base=%d", work, base)
	}
	if !(work > base) {
		t.Errorf("lower-better improvement must give work > baseline-work, got %d vs %d", work, base)
	}
	// A regression (candidate worse than baseline) must NOT read as a gain.
	w2, b2 := scaleWorkPair(7, 5, true)
	if w2 > b2 {
		t.Errorf("lower-better regression must give work <= baseline-work, got %d vs %d", w2, b2)
	}
}

func TestDosImproveArgs_CarriesWitnessesScaledWorkAndVerdict(t *testing.T) {
	r := rsiloop.Row{
		Cycle: 3, MetricName: "near_misses_caught", Baseline: 0, Candidate_: 4,
		Measured: true, LowerBetter: false, Improved: true,
		SuiteGreen: true, TruthClean: true, Decision: "KEEP", BreakerCount: 0,
		Score: &rsiloop.Scorecard{
			Name:  "attention_sn",
			Value: 0.9,
			Grade: "lean",
			Components: []rsiloop.ScoreComponent{
				{Name: "mean_ratio", Value: 0.9, Unit: "ratio"},
			},
		},
	}
	joined := strings.Join(dosImproveArgs("/ws", 3, r), " ")
	for _, want := range []string{
		"improve", "--observe", "--work 4000000000", "--baseline-work 0",
		"--suite-passed", "--truth-clean", "--max-reverts 3", "--workspace /ws",
		"--lane rsiloop",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("args missing %q; got: %s", want, joined)
		}
	}
	// The loop's own verdict rides in --narrated (carried for the operator, gating nothing).
	if !strings.Contains(joined, "verdict=KEEP") {
		t.Errorf("--narrated must carry the loop verdict; got: %s", joined)
	}
	if !strings.Contains(joined, "score attention_sn=0.900") || !strings.Contains(joined, "mean_ratio=0.900") {
		t.Errorf("--narrated must carry score telemetry without changing --work; got: %s", joined)
	}
}

func TestDosImproveArgs_OmitsFalseWitnessesAndEmptyWorkspace(t *testing.T) {
	r := rsiloop.Row{Cycle: 1, MetricName: "k", Decision: "REVERT", SuiteGreen: false, TruthClean: false}
	joined := strings.Join(dosImproveArgs("", 3, r), " ")
	if strings.Contains(joined, "--suite-passed") || strings.Contains(joined, "--truth-clean") {
		t.Errorf("a red/dirty row must omit the witness flags (dos defaults them to fail-safe); got: %s", joined)
	}
	if strings.Contains(joined, "--workspace") {
		t.Errorf("an empty workspace must omit --workspace; got: %s", joined)
	}
}
