package rsiloop

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionobs"
)

func TestSessionObsHarnessKeepsOnlyClosedS0Gain(t *testing.T) {
	h := NewSessionObsDemoHarness()
	res, err := Run(h, nil, 3, 0)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Cycles != 2 {
		t.Fatalf("cycles=%d, want 2", res.Cycles)
	}
	if res.Kept != 1 {
		t.Fatalf("kept=%d, want 1 (rows: %+v)", res.Kept, res.Rows)
	}
	first, second := res.Rows[0], res.Rows[1]
	if first.Kept || first.Improved {
		t.Fatalf("no-op proposal must REVERT with no S0 movement: %+v", first)
	}
	if !second.Kept || !second.Improved {
		t.Fatalf("closed sessionobs proposal must KEEP on strict S0 loop-index gain: %+v", second)
	}
	if second.MetricName != SessionObsMetricName || second.LowerBetter {
		t.Fatalf("sessionobs objective not journaled as higher-better S0 loop-index: %+v", second)
	}
	if second.Candidate_ != 100 {
		t.Fatalf("closed candidate should drive S0 loop-index to 100, got %.0f", second.Candidate_)
	}
	if second.Score == nil || second.Score.Name != "sessionobs_s0" || second.Score.Value != 100 || second.Score.Grade != "A" {
		t.Fatalf("closed candidate should carry an S0 scorecard, got %+v", second.Score)
	}
	if got := scoreComponentValue(second.Score, "sessionobs_debt"); got != 0 {
		t.Fatalf("closed candidate score should report zero sessionobs debt, got %.0f in %+v", got, second.Score)
	}
	if got := scoreComponentValue(second.Score, "loop_consumes"); got != 1 {
		t.Fatalf("closed candidate score should report loop_consumes=true, got %.0f in %+v", got, second.Score)
	}
	if !second.SuiteGreen || !second.TruthClean {
		t.Fatalf("kept S0 candidate must have non-vacuous suite + clean report witness: %+v", second)
	}
	if res.FinalBaseline != 100 {
		t.Fatalf("final S0 baseline=%.0f, want 100 after KEEP", res.FinalBaseline)
	}
}

func TestSessionObsHarnessRevertsPartialS0GainWithoutCleanReport(t *testing.T) {
	baseline := SessionObsState{
		Corpus: []sessionobs.Record{
			{SessionID: "a", AssistantTurns: 4, Outcome: sessionobs.OutcomeUnknown},
			{SessionID: "b", AssistantTurns: 4, Outcome: sessionobs.OutcomeUnknown},
		},
	}
	partial := SessionObsState{
		Corpus: []sessionobs.Record{
			{SessionID: "a", AssistantTurns: 4, Outcome: sessionobs.OutcomeShipped, Signals: sessionobs.Signals{Commits: 1}},
			{SessionID: "b", AssistantTurns: 4, Outcome: sessionobs.OutcomeStopped, Signals: sessionobs.Signals{StopEvents: 1}},
		},
		Pipeline: sessionobs.Pipeline{LoopConsumes: true},
	}
	h := NewSessionObsHarness(baseline, []SessionObsProposal{
		{Label: "sessionobs:partial-link-only", State: partial},
	})
	res, err := Run(h, nil, 3, 1)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(res.Rows) != 1 {
		t.Fatalf("rows=%d, want 1", len(res.Rows))
	}
	row := res.Rows[0]
	if !row.Improved {
		t.Fatalf("partial proposal should raise S0 loop-index enough to mark Improved: %+v", row)
	}
	if row.Kept {
		t.Fatalf("partial proposal must REVERT because the sessionobs report is not clean: %+v", row)
	}
	if row.TruthClean {
		t.Fatalf("partial proposal should have dirty truth witness: %+v", row)
	}
	if row.Score == nil || row.Score.Name != "sessionobs_s0" {
		t.Fatalf("partial proposal should still carry score telemetry: %+v", row.Score)
	}
	if got := scoreComponentValue(row.Score, "sessionobs_debt"); got == 0 {
		t.Fatalf("partial proposal score should expose remaining sessionobs debt: %+v", row.Score)
	}
	if got := scoreComponentValue(row.Score, "corpus_committed"); got != 0 {
		t.Fatalf("partial proposal should expose missing committed corpus: %.0f in %+v", got, row.Score)
	}
}

func scoreComponentValue(score *Scorecard, name string) float64 {
	if score == nil {
		return 0
	}
	for _, c := range score.Components {
		if c.Name == name {
			return c.Value
		}
	}
	return 0
}
