package gardenbundle

import "testing"

func TestScoreResultsRanksStableWorstFirstIssues(t *testing.T) {
	score := ScoreResults([]MemberResult{
		{
			Key:      "fresh_status",
			Label:    "fresh status",
			State:    "action",
			OK:       true,
			Verdict:  "ACTION",
			Detail:   "unpushed work",
			ExitCode: 0,
		},
		{
			Key:      "scorecard",
			Label:    "scorecard control pane",
			Gates:    true,
			State:    "red",
			OK:       false,
			Verdict:  "ACTION",
			Detail:   "grade debt rose",
			ExitCode: 1,
		},
		{
			Key:     "stale_leases",
			Label:   "stale leases",
			State:   "ok",
			OK:      true,
			Verdict: "OK",
		},
	})

	if score.Score >= 100 || score.Debt == 0 {
		t.Fatalf("want non-clear score/debt, got score=%d debt=%d", score.Score, score.Debt)
	}
	if len(score.TopIssues) != 2 {
		t.Fatalf("want two top issues (ok member excluded), got %d", len(score.TopIssues))
	}
	top := score.TopIssues[0]
	if top.RecurrenceKey != "garden/scorecard/red" {
		t.Fatalf("top recurrence key = %q, want garden/scorecard/red", top.RecurrenceKey)
	}
	if !top.Gates || top.Severity <= score.TopIssues[1].Severity {
		t.Fatalf("gating red should outrank advisory action, top=%+v next=%+v", top, score.TopIssues[1])
	}
	if top.NextAction == "" {
		t.Fatalf("top issue must carry an owning next action")
	}
}

func TestPayloadMarshalBackfillsHealthForManualPayload(t *testing.T) {
	p := Payload{
		OK:      false,
		Verdict: "ACTION",
		Finding: "garden_gate_red",
		Members: []MemberResult{
			{Key: "scorecard", Label: "scorecard", Gates: true, State: "red", ExitCode: 1},
		},
	}
	score := p.score()
	if score.Score >= 100 || score.Debt == 0 || len(score.TopIssues) != 1 {
		t.Fatalf("manual payload score backfill failed: %+v", score)
	}
}
