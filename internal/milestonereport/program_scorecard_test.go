package milestonereport

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestProgramScorecardRendersUnderMilestoneAndInJSON(t *testing.T) {
	report := Fold(fixtureMaturity(), fixtureEpics(), FoldOpts{Date: "2026-07-10", Commit: "abc123"})
	card := ProgramScorecard{
		Key:       "mlp",
		Milestone: 17,
		Title:     "MLP first lovable cut",
		Verdict:   "not-yet",
		Witnessed: 1,
		Total:     2,
		Criteria: []ProgramCriterion{
			{Workstream: "B1", Title: "both runtimes", Grade: "witnessed", WitnessRef: "docs/mlp/witnesses/b1.json"},
			{Workstream: "D5", Title: "under ten minutes", Grade: "not-yet", WitnessRef: "docs/mlp/witnesses/d5.json"},
		},
	}
	report = report.WithProgramScorecard(card)

	text := Render(report)
	for _, want := range []string{
		"milestone #17 MLP first lovable cut: NOT-YET (1/2 witnessed)",
		"[B1] both runtimes - witnessed - docs/mlp/witnesses/b1.json",
		"[D5] under ten minutes - not-yet - docs/mlp/witnesses/d5.json",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("milestone render missing %q:\n%s", want, text)
		}
	}

	raw, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	var decoded struct {
		ProgramScorecards []ProgramScorecard `json:"program_scorecards"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded.ProgramScorecards) != 1 || decoded.ProgramScorecards[0].Milestone != 17 {
		t.Fatalf("program scorecard JSON = %+v", decoded.ProgramScorecards)
	}
}

func TestWithProgramScorecardReplacesSameMilestoneKey(t *testing.T) {
	report := Report{}
	report = report.WithProgramScorecard(ProgramScorecard{Key: "mlp", Milestone: 17, Verdict: "not-yet"})
	report = report.WithProgramScorecard(ProgramScorecard{Key: "mlp", Milestone: 17, Verdict: "lovable"})
	if len(report.ProgramScorecards) != 1 || report.ProgramScorecards[0].Verdict != "lovable" {
		t.Fatalf("replacement was not idempotent: %+v", report.ProgramScorecards)
	}
}
