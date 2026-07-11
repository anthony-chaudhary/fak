package dispatchtick

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

func TestDispatchChildCurveReadsWitnessedIssueCurve(t *testing.T) {
	root := t.TempDir()
	ledger := filepath.Join(root, trajctl.DefaultLedgerRel)
	obj := trajctl.Objective{ID: "issue-2550", Statement: "bind child objective", Status: trajctl.StatusActive, Scorers: []string{"commit-progress"}}
	if err := trajctl.Append(ledger, trajctl.ObjectiveRecord(obj)); err != nil {
		t.Fatal(err)
	}
	for i, value := range []float64{0.25, 0.75} {
		row := trajctl.ScoreRow{ObjectiveID: obj.ID, Method: trajctl.CommitScorerMethod, Version: "1", Witness: trajctl.W3, Value: value, UnixMillis: int64(i + 1)}
		if err := trajctl.Append(ledger, trajctl.ScoreRecord(row)); err != nil {
			t.Fatal(err)
		}
	}
	got := ChildCurve(root, 2550)
	if got["present"] != true || got["objective_id"] != "issue-2550" || got["latest"] != 0.75 || got["delta"] != 0.5 {
		t.Fatalf("curve=%v, want witnessed issue curve", got)
	}
}

func TestDispatchChildCurveReportsMissingEvidence(t *testing.T) {
	got := ChildCurve(t.TempDir(), 2550)
	if got["present"] != false || got["reason"] == "" {
		t.Fatalf("curve=%v, want explicit absence", got)
	}
}
