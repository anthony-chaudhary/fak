package resume

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

func TestBuildResumeAnchorUsesWitnessedSessionCurve(t *testing.T) {
	obj := trajctl.Objective{ID: "issue-2551", Statement: "prevent cascade drift", Status: trajctl.StatusActive, Plan: []trajctl.PlanPhase{{ID: "p1", Title: "wire anchor"}}}
	st := trajctl.Fold([]trajctl.Row{
		trajctl.ObjectiveRecord(obj),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: obj.ID, SessionID: "sess", Method: trajctl.CommitScorerMethod, Witness: trajctl.W3, Value: .2, Version: "1", UnixMillis: 1}),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: obj.ID, SessionID: "sess", Method: trajctl.CommitScorerMethod, Witness: trajctl.W3, Value: .7, Version: "1", UnixMillis: 2}),
	})
	got := BuildResumeAnchor("sess", st)
	if !got.Present || got.ObjectiveID != obj.ID || got.Curve == nil || got.Curve.Latest != .7 || got.Curve.Delta < .49 || got.Curve.Delta > .51 {
		t.Fatalf("anchor=%+v curve=%+v, want witnessed curve", got, got.Curve)
	}
	for _, want := range []string{"prevent cascade drift", "latest=0.70", "delta=+0.50", "p1 wire anchor"} {
		if !strings.Contains(got.Prompt(), want) {
			t.Fatalf("prompt missing %q: %s", want, got.Prompt())
		}
	}
}

func TestBuildResumeAnchorRejectsSelfReport(t *testing.T) {
	obj := trajctl.Objective{ID: "o", Statement: "self claim", Status: trajctl.StatusActive}
	st := trajctl.Fold([]trajctl.Row{trajctl.ObjectiveRecord(obj), trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: obj.ID, SessionID: "sess", Method: "self", Version: "1", Witness: trajctl.W0, Value: 1})})
	if got := BuildResumeAnchor("sess", st); got.Present || got.Prompt() != "" {
		t.Fatalf("self-report anchor=%+v, want absent", got)
	}
}
