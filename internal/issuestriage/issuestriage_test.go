package issuestriage

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

// TestTriageDispositions pins each surfaced issue-action shape to its expected disposition —
// the single lock that both the issues-garden pane and the garden walk now inherit by
// delegating here. Edit the Signal shape in one place and this fails if the classification
// drifts.
func TestTriageDispositions(t *testing.T) {
	cases := []struct {
		name  string
		act   Action
		want  choicetriage.Disposition
		human bool
	}{
		{"close-dormant", Action{Number: 1, Kind: "close-dormant-question", Reason: "question idle 60d", Command: "gh issue close 1 --reason \"not planned\""}, choicetriage.TakeObvious, false},
		{"mark-stale", Action{Number: 2, Kind: "mark-stale", Reason: "idle 90d", Command: "gh issue edit 2 --add-label stale"}, choicetriage.TakeObvious, false},
		{"review-priority", Action{Number: 3, Kind: "review", Reason: "needs-priority, needs-area"}, choicetriage.HumanResidual, true},
		{"review-area", Action{Number: 4, Kind: "review", Reason: "needs-area"}, choicetriage.FreshContext, false},
		{"review-kind", Action{Number: 5, Kind: "review", Reason: "needs-kind"}, choicetriage.FreshContext, false},
		{"review-dup", Action{Number: 6, Kind: "review", Reason: "likely-dup"}, choicetriage.FreshContext, false},
		{"review-multi", Action{Number: 7, Kind: "review", Reason: "needs-kind, needs-area, likely-dup"}, choicetriage.FreshContext, false},
	}
	for _, tc := range cases {
		v := Triage(tc.act)
		if v.Disposition != tc.want {
			t.Errorf("%s: disposition = %s, want %s (reason: %s)", tc.name, v.Disposition, tc.want, v.Reason)
		}
		if got := NeedsHuman(tc.act); got != tc.human {
			t.Errorf("%s: NeedsHuman = %v, want %v", tc.name, got, tc.human)
		}
	}
}

// TestSelfcheck is the deterministic proof surfaced at the CLI.
func TestSelfcheck(t *testing.T) {
	if err := Selfcheck(); err != nil {
		t.Fatalf("Selfcheck: %v", err)
	}
}

// TestReviewResolveIsActionable confirms a fleet-driven review carries a resolve hint (so the
// pane can render "here's the next move" rather than just "someone look at this").
func TestReviewResolveIsActionable(t *testing.T) {
	v := Triage(Action{Number: 8, Kind: "review", Reason: "needs-area"})
	if v.Resolve == "" {
		t.Error("a FRESH_CONTEXT review should carry a resolve hint")
	}
}
