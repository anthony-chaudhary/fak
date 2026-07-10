package gardenbundle

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

// TestTriageDecisionDispositions pins each emitted-decision shape to its expected
// choicetriage disposition — the lock against the walk silently re-centering the human. A
// ready-command act is the fleet's; a needs-area/needs-kind/likely-dup review is the fleet's
// to drive in a fresh context; only an unset-priority review waits on a person.
func TestTriageDecisionDispositions(t *testing.T) {
	cases := []struct {
		name  string
		dec   WalkDecision
		want  choicetriage.Disposition
		human bool
	}{
		{"act-mark-stale", WalkDecision{ID: 1, Disposition: DispAct, Action: "mark-stale", Command: "gh issue edit 1 --add-label stale", Reason: "idle 90d"}, choicetriage.TakeObvious, false},
		{"act-close-dormant", WalkDecision{ID: 2, Disposition: DispAct, Action: "close-dormant-question", Command: "gh issue close 2 --reason \"not planned\"", Reason: "question idle 60d"}, choicetriage.TakeObvious, false},
		{"review-needs-area", WalkDecision{ID: 3, Disposition: DispReview, Action: "review", Reason: "needs-area"}, choicetriage.FreshContext, false},
		{"review-needs-kind", WalkDecision{ID: 4, Disposition: DispReview, Action: "review", Reason: "needs-kind"}, choicetriage.FreshContext, false},
		{"review-likely-dup", WalkDecision{ID: 5, Disposition: DispReview, Action: "review", Reason: "likely-dup"}, choicetriage.FreshContext, false},
		{"review-multi", WalkDecision{ID: 6, Disposition: DispReview, Action: "review", Reason: "needs-area, likely-dup"}, choicetriage.FreshContext, false},
		{"review-priority", WalkDecision{ID: 7, Disposition: DispReview, Action: "review", Reason: "needs-priority, needs-area"}, choicetriage.HumanResidual, true},
	}
	for _, tc := range cases {
		v := TriageDecision(tc.dec)
		if v.Disposition != tc.want {
			t.Errorf("%s: disposition = %s, want %s (reason: %s)", tc.name, v.Disposition, tc.want, v.Reason)
		}
		if got := DecisionNeedsHuman(tc.dec); got != tc.human {
			t.Errorf("%s: NeedsHuman = %v, want %v", tc.name, got, tc.human)
		}
	}
}

// TestWalkAttentionSplit confirms the split partitions the emitted worklist and that only the
// priority review lands in the needs-human bucket.
func TestWalkAttentionSplit(t *testing.T) {
	plan := WalkPlan{Decisions: []WalkDecision{
		{ID: 7, Disposition: DispReview, Action: "review", Reason: "needs-priority"},
		{ID: 1, Disposition: DispAct, Action: "mark-stale", Command: "gh issue edit 1 --add-label stale"},
		{ID: 3, Disposition: DispReview, Action: "review", Reason: "needs-area"},
	}}
	needHuman, fleet := WalkAttentionSplit(plan)
	if len(needHuman) != 1 || needHuman[0].ID != 7 {
		t.Fatalf("needHuman = %+v, want exactly #7", needHuman)
	}
	if len(fleet) != 2 {
		t.Fatalf("fleetDrives = %d decisions, want 2", len(fleet))
	}
	if len(needHuman)+len(fleet) != len(plan.Decisions) {
		t.Fatalf("split lost a decision")
	}
}

// TestAttentionTriageLine checks the readout line reports the split, and is empty for a walk
// that emitted nothing.
func TestAttentionTriageLine(t *testing.T) {
	if line := AttentionTriageLine(WalkPlan{}); line != "" {
		t.Errorf("empty walk must surface no line, got %q", line)
	}
	plan := WalkPlan{Decisions: []WalkDecision{
		{ID: 7, Disposition: DispReview, Action: "review", Reason: "needs-priority"},
		{ID: 3, Disposition: DispReview, Action: "review", Reason: "needs-area"},
	}}
	line := AttentionTriageLine(plan)
	if !strings.Contains(line, "1 needs-you") || !strings.Contains(line, "1 fleet-drives") {
		t.Errorf("line = %q, want 1 needs-you + 1 fleet-drives", line)
	}
}

// TestGardenWalkTriageEnforced pins the enforce/warn switch.
func TestGardenWalkTriageEnforced(t *testing.T) {
	if !GardenWalkTriageEnforced("enforce") {
		t.Error("enforce must be on")
	}
	for _, off := range []string{"", "warn", "Warn", "off", "1"} {
		if GardenWalkTriageEnforced(off) {
			t.Errorf("%q must be off", off)
		}
	}
}

// TestWalkTriageSelfcheck is the deterministic proof surfaced at the CLI.
func TestWalkTriageSelfcheck(t *testing.T) {
	if err := TriageSelfcheck(); err != nil {
		t.Fatalf("TriageSelfcheck: %v", err)
	}
}

// TestTriageDoesNotChangePlanWalk is the soak guarantee: TriageDecision reads a plan's
// decisions but never mutates PlanWalk's output. A walk folded with and without a triage read
// produces identical decisions.
func TestTriageDoesNotChangePlanWalk(t *testing.T) {
	items := []WalkItem{
		{ID: 1, Score: 9, Disposition: DispReview, Action: "review", Reason: "needs-priority"},
		{ID: 2, Score: 5, Disposition: DispAct, Action: "mark-stale", Command: "gh issue edit 2 --add-label stale", Reason: "idle 90d"},
	}
	before := PlanWalk("issue", items, WalkPolicy{})
	// A triage read over the plan.
	_ = AttentionTriageLine(before)
	after := PlanWalk("issue", items, WalkPolicy{})
	if len(before.Decisions) != len(after.Decisions) {
		t.Fatalf("PlanWalk not deterministic under triage read")
	}
	for i := range before.Decisions {
		if before.Decisions[i] != after.Decisions[i] {
			t.Errorf("decision %d changed: %+v vs %+v", i, before.Decisions[i], after.Decisions[i])
		}
	}
}
