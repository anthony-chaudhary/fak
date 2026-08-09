package ctxplan

import "testing"

func TestReconcilePlan(t *testing.T) {
	one := StepPin{StepID: "one", Text: "inspect"}
	two := StepPin{StepID: "two", Text: "ship"}
	base := NewPlanPin([]StepPin{one, two})

	tests := []struct {
		name      string
		after     PlanPin
		step      string
		want      ObjectiveOutcome
		wantMatch bool
	}{
		{name: "preserved", after: NewPlanPin([]StepPin{one, two}), step: "one", want: ObjectivePreserved, wantMatch: true},
		{name: "reorder", after: NewPlanPin([]StepPin{two, one}), step: "one", want: ObjectiveDrifted},
		{name: "per-step drift", after: NewPlanPin([]StepPin{{StepID: "one", Text: "guess"}, two}), step: "one", want: ObjectiveDrifted},
		{name: "drop", after: NewPlanPin([]StepPin{one}), step: "two", want: ObjectiveDropped},
		{name: "add", after: NewPlanPin([]StepPin{one, two, {StepID: "three", Text: "report"}}), step: "three", want: ObjectiveQueryUser},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ReconcilePlan(base, tc.after)
			if got.AllMatch != tc.wantMatch {
				t.Fatalf("AllMatch=%v, want %v", got.AllMatch, tc.wantMatch)
			}
			if got.Decisions[tc.step].Outcome != tc.want {
				t.Fatalf("%s outcome=%q, want %q", tc.step, got.Decisions[tc.step].Outcome, tc.want)
			}
			if tc.want != ObjectivePreserved && !tc.want.Refusal() {
				t.Fatalf("non-preserved outcome %q is not refusal-category", tc.want)
			}
		})
	}
}

func TestPlanLogReplay(t *testing.T) {
	one := StepPin{StepID: "one", Text: "inspect"}
	base := NewPlanPin([]StepPin{one})
	changed := NewPlanPin([]StepPin{{StepID: "one", Text: "guess"}})
	results, allMatch := (PlanLog{Entries: []PlanLogEntry{
		{Before: base, After: base},
		{Before: base, After: changed},
	}}).Replay()
	if allMatch {
		t.Fatal("replay hid a drifted transition")
	}
	if len(results) != 2 || !results[0].AllMatch || results[1].AllMatch {
		t.Fatalf("unexpected replay results: %#v", results)
	}
}
