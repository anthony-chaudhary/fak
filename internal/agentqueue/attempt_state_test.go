package agentqueue

import (
	"reflect"
	"strings"
	"testing"
)

// TestReconcileAttemptStateAdmission ensures unsupported states cannot silently
// free a slot and schedule another attempt for the same queued intent.
// fak-test:runtime fast est=1ms lane=default
//
//enumlint:exempt Unsupported state literals are deliberate refusal fixtures; all five declared states have positive controls.
func TestReconcileAttemptStateAdmission(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name    string
		state   AttemptState
		active  bool
		invalid bool
	}{
		{name: "reserved", state: AttemptReserved, active: true},
		{name: "launching", state: AttemptLaunching, active: true},
		{name: "running", state: AttemptRunning, active: true},
		{name: "succeeded", state: AttemptSucceeded},
		{name: "failed", state: AttemptFailed},
		{name: "empty", state: "", invalid: true},
		{name: "misspelled", state: "runing", invalid: true},
		{name: "future", state: "future-state", invalid: true},
		{name: "undeclared held", state: "held", invalid: true},
		{name: "undeclared indeterminate", state: "indeterminate", invalid: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := Snapshot{
				Schema: Schema, Generation: "g1",
				Pool:    PoolSpec{ID: "pool", Desired: 1, Max: 1},
				Intents: []Intent{{ID: "job", State: IntentQueued}},
			}
			baseline, err := Reconcile(s)
			if err != nil || len(baseline.Start) != 1 || baseline.Start[0].IntentID != "job" {
				t.Fatalf("fixture does not expose replacement admission: %+v, %v", baseline, err)
			}
			s.Attempts = []Attempt{{ID: "attempt", IntentID: "job", State: tc.state}}
			got, err := Reconcile(s)
			if tc.invalid {
				if err == nil || !strings.Contains(err.Error(), "unsupported attempt state") {
					t.Fatalf("unsupported state %q admitted: receipt=%+v err=%v", tc.state, got, err)
				}
				if !reflect.DeepEqual(got, Receipt{}) {
					t.Fatalf("unsupported state returned a partial admission receipt: %+v", got)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			want := baseline
			if tc.active {
				want.Observed = 1
				want.Start = []StartAction{}
				want.Hold = []string{"AT_CAPACITY"}
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("known state %q changed admission: got %+v, want %+v", tc.state, got, want)
			}
		})
	}
}
