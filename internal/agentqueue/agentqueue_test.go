package agentqueue

import (
	"reflect"
	"testing"
)

func snap() Snapshot {
	//enumlint:exempt This reconciliation fixture deliberately contains only queued and completed intents; state-specific behavior is covered by focused tests below.
	return Snapshot{Schema: Schema, Generation: "g1", Pool: PoolSpec{ID: "p", Min: 1, Desired: 2, Max: 2}, Intents: []Intent{{ID: "b", State: IntentQueued}, {ID: "a", State: IntentQueued}, {ID: "done", State: IntentCompleted}}}
}
func TestDeterministicDuplicateTick(t *testing.T) {
	s := snap()
	a, e := Reconcile(s)
	if e != nil {
		t.Fatal(e)
	}
	b, _ := Reconcile(s)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("changed")
	}
	if a.Start[0].IntentID != "a" || a.Start[1].IntentID != "b" {
		t.Fatalf("%+v", a)
	}
}
func TestHardMaxCountsReservations(t *testing.T) {
	s := snap()
	s.Intents = append(s.Intents, Intent{ID: "x", State: IntentRunning}, Intent{ID: "y", State: IntentRunning})
	s.Attempts = []Attempt{{IntentID: "x", State: AttemptReserved}, {IntentID: "y", State: AttemptRunning}}
	r, e := Reconcile(s)
	if e != nil || r.Observed != 2 || len(r.Start) != 0 || r.Hold[0] != "AT_CAPACITY" {
		t.Fatalf("%+v %v", r, e)
	}
}
func TestCompletedAndFailedNotRespawned(t *testing.T) {
	s := snap()
	s.Pool.Desired = 1
	s.Intents = []Intent{{ID: "done", State: IntentCompleted}, {ID: "no", State: IntentFailed}, {ID: "yes", State: IntentFailed, RetryEligible: true}}
	r, e := Reconcile(s)
	if e != nil || len(r.Start) != 1 || r.Start[0].IntentID != "yes" {
		t.Fatalf("%+v %v", r, e)
	}
}
func TestDrainedQueue(t *testing.T) {
	s := snap()
	s.Intents = []Intent{{ID: "done", State: IntentCompleted}}
	r, _ := Reconcile(s)
	if len(r.Start) != 0 || r.Hold[0] != "QUEUE_EMPTY" {
		t.Fatalf("%+v", r)
	}
}
func TestInvalidBounds(t *testing.T) {
	s := snap()
	s.Pool.Min = 3
	if _, e := Reconcile(s); e == nil {
		t.Fatal("want error")
	}
}
