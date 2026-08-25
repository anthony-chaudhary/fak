package agentqueue

import (
	"context"
	"reflect"
	"testing"
)

type recordingRunner struct{ calls [][]string }

func (r *recordingRunner) Run(_ context.Context, name string, args ...string) error {
	r.calls = append(r.calls, append([]string{name}, args...))
	return nil
}

func TestActuateRoutesDisjointReservationsThroughGuardedDispatch(t *testing.T) {
	snapshot := Snapshot{Intents: []Intent{
		{ID: "docs", Launch: LaunchSpec{Issue: 101, Lane: "docs"}},
		{ID: "gateway", Launch: LaunchSpec{Issue: 202, Lane: "gateway"}},
	}}
	starts := []StartAction{
		{IntentID: "docs", IdempotencyKey: "start:docs"},
		{IntentID: "gateway", IdempotencyKey: "start:gateway"},
	}
	runner := &recordingRunner{}
	receipts, err := Actuate(context.Background(), "fak", snapshot, starts, runner)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]string{
		{"fak", "dispatch", "tick", "--target-issue", "101", "--lane", "docs", "--lease-id", "start:docs", "--live", "--json"},
		{"fak", "dispatch", "tick", "--target-issue", "202", "--lane", "gateway", "--lease-id", "start:gateway", "--live", "--json"},
	}
	if !reflect.DeepEqual(runner.calls, want) {
		t.Fatalf("calls = %#v, want %#v", runner.calls, want)
	}
	if len(receipts) != 2 || receipts[0].IdempotencyKey != "start:docs" || receipts[1].IdempotencyKey != "start:gateway" {
		t.Fatalf("receipts = %#v", receipts)
	}
}

func TestActuateRejectsUnroutableIntentBeforeExecution(t *testing.T) {
	runner := &recordingRunner{}
	_, err := Actuate(context.Background(), "fak", Snapshot{Intents: []Intent{{ID: "bad"}}}, []StartAction{{IntentID: "bad"}}, runner)
	if err == nil {
		t.Fatal("Actuate accepted intent without issue/lane")
	}
	if len(runner.calls) != 0 {
		t.Fatalf("executed %#v", runner.calls)
	}
}
