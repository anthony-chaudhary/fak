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
	handoffs := map[string]LaunchHandoff{
		"start:docs":    {AttemptID: "start:docs", Nonce: "nonce:docs", StatePath: "/state/queue.json"},
		"start:gateway": {AttemptID: "start:gateway", Nonce: "nonce:gateway", StatePath: "/state/queue.json"},
	}
	runner := &recordingRunner{}
	receipts, err := Actuate(context.Background(), "fak", snapshot, starts, handoffs, runner)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]string{
		{"fak", "dispatch", "tick", "--target-issue", "101", "--lane", "docs", "--lease-id", "start:docs", "--attempt-id", "start:docs", "--launch-nonce", "nonce:docs", "--queue-state", "/state/queue.json", "--live", "--json"},
		{"fak", "dispatch", "tick", "--target-issue", "202", "--lane", "gateway", "--lease-id", "start:gateway", "--attempt-id", "start:gateway", "--launch-nonce", "nonce:gateway", "--queue-state", "/state/queue.json", "--live", "--json"},
	}
	if !reflect.DeepEqual(runner.calls, want) {
		t.Fatalf("calls = %#v, want %#v", runner.calls, want)
	}
	if len(receipts) != 2 || receipts[0].IdempotencyKey != "start:docs" || receipts[1].IdempotencyKey != "start:gateway" {
		t.Fatalf("receipts = %#v", receipts)
	}
	if receipts[0].AttemptID != "start:docs" || receipts[0].Nonce != "nonce:docs" {
		t.Fatalf("receipt handoff = %#v", receipts[0])
	}
}

func TestActuateRejectsUnfencedStartBeforeExecution(t *testing.T) {
	snapshot := Snapshot{Intents: []Intent{{ID: "docs", Launch: LaunchSpec{Issue: 101, Lane: "docs"}}}}
	runner := &recordingRunner{}
	_, err := Actuate(context.Background(), "fak", snapshot, []StartAction{{IntentID: "docs", IdempotencyKey: "start:docs"}}, nil, runner)
	if err == nil {
		t.Fatal("Actuate launched a start without a fenced handoff")
	}
	if len(runner.calls) != 0 {
		t.Fatalf("executed %#v", runner.calls)
	}
}

func TestActuateRejectsUnroutableIntentBeforeExecution(t *testing.T) {
	runner := &recordingRunner{}
	start := StartAction{IntentID: "bad", IdempotencyKey: "start:bad"}
	handoffs := map[string]LaunchHandoff{"start:bad": {AttemptID: "start:bad", Nonce: "nonce:bad", StatePath: "/state/queue.json"}}
	_, err := Actuate(context.Background(), "fak", Snapshot{Intents: []Intent{{ID: "bad"}}}, []StartAction{start}, handoffs, runner)
	if err == nil {
		t.Fatal("Actuate accepted intent without issue/lane")
	}
	if len(runner.calls) != 0 {
		t.Fatalf("executed %#v", runner.calls)
	}
}
