package agentqueue

import (
	"reflect"
	"testing"
	"time"
)

func TestReconcileRestartLaunchingHoldsWithoutReplacement(t *testing.T) {
	now := time.Date(2026, 9, 23, 13, 0, 0, 0, time.UTC)
	startedAt := now.Add(-time.Minute)
	tests := []struct {
		name     string
		deadline time.Time
		pid      int
		startAt  time.Time
	}{
		{name: "registered", deadline: now.Add(time.Minute), pid: 4242, startAt: startedAt},
		{name: "unregistered", deadline: now.Add(time.Minute)},
		{name: "expired registered", deadline: now.Add(-time.Second), pid: 4242, startAt: startedAt},
		{name: "expired unregistered", deadline: now.Add(-time.Second)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original := Attempt{
				ID: "attempt-launching", IntentID: "intent-launching", State: AttemptLaunching,
				Nonce: "nonce-a", LaunchDeadline: tt.deadline, PID: tt.pid, StartedAt: tt.startAt,
			}
			snapshot := Snapshot{
				Schema:     Schema,
				Generation: "generation-restart",
				Pool:       PoolSpec{ID: "pool", Min: 0, Desired: 1, Max: 1},
				Intents: []Intent{
					{ID: "intent-launching", State: IntentQueued},
					{ID: "intent-competing", State: IntentQueued},
				},
				Attempts: []Attempt{original},
			}
			livenessCalls := 0
			reconciliation, updated, err := ReconcileRestart(snapshot, func(int) bool {
				livenessCalls++
				return false
			}, RestartOptions{Now: now})
			if err != nil {
				t.Fatalf("ReconcileRestart: %v", err)
			}
			if len(reconciliation.Held) != 1 || reconciliation.Held[0].IntentID != "intent-launching" {
				t.Fatalf("held dispositions = %+v, want launching intent", reconciliation.Held)
			}
			if len(reconciliation.Adopted) != 0 || len(reconciliation.Replaced) != 0 {
				t.Fatalf("unexpected restart action: adopted=%+v replaced=%+v", reconciliation.Adopted, reconciliation.Replaced)
			}
			if livenessCalls != 0 {
				t.Fatalf("liveness called %d times for indeterminate launch", livenessCalls)
			}
			if len(updated.Attempts) != 1 || !reflect.DeepEqual(updated.Attempts[0], original) {
				t.Fatalf("launching attempt mutated: got %+v, want %+v", updated.Attempts, original)
			}
			receipt, err := Reconcile(updated)
			if err != nil {
				t.Fatalf("Reconcile updated snapshot: %v", err)
			}
			if receipt.Observed != 1 || len(receipt.Start) != 0 {
				t.Fatalf("capacity after restart = observed %d, starts %+v; want 1/none", receipt.Observed, receipt.Start)
			}
		})
	}
}

func TestReconcileRestartLaunchingWinsMixedAttemptOrder(t *testing.T) {
	now := time.Date(2026, 9, 23, 13, 0, 0, 0, time.UTC)
	launching := Attempt{
		ID: "attempt-launching", IntentID: "intent-a", State: AttemptLaunching,
		Nonce: "nonce-a", LaunchDeadline: now.Add(time.Minute), PID: 4242, StartedAt: now.Add(-time.Minute),
	}
	for _, laterState := range []AttemptState{AttemptReserved, AttemptRunning} {
		t.Run(string(laterState), func(t *testing.T) {
			snapshot := Snapshot{
				Schema:     Schema,
				Generation: "generation-mixed",
				Pool:       PoolSpec{ID: "pool", Min: 0, Desired: 1, Max: 1},
				Intents: []Intent{
					{ID: "intent-a", State: IntentRunning},
					{ID: "intent-competing", State: IntentQueued},
				},
				Attempts: []Attempt{
					launching,
					{ID: "attempt-later", IntentID: "intent-a", State: laterState, PID: 9999},
				},
			}
			reconciliation, updated, err := ReconcileRestart(snapshot, func(int) bool { return false }, RestartOptions{Now: now})
			if err != nil {
				t.Fatalf("ReconcileRestart: %v", err)
			}
			if len(reconciliation.Held) != 1 || reconciliation.Held[0].IntentID != "intent-a" || len(reconciliation.Replaced) != 0 {
				t.Fatalf("restart dispositions = held %+v, replaced %+v", reconciliation.Held, reconciliation.Replaced)
			}
			if got := terminalAttempt(t, updated, launching.ID); !reflect.DeepEqual(got, launching) {
				t.Fatalf("launching attempt mutated: got %+v, want %+v", got, launching)
			}
			receipt, err := Reconcile(updated)
			if err != nil {
				t.Fatalf("Reconcile updated snapshot: %v", err)
			}
			if receipt.Observed == 0 || len(receipt.Start) != 0 {
				t.Fatalf("mixed capacity after restart = observed %d, starts %+v; want active/no starts", receipt.Observed, receipt.Start)
			}
		})
	}
}
