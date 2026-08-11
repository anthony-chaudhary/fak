package sessionregistry

import (
	"testing"
	"time"
)

func TestReconcileStaleBoundsOpenRowsAndDefeatsPIDReuse(t *testing.T) {
	now := time.Date(2026, 8, 11, 20, 0, 0, 0, time.UTC)
	oldStart := now.Add(-time.Hour)
	rows := []Record{
		{RegistrationID: "live", State: StateActive, CreatedAt: oldStart, HeartbeatAt: oldStart, Identity: Identity{PID: 7, ProcessStartedAt: oldStart}},
		{RegistrationID: "reused", State: StateActive, CreatedAt: oldStart, HeartbeatAt: oldStart, Identity: Identity{PID: 8, ProcessStartedAt: oldStart}},
		{RegistrationID: "receipt-only", State: StateRegistered, CreatedAt: oldStart},
		{RegistrationID: "fresh", State: StateRegistered, CreatedAt: now.Add(-time.Minute)},
		{RegistrationID: "done", State: StateCompleted, CreatedAt: oldStart, TerminalAt: oldStart},
	}
	observed := []ObservedProcess{
		{PID: 7, ProcessStartedAt: oldStart},
		{PID: 8, ProcessStartedAt: now.Add(-time.Minute)}, // same PID, different process
	}
	got := ReconcileStale(rows, observed, now, 10*time.Minute)
	if len(got) != 2 {
		t.Fatalf("reconciliations=%+v, want reused and receipt-only", got)
	}
	if got[0].RegistrationID != "receipt-only" || got[0].To != StateUnknown || got[0].Reason == "" {
		t.Fatalf("receipt-only=%+v", got[0])
	}
	if got[1].RegistrationID != "reused" || got[1].To != StateLost || got[1].Reason == "" {
		t.Fatalf("pid reuse=%+v", got[1])
	}
}
