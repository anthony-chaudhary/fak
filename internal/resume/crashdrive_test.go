package resume

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

// crashed is a small constructor for a CRASHED Classified keyed on id with an optional drive.
func crashed(id string, drive *sessionjournal.DriveCarry) sessionjournal.Classified {
	return sessionjournal.Classified{
		Session: sessionjournal.Session{ID: id, Drive: drive},
		Status:  sessionjournal.StatusCrashed,
		Reason:  sessionjournal.ReasonMachineReboot,
	}
}

// TestResolveCrashedDriveUUIDKeyed: a CRASHED row keyed on a transcript UUID resolves its
// gateway trace across the wall and carries its drive — the core done-condition.
func TestResolveCrashedDriveUUIDKeyed(t *testing.T) {
	join := IdentityJoin{
		TraceByUUID: map[string]string{"uuid-a": "trace-a"},
		UUIDByTrace: map[string]string{"trace-a": "uuid-a"},
	}
	got := ResolveCrashedDrive([]sessionjournal.Classified{crashed("uuid-a", &sessionjournal.DriveCarry{TurnsLeft: 2})}, join, nil)
	if len(got) != 1 {
		t.Fatalf("want 1 resolved, got %d", len(got))
	}
	r := got[0]
	if r.TranscriptUUID != "uuid-a" || r.GatewayTrace != "trace-a" {
		t.Fatalf("uuid-keyed row should resolve its trace, got uuid=%q trace=%q", r.TranscriptUUID, r.GatewayTrace)
	}
	if r.Drive == nil || r.Drive.TurnsLeft != 2 {
		t.Fatalf("carried drive should survive the join, got %+v", r.Drive)
	}
}

// TestResolveCrashedDriveTraceKeyed: a CRASHED row keyed on a gateway trace resolves its
// transcript UUID (the reverse direction).
func TestResolveCrashedDriveTraceKeyed(t *testing.T) {
	join := IdentityJoin{
		TraceByUUID: map[string]string{"uuid-b": "trace-b"},
		UUIDByTrace: map[string]string{"trace-b": "uuid-b"},
	}
	r := ResolveCrashedDrive([]sessionjournal.Classified{crashed("trace-b", nil)}, join, nil)[0]
	if r.GatewayTrace != "trace-b" || r.TranscriptUUID != "uuid-b" {
		t.Fatalf("trace-keyed row should resolve its uuid, got trace=%q uuid=%q", r.GatewayTrace, r.TranscriptUUID)
	}
	if r.Drive != nil {
		t.Fatalf("a record with no carry should resolve a nil drive, got %+v", r.Drive)
	}
}

// TestResolveCrashedDriveHold: an operator hold recorded under the session's UUID surfaces on
// the resolved row even when the record was keyed on the trace.
func TestResolveCrashedDriveHold(t *testing.T) {
	join := IdentityJoin{
		TraceByUUID: map[string]string{"uuid-c": "trace-c"},
		UUIDByTrace: map[string]string{"trace-c": "uuid-c"},
	}
	holds := map[string]WatchdogDriveState{"uuid-c": DriveStopped}
	r := ResolveCrashedDrive([]sessionjournal.Classified{crashed("trace-c", nil)}, join, holds)[0]
	if r.Hold != DriveStopped {
		t.Fatalf("hold recorded under the resolved uuid should surface, got %q", r.Hold)
	}
	if !r.Hold.HeldByOperator() {
		t.Fatalf("a stopped session must read HeldByOperator()")
	}
}

// TestResolveCrashedDriveUnmappable: an id in neither direction of the map degrades to empty
// counterparts (no panic), still carrying its drive; a nil holds map reads no hold.
func TestResolveCrashedDriveUnmappable(t *testing.T) {
	r := ResolveCrashedDrive(
		[]sessionjournal.Classified{crashed("orphan-x", &sessionjournal.DriveCarry{TokensLeft: 7})},
		IdentityJoin{}, // both directions nil
		nil,
	)[0]
	if r.SessionID != "orphan-x" {
		t.Fatalf("session id should be carried through, got %q", r.SessionID)
	}
	if r.GatewayTrace != "" || r.TranscriptUUID != "" {
		t.Fatalf("unmappable id must leave counterparts empty, got trace=%q uuid=%q", r.GatewayTrace, r.TranscriptUUID)
	}
	if r.Drive == nil || r.Drive.TokensLeft != 7 {
		t.Fatalf("drive should still attach on an unmappable row, got %+v", r.Drive)
	}
	if r.Hold != "" {
		t.Fatalf("nil holds map must read no hold, got %q", r.Hold)
	}
}
