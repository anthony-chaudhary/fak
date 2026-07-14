package hostfault

import (
	"path/filepath"
	"testing"
	"time"
)

func TestHostTerminationLedgerAndCorrelation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "host-crashes.jsonl")
	wave := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	marker := HostTerminationMarker{Schema: HostTerminationSchema, ControlType: "CTRL_CLOSE_EVENT", GuardPID: 42, ConsoleSession: 7, ObservedAt: wave.Add(500 * time.Millisecond).Format(time.RFC3339Nano)}
	if err := AppendHostTermination(path, marker); err != nil {
		t.Fatal(err)
	}
	got, err := ReadHostTerminations(path)
	if err != nil {
		t.Fatal(err)
	}
	found, ok := CorrelateHostTermination(got, wave, time.Second)
	if !ok || found != marker {
		t.Fatalf("correlation=(%+v,%v), want marker", found, ok)
	}
}

func TestForcedTerminationWithoutMarkerIsExternalUnknown(t *testing.T) {
	if _, ok := CorrelateHostTermination(nil, time.Now(), time.Second); ok {
		t.Fatal("markerless forced-owner termination must remain EXTERNAL_UNKNOWN")
	}
}
