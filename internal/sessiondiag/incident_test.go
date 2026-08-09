package sessiondiag

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestCaptureIncidentClassificationsAndRedaction(t *testing.T) {
	now := time.Date(2026, 8, 9, 4, 0, 0, 0, time.UTC)
	code := 23
	tests := []struct {
		name, want string
		in         IncidentInput
	}{
		{"direct", "DIRECT_PROCESS_FAILURE", IncidentInput{CapturedAt: now, ProcessUUID: "proc-42", ThreadID: "thread-7", LastLogAt: now.Add(-time.Second), ExitAt: now, ExitKind: ExitFailure, ExitCode: &code, OSFailureEvent: true}},
		{"intentional", "INTENTIONAL_EXIT", IncidentInput{CapturedAt: now, ExitAt: now, ExitKind: ExitIntentional, ExitCode: &code}},
		{"healthy", "HEALTHY_PROCESS", IncidentInput{CapturedAt: now, ProcessObserved: true, WALBytes: 99}},
		{"pressure", "CORRELATED_RUNTIME_PRESSURE", IncidentInput{CapturedAt: now, QueueDropDelta: 4, SlowWriteDelta: 2, WriterCount: 3, WALBytes: 4096}},
		{"missing", "MISSING_EVIDENCE", IncidentInput{CapturedAt: now}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CaptureIncident(tt.in)
			if got.Verdict != tt.want {
				t.Fatalf("verdict=%s want %s", got.Verdict, tt.want)
			}
		})
	}
	got := CaptureIncident(IncidentInput{CapturedAt: now, ProcessUUID: `C:\secret\token`, ThreadID: "ok_7"})
	b, _ := json.Marshal(got)
	if strings.Contains(string(b), "secret") || got.ProcessUUID != "" {
		t.Fatalf("unsafe identifier leaked: %s", b)
	}
	if got.ThreadID != "ok_7" || got.Schema != IncidentSchema {
		t.Fatalf("stable fields missing: %+v", got)
	}
}
