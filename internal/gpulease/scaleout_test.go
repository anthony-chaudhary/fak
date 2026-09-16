package gpulease

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fixedProbe returns an injected Progress func that always reports probe,
// letting the decision matrix be driven without a live holder.
func fixedProbe(probe HolderProbe) func(string) HolderProbe {
	return func(string) HolderProbe { return probe }
}

func TestDecideScaleOutDecisionMatrix(t *testing.T) {
	now := time.Date(2026, 9, 15, 12, 0, 0, 0, time.UTC)
	attachOwner := func(string) (OwnerCapacity, bool) {
		return OwnerCapacity{Held: true, PID: 4242, Attachable: true, Capacity: 4, InUse: 2}, true
	}

	tests := []struct {
		name         string
		err          error
		opts         ScaleOutOptions
		wantVerdict  ScaleOutVerdict
		wantReason   string
		wantRetrySec int
		wantFree     int
	}{
		{
			name:         "dead holder with a wait bound retries immediately",
			err:          &BusyError{Path: "/tmp/x", PID: 999999},
			opts:         ScaleOutOptions{Progress: fixedProbe(HolderProbe{Verdict: HolderProgressDead, Held: true, PID: 999999}), WaitBound: time.Minute},
			wantVerdict:  ScaleOutWaitWithBound,
			wantReason:   "retry",
			wantRetrySec: 0,
		},
		{
			name:        "dead holder with no wait bound refuses naming retry",
			err:         &BusyError{Path: "/tmp/x", PID: 999999},
			opts:        ScaleOutOptions{Progress: fixedProbe(HolderProbe{Verdict: HolderProgressDead, Held: true, PID: 999999})},
			wantVerdict: ScaleOutRefuse,
			wantReason:  "holder gone, retry",
		},
		{
			name:        "free lease with nil err and a wait bound retries immediately",
			err:         nil,
			opts:        ScaleOutOptions{Progress: fixedProbe(HolderProbe{Verdict: HolderProgressUnknown, Held: false, PID: HolderNotBusy}), WaitBound: time.Minute},
			wantVerdict: ScaleOutWaitWithBound,
			wantReason:  "retry",
		},
		{
			name:         "progressing unattachable holder with a wait bound waits",
			err:          &BusyError{Path: "/tmp/x", PID: 4242},
			opts:         ScaleOutOptions{Progress: fixedProbe(HolderProbe{Verdict: HolderProgressLiveProgressing, Held: true, PID: 4242, Detail: "busy"}), WaitBound: 10 * time.Minute},
			wantVerdict:  ScaleOutWaitWithBound,
			wantReason:   "progressing",
			wantRetrySec: int(DefaultWaitBound / time.Second),
		},
		{
			name:        "progressing unattachable holder with no bound refuses",
			err:         &BusyError{Path: "/tmp/x", PID: 4242},
			opts:        ScaleOutOptions{Progress: fixedProbe(HolderProbe{Verdict: HolderProgressLiveProgressing, Held: true, PID: 4242, Detail: "busy"})},
			wantVerdict: ScaleOutRefuse,
			wantReason:  "no wait bound",
		},
		{
			name:        "attachable owner with a free slot attaches",
			err:         &BusyError{Path: "/tmp/x", PID: 4242},
			opts:        ScaleOutOptions{Progress: fixedProbe(HolderProbe{Verdict: HolderProgressLiveProgressing, Held: true, PID: 4242}), OwnerCapacityProbe: attachOwner},
			wantVerdict: ScaleOutAttachOwner,
			wantFree:    2,
			wantReason:  "free slot",
		},
		{
			name: "attachable owner with zero free slots cannot attach",
			err:  &BusyError{Path: "/tmp/x", PID: 4242},
			opts: ScaleOutOptions{Progress: fixedProbe(HolderProbe{Verdict: HolderProgressLiveProgressing, Held: true, PID: 4242}), OwnerCapacityProbe: func(string) (OwnerCapacity, bool) {
				return OwnerCapacity{Held: true, PID: 4242, Attachable: true, Capacity: 2, InUse: 2}, true
			}},
			wantVerdict: ScaleOutRefuse,
		},
		{
			name:        "unknown holder refuses fail-closed",
			err:         &BusyError{Path: "/tmp/x", PID: HolderUnknownPID},
			opts:        ScaleOutOptions{Progress: fixedProbe(HolderProbe{Verdict: HolderProgressUnknown, Held: true, PID: HolderUnknownPID})},
			wantVerdict: ScaleOutRefuse,
			wantReason:  "unreadable",
		},
		{
			name:        "stalled holder refuses without an attach surface",
			err:         &BusyError{Path: "/tmp/x", PID: 4242},
			opts:        ScaleOutOptions{Progress: fixedProbe(HolderProbe{Verdict: HolderProgressStalled, Held: true, PID: 4242})},
			wantVerdict: ScaleOutRefuse,
			wantReason:  "stalled",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			opts := tc.opts
			opts.Path = filepath.Join(t.TempDir(), "gpu.lease")
			opts.Now = func() time.Time { return now }
			rec := DecideScaleOut(tc.err, opts)
			if rec.Verdict != tc.wantVerdict {
				t.Fatalf("verdict = %q, want %q (reason: %s)", rec.Verdict, tc.wantVerdict, rec.Reason)
			}
			if tc.wantReason != "" && !strings.Contains(rec.Reason, tc.wantReason) {
				t.Errorf("reason %q does not contain %q", rec.Reason, tc.wantReason)
			}
			if tc.wantRetrySec != 0 && rec.RetryAfterSeconds != tc.wantRetrySec {
				t.Errorf("retry_after_seconds = %d, want %d", rec.RetryAfterSeconds, tc.wantRetrySec)
			}
			if tc.wantFree != 0 && rec.FreeSlots != tc.wantFree {
				t.Errorf("free_slots = %d, want %d", rec.FreeSlots, tc.wantFree)
			}
			if rec.Schema != scaleOutSchema {
				t.Errorf("schema = %q, want %q", rec.Schema, scaleOutSchema)
			}
			if rec.At.IsZero() {
				t.Error("receipt At must be set")
			}
		})
	}
}

func TestScaleOutReceiptJSONRoundTrip(t *testing.T) {
	rec := ScaleOutReceipt{
		Schema:            scaleOutSchema,
		Verdict:           ScaleOutWaitWithBound,
		Reason:            "holder 4242 is progressing",
		Path:              "/tmp/gpu.lease",
		Holder:            OwnerCapacity{Held: true, PID: 4242, Progress: HolderProgressLiveProgressing, Capacity: 4, InUse: 1},
		RetryAfter:        "120s",
		RetryAfterSeconds: 120,
		At:                time.Date(2026, 9, 15, 12, 0, 0, 0, time.UTC),
	}
	line, err := rec.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	if strings.Contains(line, "\n") {
		t.Fatalf("receipt JSON must be single-line, got: %q", line)
	}
	var back ScaleOutReceipt
	if err := json.Unmarshal([]byte(line), &back); err != nil {
		t.Fatalf("round-trip unmarshal: %v", err)
	}
	if back.Schema != rec.Schema || back.Verdict != rec.Verdict || back.Reason != rec.Reason {
		t.Errorf("round-trip mismatch: %+v vs %+v", back, rec)
	}
	if back.RetryAfterSeconds != rec.RetryAfterSeconds || back.Holder.PID != rec.Holder.PID {
		t.Errorf("round-trip numeric mismatch: %+v", back)
	}
	if !back.At.Equal(rec.At) {
		t.Errorf("round-trip At = %v, want %v", back.At, rec.At)
	}
	if s := rec.String(); !strings.Contains(s, string(ScaleOutWaitWithBound)) || !strings.Contains(s, "120s") {
		t.Errorf("String() = %q, want verdict and retry bound", s)
	}
}

func TestOwnerCapacitySidecarRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")
	if _, ok := ReadOwnerCapacity(path); ok {
		t.Fatal("ReadOwnerCapacity succeeded with no sidecar published")
	}
	rec := OwnerCapacityRecord{
		PID:        4242,
		Capacity:   4,
		InUse:      1,
		AttachHint: "unix:///tmp/fak-serve.sock",
		UpdatedAt:  time.Now().UTC(),
	}
	if err := PublishOwnerCapacity(path, rec); err != nil {
		t.Fatalf("Publish: %v", err)
	}
	got, ok := ReadOwnerCapacity(path)
	if !ok {
		t.Fatal("Read back published sidecar failed")
	}
	if got.PID != 4242 || got.Capacity != 4 || got.InUse != 1 || got.AttachHint != rec.AttachHint {
		t.Fatalf("round-trip mismatch: %+v", got)
	}
	if got.Schema != ownerCapacitySchema {
		t.Errorf("schema = %q, want %q", got.Schema, ownerCapacitySchema)
	}
	cap := got.AsOwnerCapacity()
	if !cap.Attachable || cap.FreeSlots() != 3 {
		t.Fatalf("expected attachable with 3 free slots, got %+v", cap)
	}

	if err := ClearOwnerCapacity(path); err != nil {
		t.Fatalf("Clear: %v", err)
	}
	if _, ok := ReadOwnerCapacity(path); ok {
		t.Fatal("ReadOwnerCapacity succeeded after Clear")
	}
	// Clearing an absent sidecar is success (idempotent post-condition).
	if err := ClearOwnerCapacity(path); err != nil {
		t.Fatalf("Clear absent: %v", err)
	}
}

func TestOwnerCapacityStaleAndCorruptAreFailClosed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")

	// Stale: UpdatedAt older than OwnerCapacityStaleAfter must be rejected.
	if err := PublishOwnerCapacity(path, OwnerCapacityRecord{
		PID: 4242, Capacity: 4, InUse: 0, AttachHint: "unix:///x",
		UpdatedAt: time.Now().Add(-2 * OwnerCapacityStaleAfter),
	}); err != nil {
		t.Fatalf("Publish stale: %v", err)
	}
	if _, ok := ReadOwnerCapacity(path); ok {
		t.Fatal("stale sidecar must be rejected")
	}

	// Corrupt: unparseable bytes must never yield an attachable owner.
	if err := os.WriteFile(OwnerCapacityPath(path), []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, ok := ReadOwnerCapacity(path); ok {
		t.Fatal("corrupt sidecar must be rejected")
	}
	// A corrupt sidecar also must not upgrade a real decision to ATTACH_OWNER.
	err := error(&BusyError{Path: path, PID: os.Getpid()})
	rec := DecideScaleOut(err, ScaleOutOptions{
		Path:     path,
		Progress: fixedProbe(HolderProbe{Verdict: HolderProgressLiveProgressing, Held: true, PID: os.Getpid()}),
	})
	if rec.Verdict == ScaleOutAttachOwner {
		t.Fatalf("corrupt sidecar must never produce ATTACH_OWNER, got %+v", rec)
	}

	// Wrong schema is rejected even with a fresh timestamp.
	if err := os.WriteFile(OwnerCapacityPath(path), []byte(`{"schema":"other/9","pid":1,"capacity":9,"in_use":0,"attach_hint":"x","updated_at":"2026-09-15T12:00:00Z"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, ok := ReadOwnerCapacity(path); ok {
		t.Fatal("foreign-schema sidecar must be rejected")
	}
}

func TestDecideScaleOutUsesPublishedSidecarAsDefaultProbe(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")
	if err := PublishOwnerCapacity(path, OwnerCapacityRecord{
		PID: os.Getpid(), Capacity: 4, InUse: 1, AttachHint: "unix:///sock",
		UpdatedAt: time.Now().UTC(),
	}); err != nil {
		t.Fatal(err)
	}
	rec := DecideScaleOut(&BusyError{Path: path, PID: os.Getpid()}, ScaleOutOptions{
		Path:     path,
		Progress: fixedProbe(HolderProbe{Verdict: HolderProgressLiveProgressing, Held: true, PID: os.Getpid()}),
	})
	if rec.Verdict != ScaleOutAttachOwner {
		t.Fatalf("a published sidecar must make the owner attachable without an explicit probe, got %+v", rec)
	}
	if rec.FreeSlots != 3 {
		t.Fatalf("free slots = %d, want 3", rec.FreeSlots)
	}
}
