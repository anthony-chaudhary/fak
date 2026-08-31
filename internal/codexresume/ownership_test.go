package codexresume

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type fixedOwnershipProbe struct {
	witness ownershipWitness
	err     error
}

func (p fixedOwnershipProbe) inspect(string) (ownershipWitness, error) {
	return p.witness, p.err
}

func makeWriterLock(t *testing.T, home, threadID string) string {
	t.Helper()
	path := filepath.Join(home, "thread-writer-locks", threadID+".lock")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestWriterOwnershipReceiptPreventsPIDReuseMatch(t *testing.T) {
	threadID := testThreadIDOne
	lockPath := makeWriterLock(t, t.TempDir(), threadID)
	first := inspectWriterOwnership(threadID, lockPath, fixedOwnershipProbe{witness: ownershipWitness{
		source:     "test_native_witness",
		conclusive: true,
		owners: []processOwner{{
			pid: 42, startTime: "2026-08-31T10:00:00Z", startToken: 100, image: `C:\\codex.exe`,
		}},
	}})
	second := inspectWriterOwnership(threadID, lockPath, fixedOwnershipProbe{witness: ownershipWitness{
		source:     "test_native_witness",
		conclusive: true,
		owners: []processOwner{{
			pid: 42, startTime: "2026-08-31T11:00:00Z", startToken: 101, image: `C:\\codex.exe`,
		}},
	}})
	if first.Verdict != WriterOwnershipLiveOwner || second.Verdict != WriterOwnershipLiveOwner {
		t.Fatalf("ownership first=%+v second=%+v", first, second)
	}
	if first.Resource == nil || !strings.Contains(first.HandleReceiptID, first.Resource.ResourceID) {
		t.Fatalf("receipt does not bind resource: ownership=%+v", first)
	}
	if first.HandleReceiptID == "" || second.HandleReceiptID == "" || first.HandleReceiptID == second.HandleReceiptID {
		t.Fatalf("PID reuse receipts must differ: first=%q second=%q", first.HandleReceiptID, second.HandleReceiptID)
	}
	if !strings.Contains(first.HandleReceiptID, ":42:") {
		t.Fatalf("receipt does not bind PID: %q", first.HandleReceiptID)
	}
}

func TestWriterOwnershipInvalidAndMismatchedResourcesFailClosed(t *testing.T) {
	lockPath := makeWriterLock(t, t.TempDir(), testThreadIDOne)
	invalid := inspectWriterOwnership("thread-invalid", lockPath, fixedOwnershipProbe{})
	if invalid.Verdict != WriterOwnershipUnknown || invalid.Resource != nil || invalid.EvidenceSource != "validation" {
		t.Fatalf("invalid identity ownership=%+v", invalid)
	}
	thread, _ := NewCodexThreadIdentity(testThreadIDOne)
	resource, _ := NewWriterResourceHandle(thread, lockPath)
	resource.Thread, _ = NewCodexThreadIdentity(testThreadIDTwo)
	mismatched := inspectWriterResourceOwnership(resource, fixedOwnershipProbe{})
	if mismatched.Verdict != WriterOwnershipUnknown || mismatched.Resource != nil {
		t.Fatalf("mismatched ownership=%+v", mismatched)
	}
}

func TestWriterOwnershipUnsupportedNativeReceiptIsUnknown(t *testing.T) {
	lockPath := makeWriterLock(t, t.TempDir(), testThreadIDOne)
	got := inspectWriterOwnership(testThreadIDOne, lockPath, fixedOwnershipProbe{witness: ownershipWitness{
		source:     "test_native_witness",
		conclusive: true,
		owners:     []processOwner{{pid: 42}},
	}})
	if got.Verdict != WriterOwnershipUnknown || got.HandleReceiptID != "" || !got.LockPresent {
		t.Fatalf("unsupported receipt ownership=%+v", got)
	}
}

func TestWriterOwnershipPermissionFailureIsUnknown(t *testing.T) {
	lockPath := makeWriterLock(t, t.TempDir(), testThreadIDOne)
	got := inspectWriterOwnership(testThreadIDOne, lockPath, fixedOwnershipProbe{
		witness: ownershipWitness{source: "test_permission_probe"},
		err:     os.ErrPermission,
	})
	if got.Verdict != WriterOwnershipUnknown || !got.LockPresent {
		t.Fatalf("ownership=%+v", got)
	}
	if got.EvidenceSource != "test_permission_probe" || !strings.Contains(got.Detail, "permission") {
		t.Fatalf("permission evidence=%+v", got)
	}
}

func TestWriterOwnershipStaleRequiresPositiveWitness(t *testing.T) {
	lockPath := makeWriterLock(t, t.TempDir(), testThreadIDOne)
	unknown := inspectWriterOwnership(testThreadIDOne, lockPath, fixedOwnershipProbe{
		witness: ownershipWitness{source: "test_inconclusive", conclusive: false},
	})
	if unknown.Verdict != WriterOwnershipUnknown {
		t.Fatalf("inconclusive ownership=%+v", unknown)
	}
	stale := inspectWriterOwnership(testThreadIDOne, lockPath, fixedOwnershipProbe{
		witness: ownershipWitness{source: "test_positive", conclusive: true},
	})
	if stale.Verdict != WriterOwnershipStaleResidue {
		t.Fatalf("positive no-owner witness=%+v", stale)
	}
}

func TestWriterOwnershipIsJSONSafeWithTypedAndCompatibilityFields(t *testing.T) {
	thread, _ := NewCodexThreadIdentity(testThreadIDOne)
	resource, err := NewWriterResourceHandle(thread, filepath.Join(t.TempDir(), "writer.lock"))
	if err != nil {
		t.Fatal(err)
	}
	got := WriterOwnership{
		ThreadID: resource.Thread.ID, LockPath: resource.LockPath, Resource: &resource, LockPresent: true,
		Verdict: WriterOwnershipLiveOwner, PID: 7, ProcessStartTime: "2026-08-31T10:00:00Z",
		ProcessImage: `C:\\codex.exe`, EvidenceSource: "windows_restart_manager", HandleReceiptID: "writer-owner-v1:abc:7:1",
	}
	data, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"thread_id", "lock_path", "resource", "resource_id", "thread", "lock_present", "verdict", "pid", "process_start_time", "process_image", "evidence_source", "handle_receipt_id"} {
		if !strings.Contains(string(data), `"`+field+`"`) {
			t.Fatalf("JSON missing %s: %s", field, data)
		}
	}
}
