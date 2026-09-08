package debtlane

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

func TestDiscoverHeldLanes_DeadPIDReclaimed(t *testing.T) {
	tempDir := t.TempDir()
	dosDir := filepath.Join(tempDir, ".dos")
	if err := os.MkdirAll(dosDir, 0o755); err != nil {
		t.Fatal(err)
	}
	journal := filepath.Join(dosDir, "lane-journal.jsonl")

	origLiveness := pidLivenessCheck
	defer func() { pidLivenessCheck = origLiveness }()

	pidLivenessCheck = func(pid int) bool {
		if pid == 99999 {
			return false // dead PID
		}
		return true
	}

	records := []string{
		`{"op":"ACQUIRE","lane":"dead-lane","pid":99999,"mode":"exclusive"}`,
		`{"op":"ACQUIRE","lane":"live-lane","pid":0}`,
	}
	content := ""
	for _, r := range records {
		content += r + "\n"
	}
	if err := os.WriteFile(journal, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	held, err := DiscoverHeldLanes(tempDir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []string{"live-lane"}
	if !reflect.DeepEqual(held, expected) {
		t.Fatalf("expected held %v, got %v", expected, held)
	}
}

func TestDiscoverHeldLanes_AdvisoryLeaseFiltered(t *testing.T) {
	tempDir := t.TempDir()
	dosDir := filepath.Join(tempDir, ".dos")
	if err := os.MkdirAll(dosDir, 0o755); err != nil {
		t.Fatal(err)
	}
	journal := filepath.Join(dosDir, "lane-journal.jsonl")

	origLiveness := pidLivenessCheck
	defer func() { pidLivenessCheck = origLiveness }()
	pidLivenessCheck = func(pid int) bool {
		return true
	}

	records := []string{
		`{"op":"ACQUIRE","lane":"advisory-lane","pid":1001,"mode":"advisory"}`,
		`{"op":"ACQUIRE","lane":"readonly-lane","pid":1002,"mode":"readonly"}`,
		`{"op":"ACQUIRE","lane":"shared-lane","pid":1003,"mode":"shared"}`,
		`{"op":"ACQUIRE","lane":"exclusive-lane","pid":1004,"mode":"exclusive"}`,
	}
	content := ""
	for _, r := range records {
		content += r + "\n"
	}
	if err := os.WriteFile(journal, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	held, err := DiscoverHeldLanes(tempDir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []string{"exclusive-lane"}
	if !reflect.DeepEqual(held, expected) {
		t.Fatalf("expected held %v, got %v", expected, held)
	}
}

func TestDiscoverHeldLanes_AlivePIDPreserved(t *testing.T) {
	tempDir := t.TempDir()
	dosDir := filepath.Join(tempDir, ".dos")
	if err := os.MkdirAll(dosDir, 0o755); err != nil {
		t.Fatal(err)
	}
	journal := filepath.Join(dosDir, "lane-journal.jsonl")

	origLiveness := pidLivenessCheck
	defer func() { pidLivenessCheck = origLiveness }()
	pidLivenessCheck = func(pid int) bool {
		return pid == 2001 || pid == 2002
	}

	now := time.Now()
	// Active lane: acquired 5 minutes ago (recent activity < 10m -> effective TTL = 60m).
	activeEntry := LaneJournalEntry{
		Op:         "ACQUIRE",
		Lane:       "active-lane",
		PID:        2001,
		Mode:       "exclusive",
		AcquiredAt: now.Add(-5 * time.Minute),
	}
	// Expired lane: acquired 25 minutes ago, no heartbeat (base TTL = 15m + 5m grace = 20m).
	expiredEntry := LaneJournalEntry{
		Op:         "ACQUIRE",
		Lane:       "expired-lane",
		PID:        2002,
		Mode:       "exclusive",
		AcquiredAt: now.Add(-25 * time.Minute),
	}

	actBytes, _ := json.Marshal(activeEntry)
	expBytes, _ := json.Marshal(expiredEntry)

	content := string(actBytes) + "\n" + string(expBytes) + "\n"
	if err := os.WriteFile(journal, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	held, err := DiscoverHeldLanes(tempDir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := []string{"active-lane"}
	if !reflect.DeepEqual(held, expected) {
		t.Fatalf("expected held %v, got %v", expected, held)
	}
}

func TestDiscoverHeldLanes_TouchAndHeartbeat(t *testing.T) {
	tempDir := t.TempDir()
	dosDir := filepath.Join(tempDir, ".dos")
	if err := os.MkdirAll(dosDir, 0o755); err != nil {
		t.Fatal(err)
	}
	journal := filepath.Join(dosDir, "lane-journal.jsonl")

	origLiveness := pidLivenessCheck
	defer func() { pidLivenessCheck = origLiveness }()
	pidLivenessCheck = func(pid int) bool {
		return pid == 3001
	}

	now := time.Now()
	// Initial lease acquired 25 minutes ago (would be expired under base 15m + 5m grace).
	oldEntry := LaneJournalEntry{
		Op:         "ACQUIRE",
		Lane:       "beat-lane",
		PID:        3001,
		Mode:       "exclusive",
		AcquiredAt: now.Add(-25 * time.Minute),
	}
	oldBytes, _ := json.Marshal(oldEntry)
	if err := os.WriteFile(journal, append(oldBytes, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}

	// Verify it is expired before touch/heartbeat
	heldBefore, err := DiscoverHeldLanes(tempDir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(heldBefore) != 0 {
		t.Fatalf("expected 0 held lanes before heartbeat, got %v", heldBefore)
	}

	// Now touch the lane lease via TouchLaneLease
	if err := TouchLaneLease(tempDir, "beat-lane", 3001); err != nil {
		t.Fatalf("TouchLaneLease failed: %v", err)
	}

	// Verify the lane is now reclaimed / active!
	heldAfter, err := DiscoverHeldLanes(tempDir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := []string{"beat-lane"}
	if !reflect.DeepEqual(heldAfter, expected) {
		t.Fatalf("expected %v after heartbeat, got %v", expected, heldAfter)
	}

	// Also test TOUCH op explicitly
	touchRecord := `{"op":"TOUCH","lane":"touch-lane","pid":3001,"heartbeat_at":"` + time.Now().Format(time.RFC3339) + `"}` + "\n"
	f, err := os.OpenFile(journal, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(touchRecord); err != nil {
		f.Close()
		t.Fatal(err)
	}
	f.Close()

	heldWithTouch, err := DiscoverHeldLanes(tempDir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expectedWithTouch := []string{"beat-lane", "touch-lane"}
	if !reflect.DeepEqual(heldWithTouch, expectedWithTouch) {
		t.Fatalf("expected %v, got %v", expectedWithTouch, heldWithTouch)
	}
}

func TestLaneJournalEntry_UnmarshalTree(t *testing.T) {
	// 1. Array tree
	data1 := []byte(`{"op":"ACQUIRE","lane":"compute","tree":["internal/compute/a.go","internal/compute/b.go"],"pid":123}`)
	var e1 LaneJournalEntry
	if err := json.Unmarshal(data1, &e1); err != nil {
		t.Fatalf("unmarshal array tree: %v", err)
	}
	if !reflect.DeepEqual(e1.Tree, []string{"internal/compute/a.go", "internal/compute/b.go"}) {
		t.Errorf("expected array tree, got %v", e1.Tree)
	}

	// 2. String tree
	data2 := []byte(`{"op":"ACQUIRE","lane":"gateway","tree":"internal/gateway/**","pid":456}`)
	var e2 LaneJournalEntry
	if err := json.Unmarshal(data2, &e2); err != nil {
		t.Fatalf("unmarshal string tree: %v", err)
	}
	if !reflect.DeepEqual(e2.Tree, []string{"internal/gateway/**"}) {
		t.Errorf("expected string tree converted to slice, got %v", e2.Tree)
	}

	// 3. Nested lease.tree
	data3 := []byte(`{"op":"ACQUIRE","lane":"model","lease":{"tree":["internal/model/m.go"]},"pid":789}`)
	var e3 LaneJournalEntry
	if err := json.Unmarshal(data3, &e3); err != nil {
		t.Fatalf("unmarshal lease.tree: %v", err)
	}
	if !reflect.DeepEqual(e3.Tree, []string{"internal/model/m.go"}) {
		t.Errorf("expected nested lease.tree, got %v", e3.Tree)
	}
}

func TestDiscoverHeldLeasesAndTrees(t *testing.T) {
	tempDir := t.TempDir()

	origLiveness := pidLivenessCheck
	defer func() { pidLivenessCheck = origLiveness }()
	pidLivenessCheck = func(pid int) bool { return true }

	if err := AcquireTreeLease(tempDir, "compute", []string{"internal/compute/a.go"}, 5001); err != nil {
		t.Fatalf("AcquireTreeLease 1: %v", err)
	}
	if err := AcquireTreeLease(tempDir, "compute", []string{"internal/compute/b.go"}, 5002); err != nil {
		t.Fatalf("AcquireTreeLease 2: %v", err)
	}
	if err := AcquireTreeLease(tempDir, "gateway", []string{"internal/gateway/**"}, 5003); err != nil {
		t.Fatalf("AcquireTreeLease 3: %v", err)
	}

	leases, err := DiscoverHeldLeases(tempDir)
	if err != nil {
		t.Fatalf("DiscoverHeldLeases: %v", err)
	}
	if len(leases) != 3 {
		t.Fatalf("expected 3 held leases, got %d: %v", len(leases), leases)
	}

	trees, err := DiscoverHeldTrees(tempDir)
	if err != nil {
		t.Fatalf("DiscoverHeldTrees: %v", err)
	}
	if len(trees["compute"]) != 2 {
		t.Errorf("expected 2 compute trees, got %v", trees["compute"])
	}
	if len(trees["gateway"]) != 1 {
		t.Errorf("expected 1 gateway tree, got %v", trees["gateway"])
	}

	// Release one of the compute tree leases
	if err := ReleaseTreeLease(tempDir, "compute", []string{"internal/compute/a.go"}, 5001); err != nil {
		t.Fatalf("ReleaseTreeLease: %v", err)
	}

	leasesAfter, err := DiscoverHeldLeases(tempDir)
	if err != nil {
		t.Fatalf("DiscoverHeldLeases after release: %v", err)
	}
	if len(leasesAfter) != 2 {
		t.Fatalf("expected 2 held leases after release, got %d", len(leasesAfter))
	}
	foundComputeB := false
	for _, l := range leasesAfter {
		if l.Lane == "compute" && l.PID == 5002 {
			foundComputeB = true
		}
	}
	if !foundComputeB {
		t.Errorf("expected compute PID 5002 still held, got %v", leasesAfter)
	}
}
