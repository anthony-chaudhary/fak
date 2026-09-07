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
