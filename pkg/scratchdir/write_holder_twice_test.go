package scratchdir

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestWriteHolderTwiceIdempotentOverwrite: a second WriteHolder into the same
// dir must succeed and leave the dir reading held — the record is overwritten,
// not appended, so IsHeld still reads one live-holder record for this process
// at any age.
//
// fak-test:runtime fast est=20ms lane=default
func TestWriteHolderTwiceIdempotentOverwrite(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "twice")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := WriteHolder(dir); err != nil {
		t.Fatalf("first WriteHolder: %v", err)
	}
	if err := WriteHolder(dir); err != nil {
		t.Fatalf("second WriteHolder: %v", err)
	}
	if !IsHeld(dir, 0) {
		t.Errorf("IsHeld(dir, 0) = false after two writes, want true: the second write must leave a live-holder record")
	}
	if !IsHeld(dir, 48*time.Hour) {
		t.Errorf("IsHeld(dir, 48h) = false after two writes, want true: a live holder reads held whatever the age")
	}
}

// TestWriteHolderReplacesStaleRecord: WriteHolder overwrites whatever record
// the dir already carried — after it, the dir reads held for this process even
// when the previous record named a provably exited pid.
//
// fak-test:runtime fast est=1s lane=default
func TestWriteHolderReplacesStaleRecord(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "stale-then-live")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, HolderFile), []byte(deadHolderRecord(t)), 0o644); err != nil {
		t.Fatal(err)
	}
	if IsHeld(dir, 0) {
		t.Fatalf("IsHeld(dir, 0) = true for the stale record before WriteHolder, want false (fixture guard)")
	}
	if err := WriteHolder(dir); err != nil {
		t.Fatalf("WriteHolder over a stale record: %v", err)
	}
	if !IsHeld(dir, 0) {
		t.Errorf("IsHeld(dir, 0) = false after WriteHolder replaced the stale record, want true for this live process")
	}
}
