package scratchdir

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/procalive"
)

// deadHolderRecord returns a holder record naming a pid that has provably
// exited: a child process (this test binary, run with a selector that matches
// no test) is started, its start time is sampled while it is alive, then the
// child is waited to completion. The sampled start keeps the record's verdict
// correct even if the pid were reused before IsHeld reads it.
func deadHolderRecord(t *testing.T) string {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("resolve test binary: %v", err)
	}
	cmd := exec.Command(exe, "-test.run=^$")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start child process: %v", err)
	}
	pid := cmd.Process.Pid
	start := int64(0)
	for i := 0; i < 200; i++ {
		if ts, ok := procalive.StartTime(pid); ok {
			start = ts.UnixNano()
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	_ = cmd.Wait()
	if start == 0 {
		t.Logf("start time unresolved for pid %d; the record pins start=0 instead", pid)
	}
	return fmt.Sprintf("pid=%d start=%d\n", pid, start)
}

// TestWriteHolderMarksLiveHolder: WriteHolder records the calling process as
// the live holder of an already-existing dir, with the same record IsHeld and
// Sweep read.
//
// fak-test:runtime fast est=20ms lane=default
func TestWriteHolderMarksLiveHolder(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "already-there")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := WriteHolder(dir); err != nil {
		t.Fatalf("WriteHolder: %v", err)
	}
	if !IsHeld(dir, 0) {
		t.Errorf("IsHeld(dir, 0) = false, want true: WriteHolder must record this live process")
	}
	if !IsHeld(dir, 48*time.Hour) {
		t.Errorf("IsHeld(dir, 48h) = false, want true: a live holder reads held whatever the age")
	}
}

// TestWriteHolderMissingDirErrors: WriteHolder marks an existing dir; a
// missing dir is a caller bug and must error, not silently create one.
//
// fak-test:runtime fast est=1ms lane=default
func TestWriteHolderMissingDirErrors(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "never-created")
	if err := WriteHolder(missing); err == nil {
		t.Errorf("WriteHolder into a missing dir = nil error, want an error")
	}
}

// TestWriteHolderDeadPidRecordNotHeld: a holder record whose pid has exited
// reads not-held, so a crashed holder never pins a dir.
//
// fak-test:runtime fast est=1s lane=default
func TestWriteHolderDeadPidRecordNotHeld(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "dead-holder")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, HolderFile), []byte(deadHolderRecord(t)), 0o644); err != nil {
		t.Fatal(err)
	}
	if IsHeld(dir, 0) {
		t.Errorf("IsHeld(dir, 0) = true for a record whose pid has exited, want false")
	}
	if IsHeld(dir, 48*time.Hour) {
		t.Errorf("IsHeld(dir, 48h) = true for a record whose pid has exited, want false")
	}
}

// TestWriteHolderKeepsSweepFromRemoving witnesses the same record through the
// Sweep path: an old dir this process holds via WriteHolder is classified
// held and never removed.
//
// fak-test:runtime fast est=20ms lane=default
func TestWriteHolderKeepsSweepFromRemoving(t *testing.T) {
	parent := t.TempDir()
	dir := filepath.Join(parent, "wh-held")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := WriteHolder(dir); err != nil {
		t.Fatalf("WriteHolder: %v", err)
	}
	old := time.Now().Add(-48 * time.Hour)
	if err := os.Chtimes(dir, old, old); err != nil {
		t.Fatal(err)
	}
	rep := Sweep(SweepOptions{Parent: parent, MinAge: time.Hour, Apply: true})
	entry, found := Entry{}, false
	for _, e := range rep.Entries {
		if e.Path == dir {
			entry, found = e, true
		}
	}
	if !found {
		t.Fatalf("sweep report has no entry for %s (held=%d removed=%d)", dir, rep.Held, rep.Removed)
	}
	if entry.Verdict != VerdictHeld || entry.Removed {
		t.Errorf("held dir verdict=%s removed=%v, want held/false", entry.Verdict, entry.Removed)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Errorf("sweep removed a dir this process holds via WriteHolder: %v", err)
	}
}
