package main

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// TestLoadLeaseLiveness witnesses the dead-owner reaper's production lease lookup:
// a recently-active run reads ALIVE, a run silent past the stale window reads
// DEAD (orphaned owner), a witnessed/terminal run reads DEAD (owner gone), and a
// run id the ledger never recorded reads DEAD (registry row absent). This is the
// LeaseAlive that keys procguard.ClassifyDeadOwnerOrphans in `fak process-guard
// report --reap-dead-owner`. Issue #3596.
func TestLoadLeaseLiveness(t *testing.T) {
	now := time.Now().Unix()
	path := filepath.Join(t.TempDir(), "loops.jsonl")
	appendEv(t, path, "r-live", loopmgr.EventStart, "", now-60)                              // recent -> running -> alive
	appendEv(t, path, "r-dead", loopmgr.EventStart, "", now-5000)                            // silent past stale -> orphaned -> dead
	appendEv(t, path, "r-done", loopmgr.EventStart, "", now-4000)                            // started ...
	appendEv(t, path, "r-done", loopmgr.EventWitness, loopmgr.StatusWitnessedDone, now-3900) // ... witnessed -> complete -> dead

	leaseAlive, note := loadLeaseLiveness(path)
	if note != "" {
		t.Fatalf("readable ledger must not skip the mode: %s", note)
	}
	if leaseAlive == nil {
		t.Fatal("readable ledger must yield a lease lookup")
	}
	if !leaseAlive("r-live") {
		t.Fatalf("a recently-active run must read ALIVE")
	}
	if leaseAlive("r-dead") {
		t.Fatalf("a run silent past the stale window must read DEAD")
	}
	if leaseAlive("r-done") {
		t.Fatalf("a witnessed/terminal run must read DEAD (owner gone)")
	}
	if leaseAlive("r-never-seen") {
		t.Fatalf("a run id absent from the ledger must read DEAD (registry row absent)")
	}
}

// TestLoadLeaseLivenessFailClosed: an unreadable ledger path must SKIP the mode
// (return a nil lookup + a note), never treat every owner as dead.
func TestLoadLeaseLivenessFailClosed(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "does-not-exist", "loops.jsonl")
	leaseAlive, note := loadLeaseLiveness(missing)
	if leaseAlive != nil {
		t.Fatalf("an unreadable ledger must yield a nil lookup (skip the mode), got non-nil")
	}
	if note == "" {
		t.Fatalf("an unreadable ledger must carry a skip note for the operator")
	}
}
