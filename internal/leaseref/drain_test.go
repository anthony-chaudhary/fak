package leaseref

import (
	"testing"
	"time"
)

// TestReportDescriptorDrainTargetsExpiredExcludesLive is the core report-first property: with
// one live (long TTL) and one expired descriptor, the plan names exactly the expired id and
// its delete refspec, counts the live one as excluded — and, decisively, DELETES NOTHING
// (both descriptors survive the call). Producing the plan is a pure read.
func TestReportDescriptorDrainTargetsExpiredExcludesLive(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Now()
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "liveD", Host: "n1", PCBState: "RUNNING", UpdatedAt: now.Unix(), TTLSecs: 3600}); err != nil {
		t.Fatalf("Publish live: %v", err)
	}
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "deadD", Host: "n2", PCBState: "DRAINING", UpdatedAt: 100, TTLSecs: 10}); err != nil {
		t.Fatalf("Publish dead: %v", err)
	}

	plan, err := s.ReportDescriptorDrain(ctx(), "origin", now)
	if err != nil {
		t.Fatalf("ReportDescriptorDrain: %v", err)
	}
	if plan.Remote != "origin" {
		t.Fatalf("plan.Remote = %q, want origin", plan.Remote)
	}
	if len(plan.ExpiredIDs) != 1 || plan.ExpiredIDs[0] != "deadD" {
		t.Fatalf("plan.ExpiredIDs = %v, want [deadD]", plan.ExpiredIDs)
	}
	if len(plan.DeleteRefspecs) != 1 || plan.DeleteRefspecs[0] != ":refs/fak/locks/session-deadD" {
		t.Fatalf("plan.DeleteRefspecs = %v, want [:refs/fak/locks/session-deadD]", plan.DeleteRefspecs)
	}
	if plan.LiveExcluded != 1 {
		t.Fatalf("plan.LiveExcluded = %d, want 1 (the live descriptor)", plan.LiveExcluded)
	}

	// The load-bearing report-first guarantee: planning is READ-ONLY. Neither the expired
	// nor the live descriptor may be removed by producing the plan.
	if _, ok, _ := s.GetSession(ctx(), "deadD"); !ok {
		t.Fatalf("ReportDescriptorDrain deleted the expired descriptor — it must be read-only")
	}
	if _, ok, _ := s.GetSession(ctx(), "liveD"); !ok {
		t.Fatalf("ReportDescriptorDrain deleted the live descriptor — it must be read-only")
	}
}

// TestReportDescriptorDrainEmptyNamespace: with no descriptors published, the plan is empty
// with no error — the same "absence is a valid empty view" rule the readers hold.
func TestReportDescriptorDrainEmptyNamespace(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	plan, err := s.ReportDescriptorDrain(ctx(), "origin", time.Now())
	if err != nil {
		t.Fatalf("ReportDescriptorDrain empty: %v", err)
	}
	if len(plan.ExpiredIDs) != 0 || len(plan.DeleteRefspecs) != 0 || plan.LiveExcluded != 0 {
		t.Fatalf("plan over empty namespace = %+v, want all zero", plan)
	}
}

// TestReportDescriptorDrainRejectsInvalidRemote: a remote that cannot be one safe git argv
// token is refused up front (before any ref read), so a reported plan is always one a real
// push could run verbatim — argv hygiene identical to Sync's.
func TestReportDescriptorDrainRejectsInvalidRemote(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	for _, bad := range []string{"", "-origin", "a b"} {
		if _, err := s.ReportDescriptorDrain(ctx(), bad, time.Now()); err == nil {
			t.Fatalf("ReportDescriptorDrain(%q) = nil error, want a rejected-remote error", bad)
		}
	}
}

// TestReportDescriptorDrainIgnoresLockLeases is the namespace-split guarantee on the plan side:
// an expired LOCK LEASE sharing the same id stem as an expired descriptor must never appear in
// the drain plan — only session-<id> refs are drain targets, exactly as ReapSessions deletes
// only descriptors.
func TestReportDescriptorDrainIgnoresLockLeases(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Now()
	if _, err := s.Acquire(ctx(), Record{ID: "shared", TreeGlobs: []string{"a"}, Holder: "h", AcquiredAt: 100, TTLSeconds: 10}); err != nil {
		t.Fatalf("Acquire expired lock lease: %v", err)
	}
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "shared", Host: "n", PCBState: "RUNNING", UpdatedAt: 100, TTLSecs: 10}); err != nil {
		t.Fatalf("Publish expired descriptor: %v", err)
	}
	plan, err := s.ReportDescriptorDrain(ctx(), "origin", now)
	if err != nil {
		t.Fatalf("ReportDescriptorDrain: %v", err)
	}
	if len(plan.ExpiredIDs) != 1 || plan.ExpiredIDs[0] != "shared" {
		t.Fatalf("plan.ExpiredIDs = %v, want [shared] (the descriptor only, never the lock lease)", plan.ExpiredIDs)
	}
	if len(plan.DeleteRefspecs) != 1 || plan.DeleteRefspecs[0] != ":refs/fak/locks/session-shared" {
		t.Fatalf("plan.DeleteRefspecs = %v, want the session- ref only", plan.DeleteRefspecs)
	}
}
