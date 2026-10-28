package leaseref

import (
	"context"
	"reflect"
	"testing"
	"time"
)

// TestReapDeletesOnlyExpiredLeases: with one live (long TTL) and one expired lock lease,
// Reap removes exactly the expired one, returns its id, and leaves the live lease intact.
func TestReapDeletesOnlyExpiredLeases(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Now()
	if _, err := s.Acquire(ctx(), Record{ID: "live", TreeGlobs: []string{"a"}, Holder: "h", AcquiredAt: now.Unix(), TTLSeconds: 3600}); err != nil {
		t.Fatalf("Acquire live: %v", err)
	}
	if _, err := s.Acquire(ctx(), Record{ID: "dead", TreeGlobs: []string{"b"}, Holder: "h", AcquiredAt: 100, TTLSeconds: 10}); err != nil {
		t.Fatalf("Acquire dead: %v", err)
	}

	reaped, err := s.Reap(ctx(), now)
	if err != nil {
		t.Fatalf("Reap: %v", err)
	}
	if len(reaped) != 1 || reaped[0] != "dead" {
		t.Fatalf("reaped = %v, want [dead]", reaped)
	}
	if _, ok, _ := s.Get(ctx(), "dead"); ok {
		t.Fatalf("expired lease still present after Reap")
	}
	if _, ok, _ := s.Get(ctx(), "live"); !ok {
		t.Fatalf("Reap wrongly removed the live lease")
	}
}

// TestReapEmptyNamespaceNoError: reaping an empty namespace is a clean no-op (no ids, no
// error) — the same "absence is valid" rule the read side already holds.
func TestReapEmptyNamespaceNoError(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	reaped, err := s.Reap(ctx(), time.Now())
	if err != nil {
		t.Fatalf("Reap empty: %v", err)
	}
	if len(reaped) != 0 {
		t.Fatalf("reaped = %v, want none", reaped)
	}
}

// TestReapIsIdempotent: a second Reap after the first removed the expired lease finds
// nothing to do and does not error — models two peers racing the same reap, where the
// loser deletes an already-absent ref.
func TestReapIsIdempotent(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Now()
	if _, err := s.Acquire(ctx(), Record{ID: "dead", TreeGlobs: []string{"b"}, Holder: "h", AcquiredAt: 100, TTLSeconds: 10}); err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	if _, err := s.Reap(ctx(), now); err != nil {
		t.Fatalf("first Reap: %v", err)
	}
	reaped, err := s.Reap(ctx(), now)
	if err != nil {
		t.Fatalf("second Reap must be a clean no-op: %v", err)
	}
	if len(reaped) != 0 {
		t.Fatalf("second Reap reaped %v, want none", reaped)
	}
}

// TestReapSessionsDeletesOnlyExpired: the session-side reaper removes exactly the expired
// descriptor and leaves a live one — the cleanup LiveSessions anticipated but had no caller.
func TestReapSessionsDeletesOnlyExpired(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Now()
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "liveS", Host: "n1", PCBState: "RUNNING", UpdatedAt: now.Unix(), TTLSecs: 3600}); err != nil {
		t.Fatalf("Publish live: %v", err)
	}
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "deadS", Host: "n2", PCBState: "RUNNING", UpdatedAt: 100, TTLSecs: 10}); err != nil {
		t.Fatalf("Publish dead: %v", err)
	}

	reaped, err := s.ReapSessions(ctx(), now)
	if err != nil {
		t.Fatalf("ReapSessions: %v", err)
	}
	if len(reaped) != 1 || reaped[0] != "deadS" {
		t.Fatalf("reaped sessions = %v, want [deadS]", reaped)
	}
	if _, ok, _ := s.GetSession(ctx(), "deadS"); ok {
		t.Fatalf("expired session still present after ReapSessions")
	}
	if _, ok, _ := s.GetSession(ctx(), "liveS"); !ok {
		t.Fatalf("ReapSessions wrongly removed the live session")
	}
}

// TestReapKeepsLeasesAndSessionsDistinct is the load-bearing property: because lock leases
// (refs/fak/locks/<id>) and session descriptors (refs/fak/locks/session-<id>) share one ref
// prefix, a naive reaper could cross-delete. Reap must touch ONLY the lock lease and
// ReapSessions ONLY the session — even when both carry the same id stem and both are expired.
func TestReapKeepsLeasesAndSessionsDistinct(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Now()
	if _, err := s.Acquire(ctx(), Record{ID: "x", TreeGlobs: []string{"a"}, Holder: "h", AcquiredAt: 100, TTLSeconds: 10}); err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "x", Host: "n", PCBState: "RUNNING", UpdatedAt: 100, TTLSecs: 10}); err != nil {
		t.Fatalf("Publish: %v", err)
	}

	// Reap touches the lock lease only; the session descriptor x survives.
	reaped, err := s.Reap(ctx(), now)
	if err != nil {
		t.Fatalf("Reap: %v", err)
	}
	if len(reaped) != 1 || reaped[0] != "x" {
		t.Fatalf("Reap reaped %v, want [x] (the lock lease only)", reaped)
	}
	if _, ok, _ := s.GetSession(ctx(), "x"); !ok {
		t.Fatalf("Reap wrongly removed session descriptor x — namespace split violated")
	}

	// ReapSessions then removes the session; the (already-reaped) lock lease stays gone.
	rs, err := s.ReapSessions(ctx(), now)
	if err != nil {
		t.Fatalf("ReapSessions: %v", err)
	}
	if len(rs) != 1 || rs[0] != "x" {
		t.Fatalf("ReapSessions reaped %v, want [x] (the session only)", rs)
	}
	if _, ok, _ := s.GetSession(ctx(), "x"); ok {
		t.Fatalf("session x still present after ReapSessions")
	}
}

// fak-test:runtime fast est=10ms
func TestReapPreservesConcurrentRenewalAfterBatchFailure(t *testing.T) {
	g := newFakeGit()
	writer := NewWithRunner(g.run, "")
	now := time.Unix(1000, 0)
	for _, id := range []string{"renewed", "expired", "unknown"} {
		ttl := int64(10)
		if id == "unknown" {
			ttl = 0
		}
		if _, err := writer.Acquire(ctx(), Record{ID: id, TreeGlobs: []string{id}, Holder: "h", AcquiredAt: 100, TTLSeconds: ttl}); err != nil {
			t.Fatal(err)
		}
	}
	injected := false
	s := NewWithStdinRunner(g.run, func(c context.Context, dir, stdin string, args ...string) (string, int, error) {
		if len(args) == 2 && args[0] == "update-ref" && args[1] == "--stdin" && !injected {
			injected = true
			// Snapshot cost is constant: no per-lease rev-parse/cat-file spawns.
			want := [][]string{{"for-each-ref", "--format=%(refname)", refPrefix}, {"cat-file", "--batch"}}
			if !reflect.DeepEqual(g.calls, want) {
				t.Fatalf("snapshot calls = %v, want %v", g.calls, want)
			}
			// Advance the same ref after the expiry/OID snapshot but before deletion.
			if _, err := writer.Acquire(c, Record{ID: "renewed", TreeGlobs: []string{"renewed"}, Holder: "h", AcquiredAt: now.Unix(), TTLSeconds: 3600}); err != nil {
				t.Fatal(err)
			}
		}
		return g.runStdin(c, dir, stdin, args...)
	}, "")
	g.calls = nil
	reaped, err := s.Reap(ctx(), now)
	if !injected || err == nil || !reflect.DeepEqual(reaped, []string{"expired"}) {
		t.Fatalf("injected=%v reaped=%v err=%v; want refused renewal and only expired reaped", injected, reaped, err)
	}
	if rec, ok, err := s.Get(ctx(), "renewed"); err != nil || !ok || rec.Expired(now) {
		t.Fatalf("renewed lease lost: rec=%+v ok=%v err=%v", rec, ok, err)
	}
	if _, ok, err := s.Get(ctx(), "unknown"); err != nil || !ok {
		t.Fatalf("unknown TTL0 lease lost: ok=%v err=%v", ok, err)
	}
}
