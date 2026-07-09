package leaseref

import (
	"testing"
	"time"
)

// seedWorker publishes one session descriptor and binds n lock-lease registrations to it.
// Every registration is given a LONG own TTL that is NOT expired at now — the load-bearing
// detail of the cascade witness: if the lease's own TTL had lapsed, Live would already drop
// it on the TTL rule alone and the session-liveness cascade would never be exercised.
func seedWorker(t *testing.T, s *Store, sess SessionDescriptor, now time.Time, ids ...string) {
	t.Helper()
	if _, err := s.PublishSession(ctx(), sess); err != nil {
		t.Fatalf("PublishSession(%s): %v", sess.ID, err)
	}
	for _, id := range ids {
		rec := Record{
			ID:         id,
			TreeGlobs:  []string{"internal/" + id + "/**"},
			Holder:     MintHolder(sess.Host, sess.ID),
			SessionID:  sess.ID,
			AcquiredAt: now.Unix() - 10,
			TTLSeconds: 86400, // far from expiry: the OWN-TTL rule must not be what drops it
		}
		if _, err := s.Acquire(ctx(), rec); err != nil {
			t.Fatalf("Acquire(%s): %v", id, err)
		}
	}
}

// deleteRefCalls counts the `update-ref -d` argv the fake git saw — the witness that a
// reaper pass did NOT run. Cascade removal is a READ-SIDE drop: the registrations must
// vanish from the live view without any ref being deleted.
func deleteRefCalls(g *fakeGit) int {
	n := 0
	for _, argv := range g.calls {
		if len(argv) >= 2 && argv[0] == "update-ref" && argv[1] == "-d" {
			n++
		}
	}
	return n
}

// TestLiveLeasesCascadesWhenOwningSessionHeartbeatLapses is the WITNESS for #3372, asserted
// on the PRE-EXISTING arbiter feed (Store.LiveLeases) so it fails before the fix and passes
// after. A worker publishes a session descriptor and two registrations bound to it; the
// worker disconnects, so its heartbeat lapses past the descriptor's TTL. Both registrations
// must vanish from the arbiter's live_leases in ONE step, with a removal event per
// registration, and with NO reaper pass (no `update-ref -d`).
//
// Before the fix LiveLeases folded only Record.Expired (the lease's OWN TTL), never the
// owning session — so a disconnected worker's registrations kept refusing a peer for the
// whole remaining lease TTL (forever, when TTLSeconds == 0).
func TestLiveLeasesCascadesWhenOwningSessionHeartbeatLapses(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Unix(1_000_000, 0)

	// The worker's heartbeat last landed an hour ago against a 60s TTL: positively dead.
	seedWorker(t, s, SessionDescriptor{
		ID: "w1", Host: "nodeA", PCBState: "RUNNING",
		UpdatedAt: now.Unix() - 3600, TTLSecs: 60,
	}, now, "alpha", "beta")

	// Sanity: the registrations' OWN TTL has not lapsed, so nothing but the session
	// cascade can drop them. Live (the TTL-only partition) still reports both.
	liveByTTL, expiredByTTL, err := s.Live(ctx(), now)
	if err != nil {
		t.Fatalf("Live: %v", err)
	}
	if len(liveByTTL) != 2 || len(expiredByTTL) != 0 {
		t.Fatalf("TTL partition = %d live / %d expired, want 2/0 — the witness must not be decided by the own-TTL rule", len(liveByTTL), len(expiredByTTL))
	}

	// The defect: the arbiter feed must not advertise a dead worker's registrations.
	leases, err := s.LiveLeases(ctx(), now)
	if err != nil {
		t.Fatalf("LiveLeases: %v", err)
	}
	if len(leases) != 0 {
		t.Fatalf("LiveLeases = %+v, want none — the owning session's heartbeat lapsed, so its registrations must cascade out of the arbiter feed", leases)
	}

	// ...and the drop is observable as a removal event per registration.
	kept, removed, err := s.LiveRegistrations(ctx(), now)
	if err != nil {
		t.Fatalf("LiveRegistrations: %v", err)
	}
	if len(kept) != 0 {
		t.Fatalf("kept = %+v, want none", kept)
	}
	if len(removed) != 2 {
		t.Fatalf("removal events = %+v, want one per registration (alpha, beta)", removed)
	}
	for i, want := range []string{"alpha", "beta"} {
		if removed[i].LeaseID != want {
			t.Errorf("removal[%d].LeaseID = %q, want %q (id-sorted)", i, removed[i].LeaseID, want)
		}
		if removed[i].Reason != RemovalSessionExpired {
			t.Errorf("removal[%d].Reason = %q, want %q", i, removed[i].Reason, RemovalSessionExpired)
		}
		if removed[i].SessionID != "w1" {
			t.Errorf("removal[%d].SessionID = %q, want w1", i, removed[i].SessionID)
		}
		if removed[i].Node != "nodeA" {
			t.Errorf("removal[%d].Node = %q, want nodeA", i, removed[i].Node)
		}
		if removed[i].AtUnix != now.Unix() {
			t.Errorf("removal[%d].AtUnix = %d, want %d", i, removed[i].AtUnix, now.Unix())
		}
		if removed[i].Evidence == "" {
			t.Errorf("removal[%d] carries no evidence sentence", i)
		}
	}

	// NO REAPER PASS: the cascade is a read-side drop, never a ref delete.
	if n := deleteRefCalls(g); n != 0 {
		t.Fatalf("cascade issued %d `update-ref -d` calls, want 0 — the registrations must vanish without a reaper pass", n)
	}
}

// TestCascadeDropsOnTerminalStoppedSession: a worker that published the terminal PCB state
// STOPPED is positively dead by its own statement, before its heartbeat TTL has lapsed. Its
// registrations cascade out with the session-stopped reason.
func TestCascadeDropsOnTerminalStoppedSession(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Unix(1_000_000, 0)

	// Heartbeat is FRESH (updated a second ago, 3600s TTL) — only the STOPPED state kills it.
	seedWorker(t, s, SessionDescriptor{
		ID: "w2", Host: "nodeB", PCBState: "STOPPED",
		UpdatedAt: now.Unix() - 1, TTLSecs: 3600,
	}, now, "gamma")

	kept, removed, err := s.LiveRegistrations(ctx(), now)
	if err != nil {
		t.Fatalf("LiveRegistrations: %v", err)
	}
	if len(kept) != 0 || len(removed) != 1 {
		t.Fatalf("kept = %+v, removed = %+v, want 0 kept / 1 removed", kept, removed)
	}
	if removed[0].Reason != RemovalSessionStopped {
		t.Fatalf("reason = %q, want %q", removed[0].Reason, RemovalSessionStopped)
	}
	if n := deleteRefCalls(g); n != 0 {
		t.Fatalf("cascade issued %d `update-ref -d` calls, want 0", n)
	}
}

// TestCascadeRetainsRegistrationWhenSessionDescriptorIsAbsent is the FAIL-CLOSED guard, and
// the reason this cascade is keyed on POSITIVE death rather than on "the lease is gone".
// Session publishing is best-effort and fail-open (session.go), and a ref namespace
// converges across clones only as a SET — so an ABSENT descriptor means "no evidence",
// which is routinely a LIVE worker whose descriptor this clone has not fetched yet. Treating
// absence as death would let a peer steal a live worker's lane. The registration is RETAINED.
func TestCascadeRetainsRegistrationWhenSessionDescriptorIsAbsent(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Unix(1_000_000, 0)

	// A registration bound to session w3 — whose descriptor was never published here.
	rec := Record{
		ID: "delta", TreeGlobs: []string{"internal/delta/**"},
		Holder: "nodeC/w3", SessionID: "w3",
		AcquiredAt: now.Unix() - 10, TTLSeconds: 86400,
	}
	if _, err := s.Acquire(ctx(), rec); err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	kept, removed, err := s.LiveRegistrations(ctx(), now)
	if err != nil {
		t.Fatalf("LiveRegistrations: %v", err)
	}
	if len(kept) != 1 || kept[0].ID != "delta" {
		t.Fatalf("kept = %+v, want [delta] — an absent descriptor is peer-unknown, never proof of death", kept)
	}
	if len(removed) != 0 {
		t.Fatalf("removed = %+v, want none — absence of evidence must not cascade", removed)
	}
	leases, err := s.LiveLeases(ctx(), now)
	if err != nil {
		t.Fatalf("LiveLeases: %v", err)
	}
	if len(leases) != 1 || leases[0].Lane != "delta" {
		t.Fatalf("LiveLeases = %+v, want the delta lease retained (fail-closed)", leases)
	}
}

// TestCascadeRetainsRegistrationOfHeartbeatingSession: the converse of the witness — a live,
// heartbeating worker's registrations must NEVER be dropped from the arbiter feed. This is
// the lane-steal the cascade exists to avoid causing.
func TestCascadeRetainsRegistrationOfHeartbeatingSession(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Unix(1_000_000, 0)

	seedWorker(t, s, SessionDescriptor{
		ID: "w4", Host: "nodeD", PCBState: "RUNNING",
		UpdatedAt: now.Unix() - 5, TTLSecs: 60,
	}, now, "epsilon")

	kept, removed, err := s.LiveRegistrations(ctx(), now)
	if err != nil {
		t.Fatalf("LiveRegistrations: %v", err)
	}
	if len(kept) != 1 || len(removed) != 0 {
		t.Fatalf("kept = %+v, removed = %+v, want the heartbeating worker's registration retained", kept, removed)
	}
}

// TestCascadeRetainsUnboundLegacyRegistration: a pre-#2164 record carries no session_id, so
// there is no owning session to cascade from. It classifies peer-unknown and is retained —
// the cascade never widens to leases it cannot bind to a session.
func TestCascadeRetainsUnboundLegacyRegistration(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Unix(1_000_000, 0)

	// A dead session exists, but this lease is not bound to it.
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "w5", Host: "nodeE", PCBState: "RUNNING", UpdatedAt: 100, TTLSecs: 10}); err != nil {
		t.Fatalf("PublishSession: %v", err)
	}
	if _, err := s.Acquire(ctx(), Record{ID: "zeta", TreeGlobs: []string{"docs/**"}, Holder: "legacy-holder", AcquiredAt: now.Unix() - 10, TTLSeconds: 0}); err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	kept, removed, err := s.LiveRegistrations(ctx(), now)
	if err != nil {
		t.Fatalf("LiveRegistrations: %v", err)
	}
	if len(kept) != 1 || kept[0].ID != "zeta" || len(removed) != 0 {
		t.Fatalf("kept = %+v, removed = %+v, want the unbound legacy lease retained", kept, removed)
	}
}

// TestCascadeDropsTTLZeroRegistrationOfDeadWorker is the permanent-ghost case the own-TTL
// rule can never reach: TTLSeconds == 0 means "no expiry", so Record.Expired is false
// forever and Reap can never collect the lease. Before the cascade, such a registration
// blocked the arbiter for the life of the repo once its worker disconnected.
func TestCascadeDropsTTLZeroRegistrationOfDeadWorker(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Unix(1_000_000, 0)

	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "w6", Host: "nodeF", PCBState: "RUNNING", UpdatedAt: now.Unix() - 3600, TTLSecs: 60}); err != nil {
		t.Fatalf("PublishSession: %v", err)
	}
	if _, err := s.Acquire(ctx(), Record{ID: "eta", TreeGlobs: []string{"internal/eta/**"}, Holder: "nodeF/w6", SessionID: "w6", AcquiredAt: now.Unix() - 10, TTLSeconds: 0}); err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	// The TTL-only reaper is powerless here — it collects nothing.
	reaped, err := s.Reap(ctx(), now)
	if err != nil {
		t.Fatalf("Reap: %v", err)
	}
	if len(reaped) != 0 {
		t.Fatalf("Reap collected %v, want none — a TTL-0 lease never expires on the own-TTL rule", reaped)
	}

	// The cascade drops it anyway: its owning worker is provably gone.
	leases, err := s.LiveLeases(ctx(), now)
	if err != nil {
		t.Fatalf("LiveLeases: %v", err)
	}
	if len(leases) != 0 {
		t.Fatalf("LiveLeases = %+v, want none — a TTL-0 registration of a dead worker is the permanent ghost the cascade exists to clear", leases)
	}
}

// TestCascadeDropIsPure exercises the pure rule directly across the closed outcome set, with
// no store and no git — the same literal-driven discipline ClassifyLiveness's tests use.
func TestCascadeDropIsPure(t *testing.T) {
	now := time.Unix(1_000_000, 0)
	live := SessionDescriptor{ID: "s", Host: "n", PCBState: "RUNNING", UpdatedAt: now.Unix(), TTLSecs: 60}
	dead := SessionDescriptor{ID: "s", Host: "n", PCBState: "RUNNING", UpdatedAt: now.Unix() - 3600, TTLSecs: 60}
	stopped := SessionDescriptor{ID: "s", Host: "n", PCBState: "stopped", UpdatedAt: now.Unix(), TTLSecs: 60}
	bound := Record{ID: "l", Holder: "n/s", SessionID: "s"}

	cases := []struct {
		name       string
		rec        Record
		sessions   map[string]SessionDescriptor
		wantDrop   bool
		wantReason string
	}{
		{"heartbeating session retained", bound, map[string]SessionDescriptor{"s": live}, false, ""},
		{"lapsed heartbeat drops", bound, map[string]SessionDescriptor{"s": dead}, true, RemovalSessionExpired},
		{"terminal STOPPED drops (case-insensitive)", bound, map[string]SessionDescriptor{"s": stopped}, true, RemovalSessionStopped},
		{"absent descriptor retained (fail-closed)", bound, map[string]SessionDescriptor{}, false, ""},
		{"unbound record retained", Record{ID: "l"}, map[string]SessionDescriptor{"s": dead}, false, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ev, dropped := CascadeDrop(tc.rec, tc.sessions, now)
			if dropped != tc.wantDrop {
				t.Fatalf("dropped = %v, want %v (evidence: %s)", dropped, tc.wantDrop, ev.Evidence)
			}
			if !dropped {
				return
			}
			if ev.Reason != tc.wantReason {
				t.Errorf("reason = %q, want %q", ev.Reason, tc.wantReason)
			}
			if ev.LeaseID != "l" || ev.SessionID != "s" || ev.AtUnix != now.Unix() {
				t.Errorf("event = %+v, want it to name the lease, its session, and the read instant", ev)
			}
		})
	}
}
