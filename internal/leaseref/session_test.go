package leaseref

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// TestPublishSessionWritesSessionRef proves publish-on-register writes the descriptor at
// refs/fak/locks/session-<id> (the distinct namespace), and that a read-back round-trips
// the full {id, host, pcb_state, updated_at, ttl} projection.
func TestPublishSessionWritesSessionRef(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")

	d := SessionDescriptor{
		ID:        "guard-7",
		Host:      "dgx3",
		PCBState:  "RUNNING",
		UpdatedAt: 1000,
		TTLSecs:   300,
	}
	ref, err := s.PublishSession(ctx(), d)
	if err != nil {
		t.Fatalf("PublishSession: %v", err)
	}
	if ref != "refs/fak/locks/session-guard-7" {
		t.Fatalf("PublishSession ref=%q, want refs/fak/locks/session-guard-7", ref)
	}
	// The ref really landed in the session- namespace, not the bare lock namespace.
	if _, ok := g.refs["refs/fak/locks/session-guard-7"]; !ok {
		t.Fatalf("session ref not written; refs=%v", g.refs)
	}

	got, ok, err := s.GetSession(ctx(), "guard-7")
	if err != nil || !ok {
		t.Fatalf("GetSession ok=%v err=%v, want a descriptor", ok, err)
	}
	if got != d {
		t.Fatalf("GetSession returned %+v, want the published %+v", got, d)
	}
}

// TestSessionDistinctFromLockLease is the load-bearing namespace-split test: a lock lease
// and a session descriptor coexist under refs/fak/locks/, but ListSessions returns ONLY
// the session and List (lock leases) returns ONLY the lease — neither view leaks into the
// other.
func TestSessionDistinctFromLockLease(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")

	if _, err := s.Acquire(ctx(), Record{ID: "kernel-lane", TreeGlobs: []string{"internal/kernel/**"}, Holder: "A", AcquiredAt: 1, TTLSeconds: 0}); err != nil {
		t.Fatalf("Acquire lock lease: %v", err)
	}
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "sess-9", Host: "mac", PCBState: "RUNNING", UpdatedAt: 1, TTLSecs: 0}); err != nil {
		t.Fatalf("PublishSession: %v", err)
	}

	sessions, err := s.ListSessions(ctx())
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(sessions) != 1 || sessions[0].ID != "sess-9" {
		t.Fatalf("ListSessions = %+v, want only [sess-9] (lock lease excluded)", sessions)
	}

	leases, err := s.List(ctx())
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(leases) != 1 || leases[0].ID != "kernel-lane" {
		t.Fatalf("List = %+v, want only [kernel-lane] (session excluded)", leases)
	}

	// LiveLeases (the arbiter projection) must likewise NOT see the session as a lease.
	al, err := s.LiveLeases(ctx(), time.Unix(2, 0))
	if err != nil {
		t.Fatalf("LiveLeases: %v", err)
	}
	if len(al) != 1 || al[0].Lane != "kernel-lane" {
		t.Fatalf("LiveLeases = %+v, want only the kernel-lane lease, never the session", al)
	}
}

// TestPublishSessionTransitionUpdates proves update-on-transition: republishing the same
// id with a new PCB state and a later timestamp overwrites the prior descriptor in place
// (one ref per session, not an accumulating set).
func TestPublishSessionTransitionUpdates(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")

	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "s1", Host: "h", PCBState: "RUNNING", UpdatedAt: 100, TTLSecs: 600}); err != nil {
		t.Fatalf("publish register: %v", err)
	}
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "s1", Host: "h", PCBState: "PAUSED", UpdatedAt: 200, TTLSecs: 600}); err != nil {
		t.Fatalf("publish transition: %v", err)
	}

	got, ok, err := s.GetSession(ctx(), "s1")
	if err != nil || !ok {
		t.Fatalf("GetSession ok=%v err=%v", ok, err)
	}
	if got.PCBState != "PAUSED" || got.UpdatedAt != 200 {
		t.Fatalf("after transition descriptor = %+v, want PCBState=PAUSED UpdatedAt=200", got)
	}

	// Still exactly ONE session ref — a transition updates in place, never accumulates.
	all, err := s.ListSessions(ctx())
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(all) != 1 {
		t.Fatalf("ListSessions after transition = %+v, want a single in-place-updated descriptor", all)
	}
}

// TestRemoveSessionEndsLifecycle proves the stop side: RemoveSession deletes the session
// ref, GetSession then reports absent, and a second remove is idempotent (already gone).
func TestRemoveSessionEndsLifecycle(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")

	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "s2", Host: "h", PCBState: "RUNNING", UpdatedAt: 1, TTLSecs: 0}); err != nil {
		t.Fatalf("PublishSession: %v", err)
	}
	if err := s.RemoveSession(ctx(), "s2"); err != nil {
		t.Fatalf("RemoveSession: %v", err)
	}
	if _, ok, _ := s.GetSession(ctx(), "s2"); ok {
		t.Fatalf("GetSession after RemoveSession returned a descriptor, want absent")
	}
	// Idempotent: removing an already-gone session is a no-op, not an error.
	if err := s.RemoveSession(ctx(), "s2"); err != nil {
		t.Fatalf("RemoveSession of a missing session must be a no-op, got %v", err)
	}
}

// TestLiveSessionsDropsExpired proves the expire side of the lifecycle: a stale descriptor
// (TTL elapsed past its last UpdatedAt — e.g. a crashed node that stopped republishing) is
// dropped from the LIVE view and surfaced in the expired-id set so a reader can reap it.
func TestLiveSessionsDropsExpired(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")

	now := time.Now()
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "live", Host: "h", PCBState: "RUNNING", UpdatedAt: now.Unix(), TTLSecs: 3600}); err != nil {
		t.Fatalf("publish live: %v", err)
	}
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "stale", Host: "h", PCBState: "RUNNING", UpdatedAt: 100, TTLSecs: 10}); err != nil {
		t.Fatalf("publish stale: %v", err)
	}

	live, expired, err := s.LiveSessions(ctx(), now)
	if err != nil {
		t.Fatalf("LiveSessions: %v", err)
	}
	if len(live) != 1 || live[0].ID != "live" {
		t.Fatalf("LiveSessions live = %+v, want only [live]", live)
	}
	if len(expired) != 1 || expired[0] != "stale" {
		t.Fatalf("LiveSessions expired = %v, want [stale]", expired)
	}

	// Reap the expired descriptor — an ordinary ref delete that converges across clones.
	if err := s.RemoveSession(ctx(), "stale"); err != nil {
		t.Fatalf("reap RemoveSession: %v", err)
	}
	if _, ok, _ := s.GetSession(ctx(), "stale"); ok {
		t.Fatalf("reaped session still present")
	}
}

// TestPublishSessionIssuesExactPlumbing pins the EXACT git argv: write the blob, then
// point ONLY the side ref — never a branch, never a force, never HEAD. This is the #1198
// invariant (no mutation of main/HEAD/refs/heads, no force-push), asserted via the
// injectable Runner.
func TestPublishSessionIssuesExactPlumbing(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "x", Host: "h", PCBState: "RUNNING", UpdatedAt: 1}); err != nil {
		t.Fatalf("PublishSession: %v", err)
	}
	if len(g.calls) != 2 {
		t.Fatalf("PublishSession issued %d git calls, want 2 (hash-object, update-ref): %v", len(g.calls), g.calls)
	}
	if g.calls[0][0] != "hash-object" || g.calls[0][1] != "-w" {
		t.Fatalf("first call = %v, want [hash-object -w <file>]", g.calls[0])
	}
	up := g.calls[1]
	if up[0] != "update-ref" || up[1] != "refs/fak/locks/session-x" {
		t.Fatalf("second call = %v, want [update-ref refs/fak/locks/session-x <sha>]", up)
	}
	for _, c := range g.calls {
		for _, a := range c {
			if a == "-f" || a == "--force" || a == "HEAD" || strings.HasPrefix(a, "refs/heads/") {
				t.Fatalf("session publish must never force/touch a branch/HEAD; saw %q in %v", a, c)
			}
		}
	}
}

// TestSessionPeerVisibilityAfterFetch models publish-on-A -> (push/fetch) -> visible-on-B:
// once B's local ref store has the session ref (an ordinary fetch put it there), B's
// ListSessions sees A's live session — the fleet-wide visibility #1198 delivers.
func TestSessionPeerVisibilityAfterFetch(t *testing.T) {
	a := newFakeGit()
	sa := NewWithRunner(a.run, "")
	if _, err := sa.PublishSession(ctx(), SessionDescriptor{ID: "fleet-1", Host: "nodeA", PCBState: "RUNNING", UpdatedAt: 10, TTLSecs: 0}); err != nil {
		t.Fatalf("A.PublishSession: %v", err)
	}

	// "fetch": copy A's object+ref state into B (what `git fetch` does for the namespace).
	b := newFakeGit()
	for k, v := range a.blobs {
		b.blobs[k] = v
	}
	for k, v := range a.refs {
		b.refs[k] = v
	}
	sb := NewWithRunner(b.run, "")

	ds, err := sb.ListSessions(ctx())
	if err != nil {
		t.Fatalf("B.ListSessions: %v", err)
	}
	if len(ds) != 1 || ds[0].Host != "nodeA" || ds[0].ID != "fleet-1" {
		t.Fatalf("B.ListSessions = %+v, want A's live session", ds)
	}
}

// TestSessionDescriptorJSONShape pins the on-the-wire descriptor shape so a future field
// rename can't silently break a peer reading an older/newer session ref.
func TestSessionDescriptorJSONShape(t *testing.T) {
	b, _ := json.Marshal(SessionDescriptor{ID: "x", Host: "h", PCBState: "RUNNING", UpdatedAt: 1, TTLSecs: 2})
	want := `{"id":"x","host":"h","pcb_state":"RUNNING","updated_at":1,"ttl_seconds":2}`
	if string(b) != want {
		t.Fatalf("session descriptor JSON = %s, want %s", b, want)
	}
}

// TestSessionRoundTripEncodeRefDecode is the explicit encode->ref->decode round-trip the
// issue's witness asks for: a descriptor marshaled to the blob, pointed at by the ref, and
// read back through cat-file decodes to the SAME descriptor (every field preserved).
func TestSessionRoundTripEncodeRefDecode(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	in := SessionDescriptor{ID: "rt", Host: "host:sess", PCBState: "DRAINING", UpdatedAt: 424242, TTLSecs: 900}
	if _, err := s.PublishSession(ctx(), in); err != nil {
		t.Fatalf("PublishSession: %v", err)
	}
	out, ok, err := s.GetSession(ctx(), "rt")
	if err != nil || !ok {
		t.Fatalf("GetSession ok=%v err=%v", ok, err)
	}
	if out != in {
		t.Fatalf("round-trip descriptor = %+v, want the encoded %+v", out, in)
	}
}

// TestInvalidSessionIDRejected proves namespace confinement: an unsafe id, and an id that
// already carries the session- marker (which would double-prefix or collide), are both
// refused; a plain safe id is accepted.
func TestInvalidSessionIDRejected(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	for _, bad := range []string{"", "a/b", "../escape", "-flag", ".dot", "has space", "session-x"} {
		if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: bad, Host: "h", PCBState: "RUNNING"}); err == nil {
			t.Fatalf("PublishSession(%q) should reject an unsafe/double-prefixed session id", bad)
		}
	}
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "ok_sess.1-2", Host: "h", PCBState: "RUNNING", UpdatedAt: 1}); err != nil {
		t.Fatalf("PublishSession of a valid id failed: %v", err)
	}
}

// TestListSessionsSkipsUnparseableBlob proves a corrupt/forward-incompatible session ref
// is skipped, not fatal — the same resilience the lock-lease List has.
func TestListSessionsSkipsUnparseableBlob(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "good", Host: "h", PCBState: "RUNNING", UpdatedAt: 1}); err != nil {
		t.Fatalf("PublishSession: %v", err)
	}
	g.blobs["garbage"] = []byte("not json {{{")
	g.refs["refs/fak/locks/session-bad"] = "garbage"
	ds, err := s.ListSessions(ctx())
	if err != nil {
		t.Fatalf("ListSessions must not error on a corrupt descriptor: %v", err)
	}
	if len(ds) != 1 || ds[0].ID != "good" {
		t.Fatalf("ListSessions = %+v, want only the parseable [good]", ds)
	}
}

// TestSessionDescriptorExpired pins the TTL math against UpdatedAt (not AcquiredAt): a zero
// TTL never expires, and the boundary is at UpdatedAt+TTL.
func TestSessionDescriptorExpired(t *testing.T) {
	noTTL := SessionDescriptor{UpdatedAt: 1000, TTLSecs: 0}
	if noTTL.Expired(time.Unix(1000+1e6, 0)) {
		t.Fatal("a zero TTL must never expire")
	}
	d := SessionDescriptor{UpdatedAt: 1000, TTLSecs: 60}
	if d.Expired(time.Unix(1059, 0)) {
		t.Fatal("not yet expired at updated+59")
	}
	if !d.Expired(time.Unix(1060, 0)) {
		t.Fatal("expired at updated+ttl")
	}
}
