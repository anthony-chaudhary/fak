package leaseref

import (
	"context"
	"strings"
	"testing"
	"time"
)

// TestClaimIntentCollisionNamesIncumbent is the #2155 acceptance witness: two agents
// claim the SAME issue (phrased differently); the second is refused INTENT_COLLISION
// naming the first — holder, session, and the incumbent's own target words — BEFORE it
// spends a turn on the duplicate fix.
func TestClaimIntentCollisionNamesIncumbent(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	t0 := time.Unix(1000, 0)

	// Agent A claims issue 2155 at task-claim time.
	aRec, v, err := s.ClaimIntent(ctx(), IntentRecord{Target: "issue #2155", Holder: "A", SessionID: "sess-a"}, t0)
	if err != nil {
		t.Fatalf("A ClaimIntent: %v", err)
	}
	if !v.OK || aRec.Key != "issue-2155" {
		t.Fatalf("A claim: ok=%v key=%q, want ok key=issue-2155 (%+v)", v.OK, aRec.Key, v)
	}

	// Agent B claims the SAME issue two minutes later, phrased differently.
	_, bv, err := s.ClaimIntent(ctx(), IntentRecord{Target: "2155", Holder: "B", SessionID: "sess-b"}, t0.Add(2*time.Minute))
	if err != nil {
		t.Fatalf("B ClaimIntent: %v", err)
	}
	if bv.OK {
		t.Fatalf("B's duplicate claim was admitted: %+v", bv)
	}
	if bv.Reason != ReasonIntentCollision {
		t.Fatalf("B reason=%q, want %s (%+v)", bv.Reason, ReasonIntentCollision, bv)
	}
	if bv.Peer == nil || bv.Peer.Holder != "A" || bv.Peer.SessionID != "sess-a" {
		t.Fatalf("B refusal must name the incumbent A/sess-a, got %+v", bv.Peer)
	}
	if !strings.Contains(bv.Detail, "A") || !strings.Contains(bv.Detail, "sess-a") {
		t.Fatalf("B detail should name the peer and session: %q", bv.Detail)
	}
}

// TestIntentKeyNormalization: independently-phrased claims of one issue land on ONE
// key; free-form bug signatures hash deterministically and never collide with issue
// keys; "a number in a sentence" is NOT mistaken for an issue reference.
func TestIntentKeyNormalization(t *testing.T) {
	for _, tc := range []struct{ target, want string }{
		{"#2155", "issue-2155"},
		{"2155", "issue-2155"},
		{"issue 2155", "issue-2155"},
		{"issue-2155", "issue-2155"},
		{"Issue #2155", "issue-2155"},
		{"gh-2155", "issue-2155"},
		{"bug 2155", "issue-2155"},
		{"#0012", "issue-12"}, // leading zeros normalize
	} {
		if got := IntentKey(tc.target); got != tc.want {
			t.Errorf("IntentKey(%q)=%q, want %q", tc.target, got, tc.want)
		}
	}
	// A signature is hashed — deterministic, whitespace/case-insensitive, sig- prefixed.
	s1 := IntentKey("gateway drops same-tick ready events")
	s2 := IntentKey("  Gateway   drops same-tick READY events ")
	if s1 != s2 {
		t.Fatalf("signature normalization diverged: %q vs %q", s1, s2)
	}
	if !strings.HasPrefix(s1, "sig-") || len(s1) != len("sig-")+16 {
		t.Fatalf("signature key shape: %q", s1)
	}
	// A number inside a sentence is a signature, not an issue reference.
	if k := IntentKey("fix the 3 flaky tests"); strings.HasPrefix(k, "issue-") {
		t.Fatalf("sentence with a number misread as an issue key: %q", k)
	}
	if !validID(intentPrefix + s1) {
		t.Fatalf("sig key is not a safe ref segment: %q", s1)
	}
}

// TestClaimIntentRenewAndRelease: the same holder re-claiming its own live intent is a
// renew (window advances, claim identity kept); release deletes it so a NEW claimant is
// admitted; releasing an absent intent is idempotent.
func TestClaimIntentRenewAndRelease(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	t0 := time.Unix(1000, 0)

	a1, v, err := s.ClaimIntent(ctx(), IntentRecord{Target: "#77", Holder: "A", TTLSeconds: 600}, t0)
	if err != nil || !v.OK {
		t.Fatalf("claim: ok=%v err=%v", v.OK, err)
	}
	// Renew at t0+300: the window moves forward, AcquiredAt is unchanged.
	a2, v, err := s.ClaimIntent(ctx(), IntentRecord{Target: "issue 77", Holder: "A", TTLSeconds: 600}, t0.Add(5*time.Minute))
	if err != nil || !v.OK {
		t.Fatalf("renew: ok=%v err=%v (%+v)", v.OK, err, v)
	}
	if a2.AcquiredAt != a1.AcquiredAt || a2.RenewedAt != t0.Add(5*time.Minute).Unix() {
		t.Fatalf("renew identity: acquired=%d renewed=%d, want acquired=%d renewed=%d",
			a2.AcquiredAt, a2.RenewedAt, a1.AcquiredAt, t0.Add(5*time.Minute).Unix())
	}
	if a2.Expired(t0.Add(14 * time.Minute)) { // renewed at 300s + 600s TTL => live at 840s
		t.Fatalf("renewed intent wrongly expired")
	}

	// Release on ship: a new claimant is now admitted.
	if err := s.ReleaseIntent(ctx(), "77"); err != nil {
		t.Fatalf("release: %v", err)
	}
	_, v, err = s.ClaimIntent(ctx(), IntentRecord{Target: "#77", Holder: "B"}, t0.Add(6*time.Minute))
	if err != nil || !v.OK {
		t.Fatalf("post-release claim by B: ok=%v err=%v (%+v)", v.OK, err, v)
	}
	// Idempotent double-release.
	if err := s.ReleaseIntent(ctx(), "issue #12345"); err != nil {
		t.Fatalf("release of an absent intent should be idempotent: %v", err)
	}
}

// TestClaimIntentExpiredTakeover: a crashed claimant's lapsed intent never deadlocks
// the target — a peer's claim past the TTL is a takeover, and ReapIntents deletes the
// lapsed record.
func TestClaimIntentExpiredTakeover(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	t0 := time.Unix(1000, 0)

	if _, v, err := s.ClaimIntent(ctx(), IntentRecord{Target: "#9", Holder: "A", TTLSeconds: 300}, t0); err != nil || !v.OK {
		t.Fatalf("A claim: ok=%v err=%v", v.OK, err)
	}
	// Within the TTL, B is refused; past it, B takes over.
	if _, v, err := s.ClaimIntent(ctx(), IntentRecord{Target: "#9", Holder: "B"}, t0.Add(time.Minute)); err != nil || v.OK || v.Reason != ReasonIntentCollision {
		t.Fatalf("B early claim: ok=%v reason=%q err=%v", v.OK, v.Reason, err)
	}
	b, v, err := s.ClaimIntent(ctx(), IntentRecord{Target: "#9", Holder: "B"}, t0.Add(10*time.Minute))
	if err != nil || !v.OK || b.Holder != "B" {
		t.Fatalf("B takeover: ok=%v holder=%q err=%v", v.OK, b.Holder, err)
	}

	// A lapsed second intent reaps cleanly and the live one survives.
	if _, v, err := s.ClaimIntent(ctx(), IntentRecord{Target: "#10", Holder: "C", TTLSeconds: 60}, t0); err != nil || !v.OK {
		t.Fatalf("C claim: %v %+v", err, v)
	}
	reaped, err := s.ReapIntents(ctx(), t0.Add(10*time.Minute))
	if err != nil {
		t.Fatalf("reap: %v", err)
	}
	if len(reaped) != 1 || reaped[0] != "issue-10" {
		t.Fatalf("reaped=%v, want [issue-10] (B's fresh takeover of #9 is live)", reaped)
	}
	live, expired, err := s.LiveIntents(ctx(), t0.Add(10*time.Minute))
	if err != nil || len(live) != 1 || len(expired) != 0 || live[0].Key != "issue-9" {
		t.Fatalf("post-reap live=%v expired=%v err=%v", live, expired, err)
	}
}

// TestIntentNamespaceSplit: intent refs never appear in the lock-lease or session
// views, and vice versa — the three kinds stay distinct over the one shared namespace.
func TestIntentNamespaceSplit(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	t0 := time.Unix(1000, 0)

	if _, v, err := s.ClaimIntent(ctx(), IntentRecord{Target: "#2155", Holder: "A"}, t0); err != nil || !v.OK {
		t.Fatalf("intent claim: %v %+v", err, v)
	}
	if _, err := s.Acquire(ctx(), Record{ID: "kernel-lane", TreeGlobs: []string{"internal/kernel/**"}, Holder: "A"}); err != nil {
		t.Fatalf("lock acquire: %v", err)
	}
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "guard-1", Host: "h", PCBState: "RUNNING"}); err != nil {
		t.Fatalf("session publish: %v", err)
	}

	locks, err := s.List(ctx())
	if err != nil || len(locks) != 1 || locks[0].ID != "kernel-lane" {
		t.Fatalf("lock view sees %v (err=%v), want exactly kernel-lane", locks, err)
	}
	arb, err := s.LiveLeases(ctx(), t0)
	if err != nil || len(arb) != 1 || arb[0].Lane != "kernel-lane" {
		t.Fatalf("arbiter view sees %v (err=%v), want exactly kernel-lane", arb, err)
	}
	sessions, err := s.ListSessions(ctx())
	if err != nil || len(sessions) != 1 || sessions[0].ID != "guard-1" {
		t.Fatalf("session view sees %v (err=%v), want exactly guard-1", sessions, err)
	}
	intents, err := s.ListIntents(ctx())
	if err != nil || len(intents) != 1 || intents[0].Key != "issue-2155" {
		t.Fatalf("intent view sees %v (err=%v), want exactly issue-2155", intents, err)
	}
}

// TestClaimIntentCASRaceReportsWinner: when a same-host peer lands its claim between
// this claim's read and its compare-and-swap write, the lost CAS re-reads and reports
// the WINNER as INTENT_COLLISION — the race outcome is the collision the check exists
// to catch, not a bare retry error.
func TestClaimIntentCASRaceReportsWinner(t *testing.T) {
	g := newFakeGit()
	// The racer: just before the claim's update-ref CAS lands, peer B's claim appears.
	raced := false
	racer := func(c context.Context, dir string, args ...string) (string, int, error) {
		if !raced && len(args) > 0 && args[0] == "update-ref" && len(args) >= 4 {
			raced = true
			// B's record enters the object store and the ref advances to it.
			g.next++
			id := synthID(g.next)
			g.blobs[id] = []byte(`{"target":"#5","key":"issue-5","holder":"B","session_id":"sess-b","acquired_unix":1000,"ttl_seconds":3600}`)
			g.refs[args[1]] = id
		}
		return g.run(c, dir, args...)
	}
	s := NewWithRunner(racer, "")
	_, v, err := s.ClaimIntent(ctx(), IntentRecord{Target: "#5", Holder: "A"}, time.Unix(1001, 0))
	if err != nil {
		t.Fatalf("ClaimIntent: %v", err)
	}
	if v.OK || v.Reason != ReasonIntentCollision {
		t.Fatalf("raced claim: ok=%v reason=%q, want INTENT_COLLISION (%+v)", v.OK, v.Reason, v)
	}
	if v.Peer == nil || v.Peer.Holder != "B" {
		t.Fatalf("raced claim must name the winner B, got %+v", v.Peer)
	}
}
