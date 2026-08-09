package leaseref

import (
	"context"
	"testing"
	"time"
)

// TestFenceStaleHolderRefused is the #1182 acceptance fixture, the canonical
// paused-then-resumed-holder hazard (#906 §3.3): holder A acquires a lease, goes dormant
// PAST its TTL, peer B reaps and reacquires (the generation bumps), then A returns and tries
// to write. A's first write must be refused with STALE_LEASE — A is "alive", but its LEASE is
// stale — so A halts and reacquires instead of silently double-writing the leased tree.
func TestFenceStaleHolderRefused(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")

	t0 := time.Unix(1000, 0)
	lane := Record{ID: "kernel-lane", TreeGlobs: []string{"internal/kernel/**"}, Holder: "A", TTLSeconds: 300}

	// A acquires: fresh lease, generation 1.
	aRec, v, err := s.AcquireFenced(ctx(), lane, t0)
	if err != nil {
		t.Fatalf("A AcquireFenced: %v", err)
	}
	if !v.OK || aRec.Generation != 1 {
		t.Fatalf("A acquire: ok=%v gen=%d, want ok gen=1 (%+v)", v.OK, aRec.Generation, v)
	}

	// A goes dormant well past the 300s TTL. Peer B reaps + reacquires: a TRANSITION, so the
	// generation strictly bumps to 2 and the holder becomes B.
	tLater := t0.Add(10 * time.Minute) // 600s > 300s TTL
	bRec, v, err := s.AcquireFenced(ctx(), Record{ID: lane.ID, TreeGlobs: lane.TreeGlobs, Holder: "B", TTLSeconds: 300}, tLater)
	if err != nil {
		t.Fatalf("B AcquireFenced: %v", err)
	}
	if !v.OK || bRec.Generation != 2 || bRec.Holder != "B" {
		t.Fatalf("B transition: ok=%v gen=%d holder=%q, want ok gen=2 holder=B (%+v)", v.OK, bRec.Generation, bRec.Holder, v)
	}

	// A returns and presents its OLD lease (generation 1). The fence sees the live lease is
	// generation 2 and refuses A's write.
	fv, err := s.Fence(ctx(), aRec, tLater.Add(time.Second))
	if err != nil {
		t.Fatalf("A Fence: %v", err)
	}
	if fv.OK {
		t.Fatalf("A Fence admitted a stale write: %+v", fv)
	}
	if fv.Reason != ReasonStaleLease {
		t.Fatalf("A Fence reason=%q, want %s (%+v)", fv.Reason, ReasonStaleLease, fv)
	}
	if fv.Current != 2 || fv.Presented != 1 || fv.Holder != "B" {
		t.Fatalf("A Fence evidence: current=%d presented=%d holder=%q, want 2/1/B (%+v)", fv.Current, fv.Presented, fv.Holder, fv)
	}

	// And B, presenting its CURRENT lease, is still admitted — the fence refuses the stale
	// holder without locking out the live one.
	bv, err := s.Fence(ctx(), bRec, tLater.Add(time.Second))
	if err != nil {
		t.Fatalf("B Fence: %v", err)
	}
	if !bv.OK {
		t.Fatalf("B Fence refused the live holder: %+v", bv)
	}
}

// TestAcquireFencedLiveDifferentHolderRefused: a DIFFERENT holder may not acquire a lease
// that is still LIVE (un-expired) — that is the un-reapable incumbent, refused LEASE_HELD,
// distinct from the stale-generation case.
func TestAcquireFencedLiveDifferentHolderRefused(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	t0 := time.Unix(1000, 0)

	if _, v, err := s.AcquireFenced(ctx(), Record{ID: "lane", TreeGlobs: []string{"x/**"}, Holder: "A", TTLSeconds: 300}, t0); err != nil || !v.OK {
		t.Fatalf("A acquire: ok=%v err=%v", v.OK, err)
	}
	// B tries to acquire while A's lease is still live (well within the TTL).
	_, v, err := s.AcquireFenced(ctx(), Record{ID: "lane", TreeGlobs: []string{"x/**"}, Holder: "B", TTLSeconds: 300}, t0.Add(time.Minute))
	if err != nil {
		t.Fatalf("B acquire: %v", err)
	}
	if v.OK || v.Reason != ReasonLeaseHeld {
		t.Fatalf("B acquire of a live lease: ok=%v reason=%q, want refused LEASE_HELD (%+v)", v.OK, v.Reason, v)
	}
	if v.Holder != "A" {
		t.Fatalf("B refusal should name holder A, got %q", v.Holder)
	}
}

// TestAcquireFencedRenewKeepsGeneration: the same holder reacquiring its own LIVE lease is a
// RENEW — the generation is unchanged (a renew is liveness, not a new admission), RenewedAt
// advances, and the lease's expiry window moves forward.
func TestAcquireFencedRenewKeepsGeneration(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	t0 := time.Unix(1000, 0)

	a1, v, err := s.AcquireFenced(ctx(), Record{ID: "lane", TreeGlobs: []string{"x/**"}, Holder: "A", TTLSeconds: 300}, t0)
	if err != nil || !v.OK || a1.Generation != 1 {
		t.Fatalf("acquire: gen=%d ok=%v err=%v", a1.Generation, v.OK, err)
	}
	// Same holder reacquires within the TTL: a renew, generation stays 1, RenewedAt set.
	a2, v, err := s.AcquireFenced(ctx(), Record{ID: "lane", TreeGlobs: []string{"x/**"}, Holder: "A", TTLSeconds: 300}, t0.Add(2*time.Minute))
	if err != nil || !v.OK {
		t.Fatalf("renew: ok=%v err=%v", v.OK, err)
	}
	if a2.Generation != 1 {
		t.Fatalf("renew bumped the generation to %d, want it kept at 1", a2.Generation)
	}
	if a2.RenewedAt != t0.Add(2*time.Minute).Unix() {
		t.Fatalf("renew RenewedAt=%d, want %d", a2.RenewedAt, t0.Add(2*time.Minute).Unix())
	}
	// The renew extended the window: at t0+400s the original TTL would have lapsed, but the
	// renew at t0+120s keeps it live.
	if a2.Expired(t0.Add(400 * time.Second)) {
		t.Fatalf("renewed lease wrongly expired at t0+400s (RenewedAt=%d ttl=%d)", a2.RenewedAt, a2.TTLSeconds)
	}
}

// TestRenewWrongHolderStale: Renew refuses when a peer has taken the lease over (a different
// live holder) — STALE_LEASE — and refuses NO_LEASE when the lease has lapsed or is absent.
func TestRenewWrongHolderStale(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	t0 := time.Unix(1000, 0)

	if _, v, err := s.AcquireFenced(ctx(), Record{ID: "lane", TreeGlobs: []string{"x/**"}, Holder: "A", TTLSeconds: 300}, t0); err != nil || !v.OK {
		t.Fatalf("A acquire: ok=%v err=%v", v.OK, err)
	}
	// A expires; B takes over (gen 2).
	tLate := t0.Add(10 * time.Minute)
	if _, v, err := s.AcquireFenced(ctx(), Record{ID: "lane", TreeGlobs: []string{"x/**"}, Holder: "B", TTLSeconds: 300}, tLate); err != nil || !v.OK {
		t.Fatalf("B transition: ok=%v err=%v", v.OK, err)
	}
	// A tries to renew its lapsed lease — now held by B. STALE_LEASE.
	_, v, err := s.Renew(ctx(), "lane", "A", 0, tLate.Add(time.Second))
	if err != nil {
		t.Fatalf("A renew: %v", err)
	}
	if v.OK || v.Reason != ReasonStaleLease {
		t.Fatalf("A renew of a taken-over lease: ok=%v reason=%q, want STALE_LEASE (%+v)", v.OK, v.Reason, v)
	}

	// Renew on a never-acquired id is NO_LEASE.
	_, nv, err := s.Renew(ctx(), "ghost", "A", 0, t0)
	if err != nil {
		t.Fatalf("ghost renew: %v", err)
	}
	if nv.OK || nv.Reason != ReasonNoLease {
		t.Fatalf("renew of an absent lease: ok=%v reason=%q, want NO_LEASE (%+v)", nv.OK, nv.Reason, nv)
	}
}

// TestFenceNoLease: Fence returns NO_LEASE when the lease the caller thinks it holds has been
// released or was never acquired — not a stale-write, but the caller must reacquire.
func TestFenceNoLease(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	v, err := s.Fence(ctx(), Record{ID: "lane", Holder: "A", Generation: 1}, time.Unix(1000, 0))
	if err != nil {
		t.Fatalf("Fence: %v", err)
	}
	if v.OK || v.Reason != ReasonNoLease {
		t.Fatalf("Fence of an absent lease: ok=%v reason=%q, want NO_LEASE (%+v)", v.OK, v.Reason, v)
	}
}

// TestFenceExpiredIsNoLease: a lease that exists but has lapsed at `now` fences as NO_LEASE
// (it is reapable, not an authoritative live lease), so a holder whose own lease lapsed is
// told to reacquire rather than handed an OK.
func TestFenceExpiredIsNoLease(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	t0 := time.Unix(1000, 0)
	a, v, err := s.AcquireFenced(ctx(), Record{ID: "lane", TreeGlobs: []string{"x/**"}, Holder: "A", TTLSeconds: 300}, t0)
	if err != nil || !v.OK {
		t.Fatalf("acquire: ok=%v err=%v", v.OK, err)
	}
	fv, err := s.Fence(ctx(), a, t0.Add(10*time.Minute)) // past TTL, no peer took over
	if err != nil {
		t.Fatalf("Fence: %v", err)
	}
	if fv.OK || fv.Reason != ReasonNoLease {
		t.Fatalf("Fence of own lapsed lease: ok=%v reason=%q, want NO_LEASE (%+v)", fv.OK, fv.Reason, fv)
	}
}

// TestAcquireFencedCASContention: when the ref advances between the acquire's read and its
// compare-and-swap write (a same-host peer raced the generation bump), the write loses the
// CAS and returns LEASE_CONTENDED — the real same-host atomicity the visibility-only
// boundary does not promise cross-machine but enforces on one host.
func TestAcquireFencedCASContention(t *testing.T) {
	g := newFakeGit()
	t0 := time.Unix(1000, 0)
	if _, v, err := NewWithRunner(g.run, "").AcquireFenced(ctx(), Record{ID: "lane", TreeGlobs: []string{"x/**"}, Holder: "A", TTLSeconds: 300}, t0); err != nil || !v.OK {
		t.Fatalf("seed acquire: ok=%v err=%v", v.OK, err)
	}

	// A racing runner: just before A's renew CAS lands, a peer advances the ref to a
	// different object, so the old-value compare-and-swap fails.
	raced := false
	racer := func(c context.Context, dir string, args ...string) (string, int, error) {
		if !raced && len(args) > 0 && args[0] == "update-ref" && len(args) >= 4 {
			raced = true
			g.refs[args[1]] = "peerSHA" // a peer advanced the ref under us
		}
		return g.run(c, dir, args...)
	}
	s := NewWithRunner(racer, "")
	_, v, err := s.Renew(ctx(), "lane", "A", 0, t0.Add(time.Minute))
	if err != nil {
		t.Fatalf("renew: %v", err)
	}
	if v.OK || v.Reason != ReasonLeaseContended {
		t.Fatalf("contended renew: ok=%v reason=%q, want LEASE_CONTENDED (%+v)", v.OK, v.Reason, v)
	}
}

// TestPublishFencedSuppressesDisplacedHolderResult is the task-result write-boundary
// regression for #6000. Holder A pauses after finishing its result, B supersedes A's
// expired generation, then A resumes. The externally-visible sink must see zero writes.
func TestPublishFencedSuppressesDisplacedHolderResult(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	ctx := ctx()
	t0 := time.Unix(2000, 0)
	request := Record{ID: "task-result", TreeGlobs: []string{"results/**"}, Holder: "A", TTLSeconds: 10}

	aToken, v, err := s.AcquireFenced(ctx, request, t0)
	if err != nil || !v.OK {
		t.Fatalf("A AcquireFenced: verdict=%+v err=%v", v, err)
	}

	// A is paused before publication. Its lease expires and B takes generation 2.
	tResume := t0.Add(11 * time.Second)
	bToken, v, err := s.AcquireFenced(ctx, Record{ID: request.ID, TreeGlobs: request.TreeGlobs, Holder: "B", TTLSeconds: 10}, tResume)
	if err != nil || !v.OK || bToken.Generation != aToken.Generation+1 {
		t.Fatalf("B supersede: token=%+v verdict=%+v err=%v", bToken, v, err)
	}

	writes := 0
	v, err = s.PublishFenced(ctx, aToken, tResume, func() error {
		writes++
		return nil
	})
	if err != nil {
		t.Fatalf("stale PublishFenced: %v", err)
	}
	if v.OK || v.Reason != ReasonStaleLease {
		t.Fatalf("stale verdict=%+v, want typed %s refusal", v, ReasonStaleLease)
	}
	if writes != 0 {
		t.Fatalf("stale holder published %d results, want zero", writes)
	}

	v, err = s.PublishFenced(ctx, bToken, tResume.Add(time.Second), func() error {
		writes++
		return nil
	})
	if err != nil || !v.OK {
		t.Fatalf("current PublishFenced: verdict=%+v err=%v", v, err)
	}
	if writes != 1 {
		t.Fatalf("current holder writes=%d, want 1", writes)
	}
}

func TestPublishFencedSuppressesMissingOrExpiredToken(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Unix(3000, 0)
	writes := 0
	publish := func() error { writes++; return nil }

	missing := Record{ID: "missing-result", Holder: "A", Generation: 1, TTLSeconds: 10, RenewedAt: now.Unix()}
	v, err := s.PublishFenced(ctx(), missing, now, publish)
	if err != nil || v.OK || v.Reason != ReasonNoLease {
		t.Fatalf("missing verdict=%+v err=%v, want %s", v, err, ReasonNoLease)
	}

	token, v, err := s.AcquireFenced(ctx(), Record{ID: "expired-result", Holder: "A", TTLSeconds: 1}, now)
	if err != nil || !v.OK {
		t.Fatalf("acquire expired fixture: verdict=%+v err=%v", v, err)
	}
	v, err = s.PublishFenced(ctx(), token, now.Add(2*time.Second), publish)
	if err != nil || v.OK || v.Reason != ReasonNoLease {
		t.Fatalf("expired verdict=%+v err=%v, want %s", v, err, ReasonNoLease)
	}
	if writes != 0 {
		t.Fatalf("missing/expired tokens published %d results, want zero", writes)
	}
}
