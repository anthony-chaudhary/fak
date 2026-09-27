package leaseref

import (
	"context"
	"testing"
	"time"
)

// TestRenewFencedRefusesStaleSameHolderAfterReleaseReacquire covers the identity
// reuse hazard: a process holding generation 1 cannot renew generation 2 merely
// because both lease epochs use the same holder string.
func TestRenewFencedRefusesStaleSameHolderAfterReleaseReacquire(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	t0 := time.Unix(1000, 0)

	first, v, err := s.AcquireFenced(ctx(), Record{ID: "lane", Holder: "worker", TTLSeconds: 300}, t0)
	if err != nil || !v.OK {
		t.Fatalf("first acquire: verdict=%+v err=%v", v, err)
	}
	if v, err = s.ReleaseFenced(ctx(), "lane", "worker", first.Generation, t0.Add(time.Second)); err != nil || !v.OK {
		t.Fatalf("release: verdict=%+v err=%v", v, err)
	}
	second, v, err := s.AcquireFenced(ctx(), Record{ID: "lane", Holder: "worker", TTLSeconds: 300}, t0.Add(2*time.Second))
	if err != nil || !v.OK {
		t.Fatalf("reacquire: verdict=%+v err=%v", v, err)
	}
	if second.Generation <= first.Generation {
		t.Fatalf("reacquired generation=%d, want greater than released generation=%d", second.Generation, first.Generation)
	}

	_, v, err = s.RenewFenced(ctx(), "lane", "worker", first.Generation, 600, t0.Add(3*time.Second))
	if err != nil {
		t.Fatalf("stale RenewFenced: %v", err)
	}
	if v.OK || v.Reason != ReasonStaleLease || v.Presented != first.Generation || v.Current != second.Generation {
		t.Fatalf("verdict=%+v, want STALE_LEASE presented=%d current=%d", v, first.Generation, second.Generation)
	}
	got, ok, err := s.Get(ctx(), "lane")
	if err != nil || !ok {
		t.Fatalf("Get replacement: ok=%v err=%v", ok, err)
	}
	if got.Generation != second.Generation || got.RenewedAt != second.RenewedAt || got.TTLSeconds != second.TTLSeconds {
		t.Fatalf("stale renew mutated replacement: got=%+v want=%+v", got, second)
	}
}

// TestRenewFencedRefusesOmittedGenerationForPositiveLease proves generation zero
// remains legacy-only on renew as well as release.
func TestRenewFencedRefusesOmittedGenerationForPositiveLease(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	t0 := time.Unix(1000, 0)
	rec, v, err := s.AcquireFenced(ctx(), Record{ID: "lane", Holder: "worker", TTLSeconds: 300}, t0)
	if err != nil || !v.OK {
		t.Fatalf("AcquireFenced: verdict=%+v err=%v", v, err)
	}
	_, v, err = s.RenewFenced(ctx(), "lane", "worker", 0, 600, t0.Add(time.Second))
	if err != nil {
		t.Fatalf("RenewFenced: %v", err)
	}
	if v.OK || v.Reason != ReasonStaleLease || v.Presented != 0 || v.Current != rec.Generation {
		t.Fatalf("verdict=%+v, want STALE_LEASE presented=0 current=%d", v, rec.Generation)
	}
}

// TestRenewFencedCASLossIsContended deterministically advances the ref after the
// renew reads its record but before its guarded update. The valid token still
// loses closed with LEASE_CONTENDED, and the racer's record remains authoritative.
func TestRenewFencedCASLossIsContended(t *testing.T) {
	g := newFakeGit()
	t0 := time.Unix(1000, 0)
	seed := NewWithRunner(g.run, "")
	rec, v, err := seed.AcquireFenced(ctx(), Record{ID: "lane", Holder: "worker", TTLSeconds: 300}, t0)
	if err != nil || !v.OK {
		t.Fatalf("AcquireFenced: verdict=%+v err=%v", v, err)
	}

	raced := false
	runner := func(c context.Context, dir string, args ...string) (string, int, error) {
		if !raced && len(args) >= 4 && args[0] == "update-ref" && args[1] == rec.Ref() {
			raced = true
			g.next++
			racerOID := synthID(g.next)
			g.blobs[racerOID] = []byte(`{"id":"lane","holder":"racer","generation":2,"acquired_unix":1001,"ttl_seconds":300}`)
			g.refs[rec.Ref()] = racerOID
		}
		return g.run(c, dir, args...)
	}
	s := NewWithRunner(runner, "")
	_, v, err = s.RenewFenced(ctx(), "lane", "worker", rec.Generation, 600, t0.Add(time.Second))
	if err != nil {
		t.Fatalf("RenewFenced: %v", err)
	}
	if !raced {
		t.Fatal("race injection did not run")
	}
	if v.OK || v.Reason != ReasonLeaseContended {
		t.Fatalf("verdict=%+v, want LEASE_CONTENDED", v)
	}
	got, ok, err := s.Get(ctx(), "lane")
	if err != nil || !ok || got.Holder != "racer" || got.Generation != 2 {
		t.Fatalf("CAS loser overwrote racer: got=%+v ok=%v err=%v", got, ok, err)
	}
}
