package leaseref

import (
	"context"
	"strings"
	"testing"
	"time"
)

// TestReleaseFencedAbsentIsIdempotentOK pins the absent-lease asymmetry: releasing a
// lease that does not exist is an OK (the post-state already holds), not NO_LEASE, and
// issues NO delete plumbing.
func TestReleaseFencedAbsentIsIdempotentOK(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	v, err := s.ReleaseFenced(ctx(), "never-existed", "me", 0, time.Now())
	if err != nil {
		t.Fatalf("ReleaseFenced: %v", err)
	}
	if !v.OK {
		t.Fatalf("verdict = %+v, want idempotent OK for an absent lease", v)
	}
	for _, c := range g.calls {
		if c[0] == "update-ref" {
			t.Fatalf("absent-lease release must not issue update-ref; saw %v", g.calls)
		}
	}
}

// TestReleaseFencedHolderReleasesOwnLiveLease is the happy path: the live holder
// releases its own lease (with and without presenting its generation) and the ref is
// gone; the plumbing never touches a branch or HEAD.
func TestReleaseFencedHolderReleasesOwnLiveLease(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Now()
	rec, av, err := s.AcquireFenced(ctx(), Record{ID: "lane", TreeGlobs: []string{"a/**"}, Holder: "me", TTLSeconds: 3600}, now)
	if err != nil || !av.OK {
		t.Fatalf("AcquireFenced: %+v %v", av, err)
	}
	v, err := s.ReleaseFenced(ctx(), "lane", "me", rec.Generation, now)
	if err != nil {
		t.Fatalf("ReleaseFenced: %v", err)
	}
	if !v.OK {
		t.Fatalf("verdict = %+v, want OK", v)
	}
	if _, ok, _ := s.Get(ctx(), "lane"); ok {
		t.Fatalf("lease still present after release")
	}
	for _, c := range g.calls {
		for _, a := range c {
			if a == "-f" || a == "--force" || a == "HEAD" || strings.HasPrefix(a, "refs/heads/") {
				t.Fatalf("release must never force/touch a branch/HEAD; saw %q in %v", a, c)
			}
		}
	}

	// Generation 0 (not presented) also releases: the fence is optional on release,
	// holder identity is not.
	if _, av, err = s.AcquireFenced(ctx(), Record{ID: "lane", TreeGlobs: []string{"a/**"}, Holder: "me", TTLSeconds: 3600}, now); err != nil || !av.OK {
		t.Fatalf("re-AcquireFenced: %+v %v", av, err)
	}
	if v, err = s.ReleaseFenced(ctx(), "lane", "me", 0, now); err != nil || !v.OK {
		t.Fatalf("release without generation = %+v %v, want OK", v, err)
	}
}

// TestReleaseFencedRefusesOtherHoldersLiveLease pins the fence: a live lease held by a
// different holder — or an anonymous caller, or an anonymous live lease — is refused
// STALE_LEASE and the ref survives.
func TestReleaseFencedRefusesOtherHoldersLiveLease(t *testing.T) {
	now := time.Now()
	for _, tc := range []struct {
		name         string
		leaseHolder  string
		callerHolder string
	}{
		{"different holder", "peer", "me"},
		{"anonymous caller", "peer", ""},
		{"anonymous live lease", "", "me"},
	} {
		g := newFakeGit()
		s := NewWithRunner(g.run, "")
		if _, err := s.Acquire(ctx(), Record{ID: "lane", TreeGlobs: []string{"a/**"}, Holder: tc.leaseHolder, AcquiredAt: now.Unix(), TTLSeconds: 3600}); err != nil {
			t.Fatalf("%s: Acquire: %v", tc.name, err)
		}
		v, err := s.ReleaseFenced(ctx(), "lane", tc.callerHolder, 0, now)
		if err != nil {
			t.Fatalf("%s: ReleaseFenced: %v", tc.name, err)
		}
		if v.OK || v.Reason != ReasonStaleLease {
			t.Fatalf("%s: verdict = %+v, want refused STALE_LEASE", tc.name, v)
		}
		if _, ok, _ := s.Get(ctx(), "lane"); !ok {
			t.Fatalf("%s: refused release still deleted the lease", tc.name)
		}
	}
}

// TestReleaseFencedRefusesStaleGeneration pins the token check: the right holder
// presenting a WRONG non-zero generation is refused STALE_LEASE (fail closed — its
// lease instance is not the live one).
func TestReleaseFencedRefusesStaleGeneration(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Now()
	rec, av, err := s.AcquireFenced(ctx(), Record{ID: "lane", TreeGlobs: []string{"a/**"}, Holder: "me", TTLSeconds: 3600}, now)
	if err != nil || !av.OK {
		t.Fatalf("AcquireFenced: %+v %v", av, err)
	}
	v, err := s.ReleaseFenced(ctx(), "lane", "me", rec.Generation+1, now)
	if err != nil {
		t.Fatalf("ReleaseFenced: %v", err)
	}
	if v.OK || v.Reason != ReasonStaleLease || v.Current != rec.Generation {
		t.Fatalf("verdict = %+v, want refused STALE_LEASE current=%d", v, rec.Generation)
	}
	if _, ok, _ := s.Get(ctx(), "lane"); !ok {
		t.Fatalf("refused release still deleted the lease")
	}
}

// TestReleaseFencedExpiredIsSingleIDReap pins the expired asymmetry: a lapsed record is
// releasable by ANYONE (deleting it is reap semantics, not a write under the lease).
func TestReleaseFencedExpiredIsSingleIDReap(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	if _, err := s.Acquire(ctx(), Record{ID: "dead", TreeGlobs: []string{"a/**"}, Holder: "peer", AcquiredAt: 100, TTLSeconds: 10}); err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	v, err := s.ReleaseFenced(ctx(), "dead", "someone-else", 0, time.Now())
	if err != nil {
		t.Fatalf("ReleaseFenced: %v", err)
	}
	if !v.OK {
		t.Fatalf("verdict = %+v, want OK (expired record is single-id reap)", v)
	}
	if _, ok, _ := s.Get(ctx(), "dead"); ok {
		t.Fatalf("expired lease still present after release")
	}
}

// TestReleaseFencedCASLossIsContended pins the delete-side CAS: a ref that advances
// between the read and the `update-ref -d <ref> <old>` loses the delete and the verdict
// is LEASE_CONTENDED — the racer's newer record survives.
func TestReleaseFencedCASLossIsContended(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Now()
	if _, av, err := s.AcquireFenced(ctx(), Record{ID: "lane", TreeGlobs: []string{"a/**"}, Holder: "me", TTLSeconds: 3600}, now); err != nil || !av.OK {
		t.Fatalf("AcquireFenced: %+v %v", av, err)
	}
	// Race injection: the moment the CAS delete arrives, advance the ref first (a peer's
	// same-host write landing between our read and our delete), then delegate to the fake.
	raced := false
	s2 := NewWithRunner(func(c context.Context, dir string, args ...string) (string, int, error) {
		if !raced && args[0] == "update-ref" && args[1] == "-d" {
			raced = true
			g.next++
			nid := synthID(g.next)
			g.blobs[nid] = []byte(`{"id":"lane","holder":"racer"}`)
			g.refs[args[2]] = nid
		}
		return g.run(c, dir, args...)
	}, "")
	v, err := s2.ReleaseFenced(ctx(), "lane", "me", 0, now)
	if err != nil {
		t.Fatalf("ReleaseFenced: %v", err)
	}
	if v.OK || v.Reason != ReasonLeaseContended {
		t.Fatalf("verdict = %+v, want LEASE_CONTENDED on a CAS loss", v)
	}
	if _, ok, _ := s2.Get(ctx(), "lane"); !ok {
		t.Fatalf("CAS-losing release deleted the racer's record")
	}
}

// TestReleaseFencedInvalidID keeps the unsafe-ref-id guard on the release side.
func TestReleaseFencedInvalidID(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	if _, err := s.ReleaseFenced(ctx(), "a/b", "me", 0, time.Now()); err == nil {
		t.Fatalf("ReleaseFenced must reject an unsafe ref id")
	}
}
