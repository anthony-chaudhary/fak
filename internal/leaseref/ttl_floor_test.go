package leaseref

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"
)

// plantLegacyLease writes a lock-lease blob verbatim into the fake object store, bypassing
// the write-side clamp, to model a ttl-0 ref published by an older binary (the eleven
// ghosts behind fak-private#3076).
func plantLegacyLease(t *testing.T, g *fakeGit, rec Record) {
	t.Helper()
	blob, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("marshal legacy record: %v", err)
	}
	g.next++
	oid := fmt.Sprintf("%040x", 0xfeed0000+g.next)
	g.blobs[oid] = blob
	g.refs[refPrefix+rec.ID] = oid
}

// TestAcquireClampsNonPositiveTTL: the blind Acquire never persists ttl<=0.
func TestAcquireClampsNonPositiveTTL(t *testing.T) {
	for _, ttl := range []int64{0, -30} {
		g := newFakeGit()
		s := NewWithRunner(g.run, "")
		if _, err := s.Acquire(ctx(), Record{ID: "lane", TreeGlobs: []string{"a/**"}, Holder: "h", TTLSeconds: ttl}); err != nil {
			t.Fatalf("Acquire ttl=%d: %v", ttl, err)
		}
		got, ok, err := s.Get(ctx(), "lane")
		if err != nil || !ok {
			t.Fatalf("Get after Acquire ttl=%d: ok=%v err=%v", ttl, ok, err)
		}
		if got.TTLSeconds != DefaultLeaseTTLSeconds {
			t.Fatalf("persisted ttl=%d for requested ttl=%d, want DefaultLeaseTTLSeconds=%d", got.TTLSeconds, ttl, DefaultLeaseTTLSeconds)
		}
	}
}

// TestAcquireFencedClampsZeroTTL: the fenced acquire (the `fak leaseref acquire` path, whose
// --ttl default is 0) lands a bounded lease, both in the returned record and on the ref.
func TestAcquireFencedClampsZeroTTL(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	t0 := time.Unix(1_000_000, 0)
	rec, v, err := s.AcquireFenced(ctx(), Record{ID: "lane", Holder: "h", TTLSeconds: 0}, t0)
	if err != nil || !v.OK {
		t.Fatalf("AcquireFenced: verdict=%+v err=%v", v, err)
	}
	if rec.TTLSeconds != DefaultLeaseTTLSeconds {
		t.Fatalf("returned ttl=%d, want %d", rec.TTLSeconds, DefaultLeaseTTLSeconds)
	}
	got, _, _ := s.Get(ctx(), "lane")
	if got.TTLSeconds != DefaultLeaseTTLSeconds {
		t.Fatalf("persisted ttl=%d, want %d", got.TTLSeconds, DefaultLeaseTTLSeconds)
	}
	if !got.Expired(t0.Add(time.Duration(DefaultLeaseTTLSeconds) * time.Second)) {
		t.Fatal("a clamped lease must expire after DefaultLeaseTTLSeconds")
	}
}

// TestRenewUpgradesLegacyZeroTTL: renewing a live legacy ttl-0 lease without an explicit TTL
// rewrites it bounded instead of re-persisting the forever value.
func TestRenewUpgradesLegacyZeroTTL(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	t0 := time.Unix(1_000_000, 0)
	plantLegacyLease(t, g, Record{ID: "lane", Holder: "h", AcquiredAt: t0.Unix(), TTLSeconds: 0, Generation: 1})
	rec, v, err := s.RenewFenced(ctx(), "lane", "h", 1, 0, t0.Add(time.Minute))
	if err != nil || !v.OK {
		t.Fatalf("RenewFenced: verdict=%+v err=%v", v, err)
	}
	if rec.TTLSeconds != DefaultLeaseTTLSeconds {
		t.Fatalf("renewed ttl=%d, want %d", rec.TTLSeconds, DefaultLeaseTTLSeconds)
	}
	if got, _, _ := s.Get(ctx(), "lane"); got.TTLSeconds != DefaultLeaseTTLSeconds {
		t.Fatalf("persisted renewed ttl=%d, want %d", got.TTLSeconds, DefaultLeaseTTLSeconds)
	}
}

// TestLegacyZeroTTLLeaseAgesOutToReapable: a legacy ttl-0 lease older than
// LegacyNoTTLMaxAgeSeconds lands in Live's expired partition (so the normal Reap would
// collect it), while one younger than the bound stays live and is not reaped.
func TestLegacyZeroTTLLeaseAgesOutToReapable(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	now := time.Unix(2_000_000_000, 0)
	plantLegacyLease(t, g, Record{ID: "ghost", TreeGlobs: []string{"internal/rsl/**"}, Holder: "old", AcquiredAt: now.Unix() - 16*24*3600})
	plantLegacyLease(t, g, Record{ID: "young", TreeGlobs: []string{"internal/x/**"}, Holder: "new", AcquiredAt: now.Unix() - LegacyNoTTLMaxAgeSeconds + 60})
	plantLegacyLease(t, g, Record{ID: "undated", TreeGlobs: []string{"internal/y/**"}, Holder: "anon"})

	live, expiredIDs, err := s.Live(ctx(), now)
	if err != nil {
		t.Fatalf("Live: %v", err)
	}
	if len(expiredIDs) != 1 || expiredIDs[0] != "ghost" {
		t.Fatalf("expired = %v, want [ghost]", expiredIDs)
	}
	if len(live) != 2 {
		t.Fatalf("live = %+v, want young + undated (an undated record fails closed to live)", live)
	}

	reaped, err := s.Reap(ctx(), now)
	if err != nil {
		t.Fatalf("Reap: %v", err)
	}
	if len(reaped) != 1 || reaped[0] != "ghost" {
		t.Fatalf("reaped = %v, want [ghost]", reaped)
	}
	if _, ok, _ := s.Get(ctx(), "young"); !ok {
		t.Fatal("Reap removed a legacy ttl-0 lease younger than the age bound")
	}
}
