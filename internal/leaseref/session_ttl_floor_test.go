package leaseref

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"
)

// plantLegacySession writes a session-descriptor blob verbatim into the fake object store,
// bypassing PublishSession's write-side clamp, to model a ttl-0 descriptor published by an
// older binary (the RUNNING ghosts `session-publish --ttl 0` left on the remote).
func plantLegacySession(t *testing.T, g *fakeGit, d SessionDescriptor) {
	t.Helper()
	blob, err := json.Marshal(d)
	if err != nil {
		t.Fatalf("marshal legacy descriptor: %v", err)
	}
	g.next++
	oid := fmt.Sprintf("%040x", 0xbeef0000+g.next)
	g.blobs[oid] = blob
	g.refs[d.Ref()] = oid
}

// TestPublishSessionClampsNonPositiveTTL: PublishSession never persists ttl<=0, and the
// clamped descriptor lapses DefaultSessionTTLSeconds after its UpdatedAt.
//
// fak-test:runtime fast est=5ms lane=default
func TestPublishSessionClampsNonPositiveTTL(t *testing.T) {
	for _, ttl := range []int64{0, -30} {
		g := newFakeGit()
		s := NewWithRunner(g.run, "")
		d := SessionDescriptor{ID: "sess-clamp", Host: "h", PCBState: "RUNNING", UpdatedAt: 1_000_000, TTLSecs: ttl}
		if _, err := s.PublishSession(ctx(), d); err != nil {
			t.Fatalf("PublishSession ttl=%d: %v", ttl, err)
		}
		got, ok, err := s.GetSession(ctx(), "sess-clamp")
		if err != nil || !ok {
			t.Fatalf("GetSession ttl=%d: ok=%v err=%v", ttl, ok, err)
		}
		if got.TTLSecs != DefaultSessionTTLSeconds {
			t.Fatalf("persisted ttl=%d for requested ttl=%d, want DefaultSessionTTLSeconds=%d", got.TTLSecs, ttl, DefaultSessionTTLSeconds)
		}
		end := time.Unix(d.UpdatedAt+DefaultSessionTTLSeconds, 0)
		if got.Expired(end.Add(-time.Second)) || !got.Expired(end) {
			t.Fatalf("clamped descriptor must lapse exactly at UpdatedAt+%d", DefaultSessionTTLSeconds)
		}
	}
}

// TestPublishSessionKeepsPositiveTTL: an explicit positive TTL is written through untouched.
//
// fak-test:runtime fast est=2ms lane=default
func TestPublishSessionKeepsPositiveTTL(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	if _, err := s.PublishSession(ctx(), SessionDescriptor{ID: "sess-pos", Host: "h", PCBState: "RUNNING", UpdatedAt: 10, TTLSecs: 45}); err != nil {
		t.Fatalf("PublishSession: %v", err)
	}
	got, _, _ := s.GetSession(ctx(), "sess-pos")
	if got.TTLSecs != 45 {
		t.Fatalf("persisted ttl=%d, want the explicit 45", got.TTLSecs)
	}
}

// TestLegacyNoTTLSessionAgesOut: a legacy ttl-0 descriptor (planted past the clamp) stays
// LIVE while younger than LegacyNoTTLMaxAgeSeconds, drops into LiveSessions' expired set
// once its UpdatedAt is that old, and ReapSessions deletes it; an undated legacy descriptor
// has no age and fails closed to live.
//
// fak-test:runtime fast est=5ms lane=default
func TestLegacyNoTTLSessionAgesOut(t *testing.T) {
	now := time.Unix(10_000_000, 0)
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	plantLegacySession(t, g, SessionDescriptor{ID: "ghost", Host: "h", PCBState: "RUNNING", UpdatedAt: now.Unix() - LegacyNoTTLMaxAgeSeconds, TTLSecs: 0})
	plantLegacySession(t, g, SessionDescriptor{ID: "young", Host: "h", PCBState: "RUNNING", UpdatedAt: now.Unix() - LegacyNoTTLMaxAgeSeconds + 60, TTLSecs: 0})
	plantLegacySession(t, g, SessionDescriptor{ID: "undated", Host: "h", PCBState: "RUNNING", UpdatedAt: 0, TTLSecs: 0})

	live, expired, err := s.LiveSessions(ctx(), now)
	if err != nil {
		t.Fatalf("LiveSessions: %v", err)
	}
	if len(expired) != 1 || expired[0] != "ghost" {
		t.Fatalf("expired = %v, want only [ghost]", expired)
	}
	liveIDs := map[string]bool{}
	for _, d := range live {
		liveIDs[d.ID] = true
	}
	if len(live) != 2 || !liveIDs["young"] || !liveIDs["undated"] {
		t.Fatalf("live = %+v, want [young undated]", live)
	}

	reaped, err := s.ReapSessions(ctx(), now)
	if err != nil {
		t.Fatalf("ReapSessions: %v", err)
	}
	if len(reaped) != 1 || reaped[0] != "ghost" {
		t.Fatalf("reaped = %v, want only [ghost]", reaped)
	}
	if _, ok, _ := s.GetSession(ctx(), "ghost"); ok {
		t.Fatal("ghost descriptor still present after ReapSessions")
	}
}

// TestLegacyNoTTLSessionClassifiedDead: the lease-liveness classifier treats a lease whose
// owning session is an aged-out legacy ttl-0 descriptor as heartbeat-lapsed (peer dead),
// not heartbeating forever.
//
// fak-test:runtime fast est=2ms lane=default
func TestLegacyNoTTLSessionClassifiedDead(t *testing.T) {
	now := time.Unix(10_000_000, 0)
	d := SessionDescriptor{ID: "ghost", Host: "h", PCBState: "RUNNING", UpdatedAt: now.Unix() - LegacyNoTTLMaxAgeSeconds - 1, TTLSecs: 0}
	rec := Record{ID: "lane", Holder: "h", SessionID: "ghost", AcquiredAt: now.Unix() - 60, TTLSeconds: 3600}
	class, kind, _ := ClassifyLiveness(rec, map[string]SessionDescriptor{"ghost": d}, "", now)
	if class != LivenessPeerDead || kind != EvidenceHeartbeatLapsed {
		t.Fatalf("aged-out legacy owner: class=%s kind=%s, want %s/%s", class, kind, LivenessPeerDead, EvidenceHeartbeatLapsed)
	}
	fresh := d
	fresh.UpdatedAt = now.Unix() - 60
	class, kind, _ = ClassifyLiveness(rec, map[string]SessionDescriptor{"ghost": fresh}, "", now)
	if class != LivenessPeerLive || kind != EvidenceHeartbeating {
		t.Fatalf("fresh legacy owner: class=%s kind=%s, want %s/%s", class, kind, LivenessPeerLive, EvidenceHeartbeating)
	}
}
