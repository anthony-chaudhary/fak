package fleetspine

import (
	"testing"
	"time"
)

// mkHB builds a valid heartbeat with a parseable stamp for the given id at time ts.
func mkHB(id string, ts time.Time) Heartbeat {
	return Heartbeat{
		Schema:       HeartbeatSchema,
		ID:           id,
		Host:         id + "-host",
		State:        "OK",
		AppVersion:   "v1.2.3",
		Sessions:     2,
		GeneratedUTC: ts.UTC().Format(time.RFC3339),
	}
}

// TestRegistryIngestUpsertAndSelfDrop: two distinct peers are stored; a heartbeat whose id is
// our own (multicast self-echo) is dropped, so the panel stays peer-only.
func TestRegistryIngestUpsertAndSelfDrop(t *testing.T) {
	t0 := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	r := NewRegistry(RegistryConfig{SelfID: "me", Now: func() time.Time { return t0 }})

	r.Ingest(mkHB("alpha", t0), t0)
	r.Ingest(mkHB("beta", t0), t0)
	r.Ingest(mkHB("me", t0), t0) // self-echo — must be dropped

	snap := r.Snapshot(t0)
	if len(snap) != 2 {
		t.Fatalf("snapshot len = %d, want 2 (self-echo dropped)", len(snap))
	}
	if snap[0].ID != "alpha" || snap[1].ID != "beta" {
		t.Fatalf("snapshot ids = %q,%q, want alpha,beta (sorted)", snap[0].ID, snap[1].ID)
	}

	// Upsert: a newer heartbeat for alpha replaces the old one.
	r.Ingest(Heartbeat{ID: "alpha", State: "ACTION", Sessions: 9, GeneratedUTC: t0.Format(time.RFC3339)}, t0.Add(time.Second))
	snap = r.Snapshot(t0.Add(time.Second))
	if snap[0].State != "ACTION" || snap[0].Sessions != 9 {
		t.Fatalf("alpha after upsert = %+v, want state ACTION / sessions 9", snap[0])
	}
}

// TestRegistryStaleThenExpire drives the time-based liveness rule with an injected clock:
// fresh → OK, past miss-window → STALE, past hard-expiry → dropped.
func TestRegistryStaleThenExpire(t *testing.T) {
	t0 := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	r := NewRegistry(RegistryConfig{
		SelfID:     "me",
		MissWindow: 10 * time.Minute,
		HardExpiry: 20 * time.Minute,
	})
	r.Ingest(mkHB("alpha", t0), t0)

	// Fresh: within miss-window → not stale, state carries the verdict.
	maps := r.MachineMaps(t0.Add(1 * time.Minute))
	if len(maps) != 1 || maps[0]["state"] != "OK" {
		t.Fatalf("fresh maps = %+v, want 1 machine state OK", maps)
	}

	// Past miss-window, within hard-expiry → STALE, still present.
	maps = r.MachineMaps(t0.Add(11 * time.Minute))
	if len(maps) != 1 || maps[0]["state"] != "STALE" {
		t.Fatalf("stale maps = %+v, want 1 machine state STALE", maps)
	}

	// Past hard-expiry → dropped entirely (read-time prune).
	maps = r.MachineMaps(t0.Add(21 * time.Minute))
	if len(maps) != 0 {
		t.Fatalf("expired maps = %+v, want 0 machines (hard-expired)", maps)
	}
}

// TestRegistryExpireLoopDropsDeparted: the explicit Expire call (the RunExpiry body) drops a
// peer even when no read happens to trigger the inline prune.
func TestRegistryExpireLoopDropsDeparted(t *testing.T) {
	t0 := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	r := NewRegistry(RegistryConfig{MissWindow: 5 * time.Minute, HardExpiry: 10 * time.Minute})
	r.Ingest(mkHB("gamma", t0), t0)

	r.Expire(t0.Add(6 * time.Minute)) // stale but not hard-expired → kept
	if got := len(r.Snapshot(t0.Add(6 * time.Minute))); got != 1 {
		t.Fatalf("after stale Expire: %d peers, want 1", got)
	}
	r.Expire(t0.Add(11 * time.Minute)) // hard-expired → dropped
	if got := len(r.Snapshot(t0.Add(11 * time.Minute))); got != 0 {
		t.Fatalf("after hard Expire: %d peers, want 0", got)
	}
}

// TestRegistryMalformedRejected: an empty id and an unparseable stamp are both dropped.
func TestRegistryMalformedRejected(t *testing.T) {
	t0 := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	r := NewRegistry(RegistryConfig{})
	r.Ingest(Heartbeat{ID: "", GeneratedUTC: t0.Format(time.RFC3339)}, t0)    // empty id
	r.Ingest(Heartbeat{ID: "bad-stamp", GeneratedUTC: "not-a-timestamp"}, t0) // bad stamp
	r.Ingest(Heartbeat{ID: "blank-stamp", GeneratedUTC: ""}, t0)              // blank stamp
	if got := len(r.Snapshot(t0)); got != 0 {
		t.Fatalf("malformed heartbeats produced %d peers, want 0", got)
	}
}

// TestRegistryBoundedMap: at the cap, a brand-new id is ignored while a known id still updates.
func TestRegistryBoundedMap(t *testing.T) {
	t0 := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	r := NewRegistry(RegistryConfig{MaxPeers: 2})
	r.Ingest(mkHB("a", t0), t0)
	r.Ingest(mkHB("b", t0), t0)
	r.Ingest(mkHB("c", t0), t0) // map full → dropped
	if got := len(r.Snapshot(t0)); got != 2 {
		t.Fatalf("bounded map has %d peers, want 2 (new id past cap dropped)", got)
	}
	// A known id still updates at the cap.
	r.Ingest(Heartbeat{ID: "a", State: "ACTION", GeneratedUTC: t0.Format(time.RFC3339)}, t0.Add(time.Second))
	for _, p := range r.Snapshot(t0.Add(time.Second)) {
		if p.ID == "a" && p.State != "ACTION" {
			t.Fatalf("known id a not updated at cap: state=%q", p.State)
		}
	}
}

// TestRegistryRateLimit: with a MinInterval, a too-soon second heartbeat from a known id is
// ignored (the state does not advance), while one after the interval is accepted.
func TestRegistryRateLimit(t *testing.T) {
	t0 := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	r := NewRegistry(RegistryConfig{MinInterval: time.Minute})
	r.Ingest(mkHB("a", t0), t0)
	r.Ingest(Heartbeat{ID: "a", State: "ACTION", GeneratedUTC: t0.Format(time.RFC3339)}, t0.Add(10*time.Second)) // too soon
	if r.Snapshot(t0.Add(10 * time.Second))[0].State != "OK" {
		t.Fatal("rate-limited heartbeat was applied (state advanced), want ignored")
	}
	r.Ingest(Heartbeat{ID: "a", State: "ACTION", GeneratedUTC: t0.Format(time.RFC3339)}, t0.Add(2*time.Minute)) // ok
	if r.Snapshot(t0.Add(2 * time.Minute))[0].State != "ACTION" {
		t.Fatal("post-interval heartbeat was not applied")
	}
}

// TestMachineMapsShapeMatchesFold locks the union contract: MachineMaps produces exactly the
// keys the guard fleet fold (guardFleetFromDoc) reads off a machine map.
func TestMachineMapsShapeMatchesFold(t *testing.T) {
	t0 := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	r := NewRegistry(RegistryConfig{MissWindow: 10 * time.Minute, HardExpiry: 20 * time.Minute})
	r.Ingest(mkHB("alpha", t0), t0)
	m := r.MachineMaps(t0.Add(time.Minute))[0]
	for _, k := range []string{"id", "host", "state", "age_min", "sessions", "app_version", "generated_utc"} {
		if _, ok := m[k]; !ok {
			t.Fatalf("machine map missing key %q (fold reads it): %+v", k, m)
		}
	}
	if m["id"] != "alpha" || m["app_version"] != "v1.2.3" || m["sessions"] != 2 {
		t.Fatalf("machine map values = %+v", m)
	}
	if age, ok := m["age_min"].(float64); !ok || age <= 0 {
		t.Fatalf("age_min = %v, want positive float minutes", m["age_min"])
	}
}
