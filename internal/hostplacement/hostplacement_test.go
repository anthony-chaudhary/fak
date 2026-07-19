package hostplacement

import (
	"testing"
	"time"
)

// base is a fixed reference instant. All test timestamps are expressed relative
// to it, so the suite carries no wall-clock dependency: "time" is just an offset.
var base = time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)

const ttl = 30 * time.Second

// TestPlace is the DoD table: it proves all three acceptance witnesses of #3599
// over the pure placement function, plus the empty (single-host default) case.
func TestPlace(t *testing.T) {
	now := base
	// A over its saturation threshold, B under it; both fresh.
	hostASaturated := Heartbeat{Hostname: "host-a", LiveHeadroom: 5, Saturation: 0.95, TS: now}
	hostBFree := Heartbeat{Hostname: "host-b", LiveHeadroom: 3, Saturation: 0.30, TS: now}
	// A fresh + eligible; B stale (heartbeat older than the TTL).
	hostAFreshLow := Heartbeat{Hostname: "host-a", LiveHeadroom: 2, Saturation: 0.20, TS: now}
	hostBStale := Heartbeat{Hostname: "host-b", LiveHeadroom: 9, Saturation: 0.01, TS: now.Add(-ttl - time.Second)}

	cases := []struct {
		name      string
		hosts     []Heartbeat
		threshold float64
		wantHost  string
		wantLocal bool
		wantWhy   string
	}{
		{
			// Witness 1: two hosts, A over threshold -> place on B (deterministic).
			name:      "saturated_host_a_spills_to_b",
			hosts:     []Heartbeat{hostASaturated, hostBFree},
			threshold: 0.90,
			wantHost:  "host-b",
			wantWhy:   ReasonPlaced,
		},
		{
			// Witness 2: B's heartbeat age > TTL -> B is unavailable, never placed
			// on even though it is the least saturated; the fresh A wins instead.
			name:      "stale_host_never_placed",
			hosts:     []Heartbeat{hostAFreshLow, hostBStale},
			threshold: 0.90,
			wantHost:  "host-a",
			wantWhy:   ReasonPlaced,
		},
		{
			// Witness 2b: the ONLY candidate is stale -> stay local, none eligible.
			name:      "only_candidate_stale_stays_local",
			hosts:     []Heartbeat{hostBStale},
			threshold: 0.90,
			wantLocal: true,
			wantWhy:   ReasonNoneEligible,
		},
		{
			// Witness 3: single host, and it is over the threshold -> stay local.
			name:      "single_saturated_host_stays_local",
			hosts:     []Heartbeat{hostASaturated},
			threshold: 0.90,
			wantLocal: true,
			wantWhy:   ReasonNoneEligible,
		},
		{
			// Witness 3c: BOTH hosts at capacity (primary and the spill target) ->
			// the placement declines to spill anywhere and stays local.
			name: "both_hosts_full_declines",
			hosts: []Heartbeat{
				hostASaturated,
				{Hostname: "host-b", LiveHeadroom: 0, Saturation: 0.97, TS: now},
			},
			threshold: 0.90,
			wantLocal: true,
			wantWhy:   ReasonNoneEligible,
		},
		{
			// Witness 3b: no hosts configured (single-host default) -> stay local.
			name:      "no_hosts_stays_local",
			hosts:     nil,
			threshold: 0.90,
			wantLocal: true,
			wantWhy:   ReasonNoHosts,
		},
		{
			// A host exactly at the threshold is NOT eligible (threshold is a
			// ceiling): the strictly-lower host wins.
			name:      "at_threshold_is_ineligible",
			hosts:     []Heartbeat{{Hostname: "host-a", Saturation: 0.90, TS: now}, hostBFree},
			threshold: 0.90,
			wantHost:  "host-b",
			wantWhy:   ReasonPlaced,
		},
		{
			// Least-saturated wins among several eligible hosts.
			name: "picks_least_saturated",
			hosts: []Heartbeat{
				{Hostname: "host-a", Saturation: 0.50, TS: now},
				{Hostname: "host-b", Saturation: 0.10, TS: now},
				{Hostname: "host-c", Saturation: 0.30, TS: now},
			},
			threshold: 0.90,
			wantHost:  "host-b",
			wantWhy:   ReasonPlaced,
		},
		{
			// Equal saturation -> tie broken by higher LiveHeadroom (host-b) not by
			// hostname order (host-a would win lexicographically).
			name: "tie_broken_by_headroom",
			hosts: []Heartbeat{
				{Hostname: "host-a", LiveHeadroom: 1, Saturation: 0.20, TS: now},
				{Hostname: "host-b", LiveHeadroom: 8, Saturation: 0.20, TS: now},
			},
			threshold: 0.90,
			wantHost:  "host-b",
			wantWhy:   ReasonPlaced,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Place(tc.hosts, tc.threshold, now, ttl)
			if got.StayLocal != tc.wantLocal {
				t.Fatalf("StayLocal = %v, want %v (decision %+v)", got.StayLocal, tc.wantLocal, got)
			}
			if got.Host != tc.wantHost {
				t.Fatalf("Host = %q, want %q (decision %+v)", got.Host, tc.wantHost, got)
			}
			if got.Reason != tc.wantWhy {
				t.Fatalf("Reason = %q, want %q", got.Reason, tc.wantWhy)
			}
			// Determinism: a second identical call must return the same decision.
			if again := Place(tc.hosts, tc.threshold, now, ttl); again != got {
				t.Fatalf("non-deterministic: %+v then %+v", got, again)
			}
		})
	}
}

// TestRegistryFoldsLatestPerHost proves the registry keeps the newest heartbeat
// per host and drops strictly-older out-of-order reports.
func TestRegistryFoldsLatestPerHost(t *testing.T) {
	r := NewRegistry()
	r.Observe(Heartbeat{Hostname: "host-a", Saturation: 0.10, TS: base})
	r.Observe(Heartbeat{Hostname: "host-b", Saturation: 0.20, TS: base})
	// Newer report for host-a wins.
	r.Observe(Heartbeat{Hostname: "host-a", Saturation: 0.80, TS: base.Add(time.Second)})
	// Strictly-older out-of-order report for host-b is ignored.
	r.Observe(Heartbeat{Hostname: "host-b", Saturation: 0.99, TS: base.Add(-time.Second)})

	hosts := r.Hosts()
	if len(hosts) != 2 {
		t.Fatalf("Hosts() len = %d, want 2 (%+v)", len(hosts), hosts)
	}
	// Hosts() is sorted by hostname: [host-a, host-b].
	if hosts[0].Hostname != "host-a" || hosts[0].Saturation != 0.80 {
		t.Fatalf("host-a not folded to newest: %+v", hosts[0])
	}
	if hosts[1].Hostname != "host-b" || hosts[1].Saturation != 0.20 {
		t.Fatalf("host-b took a stale out-of-order report: %+v", hosts[1])
	}
}

// TestRegistryPlace proves the registry-bound Place folds latest-per-host through
// the same pure selection: host-a saturated over threshold -> spill to host-b.
func TestRegistryPlace(t *testing.T) {
	now := base
	r := NewRegistry()
	r.Observe(Heartbeat{Hostname: "host-a", Saturation: 0.10, TS: now})
	r.Observe(Heartbeat{Hostname: "host-a", Saturation: 0.95, TS: now.Add(time.Second)}) // newest: saturated
	r.Observe(Heartbeat{Hostname: "host-b", Saturation: 0.25, TS: now})

	got := r.Place(0.90, now.Add(time.Second), ttl)
	if got.StayLocal || got.Host != "host-b" || got.Reason != ReasonPlaced {
		t.Fatalf("registry Place = %+v, want host-b placed", got)
	}
}

// TestFresh spot-checks the staleness boundary directly: at exactly the TTL a
// heartbeat is still fresh; one tick past it is stale.
func TestFresh(t *testing.T) {
	hb := Heartbeat{Hostname: "h", TS: base}
	if !hb.Fresh(base.Add(ttl), ttl) {
		t.Fatal("heartbeat at exactly TTL should be fresh")
	}
	if hb.Fresh(base.Add(ttl+time.Nanosecond), ttl) {
		t.Fatal("heartbeat one tick past TTL should be stale")
	}
	if hb.Fresh(base, 0) {
		t.Fatal("non-positive TTL should make every heartbeat stale")
	}
}
