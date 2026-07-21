package dispatchtick

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostplacement"
)

// spillBase is a fixed reference instant; all heartbeat timestamps are offsets
// from it, so the suite carries no wall-clock dependency.
var spillBase = time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)

const spillTTL = 30 * time.Second

func TestParseDispatchHosts(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want []string
	}{
		{name: "empty_is_single_host_default", raw: "", want: nil},
		{name: "whitespace_only_is_nil", raw: "  \t ", want: nil},
		{name: "comma_separated", raw: "host-a,host-b", want: []string{"host-a", "host-b"}},
		{name: "mixed_commas_spaces_and_empties", raw: " host-a , host-b\thost-c ,,", want: []string{"host-a", "host-b", "host-c"}},
		{name: "duplicates_collapse_first_wins", raw: "host-a,host-b,host-a", want: []string{"host-a", "host-b"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ParseDispatchHosts(tc.raw)
			if len(got) != len(tc.want) {
				t.Fatalf("ParseDispatchHosts(%q) = %v, want %v", tc.raw, got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("ParseDispatchHosts(%q) = %v, want %v", tc.raw, got, tc.want)
				}
			}
		})
	}
}

// fakeSpillRegistry builds the fake multi-host registry the placement tests
// run against: primary host-a and spill target host-b, with the given
// saturations, both heartbeating fresh at spillBase.
func fakeSpillRegistry(satA, satB float64) *hostplacement.Registry {
	r := hostplacement.NewRegistry()
	r.Observe(hostplacement.Heartbeat{Hostname: "host-a", LiveHeadroom: 4, Saturation: satA, TS: spillBase})
	r.Observe(hostplacement.Heartbeat{Hostname: "host-b", LiveHeadroom: 6, Saturation: satB, TS: spillBase})
	return r
}

// TestResolveHostSpill is the dispatch-seam DoD table over a fake multi-host
// registry: primary-full spills to the second host, both-full declines, a
// stale heartbeat is never placed on, and the single-host default (no
// FAK_DISPATCH_HOSTS) stays local unconditionally.
func TestResolveHostSpill(t *testing.T) {
	now := spillBase

	staleB := hostplacement.NewRegistry()
	staleB.Observe(hostplacement.Heartbeat{Hostname: "host-a", LiveHeadroom: 4, Saturation: 0.95, TS: now})
	staleB.Observe(hostplacement.Heartbeat{Hostname: "host-b", LiveHeadroom: 9, Saturation: 0.05, TS: now.Add(-spillTTL - time.Second)})

	cases := []struct {
		name      string
		hostsRaw  string
		reg       *hostplacement.Registry
		wantHost  string
		wantLocal bool
		wantWhy   string
	}{
		{
			// Primary host-a at capacity -> the next worker SPILLS to host-b.
			name:     "primary_full_spills_to_second_host",
			hostsRaw: "host-a,host-b",
			reg:      fakeSpillRegistry(0.95, 0.30),
			wantHost: "host-b",
			wantWhy:  hostplacement.ReasonPlaced,
		},
		{
			// Both hosts at capacity -> the spill is DECLINED; stay local.
			name:      "both_full_declines",
			hostsRaw:  "host-a,host-b",
			reg:       fakeSpillRegistry(0.95, 0.97),
			wantLocal: true,
			wantWhy:   hostplacement.ReasonNoneEligible,
		},
		{
			// Primary under threshold keeps the worker on the least-saturated
			// eligible host (host-a itself).
			name:     "primary_free_places_on_primary",
			hostsRaw: "host-a,host-b",
			reg:      fakeSpillRegistry(0.10, 0.30),
			wantHost: "host-a",
			wantWhy:  hostplacement.ReasonPlaced,
		},
		{
			// host-b's heartbeat is older than the TTL -> unavailable, never
			// placed on, and with host-a full the spill is declined.
			name:      "stale_second_host_never_placed",
			hostsRaw:  "host-a,host-b",
			reg:       staleB,
			wantLocal: true,
			wantWhy:   hostplacement.ReasonNoneEligible,
		},
		{
			// Single-host default: no FAK_DISPATCH_HOSTS -> stay local even
			// though the registry holds a free, fresh host.
			name:      "no_hosts_env_stays_local",
			hostsRaw:  "",
			reg:       fakeSpillRegistry(0.10, 0.10),
			wantLocal: true,
			wantWhy:   hostplacement.ReasonNoHosts,
		},
		{
			// A heartbeat from an UNCONFIGURED host is ignored: only host-a is
			// configured and it is full, so free host-b cannot be placed on.
			name:      "unconfigured_host_ignored",
			hostsRaw:  "host-a",
			reg:       fakeSpillRegistry(0.95, 0.05),
			wantLocal: true,
			wantWhy:   hostplacement.ReasonNoneEligible,
		},
		{
			// Configured hosts with no heartbeat at all have no headroom fact:
			// decline, not the single-host default.
			name:      "configured_but_no_heartbeats_declines",
			hostsRaw:  "host-a,host-b",
			reg:       hostplacement.NewRegistry(),
			wantLocal: true,
			wantWhy:   hostplacement.ReasonNoneEligible,
		},
		{
			// A nil registry is a valid empty registry.
			name:      "nil_registry_declines",
			hostsRaw:  "host-a,host-b",
			reg:       nil,
			wantLocal: true,
			wantWhy:   hostplacement.ReasonNoneEligible,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ResolveHostSpill(tc.hostsRaw, tc.reg, 0.90, now, spillTTL)
			if got.StayLocal != tc.wantLocal {
				t.Fatalf("StayLocal = %v, want %v (decision %+v)", got.StayLocal, tc.wantLocal, got)
			}
			if got.Host != tc.wantHost {
				t.Fatalf("Host = %q, want %q (decision %+v)", got.Host, tc.wantHost, got)
			}
			if got.Reason != tc.wantWhy {
				t.Fatalf("Reason = %q, want %q", got.Reason, tc.wantWhy)
			}
			// Determinism: the same inputs must yield the same decision.
			if again := ResolveHostSpill(tc.hostsRaw, tc.reg, 0.90, now, spillTTL); again != got {
				t.Fatalf("non-deterministic: %+v then %+v", got, again)
			}
		})
	}
}

// TestResolveHostSpillDefaults proves a non-positive threshold/ttl selects the
// package defaults at this seam (zero knob = default), instead of the raw
// primitive's "non-positive TTL is never fresh".
func TestResolveHostSpillDefaults(t *testing.T) {
	now := spillBase
	// host-b is under the DEFAULT 0.90 threshold and its heartbeat is within
	// the DEFAULT 90s TTL (60s old) — but would be STALE under any tiny
	// explicit ttl and ineligible under a tiny explicit threshold.
	r := hostplacement.NewRegistry()
	r.Observe(hostplacement.Heartbeat{Hostname: "host-a", Saturation: 0.95, TS: now})
	r.Observe(hostplacement.Heartbeat{Hostname: "host-b", Saturation: 0.50, TS: now.Add(-60 * time.Second)})

	got := ResolveHostSpill("host-a,host-b", r, 0, now, 0)
	if got.StayLocal || got.Host != "host-b" || got.Reason != hostplacement.ReasonPlaced {
		t.Fatalf("defaults not applied: %+v, want host-b placed", got)
	}
}
