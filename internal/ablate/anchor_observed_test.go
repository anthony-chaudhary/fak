package ablate

import (
	"math"
	"strings"
	"testing"
)

// The conservative floor is exactly the gross shed minus the entire window write premium — the
// deliberate over-charge that makes a positive result a lower bound on the true net-of-burst benefit.
func TestObservedAnchorArm_ConservativeNetFloor(t *testing.T) {
	a := ObservedAnchorArm{GrossShedUSD: 2809.0038, WindowWritePremiumUSD: 351.3678, Fires: 575}
	want := 2809.0038 - 351.3678
	if got := a.ConservativeNetFloorUSD(); math.Abs(got-want) > 1e-9 {
		t.Fatalf("ConservativeNetFloorUSD = %v, want %v", got, want)
	}
}

// The three-way worded verdict, and the invariant that a directional (net-beneficial) claim is made
// only when the conservative floor is strictly positive — the worded read may not out-run the floor.
func TestObservedAnchorArm_VerdictThreeWay(t *testing.T) {
	cases := []struct {
		name        string
		arm         ObservedAnchorArm
		wantPhrase  string
		directional bool // a net-beneficial claim is earned
	}{
		{
			name:        "net_beneficial_floor_positive",
			arm:         ObservedAnchorArm{GrossShedUSD: 2809.0038, WindowWritePremiumUSD: 351.3678, Fires: 575, WindowStart: "2026-07-04", WindowEnd: "2026-07-11"},
			wantPhrase:  "IS net-beneficial",
			directional: true,
		},
		{
			name:        "undistinguished_floor_nonpositive",
			arm:         ObservedAnchorArm{GrossShedUSD: 100, WindowWritePremiumUSD: 250, Fires: 40, WindowStart: "2026-07-04", WindowEnd: "2026-07-11"},
			wantPhrase:  "NOT DISTINGUISHABLE",
			directional: false,
		},
		{
			name:        "unwitnessed_no_fires",
			arm:         ObservedAnchorArm{GrossShedUSD: 0, WindowWritePremiumUSD: 12, Fires: 0, WindowStart: "2026-07-04", WindowEnd: "2026-07-11"},
			wantPhrase:  "UNWITNESSED",
			directional: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.arm.Verdict()
			if !strings.Contains(got, tc.wantPhrase) {
				t.Errorf("Verdict() = %q, want phrase %q", got, tc.wantPhrase)
			}
			directional := strings.Contains(got, "IS net-beneficial")
			if directional != tc.directional {
				t.Errorf("directional=%v, want %v (floor=%v) for %q", directional, tc.directional, tc.arm.ConservativeNetFloorUSD(), got)
			}
			// A directional win must coincide with a strictly-positive floor.
			if directional != (tc.arm.ConservativeNetFloorUSD() > 0) {
				t.Errorf("net-beneficial claim disagrees with floor>0: floor=%v verdict=%q", tc.arm.ConservativeNetFloorUSD(), got)
			}
		})
	}
}

// Real-snapshot pin (docs/nightrun/cache-savings.jsonl, 2026-07-04..2026-07-11): 575 compaction
// fires shed a gross $2809.0038 at input price; the whole window's provider write premium is
// $351.3678. Charging the head arm every window write-dollar still leaves a $+2457.6360 floor, so
// the #1407 de-starvation switch is net positive on real traffic. These aggregates are hard-coded
// fixture inputs (not read from the mutable ledger) so the assertion stays deterministic; the
// published artifact carries the reproducing command over the committed ledger.
func TestObservedAnchorArm_RealTrafficSnapshot(t *testing.T) {
	arm := ObservedAnchorArm{
		GrossShedUSD:          2809.0038,
		WindowWritePremiumUSD: 351.3678,
		Fires:                 575,
		LedgerPath:            "docs/nightrun/cache-savings.jsonl",
		WindowStart:           "2026-07-04",
		WindowEnd:             "2026-07-11",
	}
	if got, want := arm.ConservativeNetFloorUSD(), 2457.6360; math.Abs(got-want) > 1e-4 {
		t.Fatalf("real-traffic floor = %v, want %v", got, want)
	}
	v := arm.Verdict()
	if !strings.Contains(v, "IS net-beneficial") {
		t.Fatalf("real-traffic verdict should be net-beneficial, got %q", v)
	}
	for _, s := range []string{"N=575", "2026-07-04", "2026-07-11"} {
		if !strings.Contains(v, s) {
			t.Errorf("real-traffic verdict missing %q: %q", s, v)
		}
	}
	if arm.Caveat() == "" {
		t.Errorf("observational arm must carry the single-arm caveat")
	}
}
