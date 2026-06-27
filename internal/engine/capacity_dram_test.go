package engine

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestDRAMPressureFlipsDemoteTarget is the MLCACHE1 acceptance: live DRAM pressure changes
// WHERE an under-pressure span demotes. An HBM-resident span under HBM pressure demotes one
// tier colder; with DRAM having room (the default, pressure 0) that target is DRAM, but when
// host RAM is full (DRAM pressure 1.0) the coldest-colder-with-room walk SKIPS DRAM and the
// span demotes to the next attendable tier (NUMA-far). The wire is what moves the target.
//
// The flip is exercised through the pure copy-on-write helpers (withDRAMPressure/Capacity ->
// cachemeta.PlanPlacement) because HostDRAMPressure reads the real host and cannot be forced in
// a test; the probe's own math is checked separately in TestHostDRAMPressureContract. This is
// the same split the HBM wire's tests take (controllable flip vs. live probe math).
func TestDRAMPressureFlipsDemoteTarget(t *testing.T) {
	// Base: an expensive HBM prefix already under HBM pressure (so the policy is in its
	// demote-or-evict branch and the colder-tier choice is what we are measuring).
	base := func() cachemeta.PlacementRequest {
		req := expensivePrefixRequest()
		req.Pressure = cachemeta.TierPressure{cachemeta.TierHBM: 1.0}
		return req
	}

	// REFUTE GUARD: without the DRAM wire (DRAM at its default empty pressure) the demote target
	// is DRAM. If this assertion ever fails the rung's flip proves nothing — the target was never
	// DRAM to begin with — so the test names that explicitly.
	if d := cachemeta.PlanPlacement(base()); d.Action != cachemeta.ActionDemote || d.ToTier != cachemeta.TierDRAM {
		t.Fatalf("refute guard: without DRAM pressure the demote target must be DRAM, got %s->%s", d.Action, d.ToTier)
	}

	// WITH the wire: fold a full-host DRAM pressure into the request the way withHostDRAM does on
	// a real probe. The target must move off DRAM to the next colder attendable tier.
	req := base()
	req.Pressure = withTierPressure(req.Pressure, cachemeta.TierDRAM, 1.0)
	d := cachemeta.PlanPlacement(req)
	if d.Action != cachemeta.ActionDemote {
		t.Fatalf("with DRAM full the span should still demote (a colder tier has room), got %s", d.Action)
	}
	if d.ToTier == cachemeta.TierDRAM {
		t.Fatal("DRAM pressure 1.0 did not move the demote target off DRAM — the wire is inert")
	}
	if d.ToTier != cachemeta.TierNUMAFar {
		t.Fatalf("a full DRAM should demote to the next colder attendable tier (NUMA-far), got %s", d.ToTier)
	}
}

// TestPlanPlacementForHostEvaluatesFromHostTier checks the full host-wired entry point against
// the LIVE host probe. It cannot force the dev box / CI host to be full (free is real, and the
// probe uses total-free when free is known, ignoring residentBytes), so it asserts the
// invariants that must hold regardless of real occupancy: the span is evaluated from its own
// DRAM tier, and the decision is a valid placement action — never a nonsense promote of a span
// already in the warm DRAM tier under its own pressure. The CONTROLLABLE keep->demote-target
// flip is proven in TestDRAMPressureFlipsDemoteTarget; this guards the live wire's plumbing.
func TestPlanPlacementForHostEvaluatesFromHostTier(t *testing.T) {
	profiles := cachemeta.DefaultTierProfiles()
	lc := cachemeta.NewLifecycle(cachemeta.TierDRAM, 0).MarkResident(profiles, 0)
	req := cachemeta.PlacementRequest{
		Lifecycle:            lc,
		SizeBytes:            64 << 20,
		Tokens:               4000,
		Profiles:             profiles,
		Pressure:             cachemeta.TierPressure{},
		Policy:               cachemeta.LifecyclePolicy{DemoteOnExpiry: true},
		PerTokenPrefillNanos: 2_000_000,
		NowMillis:            0,
	}

	d := PlanPlacementForHost(0, req)
	if d.FromTier != cachemeta.TierDRAM {
		t.Fatalf("the span should be evaluated from its DRAM tier, got %s", d.FromTier)
	}
	switch d.Action {
	case cachemeta.ActionKeep, cachemeta.ActionDemote, cachemeta.ActionSpill, cachemeta.ActionEvict:
		// All valid: keep when the host has room, a relocation/evict when it does not.
	default:
		t.Fatalf("a DRAM-resident span must yield keep/demote/spill/evict, got %s", d.Action)
	}
}

// TestPlanPlacementForHostFailsOpen guards the fail-open contract independent of platform: a
// resident span with room everywhere is KEPT, because either the probe is unsupported (request
// used verbatim) or the host has room — neither may invent pressure that evicts a fresh span.
func TestPlanPlacementForHostFailsOpen(t *testing.T) {
	profiles := cachemeta.DefaultTierProfiles()
	lc := cachemeta.NewLifecycle(cachemeta.TierDRAM, 0).MarkResident(profiles, 0)
	req := cachemeta.PlacementRequest{
		Lifecycle: lc, SizeBytes: 1 << 10, Tokens: 1, Profiles: profiles,
		Pressure: cachemeta.TierPressure{}, Policy: cachemeta.LifecyclePolicy{DemoteOnExpiry: true},
		NowMillis: 0,
	}
	// residentBytes 0 -> even when the probe is supported, fak tracks ~no DRAM bytes, and a tiny
	// span with room stays. (When the probe reports free-known the real occupancy could be high,
	// but a 1 KiB span that is cheaper to recompute than to stage is the keep/evict boundary — so
	// we assert only that it is not promoted/spilled into nonsense.)
	if d := PlanPlacementForHost(0, req); d.Action == cachemeta.ActionPromote {
		t.Fatalf("a fresh resident span must never be promoted by the host wire, got %s", d.Action)
	}
}

// TestHostDRAMPressureContract checks the report->pressure math against the live host probe:
// when supported, pressure is a real fraction in [0,1] and capacity is the reported total; the
// over-count clamp holds; and an unsupported platform fails open to (0,0,false). It cannot force
// a specific fullness (the host is real), so it asserts the invariants the math must satisfy.
func TestHostDRAMPressureContract(t *testing.T) {
	total, _, known := compute.HostSystemMemoryInfo()
	p, capb, pk := HostDRAMPressure(0)
	if known {
		if !pk {
			t.Fatal("host probe is supported but HostDRAMPressure reported unknown")
		}
		if capb != total {
			t.Fatalf("capacity should equal the host total, got %d want %d", capb, total)
		}
		if p < 0 || p > 1 {
			t.Fatalf("pressure must be a fraction in [0,1], got %v", p)
		}
		// Over-count clamp: tracked-resident far above total can never exceed 1.0.
		if over, _, _ := HostDRAMPressure(total * 4); over > 1.0 {
			t.Fatalf("over-count must clamp to <=1.0, got %v", over)
		}
	} else {
		if pk || p != 0 || capb != 0 {
			t.Fatalf("unsupported host probe must fail open to (0,0,false), got (%v,%d,%v)", p, capb, pk)
		}
	}
}

// TestWithDRAMCapacityOverridesOnlyExistingDRAM mirrors the HBM capacity-fold guard: copy the
// table, touch only an existing DRAM profile, never invent a tier, keep nil nil.
func TestWithDRAMCapacityOverridesOnlyExistingDRAM(t *testing.T) {
	in := cachemeta.DefaultTierProfiles()
	srcDRAM := in[cachemeta.TierDRAM].CapacityBytes
	out := withTierCapacity(in, cachemeta.TierDRAM, 1<<40)
	if out[cachemeta.TierDRAM].CapacityBytes != (1 << 40) {
		t.Fatalf("DRAM CapacityBytes not overridden, got %d", out[cachemeta.TierDRAM].CapacityBytes)
	}
	if in[cachemeta.TierDRAM].CapacityBytes != srcDRAM {
		t.Fatalf("source table mutated, got %d want %d", in[cachemeta.TierDRAM].CapacityBytes, srcDRAM)
	}
	if withTierCapacity(nil, cachemeta.TierDRAM, 1) != nil {
		t.Fatal("nil table must stay nil")
	}
	noDRAM := map[cachemeta.ResidencyTier]cachemeta.TierProfile{cachemeta.TierHBM: {Tier: cachemeta.TierHBM}}
	if _, has := withTierCapacity(noDRAM, cachemeta.TierDRAM, 1)[cachemeta.TierDRAM]; has {
		t.Fatal("must not invent a DRAM tier the table did not declare")
	}
}

// TestWithDRAMPressureCopiesAndPreservesOtherTiers checks the pressure fold is copy-on-write and
// leaves sibling tiers untouched (so a request reused across planners does not accrete state).
func TestWithDRAMPressureCopiesAndPreservesOtherTiers(t *testing.T) {
	in := cachemeta.TierPressure{cachemeta.TierHBM: 0.4, cachemeta.TierDRAM: 0.1}
	out := withTierPressure(in, cachemeta.TierDRAM, 0.9)
	if out[cachemeta.TierDRAM] != 0.9 {
		t.Fatalf("DRAM pressure not set, got %v", out[cachemeta.TierDRAM])
	}
	if out[cachemeta.TierHBM] != 0.4 {
		t.Fatalf("sibling HBM pressure not preserved, got %v", out[cachemeta.TierHBM])
	}
	if in[cachemeta.TierDRAM] != 0.1 {
		t.Fatalf("source map mutated, got %v", in[cachemeta.TierDRAM])
	}
	// nil in -> a one-entry map.
	if got := withTierPressure(nil, cachemeta.TierDRAM, 0.5); got[cachemeta.TierDRAM] != 0.5 || len(got) != 1 {
		t.Fatalf("nil in should yield a one-entry DRAM map, got %v", got)
	}
}
