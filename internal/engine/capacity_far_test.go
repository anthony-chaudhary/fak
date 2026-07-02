package engine

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestNUMAFarPressureFlipsDemoteTarget: with DRAM already full (the MLCACHE1 end
// state), the demote target is NUMA-far; folding a full far socket (pressure 1.0) the
// way withNUMAFarPressure does on a real probe moves the target one rung colder, to
// CXL. The flip is exercised through the pure copy-on-write helpers because the live
// probe reads the real box and cannot be forced in a test — the same split every
// capacity rung's tests take (controllable flip vs. live probe math).
func TestNUMAFarPressureFlipsDemoteTarget(t *testing.T) {
	base := func() cachemeta.PlacementRequest {
		req := expensivePrefixRequest()
		req.Pressure = cachemeta.TierPressure{cachemeta.TierHBM: 1.0, cachemeta.TierDRAM: 1.0}
		return req
	}

	// REFUTE GUARD: without the far wire (NUMA-far at its default empty pressure) the
	// demote target is NUMA-far. If this fails the flip below proves nothing.
	if d := cachemeta.PlanPlacement(base()); d.Action != cachemeta.ActionDemote || d.ToTier != cachemeta.TierNUMAFar {
		t.Fatalf("refute guard: with DRAM full and no far pressure the target must be NUMA-far, got %s->%s", d.Action, d.ToTier)
	}

	req := base()
	req.Pressure = withTierPressure(req.Pressure, cachemeta.TierNUMAFar, 1.0)
	d := cachemeta.PlanPlacement(req)
	if d.Action != cachemeta.ActionDemote || d.ToTier != cachemeta.TierCXL {
		t.Fatalf("a full far socket should demote to the next attendable tier (CXL), got %s->%s", d.Action, d.ToTier)
	}
}

// TestCXLPressureFlipsDemoteTarget is the #1470 acceptance: live CXL pressure changes
// WHERE an under-pressure span goes. With every hotter tier full the target is CXL
// (the refute guard — the default pressure 0 kept CXL); folding a full CXL pool
// (pressure 1.0) moves the span past it to disk (a SPILL — disk is not attendable in
// place); and with disk ALSO full nothing colder has room, so the span is evicted.
// The wire is what moves the target each step.
func TestCXLPressureFlipsDemoteTarget(t *testing.T) {
	base := func() cachemeta.PlacementRequest {
		req := expensivePrefixRequest()
		req.Pressure = cachemeta.TierPressure{
			cachemeta.TierHBM:     1.0,
			cachemeta.TierDRAM:    1.0,
			cachemeta.TierNUMAFar: 1.0,
		}
		return req
	}

	// REFUTE GUARD: without the CXL wire (CXL at its default empty pressure) the demote
	// target is CXL. If this ever fails, the flips below prove nothing — the target was
	// never CXL to begin with.
	if d := cachemeta.PlanPlacement(base()); d.Action != cachemeta.ActionDemote || d.ToTier != cachemeta.TierCXL {
		t.Fatalf("refute guard: with all hotter tiers full the demote target must be CXL, got %s->%s", d.Action, d.ToTier)
	}

	// WITH the wire: a full CXL pool moves the span past far memory to disk.
	req := base()
	req.Pressure = withTierPressure(req.Pressure, cachemeta.TierCXL, 1.0)
	d := cachemeta.PlanPlacement(req)
	if d.Action != cachemeta.ActionSpill || d.ToTier != cachemeta.TierDisk {
		t.Fatalf("CXL pressure 1.0 should spill the span to disk, got %s->%s", d.Action, d.ToTier)
	}

	// And when disk is full too, the ladder bottoms out: evict, never a doomed demote
	// into a tier with no room.
	req = base()
	req.Pressure = withTierPressure(req.Pressure, cachemeta.TierCXL, 1.0)
	req.Pressure = withTierPressure(req.Pressure, cachemeta.TierDisk, 1.0)
	d = cachemeta.PlanPlacement(req)
	if d.Action != cachemeta.ActionEvict {
		t.Fatalf("with every colder tier full the span must evict, got %s->%s", d.Action, d.ToTier)
	}
}

// TestNUMAFarPressureContract checks the probe->pressure math against the live box:
// when the topology confirms far memory, pressure is a real fraction in [0,1], the
// capacity is the reported total, and the over-count clamp holds; on a box without
// far memory (every CI runner, any non-linux host) it fails open to (0,0,false) — the
// #1470 fence that an unprobed far tier never fabricates a number.
func TestNUMAFarPressureContract(t *testing.T) {
	farPressureContract(t, "numa_far", compute.NUMAFarMemoryInfo, NUMAFarPressure)
}

// TestCXLPressureContract is the CXL twin of TestNUMAFarPressureContract.
func TestCXLPressureContract(t *testing.T) {
	farPressureContract(t, "cxl", compute.CXLMemoryInfo, CXLPressure)
}

func farPressureContract(t *testing.T, name string, info func() (int64, int64, bool), derive func(int64) (float64, int64, bool)) {
	t.Helper()
	total, _, known := info()
	p, capb, pk := derive(0)
	if known {
		if !pk {
			t.Fatalf("%s: probe confirmed the tier but the pressure wire reported unknown", name)
		}
		if capb != total {
			t.Fatalf("%s: capacity should equal the probed total, got %d want %d", name, capb, total)
		}
		if p < 0 || p > 1 {
			t.Fatalf("%s: pressure must be a fraction in [0,1], got %v", name, p)
		}
		if over, _, _ := derive(total * 4); over > 1.0 {
			t.Fatalf("%s: over-count must clamp to <=1.0, got %v", name, over)
		}
	} else if pk || p != 0 || capb != 0 {
		t.Fatalf("%s: unconfirmed tier must fail open to (0,0,false), got (%v,%d,%v)", name, p, capb, pk)
	}
}

// TestPlanPlacementForFarMemoryFailsOpen guards the fail-open contract independent of
// the host: a fresh resident span with room everywhere is never promoted or otherwise
// moved into nonsense by the far wire — either the box has no far memory (request used
// verbatim) or it does and a fresh span still fits where it is.
func TestPlanPlacementForFarMemoryFailsOpen(t *testing.T) {
	profiles := cachemeta.DefaultTierProfiles()
	lc := cachemeta.NewLifecycle(cachemeta.TierDRAM, 0).MarkResident(profiles, 0)
	req := cachemeta.PlacementRequest{
		Lifecycle: lc, SizeBytes: 1 << 10, Tokens: 1, Profiles: profiles,
		Pressure: cachemeta.TierPressure{}, Policy: cachemeta.LifecyclePolicy{DemoteOnExpiry: true},
		NowMillis: 0,
	}
	if d := PlanPlacementForFarMemory(0, 0, req); d.Action == cachemeta.ActionPromote {
		t.Fatalf("a fresh resident span must never be promoted by the far wire, got %s", d.Action)
	}
	// The folds are copy-on-write: the caller's request must not have accreted state.
	if len(req.Pressure) != 0 {
		t.Fatalf("caller's Pressure map mutated by the far wire: %v", req.Pressure)
	}
}

// TestPlanPlacementForLocalLadderPlumbing drives the full five-tier wire against the
// live host (cpu-ref backend, real probes): it cannot force this box's fullness, so it
// asserts the invariants that must hold regardless — the span is evaluated from its
// own tier and the decision is a valid placement action.
func TestPlanPlacementForLocalLadderPlumbing(t *testing.T) {
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
	d := PlanPlacementForLocalLadder(compute.Default(), 0, 0, t.TempDir(), req)
	if d.FromTier != cachemeta.TierDRAM {
		t.Fatalf("the span should be evaluated from its DRAM tier, got %s", d.FromTier)
	}
	switch d.Action {
	case cachemeta.ActionKeep, cachemeta.ActionDemote, cachemeta.ActionSpill, cachemeta.ActionEvict:
		// All valid: keep when the box has room, a relocation/evict when it does not.
	default:
		t.Fatalf("a DRAM-resident span must yield keep/demote/spill/evict, got %s", d.Action)
	}
}
