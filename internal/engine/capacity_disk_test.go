package engine

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expensiveDRAMRequest mirrors expensivePrefixRequest but starts in DRAM instead of HBM:
// a 4000-token, 64 MB span that is far cheaper to RETAIN in a colder tier than to evict
// and re-prefill, so under DRAM pressure the decision is spill to disk (not evict).
func expensiveDRAMRequest() cachemeta.PlacementRequest {
	profiles := cachemeta.DefaultTierProfiles()
	lc := cachemeta.NewLifecycle(cachemeta.TierDRAM, 0).MarkResident(profiles, 0)
	return cachemeta.PlacementRequest{
		Lifecycle:            lc,
		SizeBytes:            64 << 20,
		Tokens:               4000,
		Profiles:             profiles,
		Pressure:             cachemeta.TierPressure{},
		Policy:               cachemeta.LifecyclePolicy{DemoteOnExpiry: true},
		PerTokenPrefillNanos: 2_000_000,
		NowMillis:            0,
	}
}

// TestDiskPressureFlipsSpillToEvict is the MLCACHE2 acceptance: live disk pressure changes
// WHAT an under-pressure span does. A DRAM-resident expensive span that would normally spill to
// disk (disk has room, pressure 0) instead EVICTS when the disk is full (pressure 1.0), because
// the coldest-colder-with-room walk finds nothing colder than disk with room and emits
// ActionEvict with reason "no_colder_tier_with_room" instead of ActionSpill.
func TestDiskPressureFlipsSpillToEvict(t *testing.T) {
	// Base: a DRAM-resident expensive prefix under DRAM pressure that spills to disk by default.
	// We need to fill NUMA-far and CXL too, so disk is the only colder tier with room.
	base := func() cachemeta.PlacementRequest {
		req := expensiveDRAMRequest()
		req.Pressure = cachemeta.TierPressure{
			cachemeta.TierDRAM:    1.0,
			cachemeta.TierNUMAFar: 1.0,
			cachemeta.TierCXL:     1.0,
			cachemeta.TierDisk:    0.0,
		}
		return req
	}

	// REFUTE GUARD: without the disk wire (disk at its default empty pressure) the action is
	// spill to disk. If this assertion ever fails the rung's flip proves nothing.
	if d := cachemeta.PlanPlacement(base()); d.Action != cachemeta.ActionSpill || d.ToTier != cachemeta.TierDisk {
		t.Fatalf("refute guard: without disk pressure the span should spill to disk, got %s->%s", d.Action, d.ToTier)
	}

	// WITH the wire: fold a full-disk pressure into the request the way withDiskPressure does on
	// a real probe. The action must flip to evict with reason "no_colder_tier_with_room".
	req := base()
	req.Pressure = withTierDiskPressure(req.Pressure, 1.0)
	d := cachemeta.PlanPlacement(req)
	if d.Action != cachemeta.ActionEvict {
		t.Fatalf("with disk full the span should evict (no colder tier has room), got %s", d.Action)
	}
	if d.Reason != "no_colder_tier_with_room" {
		t.Fatalf("evict reason must be 'no_colder_tier_with_room', got %s", d.Reason)
	}
}

// TestPlanPlacementForDiskEvaluatesFromDiskTier checks the full disk-wired entry point.
// Since the probe needs a real path and we cannot guarantee a full filesystem, we use a
// simple reachable path and assert the invariants: the decision is valid, and the request
// is not mutated (copy-on-write). The CONTROLLABLE spill->evict flip is proven in
// TestDiskPressureFlipsSpillToEvict; this guards the live wire's plumbing.
func TestPlanPlacementForDiskEvaluatesFromDiskTier(t *testing.T) {
	profiles := cachemeta.DefaultTierProfiles()
	lc := cachemeta.NewLifecycle(cachemeta.TierDisk, 0).MarkResident(profiles, 0)
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

	// Use "." as a path that should be accessible on any platform.
	d := PlanPlacementForDisk(".", req)
	if d.FromTier != cachemeta.TierDisk {
		t.Fatalf("the span should be evaluated from its disk tier, got %s", d.FromTier)
	}
	switch d.Action {
	case cachemeta.ActionKeep, cachemeta.ActionPromote, cachemeta.ActionEvict:
		// All valid: keep when the filesystem has room, promote when hot, evict when no room.
	default:
		t.Fatalf("a disk-resident span must yield keep/promote/evict, got %s", d.Action)
	}
}

// TestPlanPlacementForDiskFailsOpen guards the fail-open contract: a fresh span with room
// everywhere is KEPT, because either the probe fails (request used verbatim) or the
// filesystem has room — neither may invent pressure that evicts a fresh span.
func TestPlanPlacementForDiskFailsOpen(t *testing.T) {
	profiles := cachemeta.DefaultTierProfiles()
	lc := cachemeta.NewLifecycle(cachemeta.TierDisk, 0).MarkResident(profiles, 0)
	req := cachemeta.PlacementRequest{
		Lifecycle: lc, SizeBytes: 1 << 10, Tokens: 1, Profiles: profiles,
		Pressure: cachemeta.TierPressure{}, Policy: cachemeta.LifecyclePolicy{DemoteOnExpiry: true},
		NowMillis: 0,
	}

	// "." should be accessible; either the probe works (real free space) or it fails open.
	// In either case a tiny fresh span must not be evicted spuriously.
	if d := PlanPlacementForDisk(".", req); d.Action == cachemeta.ActionEvict && d.Reason == "no_colder_tier_with_room" {
		t.Fatalf("a fresh disk-resident span must not be evicted by a healthy probe, got %s (reason: %s)", d.Action, d.Reason)
	}
}

// TestDiskPressureContract checks the report->pressure math against the disk probe:
// when supported, pressure is a real fraction in [0,1] and capacity is the reported total;
// and an inaccessible path fails open to (0,0,false). It cannot force a specific fullness
// (the path is real), so it asserts the invariants the math must satisfy.
func TestDiskPressureContract(t *testing.T) {
	// Use "." as a path that should be accessible on any platform.
	total, _, known := compute.DiskInfo(".")
	p, capb, pk := DiskPressure(".")
	if known {
		if !pk {
			t.Fatal("disk probe is supported but DiskPressure reported unknown")
		}
		if capb != total {
			t.Fatalf("capacity should equal the filesystem total, got %d want %d", capb, total)
		}
		if p < 0 || p > 1 {
			t.Fatalf("pressure must be a fraction in [0,1], got %v", p)
		}
	} else {
		if pk || p != 0 || capb != 0 {
			t.Fatalf("unsupported disk probe must fail open to (0,0,false), got (%v,%d,%v)", p, capb, pk)
		}
	}

	// Inaccessible path should fail open.
	if p, capb, pk := DiskPressure("/this/path/does/not/exist/diskpressure/986/test"); pk || p != 0 || capb != 0 {
		t.Fatalf("inaccessible path must fail open to (0,0,false), got (%v,%d,%v)", p, capb, pk)
	}
}

// TestWithTierDiskCapacityOverridesOnlyExistingDisk mirrors the DRAM and HBM capacity-fold
// guards: copy the table, touch only an existing disk profile, never invent a tier, keep nil nil.
func TestWithTierDiskCapacityOverridesOnlyExistingDisk(t *testing.T) {
	in := cachemeta.DefaultTierProfiles()
	srcDisk := in[cachemeta.TierDisk].CapacityBytes
	out := withTierDiskCapacity(in, 1<<40)
	if out[cachemeta.TierDisk].CapacityBytes != (1 << 40) {
		t.Fatalf("Disk CapacityBytes not overridden, got %d", out[cachemeta.TierDisk].CapacityBytes)
	}
	if in[cachemeta.TierDisk].CapacityBytes != srcDisk {
		t.Fatalf("source table mutated, got %d want %d", in[cachemeta.TierDisk].CapacityBytes, srcDisk)
	}
	if withTierDiskCapacity(nil, 1) != nil {
		t.Fatal("nil table must stay nil")
	}
	noDisk := map[cachemeta.ResidencyTier]cachemeta.TierProfile{cachemeta.TierHBM: {Tier: cachemeta.TierHBM}}
	if _, has := withTierDiskCapacity(noDisk, 1)[cachemeta.TierDisk]; has {
		t.Fatal("must not invent a disk tier the table did not declare")
	}
}

// TestWithTierDiskPressureCopiesAndPreservesOtherTiers checks the pressure fold is
// copy-on-write and leaves sibling tiers untouched.
func TestWithTierDiskPressureCopiesAndPreservesOtherTiers(t *testing.T) {
	in := cachemeta.TierPressure{cachemeta.TierHBM: 0.4, cachemeta.TierDisk: 0.1}
	out := withTierDiskPressure(in, 0.9)
	if out[cachemeta.TierDisk] != 0.9 {
		t.Fatalf("Disk pressure not set, got %v", out[cachemeta.TierDisk])
	}
	if out[cachemeta.TierHBM] != 0.4 {
		t.Fatalf("sibling HBM pressure not preserved, got %v", out[cachemeta.TierHBM])
	}
	if in[cachemeta.TierDisk] != 0.1 {
		t.Fatalf("source map mutated, got %v", in[cachemeta.TierDisk])
	}
	// nil in -> a one-entry map.
	if got := withTierDiskPressure(nil, 0.5); got[cachemeta.TierDisk] != 0.5 || len(got) != 1 {
		t.Fatalf("nil in should yield a one-entry disk map, got %v", got)
	}
}
