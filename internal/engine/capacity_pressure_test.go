package engine

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fakeCapBackend is a compute.Backend that reports a device capacity — the test stand-in for
// a real accelerator (cuda/metal) implementing compute.DeviceCapacity. It embeds the registered
// reference backend for the full Backend surface and overrides only Caps (to advertise the
// probe) and DeviceMemory. With probe=false it models the cpu-ref/wasm floor (known=false).
type fakeCapBackend struct {
	compute.Backend
	total, free int64
	probe       bool
}

func (f fakeCapBackend) Caps() compute.Caps {
	return compute.Caps{Async: true, DeviceMemory: true, CapacityProbe: f.probe}
}
func (f fakeCapBackend) DeviceMemory() (int64, int64, bool) { return f.total, f.free, f.probe }

// expensivePrefixRequest mirrors cmd/hwcachedemo's 4000-token, 64 MB hot prefix: a span the
// cost model deems far cheaper to RETAIN one tier colder than to evict and re-prefill, so the
// under-pressure decision is demote (not evict). Pressure starts EMPTY — the device wire is the
// only thing that can raise HBM pressure and flip the decision.
func expensivePrefixRequest() cachemeta.PlacementRequest {
	profiles := cachemeta.DefaultTierProfiles()
	lc := cachemeta.NewLifecycle(cachemeta.TierHBM, 0).MarkResident(profiles, 0)
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

// TestPlanPlacementForDeviceFlipsKeepToDemote is the issue-#707 acceptance: a high-occupancy
// device raises HBM pressure enough to flip the placement from keep -> demote. The SAME request
// keeps on a near-empty device and falls back to keep on a backend that cannot probe (cpu-ref).
func TestPlanPlacementForDeviceFlipsKeepToDemote(t *testing.T) {
	const total = 24 << 30 // 24 GiB device, free not yet probeable (the cuda-before-#363 case)

	// Sanity: with no device wiring and empty pressure, the policy KEEPS — so any flip below is
	// caused by the device-derived pressure, not the request itself.
	if d := cachemeta.PlanPlacement(expensivePrefixRequest()); d.Action != cachemeta.ActionKeep {
		t.Fatalf("baseline (no device pressure) must keep, got %s", d.Action)
	}

	// High occupancy: tracked-resident bytes fill the device -> HBM pressure 1.0 -> demote.
	full := fakeCapBackend{Backend: compute.Default(), total: total, free: compute.FreeUnknown, probe: true}
	if d := PlanPlacementForDevice(full, total, expensivePrefixRequest()); d.Action != cachemeta.ActionDemote {
		t.Fatalf("a full device must flip keep->demote, got %s (to %s)", d.Action, d.ToTier)
	} else if d.FromTier != cachemeta.TierHBM || d.ToTier != cachemeta.TierDRAM {
		t.Fatalf("demote should relocate HBM->DRAM, got %s->%s", d.FromTier, d.ToTier)
	}

	// Low occupancy: a near-empty device leaves HBM with room -> keep.
	if d := PlanPlacementForDevice(full, 1<<20, expensivePrefixRequest()); d.Action != cachemeta.ActionKeep {
		t.Fatalf("a near-empty device must keep, got %s", d.Action)
	}

	// cpu-ref (cannot probe): falls back to the profile default (empty pressure) -> keep,
	// unchanged from a path that never knew about the device. Test both a non-probing fake and
	// the real reference floor.
	cpuRef := fakeCapBackend{Backend: compute.Default(), total: total, free: total, probe: false}
	if d := PlanPlacementForDevice(cpuRef, total, expensivePrefixRequest()); d.Action != cachemeta.ActionKeep {
		t.Fatalf("a non-probing backend must fall back to keep, got %s", d.Action)
	}
	if d := PlanPlacementForDevice(compute.Default(), total, expensivePrefixRequest()); d.Action != cachemeta.ActionKeep {
		t.Fatalf("the cpu-ref floor must fall back to keep, got %s", d.Action)
	}
}

// TestDeviceHBMPressure checks the report->pressure math directly across the free-known,
// free-unknown, clamp, and unknown-capacity cases.
func TestDeviceHBMPressure(t *testing.T) {
	const total = 24 << 30

	// free KNOWN: used = total - free; pressure = used/total, capacity = total.
	dev := fakeCapBackend{Backend: compute.Default(), total: total, free: 6 << 30, probe: true}
	if p, cap, known := DeviceHBMPressure(dev, 0); !known || cap != total || p < 0.749 || p > 0.751 {
		t.Fatalf("free-known: want known, cap=%d, p≈0.75; got known=%v cap=%d p=%v", total, known, cap, p)
	}

	// free FreeUnknown: pressure derives from total vs tracked-resident bytes.
	un := fakeCapBackend{Backend: compute.Default(), total: total, free: compute.FreeUnknown, probe: true}
	if p, _, known := DeviceHBMPressure(un, 12<<30); !known || p < 0.499 || p > 0.501 {
		t.Fatalf("free-unknown: want known p≈0.5, got known=%v p=%v", known, p)
	}
	// A stale over-count clamps to a full (1.0) device, never past it.
	if p, _, _ := DeviceHBMPressure(un, total*2); p != 1.0 {
		t.Fatalf("over-count must clamp to 1.0, got %v", p)
	}

	// unknown capacity: fail open -> known=false, no pressure/capacity to apply.
	off := fakeCapBackend{Backend: compute.Default(), total: total, free: 0, probe: false}
	if p, cap, known := DeviceHBMPressure(off, total); known || p != 0 || cap != 0 {
		t.Fatalf("non-probing backend must report unknown (0,0,false), got (%v,%d,%v)", p, cap, known)
	}
}

// TestPlanPlacementForDeviceDoesNotMutateRequest guards the copy-on-write contract: a request
// value reused across backends must not accrete the HBM override from a prior call.
func TestPlanPlacementForDeviceDoesNotMutateRequest(t *testing.T) {
	req := expensivePrefixRequest()
	req.Pressure[cachemeta.TierDRAM] = 0.25 // a caller-set value that must survive untouched
	full := fakeCapBackend{Backend: compute.Default(), total: 24 << 30, free: compute.FreeUnknown, probe: true}

	_ = PlanPlacementForDevice(full, 24<<30, req)

	if _, set := req.Pressure[cachemeta.TierHBM]; set {
		t.Fatal("PlanPlacementForDevice mutated the caller's Pressure map (HBM leaked in)")
	}
	if req.Pressure[cachemeta.TierDRAM] != 0.25 {
		t.Fatalf("caller's DRAM pressure was altered, got %v", req.Pressure[cachemeta.TierDRAM])
	}
	if got := req.Profiles[cachemeta.TierHBM].CapacityBytes; got != (80 << 30) {
		t.Fatalf("caller's HBM CapacityBytes was overridden in place, got %d", got)
	}
}

// TestWithHBMCapacityOverridesOnlyExistingHBM checks the capacity fold copies the table and
// touches only an existing HBM profile (it never invents a tier the box did not declare).
func TestWithHBMCapacityOverridesOnlyExistingHBM(t *testing.T) {
	in := cachemeta.DefaultTierProfiles()
	out := withHBMCapacity(in, 24<<30)
	if out[cachemeta.TierHBM].CapacityBytes != (24 << 30) {
		t.Fatalf("HBM CapacityBytes not overridden, got %d", out[cachemeta.TierHBM].CapacityBytes)
	}
	if in[cachemeta.TierHBM].CapacityBytes != (80 << 30) {
		t.Fatalf("source table mutated, got %d", in[cachemeta.TierHBM].CapacityBytes)
	}
	if withHBMCapacity(nil, 1) != nil {
		t.Fatal("nil table must stay nil")
	}
	noHBM := map[cachemeta.ResidencyTier]cachemeta.TierProfile{cachemeta.TierDRAM: {Tier: cachemeta.TierDRAM}}
	if _, has := withHBMCapacity(noHBM, 1)[cachemeta.TierHBM]; has {
		t.Fatal("must not invent an HBM tier the table did not declare")
	}
}
