package engine

// capacity_tier.go — the shared, tier-parameterized core the per-tier capacity wires
// (capacity_pressure.go for HBM, capacity_dram.go for DRAM, capacity_disk.go for disk)
// all build on. Each wire differs only in WHERE it probes its fullness from; the
// copy-on-write fold of a derived pressure/capacity into a placement request, and the
// used->pressure clamp, are identical across tiers and live here once.

import (
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// pressureFromUsedTotal returns the USED fraction of total in [0,1], clamping used into
// [0,total] first so a stale or racy over-count can never push pressure outside [0,1] and
// trip the placement math into a nonsense decision. total is assumed > 0 (the caller checks
// the probe succeeded before calling). Shared by every per-tier *Pressure derivation.
func pressureFromUsedTotal(used, total int64) float64 {
	if used < 0 {
		used = 0
	}
	if used > total {
		used = total
	}
	return float64(used) / float64(total)
}

// pressureFromProbe converts a (total, free, ok) memory probe plus fak's tracked-resident
// byte count into the (pressure, capacity, known) triple the per-tier *Pressure derivations
// return. ok=false or a non-positive total is the fail-open case (known=false; the caller
// falls back to the profile default). When free is known (>= 0, not the FreeUnknown sentinel)
// used = total-free so every consumer of the resource counts; when free is unknown used falls
// back to residentBytes. Shared by HostDRAMPressure and DeviceHBMPressure, whose only real
// difference is WHICH probe fills total/free.
func pressureFromProbe(total, free, residentBytes int64, ok bool) (pressure float64, capacityBytes int64, known bool) {
	if !ok || total <= 0 {
		return 0, 0, false
	}
	used := residentBytes
	if free >= 0 { // free is known (not the FreeUnknown sentinel)
		used = total - free
	}
	return pressureFromUsedTotal(used, total), total, true
}

// withTierPressure returns a COPY of in with tier's pressure set to p. A nil in yields a
// one-entry map; every other tier is carried over unchanged. Copying keeps a caller's
// request value reusable across backends/planners. Shared by the per-tier withHBM/withDRAM/
// withDisk pressure folds.
func withTierPressure(in cachemeta.TierPressure, tier cachemeta.ResidencyTier, p float64) cachemeta.TierPressure {
	out := cachemeta.TierPressure{tier: p}
	for t, v := range in {
		if t == tier {
			continue
		}
		out[t] = v
	}
	return out
}

// withTierCapacity returns a COPY of in with tier's CapacityBytes overridden by capacity.
// A nil in stays nil (nothing to override); a tier table without an entry for tier is copied
// unchanged (only an existing profile is updated — these wires report a real ceiling, they
// do not invent a tier the box did not declare). Shared by the per-tier capacity folds.
func withTierCapacity(in map[cachemeta.ResidencyTier]cachemeta.TierProfile, tier cachemeta.ResidencyTier, capacity int64) map[cachemeta.ResidencyTier]cachemeta.TierProfile {
	if in == nil {
		return nil
	}
	out := make(map[cachemeta.ResidencyTier]cachemeta.TierProfile, len(in))
	for t, prof := range in {
		out[t] = prof
	}
	if prof, ok := out[tier]; ok {
		prof.CapacityBytes = capacity
		out[tier] = prof
	}
	return out
}

// withDeviceHBM folds live device-HBM pressure + capacity into a COPY of req when backend b
// can probe its memory; on a backend that cannot (cpu-ref, wasm; known=false) it returns req
// unchanged. Factored so PlanPlacementForDevice and the multi-tier device planners share one
// fail-open HBM override path — the device sibling of withHostDRAM and withDiskPressure.
func withDeviceHBM(b compute.Backend, residentBytes int64, req cachemeta.PlacementRequest) cachemeta.PlacementRequest {
	if pressure, capacity, known := DeviceHBMPressure(b, residentBytes); known {
		req.Pressure = withTierPressure(req.Pressure, cachemeta.TierHBM, pressure)
		req.Profiles = withTierCapacity(req.Profiles, cachemeta.TierHBM, capacity)
	}
	return req
}
