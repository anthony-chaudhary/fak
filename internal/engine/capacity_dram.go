package engine

// capacity_dram.go — rung 1 of the multilevel-default-cache epic
// (docs/serving/multilevel-default-cache-epic.md, MLCACHE1): the L2/DRAM analogue of the
// HBM report->policy wire in capacity_pressure.go.
//
// After #707 the placement policy planned the HBM tier against the device that ACTUALLY
// exists (DeviceHBMPressure -> cachemeta.TierPressure[HBM]) but every colder local tier —
// DRAM first among them — still planned against cachemeta.DefaultTierProfiles' representative
// numbers and an assumed-empty pressure of 0. So the demote ladder was hardware-aware at its
// top rung and BLIND one step below it: a span the policy decided to demote INTO host DRAM
// would be staged there even on a box whose RAM is already full, where the honest move is to
// skip DRAM for a colder tier that has room.
//
// This file closes that asymmetry for L2. It derives the DRAM tier's live fullness from the
// host's real physical memory (compute.HostSystemMemoryInfo — a backend-FREE probe, so it
// works on the pure-Go cpu-ref floor that has no device at all, exactly where the HBM wire
// reports known=false and contributes nothing) and folds it into a placement request the same
// copy-on-write way withHBMPressure does, leaving the caller's request value reusable.
//
// Fail open, the contract every capacity plank honors: when the host memory probe is
// unsupported (HostSystemMemoryInfo reports known=false on an unhandled platform) this derives
// NO pressure and overrides NO capacity, so PlanPlacement decides exactly as it did before the
// wire existed and no path that works today regresses.

import (
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// HostDRAMPressure derives the DRAM tier's live fullness and capacity from the process host's
// real physical memory — the L2 analogue of DeviceHBMPressure, expressed as plain math over
// compute.HostSystemMemoryInfo so it is trivially testable and needs no backend.
//
//	known == false : the platform's host-memory probe is unsupported. pressure/capacityBytes
//	                 are 0 and MUST be ignored — the caller falls back to the profile default
//	                 (the fail-open contract).
//	known == true  : total > 0 is the host RAM ceiling (capacityBytes). pressure is the USED
//	                 fraction in [0,1]:
//	                   - free known       -> used = total - free  (every consumer of host RAM
//	                                         counts, not only fak's resident bytes)
//	                   - free FreeUnknown -> used = residentBytes  (the bytes fak tracks
//	                                         resident in DRAM, when the probe gives total but
//	                                         not free)
//
// residentBytes is clamped into [0,total] so a stale or racy over-count can never push pressure
// outside [0,1] and trip the placement math into a nonsense decision — identical to the HBM
// wire's clamp.
func HostDRAMPressure(residentBytes int64) (pressure float64, capacityBytes int64, known bool) {
	total, free, ok := compute.HostSystemMemoryInfo()
	if !ok || total <= 0 {
		return 0, 0, false
	}
	used := residentBytes
	if free >= 0 { // free is known (not the FreeUnknown sentinel)
		used = total - free
	}
	if used < 0 {
		used = 0
	}
	if used > total {
		used = total
	}
	return float64(used) / float64(total), total, true
}

// PlanPlacementForHost plans a cachemeta placement against the HOST that actually exists: it
// derives live DRAM pressure + capacity (HostDRAMPressure), folds them into req, and calls
// cachemeta.PlanPlacement. It is the L2 sibling of PlanPlacementForDevice — use it when the
// demote target under consideration is host DRAM and the box has no device backend to probe
// (the cpu-ref/host-only serve path).
//
// Fail open: when the host memory probe is unsupported (known=false) req is used VERBATIM, so a
// path that worked before is unchanged. req is not mutated: its Pressure and Profiles maps are
// copied before the DRAM override is applied, so a shared request value handed to several
// planners stays clean.
func PlanPlacementForHost(residentBytes int64, req cachemeta.PlacementRequest) cachemeta.PlacementDecision {
	req = withHostDRAM(residentBytes, req)
	return cachemeta.PlanPlacement(req)
}

// PlanPlacementForDeviceAndHost plans against BOTH live top tiers at once: HBM pressure from
// the device backend (when present) and DRAM pressure from the host probe. This is the wire a
// served decode loop on a GPU box wants — a span under HBM pressure may demote to DRAM, and
// that DRAM target must itself be planned against real host fullness, or the demote lands in a
// tier that is already full. Each override is independently fail-open: a missing device leaves
// HBM at its profile default, an unsupported host probe leaves DRAM at its default.
func PlanPlacementForDeviceAndHost(b compute.Backend, hbmResidentBytes, dramResidentBytes int64, req cachemeta.PlacementRequest) cachemeta.PlacementDecision {
	if pressure, capacity, known := DeviceHBMPressure(b, hbmResidentBytes); known {
		req.Pressure = withHBMPressure(req.Pressure, pressure)
		req.Profiles = withHBMCapacity(req.Profiles, capacity)
	}
	req = withHostDRAM(dramResidentBytes, req)
	return cachemeta.PlanPlacement(req)
}

// withHostDRAM folds live host-DRAM pressure + capacity into a COPY of req when the host probe
// succeeds; on an unsupported probe it returns req unchanged. Factored so PlanPlacementForHost
// and PlanPlacementForDeviceAndHost share one fail-open override path.
func withHostDRAM(residentBytes int64, req cachemeta.PlacementRequest) cachemeta.PlacementRequest {
	pressure, capacity, known := HostDRAMPressure(residentBytes)
	if !known {
		return req
	}
	req.Pressure = withDRAMPressure(req.Pressure, pressure)
	req.Profiles = withDRAMCapacity(req.Profiles, capacity)
	return req
}

// withDRAMPressure returns a COPY of in with the DRAM tier's pressure set to p. A nil in yields
// a one-entry map; every non-DRAM tier is carried over unchanged. Copying keeps a caller's
// request value reusable across planners — the same posture as withHBMPressure.
func withDRAMPressure(in cachemeta.TierPressure, p float64) cachemeta.TierPressure {
	out := cachemeta.TierPressure{cachemeta.TierDRAM: p}
	for t, v := range in {
		if t == cachemeta.TierDRAM {
			continue
		}
		out[t] = v
	}
	return out
}

// withDRAMCapacity returns a COPY of in with the DRAM tier's CapacityBytes overridden by the
// host's real total. A nil in stays nil (nothing to override); a tier table without a DRAM
// entry is copied unchanged (only an existing DRAM profile is updated — this wire reports the
// host ceiling, it does not invent a tier the box did not declare). Mirrors withHBMCapacity.
func withDRAMCapacity(in map[cachemeta.ResidencyTier]cachemeta.TierProfile, capacity int64) map[cachemeta.ResidencyTier]cachemeta.TierProfile {
	if in == nil {
		return nil
	}
	out := make(map[cachemeta.ResidencyTier]cachemeta.TierProfile, len(in))
	for t, prof := range in {
		out[t] = prof
	}
	if dram, ok := out[cachemeta.TierDRAM]; ok {
		dram.CapacityBytes = capacity
		out[cachemeta.TierDRAM] = dram
	}
	return out
}
