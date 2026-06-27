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
	return pressureFromProbe(total, free, residentBytes, ok)
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
	req = withDeviceHBM(b, hbmResidentBytes, req)
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
	req.Pressure = withTierPressure(req.Pressure, cachemeta.TierDRAM, pressure)
	req.Profiles = withTierCapacity(req.Profiles, cachemeta.TierDRAM, capacity)
	return req
}
