package engine

// capacity_pressure.go — Plank 3 of the hardware-capacity bridge (issue #707): the
// REPORT -> POLICY wire. It derives a live cachemeta.TierPressure (and the HBM tier's
// real CapacityBytes) from the active backend's actual device memory, so
// cachemeta.PlanPlacement plans demote-not-evict against the device that EXISTS rather
// than against DefaultTierProfiles' "representative order-of-magnitude defaults" and the
// hand-injected pressure that only cmd/hwcachedemo ever supplied.
//
// The two planes the explainer (docs/explainers/hardware-limits-and-capacity.md §2) draws
// met only at the meter: compute.DeviceMemoryInfo (Plank 1) lets a backend REPORT its
// ceiling, and CapacityAdapter (Plank 4) EXECUTES a placement directive — but nothing
// turned the report into the PRESSURE the policy plans against. This file is that missing
// arrow: compute.DeviceMemoryInfo(b) -> cachemeta.TierPressure[HBM].
//
// Fail open, the same contract every other capacity plank honors. When the backend cannot
// probe its memory (the pure-Go cpu-ref floor, a wasm target — DeviceMemoryInfo reports
// known=false) this derives NO pressure and overrides NO capacity: the caller's profile
// default stands, so PlanPlacement decides exactly as it did before the wire existed and no
// path that works today regresses.

import (
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// DeviceHBMPressure derives the HBM tier's live fullness and capacity for backend b from its
// real device memory — the Plank-3 report->policy conversion, expressed as plain math over
// compute.DeviceMemoryInfo so it is trivially testable.
//
//	known == false : b cannot probe its capacity (cpu-ref, wasm, any non-DeviceCapacity
//	                 backend). pressure/capacityBytes are 0 and MUST be ignored — the caller
//	                 falls back to the profile default (the fail-open contract).
//	known == true  : total > 0 is the device ceiling (capacityBytes). pressure is the USED
//	                 fraction in [0,1]:
//	                   - free known        -> used = total - free  (every consumer of the
//	                                          device counts, not only fak's tensors)
//	                   - free FreeUnknown   -> used = residentBytes (the bytes fak tracks
//	                                          resident in HBM). This is the cuda producer's
//	                                          case until cudaMemGetInfo is wired (#363
//	                                          follow-up): total is known, free is not, so
//	                                          pressure derives from total vs tracked-resident.
//
// residentBytes is clamped into [0,total] so a stale or racy over-count can never push
// pressure outside [0,1] and trip the placement math into a nonsense decision.
func DeviceHBMPressure(b compute.Backend, residentBytes int64) (pressure float64, capacityBytes int64, known bool) {
	total, free, ok := compute.DeviceMemoryInfo(b)
	return pressureFromProbe(total, free, residentBytes, ok)
}

// PlanPlacementForDevice plans a cachemeta placement against the device that ACTUALLY exists:
// it derives live HBM pressure + capacity for backend b (DeviceHBMPressure), folds them into
// req, and calls cachemeta.PlanPlacement. This is the wired path Plank 3 names — the live
// signal nothing on the serving path computed before; until now only cmd/hwcachedemo injected
// pressure, by hand.
//
// Fail open: when b cannot probe its capacity (cpu-ref, wasm; known=false) req is used
// VERBATIM, so a path that worked before is unchanged. residentBytes is fak's tracked-resident
// HBM byte count — the pressure basis while the cuda producer reports total but not free
// (#363). req is not mutated: its Pressure and Profiles maps are copied before the HBM
// override is applied, so a shared request value handed to several backends stays clean.
func PlanPlacementForDevice(b compute.Backend, residentBytes int64, req cachemeta.PlacementRequest) cachemeta.PlacementDecision {
	return cachemeta.PlanPlacement(withDeviceHBM(b, residentBytes, req))
}
