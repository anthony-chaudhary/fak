package engine

// capacity_far.go — the NUMA-far + CXL rung of the multilevel-default-cache spine
// (#1470, phase-2 child P2-4 of epic #1463): the far-memory analogue of the HBM wire
// (capacity_pressure.go, #707), the DRAM wire (capacity_dram.go, MLCACHE1), and the
// disk wire (capacity_disk.go, MLCACHE2).
//
// After MLCACHE1/2 the demote ladder was planned against the box that actually exists
// at its top (HBM), its second rung (DRAM), and its bottom (disk) — but the two
// byte-addressable far tiers BETWEEN them, NUMA-far and CXL, the attendable-in-place
// demote targets that make "don't evict, relocate" pay
// (internal/cachemeta/hardware.go), still planned against the assumed-empty
// placeholder pressure of 0. So coldestColderWithRoom always believed far memory had
// room, and a demote ladder could pile spans into a far-memory pool that is actually
// full — the exact trap the spine's honest-boundary names ("a rung must not claim
// hardware-aware for a tier whose pressure is still the 0 placeholder").
//
// This file closes that asymmetry for the far tiers. It derives their live fullness
// from the kernel's NUMA topology (compute.NUMAFarMemoryInfo for the other socket's
// DRAM, compute.CXLMemoryInfo for memory-only expansion nodes — backend-FREE probes,
// so they work on the cpu-ref floor) and folds them into a placement request the same
// copy-on-write way withHostDRAM and withDiskPressure do.
//
// Fail open, the contract every capacity plank honors — with one wrinkle the hotter
// tiers do not have: DRAM and disk always EXIST, so their probes only ever refine a
// tier already in the ladder, while far memory is usually ABSENT. A box with no far
// node reports known=false, no pressure is derived, no capacity is overridden, and
// PlanPlacement decides exactly as it did before the wire existed (the #1470 fence:
// never fabricate a pressure for a tier the box cannot prove).

import (
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// NUMAFarPressure derives the NUMA-far tier's live fullness and capacity from the far
// sockets' real memory — the far-memory analogue of HostDRAMPressure, expressed as
// plain math over compute.NUMAFarMemoryInfo so it is trivially testable and needs no
// backend.
//
//	known == false : the box has no far NUMA node (single socket), the topology cannot
//	                 be read, or the platform has no probe. pressure/capacityBytes are
//	                 0 and MUST be ignored — the caller falls back to the profile
//	                 default (the fail-open contract).
//	known == true  : total > 0 is the far sockets' memory ceiling (capacityBytes).
//	                 pressure is the USED fraction in [0,1], with residentBytes as the
//	                 fallback when free is unreported — the same shape as the DRAM wire.
func NUMAFarPressure(residentBytes int64) (pressure float64, capacityBytes int64, known bool) {
	total, free, ok := compute.NUMAFarMemoryInfo()
	return pressureFromProbe(total, free, residentBytes, ok)
}

// CXLPressure derives the CXL tier's live fullness and capacity from the box's
// CPU-less expansion memory (memory-only NUMA nodes, CXL.mem the canonical instance)
// — the CXL analogue of NUMAFarPressure, with the identical fail-open contract: a box
// with no expansion node reports known=false and the caller keeps today's behavior.
func CXLPressure(residentBytes int64) (pressure float64, capacityBytes int64, known bool) {
	total, free, ok := compute.CXLMemoryInfo()
	return pressureFromProbe(total, free, residentBytes, ok)
}

// withNUMAFarPressure folds live NUMA-far pressure + capacity into a COPY of req when
// the far-memory probe confirms the tier; on an unconfirmed tier it returns req
// unchanged. Factored so the pattern stays identical to the HBM/DRAM/disk wires. Note
// withTierCapacity only refines a tier already in req.Profiles — the fold never
// inserts a NUMA-far tier into a ladder that does not carry one (that is
// cachemeta.ProbedTierProfiles' job, from the same probe).
func withNUMAFarPressure(residentBytes int64, req cachemeta.PlacementRequest) cachemeta.PlacementRequest {
	pressure, capacity, known := NUMAFarPressure(residentBytes)
	if !known {
		return req
	}
	req.Pressure = withTierPressure(req.Pressure, cachemeta.TierNUMAFar, pressure)
	req.Profiles = withTierCapacity(req.Profiles, cachemeta.TierNUMAFar, capacity)
	return req
}

// withCXLPressure folds live CXL pressure + capacity into a COPY of req when the
// expansion-memory probe confirms the tier; on an unconfirmed tier it returns req
// unchanged.
func withCXLPressure(residentBytes int64, req cachemeta.PlacementRequest) cachemeta.PlacementRequest {
	pressure, capacity, known := CXLPressure(residentBytes)
	if !known {
		return req
	}
	req.Pressure = withTierPressure(req.Pressure, cachemeta.TierCXL, pressure)
	req.Profiles = withTierCapacity(req.Profiles, cachemeta.TierCXL, capacity)
	return req
}

// PlanPlacementForFarMemory plans a cachemeta placement against the far memory that
// actually exists: it derives live NUMA-far and CXL pressure + capacity, folds them
// into req, and calls cachemeta.PlanPlacement — the far-memory sibling of
// PlanPlacementForHost and PlanPlacementForDisk. Each fold is independently fail-open
// (a box with a far socket but no expansion node refines only NUMA-far), and req is
// not mutated.
func PlanPlacementForFarMemory(numaResidentBytes, cxlResidentBytes int64, req cachemeta.PlacementRequest) cachemeta.PlacementDecision {
	req = withNUMAFarPressure(numaResidentBytes, req)
	req = withCXLPressure(cxlResidentBytes, req)
	return cachemeta.PlanPlacement(req)
}

// PlanPlacementForLocalLadder plans against EVERY live local tier at once — HBM from
// the device backend, DRAM from the host probe, NUMA-far + CXL from the NUMA topology,
// and disk from the filesystem probe: the whole in-box relocation ladder
// (cachemeta.NextColderTier's HBM->DRAM->NUMA-far->CXL->Disk walk), each rung planned
// against real fullness. This supersedes PlanPlacementForDeviceHostAndDisk as the full
// wire a served decode loop wants once far tiers can be live too; the far folds pass
// residentBytes 0 because fak keeps no far-resident counter yet and the topology probe
// always reports free when it reports at all, so the fallback never engages. Every
// override remains independently fail-open.
func PlanPlacementForLocalLadder(b compute.Backend, hbmResidentBytes, dramResidentBytes int64, diskPath string, req cachemeta.PlacementRequest) cachemeta.PlacementDecision {
	req = withDeviceHBM(b, hbmResidentBytes, req)
	req = withHostDRAM(dramResidentBytes, req)
	req = withNUMAFarPressure(0, req)
	req = withCXLPressure(0, req)
	req = withDiskPressure(diskPath, req)
	return cachemeta.PlanPlacement(req)
}
