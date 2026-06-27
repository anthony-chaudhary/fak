package engine

// capacity_disk.go — rung 2 of the multilevel-default-cache epic
// (docs/serving/multilevel-default-cache-epic.md, MLCACHE2): the L3/disk analogue of the
// HBM report->policy wire in capacity_pressure.go and the DRAM wire in capacity_dram.go.
//
// After MLCACHE1 the placement policy planned the DRAM tier against the host's real RAM
// (HostDRAMPressure -> cachemeta.TierPressure[DRAM]) but the spill decision (demote to
// disk) still planned against cachemeta.DefaultTierProfiles' representative numbers and
// an assumed-empty pressure of 0. So a span that beats recompute but whose target disk is
// already full would attempt a spill that fails, rather than evict immediately.
//
// This file closes that asymmetry for L3. It derives the disk tier's live fullness from
// the spill filesystem's real free space (compute.DiskInfo(path) — a backend-free probe,
// so it works on the pure-Go cpu-ref floor) and folds it into a placement request the same
// copy-on-write way withHBMPressure and withHostDRAM do, leaving the caller's request
// value reusable.
//
// Fail open, the contract every capacity plank honors: when the disk probe fails (path
// does not exist, permission denied, unsupported platform) this derives NO pressure and
// overrides NO capacity, so PlanPlacement decides exactly as it did before the wire
// existed and no path that works today regresses.

import (
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// DiskPressure derives the disk tier's live fullness and capacity for the filesystem at
// path — the L3 analogue of HostDRAMPressure, expressed as plain math over compute.DiskInfo
// so it is trivially testable and needs no backend.
//
//	known == false : the path cannot be probed (nonexistent, permission denied, unsupported
//	                 platform). pressure/capacityBytes are 0 and MUST be ignored — the caller
//	                 falls back to the profile default (the fail-open contract).
//	known == true  : total > 0 is the filesystem ceiling (capacityBytes). pressure is the
//	                 USED fraction in [0,1]: used = total - free.
func DiskPressure(path string) (pressure float64, capacityBytes int64, known bool) {
	total, free, ok := compute.DiskInfo(path)
	if !ok || total <= 0 {
		return 0, 0, false
	}
	used := total - free
	if used < 0 {
		used = 0
	}
	if used > total {
		used = total
	}
	return float64(used) / float64(total), total, true
}

// PlanPlacementForDisk plans a cachemeta placement against the filesystem at path that
// ACTUALLY exists: it derives live disk pressure + capacity (DiskPressure), folds them
// into req, and calls cachemeta.PlanPlacement. It is the L3 sibling of PlanPlacementForHost
// — use it when the demote target under consideration is disk and the box has no device
// backend to probe (the cpu-ref/host-only serve path).
//
// Fail open: when the disk probe fails (known=false) req is used VERBATIM, so a path that
// worked before is unchanged. req is not mutated: its Pressure and Profiles maps are
// copied before the disk override is applied, so a shared request value handed to several
// planners stays clean.
func PlanPlacementForDisk(path string, req cachemeta.PlacementRequest) cachemeta.PlacementDecision {
	req = withDiskPressure(path, req)
	return cachemeta.PlanPlacement(req)
}

// withDiskPressure folds live disk pressure + capacity into a COPY of req when the disk
// probe succeeds; on a failed probe it returns req unchanged. Factored so the pattern
// stays consistent with the HBM and DRAM wires.
func withDiskPressure(path string, req cachemeta.PlacementRequest) cachemeta.PlacementRequest {
	pressure, capacity, known := DiskPressure(path)
	if !known {
		return req
	}
	req.Pressure = withTierDiskPressure(req.Pressure, pressure)
	req.Profiles = withTierDiskCapacity(req.Profiles, capacity)
	return req
}

// withTierDiskPressure returns a COPY of in with the disk tier's pressure set to p.
// A nil in yields a one-entry map; every non-disk tier is carried over unchanged.
// Copying keeps a caller's request value reusable across planners.
func withTierDiskPressure(in cachemeta.TierPressure, p float64) cachemeta.TierPressure {
	out := cachemeta.TierPressure{cachemeta.TierDisk: p}
	for t, v := range in {
		if t == cachemeta.TierDisk {
			continue
		}
		out[t] = v
	}
	return out
}

// withTierDiskCapacity returns a COPY of in with the disk tier's CapacityBytes
// overridden by the filesystem's real total. A nil in stays nil (nothing to override);
// a tier table without a disk entry is copied unchanged (only an existing disk profile
// is updated — this wire reports the filesystem ceiling, it does not invent a tier the
// box did not declare). Mirrors withHBMCapacity and withDRAMCapacity.
func withTierDiskCapacity(in map[cachemeta.ResidencyTier]cachemeta.TierProfile, capacity int64) map[cachemeta.ResidencyTier]cachemeta.TierProfile {
	if in == nil {
		return nil
	}
	out := make(map[cachemeta.ResidencyTier]cachemeta.TierProfile, len(in))
	for t, prof := range in {
		out[t] = prof
	}
	if disk, ok := out[cachemeta.TierDisk]; ok {
		disk.CapacityBytes = capacity
		out[cachemeta.TierDisk] = disk
	}
	return out
}

// PlanPlacementForHostAndDisk plans against BOTH live local tiers at once: DRAM pressure
// from the host probe and disk pressure from the filesystem probe. This is the wire a served
// decode loop on a cpu-ref box wants — a span under DRAM pressure may demote to disk, and
// that disk target must itself be planned against real filesystem fullness, or the spill
// fails. Each override is independently fail-open: an unsupported host probe leaves DRAM
// at its default, a failed disk probe leaves disk at its default.
func PlanPlacementForHostAndDisk(dramResidentBytes int64, diskPath string, req cachemeta.PlacementRequest) cachemeta.PlacementDecision {
	req = withHostDRAM(dramResidentBytes, req)
	req = withDiskPressure(diskPath, req)
	return cachemeta.PlanPlacement(req)
}

// PlanPlacementForDeviceHostAndDisk plans against ALL THREE live tiers at once: HBM pressure
// from the device backend (when present), DRAM pressure from the host probe, and disk pressure
// from the filesystem probe. This is the full wire a served decode loop on a GPU box wants —
// a span under HBM pressure may demote to DRAM, then to disk, and each target must itself be
// planned against real fullness. Each override is independently fail-open.
func PlanPlacementForDeviceHostAndDisk(b compute.Backend, hbmResidentBytes, dramResidentBytes int64, diskPath string, req cachemeta.PlacementRequest) cachemeta.PlacementDecision {
	if pressure, capacity, known := DeviceHBMPressure(b, hbmResidentBytes); known {
		req.Pressure = withHBMPressure(req.Pressure, pressure)
		req.Profiles = withHBMCapacity(req.Profiles, capacity)
	}
	req = withHostDRAM(dramResidentBytes, req)
	req = withDiskPressure(diskPath, req)
	return cachemeta.PlanPlacement(req)
}