package cachemeta

import "testing"

// TestProbedTierProfilesNoGPUOmitsHBM is rung MLCACHE4's witness: on a box that proved no
// device, the chosen ladder has NO HBM tier, and the always-present DRAM tier is sized
// from the live probe (differs from the representative placeholder) — so the planner plans
// against the memory the box actually has, not the order-of-magnitude defaults.
func TestProbedTierProfilesNoGPUOmitsHBM(t *testing.T) {
	const probedDRAM = 1500 << 30 // a 1.5 TB-RAM host, unlike the 512 GiB placeholder
	const probedDisk = 4 << 40
	got := ProbedTierProfiles(CapacityProbe{HBMPresent: false, DRAMBytes: probedDRAM, DiskBytes: probedDisk})

	if _, ok := got[TierHBM]; ok {
		t.Fatalf("a no-GPU box must have no TierHBM in its ladder; got %+v", got)
	}
	dram, ok := got[TierDRAM]
	if !ok {
		t.Fatal("DRAM is always present; it must be in the probed ladder")
	}
	if dram.CapacityBytes != probedDRAM {
		t.Fatalf("DRAM capacity = %d; want the probed %d (must differ from the placeholder)", dram.CapacityBytes, int64(probedDRAM))
	}
	if dram.CapacityBytes == DefaultTierProfiles()[TierDRAM].CapacityBytes {
		t.Fatalf("probed DRAM capacity must differ from the representative default %d", DefaultTierProfiles()[TierDRAM].CapacityBytes)
	}
	if disk, ok := got[TierDisk]; !ok || disk.CapacityBytes != probedDisk {
		t.Fatalf("disk tier = %+v (ok=%v); want capacity %d", disk, ok, int64(probedDisk))
	}
}

// TestProbedTierProfilesWithGPUSizesHBM: when the box proved a device, HBM is present and
// sized from the probe (not the 80 GiB placeholder).
func TestProbedTierProfilesWithGPUSizesHBM(t *testing.T) {
	const probedHBM = 40 << 30 // an A100-40GB, unlike the 80 GiB placeholder
	got := ProbedTierProfiles(CapacityProbe{HBMPresent: true, HBMBytes: probedHBM, DRAMBytes: 256 << 30})

	hbm, ok := got[TierHBM]
	if !ok {
		t.Fatal("a box that proved a device must have a TierHBM")
	}
	if hbm.CapacityBytes != probedHBM {
		t.Fatalf("HBM capacity = %d; want the probed %d", hbm.CapacityBytes, int64(probedHBM))
	}
	if hbm.CapacityBytes == DefaultTierProfiles()[TierHBM].CapacityBytes {
		t.Fatalf("probed HBM capacity must differ from the representative default %d", DefaultTierProfiles()[TierHBM].CapacityBytes)
	}
	// Non-capacity physics stays at the representative value — the probe sizes, not re-measures.
	if hbm.BandwidthMBPerSec != DefaultTierProfiles()[TierHBM].BandwidthMBPerSec {
		t.Fatalf("HBM bandwidth changed: %d != %d", hbm.BandwidthMBPerSec, DefaultTierProfiles()[TierHBM].BandwidthMBPerSec)
	}
}

// TestProbedTierProfilesDropsUnprovableTiers: far tiers the probe did NOT confirm
// (NUMA-far/CXL absent from the probe) and the off-box Remote pool (which still has no
// local probe at all) are never asserted into the proved ladder.
func TestProbedTierProfilesDropsUnprovableTiers(t *testing.T) {
	got := ProbedTierProfiles(CapacityProbe{HBMPresent: true, HBMBytes: 80 << 30, DRAMBytes: 512 << 30, DiskBytes: 8 << 40})
	for _, tier := range []ResidencyTier{TierNUMAFar, TierCXL, TierRemote} {
		if _, ok := got[tier]; ok {
			t.Errorf("unproven tier %q must not be in the probed ladder", tier)
		}
	}
}

// TestProbedTierProfilesIncludesProvenFarMemory is the #1470 ladder half: a box whose
// far-memory probe confirmed NUMA-far and/or CXL gets those tiers in its proved ladder,
// sized from the probe, with the non-capacity physics staying representative — and a
// Present flag without positive bytes still drops the tier (a zero-size demote target
// is not a tier).
func TestProbedTierProfilesIncludesProvenFarMemory(t *testing.T) {
	const probedFar = 384 << 30 // the other socket's 384 GiB, unlike the 512 GiB placeholder
	const probedCXL = 1 << 40   // a 1 TiB expansion pool, unlike the 2 TiB placeholder
	got := ProbedTierProfiles(CapacityProbe{
		DRAMBytes:      512 << 30,
		NUMAFarPresent: true, NUMAFarBytes: probedFar,
		CXLPresent: true, CXLBytes: probedCXL,
	})

	far, ok := got[TierNUMAFar]
	if !ok {
		t.Fatal("a box that proved far NUMA memory must have TierNUMAFar in its ladder")
	}
	if far.CapacityBytes != probedFar {
		t.Fatalf("NUMA-far capacity = %d; want the probed %d", far.CapacityBytes, int64(probedFar))
	}
	cxl, ok := got[TierCXL]
	if !ok {
		t.Fatal("a box that proved CXL memory must have TierCXL in its ladder")
	}
	if cxl.CapacityBytes != probedCXL {
		t.Fatalf("CXL capacity = %d; want the probed %d", cxl.CapacityBytes, int64(probedCXL))
	}
	// The probe sizes the ladder; it does not re-measure the physics.
	def := DefaultTierProfiles()
	if far.BandwidthMBPerSec != def[TierNUMAFar].BandwidthMBPerSec || cxl.BandwidthMBPerSec != def[TierCXL].BandwidthMBPerSec {
		t.Fatal("far-tier bandwidth must stay at the representative value")
	}
	if far.ReadLatencyNanos != def[TierNUMAFar].ReadLatencyNanos || !cxl.AttendableInPlace() {
		t.Fatal("far-tier physics must stay representative (latency, addressability)")
	}

	// Present without positive bytes is not a proof.
	got = ProbedTierProfiles(CapacityProbe{NUMAFarPresent: true, CXLPresent: true})
	if _, ok := got[TierNUMAFar]; ok {
		t.Fatal("NUMAFarPresent with zero bytes must not enter the ladder")
	}
	if _, ok := got[TierCXL]; ok {
		t.Fatal("CXLPresent with zero bytes must not enter the ladder")
	}
}

// TestProbedTierProfilesKeepsDefaultWhenUnprobed: a non-positive reading is "unknown", not
// "zero capacity" — the always-present tiers keep their representative default so a failed
// probe degrades to today's behavior rather than claiming an empty tier.
func TestProbedTierProfilesKeepsDefaultWhenUnprobed(t *testing.T) {
	got := ProbedTierProfiles(CapacityProbe{HBMPresent: false}) // every reading zero
	def := DefaultTierProfiles()
	if got[TierDRAM].CapacityBytes != def[TierDRAM].CapacityBytes {
		t.Fatalf("unprobed DRAM = %d; want the default %d", got[TierDRAM].CapacityBytes, def[TierDRAM].CapacityBytes)
	}
	if got[TierDisk].CapacityBytes != def[TierDisk].CapacityBytes {
		t.Fatalf("unprobed disk = %d; want the default %d", got[TierDisk].CapacityBytes, def[TierDisk].CapacityBytes)
	}
	if _, ok := got[TierHBM]; ok {
		t.Fatal("HBM absent without a proved device, even when bytes are zero")
	}
}
