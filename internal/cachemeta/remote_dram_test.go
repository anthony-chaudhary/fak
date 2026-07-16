package cachemeta

import "testing"

// remote_dram_test.go is the witness for #4306: the peer-DRAM-over-RDMA paging rung
// (TierRemoteDRAM). It proves the rung's physical character, its place in the paging
// order (preferred over local disk, colder than local coherent memory), its
// prove-it-or-drop-it admission, and — the headline acceptance — that a memory-starved
// box pages an expensive KV span to a neighbor's borrowed RAM instead of to local SSD,
// while a box with NO registered lender fails closed to local disk.

// TestRemoteDRAMTierProfile pins the physical character of the paging rung: a random
// page-in an order of magnitude faster than the local NVMe it displaces, streamed
// zero-copy by the NIC (ShareRDMA), but NON-coherent — paged back on access, never
// attended in place — and volatile (a borrowed, reclaimable region).
func TestRemoteDRAMTierProfile(t *testing.T) {
	p := remoteDRAMProfile()
	disk := DefaultTierProfiles()[TierDisk]
	if p.Tier != TierRemoteDRAM {
		t.Fatalf("profile tier = %s, want %s", p.Tier, TierRemoteDRAM)
	}
	// A random RDMA page-in beats an NVMe random read AND streams faster — the whole
	// reason to page to borrowed RAM rather than to local SSD.
	if p.ReadLatencyNanos >= disk.ReadLatencyNanos {
		t.Fatalf("remote-DRAM page-in latency %d must beat disk %d", p.ReadLatencyNanos, disk.ReadLatencyNanos)
	}
	if p.BandwidthMBPerSec <= disk.BandwidthMBPerSec {
		t.Fatalf("remote-DRAM bandwidth %d must exceed disk %d", p.BandwidthMBPerSec, disk.BandwidthMBPerSec)
	}
	if p.Share != ShareRDMA || !p.Share.ZeroCopy() {
		t.Fatalf("remote-DRAM must advertise zero-copy RDMA sharing, got %q", p.Share)
	}
	// Paged, not coherent: a SPILL target, not a demote-in-place tier.
	if p.AttendableInPlace() {
		t.Fatal("remote-DRAM is paged over RDMA; it must NOT be attendable in place")
	}
	if p.Persistent {
		t.Fatal("borrowed peer RAM is volatile; remote-DRAM must not be persistent")
	}
}

// TestRemoteDRAMRankBetweenCXLAndDisk: the paging order is
// HBM > DRAM > NUMA-far > CXL > remote-DRAM > disk > object-store(remote). The rung is
// preferred OVER local disk (the issue's whole point) and colder than local coherent
// memory (CXL); the slow object-store Remote pool stays below disk. It is OFF-BOX, so it
// is not a local (in-box) relocation tier.
func TestRemoteDRAMRankBetweenCXLAndDisk(t *testing.T) {
	if !(TierRank(TierCXL) < TierRank(TierRemoteDRAM) &&
		TierRank(TierRemoteDRAM) < TierRank(TierDisk) &&
		TierRank(TierDisk) < TierRank(TierRemote)) {
		t.Fatalf("want CXL < remote-DRAM < disk < remote(object); got %d, %d, %d, %d",
			TierRank(TierCXL), TierRank(TierRemoteDRAM), TierRank(TierDisk), TierRank(TierRemote))
	}
	if IsLocalTier(TierRemoteDRAM) {
		t.Fatal("remote-DRAM is a peer's RAM; it must not be a local relocation tier")
	}
}

// TestRemoteDRAMInDemoteChain: the demote/paging walk threads remote-DRAM between CXL
// and disk, so a pager that walks NextColderTier reaches borrowed peer RAM before it
// reaches local SSD.
func TestRemoteDRAMInDemoteChain(t *testing.T) {
	if got := NextColderTier(TierCXL); got != TierRemoteDRAM {
		t.Fatalf("NextColderTier(CXL) = %s, want remote-DRAM", got)
	}
	if got := NextColderTier(TierRemoteDRAM); got != TierDisk {
		t.Fatalf("NextColderTier(remote-DRAM) = %s, want disk", got)
	}
}

// TestRemoteDRAMProbedProveItOrDropIt: the rung enters the ladder ONLY when a peer
// registered a lendable region (Present + positive bytes), sized from the probe; a
// Present flag with zero bytes, or no lender at all, drops it (fail-closed — never a
// phantom paging target). The probe sizes the rung; it does not re-measure the physics.
func TestRemoteDRAMProbedProveItOrDropIt(t *testing.T) {
	const lent = 200 << 30 // a neighbor lending 200 GiB of idle DRAM
	got := ProbedTierProfiles(CapacityProbe{
		DRAMBytes: 512 << 30, DiskBytes: 8 << 40,
		RemoteDRAMPresent: true, RemoteDRAMBytes: lent,
	})
	rdma, ok := got[TierRemoteDRAM]
	if !ok {
		t.Fatal("a registered peer lender must put remote-DRAM in the probed ladder")
	}
	if rdma.CapacityBytes != lent {
		t.Fatalf("remote-DRAM capacity = %d, want the lent %d", rdma.CapacityBytes, int64(lent))
	}
	if rdma.BandwidthMBPerSec != remoteDRAMProfile().BandwidthMBPerSec || rdma.ReadLatencyNanos != remoteDRAMProfile().ReadLatencyNanos {
		t.Fatal("the probe sizes the rung; it must not re-measure the physics")
	}
	// No registered lender -> no rung.
	if _, ok := ProbedTierProfiles(CapacityProbe{DRAMBytes: 512 << 30})[TierRemoteDRAM]; ok {
		t.Fatal("absent a registered lender, remote-DRAM must not be in the ladder")
	}
	// Present flag but zero bytes is not a proof.
	if _, ok := ProbedTierProfiles(CapacityProbe{RemoteDRAMPresent: true})[TierRemoteDRAM]; ok {
		t.Fatal("RemoteDRAMPresent with zero bytes must not enter the ladder")
	}
}

// TestPagerPrefersRemoteDRAMOverDisk is the headline acceptance (#4306): a memory-starved
// box with an expensive KV span in local DRAM, DRAM full, pages the span to a neighbor's
// borrowed RAM (SPILL -> remote-DRAM) instead of to local SSD — because the RDMA rung is
// preferred over disk in the paging order. The fail-closed dual proves the failure class:
// with NO peer lender registered, the SAME span spills to local disk, never to memory
// that is not on offer.
func TestPagerPrefersRemoteDRAMOverDisk(t *testing.T) {
	// A starved box: local DRAM + local SSD, plus a neighbor lending 200 GiB of DRAM.
	withLender := ProbedTierProfiles(CapacityProbe{
		DRAMBytes: 512 << 30, DiskBytes: 8 << 40,
		RemoteDRAMPresent: true, RemoteDRAMBytes: 200 << 30,
	})
	req := PlacementRequest{
		Lifecycle:            NewLifecycle(TierDRAM, 0).MarkResident(withLender, 0),
		SizeBytes:            64 << 20, // a big, expensive-to-rebuild KV span
		Tokens:               4000,
		Profiles:             withLender,
		Pressure:             TierPressure{TierDRAM: 1.0}, // local DRAM full
		Policy:               LifecyclePolicy{DemoteOnExpiry: true},
		PerTokenPrefillNanos: 2_000_000, // 2ms/token prefill — expensive to rebuild
		NowMillis:            1000,
	}
	d := PlanPlacement(req)
	if d.Action != ActionSpill || d.ToTier != TierRemoteDRAM {
		t.Fatalf("starved DRAM should page to borrowed peer RAM, got %s -> %s (%s)", d.Action, d.ToTier, d.Reason)
	}
	if d.Directive != KVOffload {
		t.Fatalf("a page-out should emit KVOffload, got %s", d.Directive)
	}

	// Fail-closed: no registered lender -> the same span spills to LOCAL disk, never to a
	// remote-DRAM rung that is not on offer.
	noLender := ProbedTierProfiles(CapacityProbe{DRAMBytes: 512 << 30, DiskBytes: 8 << 40})
	req.Profiles = noLender
	req.Lifecycle = NewLifecycle(TierDRAM, 0).MarkResident(noLender, 0)
	d = PlanPlacement(req)
	if d.Action != ActionSpill || d.ToTier != TierDisk {
		t.Fatalf("with no lender, the span must spill to local disk, got %s -> %s (%s)", d.Action, d.ToTier, d.Reason)
	}
}

// TestModelRemoteDRAMPageInAdvantage is the deterministic MODELED witness for the two
// quantities #4306's Witness names: random KV page-in latency (remote-DRAM-over-RDMA vs
// local-NVMe) and effective KV-pool size gained. It is hardware-free (representative
// profiles); the raw 2-node RDMA confirmation is MEASURED and tracked in #5066.
func TestModelRemoteDRAMPageInAdvantage(t *testing.T) {
	const page = 256 << 10 // a 256 KiB KV page
	adv := ModelRemoteDRAMPageInAdvantage(page, CapacityProbe{RemoteDRAMPresent: true, RemoteDRAMBytes: 200 << 30})
	if adv.Provenance != "MODELED" {
		t.Fatalf("provenance = %q, want MODELED (a modeled advantage is never reported as measured)", adv.Provenance)
	}
	// Paging to borrowed peer RAM beats paging from local NVMe.
	if adv.RemoteDRAMPageInNanos >= adv.LocalDiskPageInNanos {
		t.Fatalf("remote-DRAM page-in %dns must beat local disk %dns", adv.RemoteDRAMPageInNanos, adv.LocalDiskPageInNanos)
	}
	if adv.SpeedupX <= 1 {
		t.Fatalf("remote-DRAM must be faster than disk; modeled speedup = %.3fx", adv.SpeedupX)
	}
	// A registered lender adds exactly its lent region to the KV pool.
	if adv.BorrowedBytesGained != 200<<30 {
		t.Fatalf("effective KV capacity gained = %d, want the lent 200 GiB", adv.BorrowedBytesGained)
	}
	// Fail-closed: no registered lender gains no borrowed capacity (the latency model still holds).
	none := ModelRemoteDRAMPageInAdvantage(page, CapacityProbe{})
	if none.BorrowedBytesGained != 0 {
		t.Fatalf("no lender must gain no borrowed capacity, got %d", none.BorrowedBytesGained)
	}
	if none.SpeedupX <= 1 {
		t.Fatalf("the page-in latency advantage is independent of the lender probe; got %.3fx", none.SpeedupX)
	}
}
