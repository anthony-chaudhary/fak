package cachemeta

// remote_dram.go — the deterministic, no-hardware half of the #4306 witness. The peer-
// DRAM-over-RDMA paging rung (TierRemoteDRAM, hardware.go) models WHERE a starved span
// pages; this file quantifies WHY it should page to a neighbor's borrowed DRAM over
// local SSD, and HOW MUCH KV-pool capacity the borrow buys — the two quantities #4306's
// Witness names ("random KV page-in latency: remote-DRAM-over-RDMA vs local-NVMe" and
// "effective KV-pool size gained").
//
// It is provenance MODELED: it reads the REPRESENTATIVE tier profiles (remoteDRAMProfile
// vs TierDisk), not a measured run, so it is pure and witnessable with no RDMA fabric —
// the same posture ProbedTierProfiles takes ("pure and witnessable with no GPU"). The raw
// 2-node RDMA measurement that confirms the model on real hardware is the private-lab
// half, routed to sanctioned compute and tracked separately (#5066). Labeling the two
// halves by provenance is Law A2: a modeled advantage is never quietly reported as a
// measured one.

// RemoteDRAMPageInAdvantage is the MODELED comparison of paging a KV span to a peer's
// borrowed DRAM over RDMA versus spilling it to local disk, plus the KV-pool capacity a
// registered lender adds. SpeedupX > 1 is the model's claim that remote-DRAM wins; the
// hardware witness (#5066) is what turns MODELED into MEASURED.
type RemoteDRAMPageInAdvantage struct {
	// RemoteDRAMPageInNanos / LocalDiskPageInNanos are the modeled page-in costs (first-
	// byte latency plus streaming time) for a span of the requested size against each
	// paging target — remoteDRAMProfile and the DefaultTierProfiles disk profile.
	RemoteDRAMPageInNanos int64
	LocalDiskPageInNanos  int64
	// SpeedupX is LocalDisk / RemoteDRAM page-in latency: how many times faster paging
	// from borrowed peer RAM is than from local NVMe. > 1 means remote-DRAM wins (the
	// whole reason the rung sits above disk in the paging order).
	SpeedupX float64
	// BorrowedBytesGained is the KV capacity the registered peer lender adds (the
	// borrowed region), i.e. the "effective KV-pool size gained" the issue names. It is
	// named for the borrowed region rather than the pool it grows, so it is not read as a
	// Pool (session container) or PoolProfile (tier pooling character) quantity. 0 when
	// no lender is registered (fail-closed: a borrow that is not on offer gains nothing).
	BorrowedBytesGained int64
	// Provenance is MODELED — representative profiles, not a measured run. The 2-node RDMA
	// confirmation is MEASURED and tracked in #5066.
	Provenance string
}

// ModelRemoteDRAMPageInAdvantage computes the advantage of the peer-DRAM-over-RDMA paging
// rung for a page of pageInBytes, given a CapacityProbe. It reuses the representative
// profiles (remoteDRAMProfile / TierDisk) and the same stageNanos cost model PlanPlacement
// uses, so the witness this returns is exactly the cost the placement policy weighs — pure,
// deterministic, and hardware-free. A non-positive pageInBytes yields a zero-advantage
// record rather than a divide-by-zero.
func ModelRemoteDRAMPageInAdvantage(pageInBytes int64, probe CapacityProbe) RemoteDRAMPageInAdvantage {
	adv := RemoteDRAMPageInAdvantage{Provenance: "MODELED"}
	if pageInBytes <= 0 {
		return adv
	}
	rNanos := stageNanos(pageInBytes, remoteDRAMProfile())
	dNanos := stageNanos(pageInBytes, DefaultTierProfiles()[TierDisk])
	adv.RemoteDRAMPageInNanos = rNanos
	adv.LocalDiskPageInNanos = dNanos
	if rNanos > 0 {
		// Truncate to milli-x so the modeled ratio is a stable, comparable value across
		// runs without pulling in a math dependency (loopmgr's stdlib-only ethos).
		adv.SpeedupX = float64((dNanos*1000)/rNanos) / 1000
	}
	// Only a registered lender adds pool capacity — same prove-it-or-drop-it gate as the
	// tier itself (a borrow that is not on offer gains nothing).
	if probe.RemoteDRAMPresent && probe.RemoteDRAMBytes > 0 {
		adv.BorrowedBytesGained = probe.RemoteDRAMBytes
	}
	return adv
}

// ReclaimRemoteDRAM decides where a span currently paged to a peer's borrowed DRAM
// (TierRemoteDRAM) must go the instant the lender takes its RAM back — the "lease +
// fail-closed reclaim when the lender needs its RAM back" the #4306 sketch names. A
// reclaim voids the rung: the borrowed region is gone, so continuing to read the span
// from it is a use-after-reclaim (the failure class this models). The span MUST re-page
// to a colder LOCAL tier (disk) or, when nothing local can hold it, recompute — it must
// never keep pointing at reclaimed peer memory, and it must never be re-placed back onto
// the very rung that was just reclaimed.
//
// It does not invent a parallel policy: it pins the reclaimed rung to full pressure and
// hands the span to PlanPlacement, whose colder-with-room walk from TierRemoteDRAM is
// exactly disk -> recompute (NextColderTier threads it that way). So the reclaim path is
// the placement policy's own demote/spill/evict decision, just forced by the borrow
// vanishing rather than by ordinary pressure. A span that is NOT on the borrowed rung has
// nothing to reclaim, so it is kept in place.
func ReclaimRemoteDRAM(req PlacementRequest) PlacementDecision {
	from := req.Lifecycle.Tier
	if from != TierRemoteDRAM {
		// Nothing borrowed here to reclaim — leave the span where it is.
		return PlacementDecision{Action: ActionKeep, FromTier: from, ToTier: from, Reason: "not_on_remote_dram"}
	}
	// Copy the pressure map (the caller's is shared) and pin the reclaimed rung full, so
	// PlanPlacement sees the span as under pressure and relocates it strictly colder —
	// never a Keep on, nor a demote back to, the reclaimed borrow.
	pressure := make(TierPressure, len(req.Pressure)+1)
	for t, v := range req.Pressure {
		pressure[t] = v
	}
	pressure[TierRemoteDRAM] = 1.0
	req.Pressure = pressure
	d := PlanPlacement(req)
	// Fail-closed invariant: a reclaim can never resolve back onto the borrowed rung.
	if d.ToTier == TierRemoteDRAM {
		return PlacementDecision{
			Action: ActionEvict, FromTier: from, ToTier: TierRecompute,
			Directive: KVOffload, Reason: "reclaim_forces_recompute",
		}
	}
	return d
}
