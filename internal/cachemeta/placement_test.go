package cachemeta

import "testing"

// baseReq builds a placement request for a large, expensive-to-recompute span resident
// in HBM, with all tiers profiled and empty unless overridden.
func baseReq() PlacementRequest {
	profiles := DefaultTierProfiles()
	lc := NewLifecycle(TierHBM, 0).MarkResident(profiles, 0)
	return PlacementRequest{
		Lifecycle:            lc,
		SizeBytes:            64 << 20, // 64 MB KV span
		Tokens:               4000,     // a big system+tool prefix
		Profiles:             profiles,
		Pressure:             TierPressure{},
		Policy:               LifecyclePolicy{DemoteOnExpiry: true},
		PerTokenPrefillNanos: 2_000_000, // 2ms/token prefill — expensive to rebuild
		NowMillis:            1000,
	}
}

// TestKeepWhenFreshAndRoom: no pressure, not expiring -> keep.
func TestKeepWhenFreshAndRoom(t *testing.T) {
	d := PlanPlacement(baseReq())
	if d.Action != ActionKeep {
		t.Fatalf("fresh entry with room should be kept, got %s (%s)", d.Action, d.Reason)
	}
}

// TestDemoteBeatsEvictUnderPressure: HBM full, the big expensive span demotes to DRAM
// (the nearest colder attendable tier with room) rather than being evicted.
func TestDemoteBeatsEvictUnderPressure(t *testing.T) {
	req := baseReq()
	req.Pressure = TierPressure{TierHBM: 1.0} // HBM full
	d := PlanPlacement(req)
	if d.Action != ActionDemote || d.ToTier != TierDRAM {
		t.Fatalf("expensive span should demote HBM->DRAM, got %s->%s (%s)", d.FromTier, d.ToTier, d.Reason)
	}
	if d.Directive != KVOffload {
		t.Fatalf("demote should emit KVOffload, got %s", d.Directive)
	}
}

// TestDemoteSkipsFullTiersToCXL: HBM, DRAM, and NUMA-far all full -> the span lands in
// CXL, the first colder attendable tier with room. This is the headline behavior:
// instead of evicting + recomputing, the hot prefix relocates to CXL far memory.
func TestDemoteSkipsFullTiersToCXL(t *testing.T) {
	req := baseReq()
	req.Pressure = TierPressure{TierHBM: 1.0, TierDRAM: 1.0, TierNUMAFar: 1.0}
	d := PlanPlacement(req)
	if d.Action != ActionDemote || d.ToTier != TierCXL {
		t.Fatalf("should demote past full tiers to CXL, got %s->%s (%s)", d.FromTier, d.ToTier, d.Reason)
	}
}

// TestSpillWhenOnlyDiskHasRoom: every byte-addressable colder tier is full, only disk
// remains -> spill (KVOffload to a non-attendable tier), still cheaper than recompute.
func TestSpillWhenOnlyDiskHasRoom(t *testing.T) {
	req := baseReq()
	req.Pressure = TierPressure{TierHBM: 1.0, TierDRAM: 1.0, TierNUMAFar: 1.0, TierCXL: 1.0}
	d := PlanPlacement(req)
	if d.Action != ActionSpill || d.ToTier != TierDisk {
		t.Fatalf("should spill to disk, got %s->%s (%s)", d.Action, d.ToTier, d.Reason)
	}
}

// TestEvictWhenNothingColderHasRoom: all colder tiers full -> evict (recompute later).
func TestEvictWhenNothingColderHasRoom(t *testing.T) {
	req := baseReq()
	req.Pressure = TierPressure{
		TierHBM: 1.0, TierDRAM: 1.0, TierNUMAFar: 1.0, TierCXL: 1.0, TierDisk: 1.0,
	}
	d := PlanPlacement(req)
	if d.Action != ActionEvict || d.ToTier != TierRecompute {
		t.Fatalf("should evict when nothing colder has room, got %s (%s)", d.Action, d.Reason)
	}
}

// TestCheapSpanEvictsInsteadOfRetaining: when the only colder tier with room is SLOW
// (disk), a span that is trivially cheap to recompute is NOT worth holding — reading
// it back from disk costs more than rebuilding it, so evict rather than spill. This is
// the hardware-aware nuance: staging beats recompute for big/expensive spans (the
// headline), but a small, cheap-to-rebuild span on a slow far tier is the exception
// the cost model must still get right. A fast colder tier (DRAM) would retain it.
func TestCheapSpanEvictsInsteadOfRetaining(t *testing.T) {
	req := baseReq()
	// Only disk has room — every byte-addressable colder tier is full.
	req.Pressure = TierPressure{TierHBM: 1.0, TierDRAM: 1.0, TierNUMAFar: 1.0, TierCXL: 1.0}
	req.Tokens = 3                  // 3 tokens
	req.SizeBytes = 48 << 10        // small KV span
	req.PerTokenPrefillNanos = 1000 // 1us/token -> ~3us recompute, cheaper than a disk read-back
	d := PlanPlacement(req)
	if d.Action != ActionEvict {
		t.Fatalf("cheap span on a slow-only tier should evict, got %s->%s (%s)", d.Action, d.ToTier, d.Reason)
	}
	if d.Reason != "recompute_cheaper_than_retain" {
		t.Fatalf("unexpected reason: %s", d.Reason)
	}
	// Sanity: with a FAST colder tier (DRAM) free, the same span is worth retaining.
	req.Pressure = TierPressure{TierHBM: 1.0}
	if d2 := PlanPlacement(req); d2.Action != ActionDemote || d2.ToTier != TierDRAM {
		t.Fatalf("with DRAM free the cheap span should demote, got %s->%s (%s)", d2.Action, d2.ToTier, d2.Reason)
	}
}

// TestCompressDemoteRequiresQualityEvidence: when the exact span is too large to
// retain but a measured, quality-bounded compressed view is cheap enough to keep, the
// planner chooses compress_demote. Unmeasured or over-bound compression is refused
// and the span is evicted instead.
func TestCompressDemoteRequiresQualityEvidence(t *testing.T) {
	base := baseReq()
	base.Pressure = TierPressure{TierHBM: 1.0, TierDRAM: 1.0, TierNUMAFar: 1.0, TierCXL: 1.0}
	base.Tokens = 10
	base.PerTokenPrefillNanos = 1_000_000 // 10ms recompute
	base.SizeBytes = 1 << 30              // exact disk staging is more expensive than recompute
	base.CompressedSizeBytes = 1 << 20    // compressed disk staging is cheaper than recompute

	ok := base
	ok.Quality = QualityEvidence{Measured: true, QualityDelta: 0.01, MaxQualityDelta: 0.05}
	d := PlanPlacement(ok)
	if d.Action != ActionCompressDemote || d.ToTier != TierDisk {
		t.Fatalf("acceptable compressed span should compress-demote to disk, got %s->%s (%s)", d.Action, d.ToTier, d.Reason)
	}
	if d.EstMoveBytes != ok.CompressedSizeBytes {
		t.Fatalf("compress-demote should stage compressed bytes %d, got %d", ok.CompressedSizeBytes, d.EstMoveBytes)
	}

	for _, tc := range []struct {
		name string
		q    QualityEvidence
	}{
		{name: "unmeasured", q: QualityEvidence{}},
		{name: "over_bound", q: QualityEvidence{Measured: true, QualityDelta: 0.20, MaxQualityDelta: 0.05}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := base
			req.Quality = tc.q
			if d := PlanPlacement(req); d.Action != ActionEvict {
				t.Fatalf("unproven compressed span must evict, got %s->%s (%s)", d.Action, d.ToTier, d.Reason)
			}
		})
	}
}

// TestExpiredNoDemotePolicyEvicts: an expired entry under a policy that forbids
// demotion is dropped outright even with room below.
func TestExpiredNoDemotePolicyEvicts(t *testing.T) {
	req := baseReq()
	req.Policy = LifecyclePolicy{DemoteOnExpiry: false}
	req.Lifecycle.State = StateExpired
	d := PlanPlacement(req)
	if d.Action != ActionEvict || d.Reason != "expired_no_demote" {
		t.Fatalf("expired+no-demote should evict, got %s (%s)", d.Action, d.Reason)
	}
}

// TestPromoteHotColdEntry: a hot entry (high access rate) sitting in CXL with a warmer
// tier free is promoted toward the compute.
func TestPromoteHotColdEntry(t *testing.T) {
	profiles := DefaultTierProfiles()
	lc := NewLifecycle(TierCXL, 0).MarkResident(profiles, 0)
	for i := 0; i < 100; i++ {
		lc = lc.Touch(int64(i * 10)) // 100 hits over ~1s -> ~100/s
	}
	req := PlacementRequest{
		Lifecycle:         lc,
		SizeBytes:         1 << 20,
		Tokens:            1000,
		Profiles:          profiles,
		Pressure:          TierPressure{}, // room everywhere
		PromoteAccessRate: 50,             // promote above 50 hits/sec
		NowMillis:         1000,
	}
	d := PlanPlacement(req)
	if d.Action != ActionPromote || d.ToTier != TierNUMAFar {
		t.Fatalf("hot CXL entry should promote to NUMA-far, got %s->%s (%s)", d.FromTier, d.ToTier, d.Reason)
	}
	if d.Directive != KVRestore {
		t.Fatalf("promote should emit KVRestore, got %s", d.Directive)
	}
}

// TestExpiringEntryIsNotPromoted: an entry whose per-tier TTL just elapsed (Expiring)
// must NOT be promoted into scarcer, shorter-TTL memory on a stale lifetime-hot reading
// — expiry/turnover wins. It should relocate down or evict, never up.
func TestExpiringEntryIsNotPromoted(t *testing.T) {
	profiles := DefaultTierProfiles()
	lc := NewLifecycle(TierCXL, 0).MarkResident(profiles, 0)
	for i := 0; i < 100; i++ {
		lc = lc.Touch(int64(i * 10)) // lifetime-hot ~100/s
	}
	lc.State = StateExpiring // TTL elapsed, not revived by a Touch
	d := PlanPlacement(PlacementRequest{
		Lifecycle:         lc,
		SizeBytes:         1 << 20,
		Tokens:            1000,
		Profiles:          profiles,
		Pressure:          TierPressure{}, // room in the warmer tier too
		PromoteAccessRate: 50,
		NowMillis:         1000,
	})
	if d.Action == ActionPromote {
		t.Fatalf("an expiring entry must not be promoted, got %s->%s (%s)", d.FromTier, d.ToTier, d.Reason)
	}
}

// TestJustMovedEntryNotImmediatelyRepromoted: after a relocation resets the per-tier
// clock, an entry not touched since the move has no CURRENT-tier heat, so it must not
// re-promote on its lifetime count alone — it stays put until a fresh access.
func TestJustMovedEntryNotImmediatelyRepromoted(t *testing.T) {
	profiles := DefaultTierProfiles()
	lc := NewLifecycle(TierNUMAFar, 0).MarkResident(profiles, 0)
	for i := 0; i < 100; i++ {
		lc = lc.Touch(int64(i * 10)) // lifetime-hot
	}
	// Relocate to CXL at t=5000 WITHOUT touching: EnteredTierMillis=5000 > LastAccess=990.
	lc, _ = lc.MoveTo(TierCXL, profiles, 5000)
	d := PlanPlacement(PlacementRequest{
		Lifecycle:         lc,
		SizeBytes:         1 << 20,
		Tokens:            1000,
		Profiles:          profiles,
		Pressure:          TierPressure{}, // room everywhere
		PromoteAccessRate: 50,
		NowMillis:         6000,
	})
	if d.Action != ActionKeep {
		t.Fatalf("a just-moved, un-touched entry should stay put, got %s->%s (%s)", d.FromTier, d.ToTier, d.Reason)
	}
	// But a fresh access in the new tier re-earns promotion eligibility.
	lc = lc.Touch(6000)
	d2 := PlanPlacement(PlacementRequest{
		Lifecycle: lc, SizeBytes: 1 << 20, Tokens: 1000, Profiles: profiles,
		Pressure: TierPressure{}, PromoteAccessRate: 5, NowMillis: 6000,
	})
	if d2.Action != ActionPromote || d2.ToTier != TierNUMAFar {
		t.Fatalf("after a fresh touch the hot entry should promote, got %s->%s (%s)", d2.FromTier, d2.ToTier, d2.Reason)
	}
}

// TestApplyRoundTripsThroughLifecycle: a decision's Apply moves the lifecycle and
// reports the directive, so the planner and the state machine stay consistent.
func TestApplyRoundTrips(t *testing.T) {
	req := baseReq()
	req.Pressure = TierPressure{TierHBM: 1.0}
	d := PlanPlacement(req)
	lc, dir := d.Apply(req.Lifecycle, req.Profiles, req.NowMillis)
	if lc.Tier != TierDRAM || dir != KVOffload {
		t.Fatalf("Apply should land DRAM via KVOffload, got tier=%s dir=%s", lc.Tier, dir)
	}
}

// TestMissingTierSkipped: a box without NUMA-far (no profile) demotes HBM straight to
// DRAM, then to CXL — the absent tier is skipped, not treated as full-or-empty.
func TestMissingTierSkipped(t *testing.T) {
	req := baseReq()
	delete(req.Profiles, TierNUMAFar)
	delete(req.Profiles, TierDRAM)
	req.Pressure = TierPressure{TierHBM: 1.0}
	d := PlanPlacement(req)
	if d.Action != ActionDemote || d.ToTier != TierCXL {
		t.Fatalf("with DRAM+NUMA-far absent, should demote to CXL, got %s->%s (%s)", d.FromTier, d.ToTier, d.Reason)
	}
}
