package cachemeta

import "testing"

// TestPerTierTTLExpiry proves TTL is genuinely PER TIER: with HBM on a 2s budget and
// CXL on none, an HBM-resident span expires while a CXL-resident one never does — and
// the HBM clock is measured from when the entry entered HBM.
func TestPerTierTTLExpiry(t *testing.T) {
	profiles := DefaultTierProfiles()
	policy := LifecyclePolicy{
		TierTTL:     TierTTL{TierHBM: 2000, TierCXL: 0},
		GraceMillis: 500,
	}

	hbm := NewLifecycle(TierHBM, 0).MarkResident(profiles, 0)
	if hbm.State != StateResident {
		t.Fatalf("HBM fill should land Resident, got %s", hbm.State)
	}
	// Before TTL: stays resident.
	if lc, changed := hbm.Advance(policy, 1999); changed || lc.State != StateResident {
		t.Fatalf("HBM should still be resident at 1999ms")
	}
	// At TTL: -> Expiring.
	hbm, changed := hbm.Advance(policy, 2000)
	if !changed || hbm.State != StateExpiring {
		t.Fatalf("HBM should be Expiring at 2000ms, got %s", hbm.State)
	}
	// After grace: -> Expired.
	hbm, changed = hbm.Advance(policy, 2500)
	if !changed || hbm.State != StateExpired {
		t.Fatalf("HBM should be Expired at 2500ms, got %s", hbm.State)
	}

	// CXL has no TTL: never expires, however long it sits.
	cxl := NewLifecycle(TierCXL, 0).MarkResident(profiles, 0)
	if lc, changed := cxl.Advance(policy, 1_000_000); changed || lc.State != StateResident {
		t.Fatalf("CXL with no TTL must never expire, got %s", lc.State)
	}
}

// TestTouchRevivesDuringGrace shows an access in the Expiring window cancels expiry —
// a hot span proves itself and returns to Resident.
func TestTouchRevivesDuringGrace(t *testing.T) {
	profiles := DefaultTierProfiles()
	policy := LifecyclePolicy{TierTTL: TierTTL{TierHBM: 1000}, GraceMillis: 1000}
	lc := NewLifecycle(TierHBM, 0).MarkResident(profiles, 0)
	lc, _ = lc.Advance(policy, 1000) // -> Expiring
	if lc.State != StateExpiring {
		t.Fatalf("expected Expiring, got %s", lc.State)
	}
	lc = lc.Touch(1200) // hit during grace
	if lc.State != StateResident {
		t.Fatalf("a Touch during grace should revive to Resident, got %s", lc.State)
	}
	// And the TTL clock for the tier did NOT reset on Touch (only EnteredTierMillis does
	// on a tier move), so it can expire again later.
	lc2, changed := lc.Advance(policy, 2000)
	if !changed || lc2.State != StateExpiring {
		t.Fatalf("revived entry should be able to expire again, got %s", lc2.State)
	}
}

// TestMoveToResetsTierClockAndDirection asserts demote vs promote directionality and
// that the per-tier TTL clock resets on a move (freshness is per-tier).
func TestMoveToResetsTierClockAndDirection(t *testing.T) {
	profiles := DefaultTierProfiles()
	lc := NewLifecycle(TierHBM, 0).MarkResident(profiles, 0)

	// Demote HBM -> CXL: KVOffload, lands Resident (CXL is attendable in place),
	// EnteredTierMillis reset to the move time.
	lc, dir := lc.MoveTo(TierCXL, profiles, 5000)
	if dir != KVOffload {
		t.Fatalf("HBM->CXL should be KVOffload, got %s", dir)
	}
	if lc.Tier != TierCXL || lc.State != StateResident || lc.EnteredTierMillis != 5000 {
		t.Fatalf("after demote: tier=%s state=%s entered=%d", lc.Tier, lc.State, lc.EnteredTierMillis)
	}

	// Demote CXL -> Disk: spill (disk is not attendable in place).
	lc, dir = lc.MoveTo(TierDisk, profiles, 6000)
	if dir != KVOffload || lc.State != StateSpilled {
		t.Fatalf("CXL->Disk should KVOffload+Spilled, got dir=%s state=%s", dir, lc.State)
	}

	// Promote Disk -> DRAM: KVRestore, lands Resident.
	lc, dir = lc.MoveTo(TierDRAM, profiles, 7000)
	if dir != KVRestore || lc.State != StateResident {
		t.Fatalf("Disk->DRAM should KVRestore+Resident, got dir=%s state=%s", dir, lc.State)
	}

	// Move to Recompute == eviction.
	lc, dir = lc.MoveTo(TierRecompute, profiles, 8000)
	if lc.State != StateEvicted {
		t.Fatalf("move to Recompute should evict, got %s", lc.State)
	}
	_ = dir
}

// TestServeableStates pins which states satisfy a reuse lookup.
func TestServeableStates(t *testing.T) {
	serveable := map[EntryState]bool{
		StateResident: true, StateExpiring: true, StateSpilled: true,
		StateFilling: false, StateExpired: false, StateEvicted: false,
	}
	for s, want := range serveable {
		if s.Serveable() != want {
			t.Fatalf("%s.Serveable()=%v want %v", s, s.Serveable(), want)
		}
	}
}

// TestAccessRate sanity-checks the promote heuristic input.
func TestAccessRate(t *testing.T) {
	lc := NewLifecycle(TierCXL, 0).MarkResident(DefaultTierProfiles(), 0)
	for i := 0; i < 10; i++ {
		lc = lc.Touch(int64(i * 100))
	}
	// 10 accesses over 1000ms == 10 hits/sec.
	if r := lc.AccessRatePerSec(1000); r < 9.99 || r > 10.01 {
		t.Fatalf("access rate=%v want ~10/s", r)
	}
	if lc.AccessRatePerSec(0) != 0 {
		t.Fatalf("rate at zero elapsed must be 0")
	}
}

// TestFillingNotServeableUntilResident guards the produce-then-serve invariant.
func TestFillingNotServeableUntilResident(t *testing.T) {
	lc := NewLifecycle(TierHBM, 0)
	if lc.State != StateFilling || lc.State.Serveable() {
		t.Fatalf("a new entry must be Filling and not serveable")
	}
	lc = lc.MarkResident(DefaultTierProfiles(), 10)
	if !lc.State.Serveable() {
		t.Fatalf("after MarkResident the entry must be serveable")
	}
}
