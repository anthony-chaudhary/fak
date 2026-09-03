package gateway

import (
	"testing"
	"time"
)

func TestRefreshExternalTierRecencyWitness(t *testing.T) {
	// First witness requirements (#9917):
	// 1. Register multiple blocks resident in both device and external tiers.
	// 2. Repeatedly serve hot block from device tier, triggering cross-tier recency refresh.
	// 3. Pressure external tier capacity.
	// 4. Prove hot block survives in external tier while untouched cold block is reclaimed.

	mgr := NewMultiTierRecencyManager()
	t0 := time.Date(2026, 9, 3, 10, 0, 0, 0, time.UTC)

	// Block A (hot) and Block B (cold/untouched) both present on device and external tier at t0
	mgr.RegisterBlock("block-hot-A", true, true, t0)
	mgr.RegisterBlock("block-cold-B", true, true, t0)

	// Repeatedly serve Block A from device tier at subsequent times
	t1 := t0.Add(1 * time.Minute)
	if err := mgr.RecordDeviceHit("block-hot-A", t1); err != nil {
		t.Fatalf("RecordDeviceHit t1 failed: %v", err)
	}

	t2 := t0.Add(2 * time.Minute)
	if err := mgr.RecordDeviceHit("block-hot-A", t2); err != nil {
		t.Fatalf("RecordDeviceHit t2 failed: %v", err)
	}

	// Pressure external tier: reduce external capacity from 2 down to 1
	reclaimed := mgr.ReclaimExternalUnderPressure(1)
	if len(reclaimed) != 1 {
		t.Fatalf("expected 1 reclaimed block, got %d", len(reclaimed))
	}

	// 4. Verify the cold untouched block was reclaimed
	if reclaimed[0] != "block-cold-B" {
		t.Fatalf("expected cold block-cold-B to be reclaimed, got %s", reclaimed[0])
	}

	// Verify the hot block survived in external tier
	if !mgr.IsExternalResident("block-hot-A") {
		t.Fatal("hot block-hot-A was incorrectly reclaimed despite device hits")
	}
	if mgr.IsExternalResident("block-cold-B") {
		t.Fatal("cold block-cold-B still marked as external resident")
	}
}
