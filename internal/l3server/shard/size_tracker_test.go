package shard

import (
	"testing"
)

func TestSizeTrackerDisabled(t *testing.T) {
	tr := newValueSizeTracker(0, 0, false, nil)
	for i := 0; i < 5000; i++ {
		tr.record(4194304)
	}
	if tr.detected {
		t.Error("expected detection to remain disabled when warmupTarget=0")
	}
	snap := tr.snapshot()
	if snap.Status != "disabled" {
		t.Errorf("expected status 'disabled', got %q", snap.Status)
	}
}

func TestSizeTrackerWarmup(t *testing.T) {
	tr := newValueSizeTracker(100, 0, false, nil)
	for i := 0; i < 99; i++ {
		tr.record(4194304)
	}
	if tr.detected {
		t.Error("detected too early")
	}
	snap := tr.snapshot()
	if snap.Status != "warming_up" {
		t.Errorf("expected status 'warming_up', got %q", snap.Status)
	}

	// One more should trigger detection
	tr.record(4194304)
	if !tr.detected {
		t.Error("expected detection after reaching warmup target")
	}
	if tr.optimalSize != 4194304 {
		t.Errorf("expected optimalSize=4194304, got %d", tr.optimalSize)
	}
}

func TestSizeTrackerDominantSize(t *testing.T) {
	tr := newValueSizeTracker(100, 0, false, nil)

	// 90 entries of 4MB, 10 of 128 bytes
	for i := 0; i < 90; i++ {
		tr.record(4194304)
	}
	for i := 0; i < 10; i++ {
		tr.record(128)
	}

	if !tr.detected {
		t.Fatal("expected detection")
	}
	if tr.optimalSize != 4194304 {
		t.Errorf("expected dominant size 4194304, got %d", tr.optimalSize)
	}

	snap := tr.snapshot()
	if snap.Status != "detected" {
		t.Errorf("expected status 'detected', got %q", snap.Status)
	}
	if snap.DominantValueSize != 4194304 {
		t.Errorf("expected dominant_value_size=4194304, got %d", snap.DominantValueSize)
	}
	if snap.DominantFreqPercent < 89 || snap.DominantFreqPercent > 91 {
		t.Errorf("expected ~90%% frequency, got %.1f%%", snap.DominantFreqPercent)
	}
}

func TestSizeTrackerMultipleSizes(t *testing.T) {
	tr := newValueSizeTracker(100, 0, false, nil)

	// Even split between 3 sizes; largest count wins
	for i := 0; i < 50; i++ {
		tr.record(1048576) // 1MB â€” most frequent
	}
	for i := 0; i < 30; i++ {
		tr.record(2097152) // 2MB
	}
	for i := 0; i < 20; i++ {
		tr.record(4194304) // 4MB
	}

	if !tr.detected {
		t.Fatal("expected detection")
	}
	if tr.optimalSize != 1048576 {
		t.Errorf("expected dominant size 1048576 (most frequent), got %d", tr.optimalSize)
	}

	top := tr.topSizes(3)
	if len(top) != 3 {
		t.Fatalf("expected 3 top sizes, got %d", len(top))
	}
	// Should be sorted by count descending
	if top[0].Size != 1048576 {
		t.Errorf("expected top[0].Size=1048576, got %d", top[0].Size)
	}
	if top[1].Size != 2097152 {
		t.Errorf("expected top[1].Size=2097152, got %d", top[1].Size)
	}
	if top[2].Size != 4194304 {
		t.Errorf("expected top[2].Size=4194304, got %d", top[2].Size)
	}
}

func TestSizeTrackerReset(t *testing.T) {
	tr := newValueSizeTracker(10, 0, false, nil)
	for i := 0; i < 10; i++ {
		tr.record(4194304)
	}
	if !tr.detected {
		t.Fatal("expected detection before reset")
	}

	tr.reset()
	if tr.detected {
		t.Error("expected detected=false after reset")
	}
	if tr.optimalSize != 0 {
		t.Errorf("expected optimalSize=0 after reset, got %d", tr.optimalSize)
	}
	if tr.totalSets != 0 {
		t.Errorf("expected totalSets=0 after reset, got %d", tr.totalSets)
	}
	if len(tr.counts) != 0 {
		t.Errorf("expected empty counts after reset, got %d entries", len(tr.counts))
	}

	// Should be able to re-detect
	for i := 0; i < 10; i++ {
		tr.record(2097152)
	}
	if !tr.detected {
		t.Error("expected re-detection after reset")
	}
	if tr.optimalSize != 2097152 {
		t.Errorf("expected new optimalSize=2097152, got %d", tr.optimalSize)
	}
}

func TestSizeTrackerSnapshot(t *testing.T) {
	tr := newValueSizeTracker(50, 42, false, nil)

	// Before any records
	snap := tr.snapshot()
	if snap.Status != "warming_up" {
		t.Errorf("expected warming_up, got %q", snap.Status)
	}
	if snap.WarmupProgress != "0/50" {
		t.Errorf("expected progress '0/50', got %q", snap.WarmupProgress)
	}

	// Record enough to detect
	for i := 0; i < 50; i++ {
		tr.record(5242880)
	}
	snap = tr.snapshot()
	if snap.Status != "detected" {
		t.Errorf("expected detected, got %q", snap.Status)
	}
	if snap.DominantValueSize != 5242880 {
		t.Errorf("expected dominant_value_size=5242880, got %d", snap.DominantValueSize)
	}
	if snap.RecommendedPageBytes != 5242880 {
		t.Errorf("expected recommended_model_page_bytes=5242880, got %d", snap.RecommendedPageBytes)
	}
}

func TestSizeTrackerJustDetectedFlag(t *testing.T) {
	tr := newValueSizeTracker(10, 0, false, nil)

	// Before detection
	if tr.justDetected {
		t.Error("justDetected should be false initially")
	}

	for i := 0; i < 10; i++ {
		tr.record(4096)
	}
	if !tr.justDetected {
		t.Error("justDetected should be true after detection")
	}

	// Clear it
	tr.justDetected = false
	if tr.justDetected {
		t.Error("justDetected should be false after clearing")
	}
}

func TestSizeTrackerFrozenFlag(t *testing.T) {
	tr := newValueSizeTracker(5, 0, false, nil)

	// Detect
	for i := 0; i < 5; i++ {
		tr.record(2048)
	}
	if !tr.detected {
		t.Fatal("expected detection")
	}

	// Freeze
	tr.frozen = true
	tr.updateCachedSnapshot()

	snap := tr.snapshot()
	if snap.Status != "rebuilt" {
		t.Errorf("expected 'rebuilt', got %q", snap.Status)
	}

	// Frozen tracker should not record
	tr.record(9999)
	if tr.totalSets != 5 {
		t.Errorf("expected totalSets=5 (frozen), got %d", tr.totalSets)
	}

	// Reset clears frozen
	tr.reset()
	if tr.frozen {
		t.Error("frozen should be false after reset")
	}
	if tr.justDetected {
		t.Error("justDetected should be false after reset")
	}
	snap = tr.snapshot()
	if snap.Status != "warming_up" {
		t.Errorf("expected 'warming_up' after reset, got %q", snap.Status)
	}
}

func TestSizeTrackerCachedSnapshot(t *testing.T) {
	tr := newValueSizeTracker(5, 0, false, nil)

	// Initial snapshot should be warming_up
	snap := tr.snapshot()
	if snap.Status != "warming_up" {
		t.Errorf("expected 'warming_up', got %q", snap.Status)
	}

	// After detection
	for i := 0; i < 5; i++ {
		tr.record(1024)
	}
	snap = tr.snapshot()
	if snap.Status != "detected" {
		t.Errorf("expected 'detected', got %q", snap.Status)
	}
	if snap.DominantValueSize != 1024 {
		t.Errorf("expected dominant 1024, got %d", snap.DominantValueSize)
	}

	// Snapshot is cached â€” should return same value without calling updateCachedSnapshot again
	snap2 := tr.snapshot()
	if snap2.Status != snap.Status {
		t.Error("cached snapshot should be stable")
	}
}
