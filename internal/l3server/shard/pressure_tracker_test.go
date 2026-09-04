package shard

import (
	"math"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/alloc"
)

// mockAllocForPressure creates a minimal slab allocator for pressure tests.
func mockAllocForPressure(t *testing.T) alloc.Allocator {
	t.Helper()
	a, err := alloc.NewSlabAllocator(alloc.SlabConfig{
		MaxMemoryBytes: 16 * 1024 * 1024, // 16MB â€” small
	})
	if err != nil {
		t.Fatalf("failed to create test allocator: %v", err)
	}
	return a
}

func TestPressureTrackerBasic(t *testing.T) {
	a := mockAllocForPressure(t)
	defer a.Close()

	tracker := newClassPressureTracker(a.NumClasses())
	// Set window start slightly in the past so WindowDuration > 0
	tracker.windowStart = time.Now().Add(-100 * time.Millisecond)

	// Record some evictions and alloc ops
	tracker.recordEviction(0)
	tracker.recordEviction(0)
	tracker.recordEviction(1)
	tracker.recordAllocOp(0)
	tracker.recordAllocOp(0)
	tracker.recordAllocOp(0)
	tracker.recordAllocFailure(0)

	snap := tracker.snapshot(a)
	if snap == nil {
		t.Fatal("expected non-nil snapshot")
	}
	if snap.Evictions[0] != 2 {
		t.Errorf("class 0 evictions: got %d, want 2", snap.Evictions[0])
	}
	if snap.Evictions[1] != 1 {
		t.Errorf("class 1 evictions: got %d, want 1", snap.Evictions[1])
	}
	if snap.AllocOps[0] != 3 {
		t.Errorf("class 0 alloc ops: got %d, want 3", snap.AllocOps[0])
	}
	if snap.AllocFails[0] != 1 {
		t.Errorf("class 0 alloc fails: got %d, want 1", snap.AllocFails[0])
	}
	if snap.WindowDuration <= 0 {
		t.Error("expected positive window duration")
	}

	// loadSnapshot should return the same data
	loaded := tracker.loadSnapshot()
	if loaded == nil {
		t.Fatal("expected loadSnapshot to return non-nil")
	}
	if loaded.Evictions[0] != 2 {
		t.Errorf("loaded class 0 evictions: got %d, want 2", loaded.Evictions[0])
	}
}

func TestPressureTrackerReset(t *testing.T) {
	a := mockAllocForPressure(t)
	defer a.Close()

	tracker := newClassPressureTracker(3)
	tracker.recordEviction(0)
	tracker.recordAllocOp(1)

	// Reset with different class count
	tracker.reset(5)
	if tracker.numClasses != 5 {
		t.Errorf("numClasses: got %d, want 5", tracker.numClasses)
	}
	if len(tracker.classEvictions) != 5 {
		t.Errorf("evictions len: got %d, want 5", len(tracker.classEvictions))
	}
	// All counters should be zero
	for i := 0; i < 5; i++ {
		if tracker.classEvictions[i] != 0 {
			t.Errorf("class %d evictions not zero after reset", i)
		}
	}
}

func TestCustomWeightsInSlabConfig(t *testing.T) {
	// Verify that ClassWeights overrides the default weight logic
	cfg := alloc.SlabConfig{
		MaxMemoryBytes: 16 * 1024 * 1024,
		ModelPageBytes: 5242880,
		ClassWeights: map[uint64]float64{
			64:   50.0,
			5242880: 50.0,
		},
	}
	a, err := alloc.NewSlabAllocator(cfg)
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer a.Close()

	weights := a.CurrentWeights()
	// The two classes we weighted should have 50.0
	if w, ok := weights[64]; !ok || w != 50.0 {
		t.Errorf("class 64: got weight %.1f, want 50.0", w)
	}
	if w, ok := weights[5242880]; !ok || w != 50.0 {
		t.Errorf("class 5242880: got weight %.1f, want 50.0", w)
	}
	// Classes not in the map should get the fallback (0.5)
	if w, ok := weights[1024]; !ok || w != 0.5 {
		t.Errorf("class 1024: got weight %.1f, want 0.5 (fallback)", w)
	}
}

func TestNilClassWeightsUsesDefaults(t *testing.T) {
	// When ClassWeights is nil, the allocator should use default model-based weights
	cfg := alloc.SlabConfig{
		MaxMemoryBytes: 16 * 1024 * 1024,
		ModelPageBytes: 5242880,
		// ClassWeights: nil (not set)
	}
	a, err := alloc.NewSlabAllocator(cfg)
	if err != nil {
		t.Fatalf("NewSlabAllocator: %v", err)
	}
	defer a.Close()

	weights := a.CurrentWeights()
	// The model page class should have weight 100 (dominant)
	if w, ok := weights[5242880]; !ok || w != 100.0 {
		t.Errorf("model class: got weight %.1f, want 100.0", w)
	}
	// A small class should have weight 1.0 (default)
	if w, ok := weights[64]; !ok || w != 1.0 {
		t.Errorf("small class: got weight %.1f, want 1.0", w)
	}
}

func TestComputeClassPressure(t *testing.T) {
	// With globalUtil=0, relativeUtil == classUtil; with promoFromRate=0, promotion term is 0.
	// New formula: relativeUtil*0.30 + min(evictRate/norm,1)*0.30 + allocFailRate*0.20 + min(promoFromRate/norm,1)*0.20
	tests := []struct {
		name          string
		classUtil     float64
		globalUtil    float64
		evictRate     float64
		allocFailRate float64
		promoFromRate float64
		norm          float64
		want          float64
	}{
		{"zero inputs", 0, 0, 0, 0, 0, 10, 0},
		{"full util only (above global)", 1.0, 0, 0, 0, 0, 10, 0.3},
		{"full util but matches global", 0.8, 0.8, 0, 0, 0, 10, 0}, // relativeUtil=0
		{"high eviction only", 0, 0, 20, 0, 0, 10, 0.30},           // evictRate capped at 1.0
		{"alloc fail only", 0, 0, 0, 1.0, 0, 10, 0.2},
		{"promo from rate only", 0, 0, 0, 0, 20, 10, 0.20}, // capped at 1.0
		{"combined", 0.9, 0, 10, 0.1, 5, 10, 0.9*0.30 + 1.0*0.30 + 0.1*0.20 + 0.5*0.20},
		{"zero norm defaults to 10", 0.5, 0, 5, 0, 0, 0, 0.5*0.30 + 0.5*0.30},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ComputeClassPressure(tc.classUtil, tc.globalUtil, tc.evictRate, tc.allocFailRate, tc.promoFromRate, tc.norm)
			if math.Abs(got-tc.want) > 0.001 {
				t.Errorf("ComputeClassPressure(classUtil=%v, globalUtil=%v, evict=%v, fail=%v, promo=%v, norm=%v) = %.4f, want %.4f",
					tc.classUtil, tc.globalUtil, tc.evictRate, tc.allocFailRate, tc.promoFromRate, tc.norm, got, tc.want)
			}
		})
	}
}

func TestSetModelPageHintBounds(t *testing.T) {
	// Create a minimal manager with 1 shard
	s := newTestShard(t, 0, 1048576, 10, true)
	defer s.Allocator().Close()

	m := &Manager{
		shards:    []*Shard{s},
		numShards: 1,
		mask:      0,
	}

	// Too small â€” should be rejected (no panic, modelPageHint stays 0)
	m.SetModelPageHint(32)
	if m.modelPageHint.Load() != 0 {
		t.Errorf("expected hint to stay 0 for too-small value, got %d", m.modelPageHint.Load())
	}

	// Too large â€” should be rejected
	m.SetModelPageHint(128 * 1024 * 1024)
	if m.modelPageHint.Load() != 0 {
		t.Errorf("expected hint to stay 0 for too-large value, got %d", m.modelPageHint.Load())
	}

	// Valid â€” should be accepted
	m.SetModelPageHint(5242880)
	if m.modelPageHint.Load() != 5242880 {
		t.Errorf("expected hint 5242880, got %d", m.modelPageHint.Load())
	}

	// Boundary: exact min
	m.modelPageHint.Store(0)
	m.SetModelPageHint(minPageHintBytes)
	if m.modelPageHint.Load() != minPageHintBytes {
		t.Errorf("expected hint %d (min), got %d", minPageHintBytes, m.modelPageHint.Load())
	}

	// Boundary: exact max
	m.modelPageHint.Store(0)
	m.SetModelPageHint(maxPageHintBytes)
	if m.modelPageHint.Load() != maxPageHintBytes {
		t.Errorf("expected hint %d (max), got %d", maxPageHintBytes, m.modelPageHint.Load())
	}
}

func TestPressureDisabledDefault(t *testing.T) {
	// When PressureRebalancing=false, evaluateShardPressure should use legacy checks
	s := newTestShard(t, 0, 1048576, 10, true) // configured for 1MB
	defer s.Allocator().Close()

	// Simulate detection of different size
	for i := 0; i < 10; i++ {
		s.sizeTracker.record(524288) // 512KB
	}

	m := &Manager{
		shards:    []*Shard{s},
		numShards: 1,
		mask:      0,
		vacuumCfg: VacuumConfig{
			Enabled:              true,
			UtilizationThreshold: 0.85,
			PressureRebalancing:  false, // disabled
		},
	}

	needs, weights, _ := m.evaluateShardPressure(s)
	if !needs {
		t.Error("expected rebalance needed (size mismatch)")
	}
	if weights != nil {
		t.Error("expected nil weights when pressure disabled (legacy path)")
	}
}
