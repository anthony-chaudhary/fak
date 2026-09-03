package model

import (
	"testing"
)

func TestQwen35RecurrentPricingAndCapacityWitness(t *testing.T) {
	// First witness requirements (#9957):
	// 1. Tiny geometry fixture emits exact conv + recurrent bytes per unit
	// 2. Accepts N units at capacity
	// 3. Deterministically refuses N+1 before any backend buffer is allocated
	// 4. Free unit tracking and reuse

	cfg := Config{
		ModelType:           "qwen3_5_text",
		NumLayers:           2,
		LayerTypes:          []string{"linear_attention", "linear_attention"},
		LinearNumKeyHeads:   2,
		LinearKeyHeadDim:    8,
		LinearNumValueHeads: 4,
		LinearValueHeadDim:  8,
		LinearConvKernelDim: 4,
	}

	maxUnits := 3 // Capacity N = 3
	pricing, err := ComputeQwen35RecurrentPricing(cfg, maxUnits)
	if err != nil {
		t.Fatalf("ComputeQwen35RecurrentPricing failed: %v", err)
	}

	// 1. Verify exact bytes per layer and per unit
	if pricing.ConvDim != 64 {
		t.Fatalf("convDim = %d, want 64", pricing.ConvDim)
	}
	if pricing.ConvBytesPerLayer != 768 {
		t.Fatalf("convBytesPerLayer = %d, want 768", pricing.ConvBytesPerLayer)
	}
	if pricing.RecurrentBytesPerLayer != 1024 {
		t.Fatalf("recurrentBytesPerLayer = %d, want 1024", pricing.RecurrentBytesPerLayer)
	}
	if pricing.BytesPerLayer != 1792 {
		t.Fatalf("bytesPerLayer = %d, want 1792", pricing.BytesPerLayer)
	}
	if pricing.BytesPerUnit != 3584 {
		t.Fatalf("bytesPerUnit = %d, want 3584", pricing.BytesPerUnit)
	}
	if pricing.TotalCapacityBytes != 3584*3 {
		t.Fatalf("totalCapacityBytes = %d, want %d", pricing.TotalCapacityBytes, 3584*3)
	}

	mgr, err := NewQwen35RecurrentUnitManager(cfg, maxUnits)
	if err != nil {
		t.Fatalf("NewQwen35RecurrentUnitManager failed: %v", err)
	}

	// 2. Accept exactly N=3 units
	u0, err := mgr.Admit()
	if err != nil || u0 < 0 {
		t.Fatalf("admit unit 0 failed: %v", err)
	}
	u1, err := mgr.Admit()
	if err != nil || u1 < 0 {
		t.Fatalf("admit unit 1 failed: %v", err)
	}
	u2, err := mgr.Admit()
	if err != nil || u2 < 0 {
		t.Fatalf("admit unit 2 failed: %v", err)
	}

	if mgr.ActiveUnits() != 3 || mgr.FreeUnits() != 0 {
		t.Fatalf("expected 3 active, 0 free units; got %d active, %d free", mgr.ActiveUnits(), mgr.FreeUnits())
	}

	// 3. Deterministically refuse N+1
	u3, err := mgr.Admit()
	if err == nil {
		t.Fatalf("admit N+1 unexpectedly succeeded with unit %d", u3)
	}

	// 4. Releasing a unit permits subsequent admission
	if err := mgr.Release(u1); err != nil {
		t.Fatalf("release u1 failed: %v", err)
	}
	if mgr.ActiveUnits() != 2 || mgr.FreeUnits() != 1 {
		t.Fatalf("expected 2 active, 1 free; got %d active, %d free", mgr.ActiveUnits(), mgr.FreeUnits())
	}

	uReplaced, err := mgr.Admit()
	if err != nil || uReplaced != u1 {
		t.Fatalf("re-admission failed: got unit %d, err %v (want %d)", uReplaced, err, u1)
	}
}
