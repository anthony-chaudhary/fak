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
	if pricing.StateDtype != "f32" {
		t.Fatalf("stateDtype = %q, want f32", pricing.StateDtype)
	}
	if pricing.ElementBytes != 4 {
		t.Fatalf("elementBytes = %d, want 4", pricing.ElementBytes)
	}
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

func TestQwen35Recurrent16BitPricingAndCapacityWitness(t *testing.T) {
	// Requirements (#11103):
	// 1. 16-bit recurrent pricing halves memory consumption per unit (50% recurrent state bytes).
	// 2. Capacity allows doubling max concurrent units within the same total byte budget.
	// 3. State allocation produces exact 16-bit byte buffers for admitted units.

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

	maxUnits := 3
	f32Pricing, err := ComputeQwen35RecurrentPricing(cfg, maxUnits)
	if err != nil {
		t.Fatalf("ComputeQwen35RecurrentPricing failed: %v", err)
	}

	for _, dtype := range []string{"f16", "bf16"} {
		t.Run(dtype, func(t *testing.T) {
			pricing16, err := ComputeQwen35RecurrentPricingWithDtype(cfg, maxUnits, dtype)
			if err != nil {
				t.Fatalf("ComputeQwen35RecurrentPricingWithDtype(%s) failed: %v", dtype, err)
			}

			// 1. Prove 16-bit recurrent pricing halves memory consumption per unit
			if pricing16.StateDtype != dtype {
				t.Fatalf("stateDtype = %q, want %q", pricing16.StateDtype, dtype)
			}
			if pricing16.ElementBytes != 2 {
				t.Fatalf("elementBytes = %d, want 2", pricing16.ElementBytes)
			}
			if pricing16.RecurrentBytesPerLayer != f32Pricing.RecurrentBytesPerLayer/2 {
				t.Fatalf("%s recurrentBytesPerLayer = %d, want %d (50%% of f32 %d)",
					dtype, pricing16.RecurrentBytesPerLayer, f32Pricing.RecurrentBytesPerLayer/2, f32Pricing.RecurrentBytesPerLayer)
			}
			if pricing16.ConvBytesPerLayer != f32Pricing.ConvBytesPerLayer/2 {
				t.Fatalf("%s convBytesPerLayer = %d, want %d (50%% of f32 %d)",
					dtype, pricing16.ConvBytesPerLayer, f32Pricing.ConvBytesPerLayer/2, f32Pricing.ConvBytesPerLayer)
			}
			if pricing16.BytesPerLayer != f32Pricing.BytesPerLayer/2 {
				t.Fatalf("%s bytesPerLayer = %d, want %d (50%% of f32 %d)",
					dtype, pricing16.BytesPerLayer, f32Pricing.BytesPerLayer/2, f32Pricing.BytesPerLayer)
			}
			if pricing16.BytesPerUnit != f32Pricing.BytesPerUnit/2 {
				t.Fatalf("%s bytesPerUnit = %d, want %d (50%% of f32 %d)",
					dtype, pricing16.BytesPerUnit, f32Pricing.BytesPerUnit/2, f32Pricing.BytesPerUnit)
			}
			if pricing16.TotalCapacityBytes != f32Pricing.TotalCapacityBytes/2 {
				t.Fatalf("%s totalCapacityBytes = %d, want %d (50%% of f32 %d)",
					dtype, pricing16.TotalCapacityBytes, f32Pricing.TotalCapacityBytes/2, f32Pricing.TotalCapacityBytes)
			}
		})
	}

	t.Run("double_capacity_within_same_byte_budget", func(t *testing.T) {
		// 2. Prove capacity allows doubling max concurrent units within the same total byte budget
		// Budget: f32Pricing.TotalCapacityBytes (3584 * 3 = 10752 bytes).
		// Under 16-bit (1792 bytes/unit), double the units (6 units) requires exactly 10752 bytes.
		doubleUnits := maxUnits * 2
		pricingDouble, err := ComputeQwen35RecurrentPricingWithDtype(cfg, doubleUnits, "f16")
		if err != nil {
			t.Fatalf("ComputeQwen35RecurrentPricingWithDtype failed: %v", err)
		}

		if pricingDouble.TotalCapacityBytes != f32Pricing.TotalCapacityBytes {
			t.Fatalf("totalCapacityBytes with 2x units = %d, want exact same budget %d as f32 with 1x units",
				pricingDouble.TotalCapacityBytes, f32Pricing.TotalCapacityBytes)
		}

		mgrDouble, err := NewQwen35RecurrentUnitManagerWithDtype(cfg, doubleUnits, "f16")
		if err != nil {
			t.Fatalf("NewQwen35RecurrentUnitManagerWithDtype failed: %v", err)
		}

		admitted := make([]int, doubleUnits)
		for u := 0; u < doubleUnits; u++ {
			unitID, state, err := mgrDouble.AdmitWithState()
			if err != nil {
				t.Fatalf("admit unit %d failed: %v", u, err)
			}
			admitted[u] = unitID
			if state.TotalBytes != pricingDouble.BytesPerUnit {
				t.Fatalf("allocated state totalBytes = %d, want %d", state.TotalBytes, pricingDouble.BytesPerUnit)
			}
			if len(state.ConvBuffers) != cfg.NumLayers || len(state.RecurrentBuffers) != cfg.NumLayers {
				t.Fatalf("layer buffer count mismatch: %d conv, %d recurrent, want %d",
					len(state.ConvBuffers), len(state.RecurrentBuffers), cfg.NumLayers)
			}
			if int64(len(state.ConvBuffers[0])) != pricingDouble.ConvBytesPerLayer {
				t.Fatalf("conv buffer len = %d, want %d", len(state.ConvBuffers[0]), pricingDouble.ConvBytesPerLayer)
			}
			if int64(len(state.RecurrentBuffers[0])) != pricingDouble.RecurrentBytesPerLayer {
				t.Fatalf("recurrent buffer len = %d, want %d", len(state.RecurrentBuffers[0]), pricingDouble.RecurrentBytesPerLayer)
			}
		}

		if mgrDouble.ActiveUnits() != doubleUnits || mgrDouble.FreeUnits() != 0 {
			t.Fatalf("expected %d active, 0 free units; got %d active, %d free",
				doubleUnits, mgrDouble.ActiveUnits(), mgrDouble.FreeUnits())
		}
		if mgrDouble.CommittedBytes() != f32Pricing.TotalCapacityBytes {
			t.Fatalf("committed bytes = %d, want exact budget %d",
				mgrDouble.CommittedBytes(), f32Pricing.TotalCapacityBytes)
		}

		// (2N + 1)-th unit must be deterministically refused
		extraUnit, err := mgrDouble.Admit()
		if err == nil {
			t.Fatalf("admit 2N+1 unexpectedly succeeded with unit %d", extraUnit)
		}

		// Releasing a unit frees it and allows reuse
		if err := mgrDouble.Release(admitted[2]); err != nil {
			t.Fatalf("release unit failed: %v", err)
		}
		if mgrDouble.ActiveUnits() != doubleUnits-1 || mgrDouble.FreeUnits() != 1 {
			t.Fatalf("expected %d active, 1 free; got %d active, %d free",
				doubleUnits-1, mgrDouble.ActiveUnits(), mgrDouble.FreeUnits())
		}

		reAdmitted, err := mgrDouble.Admit()
		if err != nil || reAdmitted != admitted[2] {
			t.Fatalf("re-admission failed: got %d, err %v, want %d", reAdmitted, err, admitted[2])
		}
	})

	t.Run("validation_and_defaults", func(t *testing.T) {
		// Empty defaults to f32
		pDefault, err := ComputeQwen35RecurrentPricingWithDtype(cfg, 1, "")
		if err != nil || pDefault.StateDtype != "f32" || pDefault.ElementBytes != 4 {
			t.Fatalf("expected f32 default, got %+v, err %v", pDefault, err)
		}

		// Unsupported dtype rejected
		_, err = ComputeQwen35RecurrentPricingWithDtype(cfg, 1, "int8")
		if err == nil {
			t.Fatal("expected error for int8 dtype")
		}

		// Invalid maxUnits rejected
		_, err = ComputeQwen35RecurrentPricingWithDtype(cfg, 0, "f16")
		if err == nil {
			t.Fatal("expected error for maxUnits = 0")
		}
	})
}
