package model

import (
	"testing"
)

func TestQwen35StateLayoutConsumersAgree(t *testing.T) {
	// Acceptance witness (#12439): the preallocated state bank, the recurrent
	// capacity manager, and the canonical layout all agree on one derivation.
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

	t.Run("f32_cross_consumer_parity", func(t *testing.T) {
		layout, err := NewQwen35StateLayout(cfg, "f32")
		if err != nil {
			t.Fatalf("NewQwen35StateLayout(f32) failed: %v", err)
		}

		// 6. Explicit numeric grounding for the f32 fixture so the test is not
		// merely self-referential.
		if layout.ConvElementsPerLayer != 192 {
			t.Fatalf("convElements = %d, want 192", layout.ConvElementsPerLayer)
		}
		if layout.RecurrentElementsPerLayer != 256 {
			t.Fatalf("recurrentElements = %d, want 256", layout.RecurrentElementsPerLayer)
		}
		if layout.ConvBytesPerLayer() != 768 {
			t.Fatalf("convBytes = %d, want 768", layout.ConvBytesPerLayer())
		}
		if layout.RecurrentBytesPerLayer() != 1024 {
			t.Fatalf("recurrentBytes = %d, want 1024", layout.RecurrentBytesPerLayer())
		}
		if layout.BytesPerLayer() != 1792 {
			t.Fatalf("bytesPerLayer = %d, want 1792", layout.BytesPerLayer())
		}
		if layout.BytesPerUnit() != 3584 {
			t.Fatalf("bytesPerUnit = %d, want 3584", layout.BytesPerUnit())
		}
		if layout.TotalBytes(maxUnits) != 10752 {
			t.Fatalf("totalBytes = %d, want 10752", layout.TotalBytes(maxUnits))
		}

		// 2. Capacity manager must agree with the layout on every shared field.
		assertCapacityAgreesWithLayout(t, cfg, maxUnits, "f32", layout)
	})

	for _, dtype := range []string{"f16", "bf16"} {
		t.Run(dtype+"_cross_consumer_parity", func(t *testing.T) {
			layout, err := NewQwen35StateLayout(cfg, dtype)
			if err != nil {
				t.Fatalf("NewQwen35StateLayout(%s) failed: %v", dtype, err)
			}
			// Hardcoded 16-bit ground truth: same fixture geometry (192 conv
			// elements, 256 recurrent elements) priced at ElementBytes=2, so
			// the parity check below is anchored to literals, not to the
			// layout's own arithmetic.
			if layout.ConvElementsPerLayer != 192 {
				t.Fatalf("%s convElements = %d, want 192", dtype, layout.ConvElementsPerLayer)
			}
			if layout.RecurrentElementsPerLayer != 256 {
				t.Fatalf("%s recurrentElements = %d, want 256", dtype, layout.RecurrentElementsPerLayer)
			}
			if layout.ConvBytesPerLayer() != 384 {
				t.Fatalf("%s convBytes = %d, want 384", dtype, layout.ConvBytesPerLayer())
			}
			if layout.RecurrentBytesPerLayer() != 512 {
				t.Fatalf("%s recurrentBytes = %d, want 512", dtype, layout.RecurrentBytesPerLayer())
			}
			if layout.BytesPerLayer() != 896 {
				t.Fatalf("%s bytesPerLayer = %d, want 896", dtype, layout.BytesPerLayer())
			}
			if layout.BytesPerUnit() != 1792 {
				t.Fatalf("%s bytesPerUnit = %d, want 1792", dtype, layout.BytesPerUnit())
			}
			if layout.TotalBytes(maxUnits) != 5376 {
				t.Fatalf("%s totalBytes = %d, want 5376", dtype, layout.TotalBytes(maxUnits))
			}
			assertCapacityAgreesWithLayout(t, cfg, maxUnits, dtype, layout)
		})
	}

	t.Run("non_hybrid_layer_fallback", func(t *testing.T) {
		// No LayerTypes and no linear dims -> isLinearAttnLayer is false for
		// all layers (count 0) and IsQwen35Hybrid() is false, so the derivation
		// falls back to NumLinearLayers=1. Positive dims keep geometry valid.
		fbCfg := Config{
			ModelType:           "qwen3_5_text",
			NumLayers:           2,
			LinearNumKeyHeads:   2,
			LinearKeyHeadDim:    8,
			LinearNumValueHeads: 4,
			LinearValueHeadDim:  8,
			LinearConvKernelDim: 4,
		}
		if fbCfg.IsQwen35Hybrid() {
			t.Fatal("fixture unexpectedly reports IsQwen35Hybrid()==true")
		}
		layout, err := NewQwen35StateLayout(fbCfg, "f32")
		if err != nil {
			t.Fatalf("NewQwen35StateLayout(fallback) failed: %v", err)
		}
		if layout.NumLinearLayers != 1 {
			t.Fatalf("fallback NumLinearLayers = %d, want 1", layout.NumLinearLayers)
		}
		assertCapacityAgreesWithLayout(t, fbCfg, maxUnits, "f32", layout)
	})

	t.Run("kernel_le_one_defaults_to_4", func(t *testing.T) {
		// LinearConvKernelDim == 0 defaults to 4; conv elements are (4-1)*convDim.
		kCfg := Config{
			ModelType:           "qwen3_5_text",
			NumLayers:           2,
			LayerTypes:          []string{"linear_attention", "linear_attention"},
			LinearNumKeyHeads:   2,
			LinearKeyHeadDim:    8,
			LinearNumValueHeads: 4,
			LinearValueHeadDim:  8,
			LinearConvKernelDim: 0,
		}
		convDim := 2*2*8 + 4*8
		layout, err := NewQwen35StateLayout(kCfg, "f32")
		if err != nil {
			t.Fatalf("NewQwen35StateLayout(kernel=0) failed: %v", err)
		}
		if layout.ConvKernel != 4 {
			t.Fatalf("ConvKernel = %d, want 4", layout.ConvKernel)
		}
		if layout.ConvElementsPerLayer != (4-1)*convDim {
			t.Fatalf("convElements = %d, want %d", layout.ConvElementsPerLayer, (4-1)*convDim)
		}
		assertCapacityAgreesWithLayout(t, kCfg, maxUnits, "f32", layout)
	})

	t.Run("recurrent_nk_fallback", func(t *testing.T) {
		// LinearNumValueHeads == 0 makes nV*kHd*vHd == 0, so the recurrent
		// element count falls back to nK*kHd*vHd. convDim stays positive.
		rCfg := Config{
			ModelType:           "qwen3_5_text",
			NumLayers:           2,
			LayerTypes:          []string{"linear_attention", "linear_attention"},
			LinearNumKeyHeads:   2,
			LinearKeyHeadDim:    8,
			LinearNumValueHeads: 0,
			LinearValueHeadDim:  8,
			LinearConvKernelDim: 4,
		}
		layout, err := NewQwen35StateLayout(rCfg, "f32")
		if err != nil {
			t.Fatalf("NewQwen35StateLayout(nK fallback) failed: %v", err)
		}
		if layout.ConvDim <= 0 {
			t.Fatalf("ConvDim = %d, want positive", layout.ConvDim)
		}
		if layout.RecurrentElementsPerLayer != 2*8*8 {
			t.Fatalf("recurrentElements = %d, want %d", layout.RecurrentElementsPerLayer, 2*8*8)
		}
		assertCapacityAgreesWithLayout(t, rCfg, maxUnits, "f32", layout)
	})

	t.Run("pool_matches_layout", func(t *testing.T) {
		layout, err := NewQwen35StateLayout(cfg, "f32")
		if err != nil {
			t.Fatalf("NewQwen35StateLayout(f32) failed: %v", err)
		}

		// 3. The f32 preallocated bank must match the canonical f32 layout.
		bank, err := NewQwen35PreallocatedStateBank(cfg, maxUnits)
		if err != nil {
			t.Fatalf("NewQwen35PreallocatedStateBank failed: %v", err)
		}
		receipt := bank.Receipt()
		if receipt.TotalStateBytes != layout.TotalBytes(maxUnits) {
			t.Fatalf("pool TotalStateBytes = %d, want layout.TotalBytes = %d",
				receipt.TotalStateBytes, layout.TotalBytes(maxUnits))
		}
		if receipt.NumLinearLayers != layout.NumLinearLayers {
			t.Fatalf("pool NumLinearLayers = %d, want %d", receipt.NumLinearLayers, layout.NumLinearLayers)
		}
		if receipt.TotalStateBytes != 10752 {
			t.Fatalf("pool TotalStateBytes = %d, want 10752", receipt.TotalStateBytes)
		}
	})

	t.Run("reject_unsupported_geometry", func(t *testing.T) {
		// 4. Zero geometry must be rejected rather than silently allocating nothing.
		badCfg := Config{
			ModelType:  "qwen3_5_text",
			NumLayers:  1,
			LayerTypes: []string{"linear_attention"},
		}
		if _, err := NewQwen35StateLayout(badCfg, "f32"); err == nil {
			t.Fatal("expected error for zero linear geometry, got nil")
		}
	})

	t.Run("reject_unsupported_dtype_and_default", func(t *testing.T) {
		// 5. Unsupported dtype is rejected; empty defaults to f32.
		if _, err := NewQwen35StateLayout(cfg, "int8"); err == nil {
			t.Fatal("expected error for int8 dtype, got nil")
		}
		layout, err := NewQwen35StateLayout(cfg, "")
		if err != nil {
			t.Fatalf("NewQwen35StateLayout(empty) failed: %v", err)
		}
		if layout.StateDtype != "f32" || layout.ElementBytes != 4 {
			t.Fatalf("empty dtype default = (%s, %d), want (f32, 4)", layout.StateDtype, layout.ElementBytes)
		}
	})
}

func assertCapacityAgreesWithLayout(t *testing.T, cfg Config, maxUnits int, dtype string, layout Qwen35StateLayout) {
	t.Helper()
	mgr, err := NewQwen35RecurrentUnitManagerWithDtype(cfg, maxUnits, dtype)
	if err != nil {
		t.Fatalf("NewQwen35RecurrentUnitManagerWithDtype(%s) failed: %v", dtype, err)
	}
	p := mgr.Pricing()

	if p.StateDtype != layout.StateDtype {
		t.Fatalf("StateDtype = %q, want %q", p.StateDtype, layout.StateDtype)
	}
	if p.ElementBytes != layout.ElementBytes {
		t.Fatalf("ElementBytes = %d, want %d", p.ElementBytes, layout.ElementBytes)
	}
	if p.NumLinearLayers != layout.NumLinearLayers {
		t.Fatalf("NumLinearLayers = %d, want %d", p.NumLinearLayers, layout.NumLinearLayers)
	}
	if p.ConvDim != layout.ConvDim {
		t.Fatalf("ConvDim = %d, want %d", p.ConvDim, layout.ConvDim)
	}
	if p.ConvKernel != layout.ConvKernel {
		t.Fatalf("ConvKernel = %d, want %d", p.ConvKernel, layout.ConvKernel)
	}
	if p.ConvBytesPerLayer != layout.ConvBytesPerLayer() {
		t.Fatalf("ConvBytesPerLayer = %d, want %d", p.ConvBytesPerLayer, layout.ConvBytesPerLayer())
	}
	if p.RecurrentBytesPerLayer != layout.RecurrentBytesPerLayer() {
		t.Fatalf("RecurrentBytesPerLayer = %d, want %d", p.RecurrentBytesPerLayer, layout.RecurrentBytesPerLayer())
	}
	if p.BytesPerLayer != layout.BytesPerLayer() {
		t.Fatalf("BytesPerLayer = %d, want %d", p.BytesPerLayer, layout.BytesPerLayer())
	}
	if p.BytesPerUnit != layout.BytesPerUnit() {
		t.Fatalf("BytesPerUnit = %d, want %d", p.BytesPerUnit, layout.BytesPerUnit())
	}
	if p.TotalCapacityBytes != layout.TotalBytes(maxUnits) {
		t.Fatalf("TotalCapacityBytes = %d, want %d", p.TotalCapacityBytes, layout.TotalBytes(maxUnits))
	}
}
