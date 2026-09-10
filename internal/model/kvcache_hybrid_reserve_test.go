package model

import (
	"reflect"
	"testing"
)

// TestKVCacheReserve_HybridReserveOnlyFullAttention proves that KVCache.Reserve
// and CloneWithReserve allocate spare capacity only on full-attention layers
// in hybrid models, leaving recurrent (linear_attention) layers empty.
func TestKVCacheReserve_HybridReserveOnlyFullAttention(t *testing.T) {
	cfg := Config{
		ModelType:  "qwen3_5",
		NumLayers:  4,
		NumKVHeads: 2,
		HeadDim:    64,
		LayerTypes: []string{
			"linear_attention", // layer 0: recurrent
			"full_attention",   // layer 1: full
			"linear_attention", // layer 2: recurrent
			"full_attention",   // layer 3: full
		},
	}
	stride := cfg.NumKVHeads * cfg.HeadDim // 128
	const extraPositions = 16
	wantReserveFloats := extraPositions * stride

	t.Run("Reserve", func(t *testing.T) {
		cache := NewKVCache(cfg)
		cache.Reserve(extraPositions)

		for l := 0; l < cfg.NumLayers; l++ {
			if cfg.isLinearAttnLayer(l) {
				// Recurrent layers must remain empty
				if cap(cache.K[l]) != 0 || len(cache.K[l]) != 0 {
					t.Fatalf("layer %d (recurrent): K cap=%d len=%d, want 0", l, cap(cache.K[l]), len(cache.K[l]))
				}
				if cap(cache.Kraw[l]) != 0 || len(cache.Kraw[l]) != 0 {
					t.Fatalf("layer %d (recurrent): Kraw cap=%d len=%d, want 0", l, cap(cache.Kraw[l]), len(cache.Kraw[l]))
				}
				if cap(cache.V[l]) != 0 || len(cache.V[l]) != 0 {
					t.Fatalf("layer %d (recurrent): V cap=%d len=%d, want 0", l, cap(cache.V[l]), len(cache.V[l]))
				}
			} else {
				// Full attention layers must reserve spare capacity
				if cap(cache.K[l]) < wantReserveFloats {
					t.Fatalf("layer %d (full): K cap=%d, want >= %d", l, cap(cache.K[l]), wantReserveFloats)
				}
				if cap(cache.Kraw[l]) < wantReserveFloats {
					t.Fatalf("layer %d (full): Kraw cap=%d, want >= %d", l, cap(cache.Kraw[l]), wantReserveFloats)
				}
				if cap(cache.V[l]) < wantReserveFloats {
					t.Fatalf("layer %d (full): V cap=%d, want >= %d", l, cap(cache.V[l]), wantReserveFloats)
				}
			}
		}
	})

	t.Run("CloneWithReserve", func(t *testing.T) {
		cache := NewKVCache(cfg)
		clone := cache.CloneWithReserve(extraPositions)

		for l := 0; l < cfg.NumLayers; l++ {
			if cfg.isLinearAttnLayer(l) {
				// Recurrent layers must remain empty in clone
				if cap(clone.K[l]) != 0 || len(clone.K[l]) != 0 {
					t.Fatalf("layer %d (recurrent clone): K cap=%d len=%d, want 0", l, cap(clone.K[l]), len(clone.K[l]))
				}
				if cap(clone.Kraw[l]) != 0 || len(clone.Kraw[l]) != 0 {
					t.Fatalf("layer %d (recurrent clone): Kraw cap=%d len=%d, want 0", l, cap(clone.Kraw[l]), len(clone.Kraw[l]))
				}
				if cap(clone.V[l]) != 0 || len(clone.V[l]) != 0 {
					t.Fatalf("layer %d (recurrent clone): V cap=%d len=%d, want 0", l, cap(clone.V[l]), len(clone.V[l]))
				}
			} else {
				// Full attention layers must reserve spare capacity in clone
				if cap(clone.K[l]) < wantReserveFloats {
					t.Fatalf("layer %d (full clone): K cap=%d, want >= %d", l, cap(clone.K[l]), wantReserveFloats)
				}
				if cap(clone.Kraw[l]) < wantReserveFloats {
					t.Fatalf("layer %d (full clone): Kraw cap=%d, want >= %d", l, cap(clone.Kraw[l]), wantReserveFloats)
				}
				if cap(clone.V[l]) < wantReserveFloats {
					t.Fatalf("layer %d (full clone): V cap=%d, want >= %d", l, cap(clone.V[l]), wantReserveFloats)
				}
			}
		}
	})
}

// TestKVCacheReserve_PreservesPreExistingRecurrentKV proves that if a recurrent layer
// already contains token-indexed KV (e.g. impossible pre-existing data or custom test fixture),
// Reserve and CloneWithReserve preserve its contents and capacity rather than dropping data.
func TestKVCacheReserve_PreservesPreExistingRecurrentKV(t *testing.T) {
	cfg := Config{
		ModelType:  "qwen3_5",
		NumLayers:  2,
		NumKVHeads: 1,
		HeadDim:    4,
		LayerTypes: []string{
			"linear_attention", // layer 0: recurrent
			"full_attention",   // layer 1: full
		},
	}

	testDataK := []float32{1.0, 2.0, 3.0, 4.0}
	testDataKraw := []float32{1.1, 2.1, 3.1, 4.1}
	testDataV := []float32{5.0, 6.0, 7.0, 8.0}

	t.Run("ReservePreservesExisting", func(t *testing.T) {
		cache := NewKVCache(cfg)
		// Inject pre-existing KV with capacity 16 on recurrent layer 0
		cache.K[0] = make([]float32, len(testDataK), 16)
		copy(cache.K[0], testDataK)
		cache.Kraw[0] = make([]float32, len(testDataKraw), 16)
		copy(cache.Kraw[0], testDataKraw)
		cache.V[0] = make([]float32, len(testDataV), 16)
		copy(cache.V[0], testDataV)

		cache.Reserve(8)

		if !reflect.DeepEqual(cache.K[0], testDataK) {
			t.Fatalf("recurrent layer 0 K contents modified: got %v, want %v", cache.K[0], testDataK)
		}
		if cap(cache.K[0]) != 16 {
			t.Fatalf("recurrent layer 0 K capacity modified: got %d, want 16", cap(cache.K[0]))
		}
		if !reflect.DeepEqual(cache.Kraw[0], testDataKraw) {
			t.Fatalf("recurrent layer 0 Kraw contents modified: got %v, want %v", cache.Kraw[0], testDataKraw)
		}
		if cap(cache.Kraw[0]) != 16 {
			t.Fatalf("recurrent layer 0 Kraw capacity modified: got %d, want 16", cap(cache.Kraw[0]))
		}
		if !reflect.DeepEqual(cache.V[0], testDataV) {
			t.Fatalf("recurrent layer 0 V contents modified: got %v, want %v", cache.V[0], testDataV)
		}
		if cap(cache.V[0]) != 16 {
			t.Fatalf("recurrent layer 0 V capacity modified: got %d, want 16", cap(cache.V[0]))
		}
	})

	t.Run("CloneWithReservePreservesExisting", func(t *testing.T) {
		cache := NewKVCache(cfg)
		cache.K[0] = make([]float32, len(testDataK), 16)
		copy(cache.K[0], testDataK)
		cache.Kraw[0] = make([]float32, len(testDataKraw), 16)
		copy(cache.Kraw[0], testDataKraw)
		cache.V[0] = make([]float32, len(testDataV), 16)
		copy(cache.V[0], testDataV)

		clone := cache.CloneWithReserve(8)

		if !reflect.DeepEqual(clone.K[0], testDataK) {
			t.Fatalf("recurrent layer 0 clone K contents modified: got %v, want %v", clone.K[0], testDataK)
		}
		if cap(clone.K[0]) != 16 {
			t.Fatalf("recurrent layer 0 clone K capacity dropped: got %d, want 16", cap(clone.K[0]))
		}
		if !reflect.DeepEqual(clone.Kraw[0], testDataKraw) {
			t.Fatalf("recurrent layer 0 clone Kraw contents modified: got %v, want %v", clone.Kraw[0], testDataKraw)
		}
		if cap(clone.Kraw[0]) != 16 {
			t.Fatalf("recurrent layer 0 clone Kraw capacity dropped: got %d, want 16", cap(clone.Kraw[0]))
		}
		if !reflect.DeepEqual(clone.V[0], testDataV) {
			t.Fatalf("recurrent layer 0 clone V contents modified: got %v, want %v", clone.V[0], testDataV)
		}
		if cap(clone.V[0]) != 16 {
			t.Fatalf("recurrent layer 0 clone V capacity dropped: got %d, want 16", cap(clone.V[0]))
		}
	})
}

// TestKVCacheReserve_NonhybridUnchanged proves that nonhybrid models (where no layer
// is linear_attention) continue to reserve spare capacity on every layer.
func TestKVCacheReserve_NonhybridUnchanged(t *testing.T) {
	cfg := Config{
		ModelType:  "llama",
		NumLayers:  3,
		NumKVHeads: 2,
		HeadDim:    64,
		LayerTypes: nil, // nonhybrid
	}
	stride := cfg.NumKVHeads * cfg.HeadDim
	const extraPositions = 12
	wantReserveFloats := extraPositions * stride

	cache := NewKVCache(cfg)
	cache.Reserve(extraPositions)

	for l := 0; l < cfg.NumLayers; l++ {
		if cap(cache.K[l]) < wantReserveFloats {
			t.Fatalf("nonhybrid layer %d: K cap=%d, want >= %d", l, cap(cache.K[l]), wantReserveFloats)
		}
		if cap(cache.Kraw[l]) < wantReserveFloats {
			t.Fatalf("nonhybrid layer %d: Kraw cap=%d, want >= %d", l, cap(cache.Kraw[l]), wantReserveFloats)
		}
		if cap(cache.V[l]) < wantReserveFloats {
			t.Fatalf("nonhybrid layer %d: V cap=%d, want >= %d", l, cap(cache.V[l]), wantReserveFloats)
		}
	}

	clone := cache.CloneWithReserve(extraPositions)
	for l := 0; l < cfg.NumLayers; l++ {
		if cap(clone.K[l]) < wantReserveFloats {
			t.Fatalf("nonhybrid clone layer %d: K cap=%d, want >= %d", l, cap(clone.K[l]), wantReserveFloats)
		}
		if cap(clone.Kraw[l]) < wantReserveFloats {
			t.Fatalf("nonhybrid clone layer %d: Kraw cap=%d, want >= %d", l, cap(clone.Kraw[l]), wantReserveFloats)
		}
		if cap(clone.V[l]) < wantReserveFloats {
			t.Fatalf("nonhybrid clone layer %d: V cap=%d, want >= %d", l, cap(clone.V[l]), wantReserveFloats)
		}
	}
}
