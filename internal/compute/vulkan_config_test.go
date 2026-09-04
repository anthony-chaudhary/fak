package compute

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"
)

type recordingVulkanQ4KBackend struct {
	Backend
	profile bool
	stage   bool
}

func (b *recordingVulkanQ4KBackend) configureVulkanQ4K(profile, stage bool) {
	b.profile, b.stage = profile, stage
}

func TestConfigureVulkanQ4KReachesSelectedBackend(t *testing.T) {
	b := &recordingVulkanQ4KBackend{}
	if !ConfigureVulkanQ4K(b, true, true) || !b.profile || !b.stage {
		t.Fatalf("Vulkan Q4_K config did not reach backend: %+v", b)
	}
	if ConfigureVulkanQ4K(nil, true, true) {
		t.Fatal("nil backend accepted Vulkan Q4_K config")
	}
}

func TestQuantizeDraftTokenLength(t *testing.T) {
	tests := []struct {
		input int
		want  int
	}{
		// Non-positive bounds: <= 0 -> 1
		{-100, 1},
		{-10, 1},
		{-1, 1},
		{0, 1},
		// 1..64 power-of-two boundaries
		{1, 1},
		{2, 2},
		{3, 4},
		{4, 4},
		{5, 8},
		{6, 8},
		{7, 8},
		{8, 8},
		{9, 16},
		{15, 16},
		{16, 16},
		{17, 32},
		{31, 32},
		{32, 32},
		{33, 64},
		{63, 64},
		{64, 64},
		// Upper clamp: > 64 -> 64
		{65, 64},
		{100, 64},
		{128, 64},
		{512, 64},
		{1024, 64},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("input_%d", tc.input), func(t *testing.T) {
			got := QuantizeDraftTokenLength(tc.input)
			if got != tc.want {
				t.Fatalf("QuantizeDraftTokenLength(%d) = %d, want %d", tc.input, got, tc.want)
			}
		})
	}

	// Invariant verification across full 1..64 range:
	// Output must be >= input, <= 64, and a power of two.
	for n := 1; n <= 64; n++ {
		q := QuantizeDraftTokenLength(n)
		if q < n {
			t.Fatalf("QuantizeDraftTokenLength(%d) = %d, expected >= %d", n, q, n)
		}
		if q > 64 {
			t.Fatalf("QuantizeDraftTokenLength(%d) = %d, expected <= 64", n, q)
		}
		if q <= 0 || (q&(q-1)) != 0 {
			t.Fatalf("QuantizeDraftTokenLength(%d) = %d is not a power of two", n, q)
		}
	}
}

func TestPowerOfTwoGraphCache_HitMissAndOperations(t *testing.T) {
	cache := NewPowerOfTwoGraphCache(10)

	key16 := NewGraphCacheKey(512, 16)
	if val, ok := cache.Get(key16); ok || val != nil {
		t.Fatalf("expected cache miss on empty cache, got val=%v ok=%v", val, ok)
	}

	mockGraph16 := "vulkan_pipeline_b512_s16"
	if err := cache.Put(key16, mockGraph16); err != nil {
		t.Fatalf("Put failed: %v", err)
	}

	// Exact match hit
	val, ok := cache.Get(key16)
	if !ok || val != mockGraph16 {
		t.Fatalf("Get(%v) = (%v, %v), want (%v, true)", key16, val, ok, mockGraph16)
	}

	// Normalized sequence hit: length 15 normalizes to 16
	key15 := NewGraphCacheKey(512, 15)
	val15, ok15 := cache.Get(key15)
	if !ok15 || val15 != mockGraph16 {
		t.Fatalf("Get(key15) = (%v, %v), want normalized hit (%v, true)", val15, ok15, mockGraph16)
	}

	// Miss on different bucket
	key32 := NewGraphCacheKey(512, 32)
	if _, ok := cache.Get(key32); ok {
		t.Fatalf("expected miss on key32")
	}

	// GetOrCreate: miss creates, subsequent hit reuses
	createCount := 0
	g, hit, err := cache.GetOrCreate(key32, func() (any, error) {
		createCount++
		return "vulkan_pipeline_b512_s32", nil
	})
	if err != nil || hit || g != "vulkan_pipeline_b512_s32" || createCount != 1 {
		t.Fatalf("GetOrCreate miss failed: g=%v hit=%v err=%v count=%d", g, hit, err, createCount)
	}

	gHit, hit2, err2 := cache.GetOrCreate(key32, func() (any, error) {
		createCount++
		return "unexpected", nil
	})
	if err2 != nil || !hit2 || gHit != "vulkan_pipeline_b512_s32" || createCount != 1 {
		t.Fatalf("GetOrCreate hit failed: gHit=%v hit2=%v count=%d", gHit, hit2, createCount)
	}

	stats := cache.Stats()
	if stats.Hits < 2 || stats.Misses < 2 {
		t.Fatalf("unexpected stats: %+v", stats)
	}

	// Delete
	if !cache.Delete(key16) {
		t.Fatalf("Delete(%v) returned false", key16)
	}
	if _, ok := cache.Get(key16); ok {
		t.Fatalf("key16 still present after Delete")
	}

	// Clear
	cache.Clear()
	if cache.Len() != 0 {
		t.Fatalf("Clear failed, Len=%d", cache.Len())
	}

	// Close
	cache.Close()
	if err := cache.Put(key16, "closed_test"); err == nil {
		t.Fatalf("Put on closed cache should fail")
	}
	if _, ok := cache.Get(key16); ok {
		t.Fatalf("Get on closed cache should return false")
	}
}

func TestPowerOfTwoGraphCache_EvictionAndCapacityLimits(t *testing.T) {
	capacity := 3
	cache := NewPowerOfTwoGraphCache(capacity)

	var evictedKeys []GraphCacheKey
	var evictedVals []any
	cache.SetOnEvict(func(key GraphCacheKey, val any) {
		evictedKeys = append(evictedKeys, key)
		evictedVals = append(evictedVals, val)
	})

	k1 := NewGraphCacheKey(512, 1)
	k2 := NewGraphCacheKey(512, 2)
	k4 := NewGraphCacheKey(512, 4)
	k8 := NewGraphCacheKey(512, 8)

	_ = cache.Put(k1, "graph_1")
	_ = cache.Put(k2, "graph_2")
	_ = cache.Put(k4, "graph_4")

	if cache.Len() != 3 {
		t.Fatalf("cache.Len() = %d, want 3", cache.Len())
	}

	// Touch k1 to make it most recently used. Order MRU->LRU is now: k1, k4, k2.
	if _, ok := cache.Get(k1); !ok {
		t.Fatalf("failed to Get k1")
	}

	// Insert k8. k2 should be evicted as the least recently used entry.
	if err := cache.Put(k8, "graph_8"); err != nil {
		t.Fatalf("Put(k8) failed: %v", err)
	}

	if cache.Len() != 3 {
		t.Fatalf("cache.Len() after eviction = %d, want %d", cache.Len(), capacity)
	}

	// k2 must be gone
	if _, ok := cache.Get(k2); ok {
		t.Fatalf("k2 was not evicted as expected")
	}

	// k1, k4, k8 must still exist
	for _, k := range []GraphCacheKey{k1, k4, k8} {
		if _, ok := cache.Get(k); !ok {
			t.Fatalf("key %v missing after eviction", k)
		}
	}

	if len(evictedKeys) != 1 || evictedKeys[0].SeqLen != 2 || evictedVals[0] != "graph_2" {
		t.Fatalf("unexpected eviction callback records: keys=%v vals=%v", evictedKeys, evictedVals)
	}

	st := cache.Stats()
	if st.Evictions != 1 {
		t.Fatalf("Stats.Evictions = %d, want 1", st.Evictions)
	}
	if st.Capacity != 3 {
		t.Fatalf("Stats.Capacity = %d, want 3", st.Capacity)
	}
}

func TestPowerOfTwoGraphCache_ConcurrentSafety(t *testing.T) {
	cache := NewPowerOfTwoGraphCache(8)
	var wg sync.WaitGroup
	workers := 40
	iterations := 100

	for i := 0; i < workers; i++ {
		wg.Add(1)
		workerID := i
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(workerID * 1000)))

			for j := 0; j < iterations; j++ {
				tokenLen := rng.Intn(64) + 1
				key := NewGraphCacheKey(512, tokenLen)

				switch j % 5 {
				case 0, 1:
					_ = cache.Put(key, fmt.Sprintf("graph_%d_%d", workerID, tokenLen))
				case 2:
					_, _ = cache.Get(key)
				case 3:
					_, _, _ = cache.GetOrCreate(key, func() (any, error) {
						return fmt.Sprintf("created_%d", tokenLen), nil
					})
				case 4:
					_ = cache.Stats()
					_ = cache.Len()
				}
			}
		}()
	}

	wg.Wait()

	if cache.Len() > cache.Capacity() {
		t.Fatalf("cache.Len() = %d exceeded capacity %d", cache.Len(), cache.Capacity())
	}
}

type recordingStrixHaloMTPBackend struct {
	Backend
	configured bool
	cfg        StrixHaloMTPConfig
}

func (b *recordingStrixHaloMTPBackend) ConfigureStrixHaloMTP(cfg StrixHaloMTPConfig) error {
	b.configured = true
	b.cfg = cfg
	return nil
}

func TestStrixHaloMTPConfig_SpecDraftUbatchSize(t *testing.T) {
	// Default micro-batch size: 512
	if DefaultSpecDraftUbatchSize != 512 {
		t.Fatalf("DefaultSpecDraftUbatchSize = %d, want 512", DefaultSpecDraftUbatchSize)
	}

	defCfg := DefaultStrixHaloMTPConfig()
	if defCfg.SpecDraftUbatchSize != 512 {
		t.Fatalf("DefaultStrixHaloMTPConfig().SpecDraftUbatchSize = %d, want 512", defCfg.SpecDraftUbatchSize)
	}

	// Custom micro-batch size via functional options
	customCfg := NewStrixHaloMTPConfig(WithSpecDraftUbatchSize(256))
	if customCfg.SpecDraftUbatchSize != 256 {
		t.Fatalf("NewStrixHaloMTPConfig(256) = %d, want 256", customCfg.SpecDraftUbatchSize)
	}

	// Custom micro-batch size via builder
	builderCfg := NewStrixHaloMTPConfigBuilder().
		WithSpecDraftUbatchSize(1024).
		WithGraphCacheCapacity(32).
		WithTargetArch("gfx1151").
		WithPowerOfTwoBucketing(true).
		Build()
	if builderCfg.SpecDraftUbatchSize != 1024 {
		t.Fatalf("builderCfg.SpecDraftUbatchSize = %d, want 1024", builderCfg.SpecDraftUbatchSize)
	}
	if builderCfg.GraphCacheCapacity != 32 {
		t.Fatalf("builderCfg.GraphCacheCapacity = %d, want 32", builderCfg.GraphCacheCapacity)
	}

	// Non-positive micro-batch size falls back to default 512
	fallbackCfg := NewStrixHaloMTPConfig(WithSpecDraftUbatchSize(0))
	if fallbackCfg.SpecDraftUbatchSize != 512 {
		t.Fatalf("fallbackCfg.SpecDraftUbatchSize = %d, want 512", fallbackCfg.SpecDraftUbatchSize)
	}

	// Runtime inherits configured micro-batch size and handles quantized recording
	rt := NewStrixHaloMTPRuntime(customCfg)
	if rt.SpecDraftUbatchSize() != 256 {
		t.Fatalf("runtime.SpecDraftUbatchSize() = %d, want 256", rt.SpecDraftUbatchSize())
	}

	recordedTokens := 0
	g, hit, err := rt.GetOrRecordGraph(5, func(quantizedTokens int) (any, error) {
		recordedTokens = quantizedTokens
		return fmt.Sprintf("graph_q%d", quantizedTokens), nil
	})
	if err != nil || hit || recordedTokens != 8 || g != "graph_q8" {
		t.Fatalf("GetOrRecordGraph initial recording failed: g=%v hit=%v err=%v recorded=%d", g, hit, err, recordedTokens)
	}

	// Subsequent lookup with 7 tokens quantizes to 8 and reuses cached graph without recording
	g2, hit2, err2 := rt.GetOrRecordGraph(7, func(quantizedTokens int) (any, error) {
		t.Fatalf("recordFn should not be called on cache hit")
		return nil, nil
	})
	if err2 != nil || !hit2 || g2 != "graph_q8" {
		t.Fatalf("GetOrRecordGraph reuse failed: g2=%v hit2=%v err2=%v", g2, hit2, err2)
	}

	// Backend configuration delivery
	b := &recordingStrixHaloMTPBackend{}
	if !ConfigureStrixHaloMTP(b, customCfg) || !b.configured || b.cfg.SpecDraftUbatchSize != 256 {
		t.Fatalf("ConfigureStrixHaloMTP did not reach backend properly: %+v", b)
	}
	if ConfigureStrixHaloMTP(nil, customCfg) {
		t.Fatal("nil backend accepted Strix Halo MTP configuration")
	}
}
