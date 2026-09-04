package model

import (
	"fmt"
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func newTestQwen38HybridConfig() Config {
	layerTypes := []string{
		"linear_attention",
		"linear_attention",
		"full_attention",
		"linear_attention",
	}
	return Config{
		ModelType:             "qwen3_5_text",
		NumLayers:             len(layerTypes),
		NumKVHeads:            2,
		HeadDim:               64,
		LayerTypes:            layerTypes,
		LinearConvKernelDim:   4,
		LinearNumKeyHeads:     4,
		LinearNumValueHeads:   8,
		LinearKeyHeadDim:      32,
		LinearValueHeadDim:    32,
		FullAttentionInterval: 4,
	}
}

func newTestQwen38ModelConfig() Config {
	layerTypes := []string{
		"linear_attention",
		"linear_attention",
		"full_attention",
		"linear_attention",
	}
	return Config{
		ModelType:             "qwen3_5_text",
		HiddenSize:            32,
		NumLayers:             len(layerTypes),
		NumHeads:              4,
		NumKVHeads:            2,
		HeadDim:               8,
		IntermediateSize:      64,
		VocabSize:             97,
		RMSNormEps:            1e-5,
		RopeTheta:             10000,
		TieWordEmbeddings:     true,
		EOSTokenID:            -1,
		LayerTypes:            layerTypes,
		LinearConvKernelDim:   4,
		LinearKeyHeadDim:      8,
		LinearNumKeyHeads:     2,
		LinearValueHeadDim:    8,
		LinearNumValueHeads:   4,
		AttnOutputGate:        true,
		FullAttentionInterval: 4,
		NormGain1p:            true,
	}
}

func newTestSyntheticCache(cfg Config, tokens int) *KVCache {
	cache := NewKVCache(cfg)
	cache.pos = make([]int, tokens)
	for i := range cache.pos {
		cache.pos[i] = i * 2
	}

	stride := cache.kvStride()
	for l := 0; l < cfg.NumLayers; l++ {
		if !cfg.isLinearAttnLayer(l) {
			cache.K[l] = make([]float32, tokens*stride)
			cache.Kraw[l] = make([]float32, tokens*stride)
			cache.V[l] = make([]float32, tokens*stride)
			for i := range cache.K[l] {
				cache.K[l][i] = float32(l*1000+i) * 0.125
				cache.Kraw[l][i] = float32(l*1000+i) * 0.25
				cache.V[l][i] = float32(l*1000+i) * 0.5
			}
		}
	}

	if cache.linear == nil {
		cache.linear = &linearAttnCache{layers: make([]linearAttnLayerState, cfg.NumLayers)}
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			convRows := cfg.LinearConvKernelDim - 1
			convDim := 32
			if _, _, _, _, _, _, cd := cfg.linearAttnDims(); cd > 0 {
				convDim = cd
			}
			cache.linear.layers[l].conv = make([][]float32, convRows)
			for r := 0; r < convRows; r++ {
				row := make([]float32, convDim)
				for i := range row {
					row[i] = float32(l*100+r*10+i) * 0.05
				}
				cache.linear.layers[l].conv[r] = row
			}

			recHeads := cfg.LinearNumValueHeads
			recHeadSize := cfg.LinearKeyHeadDim * cfg.LinearValueHeadDim
			if recHeadSize <= 0 {
				recHeadSize = 32
			}
			cache.linear.layers[l].recurrent = make([][]float32, recHeads)
			for h := 0; h < recHeads; h++ {
				rec := make([]float32, recHeadSize)
				for i := range rec {
					rec[i] = float32(l*500+h*50+i) * 0.01
				}
				cache.linear.layers[l].recurrent[h] = rec
			}
		}
	}
	return cache
}

func assertFloat32ExactBits(t *testing.T, label string, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: length mismatch: want %d, got %d", label, len(want), len(got))
	}
	for i := range want {
		wBits := math.Float32bits(want[i])
		gBits := math.Float32bits(got[i])
		if wBits != gBits {
			t.Fatalf("%s[%d]: bit mismatch: want 0x%08x (%g), got 0x%08x (%g)", label, i, wBits, want[i], gBits, got[i])
		}
	}
}

func assertCachesBitExact(t *testing.T, desc *Qwen38GPUDirectDescriptor, want, got *KVCache) {
	t.Helper()
	if want.Len() != got.Len() {
		t.Fatalf("cache token count mismatch: want %d, got %d", want.Len(), got.Len())
	}
	if !reflect.DeepEqual(want.pos, got.pos) {
		t.Fatalf("cache token positions mismatch: want %v, got %v", want.pos, got.pos)
	}
	for _, l := range desc.FullLayers {
		assertFloat32ExactBits(t, fmt.Sprintf("layer %d K", l), want.K[l], got.K[l])
		assertFloat32ExactBits(t, fmt.Sprintf("layer %d Kraw", l), want.Kraw[l], got.Kraw[l])
		assertFloat32ExactBits(t, fmt.Sprintf("layer %d V", l), want.V[l], got.V[l])
	}
	if want.linear == nil || got.linear == nil {
		t.Fatalf("linear cache state is nil")
	}
	for l := 0; l < len(want.linear.layers); l++ {
		wl := &want.linear.layers[l]
		gl := &got.linear.layers[l]
		if len(wl.conv) != len(gl.conv) {
			t.Fatalf("layer %d conv row count mismatch: want %d, got %d", l, len(wl.conv), len(gl.conv))
		}
		for r := range wl.conv {
			assertFloat32ExactBits(t, fmt.Sprintf("layer %d conv[%d]", l, r), wl.conv[r], gl.conv[r])
		}
		if len(wl.recurrent) != len(gl.recurrent) {
			t.Fatalf("layer %d recurrent head count mismatch: want %d, got %d", l, len(wl.recurrent), len(gl.recurrent))
		}
		for h := range wl.recurrent {
			assertFloat32ExactBits(t, fmt.Sprintf("layer %d recurrent[%d]", l, h), wl.recurrent[h], gl.recurrent[h])
		}
	}
}

func TestQwen38GPUDirect_RoundTripStateExact(t *testing.T) {
	hal := compute.NewAMDGPUDirectHAL(compute.AMDGPUDirectConfig{})
	slab, err := compute.NewDirectStorageMemorySlab(hal, 0, 64*1024, 32, 0x8000000000)
	if err != nil {
		t.Fatalf("NewDirectStorageMemorySlab failed: %v", err)
	}

	cfg := newTestQwen38HybridConfig()
	engine, err := NewQwen38GPUDirectSwapper(slab, cfg, 16)
	if err != nil {
		t.Fatalf("NewQwen38GPUDirectSwapper failed: %v", err)
	}

	// 24 tokens spans 2 blocks (16 in block 0, 8 in block 1)
	cache := newTestSyntheticCache(cfg, 24)

	desc, err := engine.SwapOutDirect(cache, "session-roundtrip-1")
	if err != nil {
		t.Fatalf("SwapOutDirect failed: %v", err)
	}

	if desc.Magic != Qwen38GPUDirectSwapMagic {
		t.Errorf("magic mismatch: got %s, want %s", desc.Magic, Qwen38GPUDirectSwapMagic)
	}
	if desc.SessionID != "session-roundtrip-1" {
		t.Errorf("session ID mismatch: got %s", desc.SessionID)
	}
	if desc.TokenCount != 24 {
		t.Errorf("token count mismatch: got %d, want 24", desc.TokenCount)
	}
	if desc.BlockTokens != 16 {
		t.Errorf("block tokens mismatch: got %d, want 16", desc.BlockTokens)
	}
	if desc.StagingCopyCount() != 0 {
		t.Errorf("expected 0 staging copies, got %d", desc.StagingCopyCount())
	}
	if desc.TotalBytes() == 0 {
		t.Errorf("expected non-zero total bytes")
	}
	if len(desc.KVBlocks) != 2 {
		t.Errorf("expected 2 KV blocks, got %d", len(desc.KVBlocks))
	}
	if desc.GDNConvBytes == 0 {
		t.Errorf("expected non-zero GDN conv bytes")
	}
	if desc.GDNRecurrentBytes == 0 {
		t.Errorf("expected non-zero GDN recurrent bytes")
	}

	// Swap In from NVMe directly
	restored, err := engine.SwapInDirect(desc)
	if err != nil {
		t.Fatalf("SwapInDirect failed: %v", err)
	}

	if restored.Len() != cache.Len() {
		t.Fatalf("restored token length mismatch: got %d, want %d", restored.Len(), cache.Len())
	}

	// Assert exact position equivalence
	if !reflect.DeepEqual(restored.pos, cache.pos) {
		t.Fatalf("token positions mismatch: got %v, want %v", restored.pos, cache.pos)
	}

	// Assert exact float equivalence across full attention planes
	for _, l := range desc.FullLayers {
		if !reflect.DeepEqual(restored.K[l], cache.K[l]) {
			t.Fatalf("layer %d K plane mismatch", l)
		}
		if !reflect.DeepEqual(restored.Kraw[l], cache.Kraw[l]) {
			t.Fatalf("layer %d Kraw plane mismatch", l)
		}
		if !reflect.DeepEqual(restored.V[l], cache.V[l]) {
			t.Fatalf("layer %d V plane mismatch", l)
		}
	}

	// Assert exact equivalence across linear GDN conv and recurrent states
	if restored.linear == nil || cache.linear == nil {
		t.Fatalf("linear cache is nil")
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			if !reflect.DeepEqual(restored.linear.layers[l].conv, cache.linear.layers[l].conv) {
				t.Fatalf("layer %d linear conv mismatch", l)
			}
			if !reflect.DeepEqual(restored.linear.layers[l].recurrent, cache.linear.layers[l].recurrent) {
				t.Fatalf("layer %d linear recurrent mismatch", l)
			}
		}
	}

	stats := engine.Stats()
	if stats.SwapsOut != 1 || stats.SwapsIn != 1 {
		t.Errorf("unexpected stats: %+v", stats)
	}
	if stats.ZeroCopyAssertions < 2 {
		t.Errorf("expected at least 2 zero-copy assertions, got %d", stats.ZeroCopyAssertions)
	}
}

func TestQwen38GPUDirect_PrefetchAcceleration(t *testing.T) {
	hal := compute.NewAMDGPUDirectHAL(compute.AMDGPUDirectConfig{})
	slab, err := compute.NewDirectStorageMemorySlab(hal, 0, 64*1024, 32, 0x8000000000)
	if err != nil {
		t.Fatalf("NewDirectStorageMemorySlab failed: %v", err)
	}

	cfg := newTestQwen38HybridConfig()
	engine, err := NewQwen38GPUDirectSwapper(slab, cfg, 16)
	if err != nil {
		t.Fatalf("NewQwen38GPUDirectSwapper failed: %v", err)
	}

	cache := newTestSyntheticCache(cfg, 32)

	desc, err := engine.SwapOutDirect(cache, "session-prefetch-1")
	if err != nil {
		t.Fatalf("SwapOutDirect failed: %v", err)
	}

	// Evict slab blocks from VRAM so the cache is cold in memory, backing store is NVMe
	if err := engine.ReleaseSlabBlocks(desc); err != nil {
		t.Fatalf("ReleaseSlabBlocks failed: %v", err)
	}

	slabCold := slab.Stats()
	if slabCold.Allocated != 0 {
		t.Fatalf("expected 0 allocated blocks after eviction, got %d", slabCold.Allocated)
	}

	// Prefetch descriptor asynchronously to warm the slab cache
	done := engine.PrefetchDescriptor(desc)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("PrefetchDescriptor failed: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("PrefetchDescriptor timed out")
	}

	// Verify slab cache is warmed (blocks pre-read into VRAM)
	slabWarmed := slab.Stats()
	if slabWarmed.Allocated == 0 {
		t.Fatalf("expected blocks allocated after prefetch warming, got 0")
	}

	// Readmit cache via SwapIn: should hit the warmed slab cache
	restored, err := engine.SwapInDirect(desc)
	if err != nil {
		t.Fatalf("SwapInDirect failed: %v", err)
	}

	stats := engine.Stats()
	if stats.PrefetchHits <= 0 {
		t.Errorf("expected prefetch hits > 0, got %d", stats.PrefetchHits)
	}
	if slab.Stats().CacheHits <= 0 {
		t.Errorf("expected slab cache hits > 0, got %d", slab.Stats().CacheHits)
	}
	if desc.StagingCopyCount() != 0 {
		t.Errorf("expected 0 staging copies, got %d", desc.StagingCopyCount())
	}

	// Verify state integrity
	if !reflect.DeepEqual(restored.pos, cache.pos) {
		t.Errorf("restored pos mismatch")
	}
	for _, l := range desc.FullLayers {
		if !reflect.DeepEqual(restored.K[l], cache.K[l]) {
			t.Errorf("layer %d K mismatch", l)
		}
	}
}

func TestQwen38GPUDirect_DescriptorFreeAndReuse(t *testing.T) {
	hal := compute.NewAMDGPUDirectHAL(compute.AMDGPUDirectConfig{})
	// Restrict slab to 8 blocks to prove freeing works and prevents exhaustion
	slab, err := compute.NewDirectStorageMemorySlab(hal, 0, 64*1024, 8, 0x8000000000)
	if err != nil {
		t.Fatalf("NewDirectStorageMemorySlab failed: %v", err)
	}

	cfg := newTestQwen38HybridConfig()
	engine, err := NewQwen38GPUDirectSwapper(slab, cfg, 16)
	if err != nil {
		t.Fatalf("NewQwen38GPUDirectSwapper failed: %v", err)
	}

	cache := newTestSyntheticCache(cfg, 32) // 2 KV blocks + 1 conv + 1 rec = 4 slab blocks

	// First swap out
	desc1, err := engine.SwapOutDirect(cache, "session-reuse-1")
	if err != nil {
		t.Fatalf("SwapOutDirect 1 failed: %v", err)
	}

	slabStats1 := slab.Stats()
	if slabStats1.Allocated != 4 {
		t.Fatalf("expected 4 allocated slab blocks, got %d", slabStats1.Allocated)
	}

	// Collect LBAs from desc1
	lbas1 := make(map[uint64]bool)
	for _, b := range desc1.KVBlocks {
		lbas1[b.NVMeLBA] = true
	}
	if desc1.GDNConvBytes > 0 {
		lbas1[desc1.GDNConvLBA] = true
	}
	if desc1.GDNRecurrentBytes > 0 {
		lbas1[desc1.GDNRecurrentLBA] = true
	}

	// Free descriptor 1
	engine.FreeDescriptor(desc1)

	slabStatsFreed := slab.Stats()
	if slabStatsFreed.Allocated != 0 {
		t.Fatalf("expected 0 allocated slab blocks after FreeDescriptor, got %d", slabStatsFreed.Allocated)
	}

	// Second swap out: should reuse the freed slab blocks and LBAs without exhaustion error
	desc2, err := engine.SwapOutDirect(cache, "session-reuse-2")
	if err != nil {
		t.Fatalf("SwapOutDirect 2 failed: %v", err)
	}

	// Verify LBA reuse
	for _, b := range desc2.KVBlocks {
		if !lbas1[b.NVMeLBA] {
			t.Errorf("expected KV block LBA %d to be reused from desc1", b.NVMeLBA)
		}
	}
	if desc2.GDNConvBytes > 0 && !lbas1[desc2.GDNConvLBA] {
		t.Errorf("expected conv LBA %d to be reused from desc1", desc2.GDNConvLBA)
	}
	if desc2.GDNRecurrentBytes > 0 && !lbas1[desc2.GDNRecurrentLBA] {
		t.Errorf("expected recurrent LBA %d to be reused from desc1", desc2.GDNRecurrentLBA)
	}

	// Clean up desc2
	engine.FreeDescriptor(desc2)
	slabStatsFinal := slab.Stats()
	if slabStatsFinal.Allocated != 0 {
		t.Fatalf("expected 0 allocated slab blocks after final FreeDescriptor, got %d", slabStatsFinal.Allocated)
	}
}

func TestQwen38GPUDirect_LongContextParity(t *testing.T) {
	hal := compute.NewAMDGPUDirectHAL(compute.AMDGPUDirectConfig{})
	cfg := newTestQwen38ModelConfig()
	m := NewSynthetic(cfg)

	// Test sequence lengths spanning across multiple blocks: 512, 1024, and 2048 tokens.
	seqLengths := []int{512, 1024, 2048}

	for _, seqLen := range seqLengths {
		t.Run(fmt.Sprintf("tokens=%d", seqLen), func(t *testing.T) {
			// For 2048 tokens with blockTokens=16, we need 128 KV blocks + 2 GDN blocks = 130 blocks.
			// Allocate slab with 512 blocks (64 KiB each = 32 MiB).
			slab, err := compute.NewDirectStorageMemorySlab(hal, 0, 64*1024, 512, 0x8000000000)
			if err != nil {
				t.Fatalf("NewDirectStorageMemorySlab failed: %v", err)
			}

			engine, err := NewQwen38GPUDirectSwapper(slab, cfg, 16)
			if err != nil {
				t.Fatalf("NewQwen38GPUDirectSwapper failed: %v", err)
			}

			cache := newTestSyntheticCache(cfg, seqLen)

			desc, err := engine.SwapOutDirect(cache, fmt.Sprintf("session-longcontext-%d", seqLen))
			if err != nil {
				t.Fatalf("SwapOutDirect failed for seqLen %d: %v", seqLen, err)
			}

			if desc.TokenCount != seqLen {
				t.Errorf("token count mismatch: got %d, want %d", desc.TokenCount, seqLen)
			}
			expectedBlocks := (seqLen + 15) / 16
			if len(desc.KVBlocks) != expectedBlocks {
				t.Errorf("KV block count mismatch: got %d, want %d", len(desc.KVBlocks), expectedBlocks)
			}
			if desc.StagingCopyCount() != 0 {
				t.Errorf("expected 0 host staging copies, got %d", desc.StagingCopyCount())
			}

			// Evict slab blocks from VRAM so backing store is NVMe only
			if err := engine.ReleaseSlabBlocks(desc); err != nil {
				t.Fatalf("ReleaseSlabBlocks failed: %v", err)
			}
			if slab.Stats().Allocated != 0 {
				t.Fatalf("expected 0 allocated slab blocks after eviction, got %d", slab.Stats().Allocated)
			}

			// Prefetch descriptor to warm slab cache
			done := engine.PrefetchDescriptor(desc)
			if err := <-done; err != nil {
				t.Fatalf("PrefetchDescriptor failed: %v", err)
			}

			// Swap back into GPU VRAM
			restored, err := engine.SwapInDirect(desc)
			if err != nil {
				t.Fatalf("SwapInDirect failed for seqLen %d: %v", seqLen, err)
			}

			// Assert 100% bit-exact parity across full cache
			assertCachesBitExact(t, desc, cache, restored)

			// Step continuation test: unswapped baseline vs swapped session
			baseSession := &Session{M: m, Cache: cache}
			restoredSession := &Session{M: m, Cache: restored}

			stepTok := 42
			logitsBase := baseSession.Step(stepTok)
			logitsRestored := restoredSession.Step(stepTok)

			assertFloat32ExactBits(t, fmt.Sprintf("seqLen %d continuation logits", seqLen), logitsBase, logitsRestored)

			tokBase := argmaxF32(logitsBase)
			tokRestored := argmaxF32(logitsRestored)
			if tokBase != tokRestored {
				t.Fatalf("seqLen %d argmax token mismatch: base=%d, restored=%d", seqLen, tokBase, tokRestored)
			}

			engine.FreeDescriptor(desc)
			if slab.Stats().Allocated != 0 {
				t.Fatalf("expected 0 allocated slab blocks after FreeDescriptor, got %d", slab.Stats().Allocated)
			}
		})
	}
}

func TestQwen38GPUDirect_TokenGenerationExactMatch(t *testing.T) {
	hal := compute.NewAMDGPUDirectHAL(compute.AMDGPUDirectConfig{})
	slab, err := compute.NewDirectStorageMemorySlab(hal, 0, 64*1024, 64, 0x8000000000)
	if err != nil {
		t.Fatalf("NewDirectStorageMemorySlab failed: %v", err)
	}

	cfg := newTestQwen38ModelConfig()
	m := NewSynthetic(cfg)

	engine, err := NewQwen38GPUDirectSwapper(slab, cfg, 16)
	if err != nil {
		t.Fatalf("NewQwen38GPUDirectSwapper failed: %v", err)
	}

	// 24-token prompt prefill (spans 2 blocks: 16 tokens in block 0, 8 tokens in block 1)
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 2, 4}

	sessionA := m.NewSession()
	sessionB := m.NewSession()

	logitsA := sessionA.Prefill(prompt)
	logitsB := sessionB.Prefill(prompt)

	assertFloat32ExactBits(t, "prefill logits", logitsA, logitsB)

	// Session B is swapped out to NVMe via GPU Direct
	desc, err := engine.SwapOutDirect(sessionB.Cache, "session-b-tokengen")
	if err != nil {
		t.Fatalf("SwapOutDirect failed: %v", err)
	}
	if desc.StagingCopyCount() != 0 {
		t.Errorf("staging copy count = %d, want 0", desc.StagingCopyCount())
	}
	if len(desc.KVBlocks) != 2 {
		t.Fatalf("expected 2 KV blocks, got %d", len(desc.KVBlocks))
	}

	// Evict slab blocks from VRAM slab
	if err := engine.ReleaseSlabBlocks(desc); err != nil {
		t.Fatalf("ReleaseSlabBlocks failed: %v", err)
	}
	if slab.Stats().Allocated != 0 {
		t.Fatalf("expected 0 allocated slab blocks after eviction, got %d", slab.Stats().Allocated)
	}

	// Clear sessionB cache pointer (adversarial proof: no lingering VRAM/host pointer)
	sessionB.Cache = nil

	// Prefetch descriptor and swap back in via GPU Direct
	done := engine.PrefetchDescriptor(desc)
	if err := <-done; err != nil {
		t.Fatalf("PrefetchDescriptor failed: %v", err)
	}

	restored, err := engine.SwapInDirect(desc)
	if err != nil {
		t.Fatalf("SwapInDirect failed: %v", err)
	}
	sessionB.Cache = restored

	// Assert that swapped-in cache is 100% bit-exact and byte-exact to Session A's cache
	assertCachesBitExact(t, desc, sessionA.Cache, sessionB.Cache)

	// Generate 32 tokens on both sessions: Session A unswapped, Session B swapped-in
	tokA := argmaxF32(logitsA)
	tokB := argmaxF32(logitsB)
	if tokA != tokB {
		t.Fatalf("initial token argmax mismatch: tokA=%d tokB=%d", tokA, tokB)
	}

	tokensA := make([]int, 0, 32)
	tokensB := make([]int, 0, 32)

	for step := 0; step < 32; step++ {
		tokensA = append(tokensA, tokA)
		tokensB = append(tokensB, tokB)

		stepLogitsA := sessionA.Step(tokA)
		stepLogitsB := sessionB.Step(tokB)

		// Assert that floating-point logits match down to the exact bit!
		assertFloat32ExactBits(t, fmt.Sprintf("step %d continuation logits", step), stepLogitsA, stepLogitsB)

		tokA = argmaxF32(stepLogitsA)
		tokB = argmaxF32(stepLogitsB)
		if tokA != tokB {
			t.Fatalf("step %d generated token ID mismatch: tokA=%d tokB=%d", step, tokA, tokB)
		}
	}

	if !reflect.DeepEqual(tokensA, tokensB) {
		t.Fatalf("generated token stream mismatch: tokensA=%v tokensB=%v", tokensA, tokensB)
	}

	// Final verification: post-generation cache state must remain bit-exact
	assertCachesBitExact(t, desc, sessionA.Cache, sessionB.Cache)

	engine.FreeDescriptor(desc)
}

func TestQwen38GPUDirect_MultipleInterleavedSwaps(t *testing.T) {
	hal := compute.NewAMDGPUDirectHAL(compute.AMDGPUDirectConfig{})
	slab, err := compute.NewDirectStorageMemorySlab(hal, 0, 64*1024, 64, 0x8000000000)
	if err != nil {
		t.Fatalf("NewDirectStorageMemorySlab failed: %v", err)
	}

	cfg := newTestQwen38ModelConfig()
	m := NewSynthetic(cfg)

	engine, err := NewQwen38GPUDirectSwapper(slab, cfg, 16)
	if err != nil {
		t.Fatalf("NewQwen38GPUDirectSwapper failed: %v", err)
	}

	prompt := []int{5, 11, 13, 17, 23, 29, 31, 37, 41, 43, 47, 53}

	sessionA := m.NewSession()
	sessionB := m.NewSession()

	logitsA := sessionA.Prefill(prompt)
	logitsB := sessionB.Prefill(prompt)

	assertFloat32ExactBits(t, "prefill logits", logitsA, logitsB)

	tokA := argmaxF32(logitsA)
	tokB := argmaxF32(logitsB)
	if tokA != tokB {
		t.Fatalf("initial token argmax mismatch: tokA=%d tokB=%d", tokA, tokB)
	}

	const numRounds = 8
	const tokensPerRound = 4

	for round := 0; round < numRounds; round++ {
		// Generate tokens for this round
		for s := 0; s < tokensPerRound; s++ {
			stepLogitsA := sessionA.Step(tokA)
			stepLogitsB := sessionB.Step(tokB)

			assertFloat32ExactBits(t, fmt.Sprintf("round %d step %d logits", round, s), stepLogitsA, stepLogitsB)

			tokA = argmaxF32(stepLogitsA)
			tokB = argmaxF32(stepLogitsB)
			if tokA != tokB {
				t.Fatalf("round %d step %d token ID mismatch: tokA=%d, tokB=%d", round, s, tokA, tokB)
			}
		}

		// Swap out Session B to NVMe
		desc, err := engine.SwapOutDirect(sessionB.Cache, fmt.Sprintf("session-interleaved-%d", round))
		if err != nil {
			t.Fatalf("round %d SwapOutDirect failed: %v", round, err)
		}
		if desc.StagingCopyCount() != 0 {
			t.Errorf("round %d staging copy count = %d, want 0", round, desc.StagingCopyCount())
		}

		// Evict from VRAM slab
		if err := engine.ReleaseSlabBlocks(desc); err != nil {
			t.Fatalf("round %d ReleaseSlabBlocks failed: %v", round, err)
		}
		if slab.Stats().Allocated != 0 {
			t.Fatalf("round %d expected 0 allocated slab blocks, got %d", round, slab.Stats().Allocated)
		}
		sessionB.Cache = nil // wipe pointer

		// Alternating prefetch
		if round%2 == 0 {
			done := engine.PrefetchDescriptor(desc)
			if err := <-done; err != nil {
				t.Fatalf("round %d PrefetchDescriptor failed: %v", round, err)
			}
		}

		// Swap back in
		restored, err := engine.SwapInDirect(desc)
		if err != nil {
			t.Fatalf("round %d SwapInDirect failed: %v", round, err)
		}
		sessionB.Cache = restored

		// Verify 100% bit-exact cache match after swap-in
		assertCachesBitExact(t, desc, sessionA.Cache, sessionB.Cache)

		// Free descriptor for next round's reuse
		engine.FreeDescriptor(desc)
	}

	stats := engine.Stats()
	if stats.SwapsOut != numRounds {
		t.Errorf("expected %d swaps out, got %d", numRounds, stats.SwapsOut)
	}
	if stats.SwapsIn != numRounds {
		t.Errorf("expected %d swaps in, got %d", numRounds, stats.SwapsIn)
	}
	if stats.ZeroCopyAssertions < numRounds*2 {
		t.Errorf("expected at least %d zero-copy assertions, got %d", numRounds*2, stats.ZeroCopyAssertions)
	}
}

func TestQwen38GPUDirect_AdversarialBitCorruptionDetection(t *testing.T) {
	hal := compute.NewAMDGPUDirectHAL(compute.AMDGPUDirectConfig{})
	slab, err := compute.NewDirectStorageMemorySlab(hal, 0, 64*1024, 32, 0x8000000000)
	if err != nil {
		t.Fatalf("NewDirectStorageMemorySlab failed: %v", err)
	}

	cfg := newTestQwen38ModelConfig()
	engine, err := NewQwen38GPUDirectSwapper(slab, cfg, 16)
	if err != nil {
		t.Fatalf("NewQwen38GPUDirectSwapper failed: %v", err)
	}

	cache := newTestSyntheticCache(cfg, 16)
	desc, err := engine.SwapOutDirect(cache, "session-adversarial")
	if err != nil {
		t.Fatalf("SwapOutDirect failed: %v", err)
	}

	// 1. Corrupt magic
	badMagicDesc := *desc
	badMagicDesc.Magic = "CORRUPTMAGIC"
	if _, err := engine.SwapInDirect(&badMagicDesc); err == nil {
		t.Errorf("expected error for corrupted magic, got nil")
	}

	// 2. Corrupt a single bit in KV block payload and assert it fails bit-exact check
	kvLBA := desc.KVBlocks[0].NVMeLBA
	origKVPayload := append([]byte(nil), engine.nvmeStorage[kvLBA]...)

	// Flip 1 bit in float data (after 4 byte count + 16*8 pos = offset 132)
	corruptKVPayload := append([]byte(nil), origKVPayload...)
	corruptKVPayload[132] ^= 0x01
	engine.nvmeStorage[kvLBA] = corruptKVPayload

	corruptKVCache, err := engine.SwapInDirect(desc)
	if err != nil {
		t.Fatalf("SwapIn failed: %v", err)
	}

	// Verify bit mismatch is detected
	bitDiffFound := false
	for _, l := range desc.FullLayers {
		for i := range cache.K[l] {
			if math.Float32bits(cache.K[l][i]) != math.Float32bits(corruptKVCache.K[l][i]) {
				bitDiffFound = true
				break
			}
		}
	}
	if !bitDiffFound {
		t.Errorf("expected 1-bit corruption in K plane to be detected, but all bits matched")
	}
	// Restore clean payload
	engine.nvmeStorage[kvLBA] = origKVPayload

	// 3. Corrupt a single bit in GDN conv payload
	if desc.GDNConvBytes > 0 {
		origConvPayload := append([]byte(nil), engine.nvmeStorage[desc.GDNConvLBA]...)
		corruptConvPayload := append([]byte(nil), origConvPayload...)
		corruptConvPayload[len(corruptConvPayload)-1] ^= 0x01
		engine.nvmeStorage[desc.GDNConvLBA] = corruptConvPayload

		corruptConvCache, err := engine.SwapInDirect(desc)
		if err != nil {
			t.Fatalf("SwapIn conv failed: %v", err)
		}
		convDiffFound := false
		for l := range cache.linear.layers {
			for r := range cache.linear.layers[l].conv {
				for c := range cache.linear.layers[l].conv[r] {
					if math.Float32bits(cache.linear.layers[l].conv[r][c]) != math.Float32bits(corruptConvCache.linear.layers[l].conv[r][c]) {
						convDiffFound = true
						break
					}
				}
			}
		}
		if !convDiffFound {
			t.Errorf("expected 1-bit corruption in conv state to be detected")
		}
		engine.nvmeStorage[desc.GDNConvLBA] = origConvPayload
	}

	// 4. Truncate payload and verify graceful error refusal
	engine.nvmeStorage[kvLBA] = origKVPayload[:10]
	if _, err := engine.SwapInDirect(desc); err == nil {
		t.Errorf("expected error on truncated KV payload, got nil")
	}
	engine.nvmeStorage[kvLBA] = origKVPayload

	// 5. Clean up
	engine.FreeDescriptor(desc)
}
