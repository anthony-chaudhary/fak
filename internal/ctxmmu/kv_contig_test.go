package ctxmmu

import (
	"bytes"
	"math/rand"
	"sync"
	"testing"
)

func TestContig_LayoutRoundtrip(t *testing.T) {
	testCases := []struct {
		name      string
		numTokens int
		numHeads  int
		headDim   int
	}{
		{name: "SingleBlock_64", numTokens: 64, numHeads: 8, headDim: 128},
		{name: "PartialBlock_100", numTokens: 100, numHeads: 8, headDim: 128},
		{name: "TwoBlocks_128", numTokens: 128, numHeads: 8, headDim: 128},
		{name: "FourBlocks_256", numTokens: 256, numHeads: 4, headDim: 64},
		{name: "Deep_512", numTokens: 512, numHeads: 16, headDim: 128},
		{name: "Tiny_17", numTokens: 17, numHeads: 2, headDim: 32},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tokenBytes := tc.numHeads * tc.headDim * BytesPerF16
			totalBytes := tc.numTokens * tokenBytes

			kOrig := make([]byte, totalBytes)
			vOrig := make([]byte, totalBytes)

			rng := rand.New(rand.NewSource(1337))
			rng.Read(kOrig)
			rng.Read(vOrig)

			cfg := KVContigConfig{
				BlockSizeTokens: ContigBlockSizeTokens,
				NumHeads:        tc.numHeads,
				HeadDim:         tc.headDim,
				SubChannels:     LPDDR5XSubChannels,
			}
			cache := NewContigKVCache(cfg)

			// 1. Transform Strided -> Blocked [N_blocks, 64, N_heads, D_head]
			if err := cache.TransformStridedToBlocked(kOrig, vOrig, tc.numTokens, tc.numHeads, tc.headDim); err != nil {
				t.Fatalf("TransformStridedToBlocked failed: %v", err)
			}

			expectedBlocks := (tc.numTokens + ContigBlockSizeTokens - 1) / ContigBlockSizeTokens
			if cache.NumBlocks() != expectedBlocks {
				t.Errorf("NumBlocks = %d, want %d", cache.NumBlocks(), expectedBlocks)
			}
			if cache.TotalTokens() != tc.numTokens {
				t.Errorf("TotalTokens = %d, want %d", cache.TotalTokens(), tc.numTokens)
			}

			// Verify block metadata
			for b := 0; b < cache.NumBlocks(); b++ {
				blk, err := cache.ReadBlock(b)
				if err != nil {
					t.Fatalf("ReadBlock(%d) failed: %v", b, err)
				}
				if blk.BlockID != b {
					t.Errorf("block %d BlockID = %d", b, blk.BlockID)
				}
				if blk.CapacityTokens != ContigBlockSizeTokens {
					t.Errorf("block %d CapacityTokens = %d, want %d", b, blk.CapacityTokens, ContigBlockSizeTokens)
				}
				expectedCapBytes := ContigBlockSizeTokens * tokenBytes
				if len(blk.KeyData) != expectedCapBytes || len(blk.ValueData) != expectedCapBytes {
					t.Errorf("block %d data capacity mismatch: got k=%d v=%d, want %d",
						b, len(blk.KeyData), len(blk.ValueData), expectedCapBytes)
				}
			}

			// 2. Transform Blocked -> Strided (inverse)
			kRestored, vRestored, err := cache.TransformBlockedToStrided(tc.numTokens)
			if err != nil {
				t.Fatalf("TransformBlockedToStrided failed: %v", err)
			}

			// 3. Assert zero numerical drift (bit-exact identity)
			if !bytes.Equal(kOrig, kRestored) {
				t.Fatalf("Key tensor numerical drift detected: restored bytes do not match original strided bytes")
			}
			if !bytes.Equal(vOrig, vRestored) {
				t.Fatalf("Value tensor numerical drift detected: restored bytes do not match original strided bytes")
			}
		})
	}
}

func TestContig_16ChannelSymmetryAndEntropy(t *testing.T) {
	cfg := KVContigConfig{
		BlockSizeTokens: ContigBlockSizeTokens,
		NumHeads:        8,
		HeadDim:         128,
		SubChannels:     LPDDR5XSubChannels,
	}
	cache := NewContigKVCache(cfg)

	// 1. Contiguous layout: 16 of 16 sub-channels active, entropy ~ 1.0
	contigProfile := cache.ChannelAccessProfile(true)
	if contigProfile.ActiveSubChannels != 16 {
		t.Errorf("contig ActiveSubChannels = %d, want 16", contigProfile.ActiveSubChannels)
	}
	if contigProfile.TotalSubChannels != 16 {
		t.Errorf("contig TotalSubChannels = %d, want 16", contigProfile.TotalSubChannels)
	}
	if contigProfile.IdleSubChannels != 0 {
		t.Errorf("contig IdleSubChannels = %d, want 0", contigProfile.IdleSubChannels)
	}
	if contigProfile.SubChannelUtilization != 1.0 {
		t.Errorf("contig SubChannelUtilization = %.4f, want 1.0", contigProfile.SubChannelUtilization)
	}
	if contigProfile.Entropy < 0.999 {
		t.Errorf("contig Entropy = %.4f, want ~1.0", contigProfile.Entropy)
	}
	if contigProfile.RawEntropy < 3.999 {
		t.Errorf("contig RawEntropy = %.4f, want ~4.0 bits", contigProfile.RawEntropy)
	}
	if contigProfile.ChannelCampingDetected {
		t.Errorf("contig ChannelCampingDetected = true, want false")
	}
	if !contigProfile.IsContiguous {
		t.Errorf("contig IsContiguous = false, want true")
	}

	// 2. Strided layout: 4 of 16 sub-channels active, 12 idled, entropy < 0.25
	stridedProfile := cache.ChannelAccessProfile(false)
	if stridedProfile.ActiveSubChannels != 4 {
		t.Errorf("strided ActiveSubChannels = %d, want 4", stridedProfile.ActiveSubChannels)
	}
	if stridedProfile.TotalSubChannels != 16 {
		t.Errorf("strided TotalSubChannels = %d, want 16", stridedProfile.TotalSubChannels)
	}
	if stridedProfile.IdleSubChannels != 12 {
		t.Errorf("strided IdleSubChannels = %d, want 12", stridedProfile.IdleSubChannels)
	}
	if stridedProfile.SubChannelUtilization != 0.25 {
		t.Errorf("strided SubChannelUtilization = %.4f, want 0.25", stridedProfile.SubChannelUtilization)
	}
	if stridedProfile.Entropy >= 0.25 {
		t.Errorf("strided Entropy = %.4f, want < 0.25 (channel camping threshold)", stridedProfile.Entropy)
	}
	if !stridedProfile.ChannelCampingDetected {
		t.Errorf("strided ChannelCampingDetected = false, want true")
	}
	if stridedProfile.IsContiguous {
		t.Errorf("strided IsContiguous = true, want false")
	}
	if stridedProfile.DominantChannelRatio < 0.90 {
		t.Errorf("strided DominantChannelRatio = %.4f, want >= 0.90", stridedProfile.DominantChannelRatio)
	}

	// Channel counts verification: 12 channels must have 0 counts
	zeroCountChannels := 0
	for _, cnt := range stridedProfile.ChannelCounts {
		if cnt == 0 {
			zeroCountChannels++
		}
	}
	if zeroCountChannels != 12 {
		t.Errorf("strided zero count channels = %d, want 12", zeroCountChannels)
	}
}

func TestContig_BlockAllocationAndAppend(t *testing.T) {
	cfg := KVContigConfig{
		BlockSizeTokens: ContigBlockSizeTokens,
		NumHeads:        8,
		HeadDim:         128,
		SubChannels:     LPDDR5XSubChannels,
	}
	cache := NewContigKVCache(cfg)

	// Step 1: Pre-allocate 128 tokens (2 blocks of 64)
	allocated := cache.AllocateBlocks(128)
	if allocated != 2 {
		t.Fatalf("AllocateBlocks(128) = %d, want 2", allocated)
	}
	if cache.NumBlocks() != 2 {
		t.Fatalf("NumBlocks = %d, want 2", cache.NumBlocks())
	}
	if cache.TotalTokens() != 0 {
		t.Fatalf("TotalTokens initially = %d, want 0", cache.TotalTokens())
	}

	tokenBytes := cfg.NumHeads * cfg.HeadDim * BytesPerF16

	// Step 2: Append 50 tokens (into block 0)
	k50 := make([]byte, 50*tokenBytes)
	v50 := make([]byte, 50*tokenBytes)
	rng := rand.New(rand.NewSource(42))
	rng.Read(k50)
	rng.Read(v50)

	if err := cache.AppendTokens(k50, v50, 50); err != nil {
		t.Fatalf("AppendTokens(50) failed: %v", err)
	}
	if cache.TotalTokens() != 50 {
		t.Errorf("TotalTokens after 50 = %d, want 50", cache.TotalTokens())
	}
	if cache.NumBlocks() != 2 {
		t.Errorf("NumBlocks after 50 = %d, want 2 (pre-allocated)", cache.NumBlocks())
	}

	// Verify ReadToken for token 25
	kTok25, vTok25, err := cache.ReadToken(25)
	if err != nil {
		t.Fatalf("ReadToken(25) failed: %v", err)
	}
	expectedK25 := k50[25*tokenBytes : 26*tokenBytes]
	expectedV25 := v50[25*tokenBytes : 26*tokenBytes]
	if !bytes.Equal(kTok25, expectedK25) {
		t.Errorf("ReadToken(25) key mismatch")
	}
	if !bytes.Equal(vTok25, expectedV25) {
		t.Errorf("ReadToken(25) value mismatch")
	}

	// Step 3: Append 30 tokens (14 into block 0, 16 into block 1)
	k30 := make([]byte, 30*tokenBytes)
	v30 := make([]byte, 30*tokenBytes)
	rng.Read(k30)
	rng.Read(v30)

	if err := cache.AppendTokens(k30, v30, 30); err != nil {
		t.Fatalf("AppendTokens(30) failed: %v", err)
	}
	if cache.TotalTokens() != 80 {
		t.Errorf("TotalTokens after 80 = %d, want 80", cache.TotalTokens())
	}
	if cache.NumBlocks() != 2 {
		t.Errorf("NumBlocks after 80 = %d, want 2", cache.NumBlocks())
	}

	// Block 0 should now be full (64 tokens), Block 1 should have 16 tokens
	b0, _ := cache.ReadBlock(0)
	if b0.NumTokens != 64 {
		t.Errorf("block 0 NumTokens = %d, want 64", b0.NumTokens)
	}
	b1, _ := cache.ReadBlock(1)
	if b1.NumTokens != 16 {
		t.Errorf("block 1 NumTokens = %d, want 16", b1.NumTokens)
	}

	// Step 4: Append 70 tokens (48 into block 1, 22 into new block 2)
	k70 := make([]byte, 70*tokenBytes)
	v70 := make([]byte, 70*tokenBytes)
	rng.Read(k70)
	rng.Read(v70)

	if err := cache.AppendTokens(k70, v70, 70); err != nil {
		t.Fatalf("AppendTokens(70) failed: %v", err)
	}
	if cache.TotalTokens() != 150 {
		t.Errorf("TotalTokens after 150 = %d, want 150", cache.TotalTokens())
	}
	if cache.NumBlocks() != 3 {
		t.Errorf("NumBlocks after 150 = %d, want 3 (new block allocated)", cache.NumBlocks())
	}

	b2, _ := cache.ReadBlock(2)
	if b2.NumTokens != 22 {
		t.Errorf("block 2 NumTokens = %d, want 22", b2.NumTokens)
	}

	// Step 5: Read roundtrip and check bit-exact equality with combined input
	var allK, allV []byte
	allK = append(allK, k50...)
	allK = append(allK, k30...)
	allK = append(allK, k70...)
	allV = append(allV, v50...)
	allV = append(allV, v30...)
	allV = append(allV, v70...)

	restoredK, restoredV, err := cache.TransformBlockedToStrided(150)
	if err != nil {
		t.Fatalf("TransformBlockedToStrided(150) failed: %v", err)
	}
	if !bytes.Equal(allK, restoredK) {
		t.Errorf("Bit-exact roundtrip failed after appends for Key")
	}
	if !bytes.Equal(allV, restoredV) {
		t.Errorf("Bit-exact roundtrip failed after appends for Value")
	}
}

func TestContig_OutOfBoundsValidations(t *testing.T) {
	cfg := KVContigConfig{
		BlockSizeTokens: ContigBlockSizeTokens,
		NumHeads:        8,
		HeadDim:         128,
		SubChannels:     LPDDR5XSubChannels,
	}
	cache := NewContigKVCache(cfg)

	// Block reads on empty cache
	if _, err := cache.ReadBlock(-1); err == nil {
		t.Errorf("ReadBlock(-1) expected error, got nil")
	}
	if _, err := cache.ReadBlock(0); err == nil {
		t.Errorf("ReadBlock(0) on empty cache expected error, got nil")
	}

	// Token reads on empty cache
	if _, _, err := cache.ReadToken(-1); err == nil {
		t.Errorf("ReadToken(-1) expected error, got nil")
	}
	if _, _, err := cache.ReadToken(0); err == nil {
		t.Errorf("ReadToken(0) on empty cache expected error, got nil")
	}

	// Allocate 1 block
	cache.AllocateBlocks(64)
	if _, err := cache.ReadBlock(1); err == nil {
		t.Errorf("ReadBlock(1) out-of-range expected error, got nil")
	}

	// Append negative tokens
	if err := cache.AppendTokens(nil, nil, -1); err == nil {
		t.Errorf("AppendTokens(-1) expected error, got nil")
	}

	// Append buffer length mismatch
	if err := cache.AppendTokens([]byte{1, 2}, []byte{1}, 1); err == nil {
		t.Errorf("AppendTokens with buffer length mismatch expected error, got nil")
	}

	// TransformStridedToBlocked with invalid geometries
	if err := cache.TransformStridedToBlocked(nil, nil, -5, 8, 128); err == nil {
		t.Errorf("TransformStridedToBlocked with negative tokens expected error, got nil")
	}
	if err := cache.TransformStridedToBlocked(nil, nil, 64, 0, 128); err == nil {
		t.Errorf("TransformStridedToBlocked with 0 heads expected error, got nil")
	}
	if err := cache.TransformStridedToBlocked(nil, nil, 64, 8, 0); err == nil {
		t.Errorf("TransformStridedToBlocked with 0 headDim expected error, got nil")
	}
	if err := cache.TransformStridedToBlocked([]byte{1}, []byte{1}, 64, 8, 128); err == nil {
		t.Errorf("TransformStridedToBlocked with invalid buffer length expected error, got nil")
	}

	// TransformBlockedToStrided bounds
	if _, _, err := cache.TransformBlockedToStrided(-1); err == nil {
		t.Errorf("TransformBlockedToStrided(-1) expected error, got nil")
	}
	if _, _, err := cache.TransformBlockedToStrided(1000); err == nil {
		t.Errorf("TransformBlockedToStrided(1000) exceeding cache expected error, got nil")
	}
}

func TestContig_CoalescedBurstMetrics(t *testing.T) {
	cfg := KVContigConfig{
		BlockSizeTokens: ContigBlockSizeTokens,
		NumHeads:        8,
		HeadDim:         128,
		SubChannels:     LPDDR5XSubChannels,
	}
	cache := NewContigKVCache(cfg)

	numTokens := 64
	metrics := cache.CoalescedBurstMetrics(numTokens)

	// 64 tokens * 8 heads * 128 dim * 2 bytes * 2 (K+V) = 262,144 bytes
	expectedBytes := int64(64 * 8 * 128 * 2 * 2)
	if metrics.TotalBytes != expectedBytes {
		t.Errorf("TotalBytes = %d, want %d", metrics.TotalBytes, expectedBytes)
	}

	// 262,144 bytes / 32 bytes per 256-bit burst = 8192 transactions
	expectedBursts := expectedBytes / LPDDR5XBurstBytes
	if metrics.BurstTransactions != expectedBursts {
		t.Errorf("BurstTransactions = %d, want %d", metrics.BurstTransactions, expectedBursts)
	}

	if metrics.EffectiveBandwidthGBs != ContigTargetBandwidthGBs {
		t.Errorf("EffectiveBandwidthGBs = %.2f, want %.2f", metrics.EffectiveBandwidthGBs, ContigTargetBandwidthGBs)
	}
	if metrics.NominalBandwidthGBs != LPDDR5XNominalBandwidthGBs {
		t.Errorf("NominalBandwidthGBs = %.2f, want %.2f", metrics.NominalBandwidthGBs, LPDDR5XNominalBandwidthGBs)
	}
	if metrics.BusEfficiency <= 0.80 {
		t.Errorf("BusEfficiency = %.4f, want > 0.80", metrics.BusEfficiency)
	}
	if metrics.BandwidthGainRatio <= 1.90 {
		t.Errorf("BandwidthGainRatio = %.4f, want > 1.90", metrics.BandwidthGainRatio)
	}
}

func TestContig_ConcurrentAccess(t *testing.T) {
	cfg := KVContigConfig{
		BlockSizeTokens: ContigBlockSizeTokens,
		NumHeads:        4,
		HeadDim:         64,
		SubChannels:     LPDDR5XSubChannels,
	}
	cache := NewContigKVCache(cfg)
	cache.AllocateBlocks(128)

	tokenBytes := cfg.NumHeads * cfg.HeadDim * BytesPerF16

	// Seed initial 32 tokens
	kInit := make([]byte, 32*tokenBytes)
	vInit := make([]byte, 32*tokenBytes)
	_ = cache.AppendTokens(kInit, vInit, 32)

	var wg sync.WaitGroup
	numReaders := 8
	numWriters := 4

	for i := 0; i < numReaders; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for r := 0; r < 50; r++ {
				_ = cache.TotalTokens()
				_ = cache.NumBlocks()
				_ = cache.ChannelAccessProfile(true)
				_ = cache.CoalescedBurstMetrics(32)
				_, _ = cache.ReadBlock(0)
				_, _, _ = cache.ReadToken(0)
			}
		}()
	}

	for i := 0; i < numWriters; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			kTok := make([]byte, 2*tokenBytes)
			vTok := make([]byte, 2*tokenBytes)
			for w := 0; w < 10; w++ {
				_ = cache.AppendTokens(kTok, vTok, 2)
			}
		}()
	}

	wg.Wait()

	expectedTokens := 32 + (numWriters * 10 * 2)
	if cache.TotalTokens() != expectedTokens {
		t.Errorf("TotalTokens after concurrent appends = %d, want %d", cache.TotalTokens(), expectedTokens)
	}
}

func BenchmarkContig_TransformStridedToBlocked(b *testing.B) {
	numTokens := 512
	numHeads := 8
	headDim := 128
	tokenBytes := numHeads * headDim * BytesPerF16
	totalBytes := numTokens * tokenBytes

	k := make([]byte, totalBytes)
	v := make([]byte, totalBytes)
	cache := NewContigKVCache(KVContigConfig{
		BlockSizeTokens: ContigBlockSizeTokens,
		NumHeads:        numHeads,
		HeadDim:         headDim,
	})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cache.TransformStridedToBlocked(k, v, numTokens, numHeads, headDim)
	}
}

func BenchmarkContig_TransformBlockedToStrided(b *testing.B) {
	numTokens := 512
	numHeads := 8
	headDim := 128
	tokenBytes := numHeads * headDim * BytesPerF16
	totalBytes := numTokens * tokenBytes

	k := make([]byte, totalBytes)
	v := make([]byte, totalBytes)
	cache := NewContigKVCache(KVContigConfig{
		BlockSizeTokens: ContigBlockSizeTokens,
		NumHeads:        numHeads,
		HeadDim:         headDim,
	})
	_ = cache.TransformStridedToBlocked(k, v, numTokens, numHeads, headDim)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = cache.TransformBlockedToStrided(numTokens)
	}
}

func BenchmarkContig_ReadBlock(b *testing.B) {
	cache := NewContigKVCache(KVContigConfig{
		BlockSizeTokens: ContigBlockSizeTokens,
		NumHeads:        8,
		HeadDim:         128,
	})
	cache.AllocateBlocks(128)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = cache.ReadBlock(0)
	}
}

func BenchmarkContig_ReadToken(b *testing.B) {
	numTokens := 128
	numHeads := 8
	headDim := 128
	tokenBytes := numHeads * headDim * BytesPerF16
	totalBytes := numTokens * tokenBytes

	k := make([]byte, totalBytes)
	v := make([]byte, totalBytes)
	cache := NewContigKVCache(KVContigConfig{
		BlockSizeTokens: ContigBlockSizeTokens,
		NumHeads:        numHeads,
		HeadDim:         headDim,
	})
	_ = cache.TransformStridedToBlocked(k, v, numTokens, numHeads, headDim)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = cache.ReadToken(63)
	}
}

func BenchmarkContig_AppendTokens(b *testing.B) {
	numHeads := 8
	headDim := 128
	tokenBytes := numHeads * headDim * BytesPerF16

	k := make([]byte, tokenBytes)
	v := make([]byte, tokenBytes)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		cache := NewContigKVCache(KVContigConfig{
			BlockSizeTokens: ContigBlockSizeTokens,
			NumHeads:        numHeads,
			HeadDim:         headDim,
		})
		b.StartTimer()
		_ = cache.AppendTokens(k, v, 1)
	}
}
