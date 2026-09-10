// Package strix implements the hardware profile, memory bandwidth physics,
// compute HAL, and unified memory pipeline for AMD Strix Halo (Ryzen AI Max+ 395).
package strix

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// TestGQAKVPacking validates:
// 1. 100% of KV cache blocks are 128-byte burst aligned.
// 2. Mathematical equivalence between packed and unpacked GQA attention (cosine similarity > 0.999900).
// 3. Model configurations (Qwen 2.5 Coder 32B 8:1, LLaMA-3 4:1, Qwen 2.5 7B 7:1).
// 4. Zero-copy GTT buffer allocation and pointer identity.
func TestGQAKVPacking(t *testing.T) {
	t.Run("Qwen25Coder32B_Alignment_And_AttentionParity", func(t *testing.T) {
		cfg := Qwen25Coder32BGQAConfig()
		cfg.LayerCount = 4                            // Test on 4 layers for speed
		cfg.TokenBlockSize = DefaultGQATokenBlockSize // 16

		packer, err := NewGQAPacker(cfg)
		if err != nil {
			t.Fatalf("failed to create GQAPacker: %v", err)
		}
		defer packer.Close()

		rng := rand.New(rand.NewSource(42))
		const (
			layer        = 0
			headGroupIdx = 0
			numTokens    = 48 // 3 full blocks (16 * 3 = 48 tokens)
			headDim      = 128
			queryRatio   = 8
		)

		// Generate random reference K and V vectors: [numTokens, headDim]
		kUnpacked := make([]float32, numTokens*headDim)
		vUnpacked := make([]float32, numTokens*headDim)
		for i := range kUnpacked {
			kUnpacked[i] = (rng.Float32() - 0.5) * 2.0
			vUnpacked[i] = (rng.Float32() - 0.5) * 2.0
		}

		// Pack tokens into the GQA packer
		for tok := 0; tok < numTokens; tok++ {
			kVec := kUnpacked[tok*headDim : (tok+1)*headDim]
			vVec := vUnpacked[tok*headDim : (tok+1)*headDim]
			if err := packer.PackToken(layer, headGroupIdx, tok, kVec, vVec); err != nil {
				t.Fatalf("PackToken failed at tok %d: %v", tok, err)
			}
		}

		// Invariant 1: 100% of blocks are strictly 128-byte aligned
		allAligned, alignedCount, totalCount := packer.CheckAllBlocksAligned()
		if !allAligned || totalCount == 0 || alignedCount != totalCount {
			t.Fatalf("alignment check failed: aligned %d of %d blocks", alignedCount, totalCount)
		}

		for i, block := range packer.Blocks() {
			if !block.IsAligned128() {
				t.Fatalf("block %d (ID %d) failed IsAligned128()", i, block.BlockID)
			}
			if block.HostPtr%128 != 0 {
				t.Fatalf("block %d HostPtr 0x%x not 128-byte aligned", i, block.HostPtr)
			}
			if block.DevicePtr != 0 && block.DevicePtr%128 != 0 {
				t.Fatalf("block %d DevicePtr 0x%x not 128-byte aligned", i, block.DevicePtr)
			}
		}
		t.Logf("Verified 100%% 128-byte alignment across %d allocated blocks", totalCount)

		// Invariant 2: Mathematical equivalence of GQA attention (cosine similarity > 0.999900)
		qGroup := make([]float32, queryRatio*headDim)
		for i := range qGroup {
			qGroup[i] = (rng.Float32() - 0.5) * 2.0
		}

		outPacked, err := packer.ComputeAttentionGroup(layer, headGroupIdx, qGroup, 0, numTokens)
		if err != nil {
			t.Fatalf("ComputeAttentionGroup failed: %v", err)
		}

		outRef, err := ComputeAttentionGroupReference(qGroup, kUnpacked, vUnpacked, queryRatio, headDim, numTokens)
		if err != nil {
			t.Fatalf("ComputeAttentionGroupReference failed: %v", err)
		}

		sim, err := ComputeLogitCosineSimilarity(outPacked, outRef)
		if err != nil {
			t.Fatalf("ComputeLogitCosineSimilarity failed: %v", err)
		}
		t.Logf("Packed vs Reference Attention Cosine Similarity: %.8f", sim)

		if sim <= 0.999900 {
			t.Fatalf("attention cosine similarity %.8f <= 0.999900 threshold", sim)
		}

		// Verify element-wise numerical parity
		for i := range outPacked {
			diff := math.Abs(float64(outPacked[i] - outRef[i]))
			if diff > 1e-6 {
				t.Fatalf("element mismatch at index %d: packed=%f, ref=%f, diff=%e",
					i, outPacked[i], outRef[i], diff)
			}
		}
	})

	t.Run("Strided_Pack_Unpack_RoundTrip", func(t *testing.T) {
		cfg := DefaultGQAPackerConfig()
		cfg.LayerCount = 2
		packer, err := NewGQAPacker(cfg)
		if err != nil {
			t.Fatalf("NewGQAPacker failed: %v", err)
		}
		defer packer.Close()

		rng := rand.New(rand.NewSource(1337))
		const (
			seqLen = 32
			layer  = 1
		)
		totalElems := seqLen * cfg.NumKVHeads * cfg.HeadDim
		kStrided := make([]float32, totalElems)
		vStrided := make([]float32, totalElems)
		for i := range kStrided {
			kStrided[i] = rng.Float32()
			vStrided[i] = rng.Float32()
		}

		// Pack strided tensor
		if err := packer.PackStrided(layer, seqLen, kStrided, vStrided); err != nil {
			t.Fatalf("PackStrided failed: %v", err)
		}

		// Unpack back to strided tensor
		kUnpacked, vUnpacked, err := packer.UnpackStrided(layer, seqLen)
		if err != nil {
			t.Fatalf("UnpackStrided failed: %v", err)
		}

		// Bit-exact verification
		for i := range kStrided {
			if kUnpacked[i] != kStrided[i] {
				t.Fatalf("k mismatch at index %d: got %f, want %f", i, kUnpacked[i], kStrided[i])
			}
			if vUnpacked[i] != vStrided[i] {
				t.Fatalf("v mismatch at index %d: got %f, want %f", i, vUnpacked[i], vStrided[i])
			}
		}
		t.Logf("Strided round-trip verified bit-exact across %d elements", totalElems*2)
	})

	t.Run("Alternative_GQA_Geometries", func(t *testing.T) {
		testCases := []struct {
			name       string
			numQ       int
			numKV      int
			headDim    int
			ratio      int
			tokenBlock int
		}{
			{"LLaMA3_8B_4to1", 32, 8, 128, 4, 16},
			{"Qwen25_7B_7to1", 28, 4, 128, 7, 16},
			{"Custom_32Block", 64, 8, 128, 8, 32},
			{"Compact_HeadDim64", 32, 4, 64, 8, 16},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				cfg := GQAPackerConfig{
					LayerCount:     2,
					NumQHeads:      tc.numQ,
					NumKVHeads:     tc.numKV,
					HeadDim:        tc.headDim,
					TokenBlockSize: tc.tokenBlock,
				}
				packer, err := NewGQAPacker(cfg)
				if err != nil {
					t.Fatalf("NewGQAPacker failed for %s: %v", tc.name, err)
				}
				defer packer.Close()

				rng := rand.New(rand.NewSource(100))
				numTokens := tc.tokenBlock * 2
				kFlat := make([]float32, numTokens*tc.headDim)
				vFlat := make([]float32, numTokens*tc.headDim)
				for i := range kFlat {
					kFlat[i] = rng.Float32()
					vFlat[i] = rng.Float32()
				}

				// Pack via PackBlock
				_, err = packer.PackBlock(0, 0, 0, kFlat[:tc.tokenBlock*tc.headDim], vFlat[:tc.tokenBlock*tc.headDim], tc.tokenBlock)
				if err != nil {
					t.Fatalf("PackBlock 0 failed: %v", err)
				}
				_, err = packer.PackBlock(0, 0, tc.tokenBlock, kFlat[tc.tokenBlock*tc.headDim:], vFlat[tc.tokenBlock*tc.headDim:], tc.tokenBlock)
				if err != nil {
					t.Fatalf("PackBlock 1 failed: %v", err)
				}

				// Check 128-byte alignment
				allAligned, alignedCount, totalCount := packer.CheckAllBlocksAligned()
				if !allAligned || totalCount != 2 || alignedCount != 2 {
					t.Fatalf("alignment check failed for %s", tc.name)
				}

				// Attention parity
				qGroup := make([]float32, tc.ratio*tc.headDim)
				for i := range qGroup {
					qGroup[i] = rng.Float32()
				}
				outPacked, err := packer.ComputeAttentionGroup(0, 0, qGroup, 0, numTokens)
				if err != nil {
					t.Fatalf("ComputeAttentionGroup failed: %v", err)
				}
				outRef, err := ComputeAttentionGroupReference(qGroup, kFlat, vFlat, tc.ratio, tc.headDim, numTokens)
				if err != nil {
					t.Fatalf("ComputeAttentionGroupReference failed: %v", err)
				}
				sim, err := ComputeLogitCosineSimilarity(outPacked, outRef)
				if err != nil || sim <= 0.999900 {
					t.Fatalf("cosine similarity violation: sim=%.8f err=%v", sim, err)
				}
			})
		}
	})

	t.Run("ZeroCopy_GTT_Backed_Packing", func(t *testing.T) {
		cfg := DefaultGQAPackerConfig()
		cfg.LayerCount = 2
		cfg.UseGTT = true

		packer, err := NewGQAPacker(cfg)
		if err != nil {
			t.Fatalf("NewGQAPacker with GTT failed: %v", err)
		}
		defer packer.Close()

		rng := rand.New(rand.NewSource(999))
		k := make([]float32, cfg.HeadDim)
		v := make([]float32, cfg.HeadDim)
		for i := range k {
			k[i] = rng.Float32()
			v[i] = rng.Float32()
		}

		if err := packer.PackToken(0, 0, 0, k, v); err != nil {
			t.Fatalf("PackToken with GTT failed: %v", err)
		}

		blocks := packer.Blocks()
		if len(blocks) != 1 {
			t.Fatalf("expected 1 block, got %d", len(blocks))
		}
		b := blocks[0]
		if !b.IsAligned128() {
			t.Fatalf("GTT block failed IsAligned128()")
		}
		if !b.IsZeroCopy() {
			t.Fatalf("GTT block failed IsZeroCopy(): host=0x%x dev=0x%x", b.HostPtr, b.DevicePtr)
		}

		// Verify unpack matches
		kOut, vOut, err := packer.UnpackToken(0, 0, 0)
		if err != nil {
			t.Fatalf("UnpackToken failed: %v", err)
		}
		for i := range k {
			if kOut[i] != k[i] || vOut[i] != v[i] {
				t.Fatalf("mismatch at element %d", i)
			}
		}
	})
}

// TestGQAKVPackingUnalignedInput verifies defensive validation and structured error
// handling when unaligned memory or invalid geometry configurations are provided.
func TestGQAKVPackingUnalignedInput(t *testing.T) {
	t.Run("Unaligned_HostPtr_Rejection", func(t *testing.T) {
		cfg := DefaultGQAPackerConfig()
		packer, err := NewGQAPacker(cfg)
		if err != nil {
			t.Fatalf("NewGQAPacker failed: %v", err)
		}
		defer packer.Close()

		// Construct block with deliberately unaligned HostPtr (0x1001, not multiple of 128)
		unalignedBlock := &GQAPackedBlock{
			BlockID:      999,
			LayerIdx:     0,
			HeadGroupIdx: 0,
			TokenStart:   0,
			NumTokens:    1,
			Data:         make([]byte, cfg.BlockSizeBytes()),
			HostPtr:      0x1001,
			DevicePtr:    0x1001,
		}

		if unalignedBlock.IsAligned128() {
			t.Fatal("expected unaligned HostPtr to fail IsAligned128()")
		}

		err = packer.RegisterBlock(unalignedBlock)
		if !errors.Is(err, ErrUnalignedKVMemory) {
			t.Fatalf("expected ErrUnalignedKVMemory, got: %v", err)
		}

		err = packer.ValidateBlock(unalignedBlock)
		if !errors.Is(err, ErrUnalignedKVMemory) {
			t.Fatalf("expected ErrUnalignedKVMemory from ValidateBlock, got: %v", err)
		}
	})

	t.Run("Unaligned_DevicePtr_Rejection", func(t *testing.T) {
		block := &GQAPackedBlock{
			BlockID:      998,
			HostPtr:      0x2000, // 128-byte aligned (0x2000 % 128 == 0)
			DevicePtr:    0x2001, // unaligned
			Data:         make([]byte, 16384),
			LayerIdx:     0,
			HeadGroupIdx: 0,
		}

		if block.IsAligned128() {
			t.Fatal("expected unaligned DevicePtr to fail IsAligned128()")
		}
	})

	t.Run("Zero_HostPtr_Rejection", func(t *testing.T) {
		block := &GQAPackedBlock{
			BlockID:   997,
			HostPtr:   0,
			DevicePtr: 0,
		}

		if block.IsAligned128() {
			t.Fatal("expected zero HostPtr to fail IsAligned128()")
		}
	})

	t.Run("Invalid_GQAGeometry_Rejection", func(t *testing.T) {
		// Non-divisible query heads (65 Q heads, 8 KV heads)
		invalidConfig := GQAPackerConfig{
			LayerCount:     32,
			NumQHeads:      65,
			NumKVHeads:     8,
			HeadDim:        128,
			TokenBlockSize: 16,
		}
		_, err := NewGQAPacker(invalidConfig)
		if !errors.Is(err, ErrInvalidGQAGeometry) {
			t.Fatalf("expected ErrInvalidGQAGeometry for non-divisible heads, got: %v", err)
		}

		// Zero KV heads
		zeroKVConfig := GQAPackerConfig{
			LayerCount:     32,
			NumQHeads:      64,
			NumKVHeads:     0,
			HeadDim:        128,
			TokenBlockSize: 16,
		}
		_, err = NewGQAPacker(zeroKVConfig)
		if !errors.Is(err, ErrInvalidGQAGeometry) {
			t.Fatalf("expected ErrInvalidGQAGeometry for zero KV heads, got: %v", err)
		}

		// Zero HeadDim
		zeroDimConfig := GQAPackerConfig{
			LayerCount:     32,
			NumQHeads:      64,
			NumKVHeads:     8,
			HeadDim:        0,
			TokenBlockSize: 16,
		}
		_, err = NewGQAPacker(zeroDimConfig)
		if !errors.Is(err, ErrInvalidGQAGeometry) {
			t.Fatalf("expected ErrInvalidGQAGeometry for zero HeadDim, got: %v", err)
		}
	})

	t.Run("Boundary_And_Dimension_Method_Errors", func(t *testing.T) {
		cfg := DefaultGQAPackerConfig()
		cfg.LayerCount = 2
		packer, err := NewGQAPacker(cfg)
		if err != nil {
			t.Fatalf("NewGQAPacker failed: %v", err)
		}
		defer packer.Close()

		validVec := make([]float32, cfg.HeadDim)
		invalidVec := make([]float32, cfg.HeadDim-1)

		// Layer out of bounds
		if err := packer.PackToken(-1, 0, 0, validVec, validVec); !errors.Is(err, ErrInvalidGQAGeometry) {
			t.Fatalf("expected ErrInvalidGQAGeometry for layer -1, got: %v", err)
		}
		if err := packer.PackToken(2, 0, 0, validVec, validVec); !errors.Is(err, ErrInvalidGQAGeometry) {
			t.Fatalf("expected ErrInvalidGQAGeometry for layer 2, got: %v", err)
		}

		// KVHead out of bounds
		if err := packer.PackToken(0, -1, 0, validVec, validVec); !errors.Is(err, ErrInvalidGQAGeometry) {
			t.Fatalf("expected ErrInvalidGQAGeometry for kvHead -1, got: %v", err)
		}
		if err := packer.PackToken(0, 8, 0, validVec, validVec); !errors.Is(err, ErrInvalidGQAGeometry) {
			t.Fatalf("expected ErrInvalidGQAGeometry for kvHead 8, got: %v", err)
		}

		// Vector dimension mismatch
		if err := packer.PackToken(0, 0, 0, invalidVec, validVec); !errors.Is(err, ErrInvalidGQAGeometry) {
			t.Fatalf("expected ErrInvalidGQAGeometry for wrong vector length, got: %v", err)
		}

		// Attention group dimension mismatch
		wrongQGroup := make([]float32, cfg.QueryRatio()*cfg.HeadDim-1)
		_, err = packer.ComputeAttentionGroup(0, 0, wrongQGroup, 0, 16)
		if !errors.Is(err, ErrInvalidGQAGeometry) {
			t.Fatalf("expected ErrInvalidGQAGeometry for wrong qGroup length, got: %v", err)
		}
	})
}

// TestGQAKVPackingBandwidthModel verifies that:
// 1. All memory access transactions are coalesced into single 128-byte DRAM bursts (CoalescingRatio = 1.0).
// 2. Sustained memory read bandwidth reaches >= 218.44 GB/s (80% roofline target on AMD Strix Halo).
// 3. Unpacked strided layout fails the roofline ceiling due to non-coalesced burst stalls.
func TestGQAKVPackingBandwidthModel(t *testing.T) {
	cfg := DefaultGQAPackerConfig()
	packer, err := NewGQAPacker(cfg)
	if err != nil {
		t.Fatalf("NewGQAPacker failed: %v", err)
	}
	defer packer.Close()

	const (
		subagents = 12   // 12-subagent decode workload matching TICKET-19
		seqLen    = 1024 // 1,024 context tokens
	)

	// Evaluate packed layout
	metrics := packer.EvaluateBandwidthModel(subagents, seqLen)

	t.Logf("Packed GQA Bandwidth Model (subagents=%d, seqLen=%d):", subagents, seqLen)
	t.Logf("  Total Access Bytes:       %d (%.2f MB)", metrics.TotalAccessBytes, float64(metrics.TotalAccessBytes)/(1024*1024))
	t.Logf("  Burst Count:              %d bursts (%d bytes/burst)", metrics.BurstCount, metrics.BytesPerBurst)
	t.Logf("  Coalesced Bursts:         %d", metrics.CoalescedBursts)
	t.Logf("  Uncoalesced Bursts:       %d", metrics.UncoalescedBursts)
	t.Logf("  Burst Coalescing Ratio:   %.4f (100%% coalescing)", metrics.CoalescingRatio)
	t.Logf("  Sustained Bandwidth:      %.2f GB/s (Target: >= %.2f GB/s)", metrics.SustainedBandwidthGBs, Target80PercentBandwidthGBps)
	t.Logf("  Roofline Attained:        %v", metrics.RooflineAttained)
	t.Logf("  Speedup vs Unpacked:      %.2fx", metrics.SpeedupVsUnpacked)

	// Verification 1: 100% burst coalescing
	if metrics.CoalescingRatio != 1.0 {
		t.Fatalf("expected CoalescingRatio == 1.0, got %.4f", metrics.CoalescingRatio)
	}
	if metrics.UncoalescedBursts != 0 {
		t.Fatalf("expected 0 uncoalesced bursts, got %d", metrics.UncoalescedBursts)
	}
	if metrics.BytesPerBurst != LPDDR5XBurstBytes {
		t.Fatalf("expected burst size %d bytes, got %d", LPDDR5XBurstBytes, metrics.BytesPerBurst)
	}

	// Verification 2: Sustained memory read bandwidth reaches >= 218.44 GB/s
	if metrics.SustainedBandwidthGBs < Target80PercentBandwidthGBps {
		t.Fatalf("sustained bandwidth %.2f GB/s < target %.2f GB/s",
			metrics.SustainedBandwidthGBs, Target80PercentBandwidthGBps)
	}
	if !metrics.RooflineAttained {
		t.Fatalf("expected RooflineAttained == true")
	}
	if metrics.SpeedupVsUnpacked < 1.40 {
		t.Fatalf("expected speedup vs unpacked >= 1.40x, got %.2fx", metrics.SpeedupVsUnpacked)
	}

	// Verification 3: Unpacked baseline comparison
	unpacked := EvaluateUnpackedBandwidthModel(subagents, seqLen, cfg.NumKVHeads, cfg.HeadDim)
	t.Logf("Unpacked Strided Baseline Bandwidth Model:")
	t.Logf("  Burst Coalescing Ratio:   %.4f", unpacked.CoalescingRatio)
	t.Logf("  Uncoalesced Bursts:       %d", unpacked.UncoalescedBursts)
	t.Logf("  Sustained Bandwidth:      %.2f GB/s", unpacked.SustainedBandwidthGBs)
	t.Logf("  Roofline Attained:        %v", unpacked.RooflineAttained)

	if unpacked.CoalescingRatio >= 1.0 {
		t.Fatalf("unpacked baseline should not have 1.0 coalescing ratio")
	}
	if unpacked.UncoalescedBursts <= 0 {
		t.Fatalf("unpacked baseline should have positive uncoalesced bursts")
	}
	if unpacked.RooflineAttained {
		t.Fatalf("unpacked baseline should not attain 80%% roofline target")
	}
}
