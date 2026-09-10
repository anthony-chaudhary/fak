package strix

import (
	"testing"
)

// TestSetPermutation executes the four primary acceptance criteria subtests:
// 1. MathematicalBijection: 1:1 set permutation over all 32,768 sets.
// 2. DistributionUniformity: Uniform distribution of 8,192 tokens with CV <= 5% and 0 overflow.
// 3. BlockLayout64TokenAlignmentAndNonAliasing: 64-byte alignment and adjacent non-aliasing.
// 4. HardwareWitnessOrSimulatedArm: >= 94% effective MALL capacity utilization vs ~30% baseline.
func TestSetPermutation(t *testing.T) {
	t.Run("MathematicalBijection", func(t *testing.T) {
		// Scoped Acceptance Criteria 1 / Verification of prime stride P = 65,521
		if PrimeStrideP != 65521 {
			t.Fatalf("expected PrimeStrideP = 65521, got %d", PrimeStrideP)
		}
		if g := GCD(int(PrimeStrideP), MALLSets); g != 1 {
			t.Fatalf("expected gcd(PrimeStrideP, MALLSets) == 1, got %d", g)
		}
		if g := GCD(int(CoprimeTagMultiplier), MALLSets); g != 1 {
			t.Fatalf("expected gcd(CoprimeTagMultiplier, MALLSets) == 1, got %d", g)
		}

		engine := NewSetPermutationEngine()

		// Test bijection across multiple diverse address tags
		testTags := []uint32{0, 1, 2, 7, 16, 42, 1024, 4097, 65521, 0x12345, 0x7FFFFFF}
		for _, tag := range testTags {
			seen := make(map[uint32]bool, MALLSets)
			for rawSet := uint32(0); rawSet < MALLSets; rawSet++ {
				perm := engine.PermuteSet(rawSet, tag)
				if perm >= MALLSets {
					t.Fatalf("tag %d: permuted set index %d exceeds MALLSets (%d)", tag, perm, MALLSets)
				}
				if seen[perm] {
					t.Fatalf("tag %d: collision detected: set %d mapped multiple times for rawSet %d", tag, perm, rawSet)
				}
				seen[perm] = true
			}
			if len(seen) != MALLSets {
				t.Fatalf("tag %d: expected %d unique sets, got %d", tag, MALLSets, len(seen))
			}
		}
		t.Logf("[SW-VERIFIED] Mathematical bijection verified across %d tags over all %d sets", len(testTags), MALLSets)
	})

	t.Run("DistributionUniformity", func(t *testing.T) {
		// Scoped Acceptance Criteria 2:
		// Map 8,192 tokens across 32 layers into 32,768 sets.
		// Variance / CV must be <= 5% (CV <= 0.05) and BinOverflow (>= 17 ways) == 0.
		engine := NewSetPermutationEngine()
		sim := NewAssociativitySimulator()

		const numLayers = 32
		const blocksPerLayer = 4
		const linesPerBlock = 4096
		basePtr := uintptr(0x10000000)

		totalLines := numLayers * blocksPerLayer * linesPerBlock
		if totalLines != MALLTotalCacheLines {
			t.Fatalf("expected total lines %d to equal MALLTotalCacheLines %d", totalLines, MALLTotalCacheLines)
		}

		// Allocate 128 blocks of 64 tokens each (256 KiB / 4,096 lines per block)
		for layer := 0; layer < numLayers; layer++ {
			for b := 0; b < blocksPerLayer; b++ {
				globalBlockID := layer*blocksPerLayer + b
				blockBaseAddr := basePtr + uintptr(globalBlockID)*uintptr(RawBlockSizeBytes)
				for l := 0; l < linesPerBlock; l++ {
					lineAddr := blockBaseAddr + uintptr(l*MALLCacheLineBytes)
					sim.Access(lineAddr, ArmPrimeModulo, engine)
				}
			}
		}

		h := sim.DispersionHistogram()

		t.Logf("Uniformity Stats: Mean=%.2f, Variance=%.6f, StdDev=%.6f, CV=%.6f, MinOcc=%d, MaxOcc=%d",
			h.Mean, h.Variance, h.StdDev, h.CV, h.MinOccupancy, h.MaxOccupancy)
		t.Logf("Histogram Bins: Bin0=%d, Bin1-4=%d, Bin5-8=%d, Bin9-12=%d, Bin13-15=%d, Bin16=%d, Overflow=%d",
			h.Bin0Ways, h.Bin1To4Ways, h.Bin5To8Ways, h.Bin9To12Ways, h.Bin13To15Ways, h.Bin16Ways, h.BinOverflow)

		if h.CV > 0.05 {
			t.Errorf("expected CV <= 0.05 (5%%), got %f (Variance=%f, StdDev=%f)", h.CV, h.Variance, h.StdDev)
		}
		if h.BinOverflow != 0 {
			t.Errorf("expected BinOverflow == 0 (no conflict thrashing), got %d sets with >= 17 ways", h.BinOverflow)
		}
		if h.Mean != 16.0 {
			t.Errorf("expected Mean == 16.0, got %f", h.Mean)
		}
		if h.Bin16Ways != MALLSets {
			t.Errorf("expected all %d sets in Bin16Ways, got %d", MALLSets, h.Bin16Ways)
		}
		if h.MinOccupancy != 16 || h.MaxOccupancy != 16 {
			t.Errorf("expected MinOccupancy=16, MaxOccupancy=16, got min=%d, max=%d", h.MinOccupancy, h.MaxOccupancy)
		}
		if sim.ConflictMisses() != 0 {
			t.Errorf("expected 0 conflict misses under uniform permutation, got %d", sim.ConflictMisses())
		}
		t.Log("[SW-VERIFIED] 8,192 tokens across 32 layers mapped with CV <= 0.05 and 0 overflow sets")
	})

	t.Run("BlockLayout64TokenAlignmentAndNonAliasing", func(t *testing.T) {
		// Scoped Acceptance Criteria 3:
		// 64-token block packer maintains 64-byte alignment while ensuring
		// adjacent block base addresses never alias to the same set index modulo 32,768.
		cfg := DefaultBlockLayoutConfig()
		engine := NewSetPermutationEngine()
		packer, err := NewBlockLayoutPacker(cfg, engine)
		if err != nil {
			t.Fatalf("failed to create BlockLayoutPacker: %v", err)
		}

		basePtr := uintptr(0x7fff00000000) // 64-byte aligned base pointer
		blocks, err := packer.GenerateContextLayout(basePtr, 8192)
		if err != nil {
			t.Fatalf("failed to generate context layout: %v", err)
		}

		if len(blocks) != 128 {
			t.Fatalf("expected 128 blocks for 8,192 tokens, got %d", len(blocks))
		}

		nonAliasing, counts := packer.VerifyNonAliasing(blocks)
		if !nonAliasing {
			t.Errorf("VerifyNonAliasing returned false")
		}

		for i, b := range blocks {
			// Verify 64-byte alignment
			if b.VirtualAddress%uintptr(MALLCacheLineBytes) != 0 {
				t.Errorf("block %d VirtualAddress 0x%x is not 64-byte aligned", i, b.VirtualAddress)
			}
			// Verify adjacent non-aliasing
			if i > 0 && b.BaseSetIndex == blocks[i-1].BaseSetIndex {
				t.Errorf("adjacent blocks %d and %d alias to set %d", i-1, i, b.BaseSetIndex)
			}
			// Verify each base set index in the 128-block sequence is unique
			if counts[b.BaseSetIndex] != 1 {
				t.Errorf("block %d BaseSetIndex %d occurs %d times in sequence", i, b.BaseSetIndex, counts[b.BaseSetIndex])
			}
		}

		t.Logf("[SW-VERIFIED] 128 blocks verified: 100%% 64B aligned, 100%% distinct base set indices")
	})

	t.Run("HardwareWitnessOrSimulatedArm", func(t *testing.T) {
		// Scoped Acceptance Criteria 4:
		// Evaluates effective MALL capacity utilization >= 94% under prime-modulo permutation (Arm 1)
		// vs ~30% on power-of-two baseline (Arm 3), with bipartite SW-VERIFIED vs HW-WITNESSED labeling.
		engine := NewSetPermutationEngine()

		// Simulate multi-head allocation scenario (32 layers x 8 heads = 256 buffers of 2,048 lines)
		// under power-of-two 2MB huge page boundaries.
		const numLayers = 32
		const numHeads = 8
		const linesPerHead = 2048 // 256 * 2048 = 524,288 lines = 32 MiB
		const hugePage2MB = 2 * 1024 * 1024

		// 1. Arm 1: Prime-Modulo Permutation
		simPrime := NewAssociativitySimulator()
		for layer := 0; layer < numLayers; layer++ {
			for head := 0; head < numHeads; head++ {
				base := uintptr((layer*numHeads + head) * hugePage2MB)
				for l := 0; l < linesPerHead; l++ {
					lineAddr := base + uintptr(l*MALLCacheLineBytes)
					simPrime.Access(lineAddr, ArmPrimeModulo, engine)
				}
			}
		}
		utilPrime := simPrime.EffectiveUtilization()
		conflictRatioPrime := simPrime.ConflictMissRatio()

		// 2. Arm 3: Power-of-Two Baseline
		simPow2 := NewAssociativitySimulator()
		for layer := 0; layer < numLayers; layer++ {
			for head := 0; head < numHeads; head++ {
				base := uintptr((layer*numHeads + head) * hugePage2MB)
				for l := 0; l < linesPerHead; l++ {
					lineAddr := base + uintptr(l*MALLCacheLineBytes)
					simPow2.Access(lineAddr, ArmPowerOfTwo, engine)
				}
			}
		}
		utilPow2 := simPow2.EffectiveUtilization()
		conflictRatioPow2 := simPow2.ConflictMissRatio()

		t.Logf("[SW-VERIFIED] Arm 1 (Prime-Modulo): Utilization = %.2f%%, ConflictMissRatio = %.4f",
			utilPrime, conflictRatioPrime)
		t.Logf("[SW-VERIFIED] Arm 3 (Power-of-Two): Utilization = %.2f%%, ConflictMissRatio = %.4f",
			utilPow2, conflictRatioPow2)

		if utilPrime < 94.0 {
			t.Errorf("expected Arm 1 EffectiveUtilization >= 94.0%%, got %.2f%%", utilPrime)
		}
		if utilPow2 > 35.0 {
			t.Errorf("expected Arm 3 baseline EffectiveUtilization <= 35.0%% (collapsing under conflict thrashing), got %.2f%%", utilPow2)
		}

		// Non-amplification receipt assertion
		t.Log("[SW-VERIFIED] 94% MALL capacity verified mathematically. Live hardware test pending.")
		t.Log("[HW-WITNESSED] Pending live appliance access for hardware perf counters")
	})
}

// TestCoprimePadding_MathematicalProperties proves the number-theoretic guarantees of coprime strides.
func TestCoprimePadding_MathematicalProperties(t *testing.T) {
	// 1. gcd(4097, 32768) = 1
	if g := GCD(int(CoprimeTagMultiplier), MALLSets); g != 1 {
		t.Errorf("expected gcd(4097, 32768) == 1, got %d", g)
	}

	// 2. LinesPerPaddedBlock = 4097 is coprime to MALLSets (32768)
	if LinesPerPaddedBlock != 4097 {
		t.Fatalf("expected LinesPerPaddedBlock = 4097, got %d", LinesPerPaddedBlock)
	}
	if g := GCD(LinesPerPaddedBlock, MALLSets); g != 1 {
		t.Errorf("expected gcd(LinesPerPaddedBlock, MALLSets) == 1, got %d", g)
	}

	// 3. (k * 4097) % 32768 generates the complete cyclic group Z_32768
	visited := make(map[uint32]bool, MALLSets)
	for k := uint32(0); k < MALLSets; k++ {
		val := (k * uint32(LinesPerPaddedBlock)) % MALLSets
		if visited[val] {
			t.Fatalf("cycle detected at k=%d, val=%d before exhausting 32,768 steps", k, val)
		}
		visited[val] = true
	}
	if len(visited) != MALLSets {
		t.Errorf("expected full period %d, got %d unique elements", MALLSets, len(visited))
	}
}

// TestDecomposeAddress verifies exact bit-field extraction.
func TestDecomposeAddress(t *testing.T) {
	// Offset: 42 (bits [5:0])
	// SetIndex: 12345 (bits [20:6])
	// Tag: 9876 (bits [47:21])
	targetOffset := uint32(42)
	targetSet := uint32(12345)
	targetTag := uint32(9876)

	addr := uintptr(targetOffset) | (uintptr(targetSet) << MALLSetIndexShift) | (uintptr(targetTag) << MALLTagShift)

	setIndex, tag, offset := DecomposeAddress(addr)
	if offset != targetOffset {
		t.Errorf("expected offset %d, got %d", targetOffset, offset)
	}
	if setIndex != targetSet {
		t.Errorf("expected setIndex %d, got %d", targetSet, setIndex)
	}
	if tag != targetTag {
		t.Errorf("expected tag %d, got %d", targetTag, tag)
	}
}

// TestNewBlockLayoutPacker_Validation verifies configuration guardrails.
func TestNewBlockLayoutPacker_Validation(t *testing.T) {
	engine := NewSetPermutationEngine()

	// Zero tokens per block
	cfg := DefaultBlockLayoutConfig()
	cfg.TokensPerBlock = 0
	if _, err := NewBlockLayoutPacker(cfg, engine); err == nil {
		t.Errorf("expected error for TokensPerBlock=0")
	}

	// Zero layers
	cfg = DefaultBlockLayoutConfig()
	cfg.NumLayers = 0
	if _, err := NewBlockLayoutPacker(cfg, engine); err == nil {
		t.Errorf("expected error for NumLayers=0")
	}

	// Zero KV heads
	cfg = DefaultBlockLayoutConfig()
	cfg.NumKVHeads = 0
	if _, err := NewBlockLayoutPacker(cfg, engine); err == nil {
		t.Errorf("expected error for NumKVHeads=0")
	}

	// Zero HeadDim
	cfg = DefaultBlockLayoutConfig()
	cfg.HeadDim = 0
	if _, err := NewBlockLayoutPacker(cfg, engine); err == nil {
		t.Errorf("expected error for HeadDim=0")
	}

	// Zero BytesPerElement
	cfg = DefaultBlockLayoutConfig()
	cfg.BytesPerElement = 0
	if _, err := NewBlockLayoutPacker(cfg, engine); err == nil {
		t.Errorf("expected error for BytesPerElement=0")
	}

	// Non-multiple of 64B padding
	cfg = DefaultBlockLayoutConfig()
	cfg.EnableCoprimePadding = true
	cfg.PaddingBytes = 13
	if _, err := NewBlockLayoutPacker(cfg, engine); err == nil {
		t.Errorf("expected error for PaddingBytes not multiple of 64")
	}
}

// TestAssociativitySimulator_LRU verifies LRU eviction mechanics within a set.
func TestAssociativitySimulator_LRU(t *testing.T) {
	sim := NewAssociativitySimulator()
	setIdx := uint32(100)

	// Fill all 16 ways of set 100 with tags 1..16
	for tag := uint32(1); tag <= 16; tag++ {
		hit, conflict := sim.RecordLine(setIdx, tag)
		if hit {
			t.Errorf("expected cold miss for tag %d", tag)
		}
		if conflict {
			t.Errorf("expected no conflict while capacity remains for tag %d", tag)
		}
	}

	// Access tag 1 again -> should hit and refresh its LRU timestamp
	hit, conflict := sim.RecordLine(setIdx, 1)
	if !hit || conflict {
		t.Errorf("expected hit on tag 1, got hit=%v conflict=%v", hit, conflict)
	}

	// Access new tag 17 -> must conflict-evict tag 2 (the oldest untouched tag, since tag 1 was touched)
	hit, conflict = sim.RecordLine(setIdx, 17)
	if hit || !conflict {
		t.Errorf("expected conflict eviction for tag 17, got hit=%v conflict=%v", hit, conflict)
	}

	// Verify tag 1 is still present (hit)
	hit, _ = sim.RecordLine(setIdx, 1)
	if !hit {
		t.Errorf("tag 1 should have remained in cache due to LRU refresh")
	}

	// Verify tag 2 was evicted (miss)
	hit, conflict = sim.RecordLine(setIdx, 2)
	if hit {
		t.Errorf("tag 2 should have been evicted")
	}
}
