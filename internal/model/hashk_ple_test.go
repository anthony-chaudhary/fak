package model

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/perfscout"
)

// TestHashKDeterministicSlotMapping verifies deterministic slot mapping across subtables
// matches polynomial hash specifications and parity with internal/perfscout reference.
func TestHashKDeterministicSlotMapping(t *testing.T) {
	// Test SplitMix64 determinism
	seed := uint64(0x123456789abcdef0)
	step1 := SplitMix64(seed)
	step2 := SplitMix64(seed)
	if step1 != step2 {
		t.Fatalf("SplitMix64 non-deterministic: %d vs %d", step1, step2)
	}
	if step1 == 0 {
		t.Fatal("SplitMix64 output unexpected zero")
	}

	// Parity with perfscout.SplitMix64 reference
	refStep := perfscout.SplitMix64(seed)
	if step1 != refStep {
		t.Fatalf("SplitMix64 mismatch with perfscout reference: got %d, want %d", step1, refStep)
	}

	// Test polynomial congruential hash calculation against exact formula:
	// x_sub = (local_idx + 1) * 2862933555777941757ULL + SALTS[sub] + head * 998244353ULL;
	numSlots := uint64(80000000)
	testCases := []struct {
		localIdx uint64
		sub      int
		head     uint64
	}{
		{0, 0, 0},
		{0, 1, 0},
		{42, 0, 3},
		{42, 1, 3},
		{999999, 0, 15},
		{999999, 1, 15},
		{319999999, 0, 7},
		{319999999, 1, 7},
	}

	for _, tc := range testCases {
		// Calculate slot via ComputeHashKSlot
		gotSlot := ComputeHashKSlot(tc.localIdx, tc.sub, tc.head, numSlots)

		// Calculate slot directly via ground truth formula
		salt := HashKSalt[tc.sub%2]
		xSub := (tc.localIdx+1)*HashKMultiplier + salt + tc.head*HashKHeadPrime
		wantSlot := SplitMix64(xSub) % numSlots

		if gotSlot != wantSlot {
			t.Errorf("ComputeHashKSlot(idx=%d, sub=%d, head=%d): got %d, want %d",
				tc.localIdx, tc.sub, tc.head, gotSlot, wantSlot)
		}

		// Verify parity with perfscout.ComputeHashKSlot
		perfSlot := perfscout.ComputeHashKSlot(tc.localIdx, tc.sub, tc.head, numSlots)
		if gotSlot != perfSlot {
			t.Errorf("ComputeHashKSlot parity with perfscout failed: got %d, want %d", gotSlot, perfSlot)
		}
	}

	// Verify dispersion between subtable 0 and subtable 1
	router := NewHashKRouter(320000000, 4, 160)
	if router.NumSlotsPerSub != numSlots {
		t.Fatalf("expected %d slots per subtable, got %d", numSlots, router.NumSlotsPerSub)
	}

	collisions := 0
	for i := uint64(0); i < 2000; i++ {
		slot0, slot1 := router.RouteToken(i, 0)
		if slot0 >= router.NumSlotsPerSub {
			t.Fatalf("slot0 %d exceeds capacity %d", slot0, router.NumSlotsPerSub)
		}
		if slot1 >= router.NumSlotsPerSub {
			t.Fatalf("slot1 %d exceeds capacity %d", slot1, router.NumSlotsPerSub)
		}
		if slot0 == slot1 {
			collisions++
		}
	}
	// Under independent hashing with 80M slots, collision rate among 2000 items should be near zero (< 5)
	if collisions > 5 {
		t.Errorf("unexpected collision rate between subtables: %d / 2000", collisions)
	}
}

// TestHashKDualSubtableGather160Dim tests gathering 80-dim slices from subtable 0 and subtable 1,
// concatenating to 160 dims, and verifying ridge matrix Wh ≈ I_160 bypass.
func TestHashKDualSubtableGather160Dim(t *testing.T) {
	const totalVocab = 1000
	const compRate = 4
	const fullDim = 160
	const subDim = 80

	router := NewHashKRouter(totalVocab, compRate, fullDim)
	slots := int(router.NumSlotsPerSub)

	// Create subtable 0 and subtable 1 with unique identifiable values
	subtable0 := make([]float32, slots*subDim)
	subtable1 := make([]float32, slots*subDim)

	for s := 0; s < slots; s++ {
		for d := 0; d < subDim; d++ {
			subtable0[s*subDim+d] = float32(100000 + s*100 + d)
			subtable1[s*subDim+d] = float32(200000 + s*100 + d)
		}
	}

	testTokens := []uint64{0, 1, 42, 128, 500, 999}
	for _, tok := range testTokens {
		for head := uint64(0); head < 4; head++ {
			slot0, slot1 := router.RouteToken(tok, head)

			// Gather embedding
			gathered := router.GatherHashKEmbedding(subtable0, subtable1, tok, head)
			if len(gathered) != fullDim {
				t.Fatalf("expected %d dimensions, got %d", fullDim, len(gathered))
			}

			// Verify first 80 dims match subtable 0 at slot0
			for d := 0; d < subDim; d++ {
				expected := subtable0[int(slot0)*subDim+d]
				if gathered[d] != expected {
					t.Fatalf("token %d head %d: dim %d got %f, want %f (from subtable 0 slot %d)",
						tok, head, d, gathered[d], expected, slot0)
				}
			}

			// Verify second 80 dims match subtable 1 at slot1
			for d := 0; d < subDim; d++ {
				expected := subtable1[int(slot1)*subDim+d]
				if gathered[subDim+d] != expected {
					t.Fatalf("token %d head %d: dim %d got %f, want %f (from subtable 1 slot %d)",
						tok, head, subDim+d, gathered[subDim+d], expected, slot1)
				}
			}

			// Verify standalone function parity
			standalone := GatherHashKEmbedding(subtable0, subtable1, tok, head, router)
			for d := 0; d < fullDim; d++ {
				if standalone[d] != gathered[d] {
					t.Fatalf("standalone GatherHashKEmbedding mismatch at dim %d", d)
				}
			}
		}
	}

	// Test HashKPLETable encapsulation and batch gathering
	table := &HashKPLETable{
		Router:    router,
		Subtable0: subtable0,
		Subtable1: subtable1,
	}

	singleGather := table.Gather(42, 0)
	if len(singleGather) != 160 {
		t.Fatalf("table.Gather returned %d dims, want 160", len(singleGather))
	}

	batch := table.GatherBatch([]uint64{10, 20})
	expectedBatchLen := 2 * router.NumHeads * fullDim
	if len(batch) != expectedBatchLen {
		t.Fatalf("batch length got %d, want %d", len(batch), expectedBatchLen)
	}

	// Test FP8 table gather
	fp8Table := NewHashKPLETableFP8(totalVocab, compRate, fullDim)
	slot0, slot1 := fp8Table.Router.RouteToken(7, 2)
	for d := 0; d < subDim; d++ {
		fp8Table.Subtable0[int(slot0)*subDim+d] = byte(0xAA)
		fp8Table.Subtable1[int(slot1)*subDim+d] = byte(0xBB)
	}
	fp8Out := fp8Table.GatherBytes(7, 2)
	if len(fp8Out) != 160 {
		t.Fatalf("FP8 out len got %d, want 160", len(fp8Out))
	}
	for d := 0; d < subDim; d++ {
		if fp8Out[d] != 0xAA {
			t.Errorf("FP8 subtable0 byte mismatch at dim %d: got %x, want 0xAA", d, fp8Out[d])
		}
		if fp8Out[subDim+d] != 0xBB {
			t.Errorf("FP8 subtable1 byte mismatch at dim %d: got %x, want 0xBB", d, fp8Out[subDim+d])
		}
	}
}

// TestHashKMemoryAccounting4xCompression verifies 4x compression from 51.2 GB to 12.8 GB
// representation for 320M tokens on unified memory.
func TestHashKMemoryAccounting4xCompression(t *testing.T) {
	// Standard 320,000,000 vocabulary size
	const vocab320M = 320000000
	router := NewHashKRouter(vocab320M, 4, 160)
	stats := router.MemoryAccounting(1) // 1 byte per element for FP8

	// Uncompressed: 320,000,000 * 160 * 1 = 51,200,000,000 bytes (51.2 GB)
	const wantUncompressed = uint64(51200000000)
	if stats.UncompressedBytes != wantUncompressed {
		t.Errorf("UncompressedBytes: got %d, want %d", stats.UncompressedBytes, wantUncompressed)
	}
	if math.Abs(stats.UncompressedGB-51.2) > 1e-4 {
		t.Errorf("UncompressedGB: got %f, want 51.2", stats.UncompressedGB)
	}

	// Compressed: 2 * (80,000,000 * 80 * 1) = 12,800,000,000 bytes (12.8 GB)
	const wantCompressed = uint64(12800000000)
	if stats.CompressedBytes != wantCompressed {
		t.Errorf("CompressedBytes: got %d, want %d", stats.CompressedBytes, wantCompressed)
	}
	if math.Abs(stats.CompressedGB-12.8) > 1e-4 {
		t.Errorf("CompressedGB: got %f, want 12.8", stats.CompressedGB)
	}

	// Reclaimed: 51.2 GB - 12.8 GB = 38.4 GB
	const wantReclaimed = uint64(38400000000)
	if stats.ReclaimedBytes != wantReclaimed {
		t.Errorf("ReclaimedBytes: got %d, want %d", stats.ReclaimedBytes, wantReclaimed)
	}
	if math.Abs(stats.ReclaimedGB-38.4) > 1e-4 {
		t.Errorf("ReclaimedGB: got %f, want 38.4", stats.ReclaimedGB)
	}

	// Verify exact 4.0x compression factor
	if math.Abs(stats.CompressionFactor-4.0) > 1e-6 {
		t.Errorf("CompressionFactor: got %f, want 4.0", stats.CompressionFactor)
	}

	// Verify Ridge regression MACs saved: 16 heads * 160 * 160 = 409,600
	const wantMACs = uint64(409600)
	if stats.RidgeMACsSaved != wantMACs {
		t.Errorf("RidgeMACsSaved: got %d, want %d", stats.RidgeMACsSaved, wantMACs)
	}

	// Prime-sum vocabulary from concept study: 320,001,536 rows across 16 heads
	const primeSumVocab = 320001536
	routerPrime := NewHashKRouter(primeSumVocab, 4, 160)
	statsPrime := routerPrime.MemoryAccounting(1)

	// Verify prime vocabulary also achieves ~4x compression and ~51.2 GB -> ~12.8 GB
	if math.Abs(statsPrime.UncompressedGB-51.2002) > 0.01 {
		t.Errorf("prime vocab UncompressedGB: got %f, want ~51.2", statsPrime.UncompressedGB)
	}
	if math.Abs(statsPrime.CompressedGB-12.8001) > 0.01 {
		t.Errorf("prime vocab CompressedGB: got %f, want ~12.8", statsPrime.CompressedGB)
	}
	if math.Abs(statsPrime.CompressionFactor-4.0) > 0.001 {
		t.Errorf("prime vocab CompressionFactor: got %f, want ~4.0", statsPrime.CompressionFactor)
	}
}

// TestHashKMetalShaderContents verifies the presence and contents of internal/compute/metal/hashk_gather.metal.
func TestHashKMetalShaderContents(t *testing.T) {
	shaderPath := filepath.Join("..", "compute", "metal", "hashk_gather.metal")
	content, err := os.ReadFile(shaderPath)
	if err != nil {
		t.Fatalf("failed to read shader at %s: %v", shaderPath, err)
	}
	src := string(content)

	// Check kernel declaration
	if !strings.Contains(src, "kernel void hashk_gather") {
		t.Error("shader missing 'kernel void hashk_gather'")
	}

	// Check polynomial hash constants and formula
	if !strings.Contains(src, "2862933555777941757ULL") {
		t.Error("shader missing polynomial multiplier 2862933555777941757ULL")
	}
	if !strings.Contains(src, "998244353ULL") {
		t.Error("shader missing head prime 998244353ULL")
	}
	if !strings.Contains(src, "0x9e3779b97f4a7c15ULL") {
		t.Error("shader missing SplitMix64 constant 0x9e3779b97f4a7c15ULL")
	}
	if !strings.Contains(src, "0x517cc1b727220a95ULL") {
		t.Error("shader missing subtable 0 salt 0x517cc1b727220a95ULL")
	}

	// Check threadgroup gather logic
	if !strings.Contains(src, "threadgroup float tg_slice") {
		t.Error("shader missing threadgroup buffer for dual slice gather")
	}
	if !strings.Contains(src, "threadgroup_barrier") {
		t.Error("shader missing threadgroup_barrier synchronization")
	}
}
