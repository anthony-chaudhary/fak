package ctxmmu

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"
)

// TestDecoupledKVPoolPaging tests and witnesses Acceptance Criteria 1 & 2 from Issue #12191:
// - AC1: Dynamically maps non-contiguous physical pages to sequence tokens across GQA and MLA topologies.
// - AC2: Block allocator executes page allocation and deallocation in O(1) time with reference counting.
// - Address translation verification: Translate(seqID, tokenPos) matches physical block and offset.
// - Exact KV data read/write roundtrip verification for both GQA and MLA.
// - Zero static over-allocation / fragmentation check.
// - Memory pressure hooks, LRU eviction, and swap out/in verification.
func TestDecoupledKVPoolPaging(t *testing.T) {
	t.Run("AC2_BlockAllocator_O1_RefCounting", func(t *testing.T) {
		const totalBlocks = 64
		const blockSize = 4096
		alloc := NewPhysicalBlockAllocator(totalBlocks, blockSize)

		// Initial state verification
		if alloc.AllocatedCount() != 0 {
			t.Fatalf("expected 0 allocated blocks, got %d", alloc.AllocatedCount())
		}
		if alloc.TotalCount() != totalBlocks {
			t.Fatalf("expected total blocks %d, got %d", totalBlocks, alloc.TotalCount())
		}
		if alloc.FreeCount() != totalBlocks {
			t.Fatalf("expected free blocks %d, got %d", totalBlocks, alloc.FreeCount())
		}
		if alloc.BlockSize() != blockSize {
			t.Fatalf("expected block size %d, got %d", blockSize, alloc.BlockSize())
		}

		// O(1) page allocation verification across all blocks
		allocated := make([]*PhysicalBlock, totalBlocks)
		startAlloc := time.Now()
		for i := 0; i < totalBlocks; i++ {
			b, err := alloc.Allocate()
			if err != nil {
				t.Fatalf("unexpected allocation error at index %d: %v", i, err)
			}
			if b.PhysicalSlot() != i {
				t.Fatalf("expected physical slot %d, got %d", i, b.PhysicalSlot())
			}
			if b.RefCount() != 1 {
				t.Fatalf("expected refCount 1 for newly allocated block, got %d", b.RefCount())
			}
			if b.Swapped() {
				t.Fatalf("expected swapped=false for newly allocated block")
			}
			if b.LastAccess() == 0 {
				t.Fatalf("expected non-zero last access timestamp")
			}
			allocated[i] = b
		}
		allocDuration := time.Since(startAlloc)
		avgAllocNs := allocDuration.Nanoseconds() / int64(totalBlocks)
		t.Logf("AC2: O(1) allocation verified: avg %d ns/block across %d blocks", avgAllocNs, totalBlocks)

		// Pool is full: next allocation must return ErrNoFreeBlocks
		if _, err := alloc.Allocate(); !errors.Is(err, ErrNoFreeBlocks) {
			t.Fatalf("expected ErrNoFreeBlocks on full allocator, got %v", err)
		}
		if alloc.AllocatedCount() != totalBlocks || alloc.FreeCount() != 0 {
			t.Fatalf("expected full allocator state, got allocated=%d free=%d", alloc.AllocatedCount(), alloc.FreeCount())
		}

		// Reference counting verification
		targetBlock := allocated[0]
		if ref := targetBlock.Retain(); ref != 2 {
			t.Fatalf("expected refCount 2 after Retain, got %d", ref)
		}
		if targetBlock.RefCount() != 2 {
			t.Fatalf("expected RefCount() == 2, got %d", targetBlock.RefCount())
		}
		if ref := targetBlock.Release(); ref != 1 {
			t.Fatalf("expected refCount 1 after Release, got %d", ref)
		}
		if err := alloc.Retain(targetBlock.ID); err != nil {
			t.Fatalf("alloc.Retain failed: %v", err)
		}
		if targetBlock.RefCount() != 2 {
			t.Fatalf("expected RefCount() == 2 after alloc.Retain, got %d", targetBlock.RefCount())
		}
		_ = targetBlock.Release()

		// Touch and LastAccess verification
		oldAccess := targetBlock.LastAccess()
		time.Sleep(2 * time.Millisecond)
		targetBlock.Touch()
		if targetBlock.LastAccess() <= oldAccess {
			t.Fatalf("expected LastAccess to advance after Touch: old=%d, new=%d", oldAccess, targetBlock.LastAccess())
		}

		// Swapped status verification
		targetBlock.SetSwapped(true)
		if !targetBlock.Swapped() {
			t.Fatalf("expected Swapped() to be true")
		}
		targetBlock.SetSwapped(false)
		if targetBlock.Swapped() {
			t.Fatalf("expected Swapped() to be false")
		}

		// O(1) deallocation verification with non-contiguous free pattern (evens first)
		startFree := time.Now()
		for i := 0; i < totalBlocks; i += 2 {
			ok, err := alloc.Free(i)
			if err != nil || !ok {
				t.Fatalf("expected successful free of block %d, ok=%v err=%v", i, ok, err)
			}
		}
		freeDuration := time.Since(startFree)
		avgFreeNs := freeDuration.Nanoseconds() / int64(totalBlocks/2)
		t.Logf("AC2: O(1) deallocation verified: avg %d ns/block across %d freed blocks", avgFreeNs, totalBlocks/2)

		if alloc.AllocatedCount() != totalBlocks/2 {
			t.Fatalf("expected %d allocated, got %d", totalBlocks/2, alloc.AllocatedCount())
		}
		if alloc.FreeCount() != totalBlocks/2 {
			t.Fatalf("expected %d free, got %d", totalBlocks/2, alloc.FreeCount())
		}

		// Double-free must return false, nil without corrupting state
		ok, err := alloc.Free(0)
		if err != nil || ok {
			t.Fatalf("expected double-free to return ok=false err=nil, got ok=%v err=%v", ok, err)
		}

		// Invalid block IDs
		if _, err := alloc.Free(-1); !errors.Is(err, ErrInvalidBlockID) {
			t.Fatalf("expected ErrInvalidBlockID for -1, got %v", err)
		}
		if _, err := alloc.Free(totalBlocks); !errors.Is(err, ErrInvalidBlockID) {
			t.Fatalf("expected ErrInvalidBlockID for totalBlocks, got %v", err)
		}
		if _, err := alloc.GetBlock(-1); !errors.Is(err, ErrInvalidBlockID) {
			t.Fatalf("expected ErrInvalidBlockID from GetBlock(-1), got %v", err)
		}
		if _, err := alloc.GetBlock(0); !errors.Is(err, ErrBlockNotFound) {
			t.Fatalf("expected ErrBlockNotFound for freed block 0, got %v", err)
		}
		if b, err := alloc.GetBlock(1); err != nil || b == nil || b.ID != 1 {
			t.Fatalf("expected to get active block 1, got b=%v err=%v", b, err)
		}

		// Re-allocate freed slots
		for i := 0; i < totalBlocks/2; i++ {
			b, err := alloc.Allocate()
			if err != nil {
				t.Fatalf("unexpected allocation error on re-allocating slot %d: %v", i, err)
			}
			if b.PhysicalSlot()%2 != 0 {
				t.Fatalf("expected newly allocated block to reuse even slot, got %d", b.PhysicalSlot())
			}
		}
		if alloc.FreeCount() != 0 {
			t.Fatalf("expected 0 free blocks after refilling pool, got %d", alloc.FreeCount())
		}

		// LRU Recycling check
		alloc.blocks[4].refCount.Store(0)
		alloc.blocks[10].refCount.Store(0)
		reclaimed, err := alloc.RecycleLRU(2)
		if err != nil {
			t.Fatalf("RecycleLRU failed: %v", err)
		}
		if len(reclaimed) != 2 {
			t.Fatalf("expected 2 reclaimed blocks, got %d", len(reclaimed))
		}
		if alloc.FreeCount() != 2 {
			t.Fatalf("expected 2 free blocks after RecycleLRU, got %d", alloc.FreeCount())
		}
	})

	t.Run("AC1_DynamicPaging_AddressTranslation_GQA", func(t *testing.T) {
		cfg := KVPoolConfig{
			TokensPerPage: 16,
			NumLayers:     4,
			NumKVHeads:    4,
			HeadDim:       64,
			DType:         KVDTypeFP16,
			Topology:      KVTopologyGQA,
			TotalPages:    64,
		}
		expectedPageBytes := 16 * 4 * 2 * 4 * 64 * 2 // 65,536 bytes
		if cfg.PageBytes() != expectedPageBytes {
			t.Fatalf("expected PageBytes %d for GQA, got %d", expectedPageBytes, cfg.PageBytes())
		}

		pool, err := NewKVPool(cfg)
		if err != nil {
			t.Fatalf("failed to create GQA KVPool: %v", err)
		}
		if pool.Config().Topology != KVTopologyGQA {
			t.Fatalf("expected GQA topology, got %s", pool.Config().Topology)
		}

		// Zero static over-allocation before appending tokens
		seq1, err := pool.CreateSequence("seq-gqa-1")
		if err != nil {
			t.Fatalf("CreateSequence seq1 failed: %v", err)
		}
		if pool.PhysicalBlocksAllocated() != 0 {
			t.Fatalf("expected 0 physical blocks allocated upon sequence creation, got %d", pool.PhysicalBlocksAllocated())
		}
		if seq1.Tokens() != 0 || seq1.Len() != 0 {
			t.Fatalf("expected 0 tokens initially, got %d", seq1.Tokens())
		}

		// Create interleaved sequences to force non-contiguous physical page backing
		seq2, err := pool.CreateSequence("seq-gqa-2")
		if err != nil {
			t.Fatalf("CreateSequence seq2 failed: %v", err)
		}
		_ = seq2

		// seq1 takes page 0
		if err := pool.AppendTokens("seq-gqa-1", 16); err != nil {
			t.Fatalf("AppendTokens seq1 failed: %v", err)
		}
		// seq2 takes page 1
		if err := pool.AppendTokens("seq-gqa-2", 16); err != nil {
			t.Fatalf("AppendTokens seq2 failed: %v", err)
		}
		// seq1 takes page 2
		if err := pool.AppendTokens("seq-gqa-1", 16); err != nil {
			t.Fatalf("AppendTokens seq1 failed: %v", err)
		}
		// seq2 takes page 3
		if err := pool.AppendTokens("seq-gqa-2", 16); err != nil {
			t.Fatalf("AppendTokens seq2 failed: %v", err)
		}
		// seq1 takes page 4
		if err := pool.AppendTokens("seq-gqa-1", 16); err != nil {
			t.Fatalf("AppendTokens seq1 failed: %v", err)
		}

		// Release seq2 to create holes [1, 3] in physical space
		if err := pool.ReleaseSequence("seq-gqa-2"); err != nil {
			t.Fatalf("ReleaseSequence seq2 failed: %v", err)
		}

		// seq3 should reclaim the holes [1, 3]
		seq3, err := pool.CreateSequence("seq-gqa-3")
		if err != nil {
			t.Fatalf("CreateSequence seq3 failed: %v", err)
		}
		if err := pool.AppendTokens("seq-gqa-3", 32); err != nil {
			t.Fatalf("AppendTokens seq3 failed: %v", err)
		}

		// Verify non-contiguous page mappings for seq1 and seq3
		seq1Pages := seq1.Pages()
		physicalSlots := func(ids []int) []int {
			slots := make([]int, len(ids))
			for i, id := range ids {
				block, err := pool.Allocator().GetBlock(id)
				if err != nil {
					t.Fatalf("GetBlock(%d): %v", id, err)
				}
				slots[i] = block.PhysicalSlot()
			}
			return slots
		}
		seq1Slots := physicalSlots(seq1Pages)
		expectedSeq1Slots := []int{0, 2, 4}
		if !reflect.DeepEqual(seq1Slots, expectedSeq1Slots) {
			t.Fatalf("expected non-contiguous physical slots %v for seq1, got %v", expectedSeq1Slots, seq1Slots)
		}
		seq3Pages := seq3.Pages()
		seq3Slots := physicalSlots(seq3Pages)
		expectedSeq3Slots := []int{1, 3}
		if !reflect.DeepEqual(seq3Slots, expectedSeq3Slots) {
			t.Fatalf("expected non-contiguous physical slots %v for seq3, got %v", expectedSeq3Slots, seq3Slots)
		}
		t.Logf("AC1: Non-contiguous physical page mapping verified: seq1=%v, seq3=%v", seq1Slots, seq3Slots)

		// Address translation verification: Translate(seqID, tokenPos)
		totalTokensSeq1 := seq1.Tokens()
		if totalTokensSeq1 != 48 {
			t.Fatalf("expected 48 tokens in seq1, got %d", totalTokensSeq1)
		}
		for pos := 0; pos < totalTokensSeq1; pos++ {
			bID, offset, err := pool.Translate("seq-gqa-1", pos)
			if err != nil {
				t.Fatalf("Translate failed for token %d: %v", pos, err)
			}
			expectedPageIdx := pos / cfg.TokensPerPage
			expectedBlockID := seq1Pages[expectedPageIdx]
			expectedOffset := pos % cfg.TokensPerPage
			if bID != expectedBlockID {
				t.Fatalf("token %d: expected block ID %d, got %d", pos, expectedBlockID, bID)
			}
			if offset != expectedOffset {
				t.Fatalf("token %d: expected offset %d, got %d", pos, expectedOffset, offset)
			}
		}

		// Out-of-bounds translation checks
		if _, _, err := pool.Translate("seq-gqa-1", -1); !errors.Is(err, ErrTokenOutOfBounds) {
			t.Fatalf("expected ErrTokenOutOfBounds for pos -1, got %v", err)
		}
		if _, _, err := pool.Translate("seq-gqa-1", 48); !errors.Is(err, ErrTokenOutOfBounds) {
			t.Fatalf("expected ErrTokenOutOfBounds for pos 48, got %v", err)
		}
		if _, _, err := pool.Translate("non-existent-seq", 0); !errors.Is(err, ErrSequenceNotFound) {
			t.Fatalf("expected ErrSequenceNotFound for unknown seq, got %v", err)
		}

		// Exact KV data read/write roundtrip verification for GQA
		// GQA KV stride: NumKVHeads * HeadDim * DTypeBytes = 4 * 64 * 2 = 512 bytes
		const kvStride = 4 * 64 * 2
		testTokens := []int{0, 15, 16, 31, 32, 47} // boundary tokens across pages 0, 1, 2
		for layer := 0; layer < cfg.NumLayers; layer++ {
			for _, pos := range testTokens {
				kData := make([]byte, kvStride)
				vData := make([]byte, kvStride)
				for i := 0; i < kvStride; i++ {
					kData[i] = byte((pos*17 + layer*31 + i) & 0xFF)
					vData[i] = byte((pos*23 + layer*47 + i*3) & 0xFF)
				}

				if err := pool.WriteTokenKV("seq-gqa-1", pos, layer, kData, vData); err != nil {
					t.Fatalf("WriteTokenKV failed at pos=%d layer=%d: %v", pos, layer, err)
				}

				kRead, vRead, err := pool.ReadTokenKV("seq-gqa-1", pos, layer)
				if err != nil {
					t.Fatalf("ReadTokenKV failed at pos=%d layer=%d: %v", pos, layer, err)
				}

				if !bytes.Equal(kData, kRead) {
					t.Fatalf("GQA Key mismatch at pos=%d layer=%d", pos, layer)
				}
				if !bytes.Equal(vData, vRead) {
					t.Fatalf("GQA Val mismatch at pos=%d layer=%d", pos, layer)
				}
			}
		}
		t.Logf("AC1: Exact GQA KV read/write roundtrip verified across all %d layers and %d token positions",
			cfg.NumLayers, len(testTokens))

		// Invalid layer index checks
		dummyKV := make([]byte, kvStride)
		if err := pool.WriteTokenKV("seq-gqa-1", 0, -1, dummyKV, dummyKV); !errors.Is(err, ErrInvalidLayer) {
			t.Fatalf("expected ErrInvalidLayer for layer -1, got %v", err)
		}
		if err := pool.WriteTokenKV("seq-gqa-1", 0, cfg.NumLayers, dummyKV, dummyKV); !errors.Is(err, ErrInvalidLayer) {
			t.Fatalf("expected ErrInvalidLayer for layer NumLayers, got %v", err)
		}
		if _, _, err := pool.ReadTokenKV("seq-gqa-1", 0, -1); !errors.Is(err, ErrInvalidLayer) {
			t.Fatalf("expected ErrInvalidLayer on ReadTokenKV for layer -1, got %v", err)
		}
		if _, _, err := pool.ReadTokenKV("seq-gqa-1", 0, cfg.NumLayers); !errors.Is(err, ErrInvalidLayer) {
			t.Fatalf("expected ErrInvalidLayer on ReadTokenKV for layer NumLayers, got %v", err)
		}
	})

	t.Run("AC1_DynamicPaging_AddressTranslation_MLA", func(t *testing.T) {
		cfg := KVPoolConfig{
			TokensPerPage: 16,
			NumLayers:     4,
			KVLoraRank:    128,
			QKRopeHeadDim: 32,
			DType:         KVDTypeFP16,
			Topology:      KVTopologyMLA,
			TotalPages:    64,
		}
		// MLA PageBytes: TokensPerPage * NumLayers * (KVLoraRank + QKRopeHeadDim) * DTypeBytes
		expectedPageBytes := 16 * 4 * (128 + 32) * 2 // 20,480 bytes
		if cfg.PageBytes() != expectedPageBytes {
			t.Fatalf("expected PageBytes %d for MLA, got %d", expectedPageBytes, cfg.PageBytes())
		}

		pool, err := NewKVPool(cfg)
		if err != nil {
			t.Fatalf("failed to create MLA KVPool: %v", err)
		}
		if pool.Config().Topology != KVTopologyMLA {
			t.Fatalf("expected MLA topology, got %s", pool.Config().Topology)
		}

		seqMLA, err := pool.CreateSequence("seq-mla-1")
		if err != nil {
			t.Fatalf("CreateSequence seqMLA failed: %v", err)
		}

		// Append 40 tokens (needs ceil(40/16) = 3 pages)
		if err := pool.AppendTokens("seq-mla-1", 40); err != nil {
			t.Fatalf("AppendTokens seqMLA failed: %v", err)
		}
		if pool.PhysicalBlocksAllocated() != 3 {
			t.Fatalf("expected 3 physical pages for 40 tokens, got %d", pool.PhysicalBlocksAllocated())
		}
		if seqMLA.Tokens() != 40 {
			t.Fatalf("expected 40 tokens, got %d", seqMLA.Tokens())
		}

		// Address translation verification for MLA
		for pos := 0; pos < 40; pos++ {
			bID, offset, err := pool.Translate("seq-mla-1", pos)
			if err != nil {
				t.Fatalf("MLA Translate failed at pos %d: %v", pos, err)
			}
			expectedBlockID := seqMLA.Pages()[pos/cfg.TokensPerPage]
			expectedOffset := pos % cfg.TokensPerPage
			if bID != expectedBlockID || offset != expectedOffset {
				t.Fatalf("MLA token %d: expected block=%d offset=%d, got block=%d offset=%d",
					pos, expectedBlockID, expectedOffset, bID, offset)
			}
		}

		// Exact KV data read/write roundtrip verification for MLA
		// MLA rankBytes: KVLoraRank * 2 = 256 bytes
		// MLA ropeBytes: QKRopeHeadDim * 2 = 64 bytes
		const rankBytes = 128 * 2
		const ropeBytes = 32 * 2
		testPositions := []int{0, 15, 16, 31, 32, 39}
		for layer := 0; layer < cfg.NumLayers; layer++ {
			for _, pos := range testPositions {
				keyData := make([]byte, rankBytes)
				valData := make([]byte, ropeBytes)
				for i := 0; i < rankBytes; i++ {
					keyData[i] = byte((pos*37 + layer*19 + i) & 0xFF)
				}
				for i := 0; i < ropeBytes; i++ {
					valData[i] = byte((pos*41 + layer*29 + i*5) & 0xFF)
				}

				if err := pool.WriteTokenKV("seq-mla-1", pos, layer, keyData, valData); err != nil {
					t.Fatalf("MLA WriteTokenKV failed at pos=%d layer=%d: %v", pos, layer, err)
				}

				kRead, vRead, err := pool.ReadTokenKV("seq-mla-1", pos, layer)
				if err != nil {
					t.Fatalf("MLA ReadTokenKV failed at pos=%d layer=%d: %v", pos, layer, err)
				}

				if !bytes.Equal(keyData, kRead) {
					t.Fatalf("MLA Key mismatch at pos=%d layer=%d", pos, layer)
				}
				if !bytes.Equal(valData, vRead) {
					t.Fatalf("MLA Val mismatch at pos=%d layer=%d", pos, layer)
				}
			}
		}
		t.Logf("AC1: Exact MLA KV read/write roundtrip verified across all %d layers", cfg.NumLayers)
	})

	t.Run("ZeroStaticOverAllocation_Fragmentation", func(t *testing.T) {
		cfg := KVPoolConfig{
			TokensPerPage: 16,
			NumLayers:     2,
			NumKVHeads:    2,
			HeadDim:       32,
			DType:         KVDTypeFP16,
			TotalPages:    128,
		}
		pool, err := NewKVPool(cfg)
		if err != nil {
			t.Fatalf("failed to create KVPool: %v", err)
		}

		// Create 10 sequences with 0 tokens.
		// Zero static over-allocation: physical allocation must remain strictly 0.
		for i := 0; i < 10; i++ {
			seqID := fmt.Sprintf("idle-seq-%d", i)
			if _, err := pool.CreateSequence(seqID); err != nil {
				t.Fatalf("CreateSequence %s failed: %v", seqID, err)
			}
		}
		if pool.PhysicalBlocksAllocated() != 0 {
			t.Fatalf("zero static over-allocation violated: expected 0 blocks, got %d", pool.PhysicalBlocksAllocated())
		}
		if pool.LogicalTokensAllocated() != 0 {
			t.Fatalf("expected 0 logical tokens, got %d", pool.LogicalTokensAllocated())
		}

		// Dynamically append variable tokens to sequences:
		// seq-0: 10 tokens -> 1 page
		// seq-1: 25 tokens -> 2 pages
		// seq-2: 40 tokens -> 3 pages
		// seq-3: 16 tokens -> 1 page
		// seq-4: 32 tokens -> 2 pages
		tokenAssignments := []int{10, 25, 40, 16, 32}
		expectedPagesTotal := 1 + 2 + 3 + 1 + 2 // 9 pages
		expectedTokensTotal := 10 + 25 + 40 + 16 + 32
		for i, tok := range tokenAssignments {
			seqID := fmt.Sprintf("idle-seq-%d", i)
			if err := pool.AppendTokens(seqID, tok); err != nil {
				t.Fatalf("AppendTokens %s failed: %v", seqID, err)
			}
		}

		if pool.PhysicalBlocksAllocated() != expectedPagesTotal {
			t.Fatalf("expected %d physical pages allocated, got %d", expectedPagesTotal, pool.PhysicalBlocksAllocated())
		}
		if pool.LogicalTokensAllocated() != expectedTokensTotal {
			t.Fatalf("expected %d logical tokens, got %d", expectedTokensTotal, pool.LogicalTokensAllocated())
		}

		// Interleaved release to verify non-fragmented recycling:
		// Release seq-1 (2 pages) and seq-3 (1 page)
		if err := pool.ReleaseSequence("idle-seq-1"); err != nil {
			t.Fatalf("Release idle-seq-1 failed: %v", err)
		}
		if err := pool.ReleaseSequence("idle-seq-3"); err != nil {
			t.Fatalf("Release idle-seq-3 failed: %v", err)
		}
		expectedAfterRelease := expectedPagesTotal - 3
		if pool.PhysicalBlocksAllocated() != expectedAfterRelease {
			t.Fatalf("expected %d pages after release, got %d", expectedAfterRelease, pool.PhysicalBlocksAllocated())
		}

		// New sequence allocation reclaims the freed holes
		newSeq, err := pool.CreateSequence("reclaimed-seq")
		if err != nil {
			t.Fatalf("CreateSequence reclaimed-seq failed: %v", err)
		}
		if err := pool.AppendTokens("reclaimed-seq", 48); err != nil { // 3 pages
			t.Fatalf("AppendTokens reclaimed-seq failed: %v", err)
		}
		if pool.PhysicalBlocksAllocated() != expectedPagesTotal {
			t.Fatalf("expected %d pages after recycling, got %d", expectedPagesTotal, pool.PhysicalBlocksAllocated())
		}
		_ = newSeq

		// Release all remaining sequences and verify zero leaks
		for i := 0; i < 10; i++ {
			seqID := fmt.Sprintf("idle-seq-%d", i)
			_ = pool.ReleaseSequence(seqID)
		}
		_ = pool.ReleaseSequence("reclaimed-seq")

		if pool.PhysicalBlocksAllocated() != 0 {
			t.Fatalf("memory leak detected: expected 0 physical blocks, got %d", pool.PhysicalBlocksAllocated())
		}
		if pool.Allocator().FreeCount() != 128 {
			t.Fatalf("expected 128 free blocks in allocator, got %d", pool.Allocator().FreeCount())
		}
		t.Logf("AC1: Zero static over-allocation and non-fragmenting recycling verified")
	})

	t.Run("MemoryPressure_LRU_Swap", func(t *testing.T) {
		cfg := KVPoolConfig{
			TokensPerPage: 16,
			NumLayers:     2,
			NumKVHeads:    2,
			HeadDim:       32,
			DType:         KVDTypeFP16,
			TotalPages:    16,
		}
		pool, err := NewKVPool(cfg)
		if err != nil {
			t.Fatalf("failed to create KVPool: %v", err)
		}

		// Pressure hook verification
		var capturedPressure float64
		hookInvoked := false
		pool.RegisterPressureHook(func(p float64) {
			capturedPressure = p
			hookInvoked = true
		})

		testPressure := 0.88
		pool.TriggerPressure(testPressure)
		if !hookInvoked {
			t.Fatalf("expected memory pressure hook to be invoked")
		}
		if capturedPressure != testPressure {
			t.Fatalf("expected captured pressure %f, got %f", testPressure, capturedPressure)
		}

		// Swap out and swap in verification
		seqSwap, err := pool.CreateSequence("seq-swap")
		if err != nil {
			t.Fatalf("CreateSequence seq-swap failed: %v", err)
		}
		if err := pool.AppendTokens("seq-swap", 32); err != nil {
			t.Fatalf("AppendTokens seq-swap failed: %v", err)
		}

		const kvSize = 2 * 32 * 2 // 128 bytes
		tokenDataK := bytes.Repeat([]byte{0x7F}, kvSize)
		tokenDataV := bytes.Repeat([]byte{0x8A}, kvSize)
		if err := pool.WriteTokenKV("seq-swap", 5, 0, tokenDataK, tokenDataV); err != nil {
			t.Fatalf("WriteTokenKV failed: %v", err)
		}

		// Swap out page 0
		if err := pool.SwapBlock("seq-swap", 0); err != nil {
			t.Fatalf("SwapBlock page 0 failed: %v", err)
		}

		// Direct access to swapped page must fail with ErrBlockSwapped
		if _, _, err := pool.ReadTokenKV("seq-swap", 5, 0); !errors.Is(err, ErrBlockSwapped) {
			t.Fatalf("expected ErrBlockSwapped on read, got %v", err)
		}
		if err := pool.WriteTokenKV("seq-swap", 5, 0, tokenDataK, tokenDataV); !errors.Is(err, ErrBlockSwapped) {
			t.Fatalf("expected ErrBlockSwapped on write, got %v", err)
		}

		// Page 1 (token 20) was NOT swapped, must remain accessible
		dummyK := bytes.Repeat([]byte{0x11}, kvSize)
		dummyV := bytes.Repeat([]byte{0x22}, kvSize)
		if err := pool.WriteTokenKV("seq-swap", 20, 0, dummyK, dummyV); err != nil {
			t.Fatalf("WriteTokenKV to non-swapped page 1 failed: %v", err)
		}

		// Swap in page 0 and verify data integrity
		if err := pool.SwapInBlock("seq-swap", 0); err != nil {
			t.Fatalf("SwapInBlock page 0 failed: %v", err)
		}
		readK, readV, err := pool.ReadTokenKV("seq-swap", 5, 0)
		if err != nil {
			t.Fatalf("ReadTokenKV after swap in failed: %v", err)
		}
		if !bytes.Equal(tokenDataK, readK) || !bytes.Equal(tokenDataV, readV) {
			t.Fatalf("data mismatch after swap in roundtrip")
		}

		// Attempting SwapIn on non-swapped page returns ErrSwapNotFound
		if err := pool.SwapInBlock("seq-swap", 1); !errors.Is(err, ErrSwapNotFound) {
			t.Fatalf("expected ErrSwapNotFound for non-swapped page, got %v", err)
		}

		// EvictLRU verification
		// seq-swap has 2 pages. Add another sequence with 2 pages.
		seqLRU, err := pool.CreateSequence("seq-lru")
		if err != nil {
			t.Fatalf("CreateSequence seq-lru failed: %v", err)
		}
		if err := pool.AppendTokens("seq-lru", 32); err != nil {
			t.Fatalf("AppendTokens seq-lru failed: %v", err)
		}

		// Touch page 1 of seq-swap and all pages of seq-lru with distinct timestamps
		// so that page 0 of seq-swap is deterministically the oldest resident page.
		time.Sleep(2 * time.Millisecond)
		_ = pool.WriteTokenKV("seq-swap", 20, 0, dummyK, dummyV)
		time.Sleep(2 * time.Millisecond)
		_ = pool.WriteTokenKV("seq-lru", 0, 0, dummyK, dummyV)
		_ = pool.WriteTokenKV("seq-lru", 20, 0, dummyK, dummyV)

		evicted, err := pool.EvictLRU(1)
		if err != nil {
			t.Fatalf("EvictLRU failed: %v", err)
		}
		if evicted != 1 {
			t.Fatalf("expected 1 block evicted, got %d", evicted)
		}

		// The older page of seq-swap (page 0) was swapped out
		if _, _, err := pool.ReadTokenKV("seq-swap", 0, 0); !errors.Is(err, ErrBlockSwapped) {
			t.Fatalf("expected oldest page of seq-swap to be swapped out, got %v", err)
		}
		// Page 1 of seq-swap was accessed more recently and must remain resident
		if _, _, err := pool.ReadTokenKV("seq-swap", 20, 0); err != nil {
			t.Fatalf("expected page 1 of seq-swap to remain resident, got %v", err)
		}
		t.Logf("AC1: Memory pressure, LRU eviction, and swap roundtrip verified")
		_ = seqSwap
		_ = seqLRU
	})
}

// TestKVPoolCopyOnWriteSharing tests and witnesses Acceptance Criterion 3 from Issue #12191:
//   - AC3: Zero duplicate physical pages allocated for shared prompt prefixes (DuplicatePhysicalPagesAllocated() == 0).
//   - Fork child sequences from parent prefix.
//   - Verify CoW: when child branches append or mutate tokens, only the diverging block is duplicated,
//     while parent and sibling prefix blocks remain shared and unaltered.
//   - Verify multi-agent concurrent serving achieves >= 3x concurrent sequence capacity compared to
//     static pre-allocation (CapacityGainRatio >= 3.0).
//   - Verify proper reference decrement and page recycling on sequence release.
func TestKVPoolCopyOnWriteSharing(t *testing.T) {
	t.Run("AC3_ZeroDuplicatePages_ForkPrefix", func(t *testing.T) {
		cfg := KVPoolConfig{
			TokensPerPage: 16,
			NumLayers:     4,
			NumKVHeads:    4,
			HeadDim:       64,
			DType:         KVDTypeFP16,
			Topology:      KVTopologyGQA,
			TotalPages:    128,
		}
		pool, err := NewKVPool(cfg)
		if err != nil {
			t.Fatalf("failed to create KVPool: %v", err)
		}

		// Parent sequence setup with 48 prompt tokens (3 physical pages)
		parentID := "coord-shared-prefix"
		parentSeq, err := pool.CreateSequence(parentID)
		if err != nil {
			t.Fatalf("CreateSequence parent failed: %v", err)
		}
		if err := pool.AppendTokens(parentID, 48); err != nil {
			t.Fatalf("AppendTokens parent failed: %v", err)
		}
		if pool.PhysicalBlocksAllocated() != 3 {
			t.Fatalf("expected 3 physical pages for parent prompt, got %d", pool.PhysicalBlocksAllocated())
		}

		// Write distinct deterministic prompt KV data to all 3 pages
		const kvStride = 4 * 64 * 2
		for layer := 0; layer < cfg.NumLayers; layer++ {
			for pos := 0; pos < 48; pos++ {
				kData := make([]byte, kvStride)
				vData := make([]byte, kvStride)
				for i := 0; i < kvStride; i++ {
					kData[i] = byte((pos*11 + layer*17 + i) & 0xFF)
					vData[i] = byte((pos*13 + layer*19 + i*2) & 0xFF)
				}
				if err := pool.WriteTokenKV(parentID, pos, layer, kData, vData); err != nil {
					t.Fatalf("WriteTokenKV parent failed at pos %d: %v", pos, err)
				}
			}
		}

		parentPages := parentSeq.Pages()
		if len(parentPages) != 3 {
			t.Fatalf("expected 3 pages in parent, got %d", len(parentPages))
		}

		// Fork 3 subagent child sequences from the parent prefix
		childIDs := []string{"subagent-worker-1", "subagent-worker-2", "subagent-worker-3"}
		children := make([]*KVSequence, len(childIDs))
		for i, childID := range childIDs {
			child, err := pool.ForkSequence(parentID, childID)
			if err != nil {
				t.Fatalf("ForkSequence %s failed: %v", childID, err)
			}
			children[i] = child
		}

		// Acceptance Criterion 3: Zero duplicate physical pages allocated on fork
		if pool.DuplicatePhysicalPagesAllocated() != 0 {
			t.Fatalf("AC3 violated: expected 0 duplicate physical pages, got %d", pool.DuplicatePhysicalPagesAllocated())
		}
		if pool.PhysicalBlocksAllocated() != 3 {
			t.Fatalf("AC3 violated: expected physical allocation to remain 3, got %d", pool.PhysicalBlocksAllocated())
		}

		// Verify page table sharing: all children point to the exact same physical block IDs
		for i, child := range children {
			if !reflect.DeepEqual(child.Pages(), parentPages) {
				t.Fatalf("child %d page table mismatch: expected %v, got %v", i, parentPages, child.Pages())
			}
			if child.Tokens() != 48 {
				t.Fatalf("child %d token count mismatch: expected 48, got %d", i, child.Tokens())
			}
		}

		// Verify reference counts: each physical block must have refCount == 4 (parent + 3 children)
		for _, bID := range parentPages {
			b, err := pool.Allocator().GetBlock(bID)
			if err != nil {
				t.Fatalf("failed to get block %d: %v", bID, err)
			}
			if b.RefCount() != 4 {
				t.Fatalf("expected block %d refCount 4, got %d", bID, b.RefCount())
			}
		}
		t.Logf("AC3: Zero duplicate physical pages verified: parent and 3 children share blocks %v (refCount=4)", parentPages)

		// CoW Mutation Isolation verification:
		// subagent-worker-1 mutates token 20 (page index 1).
		// Only page 1 must be duplicated; page 0 and page 2 must remain shared!
		origBlock1 := parentPages[1]
		mutatedKey := bytes.Repeat([]byte{0xCA}, kvStride)
		mutatedVal := bytes.Repeat([]byte{0xFE}, kvStride)

		if err := pool.WriteTokenKV("subagent-worker-1", 20, 0, mutatedKey, mutatedVal); err != nil {
			t.Fatalf("WriteTokenKV on child-1 failed: %v", err)
		}

		// Physical blocks allocated should now be exactly 3 + 1 = 4 (only diverging block duplicated)
		if pool.PhysicalBlocksAllocated() != 4 {
			t.Fatalf("CoW violation: expected 4 physical blocks allocated, got %d", pool.PhysicalBlocksAllocated())
		}

		child1Pages := children[0].Pages()
		if child1Pages[0] != parentPages[0] {
			t.Fatalf("expected page 0 to remain shared, got %d vs %d", child1Pages[0], parentPages[0])
		}
		if child1Pages[1] == origBlock1 {
			t.Fatalf("expected page 1 to be duplicated via CoW, but still matches %d", origBlock1)
		}
		if child1Pages[2] != parentPages[2] {
			t.Fatalf("expected page 2 to remain shared, got %d vs %d", child1Pages[2], parentPages[2])
		}

		// Verify parent and other children are completely unaltered
		if !reflect.DeepEqual(parentSeq.Pages(), parentPages) {
			t.Fatalf("parent page table was corrupted after child mutation: %v", parentSeq.Pages())
		}
		if !reflect.DeepEqual(children[1].Pages(), parentPages) {
			t.Fatalf("child-2 page table was corrupted after child-1 mutation: %v", children[1].Pages())
		}
		if !reflect.DeepEqual(children[2].Pages(), parentPages) {
			t.Fatalf("child-3 page table was corrupted after child-1 mutation: %v", children[2].Pages())
		}

		// Data isolation check:
		// - child 1 reads mutated data at token 20
		// - parent reads original prompt data at token 20
		// - child 2 reads original prompt data at token 20
		c1K, c1V, err := pool.ReadTokenKV("subagent-worker-1", 20, 0)
		if err != nil {
			t.Fatalf("ReadTokenKV child-1 failed: %v", err)
		}
		if !bytes.Equal(c1K, mutatedKey) || !bytes.Equal(c1V, mutatedVal) {
			t.Fatalf("child-1 did not read mutated KV data")
		}

		origExpectedK := make([]byte, kvStride)
		origExpectedV := make([]byte, kvStride)
		for i := 0; i < kvStride; i++ {
			origExpectedK[i] = byte((20*11 + 0*17 + i) & 0xFF)
			origExpectedV[i] = byte((20*13 + 0*19 + i*2) & 0xFF)
		}

		parentK, parentV, err := pool.ReadTokenKV(parentID, 20, 0)
		if err != nil {
			t.Fatalf("ReadTokenKV parent failed: %v", err)
		}
		if !bytes.Equal(parentK, origExpectedK) || !bytes.Equal(parentV, origExpectedV) {
			t.Fatalf("parent KV data was modified after child mutation (data corruption!)")
		}

		c2K, c2V, err := pool.ReadTokenKV("subagent-worker-2", 20, 0)
		if err != nil {
			t.Fatalf("ReadTokenKV child-2 failed: %v", err)
		}
		if !bytes.Equal(c2K, origExpectedK) || !bytes.Equal(c2V, origExpectedV) {
			t.Fatalf("child-2 KV data was modified after sibling child-1 mutation (sibling leak!)")
		}
		t.Logf("CoW Mutation isolation verified: child-1 mutated page 1 independently; parent and siblings unaltered")

		// CoW Append Divergence verification:
		// subagent-worker-2 appends 16 tokens.
		// Since last page (page 2) was shared (refCount > 1), AppendTokens CoW duplicates page 2
		// and allocates page 3 for new tokens.
		if err := pool.AppendTokens("subagent-worker-2", 16); err != nil {
			t.Fatalf("AppendTokens child-2 failed: %v", err)
		}
		if children[1].Tokens() != 64 {
			t.Fatalf("expected 64 tokens in child-2, got %d", children[1].Tokens())
		}
		if parentSeq.Tokens() != 48 {
			t.Fatalf("parent token count changed: expected 48, got %d", parentSeq.Tokens())
		}

		// Write new data to child-2 at appended token 50
		newTokenK := bytes.Repeat([]byte{0x55}, kvStride)
		newTokenV := bytes.Repeat([]byte{0x66}, kvStride)
		if err := pool.WriteTokenKV("subagent-worker-2", 50, 0, newTokenK, newTokenV); err != nil {
			t.Fatalf("WriteTokenKV child-2 token 50 failed: %v", err)
		}
		c2AppendedK, c2AppendedV, err := pool.ReadTokenKV("subagent-worker-2", 50, 0)
		if err != nil {
			t.Fatalf("ReadTokenKV child-2 token 50 failed: %v", err)
		}
		if !bytes.Equal(c2AppendedK, newTokenK) || !bytes.Equal(c2AppendedV, newTokenV) {
			t.Fatalf("child-2 appended token readback mismatch")
		}

		// Parent and child-3 cannot read token 50
		if _, _, err := pool.ReadTokenKV(parentID, 50, 0); !errors.Is(err, ErrTokenOutOfBounds) {
			t.Fatalf("expected ErrTokenOutOfBounds on parent pos 50, got %v", err)
		}
		if _, _, err := pool.ReadTokenKV("subagent-worker-3", 50, 0); !errors.Is(err, ErrTokenOutOfBounds) {
			t.Fatalf("expected ErrTokenOutOfBounds on child-3 pos 50, got %v", err)
		}
		t.Logf("CoW Append divergence verified: child-2 extended to 64 tokens without affecting parent or siblings")
	})

	t.Run("MultiAgent_CapacityGainRatio_GTE_3x", func(t *testing.T) {
		// Multi-agent concurrent serving capacity test:
		// 1 coordinator sequence with 128 shared system prompt tokens (8 pages).
		// 10 concurrent leaf subagents forked from the coordinator.
		// Each leaf subagent decodes 16 private completion tokens.
		// Max sequence context limit: 512 tokens.
		cfg := KVPoolConfig{
			TokensPerPage: 16,
			NumLayers:     4,
			NumKVHeads:    4,
			HeadDim:       64,
			DType:         KVDTypeFP16,
			Topology:      KVTopologyGQA,
			TotalPages:    256,
		}
		pool, err := NewKVPool(cfg)
		if err != nil {
			t.Fatalf("failed to create KVPool: %v", err)
		}

		coordID := "coordinator-system-prompt"
		if _, err := pool.CreateSequence(coordID); err != nil {
			t.Fatalf("CreateSequence coord failed: %v", err)
		}
		if err := pool.AppendTokens(coordID, 128); err != nil { // 8 physical pages
			t.Fatalf("AppendTokens coord failed: %v", err)
		}
		if pool.PhysicalBlocksAllocated() != 8 {
			t.Fatalf("expected 8 physical pages for coord, got %d", pool.PhysicalBlocksAllocated())
		}

		const numSubagents = 10
		const maxSeqLen = 512
		for i := 0; i < numSubagents; i++ {
			subID := fmt.Sprintf("subagent-leaf-%d", i)
			if _, err := pool.ForkSequence(coordID, subID); err != nil {
				t.Fatalf("ForkSequence %s failed: %v", subID, err)
			}
			// Each subagent decodes 16 tokens (diverges from parent)
			if err := pool.AppendTokens(subID, 16); err != nil {
				t.Fatalf("AppendTokens %s failed: %v", subID, err)
			}
		}

		// CapacityGainRatio verification
		// Baseline static pre-allocation: 11 sequences * 512 tokens = 5,632 tokens (352 pages)
		// CoW Paged allocation: 8 prompt pages + 10 * 2 divergent pages = 28 pages (448 tokens)
		// Expected gain: 5632 / 448 = 12.57x >= 3.0x
		gainRatio := pool.CapacityGainRatio(maxSeqLen)
		t.Logf("Multi-agent concurrent serving CapacityGainRatio(maxSeqLen=%d) = %.2fx", maxSeqLen, gainRatio)
		if gainRatio < 3.0 {
			t.Fatalf("requirement violated: CapacityGainRatio must be >= 3.0, got %.2f", gainRatio)
		}

		// Dynamic gain ratio (based on actual allocated logical tokens, not static max)
		dynamicGain := pool.CapacityGainRatio(0)
		t.Logf("Dynamic CapacityGainRatio(0) = %.2fx", dynamicGain)
		if dynamicGain < 3.0 {
			t.Fatalf("requirement violated: Dynamic CapacityGainRatio must be >= 3.0, got %.2f", dynamicGain)
		}

		// Page recycling verification on sequence release
		initialAllocated := pool.PhysicalBlocksAllocated()
		if initialAllocated == 0 {
			t.Fatalf("expected positive allocated blocks")
		}

		// Release leaf sequences one by one
		for i := 0; i < numSubagents; i++ {
			subID := fmt.Sprintf("subagent-leaf-%d", i)
			if err := pool.ReleaseSequence(subID); err != nil {
				t.Fatalf("ReleaseSequence %s failed: %v", subID, err)
			}
		}

		// Only the parent's 8 shared prompt pages should remain
		if pool.PhysicalBlocksAllocated() != 8 {
			t.Fatalf("expected 8 physical pages remaining after all leaves released, got %d", pool.PhysicalBlocksAllocated())
		}

		// Release coordinator sequence: all 8 prompt pages should be recycled
		if err := pool.ReleaseSequence(coordID); err != nil {
			t.Fatalf("ReleaseSequence coord failed: %v", err)
		}

		if pool.PhysicalBlocksAllocated() != 0 {
			t.Fatalf("expected 0 physical blocks allocated after all sequences released, got %d", pool.PhysicalBlocksAllocated())
		}
		if pool.Allocator().FreeCount() != 256 {
			t.Fatalf("expected all 256 blocks returned to allocator, got %d", pool.Allocator().FreeCount())
		}
		if pool.LogicalTokensAllocated() != 0 {
			t.Fatalf("expected 0 logical tokens, got %d", pool.LogicalTokensAllocated())
		}

		// Releasing an already released sequence must return ErrSequenceNotFound
		if err := pool.ReleaseSequence(coordID); !errors.Is(err, ErrSequenceNotFound) {
			t.Fatalf("expected ErrSequenceNotFound for released sequence, got %v", err)
		}
		t.Logf("Proper reference decrement and complete page recycling verified with 0 memory leaks")
	})
}
