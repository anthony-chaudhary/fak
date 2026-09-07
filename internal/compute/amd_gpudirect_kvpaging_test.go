// Package compute implements hardware abstraction, tensor computation, memory slab management,
// and zero-copy device interconnect acceleration for the fak agent kernel.
package compute

import (
	"errors"
	"testing"
	"time"
)

func TestBaMKVPaging_1MTokenDirectStreaming(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
	cfg := BaMKVPagingConfig{
		NodeID:            0,
		TotalModelLayers:  32,
		TokensPerBlock:    1000,
		BytesPerBlock:     64 * 1024,
		MaxResidentFrames: 64,
		TotalNVMeLBAs:     1024 * 1024,
		SectorSizeBytes:   4096,
		QueueDepth:        128,
		PrefetchDistance:  2,
	}

	coord, err := NewBaMKVPagingCoordinator(hal, cfg)
	if err != nil {
		t.Fatalf("failed to create coordinator: %v", err)
	}

	const totalBlocks = 1000
	blockIDs := make([]uint64, totalBlocks)

	for i := 0; i < totalBlocks; i++ {
		layerID := i % cfg.TotalModelLayers
		data := make([]byte, cfg.BytesPerBlock)
		data[0] = byte(i & 0xFF)
		data[1] = byte((i >> 8) & 0xFF)
		data[len(data)-1] = 0xAA

		bid, err := coord.AllocateBlock(layerID, cfg.TokensPerBlock, data)
		if err != nil {
			t.Fatalf("failed to allocate block %d: %v", i, err)
		}
		blockIDs[i] = bid
	}

	if totalTokens := coord.TotalTokens(); totalTokens != 1_000_000 {
		t.Fatalf("expected 1,000,000 total tokens, got %d", totalTokens)
	}

	allDone := (coord.TotalTokens() == 1_000_000)
	if !allDone {
		t.Fatalf("expected allDone true")
	}

	stats := coord.Stats()
	if stats.TotalBlocks != totalBlocks {
		t.Fatalf("expected %d total blocks, got %d", totalBlocks, stats.TotalBlocks)
	}
	if stats.ResidentBlocks > uint64(cfg.MaxResidentFrames) {
		t.Fatalf("resident blocks %d exceeds max resident frames %d", stats.ResidentBlocks, cfg.MaxResidentFrames)
	}
	if stats.OffloadedBlocks == 0 {
		t.Fatalf("expected offloaded blocks for 1000 blocks with 64 resident frames")
	}

	coldPrefixCount := 100
	coldIDs := blockIDs[:coldPrefixCount]
	if err := coord.BufferColdPrefix(coldIDs); err != nil {
		t.Fatalf("BufferColdPrefix failed: %v", err)
	}

	stats = coord.Stats()
	if stats.ColdPrefixBlocks != uint64(coldPrefixCount) {
		t.Fatalf("expected %d cold prefix blocks, got %d", coldPrefixCount, stats.ColdPrefixBlocks)
	}

	metrics, err := coord.SimulateLayerComputeWithPrefetch(0, 500000)
	if err != nil {
		t.Fatalf("SimulateLayerComputeWithPrefetch failed: %v", err)
	}
	if !metrics.HidingAchieved {
		t.Fatalf("expected prefetch hiding achieved, overlap: %.2f%%", metrics.OverlapPercentage)
	}
	if metrics.OverlapPercentage < 90.0 {
		t.Fatalf("expected overlap >= 90.0%%, got %.2f%%", metrics.OverlapPercentage)
	}

	if coord.StagingCopyCount() != 0 {
		t.Fatalf("coordinator StagingCopyCount() = %d, want 0", coord.StagingCopyCount())
	}
	if stats.StagingCopies != 0 {
		t.Fatalf("stats StagingCopies = %d, want 0", stats.StagingCopies)
	}

	for i, bid := range blockIDs {
		if i%20 == 0 || i < 10 || i >= 990 {
			ok, err := coord.VerifyDataIntegrity(bid)
			if err != nil {
				t.Fatalf("VerifyDataIntegrity on block %d failed: %v", bid, err)
			}
			if !ok {
				t.Fatalf("data corruption detected on block %d", bid)
			}
		}
	}
}

func TestBaMKVPaging_LBAAllocation(t *testing.T) {
	table, err := NewLBAAllocationTable(1024, 4096)
	if err != nil {
		t.Fatalf("NewLBAAllocationTable failed: %v", err)
	}

	if table.SectorSizeBytes() != 4096 {
		t.Errorf("SectorSizeBytes = %d, want 4096", table.SectorSizeBytes())
	}
	if table.TotalCount() != 1024 {
		t.Errorf("TotalCount = %d, want 1024", table.TotalCount())
	}

	start1, count1, err := table.Allocate(1, 64*1024)
	if err != nil {
		t.Fatalf("Allocate block 1 failed: %v", err)
	}
	if start1 != 0 || count1 != 16 {
		t.Errorf("block 1: got start=%d count=%d, want start=0 count=16", start1, count1)
	}

	start2, count2, err := table.Allocate(2, 32*1024)
	if err != nil {
		t.Fatalf("Allocate block 2 failed: %v", err)
	}
	if start2 != 16 || count2 != 8 {
		t.Errorf("block 2: got start=%d count=%d, want start=16 count=8", start2, count2)
	}

	if table.AllocatedCount() != 24 {
		t.Errorf("AllocatedCount = %d, want 24", table.AllocatedCount())
	}
	if table.FreeCount() != 1000 {
		t.Errorf("FreeCount = %d, want 1000", table.FreeCount())
	}

	lba, ok := table.GetLBA(1)
	if !ok || lba != 0 {
		t.Errorf("GetLBA(1) = (%d, %v), want (0, true)", lba, ok)
	}

	_, ok = table.GetLBA(999)
	if ok {
		t.Errorf("GetLBA(999) succeeded, want false")
	}

	if err := table.Free(1); err != nil {
		t.Fatalf("Free(1) failed: %v", err)
	}
	if table.AllocatedCount() != 8 {
		t.Errorf("AllocatedCount after free = %d, want 8", table.AllocatedCount())
	}

	start3, count3, err := table.Allocate(3, 64*1024)
	if err != nil {
		t.Fatalf("Allocate block 3 failed: %v", err)
	}
	if start3 != 0 || count3 != 16 {
		t.Errorf("block 3 reuse: got start=%d count=%d, want start=0 count=16", start3, count3)
	}

	_, _, err = table.Allocate(4, 2000*4096)
	if err == nil {
		t.Errorf("expected out of space error, got nil")
	}
}

func TestBaMKVPaging_QueueExhaustion(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
	pipeline, err := NewBaMIOPipeline(hal, 4)
	if err != nil {
		t.Fatalf("NewBaMIOPipeline failed: %v", err)
	}

	if pipeline.StagingCopyCount() != 0 {
		t.Errorf("pipeline StagingCopyCount = %d, want 0", pipeline.StagingCopyCount())
	}

	for i := 0; i < 4; i++ {
		_, err := pipeline.SubmitRead(uint64(i*16), 16, uintptr(0x8000000000+i*4096), 4096)
		if err != nil {
			t.Fatalf("SubmitRead %d failed: %v", i, err)
		}
	}

	if pipeline.InFlightCount() != 4 {
		t.Fatalf("InFlightCount = %d, want 4", pipeline.InFlightCount())
	}

	_, err = pipeline.SubmitRead(100, 16, 0x8000010000, 4096)
	if !errors.Is(err, ErrQueueExhausted) {
		t.Fatalf("expected ErrQueueExhausted, got %v", err)
	}
	if pipeline.ExhaustionCount() != 1 {
		t.Errorf("ExhaustionCount = %d, want 1", pipeline.ExhaustionCount())
	}

	_, err = pipeline.SubmitWrite(200, 16, 0x8000020000, 4096)
	if !errors.Is(err, ErrQueueExhausted) {
		t.Fatalf("expected ErrQueueExhausted on write, got %v", err)
	}
	if pipeline.ExhaustionCount() != 2 {
		t.Errorf("ExhaustionCount = %d, want 2", pipeline.ExhaustionCount())
	}

	resolved := pipeline.PollCompletions(2)
	if resolved != 2 {
		t.Fatalf("PollCompletions(2) = %d, want 2", resolved)
	}
	if pipeline.InFlightCount() != 2 {
		t.Errorf("InFlightCount after poll = %d, want 2", pipeline.InFlightCount())
	}

	_, err = pipeline.SubmitRead(300, 16, 0x8000030000, 4096)
	if err != nil {
		t.Fatalf("SubmitRead after poll failed: %v", err)
	}

	resolvedAll := pipeline.PollCompletions(0)
	if resolvedAll != 3 {
		t.Errorf("PollCompletions(0) = %d, want 3", resolvedAll)
	}
	if pipeline.InFlightCount() != 0 {
		t.Errorf("InFlightCount after drain = %d, want 0", pipeline.InFlightCount())
	}
}

func TestBaMKVPaging_PrefetchHitRate(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
	cfg := BaMKVPagingConfig{
		TotalModelLayers:  8,
		TokensPerBlock:    512,
		BytesPerBlock:     16 * 1024,
		MaxResidentFrames: 4,
		TotalNVMeLBAs:     1024,
		PrefetchDistance:  1,
	}

	coord, err := NewBaMKVPagingCoordinator(hal, cfg)
	if err != nil {
		t.Fatalf("NewBaMKVPagingCoordinator failed: %v", err)
	}

	b0_1, err := coord.AllocateBlock(0, 512, make([]byte, cfg.BytesPerBlock))
	if err != nil {
		t.Fatalf("allocate layer 0 failed: %v", err)
	}
	b0_2, err := coord.AllocateBlock(0, 512, make([]byte, cfg.BytesPerBlock))
	if err != nil {
		t.Fatalf("allocate layer 0 failed: %v", err)
	}
	_ = b0_1
	_ = b0_2

	b1_1, err := coord.AllocateBlock(1, 512, make([]byte, cfg.BytesPerBlock))
	if err != nil {
		t.Fatalf("allocate layer 1 failed: %v", err)
	}
	b1_2, err := coord.AllocateBlock(1, 512, make([]byte, cfg.BytesPerBlock))
	if err != nil {
		t.Fatalf("allocate layer 1 failed: %v", err)
	}
	_ = b1_1
	_ = b1_2

	b2_1, err := coord.AllocateBlock(2, 512, make([]byte, cfg.BytesPerBlock))
	if err != nil {
		t.Fatalf("allocate layer 2 failed: %v", err)
	}
	b2_2, err := coord.AllocateBlock(2, 512, make([]byte, cfg.BytesPerBlock))
	if err != nil {
		t.Fatalf("allocate layer 2 failed: %v", err)
	}
	_ = b2_1
	_ = b2_2

	if err := coord.PrefetchLayerAhead(0, 1); err != nil {
		t.Fatalf("PrefetchLayerAhead(0, 1) failed: %v", err)
	}

	stats := coord.Stats()
	if stats.PrefetchHits == 0 && stats.PrefetchMisses == 0 {
		t.Errorf("expected prefetch attempts recorded, got 0")
	}

	if err := coord.PrefetchLayerAhead(1, 7); err != nil {
		t.Errorf("expected nil for out of bounds layer, got %v", err)
	}

	stats = coord.Stats()
	if stats.PrefetchHitRate < 0.0 || stats.PrefetchHitRate > 1.0 {
		t.Errorf("invalid PrefetchHitRate: %f", stats.PrefetchHitRate)
	}
}

func TestBaMKVPaging_ErrorHandling(t *testing.T) {
	_, err := NewBaMKVPagingCoordinator(nil, BaMKVPagingConfig{})
	if err == nil {
		t.Errorf("expected error for nil HAL, got nil")
	}

	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
	cfg := BaMKVPagingConfig{
		TotalModelLayers:  4,
		TokensPerBlock:    128,
		BytesPerBlock:     4096,
		MaxResidentFrames: 2,
		TotalNVMeLBAs:     1024,
	}
	coord, err := NewBaMKVPagingCoordinator(hal, cfg)
	if err != nil {
		t.Fatalf("NewBaMKVPagingCoordinator failed: %v", err)
	}

	_, err = coord.AllocateBlock(-1, 128, nil)
	if err == nil {
		t.Errorf("expected error for negative layer, got nil")
	}
	_, err = coord.AllocateBlock(4, 128, nil)
	if err == nil {
		t.Errorf("expected error for layer >= TotalModelLayers, got nil")
	}

	if err := coord.FreeBlock(9999); err == nil {
		t.Errorf("expected error freeing non-existent block, got nil")
	}
	if err := coord.Pin(9999); err == nil {
		t.Errorf("expected error pinning non-existent block, got nil")
	}
	if err := coord.Unpin(9999); err == nil {
		t.Errorf("expected error unpinning non-existent block, got nil")
	}
	if err := coord.OffloadBlock(9999); err == nil {
		t.Errorf("expected error offloading non-existent block, got nil")
	}
	if err := coord.RestoreBlock(9999); err == nil {
		t.Errorf("expected error restoring non-existent block, got nil")
	}
	if _, err := coord.ReadBlock(9999); err == nil {
		t.Errorf("expected error reading non-existent block, got nil")
	}
	if _, err := coord.VerifyDataIntegrity(9999); err == nil {
		t.Errorf("expected error verifying non-existent block, got nil")
	}
	if _, err := coord.GetPageDirectoryEntry(9999); err == nil {
		t.Errorf("expected error getting entry for non-existent block, got nil")
	}
	if coord.IsPinned(9999) {
		t.Errorf("IsPinned for non-existent block returned true")
	}

	bid, err := coord.AllocateBlock(0, 128, []byte("test-data"))
	if err != nil {
		t.Fatalf("AllocateBlock failed: %v", err)
	}

	if err := coord.Pin(bid); err != nil {
		t.Fatalf("Pin failed: %v", err)
	}
	if !coord.IsPinned(bid) {
		t.Errorf("IsPinned returned false for pinned block")
	}

	err = coord.OffloadBlock(bid)
	if !errors.Is(err, ErrBlockPinned) {
		t.Errorf("expected ErrBlockPinned offloading pinned block, got %v", err)
	}

	if err := coord.Unpin(bid); err != nil {
		t.Fatalf("Unpin failed: %v", err)
	}
	if coord.IsPinned(bid) {
		t.Errorf("IsPinned returned true after unpin")
	}

	if err := coord.Unpin(bid); err == nil {
		t.Errorf("expected error on unpin with zero pin count, got nil")
	}

	readData, err := coord.ReadBlock(bid)
	if err != nil {
		t.Fatalf("ReadBlock failed: %v", err)
	}
	if string(readData[:9]) != "test-data" {
		t.Errorf("ReadBlock data mismatch: got %q", string(readData[:9]))
	}

	entry, err := coord.GetPageDirectoryEntry(bid)
	if err != nil {
		t.Fatalf("GetPageDirectoryEntry failed: %v", err)
	}
	if entry.BlockID != bid || entry.LayerID != 0 {
		t.Errorf("entry mismatch: %+v", entry)
	}

	if err := coord.FreeBlock(bid); err != nil {
		t.Fatalf("FreeBlock failed: %v", err)
	}

	if err := coord.FreeBlock(bid); err == nil {
		t.Errorf("expected error on double free, got nil")
	}
}

func TestUMAWriteBack_2to4GiBCapacity(t *testing.T) {
	// 1. Default capacity should be 2 GiB
	ringDefault, err := NewUMADRAMWriteBackRing(UMADRAMRingConfig{})
	if err != nil {
		t.Fatalf("NewUMADRAMWriteBackRing default failed: %v", err)
	}
	if ringDefault.Config().CapacityBytes != DefaultUMADRAMRingCapacity {
		t.Errorf("default capacity = %d, want %d (2 GiB)", ringDefault.Config().CapacityBytes, DefaultUMADRAMRingCapacity)
	}

	// 2. Explicit 3 GiB should be accepted
	const threeGiB = 3 * 1024 * 1024 * 1024
	ring3G, err := NewUMADRAMWriteBackRing(UMADRAMRingConfig{CapacityBytes: threeGiB})
	if err != nil {
		t.Fatalf("NewUMADRAMWriteBackRing 3GiB failed: %v", err)
	}
	if ring3G.Config().CapacityBytes != threeGiB {
		t.Errorf("capacity = %d, want 3 GiB", ring3G.Config().CapacityBytes)
	}

	// 3. Max capacity 4 GiB should be accepted
	ring4G, err := NewUMADRAMWriteBackRing(UMADRAMRingConfig{CapacityBytes: MaxUMADRAMRingCapacity})
	if err != nil {
		t.Fatalf("NewUMADRAMWriteBackRing 4GiB failed: %v", err)
	}
	if ring4G.Config().CapacityBytes != MaxUMADRAMRingCapacity {
		t.Errorf("capacity = %d, want 4 GiB", ring4G.Config().CapacityBytes)
	}

	// 4. Capacity exceeding 4 GiB must return ErrRingCapacityExceeded
	const fiveGiB = 5 * 1024 * 1024 * 1024
	_, err = NewUMADRAMWriteBackRing(UMADRAMRingConfig{CapacityBytes: fiveGiB})
	if !errors.Is(err, ErrRingCapacityExceeded) {
		t.Errorf("expected ErrRingCapacityExceeded for 5 GiB, got: %v", err)
	}

	// 5. Non-zero capacity below 2 GiB must return ErrRingCapacityTooSmall
	const oneGiB = 1 * 1024 * 1024 * 1024
	_, err = NewUMADRAMWriteBackRing(UMADRAMRingConfig{CapacityBytes: oneGiB})
	if !errors.Is(err, ErrRingCapacityTooSmall) {
		t.Errorf("expected ErrRingCapacityTooSmall for 1 GiB, got: %v", err)
	}
}

func TestUMAWriteBack_PageCacheZeroBytes(t *testing.T) {
	var writtenExtents []UMADRAMExtent
	var alignedOffsets []uint64

	ring, err := NewUMADRAMWriteBackRing(UMADRAMRingConfig{
		CapacityBytes:       DefaultUMADRAMRingCapacity,
		ExtentCoalesceBytes: DefaultExtentCoalesceBytes,
		SectorSizeBytes:     4096,
		DiskWriter: func(offset uint64, data []byte, fd uintptr) error {
			// Assert 4096 alignment for direct I/O
			if offset%DirectIOAlignment != 0 {
				t.Errorf("disk write offset %d is not 4096-aligned", offset)
			}
			alignedOffsets = append(alignedOffsets, offset)
			writtenExtents = append(writtenExtents, UMADRAMExtent{
				ByteOffset:  offset,
				LengthBytes: uint64(len(data)),
			})
			return nil
		},
	})
	if err != nil {
		t.Fatalf("failed to construct ring: %v", err)
	}

	// Write 100 blocks of 4KB each
	const numBlocks = 100
	const blockSize = 4096
	blockData := make([]byte, blockSize)
	for i := range blockData {
		blockData[i] = byte(i & 0xFF)
	}

	for i := 0; i < numBlocks; i++ {
		lba := uint64(i)
		if err := ring.WriteBlock(lba, blockData); err != nil {
			t.Fatalf("WriteBlock(%d) failed: %v", lba, err)
		}
	}

	// Verify page cache resident bytes remains 0 before flush
	if !ring.AssertZeroPageCache() {
		t.Errorf("AssertZeroPageCache before flush returned false")
	}

	flushedBytes, extCount, err := ring.Flush()
	if err != nil {
		t.Fatalf("Flush failed: %v", err)
	}
	if flushedBytes != numBlocks*blockSize {
		t.Errorf("flushed bytes = %d, want %d", flushedBytes, numBlocks*blockSize)
	}
	if extCount == 0 {
		t.Errorf("extent count = 0, want > 0")
	}

	// Verify page cache resident bytes is strictly 0 after flush
	stats := ring.Stats()
	if stats.PageCacheResidentBytes != 0 {
		t.Errorf("PageCacheResidentBytes = %d, want 0", stats.PageCacheResidentBytes)
	}
	if !ring.AssertZeroPageCache() {
		t.Errorf("AssertZeroPageCache after flush returned false")
	}
}

func TestUMAWriteBack_Coalesced2MBExtents(t *testing.T) {
	var flushedExtents []UMADRAMExtent

	ring, err := NewUMADRAMWriteBackRing(UMADRAMRingConfig{
		CapacityBytes:       DefaultUMADRAMRingCapacity,
		ExtentCoalesceBytes: DefaultExtentCoalesceBytes, // 2 MiB
		SectorSizeBytes:     4096,
		DiskWriter: func(offset uint64, data []byte, fd uintptr) error {
			flushedExtents = append(flushedExtents, UMADRAMExtent{
				ByteOffset:  offset,
				LengthBytes: uint64(len(data)),
			})
			return nil
		},
	})
	if err != nil {
		t.Fatalf("failed to construct ring: %v", err)
	}

	// Stage 1: Write 512 contiguous 4KB blocks = exactly 2 MiB
	const blocksPer2MB = 512
	payload4K := make([]byte, 4096)
	for i := 0; i < blocksPer2MB; i++ {
		payload4K[0] = byte(i & 0xFF)
		if err := ring.WriteBlock(uint64(i), payload4K); err != nil {
			t.Fatalf("WriteBlock failed at %d: %v", i, err)
		}
	}

	// Stage 2: Write a disconnected block at LBA 2000 (offset 8,192,000)
	const isolatedLBA = 2000
	if err := ring.WriteBlock(isolatedLBA, payload4K); err != nil {
		t.Fatalf("WriteBlock isolated failed: %v", err)
	}

	// Flush and verify extents
	flushedBytes, extCount, err := ring.Flush()
	if err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	expectedBytes := uint64((blocksPer2MB + 1) * 4096)
	if flushedBytes != expectedBytes {
		t.Errorf("flushed bytes = %d, want %d", flushedBytes, expectedBytes)
	}

	// We expect exactly 2 extents: one 2 MiB extent (512 blocks) and one 4KB extent (the isolated block)
	if extCount != 2 {
		t.Fatalf("extCount = %d, want 2", extCount)
	}
	if len(flushedExtents) != 2 {
		t.Fatalf("flushedExtents len = %d, want 2", len(flushedExtents))
	}

	if flushedExtents[0].LengthBytes != DefaultExtentCoalesceBytes {
		t.Errorf("extent 0 length = %d, want %d (2 MiB)", flushedExtents[0].LengthBytes, DefaultExtentCoalesceBytes)
	}
	if flushedExtents[0].ByteOffset != 0 {
		t.Errorf("extent 0 offset = %d, want 0", flushedExtents[0].ByteOffset)
	}

	if flushedExtents[1].LengthBytes != 4096 {
		t.Errorf("extent 1 length = %d, want 4096", flushedExtents[1].LengthBytes)
	}
	if flushedExtents[1].ByteOffset != isolatedLBA*4096 {
		t.Errorf("extent 1 offset = %d, want %d", flushedExtents[1].ByteOffset, isolatedLBA*4096)
	}
}

func TestUMAWriteBack_WAFSlashing(t *testing.T) {
	ring, err := NewUMADRAMWriteBackRing(UMADRAMRingConfig{
		CapacityBytes:       DefaultUMADRAMRingCapacity,
		ExtentCoalesceBytes: DefaultExtentCoalesceBytes,
		BaselineWAF:         30.0,
		SequentialWAF:       1.1,
		SectorSizeBytes:     4096,
	})
	if err != nil {
		t.Fatalf("failed to construct ring: %v", err)
	}

	// Simulate high-frequency random 4KB writes from agent KV paging.
	// 10,000 random 4KB writes concentrated on a working set of 512 blocks (2 MiB).
	// In unbuffered disk paging, 10,000 * 4KB = 40.96 MB written with >30x WAF = >1.2 GB NAND writes!
	// With UMA DRAM write-back dirty ring, all mutations are absorbed in DRAM and coalesce into a single 2MB extent.
	const numWrites = 10000
	const workingSetBlocks = 512
	payload := make([]byte, 4096)

	for i := 0; i < numWrites; i++ {
		lba := uint64(i % workingSetBlocks)
		payload[0] = byte(i & 0xFF)
		payload[1] = byte((i >> 8) & 0xFF)
		if err := ring.WriteBlock(lba, payload); err != nil {
			t.Fatalf("WriteBlock(%d) failed: %v", i, err)
		}
	}

	// Flush dirty ring
	flushedBytes, extCount, err := ring.Flush()
	if err != nil {
		t.Fatalf("Flush failed: %v", err)
	}
	if extCount != 1 {
		t.Errorf("expected 1 coalesced 2MB extent, got %d", extCount)
	}
	if flushedBytes != workingSetBlocks*4096 {
		t.Errorf("flushed bytes = %d, want %d", flushedBytes, workingSetBlocks*4096)
	}

	stats := ring.Stats()
	t.Logf("WAF Metrics:")
	t.Logf("  Total Random Writes:           %d", stats.TotalRandomWrites)
	t.Logf("  Total Random Write Bytes:     %d (%.2f MB)", stats.TotalRandomWriteBytes, float64(stats.TotalRandomWriteBytes)/(1024*1024))
	t.Logf("  Total Sequential Flushed:     %d (%.2f MB)", stats.TotalSequentialFlushedBytes, float64(stats.TotalSequentialFlushedBytes)/(1024*1024))
	t.Logf("  Baseline WAF:                 %.1fx", stats.BaselineWAF)
	t.Logf("  Sequential WAF:               %.1fx", stats.SequentialWAF)
	t.Logf("  Measured WAF:                 %.4fx", stats.MeasuredWAF)
	t.Logf("  WAF Reduction Factor:         %.2fx", stats.WAFReductionFactor)
	t.Logf("  Estimated Lifespan Multiplier: %.2fx", stats.LifespanExtensionFactor)

	// Invariants:
	// 1. Measured WAF must be < 1.2x (slashed from >30x)
	if stats.MeasuredWAF >= 1.2 {
		t.Errorf("MeasuredWAF = %.4f, want < 1.2x", stats.MeasuredWAF)
	}
	// 2. WAF reduction factor must be >= 25x (lifespan extended by at least 25x)
	if stats.WAFReductionFactor < 25.0 {
		t.Errorf("WAFReductionFactor = %.2fx, want >= 25.0x", stats.WAFReductionFactor)
	}
	// 3. Page cache resident bytes must be strictly 0
	if stats.PageCacheResidentBytes != 0 {
		t.Errorf("PageCacheResidentBytes = %d, want 0", stats.PageCacheResidentBytes)
	}
}

func TestUMAWriteBack_EndToEndKVPagingIntegration(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
	cfg := BaMKVPagingConfig{
		NodeID:                  0,
		TotalModelLayers:        16,
		TokensPerBlock:          512,
		BytesPerBlock:           64 * 1024,
		MaxResidentFrames:       16, // Small frame limit to force offloading
		TotalNVMeLBAs:           1024 * 1024,
		SectorSizeBytes:         4096,
		QueueDepth:              64,
		PrefetchDistance:        2,
		EnableWriteBackRing:     true,
		RingBufferCapacityBytes: 2 * 1024 * 1024 * 1024,
		ExtentCoalesceBytes:     2 * 1024 * 1024,
	}

	coord, err := NewBaMKVPagingCoordinator(hal, cfg)
	if err != nil {
		t.Fatalf("failed to create coordinator with write-back ring: %v", err)
	}

	ring := coord.DirtyRing()
	if ring == nil {
		t.Fatalf("expected coordinator DirtyRing() to be non-nil")
	}

	// Allocate 64 blocks (exceeding MaxResidentFrames=16 by 4x)
	const numBlocks = 64
	blockIDs := make([]uint64, numBlocks)

	for i := 0; i < numBlocks; i++ {
		layerID := i % cfg.TotalModelLayers
		data := make([]byte, cfg.BytesPerBlock)
		data[0] = byte(i & 0xFF)
		data[1] = byte((i >> 8) & 0xFF)
		data[len(data)-1] = 0x55

		bid, err := coord.AllocateBlock(layerID, cfg.TokensPerBlock, data)
		if err != nil {
			t.Fatalf("failed to allocate block %d: %v", i, err)
		}
		blockIDs[i] = bid
	}

	stats := coord.Stats()
	if stats.OffloadedBlocks == 0 {
		t.Fatalf("expected offloaded blocks with 64 blocks and 16 frames")
	}

	// Verify dirty blocks are staged in UMA DRAM write-back ring
	ringStats := coord.DirtyRingStats()
	if ringStats == nil {
		t.Fatalf("DirtyRingStats() is nil")
	}
	if ringStats.CurrentDirtyBytes == 0 && ringStats.TotalSequentialFlushedBytes == 0 {
		t.Errorf("expected dirty bytes in write-back ring, got 0")
	}

	// Verify data integrity for offloaded blocks before and after flush
	for _, bid := range blockIDs {
		ok, err := coord.VerifyDataIntegrity(bid)
		if err != nil {
			t.Fatalf("VerifyDataIntegrity on block %d failed: %v", bid, err)
		}
		if !ok {
			t.Fatalf("data integrity check failed on block %d", bid)
		}
	}

	// Flush dirty ring
	flushedBytes, extCount, err := coord.FlushDirtyRing()
	if err != nil {
		t.Fatalf("FlushDirtyRing failed: %v", err)
	}
	t.Logf("End-to-end FlushDirtyRing: flushed %d bytes across %d extents", flushedBytes, extCount)

	postFlushStats := coord.DirtyRingStats()
	if postFlushStats.CurrentDirtyBytes != 0 {
		t.Errorf("CurrentDirtyBytes immediately after flush = %d, want 0", postFlushStats.CurrentDirtyBytes)
	}

	// Re-verify data integrity after flush (reading non-resident blocks pages them back in, offloading frames to dirty ring)
	for _, bid := range blockIDs {
		data, err := coord.ReadBlock(bid)
		if err != nil {
			t.Fatalf("ReadBlock(%d) failed: %v", bid, err)
		}
		if data[len(data)-1] != 0x55 {
			t.Errorf("block %d corrupted: trailer byte = 0x%X, want 0x55", bid, data[len(data)-1])
		}
	}

	// Final flush of remaining evicted blocks
	if _, _, err := coord.FlushDirtyRing(); err != nil {
		t.Fatalf("final FlushDirtyRing failed: %v", err)
	}

	// Final ring stats check
	finalStats := coord.DirtyRingStats()
	if finalStats.CurrentDirtyBytes != 0 {
		t.Errorf("CurrentDirtyBytes after final flush = %d, want 0", finalStats.CurrentDirtyBytes)
	}
	if finalStats.PageCacheResidentBytes != 0 {
		t.Errorf("PageCacheResidentBytes = %d, want 0", finalStats.PageCacheResidentBytes)
	}
	if finalStats.MeasuredWAF >= 1.2 {
		t.Errorf("MeasuredWAF = %.4f, want < 1.2x", finalStats.MeasuredWAF)
	}
	if finalStats.WAFReductionFactor < 25.0 {
		t.Errorf("WAFReductionFactor = %.2fx, want >= 25.0x", finalStats.WAFReductionFactor)
	}
}

func TestUMAWriteBack_InlineFlushNoDeadlock(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
	cfg := BaMKVPagingConfig{
		NodeID:                  0,
		TotalModelLayers:        4,
		TokensPerBlock:          128,
		BytesPerBlock:           64 * 1024,
		MaxResidentFrames:       2, // Forces offload on 3rd block
		TotalNVMeLBAs:           1024,
		SectorSizeBytes:         4096,
		QueueDepth:              16,
		EnableWriteBackRing:     true,
		RingBufferCapacityBytes: 2 * 1024 * 1024 * 1024,
		ExtentCoalesceBytes:     2 * 1024 * 1024,
	}

	coord, err := NewBaMKVPagingCoordinator(hal, cfg)
	if err != nil {
		t.Fatalf("failed to construct coordinator: %v", err)
	}

	// Lower the flush threshold in the coordinator's dirty ring to force an inline flush during offloadFrameLocked
	ring := coord.DirtyRing()
	ring.mu.Lock()
	ring.config.FlushThresholdBytes = 64 * 1024 // Flush on 64KB dirty data
	ring.mu.Unlock()

	// Allocate 10 blocks: with MaxResidentFrames=2 and FlushThreshold=64KB, this triggers
	// repeated offloadFrameLocked -> WriteBlock -> flushLocked -> DiskWriter calls while coord.mu is held.
	// This proves that DiskWriter does not deadlock on coord.mu.
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 10; i++ {
			data := make([]byte, cfg.BytesPerBlock)
			data[0] = byte(i)
			_, err := coord.AllocateBlock(i%cfg.TotalModelLayers, cfg.TokensPerBlock, data)
			if err != nil {
				t.Errorf("AllocateBlock(%d) failed: %v", i, err)
				return
			}
		}
	}()

	select {
	case <-done:
		// Success: no deadlock
	case <-time.After(3 * time.Second):
		t.Fatalf("DEADLOCK detected: AllocateBlock hung during inline dirty ring flush")
	}

	stats := coord.DirtyRingStats()
	if stats == nil {
		t.Fatalf("nil stats")
	}
	if stats.TotalFlushes == 0 {
		t.Errorf("expected inline flushes to occur, got 0")
	}
}

func TestUMAWriteBack_CleanBlockEviction(t *testing.T) {
	ring, err := NewUMADRAMWriteBackRing(UMADRAMRingConfig{
		CapacityBytes:       DefaultUMADRAMRingCapacity,
		FlushThresholdBytes: 128 * 1024,
		ExtentCoalesceBytes: DefaultExtentCoalesceBytes,
		SectorSizeBytes:     4096,
	})
	if err != nil {
		t.Fatalf("failed to construct ring: %v", err)
	}

	// Write 100 blocks of 4KB
	payload := make([]byte, 4096)
	for i := 0; i < 100; i++ {
		payload[0] = byte(i)
		if err := ring.WriteBlock(uint64(i), payload); err != nil {
			t.Fatalf("WriteBlock(%d) failed: %v", i, err)
		}
	}

	// Flush to make them clean
	if _, _, err := ring.Flush(); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	if ring.DirtyBytes() != 0 {
		t.Errorf("DirtyBytes after flush = %d, want 0", ring.DirtyBytes())
	}
	if ring.DirtyPages() != 0 {
		t.Errorf("DirtyPages after flush = %d, want 0", ring.DirtyPages())
	}

	// Free a single block and verify memory accounting decrements
	preMem := ring.totalMemoryBytes
	ring.FreeBlock(0)
	if ring.totalMemoryBytes != preMem-4096 {
		t.Errorf("totalMemoryBytes after FreeBlock = %d, want %d", ring.totalMemoryBytes, preMem-4096)
	}

	// Verify evictCleanBlocksLocked frees memory under pressure
	ring.mu.Lock()
	ring.evictCleanBlocksLocked(ring.config.CapacityBytes) // request eviction of everything clean
	ring.mu.Unlock()

	if ring.totalMemoryBytes != 0 {
		t.Errorf("totalMemoryBytes after clean eviction = %d, want 0", ring.totalMemoryBytes)
	}
}
