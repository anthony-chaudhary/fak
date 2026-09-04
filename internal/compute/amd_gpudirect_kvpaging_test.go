// Package compute implements hardware abstraction, tensor computation, memory slab management,
// and zero-copy device interconnect acceleration for the fak agent kernel.
package compute

import (
	"errors"
	"testing"
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
