package compute

import (
	"bytes"
	"fmt"
	"testing"
	"time"
)

// TestHierarchicalMemoryAllocationAndRetrieval verifies multi-tier allocation and data retrieval.
func TestHierarchicalMemoryAllocationAndRetrieval(t *testing.T) {
	cfg := HierarchicalMemoryConfig{
		Tier0CapacityBytes: 10 * 1024 * 1024,
		Tier1CapacityBytes: 20 * 1024 * 1024,
		Tier2CapacityBytes: 50 * 1024 * 1024,
		Tier0HighWatermark: 0.85,
		Tier0LowWatermark:  0.70,
		Tier1HighWatermark: 0.90,
		Tier1LowWatermark:  0.75,
	}

	mgr, err := NewHierarchicalMemoryManager(cfg, nil)
	if err != nil {
		t.Fatalf("NewHierarchicalMemoryManager failed: %v", err)
	}

	// 1. Allocate and write in Tier 0 (GPU VRAM)
	b0, err := mgr.Allocate("block_vram", 1024, Tier0VRAM, false)
	if err != nil {
		t.Fatalf("Allocate Tier0 failed: %v", err)
	}
	if b0.CurrentTier != Tier0VRAM {
		t.Fatalf("expected CurrentTier Tier0VRAM, got %v", b0.CurrentTier)
	}
	vramData := []byte("payload_in_rtx5090_gddr7_vram")
	if err := mgr.Write("block_vram", vramData); err != nil {
		t.Fatalf("Write block_vram failed: %v", err)
	}
	readVRAM, err := mgr.Read("block_vram", Tier0VRAM)
	if err != nil {
		t.Fatalf("Read block_vram failed: %v", err)
	}
	if !bytes.Equal(readVRAM, vramData) {
		t.Fatalf("data mismatch in Tier 0: got %s, want %s", readVRAM, vramData)
	}

	// 2. Allocate and write in Tier 1 (Host Pinned DRAM)
	b1, err := mgr.Allocate("block_dram", 2048, Tier1HostDRAM, false)
	if err != nil {
		t.Fatalf("Allocate Tier1 failed: %v", err)
	}
	if b1.CurrentTier != Tier1HostDRAM {
		t.Fatalf("expected CurrentTier Tier1HostDRAM, got %v", b1.CurrentTier)
	}
	dramData := []byte("payload_in_ryzen5950x_pinned_host_dram")
	if err := mgr.Write("block_dram", dramData); err != nil {
		t.Fatalf("Write block_dram failed: %v", err)
	}
	readDRAM, err := mgr.Read("block_dram", Tier1HostDRAM)
	if err != nil {
		t.Fatalf("Read block_dram failed: %v", err)
	}
	if !bytes.Equal(readDRAM, dramData) {
		t.Fatalf("data mismatch in Tier 1: got %s, want %s", readDRAM, dramData)
	}

	// 3. Allocate and write in Tier 2 (Direct NVMe Storage)
	b2, err := mgr.Allocate("block_nvme", 4096, Tier2NVMeDirect, false)
	if err != nil {
		t.Fatalf("Allocate Tier2 failed: %v", err)
	}
	if b2.CurrentTier != Tier2NVMeDirect {
		t.Fatalf("expected CurrentTier Tier2NVMeDirect, got %v", b2.CurrentTier)
	}
	nvmeData := []byte("payload_in_m2a_cpu_nvme_bam_p2pdma")
	if err := mgr.Write("block_nvme", nvmeData); err != nil {
		t.Fatalf("Write block_nvme failed: %v", err)
	}
	readNVMe, err := mgr.Read("block_nvme", Tier2NVMeDirect)
	if err != nil {
		t.Fatalf("Read block_nvme failed: %v", err)
	}
	if !bytes.Equal(readNVMe, nvmeData) {
		t.Fatalf("data mismatch in Tier 2: got %s, want %s", readNVMe, nvmeData)
	}

	// Check aggregate statistics
	stats := mgr.Stats()
	if stats.Tier0BlockCount != 1 || stats.Tier1BlockCount != 1 || stats.Tier2BlockCount != 1 {
		t.Fatalf("unexpected block counts: %+v", stats)
	}
	if stats.TotalBlocks != 3 {
		t.Fatalf("expected 3 total blocks, got %d", stats.TotalBlocks)
	}
	if stats.Tier0UsageBytes != uint64(len(vramData)) {
		t.Fatalf("expected Tier0Usage %d, got %d", len(vramData), stats.Tier0UsageBytes)
	}
	if stats.Tier1UsageBytes != uint64(len(dramData)) {
		t.Fatalf("expected Tier1Usage %d, got %d", len(dramData), stats.Tier1UsageBytes)
	}
	if stats.Tier2UsageBytes != uint64(len(nvmeData)) {
		t.Fatalf("expected Tier2Usage %d, got %d", len(nvmeData), stats.Tier2UsageBytes)
	}
}

// TestHierarchicalMemoryWatermarkEviction verifies the cascading eviction Tier 0 -> Tier 1 -> Tier 2.
func TestHierarchicalMemoryWatermarkEviction(t *testing.T) {
	cfg := HierarchicalMemoryConfig{
		Tier0CapacityBytes: 1000,
		Tier0HighWatermark: 0.80, // 800 bytes
		Tier0LowWatermark:  0.50, // 500 bytes
		Tier1CapacityBytes: 1000,
		Tier1HighWatermark: 0.80, // 800 bytes
		Tier1LowWatermark:  0.50, // 500 bytes
		Tier2CapacityBytes: 10000,
	}

	mgr, err := NewHierarchicalMemoryManager(cfg, nil)
	if err != nil {
		t.Fatalf("NewHierarchicalMemoryManager failed: %v", err)
	}

	// 1. Allocate block A (400 bytes) in Tier 0
	_, err = mgr.Allocate("block_A", 400, Tier0VRAM, false)
	if err != nil {
		t.Fatalf("allocate block_A failed: %v", err)
	}
	if err := mgr.Write("block_A", bytes.Repeat([]byte("A"), 400)); err != nil {
		t.Fatalf("write block_A failed: %v", err)
	}
	time.Sleep(2 * time.Millisecond)

	// 2. Allocate block B (400 bytes) in Tier 0
	// Tier 0 is now 800 bytes (at HighWatermark)
	_, err = mgr.Allocate("block_B", 400, Tier0VRAM, false)
	if err != nil {
		t.Fatalf("allocate block_B failed: %v", err)
	}
	if err := mgr.Write("block_B", bytes.Repeat([]byte("B"), 400)); err != nil {
		t.Fatalf("write block_B failed: %v", err)
	}
	time.Sleep(2 * time.Millisecond)

	// 3. Allocate block C (300 bytes) in Tier 0
	// 800 + 300 = 1100 > 800 (HighWatermark). Eviction in Tier 0 triggers!
	// block_A (oldest) should be demoted from Tier 0 to Tier 1.
	_, err = mgr.Allocate("block_C", 300, Tier0VRAM, false)
	if err != nil {
		t.Fatalf("allocate block_C failed: %v", err)
	}
	if err := mgr.Write("block_C", bytes.Repeat([]byte("C"), 300)); err != nil {
		t.Fatalf("write block_C failed: %v", err)
	}

	blkA, err := mgr.GetBlock("block_A")
	if err != nil {
		t.Fatalf("get block_A failed: %v", err)
	}
	if blkA.CurrentTier != Tier1HostDRAM {
		t.Fatalf("expected block_A to be demoted to Tier1HostDRAM, got %v", blkA.CurrentTier)
	}

	blkB, err := mgr.GetBlock("block_B")
	if err != nil {
		t.Fatalf("get block_B failed: %v", err)
	}
	if blkB.CurrentTier != Tier0VRAM {
		t.Fatalf("expected block_B to remain in Tier0VRAM, got %v", blkB.CurrentTier)
	}

	blkC, err := mgr.GetBlock("block_C")
	if err != nil {
		t.Fatalf("get block_C failed: %v", err)
	}
	if blkC.CurrentTier != Tier0VRAM {
		t.Fatalf("expected block_C in Tier0VRAM, got %v", blkC.CurrentTier)
	}

	// 4. Now fill Tier 1 to trigger cascade from Tier 1 to Tier 2.
	// Currently Tier 1 has block_A (400 bytes).
	// Allocate block_D (300 bytes) in Tier 1.
	time.Sleep(2 * time.Millisecond)
	_, err = mgr.Allocate("block_D", 300, Tier1HostDRAM, false)
	if err != nil {
		t.Fatalf("allocate block_D failed: %v", err)
	}
	if err := mgr.Write("block_D", bytes.Repeat([]byte("D"), 300)); err != nil {
		t.Fatalf("write block_D failed: %v", err)
	}
	time.Sleep(2 * time.Millisecond)

	// Allocate block_E (300 bytes) in Tier 1.
	// 400 (A) + 300 (D) + 300 (E) = 1000 bytes > 800 (Tier1HighWatermark).
	// Eviction in Tier 1 triggers! block_A (oldest in Tier 1) cascades into Tier 2!
	_, err = mgr.Allocate("block_E", 300, Tier1HostDRAM, false)
	if err != nil {
		t.Fatalf("allocate block_E failed: %v", err)
	}
	if err := mgr.Write("block_E", bytes.Repeat([]byte("E"), 300)); err != nil {
		t.Fatalf("write block_E failed: %v", err)
	}

	blkAAfter, err := mgr.GetBlock("block_A")
	if err != nil {
		t.Fatalf("get block_A after cascade failed: %v", err)
	}
	if blkAAfter.CurrentTier != Tier2NVMeDirect {
		t.Fatalf("expected block_A to cascade to Tier2NVMeDirect, got %v", blkAAfter.CurrentTier)
	}

	// Verify data integrity of block_A after Tier0 -> Tier1 -> Tier2 cascade
	dataA, err := mgr.Read("block_A", Tier2NVMeDirect)
	if err != nil {
		t.Fatalf("read block_A from Tier 2 failed: %v", err)
	}
	if !bytes.Equal(dataA, bytes.Repeat([]byte("A"), 400)) {
		t.Fatalf("data corruption in block_A after cascade")
	}
}

// TestHierarchicalMemoryPinnedProtection verifies pinned blocks are never demoted by eviction.
func TestHierarchicalMemoryPinnedProtection(t *testing.T) {
	cfg := HierarchicalMemoryConfig{
		Tier0CapacityBytes: 1000,
		Tier0HighWatermark: 0.80, // 800 bytes
		Tier0LowWatermark:  0.50, // 500 bytes
		Tier1CapacityBytes: 5000,
		Tier1HighWatermark: 0.90,
		Tier1LowWatermark:  0.75,
		Tier2CapacityBytes: 10000,
	}

	mgr, err := NewHierarchicalMemoryManager(cfg, nil)
	if err != nil {
		t.Fatalf("NewHierarchicalMemoryManager failed: %v", err)
	}

	// 1. Allocate block P1 (pinned, 400 bytes) in Tier 0
	_, err = mgr.Allocate("pinned_P1", 400, Tier0VRAM, true)
	if err != nil {
		t.Fatalf("allocate pinned_P1 failed: %v", err)
	}
	if err := mgr.Write("pinned_P1", bytes.Repeat([]byte("P"), 400)); err != nil {
		t.Fatalf("write pinned_P1 failed: %v", err)
	}
	time.Sleep(2 * time.Millisecond)

	// 2. Allocate block U2 (unpinned, 400 bytes) in Tier 0
	_, err = mgr.Allocate("unpinned_U2", 400, Tier0VRAM, false)
	if err != nil {
		t.Fatalf("allocate unpinned_U2 failed: %v", err)
	}
	if err := mgr.Write("unpinned_U2", bytes.Repeat([]byte("U"), 400)); err != nil {
		t.Fatalf("write unpinned_U2 failed: %v", err)
	}
	time.Sleep(2 * time.Millisecond)

	// 3. Allocate block U3 (unpinned, 300 bytes) in Tier 0.
	// Total requested = 400 + 400 + 300 = 1100 > 800.
	// Eviction triggers. Although pinned_P1 is older than unpinned_U2,
	// pinned_P1 is protected. unpinned_U2 MUST be evicted instead!
	_, err = mgr.Allocate("unpinned_U3", 300, Tier0VRAM, false)
	if err != nil {
		t.Fatalf("allocate unpinned_U3 failed: %v", err)
	}
	if err := mgr.Write("unpinned_U3", bytes.Repeat([]byte("3"), 300)); err != nil {
		t.Fatalf("write unpinned_U3 failed: %v", err)
	}

	p1, err := mgr.GetBlock("pinned_P1")
	if err != nil {
		t.Fatalf("get pinned_P1 failed: %v", err)
	}
	if p1.CurrentTier != Tier0VRAM {
		t.Fatalf("pinned block P1 was evicted to %v! Expected Tier0VRAM", p1.CurrentTier)
	}

	u2, err := mgr.GetBlock("unpinned_U2")
	if err != nil {
		t.Fatalf("get unpinned_U2 failed: %v", err)
	}
	if u2.CurrentTier != Tier1HostDRAM {
		t.Fatalf("expected unpinned U2 to be evicted to Tier1HostDRAM, got %v", u2.CurrentTier)
	}

	// 4. Test explicit EvictFromTier when only pinned blocks remain in Tier 0
	_ = mgr.Demote("unpinned_U3", Tier1HostDRAM)
	evicted, err := mgr.EvictFromTier(Tier0VRAM)
	if err != nil {
		t.Fatalf("EvictFromTier returned error: %v", err)
	}
	if evicted != 0 {
		t.Fatalf("expected 0 blocks evicted when all blocks are pinned, got %d", evicted)
	}
	p1After, _ := mgr.GetBlock("pinned_P1")
	if p1After.CurrentTier != Tier0VRAM {
		t.Fatalf("pinned block P1 was evicted during explicit EvictFromTier")
	}
}

// TestHierarchicalMemoryPromotionDemotion verifies explicit tier promotion and demotion.
func TestHierarchicalMemoryPromotionDemotion(t *testing.T) {
	cfg := HierarchicalMemoryConfig{
		Tier0CapacityBytes: 1024 * 1024,
		Tier1CapacityBytes: 2048 * 1024,
		Tier2CapacityBytes: 4096 * 1024,
		Tier0HighWatermark: 0.85,
		Tier0LowWatermark:  0.70,
		Tier1HighWatermark: 0.90,
		Tier1LowWatermark:  0.75,
	}

	mgr, err := NewHierarchicalMemoryManager(cfg, nil)
	if err != nil {
		t.Fatalf("NewHierarchicalMemoryManager failed: %v", err)
	}

	// Start at Tier 2
	payload := []byte("promotion_demotion_test_payload")
	b, err := mgr.Allocate("promo_demo_block", uint64(len(payload)), Tier2NVMeDirect, false)
	if err != nil {
		t.Fatalf("Allocate failed: %v", err)
	}
	if err := mgr.Write("promo_demo_block", payload); err != nil {
		t.Fatalf("Write failed: %v", err)
	}
	if b.CurrentTier != Tier2NVMeDirect {
		t.Fatalf("expected Tier2NVMeDirect, got %v", b.CurrentTier)
	}

	// 1. Promote Tier 2 -> Tier 1
	if err := mgr.Promote("promo_demo_block", Tier1HostDRAM); err != nil {
		t.Fatalf("Promote to Tier1HostDRAM failed: %v", err)
	}
	blk, _ := mgr.GetBlock("promo_demo_block")
	if blk.CurrentTier != Tier1HostDRAM {
		t.Fatalf("expected Tier1HostDRAM, got %v", blk.CurrentTier)
	}

	// 2. Promote Tier 1 -> Tier 0
	if err := mgr.Promote("promo_demo_block", Tier0VRAM); err != nil {
		t.Fatalf("Promote to Tier0VRAM failed: %v", err)
	}
	blk, _ = mgr.GetBlock("promo_demo_block")
	if blk.CurrentTier != Tier0VRAM {
		t.Fatalf("expected Tier0VRAM, got %v", blk.CurrentTier)
	}

	// 3. Demote Tier 0 -> Tier 1
	if err := mgr.Demote("promo_demo_block", Tier1HostDRAM); err != nil {
		t.Fatalf("Demote to Tier1HostDRAM failed: %v", err)
	}
	blk, _ = mgr.GetBlock("promo_demo_block")
	if blk.CurrentTier != Tier1HostDRAM {
		t.Fatalf("expected Tier1HostDRAM, got %v", blk.CurrentTier)
	}

	// 4. Demote Tier 1 -> Tier 2
	if err := mgr.Demote("promo_demo_block", Tier2NVMeDirect); err != nil {
		t.Fatalf("Demote to Tier2NVMeDirect failed: %v", err)
	}
	blk, _ = mgr.GetBlock("promo_demo_block")
	if blk.CurrentTier != Tier2NVMeDirect {
		t.Fatalf("expected Tier2NVMeDirect, got %v", blk.CurrentTier)
	}

	// 5. Test invalid operations
	if err := mgr.Promote("promo_demo_block", Tier2NVMeDirect); err == nil {
		// Target == Current is a no-op, returns nil
	}
	if err := mgr.Demote("promo_demo_block", Tier0VRAM); err == nil {
		t.Fatalf("expected error demoting from Tier 2 to Tier 0, got nil")
	}
	if err := mgr.Promote("promo_demo_block", MemoryTier(99)); err == nil {
		t.Fatalf("expected error promoting to invalid tier")
	}

	// Verify data intact
	readBack, err := mgr.Read("promo_demo_block", Tier2NVMeDirect)
	if err != nil {
		t.Fatalf("Read failed: %v", err)
	}
	if !bytes.Equal(readBack, payload) {
		t.Fatalf("payload mismatch: got %s, want %s", readBack, payload)
	}
}

// TestHierarchicalMemoryPrefetch verifies non-blocking prefetching into a faster tier.
func TestHierarchicalMemoryPrefetch(t *testing.T) {
	cfg := HierarchicalMemoryConfig{
		Tier0CapacityBytes: 10 * 1024 * 1024,
		Tier1CapacityBytes: 20 * 1024 * 1024,
		Tier2CapacityBytes: 50 * 1024 * 1024,
		Tier0HighWatermark: 0.85,
		Tier0LowWatermark:  0.70,
		Tier1HighWatermark: 0.90,
		Tier1LowWatermark:  0.75,
	}

	mgr, err := NewHierarchicalMemoryManager(cfg, nil)
	if err != nil {
		t.Fatalf("NewHierarchicalMemoryManager failed: %v", err)
	}

	var blockIDs []string
	for i := 0; i < 5; i++ {
		id := fmt.Sprintf("prefetch_block_%d", i)
		blockIDs = append(blockIDs, id)
		data := []byte(fmt.Sprintf("prefetch_content_payload_%d", i))
		_, err := mgr.Allocate(id, uint64(len(data)), Tier2NVMeDirect, false)
		if err != nil {
			t.Fatalf("Allocate %s failed: %v", id, err)
		}
		if err := mgr.Write(id, data); err != nil {
			t.Fatalf("Write %s failed: %v", id, err)
		}
	}

	// Ensure all are in Tier 2
	for _, id := range blockIDs {
		blk, _ := mgr.GetBlock(id)
		if blk.CurrentTier != Tier2NVMeDirect {
			t.Fatalf("block %s expected in Tier2NVMeDirect, got %v", id, blk.CurrentTier)
		}
	}

	// Trigger non-blocking prefetch to Tier 0
	if err := mgr.Prefetch(blockIDs, Tier0VRAM); err != nil {
		t.Fatalf("Prefetch failed: %v", err)
	}

	// Await completion via synchronization helper
	mgr.WaitForPrefetch()

	// Verify all blocks arrived in Tier 0 and data is verified
	for i, id := range blockIDs {
		blk, err := mgr.GetBlock(id)
		if err != nil {
			t.Fatalf("GetBlock %s failed: %v", id, err)
		}
		if blk.CurrentTier != Tier0VRAM {
			t.Fatalf("block %s expected in Tier0VRAM after prefetch, got %v", id, blk.CurrentTier)
		}

		expected := []byte(fmt.Sprintf("prefetch_content_payload_%d", i))
		readData, err := mgr.Read(id, Tier0VRAM)
		if err != nil {
			t.Fatalf("Read %s failed: %v", id, err)
		}
		if !bytes.Equal(readData, expected) {
			t.Fatalf("data mismatch for %s: got %s, want %s", id, readData, expected)
		}
	}
}

// TestHierarchicalMemoryWorkstationDefaults verifies default sizing matching RTX 5090 FE + Ryzen 5950X + NVMe.
func TestHierarchicalMemoryWorkstationDefaults(t *testing.T) {
	mgr, err := NewHierarchicalMemoryManager(HierarchicalMemoryConfig{}, nil)
	if err != nil {
		t.Fatalf("NewHierarchicalMemoryManager failed: %v", err)
	}

	stats := mgr.Stats()
	if stats.Tier0BandwidthGBps != 1792.0 {
		t.Fatalf("expected 1792.0 GB/s for RTX 5090 GDDR7, got %f", stats.Tier0BandwidthGBps)
	}
	if stats.Tier1BandwidthGBps != 50.0 {
		t.Fatalf("expected 50.0 GB/s for DDR4 / PCIe DMA, got %f", stats.Tier1BandwidthGBps)
	}
	if stats.Tier2BandwidthGBps != 7.1 {
		t.Fatalf("expected 7.1 GB/s for NVMe direct, got %f", stats.Tier2BandwidthGBps)
	}

	// Verify allocation on default profile works
	b, err := mgr.Allocate("default_profile_blk", 64*1024, Tier0VRAM, false)
	if err != nil {
		t.Fatalf("Allocate failed: %v", err)
	}
	if b.CurrentTier != Tier0VRAM {
		t.Fatalf("expected Tier0VRAM, got %v", b.CurrentTier)
	}
}
