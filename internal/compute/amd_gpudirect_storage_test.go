package compute

import (
	"testing"
	"time"
)

func TestNVMeSubmissionQueueEntry_BinarySerialization(t *testing.T) {
	orig := NVMeSubmissionQueueEntry{
		CDW0:  0x00010002, // Read command, CID 1
		NSID:  1,
		MPTR:  0x1000,
		PRP1:  0x8000000000, // Direct VRAM BAR1
		PRP2:  0x8000010000,
		CDW10: 100, // Starting LBA low
		CDW11: 0,   // Starting LBA high
		CDW12: 15,  // 16 blocks
	}

	data := orig.MarshalBinary()
	if len(data) != NVMeSQESize {
		t.Fatalf("expected SQE size %d bytes, got %d", NVMeSQESize, len(data))
	}

	var parsed NVMeSubmissionQueueEntry
	if err := parsed.UnmarshalBinary(data); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	if parsed != orig {
		t.Errorf("parsed SQE mismatch:\ngot  %+v\nwant %+v", parsed, orig)
	}
}

func TestNVMeCompletionQueueEntry_BinarySerialization(t *testing.T) {
	orig := NVMeCompletionQueueEntry{
		DW0:    0x12345678,
		DW1:    0,
		SQHead: 4,
		SQID:   1,
		CID:    101,
		Status: 0x0001, // Phase tag 1
	}

	data := orig.MarshalBinary()
	if len(data) != NVMeCQESize {
		t.Fatalf("expected CQE size %d bytes, got %d", NVMeCQESize, len(data))
	}

	var parsed NVMeCompletionQueueEntry
	if err := parsed.UnmarshalBinary(data); err != nil {
		t.Fatalf("unmarshal failed: %v", err)
	}

	if parsed != orig {
		t.Errorf("parsed CQE mismatch:\ngot  %+v\nwant %+v", parsed, orig)
	}
}

func TestNVMeVRAMQueue_BatchSubmissionAndPolling(t *testing.T) {
	q, err := NewNVMeVRAMQueue(1, 0, 0x80000000, 64, 0xD0000000)
	if err != nil {
		t.Fatalf("NewNVMeVRAMQueue failed: %v", err)
	}

	cmds := []*NVMeP2PCommand{
		{
			CommandID:      1,
			Opcode:         NVMeOpcodeRead,
			NamespaceID:    1,
			StartingLBA:    0,
			BlockCount:     8,
			TargetVRAMAddr: 0x80010000,
			ByteLength:     4096,
		},
		{
			CommandID:      2,
			Opcode:         NVMeOpcodeWrite,
			NamespaceID:    1,
			StartingLBA:    8,
			BlockCount:     8,
			TargetVRAMAddr: 0x80020000,
			ByteLength:     4096,
		},
	}

	if err := q.SubmitBatch(cmds); err != nil {
		t.Fatalf("SubmitBatch failed: %v", err)
	}

	resolved := q.PollCompletions(10)
	if resolved != 2 {
		t.Fatalf("expected 2 resolved completions, got %d", resolved)
	}

	for _, cmd := range cmds {
		if !cmd.Completed {
			t.Errorf("expected command %d to be completed", cmd.CommandID)
		}
		if cmd.StagingCopyCount() != 0 {
			t.Errorf("expected 0 staging copies, got %d", cmd.StagingCopyCount())
		}
	}
}

func TestDirectStorageMemorySlab_AllocFreeAndSwap(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
	engine, err := NewDirectStorageMemorySlab(hal, 0, 64*1024, 16, 0x8000000000)
	if err != nil {
		t.Fatalf("NewDirectStorageMemorySlab failed: %v", err)
	}

	stats := engine.Stats()
	if stats.TotalBlocks != 16 || stats.Free != 16 || stats.Allocated != 0 {
		t.Fatalf("unexpected initial stats: %+v", stats)
	}

	// Allocate block for LBA 1000
	blk, err := engine.AllocBlock(1000)
	if err != nil {
		t.Fatalf("AllocBlock failed: %v", err)
	}
	if blk.NVMeLBA != 1000 {
		t.Errorf("expected LBA 1000, got %d", blk.NVMeLBA)
	}

	// Direct NVMe Swap In (Disk -> VRAM, zero host copies)
	if err := engine.DirectNVMeSwapIn(blk, 128); err != nil {
		t.Fatalf("DirectNVMeSwapIn failed: %v", err)
	}

	// Direct NVMe Swap Out (VRAM -> Disk, zero host copies)
	if err := engine.DirectNVMeSwapOut(blk, 128); err != nil {
		t.Fatalf("DirectNVMeSwapOut failed: %v", err)
	}

	// Hit same LBA
	blkHit, err := engine.AllocBlock(1000)
	if err != nil {
		t.Fatalf("AllocBlock hit failed: %v", err)
	}
	if blkHit.BlockID != blk.BlockID {
		t.Errorf("expected cache hit on block %d, got %d", blk.BlockID, blkHit.BlockID)
	}

	stats = engine.Stats()
	if stats.CacheHits != 1 || stats.CacheMisses != 1 {
		t.Errorf("expected 1 hit, 1 miss; got: %+v", stats)
	}
	if stats.BytesRead != 64*1024 || stats.BytesWritten != 64*1024 {
		t.Errorf("expected 64KB read and written; got: %+v", stats)
	}

	// Free block
	if err := engine.FreeBlock(blk.BlockID); err != nil {
		t.Fatalf("FreeBlock failed: %v", err)
	}

	stats = engine.Stats()
	if stats.Free != 16 {
		t.Errorf("expected 16 free blocks after free, got %d", stats.Free)
	}
}

func TestDirectStorageMemorySlab_Prefetch(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
	engine, err := NewDirectStorageMemorySlab(hal, 0, 64*1024, 32, 0x8000000000)
	if err != nil {
		t.Fatalf("NewDirectStorageMemorySlab failed: %v", err)
	}

	done := engine.PrefetchBlocks(5000, 4)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("PrefetchBlocks failed: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for PrefetchBlocks")
	}

	stats := engine.Stats()
	if stats.Allocated != 4 {
		t.Errorf("expected 4 allocated blocks, got %d", stats.Allocated)
	}
	if stats.BytesRead != 4*64*1024 {
		t.Errorf("expected %d bytes read, got %d", 4*64*1024, stats.BytesRead)
	}
}

func TestDirectStorageMemorySlab_ExhaustionError(t *testing.T) {
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
	engine, err := NewDirectStorageMemorySlab(hal, 0, 4096, 2, 0x8000000000)
	if err != nil {
		t.Fatalf("NewDirectStorageMemorySlab failed: %v", err)
	}

	_, _ = engine.AllocBlock(100)
	_, _ = engine.AllocBlock(200)

	// Third allocation should fail
	_, err = engine.AllocBlock(300)
	if err == nil {
		t.Fatalf("expected exhaustion error, got nil")
	}
}
