package compute

import (
	"bytes"
	"testing"
	"time"
)

func TestCUDANVMeSQECQEMarshal(t *testing.T) {
	// Test 64-byte SQE serialization and deserialization
	origSQE := CUDANVMeSubmissionQueueEntry{
		CDW0:      0x00010002, // Read command, CID 1
		NSID:      1,
		Reserved0: 0x0102030405060708,
		MPTR:      0x10002000,
		PRP1:      0x8000000000, // Blackwell BAR1 VRAM base
		PRP2:      0x8000010000, // Subsequent physical page
		CDW10:     1000,         // Starting LBA low
		CDW11:     5,            // Starting LBA high
		CDW12:     127,          // 128 logical blocks
		CDW13:     0xABCD1234,
		CDW14:     0x567890AB,
		CDW15:     0xCDEF0123,
	}

	sqeBytes := origSQE.MarshalBinary()
	if len(sqeBytes) != CUDANVMeSQESize {
		t.Fatalf("expected SQE size %d, got %d", CUDANVMeSQESize, len(sqeBytes))
	}

	var parsedSQE CUDANVMeSubmissionQueueEntry
	if err := parsedSQE.UnmarshalBinary(sqeBytes); err != nil {
		t.Fatalf("SQE unmarshal failed: %v", err)
	}

	if parsedSQE != origSQE {
		t.Errorf("SQE mismatch:\ngot  %+v\nwant %+v", parsedSQE, origSQE)
	}

	// Verify CDW indexing accessors
	if parsedSQE.CDW(0) != 0x00010002 || parsedSQE.CDW(1) != 1 || parsedSQE.CDW(10) != 1000 {
		t.Errorf("CDW accessor failed: CDW0=%x, CDW1=%d, CDW10=%d", parsedSQE.CDW(0), parsedSQE.CDW(1), parsedSQE.CDW(10))
	}

	// Test SQE short buffer failure
	var shortSQE CUDANVMeSubmissionQueueEntry
	if err := shortSQE.UnmarshalBinary(make([]byte, CUDANVMeSQESize-1)); err == nil {
		t.Errorf("expected error on truncated SQE buffer, got nil")
	}

	// Test 16-byte CQE serialization and deserialization
	origCQE := CUDANVMeCompletionQueueEntry{
		DW0:    0x12345678,
		DW1:    0x87654321,
		SQHead: 16,
		SQID:   2,
		CID:    101,
		Status: 0x0001, // Phase tag bit 0 = 1
	}

	cqeBytes := origCQE.MarshalBinary()
	if len(cqeBytes) != CUDANVMeCQESize {
		t.Fatalf("expected CQE size %d, got %d", CUDANVMeCQESize, len(cqeBytes))
	}

	var parsedCQE CUDANVMeCompletionQueueEntry
	if err := parsedCQE.UnmarshalBinary(cqeBytes); err != nil {
		t.Fatalf("CQE unmarshal failed: %v", err)
	}

	if parsedCQE != origCQE {
		t.Errorf("CQE mismatch:\ngot  %+v\nwant %+v", parsedCQE, origCQE)
	}

	if !parsedCQE.PhaseTag() {
		t.Errorf("expected PhaseTag == true")
	}
	if parsedCQE.StatusCode() != 0 {
		t.Errorf("expected StatusCode == 0, got %d", parsedCQE.StatusCode())
	}
	if parsedCQE.DoNotRetry() {
		t.Errorf("expected DoNotRetry == false")
	}

	// Test CQE short buffer failure
	var shortCQE CUDANVMeCompletionQueueEntry
	if err := shortCQE.UnmarshalBinary(make([]byte, CUDANVMeCQESize-1)); err == nil {
		t.Errorf("expected error on truncated CQE buffer, got nil")
	}
}

func TestCUDABaMVRAMQueueSubmitAndPoll(t *testing.T) {
	// Reject zero VRAM base address
	_, err := NewCUDABaMVRAMQueue(1, 0, 0, 64, 0xD0000000)
	if err == nil {
		t.Fatalf("expected error for vramBase == 0, got nil")
	}

	q, err := NewCUDABaMVRAMQueue(1, 0, 0x8000000000, 64, 0xD0000000)
	if err != nil {
		t.Fatalf("NewCUDABaMVRAMQueue failed: %v", err)
	}

	if q.Arch != CUDABlackwellArch {
		t.Errorf("Arch = %s, want %s", q.Arch, CUDABlackwellArch)
	}
	if !q.Phase() {
		t.Errorf("initial Phase = false, want true")
	}
	if q.DoorbellRings() != 0 {
		t.Errorf("initial DoorbellRings = %d, want 0", q.DoorbellRings())
	}

	cmds := []*CUDANVMeP2PCommand{
		{
			CommandID:      1,
			Opcode:         CUDANVMeOpcodeRead,
			NamespaceID:    1,
			StartingLBA:    0,
			BlockCount:     8,
			TargetVRAMAddr: 0x8000000000,
			ByteLength:     4096,
		},
		{
			CommandID:      2,
			Opcode:         CUDANVMeOpcodeWrite,
			NamespaceID:    1,
			StartingLBA:    8,
			BlockCount:     8,
			TargetVRAMAddr: 0x8000001000,
			ByteLength:     4096,
		},
	}

	if err := q.SubmitBatch(cmds); err != nil {
		t.Fatalf("SubmitBatch failed: %v", err)
	}

	if q.DoorbellRings() != 1 {
		t.Errorf("DoorbellRings = %d, want 1", q.DoorbellRings())
	}
	if q.PendingCount() != 2 {
		t.Errorf("PendingCount = %d, want 2", q.PendingCount())
	}
	if q.SQTail() != 2 {
		t.Errorf("SQTail = %d, want 2", q.SQTail())
	}

	resolved := q.PollCompletions(10)
	if resolved != 2 {
		t.Fatalf("resolved = %d, want 2", resolved)
	}

	if q.PendingCount() != 0 {
		t.Errorf("PendingCount after poll = %d, want 0", q.PendingCount())
	}

	for _, cmd := range cmds {
		if !cmd.Completed {
			t.Errorf("expected command %d to be completed", cmd.CommandID)
		}
		if cmd.Status != 0 {
			t.Errorf("expected command %d status 0, got %d", cmd.CommandID, cmd.Status)
		}
		if cmd.StagingCopyCount() != 0 {
			t.Errorf("expected 0 staging copies, got %d", cmd.StagingCopyCount())
		}
	}

	// Test queue full capacity condition
	smallQueue, err := NewCUDABaMVRAMQueue(2, 0, 0x8000000000, 3, 0xD0000000)
	if err != nil {
		t.Fatalf("NewCUDABaMVRAMQueue failed: %v", err)
	}

	fillCmds := []*CUDANVMeP2PCommand{
		{CommandID: 10, Opcode: CUDANVMeOpcodeRead, BlockCount: 1},
		{CommandID: 11, Opcode: CUDANVMeOpcodeRead, BlockCount: 1},
		{CommandID: 12, Opcode: CUDANVMeOpcodeRead, BlockCount: 1},
	}
	if err := smallQueue.SubmitBatch(fillCmds); err == nil {
		t.Fatalf("expected queue full error on overflow, got nil")
	}
}

func TestCUDADirectStorageMemorySlab(t *testing.T) {
	cfg := CUDADirectStorageConfig{
		NodeID:      0,
		BlockSize:   64 * 1024,
		TotalBlocks: 16,
		BaseAddress: 0x8000000000,
	}

	slab, err := NewCUDADirectStorageMemorySlab(cfg)
	if err != nil {
		t.Fatalf("NewCUDADirectStorageMemorySlab failed: %v", err)
	}

	stats := slab.Stats()
	if stats.TotalBlocks != 16 || stats.FreeBlocks != 16 || stats.AllocatedBlocks != 0 {
		t.Fatalf("unexpected initial stats: %+v", stats)
	}

	// 1. AllocBlock
	blk, err := slab.AllocBlock(100)
	if err != nil {
		t.Fatalf("AllocBlock failed: %v", err)
	}
	if blk.NVMeLBA != 100 {
		t.Errorf("expected LBA 100, got %d", blk.NVMeLBA)
	}
	if blk.AccessCount != 1 {
		t.Errorf("expected AccessCount 1, got %d", blk.AccessCount)
	}
	if blk.LastAccess <= 0 {
		t.Errorf("expected LastAccess > 0, got %d", blk.LastAccess)
	}

	// 2. WriteBlock
	testPayload := []byte("CUDA-Blackwell-RTX5090-BaM-P2PDMA-Storage-Direct")
	if err := slab.WriteBlock(100, testPayload); err != nil {
		t.Fatalf("WriteBlock failed: %v", err)
	}
	if !blk.IsDirty {
		t.Errorf("expected block to be marked dirty after write")
	}

	// 3. ReadBlock (Cache Hit)
	data, err := slab.ReadBlock(100)
	if err != nil {
		t.Fatalf("ReadBlock failed: %v", err)
	}
	if !bytes.Equal(data[:len(testPayload)], testPayload) {
		t.Errorf("ReadBlock data mismatch: got %s, want %s", string(data[:len(testPayload)]), string(testPayload))
	}

	// 4. ReadBlock (Cache Miss for new LBA 200)
	missData, err := slab.ReadBlock(200)
	if err != nil {
		t.Fatalf("ReadBlock miss failed: %v", err)
	}
	if len(missData) != 64*1024 {
		t.Errorf("expected 64KB read data, got %d", len(missData))
	}

	stats = slab.Stats()
	if stats.Hits < 1 {
		t.Errorf("expected Hits >= 1, got %d", stats.Hits)
	}
	if stats.Misses < 1 {
		t.Errorf("expected Misses >= 1, got %d", stats.Misses)
	}
	if stats.BytesWritten != uint64(len(testPayload)) {
		t.Errorf("BytesWritten = %d, want %d", stats.BytesWritten, len(testPayload))
	}
	if stats.BytesRead < 64*1024 {
		t.Errorf("BytesRead = %d, want >= 64KB", stats.BytesRead)
	}

	// 5. LRU access tracking
	// Access LBA 100 again to make LBA 200 the least recently used
	time.Sleep(2 * time.Millisecond)
	_, _ = slab.ReadBlock(100)

	oldest := slab.GetLRUBlock()
	if oldest == nil || oldest.NVMeLBA != 200 {
		t.Fatalf("expected LRU block to be LBA 200, got: %+v", oldest)
	}

	// Evict LRU block (LBA 200)
	evicted, err := slab.EvictLRU()
	if err != nil {
		t.Fatalf("EvictLRU failed: %v", err)
	}
	if evicted.BlockID != oldest.BlockID {
		t.Errorf("evicted block %d != expected %d", evicted.BlockID, oldest.BlockID)
	}

	// 6. FreeBlock
	if err := slab.FreeBlock(blk.BlockID); err != nil {
		t.Fatalf("FreeBlock failed: %v", err)
	}

	stats = slab.Stats()
	if stats.FreeBlocks != 16 {
		t.Errorf("expected 16 free blocks after cleanup, got %d", stats.FreeBlocks)
	}

	// 7. Exhaustion error test
	exhaustionSlab, err := NewCUDADirectStorageMemorySlab(CUDADirectStorageConfig{
		BlockSize:   4096,
		TotalBlocks: 2,
		BaseAddress: 0x8000000000,
	})
	if err != nil {
		t.Fatalf("NewCUDADirectStorageMemorySlab failed: %v", err)
	}

	if _, err := exhaustionSlab.AllocBlock(1); err != nil {
		t.Fatalf("AllocBlock 1 failed: %v", err)
	}
	if _, err := exhaustionSlab.AllocBlock(2); err != nil {
		t.Fatalf("AllocBlock 2 failed: %v", err)
	}
	// Third allocation must fail with slab exhaustion error
	if _, err := exhaustionSlab.AllocBlock(3); err == nil {
		t.Fatalf("expected slab exhaustion error, got nil")
	}
}

func TestCUDADirectStorageZeroCopyAssertion(t *testing.T) {
	// Verify zero host DRAM bounce copies: desc.StagingCopyCount() == 0
	cmd := &CUDANVMeP2PCommand{
		CommandID:      1,
		Opcode:         CUDANVMeOpcodeRead,
		NamespaceID:    1,
		StartingLBA:    0,
		BlockCount:     16,
		TargetVRAMAddr: 0x8000000000,
		ByteLength:     64 * 1024,
	}

	if cmd.StagingCopyCount() != 0 {
		t.Fatalf("cmd.StagingCopyCount() = %d, want 0", cmd.StagingCopyCount())
	}

	slab, err := NewCUDADirectStorageMemorySlab(CUDADirectStorageConfig{
		BlockSize:   64 * 1024,
		TotalBlocks: 8,
		BaseAddress: 0x8000000000,
	})
	if err != nil {
		t.Fatalf("NewCUDADirectStorageMemorySlab failed: %v", err)
	}

	if slab.StagingCopyCount() != 0 {
		t.Fatalf("slab.StagingCopyCount() = %d, want 0", slab.StagingCopyCount())
	}

	blk, err := slab.AllocBlock(42)
	if err != nil {
		t.Fatalf("AllocBlock failed: %v", err)
	}
	if blk.StagingCopyCount() != 0 {
		t.Fatalf("blk.StagingCopyCount() = %d, want 0", blk.StagingCopyCount())
	}

	stats := slab.Stats()
	if stats.StagingCopyCount != 0 {
		t.Fatalf("stats.StagingCopyCount = %d, want 0", stats.StagingCopyCount)
	}

	desc, err := slab.PrefetchBlocks([]uint64{101, 102})
	if err != nil {
		t.Fatalf("PrefetchBlocks failed: %v", err)
	}
	if err := desc.Wait(); err != nil {
		t.Fatalf("desc.Wait failed: %v", err)
	}

	if desc.StagingCopyCount() != 0 {
		t.Fatalf("desc.StagingCopyCount() = %d, want 0", desc.StagingCopyCount())
	}
}

func TestCUDAPrefetchDescriptor(t *testing.T) {
	slab, err := NewCUDADirectStorageMemorySlab(CUDADirectStorageConfig{
		BlockSize:   64 * 1024,
		TotalBlocks: 32,
		BaseAddress: 0x8000000000,
	})
	if err != nil {
		t.Fatalf("NewCUDADirectStorageMemorySlab failed: %v", err)
	}

	// Nil/empty LBA list returns error
	if _, err := slab.PrefetchBlocks(nil); err == nil {
		t.Fatalf("expected error on nil LBAs, got nil")
	}

	lbas := []uint64{1000, 1001, 1002, 1003}
	desc, err := slab.PrefetchBlocks(lbas)
	if err != nil {
		t.Fatalf("PrefetchBlocks failed: %v", err)
	}

	if desc.BlockCount != 4 {
		t.Errorf("BlockCount = %d, want 4", desc.BlockCount)
	}

	select {
	case <-desc.Done():
		if desc.Error() != nil {
			t.Fatalf("prefetch pipeline completed with error: %v", desc.Error())
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for PrefetchBlocks")
	}

	if !desc.IsComplete() {
		t.Errorf("IsComplete() = false, want true")
	}

	expectedBytes := uint64(4 * 64 * 1024)
	if desc.BytesPrefetched != expectedBytes {
		t.Errorf("BytesPrefetched = %d, want %d", desc.BytesPrefetched, expectedBytes)
	}

	if desc.ThroughputGBps() <= 0 {
		t.Errorf("ThroughputGBps = %f, want > 0", desc.ThroughputGBps())
	}

	stats := slab.Stats()
	if stats.AllocatedBlocks != 4 {
		t.Errorf("AllocatedBlocks = %d, want 4", stats.AllocatedBlocks)
	}
	if stats.BytesRead != expectedBytes {
		t.Errorf("BytesRead = %d, want %d", stats.BytesRead, expectedBytes)
	}
}

func BenchmarkCUDABaMStorageTransfer(b *testing.B) {
	blockSize := uint64(64 * 1024)
	slab, err := NewCUDADirectStorageMemorySlab(CUDADirectStorageConfig{
		BlockSize:   blockSize,
		TotalBlocks: 1024,
		BaseAddress: 0x8000000000,
	})
	if err != nil {
		b.Fatalf("NewCUDADirectStorageMemorySlab failed: %v", err)
	}

	payload := make([]byte, blockSize)
	for i := range payload {
		payload[i] = byte(i % 256)
	}

	b.SetBytes(int64(blockSize))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		lba := uint64(i % 1024)
		if err := slab.WriteBlock(lba, payload); err != nil {
			b.Fatalf("WriteBlock failed: %v", err)
		}
		if _, err := slab.ReadBlock(lba); err != nil {
			b.Fatalf("ReadBlock failed: %v", err)
		}
	}
}
