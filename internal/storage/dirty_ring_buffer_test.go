package storage

import (
	"bytes"
	"math/rand"
	"sync"
	"testing"
)

func TestDirtyRingBuffer_BasicWriteReadFlush(t *testing.T) {
	var writtenExtents []Extent
	var diskMu sync.Mutex

	diskWriter := func(offset uint64, data []byte) error {
		diskMu.Lock()
		defer diskMu.Unlock()
		writtenExtents = append(writtenExtents, Extent{
			Offset: offset,
			Length: uint64(len(data)),
			Data:   append([]byte(nil), data...),
		})
		return nil
	}

	cfg := DirtyRingBufferConfig{
		BufferCapacityBytes: 8 * 1024 * 1024, // 8 MiB
		FlushThresholdBytes: 4 * 1024 * 1024,
		ChunkSizeBytes:      2 * 1024 * 1024,
		MaxDirtyPages:       1024,
		DiskWriter:          diskWriter,
	}

	buf, err := NewDirtyRingBuffer(cfg)
	if err != nil {
		t.Fatalf("failed to create buffer: %v", err)
	}
	defer buf.Close()

	// Write 3 pages
	page1 := bytes.Repeat([]byte{0xAA}, 4096)
	page2 := bytes.Repeat([]byte{0xBB}, 4096)
	page3 := bytes.Repeat([]byte{0xCC}, 4096)

	if err := buf.WritePage(1, 0, page1); err != nil {
		t.Fatalf("WritePage 1 failed: %v", err)
	}
	if err := buf.WritePage(2, 4096, page2); err != nil {
		t.Fatalf("WritePage 2 failed: %v", err)
	}
	if err := buf.WritePage(3, 8192, page3); err != nil {
		t.Fatalf("WritePage 3 failed: %v", err)
	}

	// Read resident pages by ID
	read1, err := buf.ReadPage(1)
	if err != nil || !bytes.Equal(read1, page1) {
		t.Fatalf("ReadPage 1 mismatch: err=%v, got=%x", err, read1[:8])
	}
	read2, err := buf.ReadPage(2)
	if err != nil || !bytes.Equal(read2, page2) {
		t.Fatalf("ReadPage 2 mismatch: err=%v", err)
	}

	// Non-existent page
	if _, err := buf.ReadPage(999); err != ErrPageNotFound {
		t.Fatalf("expected ErrPageNotFound, got %v", err)
	}

	// ReadAt offset
	target := make([]byte, 8192)
	n, err := buf.ReadAt(0, target)
	if err != nil || n < 8192 {
		t.Fatalf("ReadAt failed: n=%d, err=%v", n, err)
	}
	if !bytes.Equal(target[:4096], page1) || !bytes.Equal(target[4096:8192], page2) {
		t.Fatalf("ReadAt content mismatch")
	}

	// Flush pending
	flushedBytes, extentCount, err := buf.FlushPending()
	if err != nil {
		t.Fatalf("FlushPending failed: %v", err)
	}
	if flushedBytes != 12288 {
		t.Fatalf("expected 12288 flushed bytes, got %d", flushedBytes)
	}
	if extentCount != 1 {
		t.Fatalf("expected 1 coalesced extent, got %d", extentCount)
	}

	// Pages remain readable in DRAM cache after flush
	readClean, err := buf.ReadPage(1)
	if err != nil || !bytes.Equal(readClean, page1) {
		t.Fatalf("ReadPage 1 after flush mismatch: err=%v", err)
	}

	stats := buf.Stats()
	if stats.CurrentDirtyBytes != 0 || stats.CurrentDirtyPages != 0 {
		t.Fatalf("dirty stats not reset: bytes=%d, pages=%d", stats.CurrentDirtyBytes, stats.CurrentDirtyPages)
	}
	if stats.TotalRandomWrites != 3 {
		t.Fatalf("expected 3 random writes, got %d", stats.TotalRandomWrites)
	}
	if stats.TotalSequentialFlushedBytes != 12288 {
		t.Fatalf("expected 12288 sequential flushed bytes, got %d", stats.TotalSequentialFlushedBytes)
	}
}

func TestDirtyRingBuffer_CoalescingAdjacentWrites(t *testing.T) {
	var flushedExtents []Extent
	var mu sync.Mutex

	diskWriter := func(offset uint64, data []byte) error {
		mu.Lock()
		defer mu.Unlock()
		flushedExtents = append(flushedExtents, Extent{
			Offset: offset,
			Length: uint64(len(data)),
			Data:   append([]byte(nil), data...),
		})
		return nil
	}

	cfg := DirtyRingBufferConfig{
		BufferCapacityBytes: 16 * 1024 * 1024,
		FlushThresholdBytes: 16 * 1024 * 1024, // Don't auto-flush during writes
		ChunkSizeBytes:      2 * 1024 * 1024,  // 2 MiB
		MaxDirtyPages:       4096,
		DiskWriter:          diskWriter,
	}

	buf, err := NewDirtyRingBuffer(cfg)
	if err != nil {
		t.Fatalf("failed to create buffer: %v", err)
	}
	defer buf.Close()

	// 512 adjacent 4KB writes: 512 * 4096 = 2,097,152 bytes = 2 MiB
	const numPages = 512
	const pageSize = 4096

	for i := 0; i < numPages; i++ {
		pageID := uint64(i)
		offset := pageID * pageSize
		data := make([]byte, pageSize)
		data[0] = byte(i & 0xFF)
		data[1] = byte((i >> 8) & 0xFF)
		data[pageSize-1] = 0xEE

		if err := buf.WritePage(pageID, offset, data); err != nil {
			t.Fatalf("WritePage(%d) failed: %v", i, err)
		}
	}

	// Flush all 512 adjacent pages
	flushedBytes, extentCount, err := buf.FlushPending()
	if err != nil {
		t.Fatalf("FlushPending failed: %v", err)
	}

	const expectedBytes = 2 * 1024 * 1024 // 2 MiB
	if flushedBytes != expectedBytes {
		t.Fatalf("expected %d flushed bytes, got %d", expectedBytes, flushedBytes)
	}
	if extentCount != 1 {
		t.Fatalf("expected exactly 1 coalesced extent for 512 adjacent pages, got %d", extentCount)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(flushedExtents) != 1 {
		t.Fatalf("expected 1 flushed extent sent to disk, got %d", len(flushedExtents))
	}
	if flushedExtents[0].Offset != 0 {
		t.Fatalf("expected extent offset 0, got %d", flushedExtents[0].Offset)
	}
	if flushedExtents[0].Length != expectedBytes {
		t.Fatalf("expected extent length %d, got %d", expectedBytes, flushedExtents[0].Length)
	}

	// Verify data continuity within the 2MB extent
	extentData := flushedExtents[0].Data
	for i := 0; i < numPages; i++ {
		offset := i * pageSize
		if extentData[offset] != byte(i&0xFF) || extentData[offset+1] != byte((i>>8)&0xFF) {
			t.Fatalf("extent data corrupted at page %d", i)
		}
		if extentData[offset+pageSize-1] != 0xEE {
			t.Fatalf("extent data footer corrupted at page %d", i)
		}
	}
}

func TestDirtyRingBuffer_ConcurrentWrites_500Agents(t *testing.T) {
	cfg := DirtyRingBufferConfig{
		BufferCapacityBytes: 64 * 1024 * 1024, // 64 MiB
		FlushThresholdBytes: 32 * 1024 * 1024,
		ChunkSizeBytes:      2 * 1024 * 1024,
		MaxDirtyPages:       16384,
	}

	buf, err := NewDirtyRingBuffer(cfg)
	if err != nil {
		t.Fatalf("failed to create buffer: %v", err)
	}
	defer buf.Close()

	const numAgents = 500
	const writesPerAgent = 10
	var wg sync.WaitGroup
	wg.Add(numAgents)

	for i := 0; i < numAgents; i++ {
		agentID := i
		go func() {
			defer wg.Done()
			for j := 0; j < writesPerAgent; j++ {
				pageID := uint64(agentID*writesPerAgent + j)
				offset := pageID * 4096
				data := make([]byte, 4096)
				data[0] = byte(agentID & 0xFF)
				data[1] = byte(j)
				if err := buf.WritePage(pageID, offset, data); err != nil {
					t.Errorf("agent %d write failed: %v", agentID, err)
					return
				}
			}
		}()
	}

	wg.Wait()

	expectedWrites := uint64(numAgents * writesPerAgent)
	stats := buf.Stats()
	if stats.TotalRandomWrites != expectedWrites {
		t.Fatalf("expected %d total random writes, got %d", expectedWrites, stats.TotalRandomWrites)
	}

	flushedBytes, extentCount, err := buf.FlushPending()
	if err != nil {
		t.Fatalf("FlushPending failed: %v", err)
	}
	if flushedBytes == 0 || extentCount == 0 {
		t.Fatalf("expected non-zero flush: flushedBytes=%d, extentCount=%d", flushedBytes, extentCount)
	}
	t.Logf("500 agents: completed %d writes, flushed %d bytes across %d extents", expectedWrites, flushedBytes, extentCount)
}

func TestDirtyRingBuffer_WAFReduction_RealisticWorkload(t *testing.T) {
	cfg := DirtyRingBufferConfig{
		BufferCapacityBytes: 32 * 1024 * 1024, // 32 MiB
		FlushThresholdBytes: 24 * 1024 * 1024,
		ChunkSizeBytes:      2 * 1024 * 1024,
		BaselineWAF:         30.0,
		SequentialWAF:       1.1,
	}

	buf, err := NewDirtyRingBuffer(cfg)
	if err != nil {
		t.Fatalf("failed to create buffer: %v", err)
	}
	defer buf.Close()

	// Realistic KV-cache / agent workload:
	// 512 base pages representing active working set, written 4 times each with updates (2048 total writes)
	const uniquePages = 512
	const totalWrites = 2048
	rng := rand.New(rand.NewSource(42))

	for w := 0; w < totalWrites; w++ {
		pageIdx := rng.Intn(uniquePages)
		pageID := uint64(pageIdx)
		offset := pageID * 4096
		data := make([]byte, 4096)
		data[0] = byte(w & 0xFF)
		data[1] = byte(pageIdx & 0xFF)

		if err := buf.WritePage(pageID, offset, data); err != nil {
			t.Fatalf("WritePage failed: %v", err)
		}
	}

	flushedBytes, extentCount, err := buf.FlushPending()
	if err != nil {
		t.Fatalf("FlushPending failed: %v", err)
	}

	stats := buf.Stats()
	t.Logf("Realistic workload: TotalWrites=%d, TotalRandomBytes=%d, FlushedBytes=%d, Extents=%d",
		stats.TotalRandomWrites, stats.TotalRandomWriteBytes, flushedBytes, extentCount)
	t.Logf("MeasuredWAF=%.3fx, WAFReductionFactor=%.2fx, EstimatedLifespanMultiplier=%.2fx",
		stats.MeasuredWAF, stats.WAFReductionFactor, stats.EstimatedLifespanMultiplier)

	// In this workload:
	// Total random writes = 2048 * 4KB = 8 MiB.
	// Unique pages flushed = 512 * 4KB = 2 MiB.
	// Coalesced sequential write has WAF ~1.1x.
	// Measured WAF = (2 MiB * 1.1) / 8 MiB = 0.275x.
	// WAF reduction factor = 30.0 / 0.275 = ~109x >= 25x!
	if stats.WAFReductionFactor < 25.0 {
		t.Fatalf("expected WAF reduction factor >= 25x, got %.2fx", stats.WAFReductionFactor)
	}
	if stats.MeasuredWAF > 1.2 {
		t.Fatalf("expected MeasuredWAF <= 1.2, got %.2fx", stats.MeasuredWAF)
	}
	if stats.EstimatedLifespanMultiplier < 25.0 {
		t.Fatalf("expected LifespanMultiplier >= 25x, got %.2fx", stats.EstimatedLifespanMultiplier)
	}

	// Also verify that even without overwrites (1 write per page), coalescing achieves >= 25x reduction
	buf2, err := NewDirtyRingBuffer(DirtyRingBufferConfig{
		BufferCapacityBytes: 16 * 1024 * 1024,
		ChunkSizeBytes:      2 * 1024 * 1024,
		BaselineWAF:         30.0,
		SequentialWAF:       1.1,
	})
	if err != nil {
		t.Fatalf("failed to create buffer2: %v", err)
	}
	defer buf2.Close()

	for i := 0; i < 512; i++ {
		pageID := uint64(i)
		offset := pageID * 4096
		data := make([]byte, 4096)
		data[0] = byte(i)
		if err := buf2.WritePage(pageID, offset, data); err != nil {
			t.Fatalf("buf2 write failed: %v", err)
		}
	}
	if _, _, err := buf2.FlushPending(); err != nil {
		t.Fatalf("buf2 flush failed: %v", err)
	}
	stats2 := buf2.Stats()
	t.Logf("Single-write coalesced workload: MeasuredWAF=%.3fx, WAFReduction=%.2fx", stats2.MeasuredWAF, stats2.WAFReductionFactor)
	if stats2.WAFReductionFactor < 25.0 {
		t.Fatalf("expected single-write WAF reduction >= 25x (30.0 / 1.1 = 27.27x), got %.2fx", stats2.WAFReductionFactor)
	}
}

func TestDirtyRingBuffer_CapacityBoundariesAndFlushTrigger(t *testing.T) {
	// 1. Default capacity is 2 GiB
	defaultBuf, err := NewDirtyRingBuffer(DirtyRingBufferConfig{})
	if err != nil {
		t.Fatalf("default buffer creation failed: %v", err)
	}
	if defaultBuf.Config().BufferCapacityBytes != DefaultBufferCapacityBytes {
		t.Fatalf("expected default 2 GiB capacity (%d), got %d",
			DefaultBufferCapacityBytes, defaultBuf.Config().BufferCapacityBytes)
	}
	defaultBuf.Close()

	// 2. Max capacity of 4 GiB succeeds
	maxBuf, err := NewDirtyRingBuffer(DirtyRingBufferConfig{
		BufferCapacityBytes: MaxBufferCapacityBytes,
	})
	if err != nil {
		t.Fatalf("4 GiB buffer creation failed: %v", err)
	}
	if maxBuf.Config().BufferCapacityBytes != MaxBufferCapacityBytes {
		t.Fatalf("expected 4 GiB, got %d", maxBuf.Config().BufferCapacityBytes)
	}
	maxBuf.Close()

	// 3. Exceeding 4 GiB fails with ErrCapacityExceeded
	_, err = NewDirtyRingBuffer(DirtyRingBufferConfig{
		BufferCapacityBytes: 5 * 1024 * 1024 * 1024,
	})
	if err != ErrCapacityExceeded {
		t.Fatalf("expected ErrCapacityExceeded, got %v", err)
	}

	// 4. Flush triggering when reaching threshold
	flushesObserved := 0
	var fMu sync.Mutex
	flushWriter := func(offset uint64, data []byte) error {
		fMu.Lock()
		flushesObserved++
		fMu.Unlock()
		return nil
	}

	thresholdBuf, err := NewDirtyRingBuffer(DirtyRingBufferConfig{
		BufferCapacityBytes: 64 * 1024, // 64 KiB
		FlushThresholdBytes: 32 * 1024, // 32 KiB
		ChunkSizeBytes:      32 * 1024,
		MaxDirtyPages:       8,
		DiskWriter:          flushWriter,
	})
	if err != nil {
		t.Fatalf("thresholdBuf creation failed: %v", err)
	}
	defer thresholdBuf.Close()

	// Write 12 pages of 4KB each (48KB > 32KB threshold)
	for i := 0; i < 12; i++ {
		pageID := uint64(i)
		offset := pageID * 4096
		data := make([]byte, 4096)
		data[0] = byte(i)
		if err := thresholdBuf.WritePage(pageID, offset, data); err != nil {
			t.Fatalf("WritePage %d failed: %v", i, err)
		}
	}

	stats := thresholdBuf.Stats()
	if stats.TotalFlushes == 0 {
		t.Fatalf("expected automatic flush to be triggered, got 0 flushes")
	}
	fMu.Lock()
	if flushesObserved == 0 {
		t.Fatalf("expected DiskWriter to be called during auto-flush")
	}
	fMu.Unlock()

	// 5. Payload too large
	hugeData := make([]byte, 128*1024)
	if err := thresholdBuf.WritePage(100, 0, hugeData); err != ErrPayloadTooLarge {
		t.Fatalf("expected ErrPayloadTooLarge, got %v", err)
	}

	// 6. Invalid empty page
	if err := thresholdBuf.WritePage(101, 0, nil); err != ErrInvalidPageSize {
		t.Fatalf("expected ErrInvalidPageSize, got %v", err)
	}
}

func TestDirtyRingBuffer_Close(t *testing.T) {
	buf, err := NewDirtyRingBuffer(DirtyRingBufferConfig{
		BufferCapacityBytes: 1 * 1024 * 1024,
	})
	if err != nil {
		t.Fatalf("creation failed: %v", err)
	}

	data := []byte("session-checkpoint-token-42")
	if err := buf.WritePage(1, 0, data); err != nil {
		t.Fatalf("write failed: %v", err)
	}

	if err := buf.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}

	// Subsequent operations return ErrClosed
	if err := buf.WritePage(2, 4096, data); err != ErrClosed {
		t.Fatalf("expected ErrClosed on write, got %v", err)
	}
	if _, err := buf.ReadPage(1); err != ErrClosed {
		t.Fatalf("expected ErrClosed on read, got %v", err)
	}
	if _, _, err := buf.FlushPending(); err != ErrClosed {
		t.Fatalf("expected ErrClosed on flush, got %v", err)
	}
	// Calling Close again is idempotent
	if err := buf.Close(); err != nil {
		t.Fatalf("second close failed: %v", err)
	}
}

func BenchmarkDirtyRingBuffer_WritePage(b *testing.B) {
	buf, err := NewDirtyRingBuffer(DirtyRingBufferConfig{
		BufferCapacityBytes: 256 * 1024 * 1024,
		FlushThresholdBytes: 200 * 1024 * 1024,
		ChunkSizeBytes:      2 * 1024 * 1024,
	})
	if err != nil {
		b.Fatalf("failed to create buffer: %v", err)
	}
	defer buf.Close()

	data := make([]byte, 4096)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		pageID := uint64(i % 10000)
		offset := pageID * 4096
		_ = buf.WritePage(pageID, offset, data)
	}
}
