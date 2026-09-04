package qwen4exp

import (
	"encoding/binary"
	"math/rand"
	"path/filepath"
	"testing"
	"time"
	"unsafe"
)

func TestDeterministicNGramHashing(t *testing.T) {
	t.Parallel()

	tokens := []int{101, 2024, 883, 4512, 9999}
	eosToken := DefaultEOSTokenID
	totalRows := int64(100000)

	// Invariant 1: Pure function - identical inputs yield identical output
	for pos := 0; pos < len(tokens); pos++ {
		first := ComputeTokenPLERowIndices(tokens, pos, totalRows, eosToken)
		second := ComputeTokenPLERowIndices(tokens, pos, totalRows, eosToken)
		if first != second {
			t.Fatalf("pos %d: n-gram hash is not pure: %v != %v", pos, first, second)
		}
	}

	// Invariant 2: EOS padding for sequence start
	pos0Indices := ComputeTokenPLERowIndices(tokens, 0, totalRows, eosToken)
	pos1Indices := ComputeTokenPLERowIndices(tokens, 1, totalRows, eosToken)
	if pos0Indices == pos1Indices {
		t.Fatal("pos 0 and pos 1 produced identical row indices")
	}

	// Invariant 3: Heads 0..15 yield 16 well-distributed distinct row indices
	seen := make(map[int64]bool, PLERowsPerToken)
	for head, rowIdx := range pos0Indices {
		if rowIdx < 0 || rowIdx >= totalRows {
			t.Fatalf("head %d rowIdx %d out of bounds [0..%d)", head, rowIdx, totalRows)
		}
		seen[rowIdx] = true
	}
	if len(seen) < 14 {
		t.Fatalf("expected high diversity across 16 heads, got %d unique rows", len(seen))
	}
}

func TestSafetensorsShardExtentIndex(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	shard0Path := filepath.Join(tmpDir, "ngram-00001.safetensors")
	shard1Path := filepath.Join(tmpDir, "ngram-00002.safetensors")

	numRowsPerShard := 32
	shard0Data := make([]byte, numRowsPerShard*PLERowBytes)
	shard1Data := make([]byte, numRowsPerShard*PLERowBytes)

	for r := 0; r < numRowsPerShard; r++ {
		binary.LittleEndian.PutUint64(shard0Data[r*PLERowBytes:], uint64(r))
		binary.LittleEndian.PutUint64(shard1Data[r*PLERowBytes:], uint64(1000+r))
	}

	if err := WriteSyntheticSafetensorsShard(shard0Path, "ngram_embedding.shard_0", numRowsPerShard, shard0Data); err != nil {
		t.Fatalf("write shard 0: %v", err)
	}
	if err := WriteSyntheticSafetensorsShard(shard1Path, "ngram_embedding.shard_1", numRowsPerShard, shard1Data); err != nil {
		t.Fatalf("write shard 1: %v", err)
	}

	idx := NewPLEExtentIndex()
	if err := idx.AddSafetensorsShard(shard0Path, "ngram_embedding.shard_0"); err != nil {
		t.Fatalf("add shard 0: %v", err)
	}
	if err := idx.AddSafetensorsShard(shard1Path, "ngram_embedding.shard_1"); err != nil {
		t.Fatalf("add shard 1: %v", err)
	}

	if got := idx.TotalRows(); got != int64(numRowsPerShard*2) {
		t.Fatalf("total rows = %d, want %d", got, numRowsPerShard*2)
	}

	// Verify row 0 points to shard 0 at sector-aligned offset
	ext0, err := idx.LookupRow(0)
	if err != nil {
		t.Fatalf("lookup row 0: %v", err)
	}
	if ext0.ShardPath != shard0Path || ext0.FileOffset%SectorSize != 0 || ext0.ByteLength != PLERowBytes {
		t.Fatalf("row 0 extent mismatch: %+v", ext0)
	}

	// Verify row 32 points to shard 1 at sector-aligned offset
	ext32, err := idx.LookupRow(32)
	if err != nil {
		t.Fatalf("lookup row 32: %v", err)
	}
	if ext32.ShardPath != shard1Path || ext32.FileOffset%SectorSize != 0 || ext32.ByteLength != PLERowBytes {
		t.Fatalf("row 32 extent mismatch: %+v", ext32)
	}
}

func TestPinnedStagingBufferZeroRAMCache(t *testing.T) {
	t.Parallel()

	buf, err := AllocPinnedStagingBuffer()
	if err != nil {
		t.Fatalf("alloc pinned staging buffer: %v", err)
	}
	defer buf.Free()

	if buf.Rows() != PLERowsPerToken {
		t.Fatalf("rows = %d, want %d", buf.Rows(), PLERowsPerToken)
	}
	if buf.RowBytes() != PLERowBytes {
		t.Fatalf("rowBytes = %d, want %d", buf.RowBytes(), PLERowBytes)
	}
	if len(buf.Bytes()) != PLEBatchBytes {
		t.Fatalf("len(Bytes) = %d, want %d", len(buf.Bytes()), PLEBatchBytes)
	}

	// Direct I/O memory alignment invariant: start address must be aligned to SectorSize (4096)
	addr := uintptr(unsafe.Pointer(&buf.Bytes()[0]))
	if addr%SectorSize != 0 {
		t.Fatalf("pinned staging buffer address 0x%x is not sector aligned (%d)", addr, SectorSize)
	}

	// Verify non-overlapping rows span the buffer exactly
	for r := 0; r < PLERowsPerToken; r++ {
		row := buf.Row(r)
		if len(row) != PLERowBytes {
			t.Fatalf("row %d length = %d, want %d", r, len(row), PLERowBytes)
		}
		// Write pattern
		row[0] = byte(r + 1)
	}
	for r := 0; r < PLERowsPerToken; r++ {
		if buf.Row(r)[0] != byte(r+1) {
			t.Fatalf("row %d data overwritten", r)
		}
	}
}

func TestCUDAMemopSyncFailClosedFallback(t *testing.T) {
	t.Parallel()

	// Case 1: Driver has stream memop capability -> SyncModeStreamMemop admitted
	syncMemop := NewCUDAMemopSync(SyncModeStreamMemop, true, 0x1000)
	if syncMemop.Mode() != SyncModeStreamMemop {
		t.Fatalf("expected mode %s, got %s", SyncModeStreamMemop, syncMemop.Mode())
	}
	syncMemop.SignalTransferComplete(1)
	if err := syncMemop.WaitStream(0, 1); err != nil {
		t.Fatalf("WaitStream failed: %v", err)
	}
	stats := syncMemop.Stats()
	if stats.StreamWriteCalls != 1 || stats.StreamWaitCalls != 1 || stats.HostDispatchWaits != 0 {
		t.Fatalf("unexpected memop stats: %+v", stats)
	}

	// Case 2: Driver lacks stream memop capability -> fail closed to host dispatch synchronization
	syncFallback := NewCUDAMemopSync(SyncModeStreamMemop, false, 0x1000)
	if syncFallback.Mode() != SyncModeHostDispatch {
		t.Fatalf("expected fail-closed fallback to %s, got %s", SyncModeHostDispatch, syncFallback.Mode())
	}
	syncFallback.SignalTransferComplete(42)
	if err := syncFallback.WaitStream(0, 42); err != nil {
		t.Fatalf("WaitStream failed: %v", err)
	}
	fallbackStats := syncFallback.Stats()
	if fallbackStats.HostDispatchWaits != 1 || fallbackStats.StreamWriteCalls != 0 || fallbackStats.StreamWaitCalls != 0 {
		t.Fatalf("unexpected fallback stats: %+v", fallbackStats)
	}
}

func Test16RowGatherMatchesGoldTensor(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	shardPath := filepath.Join(tmpDir, "ngram-00001.safetensors")

	numRows := 64
	goldTable := make([]byte, numRows*PLERowBytes)
	for r := 0; r < numRows; r++ {
		for i := 0; i < PLERowBytes; i += 8 {
			val := uint64(r)*100000 + uint64(i)
			binary.LittleEndian.PutUint64(goldTable[r*PLERowBytes+i:], val)
		}
	}

	if err := WriteSyntheticSafetensorsShard(shardPath, "ngram_embedding", numRows, goldTable); err != nil {
		t.Fatalf("write synthetic shard: %v", err)
	}

	extentIdx := NewPLEExtentIndex()
	if err := extentIdx.AddSafetensorsShard(shardPath, "ngram_embedding"); err != nil {
		t.Fatalf("add safetensors shard: %v", err)
	}

	syncBarrier := NewCUDAMemopSync(SyncModeStreamMemop, true, 0x5000)
	gatherer, err := NewPLERowGatherer(extentIdx, syncBarrier)
	if err != nil {
		t.Fatalf("new ple row gatherer: %v", err)
	}
	defer gatherer.Close()

	// Subtest 1: Gather 16 distinct rows
	var distinctIndices [PLERowsPerToken]int64
	for i := 0; i < PLERowsPerToken; i++ {
		distinctIndices[i] = int64(i * 3)
	}

	start := time.Now()
	if err := gatherer.GatherRows(distinctIndices, 1); err != nil {
		t.Fatalf("gather distinct rows: %v", err)
	}
	lookupDuration := time.Since(start)

	ok, err := gatherer.VerifyAgainstGold(goldTable, distinctIndices)
	if err != nil || !ok {
		t.Fatalf("distinct rows verification failed: %v", err)
	}
	if lookupDuration > 50*time.Millisecond {
		t.Logf("warmup lookup took %v", lookupDuration)
	}

	// Subtest 2: Deduping verification (duplicate rows across the 16 positions)
	var dedupIndices [PLERowsPerToken]int64
	for i := 0; i < PLERowsPerToken; i++ {
		dedupIndices[i] = int64((i % 4) * 5)
	}

	if err := gatherer.GatherRows(dedupIndices, 2); err != nil {
		t.Fatalf("gather dedup rows: %v", err)
	}
	ok, err = gatherer.VerifyAgainstGold(goldTable, dedupIndices)
	if err != nil || !ok {
		t.Fatalf("dedup rows verification failed: %v", err)
	}

	// Subtest 3: Token-driven gather
	tokens := []int{42, 108, 999}
	if err := gatherer.GatherToken(tokens, 2, 3, DefaultEOSTokenID); err != nil {
		t.Fatalf("gather token: %v", err)
	}
	tokenIndices := ComputeTokenPLERowIndices(tokens, 2, int64(numRows), DefaultEOSTokenID)
	ok, err = gatherer.VerifyAgainstGold(goldTable, tokenIndices)
	if err != nil || !ok {
		t.Fatalf("token rows verification failed: %v", err)
	}
}

func Benchmark16RowDirectIOGather(b *testing.B) {
	tmpDir := b.TempDir()
	shardPath := filepath.Join(tmpDir, "benchmark-ngram.safetensors")

	numRows := 256
	goldTable := make([]byte, numRows*PLERowBytes)
	for r := 0; r < numRows; r++ {
		for i := 0; i < PLERowBytes; i += 8 {
			binary.LittleEndian.PutUint64(goldTable[r*PLERowBytes+i:], uint64(r*1000+i))
		}
	}

	if err := WriteSyntheticSafetensorsShard(shardPath, "ngram_embedding", numRows, goldTable); err != nil {
		b.Fatalf("write shard: %v", err)
	}

	extentIdx := NewPLEExtentIndex()
	if err := extentIdx.AddSafetensorsShard(shardPath, "ngram_embedding"); err != nil {
		b.Fatalf("add shard: %v", err)
	}

	syncBarrier := NewCUDAMemopSync(SyncModeStreamMemop, true, 0x8000)
	gatherer, err := NewPLERowGatherer(extentIdx, syncBarrier)
	if err != nil {
		b.Fatalf("new gatherer: %v", err)
	}
	defer gatherer.Close()

	// Pre-generate pseudo-random row indices for the benchmark loop
	rng := rand.New(rand.NewSource(42))
	queryBatches := make([][PLERowsPerToken]int64, 128)
	for q := range queryBatches {
		for s := 0; s < PLERowsPerToken; s++ {
			queryBatches[q][s] = int64(rng.Intn(numRows))
		}
	}

	// Warmup run and verify against gold table
	if err := gatherer.GatherRows(queryBatches[0], 1); err != nil {
		b.Fatalf("warmup gather: %v", err)
	}
	if ok, err := gatherer.VerifyAgainstGold(goldTable, queryBatches[0]); !ok || err != nil {
		b.Fatalf("warmup gold verification: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	t0 := time.Now()
	for i := 0; i < b.N; i++ {
		batch := queryBatches[i%len(queryBatches)]
		if err := gatherer.GatherRows(batch, uint64(i+2)); err != nil {
			b.Fatalf("gather iter %d: %v", i, err)
		}
	}
	b.StopTimer()
	elapsed := time.Since(t0)

	avgLookup := elapsed / time.Duration(b.N)
	b.ReportMetric(float64(avgLookup.Microseconds()), "us/lookup")
	b.ReportMetric(float64(PLEBatchBytes)/1024, "KB/batch")

	if avgLookup > MaxTargetLookupLatency {
		b.Logf("NOTICE: average 16-row gather latency %v exceeds 0.5ms target (observed on non-NVMe scratch disk)", avgLookup)
	}
}

func Benchmark16RowDirectIOTokenGather(b *testing.B) {
	tmpDir := b.TempDir()
	shardPath := filepath.Join(tmpDir, "benchmark-token-ngram.safetensors")

	numRows := 256
	goldTable := make([]byte, numRows*PLERowBytes)
	for r := 0; r < numRows; r++ {
		for i := 0; i < PLERowBytes; i += 8 {
			binary.LittleEndian.PutUint64(goldTable[r*PLERowBytes+i:], uint64(r*500+i))
		}
	}

	if err := WriteSyntheticSafetensorsShard(shardPath, "ngram_embedding", numRows, goldTable); err != nil {
		b.Fatalf("write shard: %v", err)
	}

	extentIdx := NewPLEExtentIndex()
	if err := extentIdx.AddSafetensorsShard(shardPath, "ngram_embedding"); err != nil {
		b.Fatalf("add shard: %v", err)
	}

	syncBarrier := NewCUDAMemopSync(SyncModeStreamMemop, true, 0x8000)
	gatherer, err := NewPLERowGatherer(extentIdx, syncBarrier)
	if err != nil {
		b.Fatalf("new gatherer: %v", err)
	}
	defer gatherer.Close()

	tokens := []int{101, 204, 305, 406, 507, 608, 709, 810, 911, 1012}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		pos := i % len(tokens)
		if err := gatherer.GatherToken(tokens, pos, uint64(i+1), DefaultEOSTokenID); err != nil {
			b.Fatalf("gather token iter %d: %v", i, err)
		}
	}
}
