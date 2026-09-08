package ctxmmu

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// recordingAdvisor records madvise calls for deterministic verification.
type recordingAdvisor struct {
	mu            sync.Mutex
	randomCalled  bool
	willneedCalls []willneedRecord
}

type willneedRecord struct {
	Off    int
	Length int
}

func (r *recordingAdvisor) MadviseRandom(data []byte) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.randomCalled = true
	return true
}

func (r *recordingAdvisor) MadviseWillneed(data []byte, off, length int) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.willneedCalls = append(r.willneedCalls, willneedRecord{Off: off, Length: length})
	return true
}

func TestLazyTensorGather_MADVRandomSuppressesSequentialReadahead(t *testing.T) {
	advisor := &recordingAdvisor{}
	opts := DefaultGatherOptions()
	opts.Advisor = advisor
	opts.PinEngramRAM = false // Standard APU mode

	// Allocate a synthetic 1 MB buffer for testing
	buf := make([]byte, 1024*1024)
	gather, err := NewLazyTensorGatherFromData(buf, opts)
	if err != nil {
		t.Fatalf("failed to create LazyTensorGather: %v", err)
	}
	defer gather.Close()

	stats := gather.Stats()
	if !stats.MADVRandomApplied {
		t.Fatal("expected MADV_RANDOM to be marked as applied in stats")
	}

	advisor.mu.Lock()
	defer advisor.mu.Unlock()
	if !advisor.randomCalled {
		t.Fatal("expected advisor.MadviseRandom to be invoked during initialization")
	}
}

func TestLazyTensorGather_UBatchPrefetchWILLNEED(t *testing.T) {
	advisor := &recordingAdvisor{}
	opts := DefaultGatherOptions()
	opts.Advisor = advisor
	opts.NumHeads = 4
	opts.VocabPerHead = 100
	opts.RowSizeBytes = 32
	opts.TotalRows = 400 // 4 heads * 100 vocab

	data := make([]byte, 400*32)
	// Fill data with known patterns
	for r := 0; r < 400; r++ {
		for b := 0; b < 32; b++ {
			data[r*32+b] = byte((r + b) & 0xFF)
		}
	}

	gather, err := NewLazyTensorGatherFromData(data, opts)
	if err != nil {
		t.Fatalf("NewLazyTensorGatherFromData failed: %v", err)
	}
	defer gather.Close()

	// Micro-batch of token IDs
	ubatch := []int{5, 12, 99}
	prefetchedCount, err := gather.PrefetchUBatch(ubatch)
	if err != nil {
		t.Fatalf("PrefetchUBatch failed: %v", err)
	}

	// 3 tokens * 4 heads = 12 unique rows
	if prefetchedCount != 12 {
		t.Fatalf("expected 12 prefetched rows, got %d", prefetchedCount)
	}

	advisor.mu.Lock()
	defer advisor.mu.Unlock()
	if len(advisor.willneedCalls) != 12 {
		t.Fatalf("expected 12 WILLNEED calls, got %d", len(advisor.willneedCalls))
	}

	// Verify all offsets are 32-byte aligned and length is RowSizeBytes (32)
	for _, call := range advisor.willneedCalls {
		if call.Length != 32 {
			t.Errorf("expected call length 32, got %d", call.Length)
		}
		if call.Off%32 != 0 {
			t.Errorf("expected call offset multiple of 32, got %d", call.Off)
		}
	}

	// Verify stats
	stats := gather.Stats()
	if stats.PrefetchedRows != 12 {
		t.Errorf("stats.PrefetchedRows = %d, want 12", stats.PrefetchedRows)
	}
	if stats.PrefetchBatches != 1 {
		t.Errorf("stats.PrefetchBatches = %d, want 1", stats.PrefetchBatches)
	}
	if stats.SequentialBytesAvoided <= 0 {
		t.Errorf("expected positive sequential bytes avoided, got %d", stats.SequentialBytesAvoided)
	}
}

func TestLazyTensorGather_MemoryCarveoutPreserved(t *testing.T) {
	opts := DefaultGatherOptions()
	opts.HostMapped = true // Required

	gather, err := NewLazyTensorGatherFromData(make([]byte, 1024), opts)
	if err != nil {
		t.Fatalf("NewLazyTensorGatherFromData failed: %v", err)
	}
	defer gather.Close()

	if !gather.IsHostMapped() {
		t.Fatal("expected IsHostMapped() to be true")
	}
	if gather.IsGTTAllocated() {
		t.Fatal("expected IsGTTAllocated() to be false (never uploaded to GTT)")
	}

	if err := gather.ValidateMemoryCarveout(); err != nil {
		t.Fatalf("ValidateMemoryCarveout failed: %v", err)
	}
}

func TestLazyTensorGather_MemoryCarveoutViolationRejection(t *testing.T) {
	opts := DefaultGatherOptions()
	opts.HostMapped = false // Attempting to place outside host memory

	_, err := NewLazyTensorGatherFromData(make([]byte, 1024), opts)
	if !errors.Is(err, ErrGTTCarveoutViolation) {
		t.Fatalf("expected ErrGTTCarveoutViolation, got %v", err)
	}
}

func TestLazyTensorGather_PinEngramRAMFallback(t *testing.T) {
	advisor := &recordingAdvisor{}
	opts := DefaultGatherOptions()
	opts.Advisor = advisor
	opts.PinEngramRAM = true // High-capacity DRAM mode (>= 256 GB)

	buf := make([]byte, 65536)
	gather, err := NewLazyTensorGatherFromData(buf, opts)
	if err != nil {
		t.Fatalf("NewLazyTensorGatherFromData failed: %v", err)
	}
	defer gather.Close()

	stats := gather.Stats()
	if !stats.PinnedRAM {
		t.Fatal("expected PinnedRAM to be true")
	}
	if stats.MADVRandomApplied {
		t.Fatal("expected MADV_RANDOM NOT to be applied when PinEngramRAM is enabled")
	}

	advisor.mu.Lock()
	defer advisor.mu.Unlock()
	if len(advisor.willneedCalls) == 0 {
		t.Fatal("expected MADV_WILLNEED to be invoked across entire buffer upon initialization")
	}
	if advisor.willneedCalls[0].Off != 0 || advisor.willneedCalls[0].Length != 65536 {
		t.Fatalf("expected WILLNEED call for [0, 65536), got %+v", advisor.willneedCalls[0])
	}
}

func TestLazyTensorGather_ResidentRSSUnder2GiB(t *testing.T) {
	opts := DefaultGatherOptions()
	opts.MaxResidentRSS = DefaultMaxResidentRSS // 2.0 GiB

	gather, err := NewLazyTensorGatherFromData(make([]byte, 100000), opts)
	if err != nil {
		t.Fatalf("NewLazyTensorGatherFromData failed: %v", err)
	}
	defer gather.Close()

	// Prefetch several micro-batches
	_, _ = gather.PrefetchUBatch([]int{1, 2, 3, 4, 5})
	_, _ = gather.PrefetchUBatch([]int{10, 20, 30})

	rss := gather.EstimatedResidentRSS()
	if rss > DefaultMaxResidentRSS {
		t.Fatalf("resident RSS %d exceeds 2.0 GiB ceiling (%d)", rss, DefaultMaxResidentRSS)
	}

	if err := gather.AssertResidentRAMFootprint(DefaultMaxResidentRSS); err != nil {
		t.Fatalf("AssertResidentRAMFootprint failed: %v", err)
	}
}

func TestLazyTensorGather_BandwidthSavingsVerification(t *testing.T) {
	opts := DefaultGatherOptions()
	opts.RowSizeBytes = 130
	opts.NumHeads = 16
	opts.VocabPerHead = 1000
	opts.TotalRows = 16000

	gather, err := NewLazyTensorGatherFromData(make([]byte, 16000*130), opts)
	if err != nil {
		t.Fatalf("NewLazyTensorGatherFromData failed: %v", err)
	}
	defer gather.Close()

	// Prefetch a 512-token prompt
	tokens := make([]int, 512)
	for i := 0; i < 512; i++ {
		tokens[i] = i
	}

	count, err := gather.PrefetchUBatch(tokens)
	if err != nil {
		t.Fatalf("PrefetchUBatch failed: %v", err)
	}

	stats := gather.Stats()
	if stats.PrefetchedRows != int64(count) {
		t.Fatalf("expected %d prefetched rows, got %d", count, stats.PrefetchedRows)
	}

	// For 512 tokens * 16 heads = 8,192 rows
	// Without MADV_RANDOM: 8192 * 128 KiB = 1,073,741,824 bytes (~1 GiB)
	// With batched readahead: 8192 * 130 = 1,064,960 bytes (~1.01 MB)
	// Expected avoided: 8192 * (131072 - 130) = 1,072,676,864 bytes (~1.07 GB saved!)
	expectedMinAvoided := int64(count) * (DefaultOSReadaheadChunk - 130)
	if stats.SequentialBytesAvoided != expectedMinAvoided {
		t.Fatalf("expected %d bytes avoided, got %d", expectedMinAvoided, stats.SequentialBytesAvoided)
	}

	// Verify bandwidth reduction is > 99%
	standardSequentialBytes := int64(count) * DefaultOSReadaheadChunk
	actualBytes := int64(count) * 130
	reductionRatio := float64(standardSequentialBytes) / float64(actualBytes)
	if reductionRatio < 900.0 {
		t.Fatalf("expected readahead reduction ratio > 900x, got %.2fx", reductionRatio)
	}
}

func TestLazyTensorGather_PrefillThroughputFloor(t *testing.T) {
	opts := DefaultGatherOptions()
	gather, err := NewLazyTensorGatherFromData(make([]byte, 1024), opts)
	if err != nil {
		t.Fatalf("NewLazyTensorGatherFromData failed: %v", err)
	}
	defer gather.Close()

	// High-performance prefill witness (e.g. 352 tok/s on Strix Halo)
	gather.SetMeasuredPrefillThroughput(352.0)
	if err := gather.ValidatePrefillThroughput(352.0); err != nil {
		t.Fatalf("ValidatePrefillThroughput failed for 352 tok/s: %v", err)
	}

	// Sub-threshold throughput (< 300 tok/s) should fail validation
	if err := gather.ValidatePrefillThroughput(250.0); !errors.Is(err, ErrPrefillThroughputTooLow) {
		t.Fatalf("expected ErrPrefillThroughputTooLow for 250 tok/s, got %v", err)
	}
}

func TestLazyTensorGather_GatherRowAndBatchParity(t *testing.T) {
	opts := DefaultGatherOptions()
	opts.NumHeads = 2
	opts.VocabPerHead = 10
	opts.RowSizeBytes = 8
	opts.TotalRows = 20

	data := make([]byte, 20*8)
	for i := 0; i < len(data); i++ {
		data[i] = byte(i & 0xFF)
	}

	gather, err := NewLazyTensorGatherFromData(data, opts)
	if err != nil {
		t.Fatalf("NewLazyTensorGatherFromData failed: %v", err)
	}
	defer gather.Close()

	// Single row gather
	row0, err := gather.GatherRow(0)
	if err != nil {
		t.Fatalf("GatherRow(0) failed: %v", err)
	}
	if !bytes.Equal(row0, data[0:8]) {
		t.Fatalf("row 0 data mismatch: got %v, want %v", row0, data[0:8])
	}

	// Token embedding gather: token 3, head 1 => row index = 1*10 + 3 = 13
	row13, err := gather.GatherTokenEmbedding(3, 1)
	if err != nil {
		t.Fatalf("GatherTokenEmbedding failed: %v", err)
	}
	if !bytes.Equal(row13, data[13*8:14*8]) {
		t.Fatalf("token embedding mismatch: got %v, want %v", row13, data[13*8:14*8])
	}

	// Batch gather
	batch, err := gather.GatherBatch([]int{1, 4})
	if err != nil {
		t.Fatalf("GatherBatch failed: %v", err)
	}
	if len(batch) != 2 || len(batch[0]) != 2 || len(batch[1]) != 2 {
		t.Fatalf("unexpected batch shape: %d x %d", len(batch), len(batch[0]))
	}
	if !bytes.Equal(batch[0][0], data[1*8:2*8]) {
		t.Errorf("batch[0][0] mismatch")
	}
	if !bytes.Equal(batch[1][1], data[14*8:15*8]) {
		t.Errorf("batch[1][1] mismatch")
	}
}

func TestLazyTensorGather_FileBackedRead(t *testing.T) {
	tempDir := t.TempDir()
	filePath := filepath.Join(tempDir, "engram_table.bin")

	// Create test file with 16 rows of 16 bytes each
	content := make([]byte, 16*16)
	for i := 0; i < len(content); i++ {
		content[i] = byte((i * 3) & 0xFF)
	}
	if err := os.WriteFile(filePath, content, 0644); err != nil {
		t.Fatalf("failed to write test file: %v", err)
	}

	opts := DefaultGatherOptions()
	opts.TablePath = filePath
	opts.RowSizeBytes = 16
	opts.NumHeads = 2
	opts.VocabPerHead = 8
	opts.TotalRows = 16

	gather, err := NewLazyTensorGather(opts)
	if err != nil {
		t.Fatalf("NewLazyTensorGather failed: %v", err)
	}
	defer gather.Close()

	row, err := gather.GatherRow(5)
	if err != nil {
		t.Fatalf("GatherRow(5) failed: %v", err)
	}
	if !bytes.Equal(row, content[5*16:6*16]) {
		t.Fatalf("file-backed row mismatch: got %v, want %v", row, content[5*16:6*16])
	}
}

func TestLazyTensorGather_BoundsAndErrorHandling(t *testing.T) {
	opts := DefaultGatherOptions()
	opts.TotalRows = 10
	opts.RowSizeBytes = 8
	opts.NumHeads = 2
	opts.VocabPerHead = 5

	gather, err := NewLazyTensorGatherFromData(make([]byte, 80), opts)
	if err != nil {
		t.Fatalf("NewLazyTensorGatherFromData failed: %v", err)
	}
	defer gather.Close()

	// Out of bounds row
	if _, err := gather.GatherRow(-1); !errors.Is(err, ErrInvalidRowIndex) {
		t.Errorf("expected ErrInvalidRowIndex for -1, got %v", err)
	}
	if _, err := gather.GatherRow(10); !errors.Is(err, ErrInvalidRowIndex) {
		t.Errorf("expected ErrInvalidRowIndex for 10, got %v", err)
	}

	// Invalid token ID
	if _, err := gather.GatherTokenEmbedding(-1, 0); !errors.Is(err, ErrInvalidTokenID) {
		t.Errorf("expected ErrInvalidTokenID for negative token, got %v", err)
	}

	// Invalid head index
	if _, err := gather.GatherTokenEmbedding(0, 2); !errors.Is(err, ErrInvalidHeadIndex) {
		t.Errorf("expected ErrInvalidHeadIndex for head 2, got %v", err)
	}

	// Prefetch closed
	if err := gather.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
	if _, err := gather.PrefetchUBatch([]int{0}); !errors.Is(err, ErrGatherClosed) {
		t.Errorf("expected ErrGatherClosed, got %v", err)
	}
	if _, err := gather.GatherRow(0); !errors.Is(err, ErrGatherClosed) {
		t.Errorf("expected ErrGatherClosed, got %v", err)
	}
}

func TestLazyTensorGather_ConcurrentAccess(t *testing.T) {
	opts := DefaultGatherOptions()
	opts.NumHeads = 4
	opts.VocabPerHead = 100
	opts.RowSizeBytes = 16
	opts.TotalRows = 400

	data := make([]byte, 400*16)
	gather, err := NewLazyTensorGatherFromData(data, opts)
	if err != nil {
		t.Fatalf("NewLazyTensorGatherFromData failed: %v", err)
	}
	defer gather.Close()

	var wg sync.WaitGroup
	workers := 8
	iterations := 50

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				tok := (workerID*10 + i) % 100
				_, _ = gather.PrefetchUBatch([]int{tok})
				_, _ = gather.GatherTokenEmbedding(tok, workerID%4)
				_ = gather.Stats()
			}
		}(w)
	}

	wg.Wait()

	stats := gather.Stats()
	if stats.PrefetchedRows == 0 {
		t.Fatal("expected positive prefetched rows under concurrency")
	}
}

func TestLazyTensorGather_TensorReadLazyDefaultAndStats(t *testing.T) {
	opts := DefaultGatherOptions()
	if !opts.TensorReadLazy {
		t.Fatal("expected DefaultGatherOptions().TensorReadLazy to default to true")
	}

	buf := make([]byte, 1024)
	gather, err := NewLazyTensorGatherFromData(buf, opts)
	if err != nil {
		t.Fatalf("NewLazyTensorGatherFromData failed: %v", err)
	}
	defer gather.Close()

	if !gather.IsTensorReadLazy() {
		t.Fatal("expected IsTensorReadLazy() to be true")
	}

	stats := gather.Stats()
	if !stats.TensorReadLazy {
		t.Fatal("expected GatherStats.TensorReadLazy to be true")
	}

	// Verify normalizeOptions defaults TensorReadLazy to true on zero-value options (!PinEngramRAM)
	zeroOpts := GatherOptions{HostMapped: true}
	gatherZero, err := NewLazyTensorGatherFromData(buf, zeroOpts)
	if err != nil {
		t.Fatalf("NewLazyTensorGatherFromData with zero options failed: %v", err)
	}
	defer gatherZero.Close()

	if !gatherZero.IsTensorReadLazy() {
		t.Fatal("expected gatherZero.IsTensorReadLazy() to default to true")
	}
	if !gatherZero.Stats().TensorReadLazy {
		t.Fatal("expected gatherZero.Stats().TensorReadLazy to default to true")
	}

	// Verify that when PinEngramRAM is set, TensorReadLazy is false
	pinnedOpts := DefaultGatherOptions()
	pinnedOpts.PinEngramRAM = true
	gatherPinned, err := NewLazyTensorGatherFromData(buf, pinnedOpts)
	if err != nil {
		t.Fatalf("NewLazyTensorGatherFromData with PinEngramRAM failed: %v", err)
	}
	defer gatherPinned.Close()

	if gatherPinned.IsTensorReadLazy() {
		t.Fatal("expected IsTensorReadLazy() to be false when PinEngramRAM is true")
	}
	if gatherPinned.Stats().TensorReadLazy {
		t.Fatal("expected GatherStats.TensorReadLazy to be false when PinEngramRAM is true")
	}
}
