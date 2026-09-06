package l3kv

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// mockEvictBackend implements abi.KVBackend and SpanFileResolver for testing.
type mockEvictBackend struct {
	evictCalls int
	file       *os.File
	offset     int64
	length     int64
}

func (m *mockEvictBackend) Len() int                    { return 100 }
func (m *mockEvictBackend) Prefill(ids []int) []float32 { return nil }
func (m *mockEvictBackend) Evict(from, n int) int {
	m.evictCalls++
	return n
}
func (m *mockEvictBackend) ModelID() string { return "mock" }
func (m *mockEvictBackend) StageSpan(ctx context.Context, digest string, from, n int) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyOK}, nil
}
func (m *mockEvictBackend) RestoreSpan(ctx context.Context, digest string) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyOK}, nil
}

func (m *mockEvictBackend) SpanFileRange(from, n int) (file *os.File, offset, length int64, ok bool) {
	if m.file != nil {
		return m.file, m.offset, m.length, true
	}
	return nil, 0, 0, false
}

func TestEvictionIssuesDeallocate(t *testing.T) {
	dir := t.TempDir()
	filePath := filepath.Join(dir, "cache_span.bin")

	const fileSize = 128 * 1024 // 128 KiB
	initialData := bytes.Repeat([]byte{0xAA}, fileSize)
	if err := os.WriteFile(filePath, initialData, 0o644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	file, err := os.OpenFile(filePath, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatalf("OpenFile failed: %v", err)
	}
	defer file.Close()

	dealloc := NewAsyncDeallocator(64)
	defer dealloc.Close()

	// Span to evict and deallocate: middle 64 KiB [32768, 98304)
	const deallocOffset int64 = 32 * 1024
	const deallocLength int64 = 64 * 1024

	mockInner := &mockEvictBackend{
		file:   file,
		offset: deallocOffset,
		length: deallocLength,
	}

	mem := newMemStore()
	b := New(mockInner, mem)
	b = WithDeallocator(b, dealloc)

	// Trigger span eviction
	removed := b.Evict(0, 10)
	if removed != 10 {
		t.Fatalf("Evict returned %d, want 10", removed)
	}
	if mockInner.evictCalls != 1 {
		t.Fatalf("mockInner.evictCalls = %d, want 1", mockInner.evictCalls)
	}

	// Close the deallocator to ensure all queued background tasks finish draining.
	if err := dealloc.Close(); err != nil {
		t.Fatalf("Close deallocator failed: %v", err)
	}

	stats := dealloc.Stats()
	if stats.SubmittedCount != 1 {
		t.Errorf("SubmittedCount = %d, want 1", stats.SubmittedCount)
	}
	if stats.ProcessedCount != 1 {
		t.Errorf("ProcessedCount = %d, want 1", stats.ProcessedCount)
	}
	if stats.ErrorCount != 0 {
		t.Errorf("ErrorCount = %d, want 0", stats.ErrorCount)
	}

	// Verify the file content:
	// Leading section [0, deallocOffset) must still be 0xAA.
	lead := make([]byte, deallocOffset)
	if _, err := file.ReadAt(lead, 0); err != nil {
		t.Fatalf("ReadAt leading failed: %v", err)
	}
	if !bytes.Equal(lead, bytes.Repeat([]byte{0xAA}, int(deallocOffset))) {
		t.Errorf("leading data corrupted after deallocate")
	}

	// Deallocated range [deallocOffset, deallocOffset+deallocLength) must now be zeroed.
	deallocated := make([]byte, deallocLength)
	if _, err := file.ReadAt(deallocated, deallocOffset); err != nil {
		t.Fatalf("ReadAt deallocated range failed: %v", err)
	}
	if !bytes.Equal(deallocated, make([]byte, deallocLength)) {
		t.Errorf("deallocated range contains non-zero data")
	}

	// Trailing section [deallocOffset+deallocLength, fileSize) must still be 0xAA.
	trailLen := fileSize - (deallocOffset + deallocLength)
	trail := make([]byte, trailLen)
	if _, err := file.ReadAt(trail, deallocOffset+deallocLength); err != nil {
		t.Fatalf("ReadAt trailing failed: %v", err)
	}
	if !bytes.Equal(trail, bytes.Repeat([]byte{0xAA}, int(trailLen))) {
		t.Errorf("trailing data corrupted after deallocate")
	}
}

func TestAsyncDeallocatorProcessing(t *testing.T) {
	dir := t.TempDir()
	dealloc := NewAsyncDeallocator(128, 2)
	defer dealloc.Close()

	const numFiles = 10
	const blockSize = 16 * 1024
	files := make([]*os.File, numFiles)

	for i := 0; i < numFiles; i++ {
		p := filepath.Join(dir, fmt.Sprintf("chunk_%d.bin", i))
		if err := os.WriteFile(p, bytes.Repeat([]byte{byte(i + 1)}, blockSize), 0o644); err != nil {
			t.Fatalf("WriteFile failed: %v", err)
		}
		f, err := os.OpenFile(p, os.O_RDWR, 0o644)
		if err != nil {
			t.Fatalf("OpenFile failed: %v", err)
		}
		defer f.Close()
		files[i] = f
	}

	var completed atomic.Int64
	var wg sync.WaitGroup

	for i := 0; i < numFiles; i++ {
		wg.Add(1)
		f := files[i]
		err := dealloc.SubmitWithCallback(f, 0, int64(blockSize), func(err error) {
			defer wg.Done()
			if err != nil {
				t.Errorf("unexpected dealloc error: %v", err)
			}
			completed.Add(1)
		})
		if err != nil {
			t.Fatalf("Submit failed: %v", err)
		}
	}

	wg.Wait()

	if got := completed.Load(); got != numFiles {
		t.Errorf("completed callbacks = %d, want %d", got, numFiles)
	}

	stats := dealloc.Stats()
	if stats.ProcessedCount != numFiles {
		t.Errorf("ProcessedCount = %d, want %d", stats.ProcessedCount, numFiles)
	}
	if stats.ErrorCount != 0 {
		t.Errorf("ErrorCount = %d, want 0", stats.ErrorCount)
	}

	// Verify all files were zeroed or truncated.
	for i, f := range files {
		buf := make([]byte, blockSize)
		n, _ := f.ReadAt(buf, 0)
		if n > 0 && !bytes.Equal(buf[:n], make([]byte, n)) {
			t.Errorf("file %d not zeroed", i)
		}
	}
}

func TestAsyncDeallocatorQueueFullNonBlocking(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "drop_test.bin")
	_ = os.WriteFile(p, make([]byte, 1024), 0o644)
	f, err := os.OpenFile(p, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatalf("OpenFile failed: %v", err)
	}
	defer f.Close()

	// Queue with capacity 1 to easily trigger saturation.
	dealloc := NewAsyncDeallocator(1, 0)
	defer dealloc.Close()

	// Rapidly submit requests to test non-blocking drop behavior.
	dropped := 0
	for i := 0; i < 50; i++ {
		err := dealloc.Submit(f, 0, 512)
		if errors.Is(err, ErrQueueFull) {
			dropped++
		}
	}

	stats := dealloc.Stats()
	if stats.DroppedCount == 0 && dropped == 0 {
		t.Logf("Note: worker processed requests before queue saturated")
	} else if stats.DroppedCount != uint64(dropped) {
		t.Errorf("Stats DroppedCount %d != dropped errors %d", stats.DroppedCount, dropped)
	}
}

func TestAsyncDeallocatorGracefulShutdown(t *testing.T) {
	dealloc := NewAsyncDeallocator(16)

	dir := t.TempDir()
	p := filepath.Join(dir, "close_test.bin")
	_ = os.WriteFile(p, make([]byte, 4096), 0o644)
	f, err := os.OpenFile(p, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatalf("OpenFile failed: %v", err)
	}
	defer f.Close()

	// Submit one valid job
	if err := dealloc.Submit(f, 0, 1024); err != nil {
		t.Fatalf("Submit failed: %v", err)
	}

	// Close deallocator
	if err := dealloc.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Second close must be safe and return nil
	if err := dealloc.Close(); err != nil {
		t.Errorf("Second Close failed: %v", err)
	}

	// Submit after close must return ErrDeallocatorClosed
	if err := dealloc.Submit(f, 0, 1024); !errors.Is(err, ErrDeallocatorClosed) {
		t.Errorf("Submit after Close = %v, want ErrDeallocatorClosed", err)
	}
}

func TestDeallocateFileRangeValidation(t *testing.T) {
	if err := DeallocateFileRange(nil, 0, 100); err == nil {
		t.Error("DeallocateFileRange(nil) should return error")
	}

	dir := t.TempDir()
	p := filepath.Join(dir, "val_test.bin")
	_ = os.WriteFile(p, make([]byte, 1024), 0o644)
	f, err := os.OpenFile(p, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatalf("OpenFile failed: %v", err)
	}
	defer f.Close()

	if err := DeallocateFileRange(f, -1, 100); err == nil {
		t.Error("DeallocateFileRange with negative offset should fail")
	}

	if err := DeallocateFileRange(f, 0, -1); err == nil {
		t.Error("DeallocateFileRange with negative length should fail")
	}

	if err := DeallocateFileRange(f, 0, 0); err != nil {
		t.Errorf("DeallocateFileRange with length 0 should succeed, got: %v", err)
	}
}

func TestDeallocateFileRangeTruncationFallback(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "trunc_test.bin")
	const size = 64 * 1024
	if err := os.WriteFile(p, bytes.Repeat([]byte{0xBB}, size), 0o644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}
	f, err := os.OpenFile(p, os.O_RDWR, 0o644)
	if err != nil {
		t.Fatalf("OpenFile failed: %v", err)
	}
	defer f.Close()

	// Deallocate range extending to EOF: [32768, 65536)
	if err := fallbackDeallocate(f, 32*1024, 32*1024); err != nil {
		t.Fatalf("fallbackDeallocate at EOF failed: %v", err)
	}

	fi, err := f.Stat()
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	if fi.Size() != 32*1024 {
		t.Errorf("file size after EOF deallocation = %d, want %d", fi.Size(), 32*1024)
	}

	// Deallocate remaining range starting at 0
	if err := fallbackDeallocate(f, 0, 32*1024); err != nil {
		t.Fatalf("fallbackDeallocate from 0 failed: %v", err)
	}
	fi, err = f.Stat()
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	if fi.Size() != 0 {
		t.Errorf("file size after full deallocation = %d, want 0", fi.Size())
	}

	// Deallocating beyond EOF must be a no-op and never grow the file
	if err := fallbackDeallocate(f, 1024, 2048); err != nil {
		t.Fatalf("fallbackDeallocate beyond EOF failed: %v", err)
	}
	fi, err = f.Stat()
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	if fi.Size() != 0 {
		t.Errorf("file size after beyond-EOF deallocation = %d, want 0 (file was extended!)", fi.Size())
	}
}
