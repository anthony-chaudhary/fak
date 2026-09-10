package strix

import (
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
	"unsafe"
)

// failingBackingProvider always returns an error to test heap fallback.
type failingBackingProvider struct{}

func (f *failingBackingProvider) AllocateSlab(size int64) (unsafe.Pointer, []byte, error) {
	return nil, nil, errors.New("simulated DRM GTT allocation failure")
}

func (f *failingBackingProvider) FreeSlab(ptr unsafe.Pointer, size int64) error {
	return nil
}

func (f *failingBackingProvider) IsDirectGTT() bool {
	return false
}

func (f *failingBackingProvider) IoctlCount() int64 {
	return 0
}

// TestBlockCachingAllocator_PowerOfTwoBins verifies that all allocation sizes
// are mapped to exact segregated power-of-two bins from 64 KiB to 32 MiB,
// and enforces UMA zero-copy pointer identity (DevPtr == HostPtr).
func TestBlockCachingAllocator_PowerOfTwoBins(t *testing.T) {
	alloc := NewBlockCachingAllocator(DefaultAllocatorConfig())
	defer alloc.Reset()

	testCases := []struct {
		requested       int64
		expectedBinSize int64
		expectedBinIdx  int
	}{
		{requested: 1024, expectedBinSize: 64 * 1024, expectedBinIdx: 0},
		{requested: 64 * 1024, expectedBinSize: 64 * 1024, expectedBinIdx: 0},
		{requested: 64*1024 + 1, expectedBinSize: 128 * 1024, expectedBinIdx: 1},
		{requested: 130 * 1024, expectedBinSize: 256 * 1024, expectedBinIdx: 2},
		{requested: 300 * 1024, expectedBinSize: 512 * 1024, expectedBinIdx: 3},
		{requested: 700 * 1024, expectedBinSize: 1 * 1024 * 1024, expectedBinIdx: 4},
		{requested: 1500 * 1024, expectedBinSize: 2 * 1024 * 1024, expectedBinIdx: 5},
		{requested: 3 * 1024 * 1024, expectedBinSize: 4 * 1024 * 1024, expectedBinIdx: 6},
		{requested: 7 * 1024 * 1024, expectedBinSize: 8 * 1024 * 1024, expectedBinIdx: 7},
		{requested: 12 * 1024 * 1024, expectedBinSize: 16 * 1024 * 1024, expectedBinIdx: 8},
		{requested: 25 * 1024 * 1024, expectedBinSize: 32 * 1024 * 1024, expectedBinIdx: 9},
	}

	allocatedBlocks := make([]*BlockDescriptor, 0, len(testCases))
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("size_%d", tc.requested), func(t *testing.T) {
			b, err := alloc.Allocate(tc.requested)
			if err != nil {
				t.Fatalf("failed to allocate %d bytes: %v", tc.requested, err)
			}
			if b.BinSize != tc.expectedBinSize {
				t.Errorf("expected bin size %d, got %d", tc.expectedBinSize, b.BinSize)
			}
			if b.BinIndex != tc.expectedBinIdx {
				t.Errorf("expected bin index %d, got %d", tc.expectedBinIdx, b.BinIndex)
			}
			if b.ActualSize != tc.requested {
				t.Errorf("expected actual size %d, got %d", tc.requested, b.ActualSize)
			}
			if len(b.Slice()) != int(tc.requested) {
				t.Errorf("expected slice len %d, got %d", tc.requested, len(b.Slice()))
			}
			if b.DevPtr() != b.HostPtr() {
				t.Errorf("UMA pointer identity violated: DevPtr %x != HostPtr %x", b.DevPtr(), b.HostPtr())
			}
			allocatedBlocks = append(allocatedBlocks, b)
		})
	}

	// Free all blocks
	for _, b := range allocatedBlocks {
		if err := alloc.Free(b); err != nil {
			t.Fatalf("failed to free block: %v", err)
		}
	}

	telem := alloc.Telemetry()
	if telem.ActiveBytes != 0 {
		t.Errorf("expected active bytes 0 after free, got %d", telem.ActiveBytes)
	}
	if telem.ReleaseCount != int64(len(testCases)) {
		t.Errorf("expected release count %d, got %d", len(testCases), telem.ReleaseCount)
	}
}

// TestBlockCachingAllocator_SubMicrosecondRecycle verifies Criterion 2:
// in-process buffer recycling in power-of-two bins executes in < 1 microsecond (O(1)).
func TestBlockCachingAllocator_SubMicrosecondRecycle(t *testing.T) {
	alloc := NewBlockCachingAllocator(DefaultAllocatorConfig())
	defer alloc.Reset()

	const count = 1000
	blocks := make([]*BlockDescriptor, count)

	// Pre-populate pool with 1,000 blocks across multiple slabs.
	for i := 0; i < count; i++ {
		b, err := alloc.Allocate(128 * 1024)
		if err != nil {
			t.Fatalf("warmup allocate %d failed: %v", i, err)
		}
		blocks[i] = b
	}
	for i := 0; i < count; i++ {
		if err := alloc.Free(blocks[i]); err != nil {
			t.Fatalf("warmup free %d failed: %v", i, err)
		}
		blocks[i] = nil
	}

	// Measure O(1) recycling of 1,000 cached buffers from the pool.
	startRecycle := time.Now()
	for i := 0; i < count; i++ {
		b, err := alloc.Allocate(128 * 1024)
		if err != nil {
			t.Fatalf("recycle %d failed: %v", i, err)
		}
		blocks[i] = b
	}
	recycleElapsed := time.Since(startRecycle)
	avgRecycle := recycleElapsed / time.Duration(count)

	// Measure O(1) return of 1,000 buffers back to the pool.
	startFree := time.Now()
	for i := 0; i < count; i++ {
		if err := alloc.Free(blocks[i]); err != nil {
			t.Fatalf("free %d failed: %v", i, err)
		}
	}
	freeElapsed := time.Since(startFree)
	avgFree := freeElapsed / time.Duration(count)

	t.Logf("Recycled %d blocks in %v (avg recycle: %v, avg free: %v)", count, recycleElapsed, avgRecycle, avgFree)

	// Criterion 2: recycling cached buffers must execute in < 1 microsecond (1,000 ns).
	if avgRecycle > 1*time.Microsecond {
		t.Errorf("average pool recycle time %v exceeds 1 microsecond ceiling", avgRecycle)
	}
	if avgFree > 1*time.Microsecond {
		t.Errorf("average pool free time %v exceeds 1 microsecond ceiling", avgFree)
	}

	telem := alloc.Telemetry()
	if telem.PoolReclaims < int64(count) {
		t.Errorf("expected at least %d pool reclaims, got %d", count, telem.PoolReclaims)
	}
	if telem.IoctlBypassCount < int64(count) {
		t.Errorf("expected at least %d ioctl bypasses, got %d", count, telem.IoctlBypassCount)
	}
}

// TestBlockCachingAllocator_ZeroIoctlOnWarmPath verifies Criterion 4:
// under sustained token generation decode simulation, zero runtime GEM ioctls occur on the hot path.
func TestBlockCachingAllocator_ZeroIoctlOnWarmPath(t *testing.T) {
	provider := NewUMAPointerBackingProvider(nil)
	cfg := DefaultAllocatorConfig()
	cfg.BackingProvider = provider
	alloc := NewBlockCachingAllocator(cfg)
	defer alloc.Reset()

	// Initial warm-up: allocate 16 blocks (matching 16 Attention Heads / KV Cache slots).
	const numSlots = 16
	slots := make([]*BlockDescriptor, numSlots)
	for i := 0; i < numSlots; i++ {
		b, err := alloc.Allocate(64 * 1024) // 64 KiB block
		if err != nil {
			t.Fatalf("warmup allocation %d failed: %v", i, err)
		}
		slots[i] = b
	}

	for i := 0; i < numSlots; i++ {
		if err := alloc.Free(slots[i]); err != nil {
			t.Fatalf("warmup free %d failed: %v", i, err)
		}
		slots[i] = nil
	}

	// Capture baseline ioctl count after warm-up.
	baselineIoctls := provider.IoctlCount()

	// Simulate 200 autoregressive token generation decode steps.
	const tokenSteps = 200
	for step := 0; step < tokenSteps; step++ {
		// Allocate active KV cache block for this step
		slot, err := alloc.Allocate(64 * 1024)
		if err != nil {
			t.Fatalf("step %d: allocation failed: %v", step, err)
		}

		// Write dummy attention KV tokens
		data := slot.Slice()
		data[0] = byte(step & 0xFF)

		// Free at step completion
		if err := alloc.Free(slot); err != nil {
			t.Fatalf("step %d: free failed: %v", step, err)
		}
	}

	currentIoctls := provider.IoctlCount()
	deltaIoctls := currentIoctls - baselineIoctls

	if deltaIoctls != 0 {
		t.Errorf("expected 0 runtime ioctls on warm decode path, got %d", deltaIoctls)
	}

	telem := alloc.Telemetry()
	if telem.IoctlBypassCount < int64(tokenSteps) {
		t.Errorf("expected at least %d ioctl bypasses, got %d", tokenSteps, telem.IoctlBypassCount)
	}
}

// TestBlockCachingAllocator_HighWatermarkAndTrim verifies Criterion 3:
// memory pressure trims safely release unreferenced slabs back to the OS.
func TestBlockCachingAllocator_HighWatermarkAndTrim(t *testing.T) {
	alloc := NewBlockCachingAllocator(DefaultAllocatorConfig())
	defer alloc.Reset()

	// Allocate multiple 2MB slabs
	b1, err := alloc.Allocate(2 * 1024 * 1024)
	if err != nil {
		t.Fatalf("alloc b1 failed: %v", err)
	}
	b2, err := alloc.Allocate(2 * 1024 * 1024)
	if err != nil {
		t.Fatalf("alloc b2 failed: %v", err)
	}
	b3, err := alloc.Allocate(2 * 1024 * 1024)
	if err != nil {
		t.Fatalf("alloc b3 failed: %v", err)
	}
	b4, err := alloc.Allocate(2 * 1024 * 1024)
	if err != nil {
		t.Fatalf("alloc b4 failed: %v", err)
	}

	// Free all 4 blocks
	_ = alloc.Free(b1)
	_ = alloc.Free(b2)
	_ = alloc.Free(b3)
	_ = alloc.Free(b4)

	telemBefore := alloc.Telemetry()
	if telemBefore.CachedBytes < 8*1024*1024 {
		t.Fatalf("expected at least 8MB cached bytes, got %d", telemBefore.CachedBytes)
	}

	// Trim under yellow pressure (25%)
	trimmedYellow, err := alloc.TrimUnderPressure(PressureYellow)
	if err != nil {
		t.Fatalf("trim yellow failed: %v", err)
	}
	if trimmedYellow <= 0 {
		t.Errorf("expected trimmed bytes under yellow pressure, got %d", trimmedYellow)
	}

	// Trim under black pressure (100%)
	trimmedBlack, err := alloc.TrimUnderPressure(PressureBlack)
	if err != nil {
		t.Fatalf("trim black failed: %v", err)
	}
	if trimmedBlack <= 0 {
		t.Errorf("expected trimmed bytes under black pressure, got %d", trimmedBlack)
	}

	telemAfter := alloc.Telemetry()
	if telemAfter.CachedBytes != 0 {
		t.Errorf("expected 0 cached bytes after full trim, got %d", telemAfter.CachedBytes)
	}
}

// TestBlockCachingAllocator_ConcurrentAccess verifies multi-agent thread safety
// and zero data races under concurrent allocate/free cycles.
func TestBlockCachingAllocator_ConcurrentAccess(t *testing.T) {
	alloc := NewBlockCachingAllocator(DefaultAllocatorConfig())
	defer alloc.Reset()

	const numWorkers = 8
	const opsPerWorker = 200

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		go func(workerID int) {
			defer wg.Done()
			sizes := []int64{
				64 * 1024,
				128 * 1024,
				256 * 1024,
				512 * 1024,
				1024 * 1024,
			}

			for i := 0; i < opsPerWorker; i++ {
				sz := sizes[(workerID+i)%len(sizes)]
				b, err := alloc.Allocate(sz)
				if err != nil {
					t.Errorf("worker %d iter %d: allocate failed: %v", workerID, i, err)
					return
				}

				// Touch memory to verify valid mapped page
				slice := b.Slice()
				slice[0] = byte(workerID)
				slice[len(slice)-1] = byte(i)

				if err := alloc.Free(b); err != nil {
					t.Errorf("worker %d iter %d: free failed: %v", workerID, i, err)
					return
				}
			}
		}(w)
	}

	wg.Wait()

	telem := alloc.Telemetry()
	if telem.ActiveBytes != 0 {
		t.Errorf("expected active bytes 0 after concurrent run, got %d", telem.ActiveBytes)
	}
	expectedOps := int64(numWorkers * opsPerWorker)
	if telem.AllocationRequests != expectedOps {
		t.Errorf("expected %d allocation requests, got %d", expectedOps, telem.AllocationRequests)
	}
	if telem.ReleaseCount != expectedOps {
		t.Errorf("expected %d release count, got %d", expectedOps, telem.ReleaseCount)
	}
}

// TestBlockCachingAllocator_FallbackHeap verifies that when backing provider
// fails, the allocator gracefully falls back to heap backing.
func TestBlockCachingAllocator_FallbackHeap(t *testing.T) {
	cfg := DefaultAllocatorConfig()
	cfg.BackingProvider = &failingBackingProvider{}
	cfg.FallbackToHeap = true

	alloc := NewBlockCachingAllocator(cfg)
	defer alloc.Reset()

	b, err := alloc.Allocate(128 * 1024)
	if err != nil {
		t.Fatalf("expected fallback to heap to succeed, got: %v", err)
	}
	if b.BinSize != 128*1024 {
		t.Errorf("expected bin size 128KB, got %d", b.BinSize)
	}

	slice := b.Slice()
	slice[0] = 42
	if slice[0] != 42 {
		t.Errorf("heap buffer data corrupted")
	}

	if err := alloc.Free(b); err != nil {
		t.Fatalf("failed to free fallback heap block: %v", err)
	}

	telem := alloc.Telemetry()
	if telem.FallbackHeapCount != 1 {
		t.Errorf("expected FallbackHeapCount 1, got %d", telem.FallbackHeapCount)
	}
}

// TestBlockCachingAllocator_Errors verifies error handling on invalid requests.
func TestBlockCachingAllocator_Errors(t *testing.T) {
	alloc := NewBlockCachingAllocator(DefaultAllocatorConfig())
	defer alloc.Reset()

	// Zero or negative size
	if _, err := alloc.Allocate(0); !errors.Is(err, ErrInvalidBufferSize) {
		t.Errorf("expected ErrInvalidBufferSize on size 0, got %v", err)
	}
	if _, err := alloc.Allocate(-100); !errors.Is(err, ErrInvalidBufferSize) {
		t.Errorf("expected ErrInvalidBufferSize on negative size, got %v", err)
	}

	// Oversized request
	if _, err := alloc.Allocate(64 * 1024 * 1024); !errors.Is(err, ErrInvalidBufferSize) {
		t.Errorf("expected ErrInvalidBufferSize on >32MB request, got %v", err)
	}

	// Nil free
	if err := alloc.Free(nil); !errors.Is(err, ErrNilBlock) {
		t.Errorf("expected ErrNilBlock on Free(nil), got %v", err)
	}

	// Double free
	b, err := alloc.Allocate(64 * 1024)
	if err != nil {
		t.Fatalf("allocate failed: %v", err)
	}
	if err := alloc.Free(b); err != nil {
		t.Fatalf("first free failed: %v", err)
	}
	if err := alloc.Free(b); !errors.Is(err, ErrBlockClosed) {
		t.Errorf("expected ErrBlockClosed on double free, got %v", err)
	}
}
