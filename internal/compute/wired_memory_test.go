//go:build darwin

package compute

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"
	"syscall"
	"testing"
	"unsafe"
)

func allocateTestBuffer(t *testing.T, size int) ([]byte, uintptr) {
	t.Helper()
	pageSize := os.Getpagesize()
	if size%pageSize != 0 {
		t.Fatalf("size %d must be page-aligned (page size %d)", size, pageSize)
	}

	b, err := syscall.Mmap(-1, 0, size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap failed for size %d: %v", size, err)
	}
	addr := uintptr(unsafe.Pointer(&b[0]))
	if addr%uintptr(pageSize) != 0 {
		_ = syscall.Munmap(b)
		t.Fatalf("mmap returned non-page-aligned buffer: %x", addr)
	}
	return b, addr
}

func freeTestBuffer(t *testing.T, b []byte) {
	t.Helper()
	if err := syscall.Munmap(b); err != nil {
		t.Errorf("munmap failed: %v", err)
	}
}

func TestWiredMemoryDarwin_LockAndRelease(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("skipping macOS-specific wired memory test on non-darwin platform")
	}

	pageSize := os.Getpagesize()
	size := pageSize * 4
	b, addr := allocateTestBuffer(t, size)
	defer freeTestBuffer(t, b)

	// Fault in the pages by writing to each page
	for i := range b {
		b[i] = byte(i & 0xFF)
	}

	// Step 1: Check initial wired status (should be unwired)
	wired, count, err := IsMemoryWired(addr)
	if errors.Is(err, ErrWiredMemoryUnavailable) {
		t.Skip("skipping macOS wired memory test: wired memory unsupported (e.g. CGO_ENABLED=0)")
	}
	if err != nil {
		t.Fatalf("IsMemoryWired before lock failed: %v", err)
	}
	if wired || count != 0 {
		t.Fatalf("expected memory initially unwired, got wired=%v count=%d", wired, count)
	}

	// Step 2: Wire memory
	receipt, err := WireMemory(addr, uint64(size))
	if err != nil {
		t.Fatalf("WireMemory failed: %v", err)
	}
	if !receipt.Wired {
		t.Fatalf("expected receipt.Wired == true, got false")
	}
	if receipt.Method != WireMethodMachVMWire && receipt.Method != WireMethodMLock {
		t.Fatalf("expected wire method mach_vm_wire or mlock, got %v (%s)", receipt.Method, receipt.Method.String())
	}
	t.Logf("wired %d bytes at 0x%x using method: %s", size, addr, receipt.Method.String())

	// Step 3: Check wired status confirms wired count >= 1
	wired, count, err = IsMemoryWired(addr)
	if err != nil {
		t.Fatalf("IsMemoryWired after lock failed: %v", err)
	}
	if !wired || count < 1 {
		t.Fatalf("expected wired=true and count>=1, got wired=%v count=%d", wired, count)
	}

	// Step 4: Unwire memory
	err = UnwireMemory(addr, uint64(size), receipt.Method)
	if err != nil {
		t.Fatalf("UnwireMemory failed: %v", err)
	}

	// Step 5: Verify unwired
	wired, count, err = IsMemoryWired(addr)
	if err != nil {
		t.Fatalf("IsMemoryWired after unwire failed: %v", err)
	}
	if wired || count != 0 {
		t.Fatalf("expected memory unwired after UnwireMemory, got wired=%v count=%d", wired, count)
	}
}

func TestWiredMemory_AlignmentValidation(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("skipping macOS-specific alignment test on non-darwin platform")
	}

	pageSize := os.Getpagesize()
	size := pageSize * 2
	b, addr := allocateTestBuffer(t, size)
	defer freeTestBuffer(t, b)

	unalignedAddr := addr + 1
	receipt, err := WireMemory(unalignedAddr, uint64(size))
	if errors.Is(err, ErrWiredMemoryUnavailable) {
		t.Skip("skipping alignment test: wired memory unsupported (e.g. CGO_ENABLED=0)")
	}
	if !errors.Is(err, ErrNotPageAligned) {
		t.Fatalf("expected ErrNotPageAligned for unaligned addr, got receipt=%+v err=%v", receipt, err)
	}

	err = UnwireMemory(unalignedAddr, uint64(size), WireMethodMLock)
	if !errors.Is(err, ErrNotPageAligned) {
		t.Fatalf("expected ErrNotPageAligned for unaligned unwire, got %v", err)
	}
}

func TestWiredMemory_WorkingSetLimits(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("skipping macOS-specific working set limits test on non-darwin platform")
	}

	limits := DarwinWorkingSetLimits()
	if limits.RecommendedMaxWorkingSet == 0 && limits.MaxBufferLength == 0 {
		t.Skip("skipping working set limits: Metal device unavailable or CGO_ENABLED=0")
	}
	t.Logf("DarwinWorkingSetLimits: recommended_max=%d (%d MB), max_buffer=%d, unified=%v, current_alloc=%d",
		limits.RecommendedMaxWorkingSet, limits.RecommendedMaxWorkingSet/(1024*1024),
		limits.MaxBufferLength, limits.HasUnifiedMemory, limits.CurrentAllocatedSize)

	if runtime.GOARCH == "arm64" {
		if limits.RecommendedMaxWorkingSet == 0 {
			t.Fatalf("expected non-zero RecommendedMaxWorkingSet on Apple Silicon, got 0")
		}
		if !limits.HasUnifiedMemory {
			t.Fatalf("expected HasUnifiedMemory == true on Apple Silicon")
		}
		if limits.MaxBufferLength == 0 {
			t.Fatalf("expected non-zero MaxBufferLength on Apple Silicon, got 0")
		}
	}

	rec := RecommendedMaxWorkingSetSize()
	if rec != limits.RecommendedMaxWorkingSet {
		t.Fatalf("RecommendedMaxWorkingSetSize %d != limits.RecommendedMaxWorkingSet %d", rec, limits.RecommendedMaxWorkingSet)
	}
}

func TestWiredMemory_GracefulFallback(t *testing.T) {
	// Zero address
	_, err := WireMemory(0, 4096)
	if !errors.Is(err, ErrInvalidAddress) && !errors.Is(err, ErrWiredMemoryUnavailable) {
		t.Fatalf("expected ErrInvalidAddress or ErrWiredMemoryUnavailable for addr=0, got %v", err)
	}

	// Zero size
	pageSize := os.Getpagesize()
	b, addr := allocateTestBuffer(t, pageSize)
	defer freeTestBuffer(t, b)

	_, err = WireMemory(addr, 0)
	if !errors.Is(err, ErrInvalidSize) && !errors.Is(err, ErrWiredMemoryUnavailable) {
		t.Fatalf("expected ErrInvalidSize or ErrWiredMemoryUnavailable for size=0, got %v", err)
	}

	// Unwire with zero address
	err = UnwireMemory(0, 4096, WireMethodMLock)
	if !errors.Is(err, ErrInvalidAddress) && !errors.Is(err, ErrWiredMemoryUnavailable) {
		t.Fatalf("expected ErrInvalidAddress or ErrWiredMemoryUnavailable for unwire addr=0, got %v", err)
	}

	// Unwire with zero size
	err = UnwireMemory(addr, 0, WireMethodMLock)
	if !errors.Is(err, ErrInvalidSize) && !errors.Is(err, ErrWiredMemoryUnavailable) {
		t.Fatalf("expected ErrInvalidSize or ErrWiredMemoryUnavailable for unwire size=0, got %v", err)
	}

	// IsMemoryWired with zero address
	_, _, err = IsMemoryWired(0)
	if !errors.Is(err, ErrInvalidAddress) && !errors.Is(err, ErrWiredMemoryUnavailable) {
		t.Fatalf("expected ErrInvalidAddress or ErrWiredMemoryUnavailable for IsMemoryWired(0), got %v", err)
	}
}

func TestWiredMemory_HighLoadLock(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("skipping macOS-specific high load wired memory test on non-darwin platform")
	}

	// Allocate a multi-megabyte buffer (16 MB)
	size := 16 * 1024 * 1024
	pageSize := os.Getpagesize()
	if size%pageSize != 0 {
		size = ((size + pageSize - 1) / pageSize) * pageSize
	}

	b, addr := allocateTestBuffer(t, size)
	defer freeTestBuffer(t, b)

	// Fault in and write initial pattern
	for i := 0; i < size; i += pageSize {
		b[i] = byte(0x55)
		b[i+pageSize-1] = byte(0xAA)
	}

	// Wire the entire 16MB working set
	receipt, err := WireMemory(addr, uint64(size))
	if errors.Is(err, ErrWiredMemoryUnavailable) {
		t.Skip("skipping high load test: wired memory unsupported (e.g. CGO_ENABLED=0)")
	}
	if err != nil {
		t.Fatalf("WireMemory failed for %d MB buffer: %v", size/(1024*1024), err)
	}
	if !receipt.Wired {
		t.Fatalf("expected receipt.Wired == true for high-load buffer")
	}
	t.Logf("high-load wired %d MB successfully using %s", size/(1024*1024), receipt.Method.String())

	// Verify wired at start, midpoint, and near end
	for _, offset := range []uintptr{0, uintptr(size / 2), uintptr(size - pageSize)} {
		wired, count, err := IsMemoryWired(addr + offset)
		if err != nil {
			t.Fatalf("IsMemoryWired at offset %d failed: %v", offset, err)
		}
		if !wired || count < 1 {
			t.Fatalf("expected wired=true at offset %d, got wired=%v count=%d", offset, wired, count)
		}
	}

	// Perform memory-intensive write while wired
	for i := 0; i < size; i += pageSize {
		b[i] = byte(0x77)
	}

	// Verify contents are intact
	for i := 0; i < size; i += pageSize {
		if b[i] != byte(0x77) {
			t.Fatalf("memory corruption at index %d: expected 0x77, got 0x%x", i, b[i])
		}
	}

	// Unwire cleanly
	if err := UnwireMemory(addr, uint64(size), receipt.Method); err != nil {
		t.Fatalf("UnwireMemory failed on high-load buffer: %v", err)
	}

	// Verify unwired
	wired, count, err := IsMemoryWired(addr)
	if err != nil {
		t.Fatalf("IsMemoryWired after unwire failed: %v", err)
	}
	if wired || count != 0 {
		t.Fatalf("expected unwired after high-load UnwireMemory, got wired=%v count=%d", wired, count)
	}
}

func TestWiredMemory_ConcurrentLockAndRelease(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("skipping macOS-specific concurrent wired memory test on non-darwin platform")
	}

	const workers = 8
	pageSize := os.Getpagesize()
	bufferSize := pageSize * 4

	var wg sync.WaitGroup
	errCh := make(chan error, workers)

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			b, addr := allocateTestBuffer(t, bufferSize)
			defer freeTestBuffer(t, b)

			// Touch pages
			for i := range b {
				b[i] = byte(workerID)
			}

			// Wire memory
			receipt, err := WireMemory(addr, uint64(bufferSize))
			if err != nil {
				errCh <- fmt.Errorf("worker %d: WireMemory failed: %w", workerID, err)
				return
			}
			if !receipt.Wired {
				errCh <- fmt.Errorf("worker %d: receipt not wired", workerID)
				return
			}

			// Check wired
			wired, count, err := IsMemoryWired(addr)
			if err != nil {
				errCh <- fmt.Errorf("worker %d: IsMemoryWired failed: %w", workerID, err)
				return
			}
			if !wired || count < 1 {
				errCh <- fmt.Errorf("worker %d: expected wired=true, got wired=%v count=%d", workerID, wired, count)
				return
			}

			// Unwire memory
			if err := UnwireMemory(addr, uint64(bufferSize), receipt.Method); err != nil {
				errCh <- fmt.Errorf("worker %d: UnwireMemory failed: %w", workerID, err)
				return
			}

			// Verify unwired
			wired, count, err = IsMemoryWired(addr)
			if err != nil {
				errCh <- fmt.Errorf("worker %d: IsMemoryWired after unwire failed: %w", workerID, err)
				return
			}
			if wired || count != 0 {
				errCh <- fmt.Errorf("worker %d: expected unwired, got wired=%v count=%d", workerID, wired, count)
				return
			}
		}(w)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Errorf("concurrent error: %v", err)
	}
}

func TestWiredMemory_UnmappedAddress(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("skipping macOS-specific unmapped address test on non-darwin platform")
	}

	// Address far out in canonical user address space that is not mapped
	unmappedAddr := uintptr(0x7ffff0000000)

	wired, _, err := IsMemoryWired(unmappedAddr)
	if err == nil && wired {
		t.Fatalf("expected error or unmapped for 0x%x, got wired=true err=nil", unmappedAddr)
	}

	_, err = WireMemory(unmappedAddr, 4096)
	if err == nil {
		t.Fatalf("expected WireMemory on unmapped address 0x%x to fail, got nil", unmappedAddr)
	}
}

func TestWiredMemory_ReadOnlyMapping(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("skipping macOS-specific read-only wired memory test on non-darwin platform")
	}

	pageSize := os.Getpagesize()
	b, err := syscall.Mmap(-1, 0, pageSize, syscall.PROT_READ, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap failed: %v", err)
	}
	defer func() { _ = syscall.Munmap(b) }()

	addr := uintptr(unsafe.Pointer(&b[0]))
	receipt, err := WireMemory(addr, uint64(pageSize))
	if err != nil {
		t.Fatalf("WireMemory on read-only mapping failed: %v", err)
	}
	if !receipt.Wired {
		t.Fatalf("expected receipt.Wired == true for read-only mapping")
	}

	wired, count, err := IsMemoryWired(addr)
	if err != nil {
		t.Fatalf("IsMemoryWired failed: %v", err)
	}
	if !wired || count < 1 {
		t.Fatalf("expected wired=true count>=1 on read-only mapping, got wired=%v count=%d", wired, count)
	}

	if err := UnwireMemory(addr, uint64(pageSize), receipt.Method); err != nil {
		t.Fatalf("UnwireMemory failed: %v", err)
	}
}

func TestWireModelWeights_LoadAndTeardown(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("skipping macOS-specific wire model weights test on non-darwin platform")
	}

	pageSize := os.Getpagesize()
	weightSize := pageSize * 8
	buf, addr := allocateTestBuffer(t, weightSize)
	defer freeTestBuffer(t, buf)

	// Simulate simulated model weights initialized in mmap memory
	for i := range buf {
		buf[i] = byte((i * 31) & 0xFF)
	}

	// Step 1: Wire model weights upon model load
	receipt, teardown, err := WireModelBuffer(buf)
	if errors.Is(err, ErrWiredMemoryUnavailable) {
		t.Skip("skipping test: wired memory unsupported in this build environment")
	}
	if err != nil {
		t.Fatalf("WireModelBuffer failed: %v", err)
	}
	if !receipt.Wired {
		t.Fatalf("expected model weights wired == true, got false")
	}
	t.Logf("model weights wired: addr=0x%x size=%d method=%s", receipt.Addr, receipt.Size, receipt.Method.String())

	// Step 2: Verify memory is wired
	wired, count, err := IsMemoryWired(addr)
	if err != nil {
		t.Fatalf("IsMemoryWired failed: %v", err)
	}
	if !wired || count < 1 {
		t.Fatalf("expected wired=true count>=1, got wired=%v count=%d", wired, count)
	}

	// Step 3: Automated teardown unwiring upon model release
	teardown()

	// Idempotency check: calling teardown twice must not panic or error
	teardown()

	// Step 4: Verify unwired
	wired, count, err = IsMemoryWired(addr)
	if err != nil {
		t.Fatalf("IsMemoryWired after teardown failed: %v", err)
	}
	if wired || count != 0 {
		t.Fatalf("expected memory unwired after model release teardown, got wired=%v count=%d", wired, count)
	}
}
