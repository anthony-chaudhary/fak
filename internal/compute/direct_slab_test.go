package compute

import (
	"bytes"
	"math/rand"
	"os"
	"sync"
	"testing"
	"unsafe"
)

func TestDirectSlabAllocator_Allocation(t *testing.T) {
	cfg := DirectSlabConfig{
		TotalBytes:         64 * 1024,
		Alignment:          64,
		PinMemory:          true,
		DirectInterconnect: true,
		DeviceAccessible:   true,
	}

	allocator, err := NewDirectSlabAllocator(cfg)
	if err != nil {
		t.Fatalf("failed to create DirectSlabAllocator: %v", err)
	}
	defer allocator.Close()

	if allocator.TotalBytes() != 64*1024 {
		t.Fatalf("expected TotalBytes 65536, got %d", allocator.TotalBytes())
	}
	if allocator.Available() != 64*1024 {
		t.Fatalf("expected Available 65536, got %d", allocator.Available())
	}
	if allocator.Used() != 0 {
		t.Fatalf("expected Used 0, got %d", allocator.Used())
	}

	// Allocate multiple blocks of different sizes.
	sizes := []int64{100, 256, 1024, 4096}
	allocs := make([]*SlabAllocation, len(sizes))

	for i, sz := range sizes {
		alloc, err := allocator.Allocate(sz)
		if err != nil {
			t.Fatalf("allocation %d (size %d) failed: %v", i, sz, err)
		}
		if alloc.Size != sz {
			t.Fatalf("allocation %d expected Size %d, got %d", i, sz, alloc.Size)
		}
		if int64(len(alloc.Data)) != sz {
			t.Fatalf("allocation %d expected len(Data) %d, got %d", i, sz, len(alloc.Data))
		}

		addr := uintptr(unsafe.Pointer(&alloc.Data[0]))
		if addr%64 != 0 {
			t.Fatalf("allocation %d addr 0x%x not 64-byte aligned", i, addr)
		}

		// Write distinct byte pattern.
		pattern := byte(i + 1)
		for j := range alloc.Data {
			alloc.Data[j] = pattern
		}
		allocs[i] = alloc
	}

	// Verify all subslice data patterns remain intact (no memory overlap).
	for i, alloc := range allocs {
		pattern := byte(i + 1)
		for j, b := range alloc.Data {
			if b != pattern {
				t.Fatalf("allocation %d corrupted at byte %d: expected 0x%x, got 0x%x", i, j, pattern, b)
			}
		}
	}

	// Free the second allocation (size 256).
	if err := allocator.Free(allocs[1]); err != nil {
		t.Fatalf("failed to free allocation 1: %v", err)
	}

	// Double free should fail.
	if err := allocator.Free(allocs[1]); err == nil {
		t.Fatalf("expected error on double free, got nil")
	}

	// Allocate a new block of 200 bytes, which should reuse the freed space.
	newAlloc, err := allocator.Allocate(200)
	if err != nil {
		t.Fatalf("allocation after free failed: %v", err)
	}
	if newAlloc.Size != 200 {
		t.Fatalf("expected newAlloc Size 200, got %d", newAlloc.Size)
	}

	// Test Release() func callback.
	newAlloc.Release()
	if err := allocator.Free(newAlloc); err == nil {
		t.Fatalf("expected error freeing already released allocation, got nil")
	}

	// Test Reset().
	allocator.Reset()
	if allocator.Used() != 0 {
		t.Fatalf("expected Used 0 after Reset, got %d", allocator.Used())
	}
	if allocator.Available() != 64*1024 {
		t.Fatalf("expected Available 65536 after Reset, got %d", allocator.Available())
	}
}

func TestDirectSlabAllocator_ZeroCopyVerification(t *testing.T) {
	cfg := DirectSlabConfig{
		TotalBytes:         128 * 1024,
		Alignment:          64,
		PinMemory:          true,
		DirectInterconnect: true,
		DeviceAccessible:   true,
		LKey:               0x4200,
	}

	allocator, err := NewDirectSlabAllocator(cfg)
	if err != nil {
		t.Fatalf("failed to create allocator: %v", err)
	}
	defer allocator.Close()

	if !allocator.IsZeroCopy() {
		t.Fatalf("expected IsZeroCopy to be true")
	}
	if allocator.StagingCopyCount() != 0 {
		t.Fatalf("initial StagingCopyCount must be 0, got %d", allocator.StagingCopyCount())
	}

	// Allocate tensor buffer for prefill / collective transfer.
	alloc, err := allocator.Allocate(4096)
	if err != nil {
		t.Fatalf("Allocate failed: %v", err)
	}
	defer alloc.Free()

	if !alloc.Direct {
		t.Fatalf("expected alloc.Direct to be true")
	}
	if !alloc.Registered {
		t.Fatalf("expected alloc.Registered to be true")
	}

	// Populate tensor data directly in UMA device-accessible memory.
	testPattern := bytes.Repeat([]byte{0xDE, 0xAD, 0xBE, 0xEF}, 1024)
	copy(alloc.Data, testPattern)

	// Retrieve verbs SGE.
	sge, err := allocator.GetSGE(alloc)
	if err != nil {
		t.Fatalf("GetSGE failed: %v", err)
	}

	expectedAddr := uintptr(unsafe.Pointer(&alloc.Data[0]))
	if sge.Address != expectedAddr {
		t.Fatalf("SGE Address mismatch: expected 0x%x, got 0x%x", expectedAddr, sge.Address)
	}
	if sge.Length != 4096 {
		t.Fatalf("SGE Length mismatch: expected 4096, got %d", sge.Length)
	}
	if sge.LKey != 0x4200 {
		t.Fatalf("SGE LKey mismatch: expected 0x4200, got 0x%x", sge.LKey)
	}

	// SGE through allocation method should yield identical result.
	sgeMethod, err := alloc.GetSGE()
	if err != nil {
		t.Fatalf("alloc.GetSGE failed: %v", err)
	}
	if sgeMethod != sge {
		t.Fatalf("alloc.GetSGE() != allocator.GetSGE(alloc)")
	}

	// Crucial DS4 borrow invariant: staging copy count must be exactly 0.
	if count := allocator.StagingCopyCount(); count != 0 {
		t.Fatalf("DS4 zero-copy invariant violated: StagingCopyCount expected 0, got %d", count)
	}

	// Verify environment variable support DS4_TP_BIG_DIRECT=1.
	os.Setenv("DS4_TP_BIG_DIRECT", "1")
	defer os.Unsetenv("DS4_TP_BIG_DIRECT")

	envAlloc, err := NewDirectSlabAllocator(DirectSlabConfig{
		TotalBytes: 4096,
		Alignment:  64,
	})
	if err != nil {
		t.Fatalf("failed to create env-configured allocator: %v", err)
	}
	defer envAlloc.Close()

	if !envAlloc.IsZeroCopy() {
		t.Fatalf("expected IsZeroCopy to be true under DS4_TP_BIG_DIRECT=1")
	}
}

func TestDirectSlabAllocator_Exhaustion(t *testing.T) {
	cfg := DirectSlabConfig{
		TotalBytes: 1024,
		Alignment:  64,
	}

	allocator, err := NewDirectSlabAllocator(cfg)
	if err != nil {
		t.Fatalf("failed to create allocator: %v", err)
	}
	defer allocator.Close()

	// Invalid allocation requests.
	if _, err := allocator.Allocate(0); err == nil {
		t.Fatalf("expected error on Allocate(0), got nil")
	}
	if _, err := allocator.Allocate(-10); err == nil {
		t.Fatalf("expected error on Allocate(-10), got nil")
	}

	// Allocate two 512-byte blocks.
	a1, err := allocator.Allocate(512)
	if err != nil {
		t.Fatalf("a1 allocate failed: %v", err)
	}
	a2, err := allocator.Allocate(512)
	if err != nil {
		t.Fatalf("a2 allocate failed: %v", err)
	}

	if allocator.Available() != 0 {
		t.Fatalf("expected Available 0, got %d", allocator.Available())
	}
	if allocator.Used() != 1024 {
		t.Fatalf("expected Used 1024, got %d", allocator.Used())
	}

	// Out of memory allocation should fail with clear error.
	_, err = allocator.Allocate(1)
	if err == nil {
		t.Fatalf("expected allocation error on full slab, got nil")
	}

	// Free a1 and allocate smaller chunk.
	if err := allocator.Free(a1); err != nil {
		t.Fatalf("free a1 failed: %v", err)
	}
	if allocator.Available() != 512 {
		t.Fatalf("expected Available 512 after free, got %d", allocator.Available())
	}

	a3, err := allocator.Allocate(256)
	if err != nil {
		t.Fatalf("a3 allocate failed: %v", err)
	}

	// Cannot allocate 512 bytes now because only 256 remains.
	if _, err := allocator.Allocate(512); err == nil {
		t.Fatalf("expected allocation failure when requesting more than available, got nil")
	}

	// Free remainder.
	if err := allocator.Free(a2); err != nil {
		t.Fatalf("free a2 failed: %v", err)
	}
	if err := allocator.Free(a3); err != nil {
		t.Fatalf("free a3 failed: %v", err)
	}

	if allocator.Used() != 0 {
		t.Fatalf("expected Used 0 after all frees, got %d", allocator.Used())
	}
	if allocator.Available() != 1024 {
		t.Fatalf("expected Available 1024 after all frees, got %d", allocator.Available())
	}

	// Allocate entire slab.
	fullAlloc, err := allocator.Allocate(1024)
	if err != nil {
		t.Fatalf("allocating entire slab failed: %v", err)
	}
	fullAlloc.Release()
}

func TestDirectSlabAllocator_Concurrent(t *testing.T) {
	cfg := DirectSlabConfig{
		TotalBytes:         1024 * 1024,
		Alignment:          64,
		PinMemory:          true,
		DirectInterconnect: true,
		DeviceAccessible:   true,
	}

	allocator, err := NewDirectSlabAllocator(cfg)
	if err != nil {
		t.Fatalf("failed to create allocator: %v", err)
	}
	defer allocator.Close()

	const numWorkers = 16
	const iterations = 50

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for workerID := 0; workerID < numWorkers; workerID++ {
		go func(id int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(id * 1000)))

			for i := 0; i < iterations; i++ {
				// Allocate between 64 and 2048 bytes.
				sz := int64(64 + rng.Intn(1984))
				alloc, err := allocator.Allocate(sz)
				if err != nil {
					// In case of high concurrency contention / fragmentation, continue.
					continue
				}

				// Check alignment.
				addr := uintptr(unsafe.Pointer(&alloc.Data[0]))
				if addr%64 != 0 {
					t.Errorf("worker %d: addr 0x%x not aligned to 64", id, addr)
				}

				// Write and verify.
				pattern := byte(id)
				for j := range alloc.Data {
					alloc.Data[j] = pattern
				}
				for j, b := range alloc.Data {
					if b != pattern {
						t.Errorf("worker %d: corrupted byte at %d", id, j)
						break
					}
				}

				// Check SGE.
				sge, err := allocator.GetSGE(alloc)
				if err != nil {
					t.Errorf("worker %d: GetSGE failed: %v", id, err)
				} else if sge.Length != uint32(sz) {
					t.Errorf("worker %d: SGE length %d != sz %d", id, sge.Length, sz)
				}

				if err := allocator.Free(alloc); err != nil {
					t.Errorf("worker %d: Free failed: %v", id, err)
				}
			}
		}(workerID)
	}

	wg.Wait()

	if allocator.Used() != 0 {
		t.Fatalf("expected Used 0 after concurrent test, got %d", allocator.Used())
	}
	if allocator.Available() != 1024*1024 {
		t.Fatalf("expected Available 1MB after concurrent test, got %d", allocator.Available())
	}
	if allocator.StagingCopyCount() != 0 {
		t.Fatalf("expected StagingCopyCount 0, got %d", allocator.StagingCopyCount())
	}
}

func TestDirectSlabAllocator_Alignment(t *testing.T) {
	// Test 64-byte alignment with irregular sizes.
	t.Run("Alignment64", func(t *testing.T) {
		allocator, err := NewDirectSlabAllocator(DirectSlabConfig{
			TotalBytes: 32 * 1024,
			Alignment:  64,
		})
		if err != nil {
			t.Fatalf("failed to create allocator: %v", err)
		}
		defer allocator.Close()

		irregularSizes := []int64{1, 3, 7, 13, 17, 33, 65, 127, 255}
		allocs := make([]*SlabAllocation, len(irregularSizes))

		for i, sz := range irregularSizes {
			alloc, err := allocator.Allocate(sz)
			if err != nil {
				t.Fatalf("size %d allocate failed: %v", sz, err)
			}
			addr := uintptr(unsafe.Pointer(&alloc.Data[0]))
			if addr%64 != 0 {
				t.Fatalf("size %d addr 0x%x is not 64-byte aligned", sz, addr)
			}
			if alloc.Offset%64 != 0 {
				t.Fatalf("size %d offset %d is not a multiple of 64", sz, alloc.Offset)
			}
			allocs[i] = alloc
		}

		for _, alloc := range allocs {
			_ = allocator.Free(alloc)
		}
	})

	// Test 4096-byte alignment (UMA/RDMA page alignment).
	t.Run("Alignment4096", func(t *testing.T) {
		allocator, err := NewDirectSlabAllocator(DirectSlabConfig{
			TotalBytes: 64 * 1024,
			Alignment:  4096,
		})
		if err != nil {
			t.Fatalf("failed to create allocator: %v", err)
		}
		defer allocator.Close()

		pageSizes := []int64{100, 4096, 5000, 8192}
		allocs := make([]*SlabAllocation, len(pageSizes))

		for i, sz := range pageSizes {
			alloc, err := allocator.Allocate(sz)
			if err != nil {
				t.Fatalf("size %d allocate failed: %v", sz, err)
			}
			addr := uintptr(unsafe.Pointer(&alloc.Data[0]))
			if addr%4096 != 0 {
				t.Fatalf("size %d addr 0x%x is not 4096-byte aligned", sz, addr)
			}
			if alloc.Offset%4096 != 0 {
				t.Fatalf("size %d offset %d is not a multiple of 4096", sz, alloc.Offset)
			}
			allocs[i] = alloc
		}

		for _, alloc := range allocs {
			_ = allocator.Free(alloc)
		}
	})
}
