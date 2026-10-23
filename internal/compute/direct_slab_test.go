package compute

import (
	"bytes"
	"errors"
	"math/rand"
	"sync"
	"testing"
	"unsafe"
)

// This owner models only the allocator/provider software contract. It performs
// no page locking, driver registration, device mapping or hardware transfer.
type testDirectSlabRegistration struct {
	value      directSlabRegistrationFacts
	live       bool
	releases   int
	releaseErr error
}

func (r *testDirectSlabRegistration) facts() (directSlabRegistrationFacts, bool) {
	return r.value, r.live
}

func (r *testDirectSlabRegistration) release() error {
	r.releases++
	if r.releaseErr == nil {
		r.live = false
	}
	return r.releaseErr
}

func bindTestDirectSlabRegistration(t *testing.T, slab *DirectSlabAllocator) *testDirectSlabRegistration {
	t.Helper()
	if slab.IsZeroCopy() {
		t.Fatal("flags-only slab claims physical registration")
	}
	probe, err := slab.Allocate(1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := probe.GetSGE(); err == nil {
		t.Fatal("flags-only slab returned physical SGE")
	}
	if _, err := RegisterDirectSlabRegion(slab, probe); err == nil {
		t.Fatal("flags-only slab returned registered region")
	}
	if err := probe.Free(); err != nil {
		t.Fatal(err)
	}
	r := &testDirectSlabRegistration{
		value: directSlabRegistrationFacts{
			allocator: slab, domain: "test-only-model", base: slab.BaseAddress(),
			length: uint64(slab.TotalBytes()), lkey: 0x1000, deviceAccessible: true,
		},
		live: true,
	}
	if err := slab.bindRegistration(r); err != nil {
		t.Fatal(err)
	}
	return r
}

// fak-test:runtime fast est=1s
func TestDirectSlabRegistrationEvidence(t *testing.T) {
	newSlab := func(t *testing.T) *DirectSlabAllocator {
		t.Helper()
		d, err := NewDirectSlabAllocator(DirectSlabConfig{TotalBytes: 1024})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = d.Close() })
		return d
	}
	for _, env := range []string{"", "DS4_TP_BIG_DIRECT", "FAK_DIRECT_SLAB"} {
		t.Run("heap_"+env, func(t *testing.T) {
			t.Setenv("DS4_TP_BIG_DIRECT", "")
			t.Setenv("FAK_DIRECT_SLAB", "")
			if env != "" {
				t.Setenv(env, "1")
			}
			d := newSlab(t)
			a, err := d.Allocate(64)
			if err != nil {
				t.Fatal(err)
			}
			defer a.Free()
			if d.IsZeroCopy() || a.Direct || a.Registered {
				t.Fatal("heap flags became registration evidence")
			}
			if _, err := a.GetSGE(); err == nil {
				t.Fatal("heap returned physical SGE")
			}
		})
	}
	for _, kind := range []string{"forged", "allocator", "domain", "base", "length", "retired"} {
		t.Run(kind, func(t *testing.T) {
			d := newSlab(t)
			r := &testDirectSlabRegistration{value: directSlabRegistrationFacts{
				allocator: d, domain: "test-only-model", base: d.BaseAddress(), length: 1024,
			}, live: true}
			switch kind {
			case "forged":
				r.value = directSlabRegistrationFacts{lkey: 0x1000}
			case "allocator":
				r.value.allocator = newSlab(t)
			case "domain":
				r.value.domain = ""
			case "base":
				r.value.base++
			case "length":
				r.value.length++
			case "retired":
				r.live = false
			}
			if err := d.bindRegistration(r); err == nil || d.IsZeroCopy() {
				t.Fatal("invalid owner receipt accepted")
			}
			if r.releases != 0 {
				t.Fatal("refused receipt ownership was stolen")
			}
		})
	}
	t.Run("provider_lifetime", func(t *testing.T) {
		d := newSlab(t)
		r := bindTestDirectSlabRegistration(t, d)
		a, err := d.Allocate(64)
		if err != nil {
			t.Fatal(err)
		}
		sge, err := a.GetSGE()
		if err != nil || sge.Address != d.BaseAddress() || sge.Length != 64 || sge.LKey != r.value.lkey {
			t.Fatalf("software provider receipt not applied: sge=%+v err=%v", sge, err)
		}
		if !d.IsZeroCopy() || !a.Direct || !a.Registered {
			t.Fatal("provider contract eligibility lost")
		}
		if err := d.Close(); err == nil || r.releases != 0 {
			t.Fatal("live allocation did not retain registration")
		}
		if err := d.Reset(); err == nil || r.releases != 0 {
			t.Fatal("reset retired a live provider user")
		}
		copied := *a
		if _, err := copied.GetSGE(); err == nil {
			t.Fatal("copied allocation acquired another allocation's receipt")
		}
		original := r.value
		for _, corrupt := range []string{"offset", "size", "domain", "base", "length", "lkey", "retired"} {
			a.Offset, a.Size = 0, 64
			r.value, r.live = original, true
			switch corrupt {
			case "offset":
				a.Offset = 1024
			case "size":
				a.Size = -1
			case "domain":
				r.value.domain = "different-domain"
			case "base":
				r.value.base++
			case "length":
				r.value.length--
			case "lkey":
				r.value.lkey++
			case "retired":
				r.live = false
			}
			if _, err := a.GetSGE(); err == nil {
				t.Fatalf("accepted %s mutation", corrupt)
			}
		}
		a.Offset, a.Size = 0, 64
		r.value, r.live = original, true
		if err := a.Free(); err != nil {
			t.Fatal(err)
		}
		reused, err := d.Allocate(64)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := a.GetSGE(); err == nil {
			t.Fatal("freed allocation accessed reused region")
		}
		if err := reused.Free(); err != nil {
			t.Fatal(err)
		}
		if err := d.Close(); err != nil {
			t.Fatal(err)
		}
		if err := d.Close(); err != nil || r.releases != 1 || d.IsZeroCopy() {
			t.Fatal("close did not retire registration exactly once")
		}
		if _, err := a.GetSGE(); err == nil {
			t.Fatal("closed allocator returned SGE")
		}
		if _, err := d.Allocate(64); err == nil {
			t.Fatal("closed allocator accepted allocation")
		}
		if err := d.Reset(); err == nil {
			t.Fatal("closed allocator accepted reset")
		}
	})
	t.Run("registered_not_device_mapped", func(t *testing.T) {
		d := newSlab(t)
		r := &testDirectSlabRegistration{value: directSlabRegistrationFacts{
			allocator: d, domain: "test-only-transport", base: d.BaseAddress(), length: 1024,
		}, live: true}
		if err := d.bindRegistration(r); err != nil {
			t.Fatal(err)
		}
		a, err := d.Allocate(64)
		if err != nil {
			t.Fatal(err)
		}
		defer a.Free()
		if _, err := a.GetSGE(); err != nil || !a.Registered || a.Direct || d.IsZeroCopy() {
			t.Fatal("transport registration was conflated with device mapping")
		}
	})
	t.Run("release_failure", func(t *testing.T) {
		d := newSlab(t)
		r := bindTestDirectSlabRegistration(t, d)
		r.releaseErr = errors.New("test-only release failure")
		for attempt := 0; attempt < 2; attempt++ {
			if err := d.Close(); err != r.releaseErr {
				t.Fatalf("release failure lost: %v", err)
			}
		}
		if r.releases != 1 || d.owner == nil || d.rawBuffer == nil || d.IsZeroCopy() {
			t.Fatal("failed release discarded backing or repeated owner release")
		}
	})
}

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

// fak-test:runtime fast est=1s
func TestDirectSlabAllocator_ZeroCopyVerification(t *testing.T) {
	allocator, err := NewDirectSlabAllocator(DirectSlabConfig{
		TotalBytes: 4096, PinMemory: true, DirectInterconnect: true,
		DeviceAccessible: true, LKey: 0x4200,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer allocator.Close()
	alloc, err := allocator.Allocate(4096)
	if err != nil {
		t.Fatal(err)
	}
	defer alloc.Free()
	pattern := bytes.Repeat([]byte{0xDE, 0xAD, 0xBE, 0xEF}, 1024)
	copy(alloc.Data, pattern)
	if !bytes.Equal(alloc.Data, pattern) || allocator.StagingCopyCount() != 0 {
		t.Fatal("heap layout or copy accounting changed")
	}
	if allocator.IsZeroCopy() || alloc.Direct || alloc.Registered {
		t.Fatal("configuration fabricated physical registration")
	}
	if _, err := alloc.GetSGE(); err == nil {
		t.Fatal("unregistered heap returned a physical SGE")
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

				// Heap allocation remains usable without physical registration.
				if _, err := allocator.GetSGE(alloc); err == nil {
					t.Errorf("worker %d: heap returned a physical SGE", id)
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
