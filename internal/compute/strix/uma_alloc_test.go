package strix

import (
	"errors"
	"fmt"
	"sync"
	"testing"
)

func TestUMAZeroCopyPointerIdentity(t *testing.T) {
	t.Run("Enforces2MBHugepageAlignmentAndPointerIdentity", func(t *testing.T) {
		alloc := NewUMAZeroCopyAllocator(nil)

		testSizes := []int64{
			64,
			4096,
			1024 * 1024,     // 1MB
			2 * 1024 * 1024, // 2MB
			5 * 1024 * 1024, // 5MB
			16 * 1024 * 1024,
		}

		for _, size := range testSizes {
			t.Run(fmt.Sprintf("Size_%d", size), func(t *testing.T) {
				buf, err := alloc.Allocate(size)
				if err != nil {
					t.Fatalf("Allocate(%d) failed: %v", size, err)
				}
				defer func() {
					if err := alloc.Free(buf); err != nil {
						t.Errorf("Free failed: %v", err)
					}
				}()

				hostPtr := buf.HostPtr()
				devPtr := buf.DevPtr()

				if hostPtr == 0 {
					t.Fatalf("HostPtr is zero")
				}
				if devPtr == 0 {
					t.Fatalf("DevPtr is zero")
				}
				if hostPtr != devPtr {
					t.Fatalf("Pointer identity violation: HostPtr 0x%x != DevPtr 0x%x", hostPtr, devPtr)
				}
				if hostPtr%uintptr(HugepageSize2MB) != 0 {
					t.Fatalf("HostPtr 0x%x is not aligned to 2MB (remainder %d)", hostPtr, hostPtr%uintptr(HugepageSize2MB))
				}
				if buf.AlignedSize()%HugepageSize2MB != 0 {
					t.Fatalf("AlignedSize %d is not multiple of 2MB", buf.AlignedSize())
				}
				if !buf.IsZeroCopy() {
					t.Fatalf("IsZeroCopy() returned false for valid buffer")
				}
				if !alloc.AssertPointerIdentity(buf) {
					t.Fatalf("AssertPointerIdentity returned false")
				}
				if buf.Len() != int(size) {
					t.Fatalf("expected Len()=%d, got %d", size, buf.Len())
				}
			})
		}
	})

	t.Run("ZeroCopyHostDeviceObservability", func(t *testing.T) {
		alloc := NewUMAZeroCopyAllocator(nil)
		const bufSize = 2 * 1024 * 1024

		buf, err := alloc.Allocate(bufSize)
		if err != nil {
			t.Fatalf("Allocate failed: %v", err)
		}
		defer func() { _ = alloc.Free(buf) }()

		hostSlice := buf.Slice()
		devPtr := buf.DevPtr()
		hostPtr := buf.HostPtr()
		if hostPtr == 0 || devPtr == 0 || hostPtr != devPtr {
			t.Fatalf("pointer identity mismatch: HostPtr 0x%x, DevPtr 0x%x", hostPtr, devPtr)
		}

		// Write magic pattern via host slice
		for i := 0; i < 1024; i++ {
			hostSlice[i] = byte(i % 256)
		}

		// Read directly from device pointer via unsafe pointer dereference
		uptr := buf.UnsafePointer()
		if uptr == nil {
			t.Fatalf("UnsafePointer is nil")
		}
		devBytes := (*[1024]byte)(uptr)
		for i := 0; i < 1024; i++ {
			if devBytes[i] != byte(i%256) {
				t.Fatalf("byte at offset %d mismatch: slice=%d, devPtr=%d", i, hostSlice[i], devBytes[i])
			}
		}

		// Mutate directly via device pointer
		devBytes[42] = 0xAA
		devBytes[99] = 0xBB

		// Verify instant host visibility with zero memory copies
		if hostSlice[42] != 0xAA {
			t.Fatalf("hostSlice[42] expected 0xAA, got 0x%x", hostSlice[42])
		}
		if hostSlice[99] != 0xBB {
			t.Fatalf("hostSlice[99] expected 0xBB, got 0x%x", hostSlice[99])
		}
	})

	t.Run("NativeDRMKFDIoctlHookPath", func(t *testing.T) {
		alloc := NewUMAZeroCopyAllocator(nil)

		var gemCreated bool
		var gemMmapped bool
		var kfdAllocated bool

		alloc.SetMockIoctlHook(func(op string, in interface{}, out interface{}) error {
			switch op {
			case "DRM_IOCTL_AMDGPU_GEM_CREATE":
				createIn, ok := in.(DRMAMDGPUGEMCreateIn)
				if !ok {
					return fmt.Errorf("unexpected in type %T", in)
				}
				if createIn.Domains != AMDGPU_GEM_DOMAIN_GTT {
					return fmt.Errorf("expected GTT domain, got %d", createIn.Domains)
				}
				if (createIn.DomainFlags & AMDGPU_GEM_CREATE_CPU_GTT_USWC) == 0 {
					return fmt.Errorf("expected USWC flag in domain_flags: 0x%x", createIn.DomainFlags)
				}
				createOut, ok := out.(*DRMAMDGPUGEMCreateOut)
				if !ok {
					return fmt.Errorf("unexpected out type %T", out)
				}
				createOut.Handle = 42
				gemCreated = true
				return nil

			case "DRM_IOCTL_AMDGPU_GEM_MMAP":
				mmapIn, ok := in.(DRMAMDGPUGEMMmapIn)
				if !ok || mmapIn.Handle != 42 {
					return fmt.Errorf("unexpected mmap handle %d", mmapIn.Handle)
				}
				mmapOut, ok := out.(*DRMAMDGPUGEMMmapOut)
				if !ok {
					return fmt.Errorf("unexpected out type %T", out)
				}
				mmapOut.AddrPtr = 0x7f0020000000
				gemMmapped = true
				return nil

			case "AMDKFD_IOCTL_ALLOC_MEM_OF_GPU":
				kfdAllocIn, ok := in.(KFDAllocMemArgs)
				if !ok {
					return fmt.Errorf("unexpected in type %T", in)
				}
				if kfdAllocIn.Flags&KFD_IOC_ALLOC_MEM_FLAGS_GTT == 0 {
					return fmt.Errorf("expected KFD GTT flag: 0x%x", kfdAllocIn.Flags)
				}
				kfdAllocated = true
				return nil

			default:
				return fmt.Errorf("unexpected op: %s", op)
			}
		})
		defer alloc.ResetMockIoctlHook()

		buf, err := alloc.Allocate(4 * 1024 * 1024)
		if err != nil {
			t.Fatalf("native Allocate failed: %v", err)
		}
		defer func() { _ = alloc.Free(buf) }()

		if !gemCreated || !gemMmapped || !kfdAllocated {
			t.Fatalf("expected all ioctls called: gemCreated=%v, gemMmapped=%v, kfdAllocated=%v",
				gemCreated, gemMmapped, kfdAllocated)
		}

		if buf.Mode() != UMAMappingModeNativeDRMKFD {
			t.Fatalf("expected mode %s, got %s", UMAMappingModeNativeDRMKFD, buf.Mode())
		}
		if buf.GEMHandle() != 42 {
			t.Fatalf("expected GEM handle 42, got %d", buf.GEMHandle())
		}
		if !buf.IsZeroCopy() {
			t.Fatalf("expected IsZeroCopy() to be true")
		}
		if buf.HostPtr() != buf.DevPtr() {
			t.Fatalf("HostPtr 0x%x != DevPtr 0x%x", buf.HostPtr(), buf.DevPtr())
		}
	})

	t.Run("FallbackBehaviorWhenDRMUnavailable", func(t *testing.T) {
		cfg := DefaultUMAZeroCopyConfig()
		cfg.ForceSimulated = true
		alloc := NewUMAZeroCopyAllocator(cfg)

		buf, err := alloc.Allocate(2 * 1024 * 1024)
		if err != nil {
			t.Fatalf("Allocate failed under simulated fallback: %v", err)
		}
		defer func() { _ = alloc.Free(buf) }()

		if buf.Mode() != UMAMappingModeSimulatedPinned {
			t.Fatalf("expected mode %s, got %s", UMAMappingModeSimulatedPinned, buf.Mode())
		}
		if !buf.IsZeroCopy() {
			t.Fatalf("expected IsZeroCopy() true under simulation")
		}
		if buf.HostPtr()%uintptr(HugepageSize2MB) != 0 {
			t.Fatalf("address not 2MB aligned: 0x%x", buf.HostPtr())
		}
		if buf.HostPtr() != buf.DevPtr() {
			t.Fatalf("pointer identity violated under simulation: HostPtr 0x%x != DevPtr 0x%x",
				buf.HostPtr(), buf.DevPtr())
		}
	})

	t.Run("LifecycleAndErrorBounds", func(t *testing.T) {
		alloc := NewUMAZeroCopyAllocator(nil)

		// Invalid sizes
		if _, err := alloc.Allocate(0); !errors.Is(err, ErrInvalidBufferSize) {
			t.Errorf("expected ErrInvalidBufferSize on size 0, got %v", err)
		}
		if _, err := alloc.Allocate(-1024); !errors.Is(err, ErrInvalidBufferSize) {
			t.Errorf("expected ErrInvalidBufferSize on negative size, got %v", err)
		}
		if _, err := alloc.Allocate(MaxGTTAllocPoolBytes + 1); !errors.Is(err, ErrInvalidBufferSize) {
			t.Errorf("expected ErrInvalidBufferSize on exceeding max pool, got %v", err)
		}

		// Free nil buffer
		if err := alloc.Free(nil); !errors.Is(err, ErrBufferClosed) {
			t.Errorf("expected ErrBufferClosed when freeing nil, got %v", err)
		}

		// Double free
		buf, err := alloc.Allocate(1024)
		if err != nil {
			t.Fatalf("Allocate failed: %v", err)
		}
		if err := alloc.Free(buf); err != nil {
			t.Fatalf("Free failed: %v", err)
		}
		if err := alloc.Free(buf); !errors.Is(err, ErrBufferClosed) {
			t.Errorf("expected ErrBufferClosed on double free, got %v", err)
		}
		if buf.HostPtr() != 0 || buf.DevPtr() != 0 {
			t.Errorf("expected HostPtr and DevPtr to be 0 after close, got 0x%x, 0x%x",
				buf.HostPtr(), buf.DevPtr())
		}
		if buf.Slice() != nil {
			t.Errorf("expected Slice() to be nil after close")
		}
	})

	t.Run("ConcurrentAllocationAndRaceSafety", func(t *testing.T) {
		alloc := NewUMAZeroCopyAllocator(nil)
		const workers = 8
		const iterationsPerWorker = 50

		var wg sync.WaitGroup
		wg.Add(workers)

		for w := 0; w < workers; w++ {
			go func(workerID int) {
				defer wg.Done()
				for i := 0; i < iterationsPerWorker; i++ {
					size := int64((workerID+1)*64*1024 + i*4096)
					buf, err := alloc.Allocate(size)
					if err != nil {
						t.Errorf("worker %d iteration %d: Allocate failed: %v", workerID, i, err)
						return
					}

					if !alloc.AssertPointerIdentity(buf) {
						t.Errorf("worker %d iteration %d: AssertPointerIdentity failed", workerID, i)
					}

					slice := buf.Slice()
					if len(slice) > 0 {
						slice[0] = byte(workerID)
						slice[len(slice)-1] = byte(i)
					}

					if err := alloc.Free(buf); err != nil {
						t.Errorf("worker %d iteration %d: Free failed: %v", workerID, i, err)
					}
				}
			}(w)
		}

		wg.Wait()

		telem := alloc.Telemetry()
		if telem.ActiveAllocations != 0 {
			t.Errorf("expected 0 active allocations, got %d", telem.ActiveAllocations)
		}
		if telem.TotalAllocatedBytes <= 0 {
			t.Errorf("expected positive TotalAllocatedBytes, got %d", telem.TotalAllocatedBytes)
		}
		if !telem.ZeroCopyVerified {
			t.Errorf("expected ZeroCopyVerified to be true")
		}
	})

	t.Run("AllocationFlagsInspection", func(t *testing.T) {
		flags := UMAFlagUSWC | UMAFlag2MBHugepages | UMAFlagCoherent
		if !flags.Has(UMAFlagUSWC) {
			t.Errorf("expected Has(UMAFlagUSWC) true")
		}
		if !flags.Has(UMAFlag2MBHugepages) {
			t.Errorf("expected Has(UMAFlag2MBHugepages) true")
		}
		if flags.Has(UMAFlagNoEvict) {
			t.Errorf("expected Has(UMAFlagNoEvict) false")
		}
		str := flags.String()
		if str == "" || str == "NONE" {
			t.Errorf("unexpected flags string representation: %s", str)
		}
	})
}
