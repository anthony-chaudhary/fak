package strix

import (
	"errors"
	"math/rand"
	"sync"
	"testing"
	"unsafe"
)

func TestPointerIdentityUMA(t *testing.T) {
	t.Run("RandomizedAllocations10k", func(t *testing.T) {
		mgr := NewUMAPointerManager()
		rng := rand.New(rand.NewSource(42))

		alignments := []int64{64, 128, 256, 512, 1024, 2048, 4096}
		const totalIterations = 10000

		for i := 0; i < totalIterations; i++ {
			size := int64(rng.Intn(4096) + 1)
			align := alignments[rng.Intn(len(alignments))]

			buf, err := mgr.AllocateUMABuffer(size, align)
			if err != nil {
				t.Fatalf("iteration %d: AllocateUMABuffer(%d, %d) failed: %v", i, size, align, err)
			}

			hostPtr := buf.HostPtr()
			devPtr := buf.DevPtr()
			unsafePtr := buf.UnsafePointer()

			if hostPtr == 0 {
				t.Fatalf("iteration %d: HostPtr is 0", i)
			}
			if devPtr == 0 {
				t.Fatalf("iteration %d: DevPtr is 0", i)
			}
			if hostPtr != devPtr {
				t.Fatalf("iteration %d: pointer identity mismatch: HostPtr 0x%x != DevPtr 0x%x", i, hostPtr, devPtr)
			}
			if uintptr(unsafePtr) != devPtr {
				t.Fatalf("iteration %d: uintptr(UnsafePointer) 0x%x != DevPtr 0x%x", i, uintptr(unsafePtr), devPtr)
			}
			if !mgr.AssertPointerIdentity(unsafePtr, devPtr) {
				t.Fatalf("iteration %d: AssertPointerIdentity failed for HostPtr 0x%x, DevPtr 0x%x", i, hostPtr, devPtr)
			}
			if hostPtr%uintptr(align) != 0 {
				t.Fatalf("iteration %d: address 0x%x not aligned to %d (remainder %d)", i, hostPtr, align, hostPtr%uintptr(align))
			}
			if !buf.IsZeroCopy() {
				t.Fatalf("iteration %d: IsZeroCopy returned false for valid buffer", i)
			}
			if buf.Len() != int(size) {
				t.Fatalf("iteration %d: Len() = %d, expected %d", i, buf.Len(), size)
			}

			if err := mgr.FreeUMABuffer(buf); err != nil {
				t.Fatalf("iteration %d: FreeUMABuffer failed: %v", i, err)
			}
		}

		if active := mgr.ActiveAllocations(); active != 0 {
			t.Errorf("expected 0 active allocations after freeing all, got %d", active)
		}
		if passed := mgr.AssertionChecksPassed(); passed < totalIterations {
			t.Errorf("expected at least %d assertions passed, got %d", totalIterations, passed)
		}
		if totalAlloc := mgr.TotalAllocatedBytes(); totalAlloc <= 0 {
			t.Errorf("expected positive TotalAllocatedBytes, got %d", totalAlloc)
		}
	})

	t.Run("ZeroCopyConversions", func(t *testing.T) {
		mgr := NewUMAPointerManager()

		// Test BytesToDevPtr
		byteData := make([]byte, 1024)
		for i := range byteData {
			byteData[i] = byte(i & 0xFF)
		}

		expectedByteAddr := uintptr(unsafe.Pointer(&byteData[0]))
		devPtr, err := mgr.BytesToDevPtr(byteData)
		if err != nil {
			t.Fatalf("BytesToDevPtr failed: %v", err)
		}
		if devPtr != expectedByteAddr {
			t.Fatalf("BytesToDevPtr returned 0x%x, expected 0x%x", devPtr, expectedByteAddr)
		}
		if !mgr.AssertPointerIdentity(unsafe.Pointer(&byteData[0]), devPtr) {
			t.Fatalf("AssertPointerIdentity failed for BytesToDevPtr")
		}

		// Test Float16ToDevPtr
		f16Data := make([]uint16, 512)
		for i := range f16Data {
			f16Data[i] = uint16(i & 0xFFFF)
		}

		expectedF16Addr := uintptr(unsafe.Pointer(&f16Data[0]))
		f16DevPtr, err := mgr.Float16ToDevPtr(f16Data)
		if err != nil {
			t.Fatalf("Float16ToDevPtr failed: %v", err)
		}
		if f16DevPtr != expectedF16Addr {
			t.Fatalf("Float16ToDevPtr returned 0x%x, expected 0x%x", f16DevPtr, expectedF16Addr)
		}
		if !mgr.AssertPointerIdentity(unsafe.Pointer(&f16Data[0]), f16DevPtr) {
			t.Fatalf("AssertPointerIdentity failed for Float16ToDevPtr")
		}

		if conversions := mgr.ZeroCopyConversions(); conversions < 2 {
			t.Errorf("expected at least 2 zero copy conversions, got %d", conversions)
		}
	})

	t.Run("CPUWriteVisibilityWithoutCopying", func(t *testing.T) {
		buf, err := AllocateUMABuffer(1024, 64)
		if err != nil {
			t.Fatalf("AllocateUMABuffer failed: %v", err)
		}
		defer FreeUMABuffer(buf)

		devPtr := buf.DevPtr()
		if devPtr == 0 {
			t.Fatalf("DevPtr is 0")
		}

		byteView := DevPtrToBytes(devPtr, 1024)
		if byteView == nil {
			t.Fatalf("DevPtrToBytes returned nil")
		}
		f16View := DevPtrToFloat16(devPtr, 512)
		if f16View == nil {
			t.Fatalf("DevPtrToFloat16 returned nil")
		}

		// 1. Host write via buf.Slice()
		slice := buf.Slice()
		slice[0] = 0x34
		slice[1] = 0x12
		slice[2] = 0x78
		slice[3] = 0x56

		// Verify immediate visibility through byteView without copying
		if byteView[0] != 0x34 || byteView[1] != 0x12 {
			t.Errorf("byteView read mismatch: expected [0x34, 0x12], got [0x%x, 0x%x]", byteView[0], byteView[1])
		}
		// Verify immediate visibility through f16View (little-endian: 0x1234, 0x5678)
		if f16View[0] != 0x1234 {
			t.Errorf("f16View[0] expected 0x1234, got 0x%x", f16View[0])
		}
		if f16View[1] != 0x5678 {
			t.Errorf("f16View[1] expected 0x5678, got 0x%x", f16View[1])
		}

		// 2. Write via f16View
		f16View[2] = 0xABCD
		// Verify immediate visibility in slice and byteView
		if slice[4] != 0xCD || slice[5] != 0xAB {
			t.Errorf("slice after f16View mutation: expected [0xCD, 0xAB], got [0x%x, 0x%x]", slice[4], slice[5])
		}
		if byteView[4] != 0xCD || byteView[5] != 0xAB {
			t.Errorf("byteView after f16View mutation: expected [0xCD, 0xAB], got [0x%x, 0x%x]", byteView[4], byteView[5])
		}

		// 3. Write via byteView
		byteView[6] = 0xEF
		byteView[7] = 0xBE
		if slice[6] != 0xEF || slice[7] != 0xBE {
			t.Errorf("slice after byteView mutation: expected [0xEF, 0xBE], got [0x%x, 0x%x]", slice[6], slice[7])
		}
		if f16View[3] != 0xBEEF {
			t.Errorf("f16View[3] after byteView mutation: expected 0xBEEF, got 0x%x", f16View[3])
		}

		// 4. Verify Float16Slice method on buffer
		bufF16 := buf.Float16Slice()
		if bufF16 == nil {
			t.Fatalf("buf.Float16Slice returned nil")
		}
		if bufF16[0] != 0x1234 || bufF16[1] != 0x5678 || bufF16[2] != 0xABCD || bufF16[3] != 0xBEEF {
			t.Errorf("bufF16 content mismatch: got %x", bufF16[:4])
		}

		// Zero copies verified
		if bc := buf.BytesCopied(); bc != 0 {
			t.Errorf("expected 0 bytes copied, got %d", bc)
		}
	})

	t.Run("ConcurrencySafetyRace", func(t *testing.T) {
		mgr := NewUMAPointerManager()
		const numGoroutines = 16
		const opsPerGoroutine = 300

		var wg sync.WaitGroup
		wg.Add(numGoroutines)

		for g := 0; g < numGoroutines; g++ {
			go func(gid int) {
				defer wg.Done()
				rng := rand.New(rand.NewSource(int64(gid*1000 + 7)))

				for i := 0; i < opsPerGoroutine; i++ {
					size := int64(rng.Intn(1024) + 64)
					buf, err := mgr.AllocateUMABuffer(size, 64)
					if err != nil {
						t.Errorf("goroutine %d: allocate failed: %v", gid, err)
						return
					}

					hp := buf.HostPtr()
					dp := buf.DevPtr()
					if hp != dp || hp == 0 {
						t.Errorf("goroutine %d: pointer mismatch: hp=0x%x dp=0x%x", gid, hp, dp)
					}
					if !mgr.AssertPointerIdentity(buf.UnsafePointer(), dp) {
						t.Errorf("goroutine %d: AssertPointerIdentity failed", gid)
					}

					bv := mgr.DevPtrToBytes(dp, int(size))
					if bv != nil {
						bv[0] = byte(gid)
					}

					if err := mgr.FreeUMABuffer(buf); err != nil {
						t.Errorf("goroutine %d: free failed: %v", gid, err)
					}
				}
			}(g)
		}

		wg.Wait()

		if active := mgr.ActiveAllocations(); active != 0 {
			t.Errorf("expected 0 active allocations after concurrent test, got %d", active)
		}
		if passed := mgr.AssertionChecksPassed(); passed < int64(numGoroutines*opsPerGoroutine) {
			t.Errorf("expected at least %d assertions passed, got %d", numGoroutines*opsPerGoroutine, passed)
		}
	})

	t.Run("ErrorHandling", func(t *testing.T) {
		mgr := NewUMAPointerManager()

		// 1. Invalid buffer size
		if _, err := mgr.AllocateUMABuffer(0, 64); !errors.Is(err, ErrInvalidBufferSize) {
			t.Errorf("expected ErrInvalidBufferSize for size=0, got %v", err)
		}
		if _, err := mgr.AllocateUMABuffer(-10, 64); !errors.Is(err, ErrInvalidBufferSize) {
			t.Errorf("expected ErrInvalidBufferSize for size=-10, got %v", err)
		}

		// 2. Invalid alignment (not power of 2, or negative)
		if _, err := mgr.AllocateUMABuffer(128, 3); !errors.Is(err, ErrInvalidAlignment) {
			t.Errorf("expected ErrInvalidAlignment for align=3, got %v", err)
		}
		if _, err := mgr.AllocateUMABuffer(128, -64); !errors.Is(err, ErrInvalidAlignment) {
			t.Errorf("expected ErrInvalidAlignment for align=-64, got %v", err)
		}

		// 3. Nil and empty slice conversions
		if _, err := mgr.BytesToDevPtr(nil); !errors.Is(err, ErrEmptySlice) {
			t.Errorf("expected ErrEmptySlice for nil bytes, got %v", err)
		}
		if _, err := mgr.BytesToDevPtr([]byte{}); !errors.Is(err, ErrEmptySlice) {
			t.Errorf("expected ErrEmptySlice for empty bytes, got %v", err)
		}
		if _, err := mgr.Float16ToDevPtr(nil); !errors.Is(err, ErrEmptySlice) {
			t.Errorf("expected ErrEmptySlice for nil float16, got %v", err)
		}
		if _, err := mgr.Float16ToDevPtr([]uint16{}); !errors.Is(err, ErrEmptySlice) {
			t.Errorf("expected ErrEmptySlice for empty float16, got %v", err)
		}

		// 4. Invalid dev pointers
		if b := mgr.DevPtrToBytes(0, 100); b != nil {
			t.Errorf("expected nil byte slice for devPtr=0, got %v", b)
		}
		if b := mgr.DevPtrToBytes(0x1000, 0); b != nil {
			t.Errorf("expected nil byte slice for size=0, got %v", b)
		}
		if b := mgr.DevPtrToBytes(0x1000, -1); b != nil {
			t.Errorf("expected nil byte slice for negative size, got %v", b)
		}
		if f := mgr.DevPtrToFloat16(0, 100); f != nil {
			t.Errorf("expected nil float16 slice for devPtr=0, got %v", f)
		}
		if f := mgr.DevPtrToFloat16(0x1000, 0); f != nil {
			t.Errorf("expected nil float16 slice for size=0, got %v", f)
		}
		if f := mgr.DevPtrToFloat16(0x1000, -5); f != nil {
			t.Errorf("expected nil float16 slice for negative size, got %v", f)
		}
		if f := mgr.DevPtrToFloat16(0x1001, 10); f != nil {
			t.Errorf("expected nil float16 slice for misaligned devPtr (odd address), got %v", f)
		}

		// 5. Free error handling
		if err := mgr.FreeUMABuffer(nil); !errors.Is(err, ErrBufferClosed) {
			t.Errorf("expected ErrBufferClosed for nil buffer, got %v", err)
		}

		buf, err := mgr.AllocateUMABuffer(64, 64)
		if err != nil {
			t.Fatalf("allocate failed: %v", err)
		}
		if err := mgr.FreeUMABuffer(buf); err != nil {
			t.Fatalf("first free failed: %v", err)
		}
		if err := mgr.FreeUMABuffer(buf); !errors.Is(err, ErrBufferClosed) {
			t.Errorf("expected ErrBufferClosed for double free, got %v", err)
		}

		// 6. Closed buffer inspection
		if hp := buf.HostPtr(); hp != 0 {
			t.Errorf("expected HostPtr=0 on closed buffer, got 0x%x", hp)
		}
		if dp := buf.DevPtr(); dp != 0 {
			t.Errorf("expected DevPtr=0 on closed buffer, got 0x%x", dp)
		}
		if buf.IsZeroCopy() {
			t.Errorf("expected IsZeroCopy=false on closed buffer")
		}
		if f := buf.Float16Slice(); f != nil {
			t.Errorf("expected nil Float16Slice on closed buffer, got %v", f)
		}
		if _, err := buf.AsActiveKVBlock(1, "s1"); !errors.Is(err, ErrBufferClosed) {
			t.Errorf("expected ErrBufferClosed for AsActiveKVBlock on closed buffer, got %v", err)
		}

		// 7. AssertPointerIdentity edge cases
		if mgr.AssertPointerIdentity(nil, 0) {
			t.Errorf("AssertPointerIdentity(nil, 0) should return false")
		}
		dummy := 42
		if mgr.AssertPointerIdentity(unsafe.Pointer(&dummy), 0) {
			t.Errorf("AssertPointerIdentity(ptr, 0) should return false")
		}
		if mgr.AssertPointerIdentity(nil, 0x1234) {
			t.Errorf("AssertPointerIdentity(nil, ptr) should return false")
		}
		if mgr.AssertPointerIdentity(unsafe.Pointer(&dummy), uintptr(unsafe.Pointer(&dummy))+1) {
			t.Errorf("AssertPointerIdentity with mismatched pointers should return false")
		}
	})

	t.Run("HugepageAlignment", func(t *testing.T) {
		buf, err := AllocateUMABuffer(1024, HugepageSize2MB)
		if err != nil {
			t.Fatalf("AllocateUMABuffer with 2MB hugepage alignment failed: %v", err)
		}
		defer FreeUMABuffer(buf)

		hp := buf.HostPtr()
		dp := buf.DevPtr()
		if hp != dp {
			t.Errorf("HostPtr 0x%x != DevPtr 0x%x", hp, dp)
		}
		if hp%uintptr(HugepageSize2MB) != 0 {
			t.Errorf("address 0x%x not aligned to 2MB hugepage (mod %d = %d)", hp, HugepageSize2MB, hp%uintptr(HugepageSize2MB))
		}
		if !buf.IsZeroCopy() {
			t.Errorf("IsZeroCopy returned false")
		}
	})

	t.Run("ActiveKVBlockIntegration", func(t *testing.T) {
		buf, err := AllocateUMABuffer(4096, 128)
		if err != nil {
			t.Fatalf("AllocateUMABuffer failed: %v", err)
		}
		defer FreeUMABuffer(buf)

		block, err := buf.AsActiveKVBlock(101, "sess-test-42")
		if err != nil {
			t.Fatalf("AsActiveKVBlock failed: %v", err)
		}
		if block.BlockID != 101 {
			t.Errorf("BlockID = %d, expected 101", block.BlockID)
		}
		if block.SessionID != "sess-test-42" {
			t.Errorf("SessionID = %s, expected sess-test-42", block.SessionID)
		}
		if block.HostPtr != buf.HostPtr() {
			t.Errorf("block HostPtr 0x%x != buf HostPtr 0x%x", block.HostPtr, buf.HostPtr())
		}
		if block.DevicePtr != buf.DevPtr() {
			t.Errorf("block DevicePtr 0x%x != buf DevPtr 0x%x", block.DevicePtr, buf.DevPtr())
		}
		if !block.IsZeroCopy() {
			t.Errorf("block.IsZeroCopy() returned false")
		}
		if block.SizeBytes != 4096 {
			t.Errorf("block.SizeBytes = %d, expected 4096", block.SizeBytes)
		}
	})

	t.Run("CountersAndReset", func(t *testing.T) {
		mgr := NewUMAPointerManager()
		buf, err := mgr.AllocateUMABuffer(2048, 64)
		if err != nil {
			t.Fatalf("allocate failed: %v", err)
		}

		b := []byte{1, 2, 3}
		_, _ = mgr.BytesToDevPtr(b)
		mgr.AssertPointerIdentity(buf.UnsafePointer(), buf.DevPtr())

		stats := mgr.Stats()
		if stats.TotalAllocatedBytes != 2048 {
			t.Errorf("TotalAllocatedBytes = %d, expected 2048", stats.TotalAllocatedBytes)
		}
		if stats.ActiveAllocations != 1 {
			t.Errorf("ActiveAllocations = %d, expected 1", stats.ActiveAllocations)
		}
		if stats.ZeroCopyConversions != 1 {
			t.Errorf("ZeroCopyConversions = %d, expected 1", stats.ZeroCopyConversions)
		}
		if stats.AssertionChecksPassed != 1 {
			t.Errorf("AssertionChecksPassed = %d, expected 1", stats.AssertionChecksPassed)
		}

		mgr.ResetCounters()
		if mgr.TotalAllocatedBytes() != 0 {
			t.Errorf("TotalAllocatedBytes after reset = %d, expected 0", mgr.TotalAllocatedBytes())
		}
		if mgr.ActiveAllocations() != 1 {
			t.Errorf("ActiveAllocations after reset should preserve active count 1, got %d", mgr.ActiveAllocations())
		}
		if mgr.ZeroCopyConversions() != 0 {
			t.Errorf("ZeroCopyConversions after reset = %d, expected 0", mgr.ZeroCopyConversions())
		}
		if mgr.AssertionChecksPassed() != 0 {
			t.Errorf("AssertionChecksPassed after reset = %d, expected 0", mgr.AssertionChecksPassed())
		}

		_ = mgr.FreeUMABuffer(buf)
	})
}
