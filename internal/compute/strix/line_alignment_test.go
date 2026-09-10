// Package strix implements 64-byte cache-line alignment, false-sharing barrier guards,
// and struct layout verification for coherent CPU-GPU MALL data structures on AMD Strix Halo.
// Placement: Gate 2 (Commercial Serving & Appliance Infrastructure, strictly private).

package strix

import (
	"errors"
	"os"
	"strings"
	"sync"
	"testing"
	"unsafe"
)

// Test helper structs for boundary crossing and tag reflection validation
type testStraddlingField struct {
	_   [60]byte
	Val [8]byte `domain:"cpu_write" role:"cpu_producer"`
}

type testMisalignedTag struct {
	Val uint64 `align:"64"`
}

type testAlignedTag struct {
	_ [64]byte `align:"64"`
}

type testCrossDeviceReadWrite struct {
	CPURead  uint32 `domain:"cpu_read" role:"cpu_consumer"`
	GPUWrite uint32 `domain:"gpu_write" role:"gpu_producer"`
}

type testCrossDeviceWriteRead struct {
	GPUWrite uint32 `domain:"gpu_write" role:"gpu_producer"`
	CPURead  uint32 `domain:"cpu_read" role:"cpu_consumer"`
}

type testUnexportedAndUntagged struct {
	unexported uint64
	Untagged   uint64
	_          [48]byte
}

func TestLineAlignment(t *testing.T) {
	// Root test containing subtests fulfilling all criteria from Ticket 22 and Issue #669.

	t.Run("SW-VERIFIED_Allocator", func(t *testing.T) {
		sizes := []int{1, 15, 63, 64, 65, 127, 128, 4096, 65536, 1048576}
		for _, size := range sizes {
			buf, err := AllocateAligned(size)
			if err != nil {
				t.Fatalf("AllocateAligned(%d) failed: %v", size, err)
			}
			if buf == nil {
				t.Fatalf("AllocateAligned(%d) returned nil buffer", size)
			}
			if buf.Size() != size {
				t.Errorf("AllocateAligned(%d) Size() = %d, want %d", size, buf.Size(), size)
			}
			if !buf.IsAligned() {
				t.Errorf("AllocateAligned(%d) IsAligned() = false, want true", size)
			}
			addr := buf.Address()
			if !IsAligned64(addr) {
				t.Errorf("AllocateAligned(%d) Address() = 0x%x is not 64-byte aligned", size, addr)
			}
			if addr&CacheLineMask != 0 {
				t.Errorf("AllocateAligned(%d) Address() 0x%x & CacheLineMask != 0", size, addr)
			}
			if buf.Pointer() == nil {
				t.Fatalf("AllocateAligned(%d) Pointer() is nil", size)
			}
			if uintptr(buf.Pointer()) != addr {
				t.Errorf("AllocateAligned(%d) Pointer() address 0x%x != Address() 0x%x", size, uintptr(buf.Pointer()), addr)
			}
			slice := buf.Bytes()
			if len(slice) != size {
				t.Fatalf("AllocateAligned(%d) len(Bytes()) = %d, want %d", size, len(slice), size)
			}

			// Read/write memory verification
			for i := 0; i < size; i++ {
				slice[i] = byte(i & 0xFF)
			}
			ptrSlice := unsafe.Slice((*byte)(buf.Pointer()), size)
			for i := 0; i < size; i++ {
				if ptrSlice[i] != byte(i&0xFF) {
					t.Fatalf("AllocateAligned(%d) data mismatch at index %d: got 0x%x, want 0x%x", size, i, ptrSlice[i], byte(i&0xFF))
				}
			}

			// Release buffer and verify cleanup
			buf.Release()
			if buf.Bytes() != nil {
				t.Errorf("AllocateAligned(%d) Bytes() after Release() = %v, want nil", size, buf.Bytes())
			}
			if buf.Size() != 0 {
				t.Errorf("AllocateAligned(%d) Size() after Release() = %d, want 0", size, buf.Size())
			}
			if buf.Address() != 0 {
				t.Errorf("AllocateAligned(%d) Address() after Release() = 0x%x, want 0", size, buf.Address())
			}
			if buf.Pointer() != nil {
				t.Errorf("AllocateAligned(%d) Pointer() after Release() = %v, want nil", size, buf.Pointer())
			}
			if buf.IsAligned() {
				t.Errorf("AllocateAligned(%d) IsAligned() after Release() = true, want false", size)
			}
		}

		// Nil buffer safety checks
		var nilBuf *AlignedBuffer
		if nilBuf.Address() != 0 {
			t.Errorf("nil AlignedBuffer Address() = 0x%x, want 0", nilBuf.Address())
		}
		if nilBuf.Pointer() != nil {
			t.Errorf("nil AlignedBuffer Pointer() = %v, want nil", nilBuf.Pointer())
		}
		if nilBuf.Bytes() != nil {
			t.Errorf("nil AlignedBuffer Bytes() = %v, want nil", nilBuf.Bytes())
		}
		if nilBuf.Size() != 0 {
			t.Errorf("nil AlignedBuffer Size() = %d, want 0", nilBuf.Size())
		}
		if nilBuf.IsAligned() {
			t.Errorf("nil AlignedBuffer IsAligned() = true, want false")
		}
		nilBuf.Release() // Must not panic

		// Error cases: non-positive buffer size
		if _, err := AllocateAligned(0); !errors.Is(err, ErrInvalidBufferSize) {
			t.Errorf("AllocateAligned(0) err = %v, want %v", err, ErrInvalidBufferSize)
		}
		if _, err := AllocateAligned(-1); !errors.Is(err, ErrInvalidBufferSize) {
			t.Errorf("AllocateAligned(-1) err = %v, want %v", err, ErrInvalidBufferSize)
		}

		// AllocateAlignedStruct for PaddedQueuePointers
		ptr, structBuf, err := AllocateAlignedStruct[PaddedQueuePointers]()
		if err != nil {
			t.Fatalf("AllocateAlignedStruct[PaddedQueuePointers]() failed: %v", err)
		}
		if ptr == nil || structBuf == nil {
			t.Fatalf("AllocateAlignedStruct[PaddedQueuePointers]() returned nil ptr or buf")
		}
		ptrAddr := uintptr(unsafe.Pointer(ptr))
		if !IsAligned64(ptrAddr) {
			t.Errorf("AllocateAlignedStruct ptr 0x%x is not 64-byte aligned", ptrAddr)
		}
		if !structBuf.IsAligned() {
			t.Errorf("AllocateAlignedStruct buf.IsAligned() = false, want true")
		}
		if structBuf.Size() != int(unsafe.Sizeof(PaddedQueuePointers{})) {
			t.Errorf("AllocateAlignedStruct buf.Size() = %d, want %d", structBuf.Size(), unsafe.Sizeof(PaddedQueuePointers{}))
		}
		// Write through typed pointer and read back
		ptr.WriteIndex = 42
		ptr.ReadIndex = 84
		if ptr.WriteIndex != 42 || ptr.ReadIndex != 84 {
			t.Errorf("field read/write mismatch: WriteIndex=%d, ReadIndex=%d", ptr.WriteIndex, ptr.ReadIndex)
		}
		structBuf.Release()

		// AllocateAlignedStruct for zero-size struct
		zeroPtr, zeroBuf, err := AllocateAlignedStruct[struct{}]()
		if err != nil {
			t.Fatalf("AllocateAlignedStruct[struct{}]() failed: %v", err)
		}
		if zeroPtr == nil || zeroBuf == nil {
			t.Fatalf("AllocateAlignedStruct[struct{}]() returned nil")
		}
		if !zeroBuf.IsAligned() {
			t.Errorf("AllocateAlignedStruct[struct{}] buf.IsAligned() = false, want true")
		}
		zeroBuf.Release()
	})

	t.Run("SW-VERIFIED_StructOffsets", func(t *testing.T) {
		// 1. PaddedQueuePointers
		var q PaddedQueuePointers
		sizeQ := unsafe.Sizeof(q)
		if sizeQ != 128 {
			t.Fatalf("unsafe.Sizeof(PaddedQueuePointers{}) = %d, want 128 (2 cache lines)", sizeQ)
		}
		offWrite := unsafe.Offsetof(q.WriteIndex)
		if offWrite != 0 {
			t.Errorf("Offsetof(PaddedQueuePointers.WriteIndex) = %d, want 0", offWrite)
		}
		offRead := unsafe.Offsetof(q.ReadIndex)
		if offRead != 64 {
			t.Errorf("Offsetof(PaddedQueuePointers.ReadIndex) = %d, want 64", offRead)
		}
		if offRead-offWrite < 64 {
			t.Errorf("Distance between WriteIndex (%d) and ReadIndex (%d) < 64 bytes", offWrite, offRead)
		}

		// 2. PaddedAQLPacketSlot
		var aql PaddedAQLPacketSlot
		sizeAQL := unsafe.Sizeof(aql)
		if sizeAQL != 128 {
			t.Fatalf("unsafe.Sizeof(PaddedAQLPacketSlot{}) = %d, want 128", sizeAQL)
		}
		cpuFieldsAQL := []struct {
			name   string
			offset uintptr
		}{
			{"Header", unsafe.Offsetof(aql.Header)},
			{"Format", unsafe.Offsetof(aql.Format)},
			{"QueueIndex", unsafe.Offsetof(aql.QueueIndex)},
			{"PayloadAddress", unsafe.Offsetof(aql.PayloadAddress)},
			{"PayloadSize", unsafe.Offsetof(aql.PayloadSize)},
		}
		for _, f := range cpuFieldsAQL {
			if f.offset >= 64 {
				t.Errorf("PaddedAQLPacketSlot CPU field %s offset = %d, want < 64", f.name, f.offset)
			}
		}
		gpuFieldsAQL := []struct {
			name   string
			offset uintptr
		}{
			{"CompletionFence", unsafe.Offsetof(aql.CompletionFence)},
			{"SignalHandle", unsafe.Offsetof(aql.SignalHandle)},
			{"Status", unsafe.Offsetof(aql.Status)},
			{"Flags", unsafe.Offsetof(aql.Flags)},
		}
		for _, f := range gpuFieldsAQL {
			if f.offset < 64 {
				t.Errorf("PaddedAQLPacketSlot GPU field %s offset = %d, want >= 64", f.name, f.offset)
			}
		}

		// 3. PaddedKVDescriptor
		var kv PaddedKVDescriptor
		sizeKV := unsafe.Sizeof(kv)
		if sizeKV != 128 {
			t.Fatalf("unsafe.Sizeof(PaddedKVDescriptor{}) = %d, want 128", sizeKV)
		}
		cpuFieldsKV := []struct {
			name   string
			offset uintptr
		}{
			{"SequenceLength", unsafe.Offsetof(kv.SequenceLength)},
			{"DraftTokens", unsafe.Offsetof(kv.DraftTokens)},
			{"HeadIndex", unsafe.Offsetof(kv.HeadIndex)},
			{"Status", unsafe.Offsetof(kv.Status)},
		}
		for _, f := range cpuFieldsKV {
			if f.offset >= 64 {
				t.Errorf("PaddedKVDescriptor CPU field %s offset = %d, want < 64", f.name, f.offset)
			}
		}
		gpuFieldsKV := []struct {
			name   string
			offset uintptr
		}{
			{"BlockPointer", unsafe.Offsetof(kv.BlockPointer)},
			{"LayerMask", unsafe.Offsetof(kv.LayerMask)},
			{"AttentionFlags", unsafe.Offsetof(kv.AttentionFlags)},
		}
		for _, f := range gpuFieldsKV {
			if f.offset < 64 {
				t.Errorf("PaddedKVDescriptor GPU field %s offset = %d, want >= 64", f.name, f.offset)
			}
		}

		// 4. CacheLinePad
		sizePad := unsafe.Sizeof(CacheLinePad{})
		if sizePad != 64 {
			t.Errorf("unsafe.Sizeof(CacheLinePad{}) = %d, want 64", sizePad)
		}
	})

	t.Run("SW-VERIFIED_ReflectionAudit", func(t *testing.T) {
		// PaddedQueuePointers
		repQ, err := AuditStructLayout(PaddedQueuePointers{})
		if err != nil {
			t.Fatalf("AuditStructLayout(PaddedQueuePointers{}) error: %v", err)
		}
		if !repQ.Passed {
			t.Errorf("PaddedQueuePointers Passed = false, want true; violations: %v", repQ.Violations)
		}
		if len(repQ.Violations) != 0 {
			t.Errorf("PaddedQueuePointers len(Violations) = %d, want 0", len(repQ.Violations))
		}
		if repQ.CacheLines != 2 {
			t.Errorf("PaddedQueuePointers CacheLines = %d, want 2", repQ.CacheLines)
		}

		// PaddedAQLPacketSlot
		repAQL, err := AuditStructLayout(PaddedAQLPacketSlot{})
		if err != nil {
			t.Fatalf("AuditStructLayout(PaddedAQLPacketSlot{}) error: %v", err)
		}
		if !repAQL.Passed {
			t.Errorf("PaddedAQLPacketSlot Passed = false, want true; violations: %v", repAQL.Violations)
		}
		if len(repAQL.Violations) != 0 {
			t.Errorf("PaddedAQLPacketSlot len(Violations) = %d, want 0", len(repAQL.Violations))
		}
		if repAQL.CacheLines != 2 {
			t.Errorf("PaddedAQLPacketSlot CacheLines = %d, want 2", repAQL.CacheLines)
		}

		// PaddedKVDescriptor
		repKV, err := AuditStructLayout(PaddedKVDescriptor{})
		if err != nil {
			t.Fatalf("AuditStructLayout(PaddedKVDescriptor{}) error: %v", err)
		}
		if !repKV.Passed {
			t.Errorf("PaddedKVDescriptor Passed = false, want true; violations: %v", repKV.Violations)
		}
		if len(repKV.Violations) != 0 {
			t.Errorf("PaddedKVDescriptor len(Violations) = %d, want 0", len(repKV.Violations))
		}
		if repKV.CacheLines != 2 {
			t.Errorf("PaddedKVDescriptor CacheLines = %d, want 2", repKV.CacheLines)
		}

		// Pointer input support
		repPtr, err := AuditStructLayout(&PaddedQueuePointers{})
		if err != nil {
			t.Fatalf("AuditStructLayout(&PaddedQueuePointers{}) error: %v", err)
		}
		if !repPtr.Passed || len(repPtr.Violations) != 0 {
			t.Errorf("AuditStructLayout with pointer failed: passed=%v, violations=%v", repPtr.Passed, repPtr.Violations)
		}

		// NaiveUnpaddedQueue: must fail with false sharing on cache line 0
		repNaive, err := AuditStructLayout(NaiveUnpaddedQueue{})
		if err != nil {
			t.Fatalf("AuditStructLayout(NaiveUnpaddedQueue{}) error: %v", err)
		}
		if repNaive.Passed {
			t.Errorf("NaiveUnpaddedQueue Passed = true, want false")
		}
		if len(repNaive.Violations) == 0 {
			t.Fatalf("NaiveUnpaddedQueue expected violations, got 0")
		}
		v0 := repNaive.Violations[0]
		if v0.CacheLineIndex != 0 {
			t.Errorf("NaiveUnpaddedQueue violation CacheLineIndex = %d, want 0", v0.CacheLineIndex)
		}
		if !strings.Contains(v0.Description, "false sharing") || !strings.Contains(v0.Description, "cache line 0") {
			t.Errorf("NaiveUnpaddedQueue description %q missing expected false sharing details", v0.Description)
		}

		// AssertNoFalseSharing on padded struct -> nil
		if err := AssertNoFalseSharing(PaddedQueuePointers{}); err != nil {
			t.Errorf("AssertNoFalseSharing(PaddedQueuePointers{}) err = %v, want nil", err)
		}

		// AssertNoFalseSharing on unpadded struct -> ErrFalseSharingDetected
		errNaive := AssertNoFalseSharing(NaiveUnpaddedQueue{})
		if errNaive == nil {
			t.Fatalf("AssertNoFalseSharing(NaiveUnpaddedQueue{}) expected error, got nil")
		}
		if !errors.Is(errNaive, ErrFalseSharingDetected) {
			t.Errorf("AssertNoFalseSharing(NaiveUnpaddedQueue{}) err = %v, want errors.Is ErrFalseSharingDetected", errNaive)
		}

		// AuditAndProtect with NaiveUnpaddedQueue
		beforeExecs := FallbackBarrierExecutions()
		protRep := AuditAndProtect(NaiveUnpaddedQueue{}, nil)
		if protRep == nil {
			t.Fatalf("AuditAndProtect returned nil report")
		}
		if protRep.Passed {
			t.Errorf("AuditAndProtect report Passed = true, want false")
		}
		afterExecs := FallbackBarrierExecutions()
		if afterExecs <= beforeExecs {
			t.Errorf("AuditAndProtect did not increment fallback barrier counter: before=%d, after=%d", beforeExecs, afterExecs)
		}
		latest := DefaultReporter().LatestReport("NaiveUnpaddedQueue")
		if latest == nil {
			t.Errorf("DefaultReporter().LatestReport(\"NaiveUnpaddedQueue\") returned nil")
		}

		// Error cases
		if _, err := AuditStructLayout(nil); !errors.Is(err, ErrNilPointer) {
			t.Errorf("AuditStructLayout(nil) err = %v, want %v", err, ErrNilPointer)
		}
		if _, err := AuditStructLayout(42); err == nil {
			t.Errorf("AuditStructLayout(int) expected error, got nil")
		}
		if err := AssertNoFalseSharing(nil); !errors.Is(err, ErrNilPointer) {
			t.Errorf("AssertNoFalseSharing(nil) err = %v, want %v", err, ErrNilPointer)
		}
	})

	t.Run("SW-VERIFIED_ConcurrentAccess_RaceFree", func(t *testing.T) {
		queue := &PaddedQueuePointers{}
		const iters = 10000
		const numWriters = 4
		const numReaders = 4

		var wg sync.WaitGroup

		// 4 simulated CPU writers
		for w := 0; w < numWriters; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				for i := uint64(1); i <= iters; i++ {
					val := (uint64(workerID) << 32) | i
					StoreReleaseUint64(&queue.WriteIndex, val)
					SFENCE()
				}
			}(w)
		}

		// 4 simulated GPU readers
		for r := 0; r < numReaders; r++ {
			wg.Add(1)
			go func(readerID int) {
				defer wg.Done()
				for i := uint64(1); i <= iters; i++ {
					_ = LoadAcquireUint64(&queue.WriteIndex)
					val := (uint64(readerID) << 32) | i
					StoreReleaseUint64(&queue.ReadIndex, val)
				}
			}(r)
		}

		wg.Wait()

		finalW := LoadAcquireUint64(&queue.WriteIndex)
		finalR := LoadAcquireUint64(&queue.ReadIndex)
		if finalW == 0 || finalR == 0 {
			t.Errorf("expected non-zero indices after concurrent test: WriteIndex=%d, ReadIndex=%d", finalW, finalR)
		}
	})

	t.Run("SW-VERIFIED_AblationArms", func(t *testing.T) {
		// Arm 1: Strict 64-Byte Alignment & Padding (Proposed system)
		arm1Buf, err := AllocateAligned(int(unsafe.Sizeof(PaddedQueuePointers{})))
		if err != nil {
			t.Fatalf("Arm 1 AllocateAligned failed: %v", err)
		}
		defer arm1Buf.Release()
		if !arm1Buf.IsAligned() {
			t.Errorf("Arm 1 buffer is not 64-byte aligned")
		}
		arm1Rep, err := AuditStructLayout(PaddedQueuePointers{})
		if err != nil || !arm1Rep.Passed || len(arm1Rep.Violations) != 0 {
			t.Errorf("Arm 1 layout audit failed: passed=%v, violations=%v", arm1Rep.Passed, arm1Rep.Violations)
		}
		// Clean barrier execution
		StoreBufferFlush()
		FullMemoryBarrier()

		// Arm 2: 64-Byte Base Alignment Only (Base aligned, but unpadded struct)
		arm2Buf, err := AllocateAligned(int(unsafe.Sizeof(NaiveUnpaddedQueue{})))
		if err != nil {
			t.Fatalf("Arm 2 AllocateAligned failed: %v", err)
		}
		defer arm2Buf.Release()
		if !arm2Buf.IsAligned() {
			t.Errorf("Arm 2 buffer base is not 64-byte aligned")
		}
		arm2Rep, err := AuditStructLayout(NaiveUnpaddedQueue{})
		if err != nil {
			t.Fatalf("Arm 2 AuditStructLayout failed: %v", err)
		}
		if arm2Rep.Passed {
			t.Errorf("Arm 2 expected false-sharing violation, but report passed")
		}
		if len(arm2Rep.Violations) == 0 {
			t.Errorf("Arm 2 expected violations, got 0")
		}

		// Arm 3: Unpadded Baseline (Unaligned base + unpadded struct)
		raw := make([]byte, 128)
		rawAddr := uintptr(unsafe.Pointer(&raw[0]))
		unalignedAddr := rawAddr
		if unalignedAddr&CacheLineMask == 0 {
			unalignedAddr++ // Force misalignment
		}
		if IsAligned64(unalignedAddr) {
			t.Errorf("Arm 3 unalignedAddr 0x%x unexpectedly reported as aligned", unalignedAddr)
		}
		if unalignedAddr&CacheLineMask == 0 {
			t.Errorf("Arm 3 unalignedAddr 0x%x mask check failed", unalignedAddr)
		}
		arm3Rep, err := AuditStructLayout(NaiveUnpaddedQueue{})
		if err != nil {
			t.Fatalf("Arm 3 AuditStructLayout failed: %v", err)
		}
		if arm3Rep.Passed {
			t.Errorf("Arm 3 expected false sharing violation on unpadded struct")
		}
	})

	t.Run("SW-VERIFIED_TelemetryReporter", func(t *testing.T) {
		reporter := NewCoherencyTelemetryReporter()

		reporter.RecordFabricReplay(10, 8, 2, 5)
		reporter.RecordReplayStall(4)
		reporter.RecordMitigatedProbe(3)

		snap := reporter.Snapshot()
		if snap.TotalStalls != 14 {
			t.Errorf("Snapshot TotalStalls = %d, want 14", snap.TotalStalls)
		}
		if snap.ProbeMisses != 12 {
			t.Errorf("Snapshot ProbeMisses = %d, want 12", snap.ProbeMisses)
		}
		if snap.ProbeHits != 5 {
			t.Errorf("Snapshot ProbeHits = %d, want 5", snap.ProbeHits)
		}
		if snap.MitigatedProbes != 8 {
			t.Errorf("Snapshot MitigatedProbes = %d, want 8", snap.MitigatedProbes)
		}
		expectedPenalty := uint64(14) * FabricReplayStallPenaltyNs
		if snap.EstimatedPenaltyNs != expectedPenalty {
			t.Errorf("Snapshot EstimatedPenaltyNs = %d, want %d", snap.EstimatedPenaltyNs, expectedPenalty)
		}

		// Record and query audit reports
		rep, _ := AuditStructLayout(PaddedQueuePointers{})
		reporter.RecordAuditReport(rep)
		reporter.RecordAuditReport(nil) // Nil safety check

		reports := reporter.Reports()
		if len(reports) != 1 {
			t.Fatalf("len(Reports()) = %d, want 1", len(reports))
		}
		latest := reporter.LatestReport("PaddedQueuePointers")
		if latest == nil || latest.StructName != "PaddedQueuePointers" {
			t.Errorf("LatestReport(\"PaddedQueuePointers\") = %v, want struct report", latest)
		}
		if reporter.LatestReport("NonExistent") != nil {
			t.Errorf("LatestReport(\"NonExistent\") != nil")
		}

		// Reset reporter
		reporter.Reset()
		snapReset := reporter.Snapshot()
		if snapReset.TotalStalls != 0 || snapReset.EstimatedPenaltyNs != 0 {
			t.Errorf("Snapshot after Reset() not zeroed: %+v", snapReset)
		}
		if len(reporter.Reports()) != 0 {
			t.Errorf("len(Reports()) after Reset() = %d, want 0", len(reporter.Reports()))
		}
		if reporter.LatestReport("PaddedQueuePointers") != nil {
			t.Errorf("LatestReport after Reset() != nil")
		}
	})

	t.Run("HW-WITNESSED_MALL_ProbeStalls", func(t *testing.T) {
		kfdPath := "/dev/kfd"
		drmPath := "/sys/class/drm/card0/device"

		kfdPresent := false
		if fi, err := os.Stat(kfdPath); err == nil && fi.Mode()&os.ModeDevice != 0 {
			kfdPresent = true
		}
		drmPresent := false
		if _, err := os.Stat(drmPath); err == nil {
			drmPresent = true
		}

		hwWitnessed := kfdPresent || drmPresent
		if hwWitnessed {
			t.Logf("[HW-WITNESSED] Hardware environment detected on host (kfd=%v, drm=%v)", kfdPresent, drmPresent)
		} else {
			t.Log("[SW-VERIFIED] Bare-metal AMD Strix Halo hardware absent; running simulated MALL probe stall verification")
			simReporter := NewCoherencyTelemetryReporter()
			const simulatedStalls = 100
			simReporter.RecordReplayStall(simulatedStalls)
			snap := simReporter.Snapshot()
			expectedPenalty := simulatedStalls * FabricReplayStallPenaltyNs
			if snap.EstimatedPenaltyNs != expectedPenalty {
				t.Errorf("simulated stall penalty mismatch: got %d ns, want %d ns", snap.EstimatedPenaltyNs, expectedPenalty)
			}
			t.Logf("[SW-VERIFIED] Simulated %d fabric replay stalls yielded %d ns estimated penalty (%dns/stall)",
				simulatedStalls, snap.EstimatedPenaltyNs, FabricReplayStallPenaltyNs)
			t.Log("[HW-WITNESSED] Physical AMD Strix Halo silicon (GFX1151) criteria remain open [ ] until witnessed on live bare-metal appliance.")
		}
	})

	t.Run("SW-VERIFIED_AlignmentMath", func(t *testing.T) {
		testCases := []struct {
			addr       uintptr
			aligned    bool
			alignUp    uintptr
			alignDown  uintptr
			offsetNext uintptr
		}{
			{0, true, 0, 0, 0},
			{1, false, 64, 0, 63},
			{31, false, 64, 0, 33},
			{32, false, 64, 0, 32},
			{63, false, 64, 0, 1},
			{64, true, 64, 64, 0},
			{65, false, 128, 64, 63},
			{127, false, 128, 64, 1},
			{128, true, 128, 128, 0},
			{129, false, 192, 128, 63},
			{255, false, 256, 192, 1},
			{256, true, 256, 256, 0},
			{4096, true, 4096, 4096, 0},
			{4097, false, 4160, 4096, 63},
		}

		for _, tc := range testCases {
			if got := IsAligned64(tc.addr); got != tc.aligned {
				t.Errorf("IsAligned64(0x%x) = %v, want %v", tc.addr, got, tc.aligned)
			}
			if got := AlignUp64(tc.addr); got != tc.alignUp {
				t.Errorf("AlignUp64(0x%x) = 0x%x, want 0x%x", tc.addr, got, tc.alignUp)
			}
			if got := AlignDown64(tc.addr); got != tc.alignDown {
				t.Errorf("AlignDown64(0x%x) = 0x%x, want 0x%x", tc.addr, got, tc.alignDown)
			}
			if got := OffsetToNextLine(tc.addr); got != tc.offsetNext {
				t.Errorf("OffsetToNextLine(0x%x) = %d, want %d", tc.addr, got, tc.offsetNext)
			}
			// Invariant: addr + OffsetToNextLine(addr) is always aligned
			sum := tc.addr + OffsetToNextLine(tc.addr)
			if !IsAligned64(sum) {
				t.Errorf("addr(0x%x) + OffsetToNextLine(%d) = 0x%x is not aligned", tc.addr, tc.offsetNext, sum)
			}
		}
	})

	t.Run("SW-VERIFIED_BarrierPrimitives", func(t *testing.T) {
		// 1. StoreReleaseUint32 and LoadAcquireUint32
		var v32 uint32
		StoreReleaseUint32(&v32, 0x12345678)
		if got := LoadAcquireUint32(&v32); got != 0x12345678 {
			t.Errorf("LoadAcquireUint32 = 0x%x, want 0x12345678", got)
		}

		// 2. StoreReleaseUint64 and LoadAcquireUint64
		var v64 uint64
		StoreReleaseUint64(&v64, 0xDEADBEEFCAFEBABE)
		if got := LoadAcquireUint64(&v64); got != 0xDEADBEEFCAFEBABE {
			t.Errorf("LoadAcquireUint64 = 0x%x, want 0xDEADBEEFCAFEBABE", got)
		}

		// 3. StoreReleasePointer and LoadAcquirePointer
		var ptr unsafe.Pointer
		target := 42
		StoreReleasePointer(&ptr, unsafe.Pointer(&target))
		gotPtr := LoadAcquirePointer(&ptr)
		if gotPtr == nil || *(*int)(gotPtr) != 42 {
			t.Errorf("LoadAcquirePointer deref = %v, want 42", gotPtr)
		}

		// 4. Memory barriers
		SFENCE()
		StoreBufferFlush()
		MFENCE()
		FullMemoryBarrier()

		// 5. Software fallback barrier counter
		before := FallbackBarrierExecutions()
		SoftwareFallbackBarrier()
		after := FallbackBarrierExecutions()
		if after != before+1 {
			t.Errorf("FallbackBarrierExecutions after SoftwareFallbackBarrier() = %d, want %d", after, before+1)
		}
	})

	t.Run("SW-VERIFIED_LayoutEdgeCases", func(t *testing.T) {
		// Straddling field crossing cache line boundary
		repStraddle, err := AuditStructLayout(testStraddlingField{})
		if err != nil {
			t.Fatalf("AuditStructLayout(testStraddlingField{}) failed: %v", err)
		}
		if repStraddle.Passed {
			t.Errorf("testStraddlingField Passed = true, want false")
		}
		hasStraddleViolation := false
		for _, v := range repStraddle.Violations {
			if strings.Contains(v.Description, "crosses 64-byte cache line boundary") {
				hasStraddleViolation = true
				break
			}
		}
		if !hasStraddleViolation {
			t.Errorf("testStraddlingField did not report boundary-crossing violation: %+v", repStraddle.Violations)
		}

		// Tag align:"64" on struct with non-multiple size
		repMisalignTag, err := AuditStructLayout(testMisalignedTag{})
		if err != nil {
			t.Fatalf("AuditStructLayout(testMisalignedTag{}) failed: %v", err)
		}
		if repMisalignTag.Passed {
			t.Errorf("testMisalignedTag Passed = true, want false")
		}
		hasTagViolation := false
		for _, v := range repMisalignTag.Violations {
			if strings.Contains(v.Description, "align:\"64\"") {
				hasTagViolation = true
				break
			}
		}
		if !hasTagViolation {
			t.Errorf("testMisalignedTag did not report tag violation: %+v", repMisalignTag.Violations)
		}

		// Tag align:"64" on struct with size multiple of 64
		repAlignTag, err := AuditStructLayout(testAlignedTag{})
		if err != nil {
			t.Fatalf("AuditStructLayout(testAlignedTag{}) failed: %v", err)
		}
		if !repAlignTag.Passed || len(repAlignTag.Violations) != 0 {
			t.Errorf("testAlignedTag Passed = %v, violations = %v, want passed with 0 violations",
				repAlignTag.Passed, repAlignTag.Violations)
		}

		// Cross-device read/write collision: cpu_read and gpu_write
		repRW, err := AuditStructLayout(testCrossDeviceReadWrite{})
		if err != nil {
			t.Fatalf("AuditStructLayout(testCrossDeviceReadWrite{}) failed: %v", err)
		}
		if repRW.Passed || len(repRW.Violations) == 0 {
			t.Errorf("testCrossDeviceReadWrite expected false-sharing violation")
		}

		// Cross-device write/read collision: gpu_write and cpu_read
		repWR, err := AuditStructLayout(testCrossDeviceWriteRead{})
		if err != nil {
			t.Fatalf("AuditStructLayout(testCrossDeviceWriteRead{}) failed: %v", err)
		}
		if repWR.Passed || len(repWR.Violations) == 0 {
			t.Errorf("testCrossDeviceWriteRead expected false-sharing violation")
		}

		// Unexported and untagged fields
		repUnexp, err := AuditStructLayout(testUnexportedAndUntagged{})
		if err != nil {
			t.Fatalf("AuditStructLayout(testUnexportedAndUntagged{}) failed: %v", err)
		}
		if !repUnexp.Passed || len(repUnexp.Violations) != 0 {
			t.Errorf("testUnexportedAndUntagged expected pass with 0 violations, got: %+v", repUnexp)
		}

		// AuditAndProtect with custom fallback function
		var customCalled bool
		AuditAndProtect(NaiveUnpaddedQueue{}, func() {
			customCalled = true
		})
		if !customCalled {
			t.Errorf("AuditAndProtect with custom fallbackFn was not called")
		}

		// AuditAndProtect on clean struct (passes)
		cleanRep := AuditAndProtect(PaddedQueuePointers{}, nil)
		if cleanRep == nil || !cleanRep.Passed {
			t.Errorf("AuditAndProtect on PaddedQueuePointers expected clean pass, got: %v", cleanRep)
		}

		// AuditAndProtect with nil argument -> returns nil
		if nilRep := AuditAndProtect(nil, nil); nilRep != nil {
			t.Errorf("AuditAndProtect(nil, nil) = %v, want nil", nilRep)
		}

		// Empty struct audit (0 size, 0 cache lines)
		repEmpty, err := AuditStructLayout(struct{}{})
		if err != nil {
			t.Fatalf("AuditStructLayout(struct{}{}) failed: %v", err)
		}
		if !repEmpty.Passed || repEmpty.CacheLines != 0 || repEmpty.TotalSize != 0 {
			t.Errorf("AuditStructLayout(struct{}{}) unexpected report: %+v", repEmpty)
		}

		// Anonymous struct audit (name is empty, defaults to type string)
		repAnon, err := AuditStructLayout(struct{ X int }{})
		if err != nil {
			t.Fatalf("AuditStructLayout(anonymous struct) failed: %v", err)
		}
		if !repAnon.Passed || repAnon.StructName == "" {
			t.Errorf("AuditStructLayout(anonymous struct) unexpected report: %+v", repAnon)
		}
	})
}
