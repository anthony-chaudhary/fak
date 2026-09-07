package compute

import (
	"encoding/binary"
	"errors"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
)

type mockHSADoorbell struct {
	rings uint64
}

func (m *mockHSADoorbell) Ring(writePtr uint64) {
	atomic.AddUint64(&m.rings, 1)
}

// TestStrixAQL_HSAKernelDispatchPacketLayoutAndAlignment asserts that HSAKernelDispatchPacket
// is exactly 64 bytes (Zen 5 CPU / RDNA 3.5 L2 cache line width) and verifies all field offsets.
func TestStrixAQL_HSAKernelDispatchPacketLayoutAndAlignment(t *testing.T) {
	pktSize := unsafe.Sizeof(HSAKernelDispatchPacket{})
	if pktSize != 64 {
		t.Fatalf("HSAKernelDispatchPacket size must be exactly 64 bytes, got %d", pktSize)
	}

	p := &HSAKernelDispatchPacket{}

	if off := unsafe.Offsetof(p.Header); off != 0 {
		t.Errorf("expected Header at offset 0, got %d", off)
	}
	if off := unsafe.Offsetof(p.Setup); off != 2 {
		t.Errorf("expected Setup at offset 2, got %d", off)
	}
	if off := unsafe.Offsetof(p.WorkgroupSizeX); off != 4 {
		t.Errorf("expected WorkgroupSizeX at offset 4, got %d", off)
	}
	if off := unsafe.Offsetof(p.WorkgroupSizeY); off != 6 {
		t.Errorf("expected WorkgroupSizeY at offset 6, got %d", off)
	}
	if off := unsafe.Offsetof(p.WorkgroupSizeZ); off != 8 {
		t.Errorf("expected WorkgroupSizeZ at offset 8, got %d", off)
	}
	if off := unsafe.Offsetof(p.Reserved0); off != 10 {
		t.Errorf("expected Reserved0 at offset 10, got %d", off)
	}
	if off := unsafe.Offsetof(p.GridSizeX); off != 12 {
		t.Errorf("expected GridSizeX at offset 12, got %d", off)
	}
	if off := unsafe.Offsetof(p.GridSizeY); off != 16 {
		t.Errorf("expected GridSizeY at offset 16, got %d", off)
	}
	if off := unsafe.Offsetof(p.GridSizeZ); off != 20 {
		t.Errorf("expected GridSizeZ at offset 20, got %d", off)
	}
	if off := unsafe.Offsetof(p.PrivateSegmentSize); off != 24 {
		t.Errorf("expected PrivateSegmentSize at offset 24, got %d", off)
	}
	if off := unsafe.Offsetof(p.GroupSegmentSize); off != 28 {
		t.Errorf("expected GroupSegmentSize at offset 28, got %d", off)
	}
	if off := unsafe.Offsetof(p.KernelObject); off != 32 {
		t.Errorf("expected KernelObject at offset 32, got %d", off)
	}
	if off := unsafe.Offsetof(p.KernargAddress); off != 40 {
		t.Errorf("expected KernargAddress at offset 40, got %d", off)
	}
	if off := unsafe.Offsetof(p.Reserved2); off != 48 {
		t.Errorf("expected Reserved2 at offset 48, got %d", off)
	}
	if off := unsafe.Offsetof(p.CompletionSignal); off != 56 {
		t.Errorf("expected CompletionSignal at offset 56, got %d", off)
	}
}

// TestStrixAQL_PacketCreationAndSerialization tests packet construction, bitfield packing,
// 64-byte serialization, and deserialization roundtrip.
func TestStrixAQL_PacketCreationAndSerialization(t *testing.T) {
	pkt := NewKernelDispatchPacket(
		2,         // dimensions = 2
		256, 1, 1, // wg: 256, 1, 1
		2048, 4, 1, // grid: 2048, 4, 1
		0x7FFF00001000, // kernel object GPU VA
		0x7FFF00002000, // kernarg address GPU VA
		16384,          // LDS = 16KB
		128,            // private = 128B
		0xCAFEBABE,     // completion signal
	)

	// Check header bitfield unpacking
	pktType := pkt.Header & HSAHeaderTypeMask
	if pktType != HSAPacketTypeKernelDispatch {
		t.Errorf("expected packet type %d, got %d", HSAPacketTypeKernelDispatch, pktType)
	}
	barrier := (pkt.Header & HSAHeaderBarrierBit) != 0
	if barrier {
		t.Errorf("expected barrier=false, got true")
	}
	acq := (pkt.Header >> HSAHeaderAcquireShift) & HSAHeaderAcquireMask
	rel := (pkt.Header >> HSAHeaderReleaseShift) & HSAHeaderReleaseMask
	if acq != HSAFenceScopeAgent || rel != HSAFenceScopeAgent {
		t.Errorf("expected agent fences (%d, %d), got (%d, %d)", HSAFenceScopeAgent, HSAFenceScopeAgent, acq, rel)
	}

	// Serialize
	raw := pkt.Serialize()
	if len(raw) != 64 {
		t.Fatalf("expected 64 bytes, got %d", len(raw))
	}

	// Verify byte content
	if binary.LittleEndian.Uint16(raw[0:2]) != pkt.Header {
		t.Errorf("header mismatch in serialized bytes")
	}
	if binary.LittleEndian.Uint16(raw[2:4]) != 2 {
		t.Errorf("setup mismatch in serialized bytes")
	}
	if binary.LittleEndian.Uint16(raw[4:6]) != 256 {
		t.Errorf("wgX mismatch in serialized bytes")
	}
	if binary.LittleEndian.Uint32(raw[12:16]) != 2048 {
		t.Errorf("gridX mismatch in serialized bytes")
	}
	if binary.LittleEndian.Uint64(raw[32:40]) != 0x7FFF00001000 {
		t.Errorf("kernelObject mismatch in serialized bytes")
	}
	if binary.LittleEndian.Uint64(raw[40:48]) != 0x7FFF00002000 {
		t.Errorf("kernargAddress mismatch in serialized bytes")
	}
	if binary.LittleEndian.Uint64(raw[56:64]) != 0xCAFEBABE {
		t.Errorf("completionSignal mismatch in serialized bytes")
	}

	// Roundtrip deserialization
	deser, err := DeserializeKernelDispatchPacket(raw[:])
	if err != nil {
		t.Fatalf("deserialization failed: %v", err)
	}
	if deser != pkt {
		t.Fatalf("deserialized packet %+v != original %+v", deser, pkt)
	}

	// Invalid short packet
	_, err = DeserializeKernelDispatchPacket(raw[:63])
	if !errors.Is(err, ErrInvalidPacketSize) {
		t.Fatalf("expected ErrInvalidPacketSize for 63 bytes, got %v", err)
	}
}

// TestStrixAQL_PM4HeaderEncoding verifies PM4 Type-3 packet header calculation.
func TestStrixAQL_PM4HeaderEncoding(t *testing.T) {
	cases := []struct {
		opcode uint8
		count  uint16
	}{
		{PKT3_DISPATCH_DIRECT, 3},
		{PKT3_SET_SH_REG, 2},
		{PKT3_ACQUIRE_MEM, 5},
		{PKT3_RELEASE_MEM, 6},
	}

	for _, c := range cases {
		hdr := PM4Packet3Header(c.opcode, c.count)
		pktType := (hdr >> 30) & 0x3
		cnt := uint16((hdr >> 16) & 0x3FFF)
		op := uint8((hdr >> 8) & 0xFF)
		reserved := hdr & 0xFF

		if pktType != 3 {
			t.Errorf("expected type 3, got %d", pktType)
		}
		if cnt != c.count {
			t.Errorf("expected count %d, got %d", c.count, cnt)
		}
		if op != c.opcode {
			t.Errorf("expected opcode 0x%X, got 0x%X", c.opcode, op)
		}
		if reserved != 0 {
			t.Errorf("expected reserved bottom byte 0, got 0x%X", reserved)
		}
	}
}

// TestStrixAQL_PM4CommandBuffer_DispatchDirect tests emission of PKT3_DISPATCH_DIRECT.
func TestStrixAQL_PM4CommandBuffer_DispatchDirect(t *testing.T) {
	cb := NewPM4CommandBuffer()
	cb.EmitDispatchDirect(1024, 2, 1, 0x00000001)

	dwords := cb.Dwords()
	if len(dwords) != 5 {
		t.Fatalf("expected 5 DWORDs, got %d", len(dwords))
	}

	expectedHdr := PM4Packet3Header(PKT3_DISPATCH_DIRECT, 3)
	if dwords[0] != expectedHdr {
		t.Errorf("expected header 0x%08X, got 0x%08X", expectedHdr, dwords[0])
	}
	if dwords[1] != 1024 || dwords[2] != 2 || dwords[3] != 1 {
		t.Errorf("grid dimensions mismatch: %v", dwords[1:4])
	}
	if dwords[4] != 0x00000001 {
		t.Errorf("flags mismatch: 0x%08X", dwords[4])
	}
}

// TestStrixAQL_PM4CommandBuffer_SetSHReg tests emission of PKT3_SET_SH_REG.
func TestStrixAQL_PM4CommandBuffer_SetSHReg(t *testing.T) {
	cb := NewPM4CommandBuffer()
	cb.EmitSetSHReg(RegComputePgmRsrc1, 0x12345678, 0x87654321)

	dwords := cb.Dwords()
	if len(dwords) != 4 {
		t.Fatalf("expected 4 DWORDs, got %d", len(dwords))
	}

	expectedHdr := PM4Packet3Header(PKT3_SET_SH_REG, 2)
	if dwords[0] != expectedHdr {
		t.Errorf("expected header 0x%08X, got 0x%08X", expectedHdr, dwords[0])
	}
	expectedRegOffset := RegComputePgmRsrc1 - RegSHConfigBase
	if dwords[1] != expectedRegOffset {
		t.Errorf("expected register offset 0x%08X, got 0x%08X", expectedRegOffset, dwords[1])
	}
	if dwords[2] != 0x12345678 || dwords[3] != 0x87654321 {
		t.Errorf("register values mismatch: %v", dwords[2:])
	}
}

// TestStrixAQL_PM4CommandBuffer_SetComputePgmRsrc tests Wave32 mode and LDS register configuration.
func TestStrixAQL_PM4CommandBuffer_SetComputePgmRsrc(t *testing.T) {
	cb := NewPM4CommandBuffer()
	cb.EmitSetComputePgmRsrc(true, 128, 32, 4096)

	dwords := cb.Dwords()
	if len(dwords) != 4 {
		t.Fatalf("expected 4 DWORDs, got %d", len(dwords))
	}

	rsrc1 := dwords[2]
	rsrc2 := dwords[3]

	// Verify Wave32 bit is set
	if (rsrc1 & ComputePgmRsrc1Wave32Bit) == 0 {
		t.Errorf("expected Wave32 bit to be set in PGM_RSRC1, got 0x%08X", rsrc1)
	}

	// Verify LDS allocation: 4096 bytes = 8 blocks of 512 bytes
	ldsBlocks := (rsrc2 >> 15) & 0x1FF
	if ldsBlocks != 8 {
		t.Errorf("expected 8 LDS blocks (4096B), got %d (rsrc2=0x%08X)", ldsBlocks, rsrc2)
	}
}

// TestStrixAQL_PM4CommandBuffer_AcquireMem_MALLFlush tests PKT3_ACQUIRE_MEM with MALL cache flush flags.
func TestStrixAQL_PM4CommandBuffer_AcquireMem_MALLFlush(t *testing.T) {
	cb := NewPM4CommandBuffer()
	flags := CPCoherCNTLCoherInvL1 | CPCoherCNTLCoherInvL2 | CPCoherCNTLMALLFlushAll
	cb.EmitAcquireMem(flags)

	dwords := cb.Dwords()
	if len(dwords) != 7 {
		t.Fatalf("expected 7 DWORDs, got %d", len(dwords))
	}

	expectedHdr := PM4Packet3Header(PKT3_ACQUIRE_MEM, 5)
	if dwords[0] != expectedHdr {
		t.Errorf("expected header 0x%08X, got 0x%08X", expectedHdr, dwords[0])
	}
	if dwords[1] != flags {
		t.Errorf("expected flags 0x%08X, got 0x%08X", flags, dwords[1])
	}
	if (dwords[1] & CPCoherCNTLMALLFlushAll) != CPCoherCNTLMALLFlushAll {
		t.Errorf("MALL flush flags missing from acquire mem")
	}
	if dwords[2] != 0xFFFFFFFF {
		t.Errorf("expected full range 0xFFFFFFFF, got 0x%08X", dwords[2])
	}
	if dwords[6] != 10 {
		t.Errorf("expected poll interval 10, got %d", dwords[6])
	}
}

// TestStrixAQL_PM4CommandBuffer_ReleaseMem tests memory release fence emission and byte serialization.
func TestStrixAQL_PM4CommandBuffer_ReleaseMem(t *testing.T) {
	cb := NewPM4CommandBuffer()
	event := uint32(0x10) // CACHE_FLUSH_AND_INV_TS_EVENT
	dataSel := uint32(1)  // write 64-bit int
	addr := uint64(0x7FFF00004000)
	data := uint64(0xDEADBEEFCAFE0001)

	cb.EmitReleaseMem(event, dataSel, addr, data)

	dwords := cb.Dwords()
	if len(dwords) != 8 {
		t.Fatalf("expected 8 DWORDs, got %d", len(dwords))
	}

	expectedHdr := PM4Packet3Header(PKT3_RELEASE_MEM, 6)
	if dwords[0] != expectedHdr {
		t.Errorf("expected header 0x%08X, got 0x%08X", expectedHdr, dwords[0])
	}
	if dwords[1] != event {
		t.Errorf("expected event 0x%X, got 0x%X", event, dwords[1])
	}
	if dwords[2] != dataSel {
		t.Errorf("expected dataSel 0x%X, got 0x%X", dataSel, dwords[2])
	}

	// Verify byte serialization matches little-endian DWORDs
	rawBytes := cb.Bytes()
	if len(rawBytes) != len(dwords)*4 {
		t.Fatalf("expected %d bytes, got %d", len(dwords)*4, len(rawBytes))
	}
	for i, dw := range dwords {
		bDw := binary.LittleEndian.Uint32(rawBytes[i*4 : (i+1)*4])
		if bDw != dw {
			t.Errorf("DWORD %d byte serialization mismatch: 0x%08X vs 0x%08X", i, dw, bDw)
		}
	}
}

// TestStrixAQL_StrixAQLQueue_SubmissionSaturationWrapAround tests circular ring buffer
// enqueuing, release semantics, queue saturation detection, and wrap-around.
func TestStrixAQL_StrixAQLQueue_SubmissionSaturationWrapAround(t *testing.T) {
	doorbell := NewHSADoorbell("db-test-wrap", 0x7FFF0000, 1)
	var mmioReg uint64

	queue := NewStrixAQLQueue(1, 4, doorbell) // Tiny queue of 4 entries
	queue.SetDoorbellPtr(&mmioReg)

	pkt := NewKernelDispatchPacket(1, 64, 1, 1, 64, 1, 1, 0x1000, 0x2000, 0, 0, 1)

	// Submit 4 packets: saturates queue
	for i := 0; i < 4; i++ {
		nextIdx, err := queue.SubmitPacket(pkt)
		if err != nil {
			t.Fatalf("submit %d failed: %v", i, err)
		}
		if nextIdx != uint64(i+1) {
			t.Errorf("expected write pointer %d, got %d", i+1, nextIdx)
		}
	}

	// Verify doorbell received all 4 rings
	if doorbell.ReadRelaxed() != 4 {
		t.Errorf("expected doorbell 4, got %d", doorbell.ReadRelaxed())
	}
	if atomic.LoadUint64(&mmioReg) != 4 {
		t.Errorf("expected mmioReg 4, got %d", mmioReg)
	}

	// 5th submit must fail with ErrAQLQueueFull
	_, err := queue.SubmitPacket(pkt)
	if !errors.Is(err, ErrAQLQueueFull) {
		t.Fatalf("expected ErrAQLQueueFull on saturated queue, got %v", err)
	}

	// Simulate GPU CP consuming 2 packets
	queue.AdvanceReadPtr(2)
	if queue.LoadReadPtr() != 2 {
		t.Fatalf("expected read pointer 2, got %d", queue.LoadReadPtr())
	}

	// Submit 2 more packets (wrapping around circular buffer slots 0 and 1)
	for i := 4; i < 6; i++ {
		nextIdx, err := queue.SubmitPacket(pkt)
		if err != nil {
			t.Fatalf("wrap-around submit %d failed: %v", i, err)
		}
		if nextIdx != uint64(i+1) {
			t.Errorf("expected write pointer %d, got %d", i+1, nextIdx)
		}
	}

	// Queue is full again (write=6, read=2, diff=4)
	_, err = queue.SubmitPacket(pkt)
	if !errors.Is(err, ErrAQLQueueFull) {
		t.Fatalf("expected ErrAQLQueueFull after wrap-around saturation, got %v", err)
	}

	if atomic.LoadUint64(&mmioReg) != 6 {
		t.Errorf("expected mmioReg 6, got %d", mmioReg)
	}
	if doorbell.ReadRelaxed() != 6 {
		t.Errorf("expected doorbell 6, got %d", doorbell.ReadRelaxed())
	}
}

// TestStrixAQL_LaunchOverheadUnderOneMicrosecond asserts that dispatch turnaround overhead
// is strictly <= 1.0 µs (1000 ns).
func TestStrixAQL_LaunchOverheadUnderOneMicrosecond(t *testing.T) {
	doorbell := NewHSADoorbell("db-test-overhead", 0x7FFF0000, 1)
	var mmioReg uint64

	queue := NewStrixAQLQueue(1, 1024, doorbell)
	queue.SetDoorbellPtr(&mmioReg)

	pkt := NewKernelDispatchPacket(1, 32, 1, 1, 32, 1, 1, 0x1000, 0x2000, 0, 0, 1)

	const iterations = 10000

	// Warmup
	for i := 0; i < 100; i++ {
		_, _ = queue.SubmitPacket(pkt)
		queue.AdvanceReadPtr(1)
	}

	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, err := queue.SubmitPacket(pkt)
		if err != nil {
			t.Fatalf("iteration %d failed: %v", i, err)
		}
		queue.AdvanceReadPtr(1) // prevent saturation
	}
	totalElapsed := time.Since(start)

	avgNs := totalElapsed.Nanoseconds() / iterations
	avgUs := float64(avgNs) / 1000.0

	t.Logf("Strix AQL Dispatch Turnaround across %d iterations: total %v, avg %.3f µs (%d ns)",
		iterations, totalElapsed, avgUs, avgNs)

	// Hard requirement: launch overhead <= 1.0 µs (1000 ns)
	if avgNs > 1000 {
		t.Fatalf("dispatch launch overhead violated 1.0 µs SLA: avg %.3f µs (%d ns)", avgUs, avgNs)
	}
}

// BenchmarkStrixAQL_DispatchTurnaround benchmarks packet enqueue and atomic MMIO doorbell ring.
func BenchmarkStrixAQL_DispatchTurnaround(b *testing.B) {
	doorbell := NewHSADoorbell("db-test-bench", 0x7FFF0000, 1)
	var mmioReg uint64

	queue := NewStrixAQLQueue(1, 2048, doorbell)
	queue.SetDoorbellPtr(&mmioReg)

	pkt := NewKernelDispatchPacket(1, 32, 1, 1, 32, 1, 1, 0x1000, 0x2000, 0, 0, 1)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _ = queue.SubmitPacket(pkt)
		queue.AdvanceReadPtr(1)
	}
}
