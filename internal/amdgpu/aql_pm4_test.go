package amdgpu

import (
	"bytes"
	"testing"
	"unsafe"
)

// TestAQLPacketSizesAndAlignment verifies that all AQL packet types have exactly 64 bytes
// in memory and when serialized, fulfilling the AMD HSA ABI specification.
func TestAQLPacketSizesAndAlignment(t *testing.T) {
	// Static struct size checks
	if sz := unsafe.Sizeof(AQLKernelDispatchPacket{}); sz != 64 {
		t.Fatalf("AQLKernelDispatchPacket struct size = %d bytes; want exactly 64 bytes", sz)
	}
	if sz := unsafe.Sizeof(AQLBarrierAndPacket{}); sz != 64 {
		t.Fatalf("AQLBarrierAndPacket struct size = %d bytes; want exactly 64 bytes", sz)
	}
	if sz := unsafe.Sizeof(AQLBarrierOrPacket{}); sz != 64 {
		t.Fatalf("AQLBarrierOrPacket struct size = %d bytes; want exactly 64 bytes", sz)
	}

	// Dynamic serialized byte lengths
	kPkt := AQLKernelDispatchPacket{}
	kBytes, err := kPkt.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary failed: %v", err)
	}
	if len(kBytes) != 64 {
		t.Fatalf("serialized AQLKernelDispatchPacket = %d bytes; want 64", len(kBytes))
	}

	bandPkt := AQLBarrierAndPacket{}
	bandBytes, err := bandPkt.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary failed: %v", err)
	}
	if len(bandBytes) != 64 {
		t.Fatalf("serialized AQLBarrierAndPacket = %d bytes; want 64", len(bandBytes))
	}

	borPkt := AQLBarrierOrPacket{}
	borBytes, err := borPkt.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary failed: %v", err)
	}
	if len(borBytes) != 64 {
		t.Fatalf("serialized AQLBarrierOrPacket = %d bytes; want 64", len(borBytes))
	}
}

// TestAQLHeaderBitfields verifies packing and unpacking of AQL packet header bitfields:
// packet type (bits 0..7), barrier bit (bit 8), acquire fence (bits 9..10), and release fence (bits 11..12).
func TestAQLHeaderBitfields(t *testing.T) {
	cases := []struct {
		name         string
		pktType      AQLPacketType
		barrier      bool
		acquireScope AQLFenceScope
		releaseScope AQLFenceScope
		expectedHdr  uint16
	}{
		{
			name:         "KernelDispatch_NoBarrier_NoFences",
			pktType:      AQLPacketTypeKernelDispatch,
			barrier:      false,
			acquireScope: AQLFenceScopeNone,
			releaseScope: AQLFenceScopeNone,
			expectedHdr:  0x0002,
		},
		{
			name:         "KernelDispatch_Barrier_AgentFences",
			pktType:      AQLPacketTypeKernelDispatch,
			barrier:      true,
			acquireScope: AQLFenceScopeAgent,
			releaseScope: AQLFenceScopeAgent,
			// type = 2 | barrier = 0x100 | acquire = 1 << 9 (0x200) | release = 1 << 11 (0x800) -> 0x0B02
			expectedHdr: 0x0B02,
		},
		{
			name:         "BarrierAnd_SystemFences",
			pktType:      AQLPacketTypeBarrierAnd,
			barrier:      true,
			acquireScope: AQLFenceScopeSystem,
			releaseScope: AQLFenceScopeSystem,
			// type = 3 | barrier = 0x100 | acquire = 2 << 9 (0x400) | release = 2 << 11 (0x1000) -> 0x1503
			expectedHdr: 0x1503,
		},
		{
			name:         "BarrierOr_NoBarrier_SystemAcquire_AgentRelease",
			pktType:      AQLPacketTypeBarrierOr,
			barrier:      false,
			acquireScope: AQLFenceScopeSystem,
			releaseScope: AQLFenceScopeAgent,
			// type = 5 | barrier = 0 | acquire = 2 << 9 (0x400) | release = 1 << 11 (0x800) -> 0x0C05
			expectedHdr: 0x0C05,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hdr := BuildAQLHeader(tc.pktType, tc.barrier, tc.acquireScope, tc.releaseScope)
			if hdr != tc.expectedHdr {
				t.Fatalf("BuildAQLHeader() = 0x%04X, want 0x%04X", hdr, tc.expectedHdr)
			}

			pType, barrier, acq, rel := ParseAQLHeader(hdr)
			if pType != tc.pktType {
				t.Errorf("parsed packet type = %d, want %d", pType, tc.pktType)
			}
			if barrier != tc.barrier {
				t.Errorf("parsed barrier = %v, want %v", barrier, tc.barrier)
			}
			if acq != tc.acquireScope {
				t.Errorf("parsed acquireScope = %d, want %d", acq, tc.acquireScope)
			}
			if rel != tc.releaseScope {
				t.Errorf("parsed releaseScope = %d, want %d", rel, tc.releaseScope)
			}
		})
	}
}

// TestAQLKernelDispatchSerializationRoundTrip tests full fidelity marshalling and unmarshalling
// of a realistic compute dispatch packet.
func TestAQLKernelDispatchSerializationRoundTrip(t *testing.T) {
	orig := AQLKernelDispatchPacket{
		Header:             BuildAQLHeader(AQLPacketTypeKernelDispatch, true, AQLFenceScopeAgent, AQLFenceScopeAgent),
		Setup:              3, // 3D grid
		WorkgroupSizeX:     256,
		WorkgroupSizeY:     1,
		WorkgroupSizeZ:     1,
		Reserved0:          0,
		GridSizeX:          65536,
		GridSizeY:          1,
		GridSizeZ:          1,
		PrivateSegmentSize: 4096,
		GroupSegmentSize:   32768,
		KernelObject:       0x00007FFF80001000,
		KernargAddress:     0x00007FFF90000000,
		Reserved1:          0,
		CompletionSignal:   0x00007FFFA0000040,
	}

	raw, err := orig.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary failed: %v", err)
	}
	if len(raw) != 64 {
		t.Fatalf("raw packet length = %d; want 64", len(raw))
	}

	var decoded AQLKernelDispatchPacket
	if err := decoded.UnmarshalBinary(raw); err != nil {
		t.Fatalf("UnmarshalBinary failed: %v", err)
	}

	if decoded != orig {
		t.Fatalf("decoded packet does not match original:\ngot  %+v\nwant %+v", decoded, orig)
	}

	// Short buffer rejection
	if err := decoded.UnmarshalBinary(raw[:63]); err == nil {
		t.Fatal("expected error on unmarshalling short packet, got nil")
	}
}

// TestAQLBarrierSerializationRoundTrip tests Barrier-AND and Barrier-OR packet serialization.
func TestAQLBarrierSerializationRoundTrip(t *testing.T) {
	// Barrier-AND
	band := AQLBarrierAndPacket{
		Header:           BuildAQLHeader(AQLPacketTypeBarrierAnd, false, AQLFenceScopeSystem, AQLFenceScopeSystem),
		Reserved0:        0,
		Reserved1:        0,
		DepSignals:       [5]uint64{0x100, 0x200, 0x300, 0x400, 0x500},
		Reserved2:        0,
		CompletionSignal: 0x999,
	}
	data, err := band.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary BarrierAnd failed: %v", err)
	}
	var bandDecoded AQLBarrierAndPacket
	if err := bandDecoded.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary BarrierAnd failed: %v", err)
	}
	if bandDecoded != band {
		t.Fatalf("BarrierAnd mismatch: got %+v, want %+v", bandDecoded, band)
	}

	// Barrier-OR
	bor := AQLBarrierOrPacket{
		Header:           BuildAQLHeader(AQLPacketTypeBarrierOr, true, AQLFenceScopeAgent, AQLFenceScopeAgent),
		Reserved0:        0,
		Reserved1:        0,
		DepSignals:       [5]uint64{0x111, 0x222, 0, 0, 0},
		Reserved2:        0,
		CompletionSignal: 0x888,
	}
	dataOR, err := bor.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary BarrierOr failed: %v", err)
	}
	var borDecoded AQLBarrierOrPacket
	if err := borDecoded.UnmarshalBinary(dataOR); err != nil {
		t.Fatalf("UnmarshalBinary BarrierOr failed: %v", err)
	}
	if borDecoded != bor {
		t.Fatalf("BarrierOr mismatch: got %+v, want %+v", borDecoded, bor)
	}
}

// TestAQLQueueRingBuffer verifies the in-memory user-space queue ring buffer:
// power-of-2 capacity validation, contiguous 64-byte alignment, wraparound,
// and full-buffer backpressure.
func TestAQLQueueRingBuffer(t *testing.T) {
	// Capacity must be power of 2
	if _, err := NewAQLQueue(0); err == nil {
		t.Fatal("expected error for capacity 0")
	}
	if _, err := NewAQLQueue(3); err == nil {
		t.Fatal("expected error for capacity 3 (non-power-of-2)")
	}

	const cap = 4
	q, err := NewAQLQueue(cap)
	if err != nil {
		t.Fatalf("NewAQLQueue(4) failed: %v", err)
	}
	if q.Capacity() != cap {
		t.Fatalf("Capacity() = %d, want %d", q.Capacity(), cap)
	}
	if len(q.RingBytes()) != cap*64 {
		t.Fatalf("RingBytes length = %d, want %d", len(q.RingBytes()), cap*64)
	}

	// Submit 4 packets to fill the queue
	for i := uint64(0); i < cap; i++ {
		pkt := AQLKernelDispatchPacket{
			Header:       BuildAQLHeader(AQLPacketTypeKernelDispatch, false, AQLFenceScopeNone, AQLFenceScopeNone),
			GridSizeX:    uint32(100 + i),
			KernelObject: 0x1000 + i*0x100,
		}
		slot, err := q.SubmitKernelDispatch(pkt)
		if err != nil {
			t.Fatalf("SubmitKernelDispatch[%d] failed: %v", i, err)
		}
		if slot != i {
			t.Fatalf("slot index = %d, want %d", slot, i)
		}
	}

	if q.Size() != cap {
		t.Fatalf("queue size = %d, want %d", q.Size(), cap)
	}

	// 5th packet should fail with queue full
	overflowPkt := AQLKernelDispatchPacket{
		Header: BuildAQLHeader(AQLPacketTypeKernelDispatch, false, AQLFenceScopeNone, AQLFenceScopeNone),
	}
	if _, err := q.SubmitKernelDispatch(overflowPkt); err == nil {
		t.Fatal("expected error when submitting to full queue, got nil")
	}

	// Consume 2 packets
	q.AdvanceReadIndex(2)
	if q.Size() != 2 {
		t.Fatalf("queue size after consuming 2 = %d, want 2", q.Size())
	}

	// Now we can submit 2 more (testing ring wraparound)
	for i := uint64(0); i < 2; i++ {
		barrierPkt := AQLBarrierAndPacket{
			Header:     BuildAQLHeader(AQLPacketTypeBarrierAnd, true, AQLFenceScopeAgent, AQLFenceScopeAgent),
			DepSignals: [5]uint64{0x500 + i},
		}
		slot, err := q.SubmitBarrierAnd(barrierPkt)
		if err != nil {
			t.Fatalf("SubmitBarrierAnd failed during wraparound: %v", err)
		}
		expectedSlot := cap + i
		if slot != expectedSlot {
			t.Fatalf("slot index = %d, want %d", slot, expectedSlot)
		}
	}

	// Verify packet at wrapped slot
	pktBytes, err := q.PacketAt(cap)
	if err != nil {
		t.Fatalf("PacketAt(%d) failed: %v", cap, err)
	}
	var decodedBarrier AQLBarrierAndPacket
	if err := decodedBarrier.UnmarshalBinary(pktBytes); err != nil {
		t.Fatalf("UnmarshalBinary on PacketAt failed: %v", err)
	}
	if decodedBarrier.DepSignals[0] != 0x500 {
		t.Fatalf("wrapped packet DepSignal[0] = 0x%X, want 0x500", decodedBarrier.DepSignals[0])
	}
}

// TestPM4PacketStreamGenerationAndDecoding tests PM4 Type-3 packet stream creation
// and reverse decoding for all required microcode opcodes:
// IT_SET_SH_REG, IT_DISPATCH_DIRECT, IT_WAIT_REG_MEM, and IT_EVENT_WRITE.
func TestPM4PacketStreamGenerationAndDecoding(t *testing.T) {
	builder := NewPM4Builder()

	// 1. Configure shader registers (COMPUTE_PGM_LO/HI, COMPUTE_USER_DATA)
	const (
		regComputePgmLo = 0x2E00
		regValLo        = 0x80001000
		regValHi        = 0x00007FFF
		regKernargPtrLo = 0x90000000
		regKernargPtrHi = 0x00007FFF
	)
	builder.SetShReg(regComputePgmLo, regValLo, regValHi, regKernargPtrLo, regKernargPtrHi)

	// 2. Direct compute dispatch (128x1x1 workgroups)
	const (
		gridX     = 128
		gridY     = 1
		gridZ     = 1
		initiator = 1 // COMPUTE_SHADER_EN
	)
	builder.DispatchDirect(gridX, gridY, gridZ, initiator)

	// 3. Wait on memory completion flag (poll memory address until equal to ref)
	const (
		waitAddr     = 0x00007FFFA0000080
		refVal       = 0x1
		maskVal      = 0xFFFFFFFF
		pollInterval = 10
	)
	builder.WaitRegMem(
		WaitRegMemEngineME,
		WaitRegMemMemSpaceMem,
		WaitRegMemFuncEqual,
		waitAddr,
		refVal,
		maskVal,
		pollInterval,
	)

	// 4. Hardware event write (Cache flush and invalidate)
	builder.EventWrite(EventCacheFlushAndInvTS, 0)

	// Verify stream length in dwords
	// Packet 1 (SET_SH_REG): 1 hdr + 1 regOffset + 4 values = 6 dwords
	// Packet 2 (DISPATCH_DIRECT): 1 hdr + 4 body = 5 dwords
	// Packet 3 (WAIT_REG_MEM): 1 hdr + 6 body = 7 dwords
	// Packet 4 (EVENT_WRITE): 1 hdr + 1 body = 2 dwords
	// Total = 6 + 5 + 7 + 2 = 20 dwords (80 bytes)
	expectedDwords := 20
	if builder.Len() != expectedDwords {
		t.Fatalf("builder.Len() = %d, want %d", builder.Len(), expectedDwords)
	}

	byteStream := builder.Bytes()
	if len(byteStream) != expectedDwords*4 {
		t.Fatalf("byteStream len = %d, want %d", len(byteStream), expectedDwords*4)
	}

	// Decode the command stream
	packets, err := DecodePM4(builder.Dwords())
	if err != nil {
		t.Fatalf("DecodePM4 failed: %v", err)
	}
	if len(packets) != 4 {
		t.Fatalf("decoded %d packets; want 4", len(packets))
	}

	// Verify Packet 1: IT_SET_SH_REG
	p1 := packets[0]
	if p1.Opcode != IT_SET_SH_REG {
		t.Errorf("p1 opcode = 0x%02X, want IT_SET_SH_REG (0x%02X)", p1.Opcode, IT_SET_SH_REG)
	}
	// count = body dwords - 1 = 5 - 1 = 4
	if p1.Count != 4 {
		t.Errorf("p1 count = %d, want 4", p1.Count)
	}
	if p1.Payload[0] != regComputePgmLo || p1.Payload[1] != regValLo || p1.Payload[2] != regValHi {
		t.Errorf("p1 payload register values unexpected: %+v", p1.Payload)
	}

	// Verify Packet 2: IT_DISPATCH_DIRECT
	p2 := packets[1]
	if p2.Opcode != IT_DISPATCH_DIRECT {
		t.Errorf("p2 opcode = 0x%02X, want IT_DISPATCH_DIRECT (0x%02X)", p2.Opcode, IT_DISPATCH_DIRECT)
	}
	if p2.Count != 3 {
		t.Errorf("p2 count = %d, want 3", p2.Count)
	}
	if p2.Payload[0] != gridX || p2.Payload[1] != gridY || p2.Payload[2] != gridZ || p2.Payload[3] != initiator {
		t.Errorf("p2 dispatch dimensions unexpected: %+v", p2.Payload)
	}

	// Verify Packet 3: IT_WAIT_REG_MEM
	p3 := packets[2]
	if p3.Opcode != IT_WAIT_REG_MEM {
		t.Errorf("p3 opcode = 0x%02X, want IT_WAIT_REG_MEM (0x%02X)", p3.Opcode, IT_WAIT_REG_MEM)
	}
	if p3.Count != 5 {
		t.Errorf("p3 count = %d, want 5", p3.Count)
	}
	expectedAddrLo := uint32(waitAddr & 0xFFFFFFFF)
	expectedAddrHi := uint32((waitAddr >> 32) & 0xFFFFFFFF)
	if p3.Payload[1] != expectedAddrLo || p3.Payload[2] != expectedAddrHi || p3.Payload[3] != refVal {
		t.Errorf("p3 wait parameters unexpected: %+v", p3.Payload)
	}

	// Verify Packet 4: IT_EVENT_WRITE
	p4 := packets[3]
	if p4.Opcode != IT_EVENT_WRITE {
		t.Errorf("p4 opcode = 0x%02X, want IT_EVENT_WRITE (0x%02X)", p4.Opcode, IT_EVENT_WRITE)
	}
	if p4.Count != 0 {
		t.Errorf("p4 count = %d, want 0", p4.Count)
	}
	if (p4.Payload[0] & 0xFF) != EventCacheFlushAndInvTS {
		t.Errorf("p4 event type = 0x%02X, want 0x%02X", p4.Payload[0]&0xFF, EventCacheFlushAndInvTS)
	}

	// Test DecodePM4Bytes directly
	packetsFromBytes, err := DecodePM4Bytes(byteStream)
	if err != nil {
		t.Fatalf("DecodePM4Bytes failed: %v", err)
	}
	if len(packetsFromBytes) != 4 {
		t.Fatalf("decoded %d packets from bytes; want 4", len(packetsFromBytes))
	}
}

// TestPM4DecodingErrors verifies error detection on corrupted PM4 streams.
func TestPM4DecodingErrors(t *testing.T) {
	// 1. Invalid Type-3 header (not starting with 0xC0000000)
	invalidHeader := []uint32{0x80000000, 0x0}
	if _, err := DecodePM4(invalidHeader); err == nil {
		t.Fatal("expected error on invalid header mask, got nil")
	}

	// 2. Truncated packet
	truncated := []uint32{
		PM4Type3Header(IT_SET_SH_REG, 4), // claims 5 body dwords
		0x2E00,                           // only 1 dword supplied
	}
	if _, err := DecodePM4(truncated); err == nil {
		t.Fatal("expected error on truncated packet stream, got nil")
	}

	// 3. Byte slice not multiple of 4
	if _, err := DecodePM4Bytes([]byte{1, 2, 3}); err == nil {
		t.Fatal("expected error on unaligned byte slice, got nil")
	}
}

// TestAQLQueueCommitOrdering ensures payload bytes are intact before header commit.
func TestAQLQueueCommitOrdering(t *testing.T) {
	q, err := NewAQLQueue(2)
	if err != nil {
		t.Fatalf("NewAQLQueue failed: %v", err)
	}

	kPkt := AQLKernelDispatchPacket{
		Header:       BuildAQLHeader(AQLPacketTypeKernelDispatch, false, AQLFenceScopeNone, AQLFenceScopeNone),
		GridSizeX:    1024,
		KernelObject: 0xDEADBEEFCAFE,
	}

	slot, err := q.SubmitKernelDispatch(kPkt)
	if err != nil {
		t.Fatalf("SubmitKernelDispatch failed: %v", err)
	}

	raw, err := q.PacketAt(slot)
	if err != nil {
		t.Fatalf("PacketAt failed: %v", err)
	}

	var readBack AQLKernelDispatchPacket
	if err := readBack.UnmarshalBinary(raw); err != nil {
		t.Fatalf("UnmarshalBinary failed: %v", err)
	}

	if readBack.GridSizeX != 1024 || readBack.KernelObject != 0xDEADBEEFCAFE {
		t.Fatalf("read-back packet data mismatch: %+v", readBack)
	}

	// Ensure header was set
	pType, _, _, _ := ParseAQLHeader(readBack.Header)
	if pType != AQLPacketTypeKernelDispatch {
		t.Fatalf("packet type = %d, want %d", pType, AQLPacketTypeKernelDispatch)
	}

	// Test Reset
	q.Reset()
	if q.Size() != 0 || q.WriteIndex() != 0 || q.ReadIndex() != 0 {
		t.Fatalf("Reset did not clear indices: size=%d, w=%d, r=%d", q.Size(), q.WriteIndex(), q.ReadIndex())
	}
	empty := make([]byte, len(q.RingBytes()))
	if !bytes.Equal(q.RingBytes(), empty) {
		t.Fatal("Reset did not zero the ring buffer memory")
	}
}
