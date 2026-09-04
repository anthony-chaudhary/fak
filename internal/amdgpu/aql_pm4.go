// Package amdgpu provides AMD GPU facts probing, hardware governor settings,
// Strix Halo APU operational serving profiles, direct AQL/PM4 packet dispatch,
// and native HSACO code-object emission.
package amdgpu

import (
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"
)

// AQL packet constants matching the AMD HSA System Architecture Specification.
const (
	// AQLPacketSize is the byte-exact size of every AQL packet (64 bytes / 512 bits).
	AQLPacketSize = 64

	// AQLPacketAlignment is the memory alignment requirement for AQL packets.
	AQLPacketAlignment = 64
)

// AQLPacketType identifies the purpose and payload layout of an AQL packet.
type AQLPacketType uint8

const (
	AQLPacketTypeVendorSpecific AQLPacketType = 0
	AQLPacketTypeInvalid        AQLPacketType = 1
	AQLPacketTypeKernelDispatch AQLPacketType = 2
	AQLPacketTypeBarrierAnd     AQLPacketType = 3
	AQLPacketTypeAgentDispatch  AQLPacketType = 4
	AQLPacketTypeBarrierOr      AQLPacketType = 5
)

// AQLFenceScope controls memory fence synchronization breadth.
type AQLFenceScope uint8

const (
	AQLFenceScopeNone   AQLFenceScope = 0
	AQLFenceScopeAgent  AQLFenceScope = 1
	AQLFenceScopeSystem AQLFenceScope = 2
)

// AQL Header bitfield masks and shifts.
const (
	AQLHeaderTypeMask          uint16 = 0x00FF
	AQLHeaderTypeShift                = 0
	AQLHeaderBarrierBit        uint16 = 1 << 8
	AQLHeaderAcquireFenceMask  uint16 = 0x0003
	AQLHeaderAcquireFenceShift        = 9
	AQLHeaderReleaseFenceMask  uint16 = 0x0003
	AQLHeaderReleaseFenceShift        = 11
)

// BuildAQLHeader packs packet type, barrier flag, and fence scopes into a 16-bit AQL header.
func BuildAQLHeader(pktType AQLPacketType, barrier bool, acquireScope, releaseScope AQLFenceScope) uint16 {
	var h uint16 = uint16(pktType) & AQLHeaderTypeMask
	if barrier {
		h |= AQLHeaderBarrierBit
	}
	h |= (uint16(acquireScope) & AQLHeaderAcquireFenceMask) << AQLHeaderAcquireFenceShift
	h |= (uint16(releaseScope) & AQLHeaderReleaseFenceMask) << AQLHeaderReleaseFenceShift
	return h
}

// ParseAQLHeader extracts packet type, barrier flag, and fence scopes from a 16-bit AQL header.
func ParseAQLHeader(header uint16) (pktType AQLPacketType, barrier bool, acquireScope, releaseScope AQLFenceScope) {
	pktType = AQLPacketType(header & AQLHeaderTypeMask)
	barrier = (header & AQLHeaderBarrierBit) != 0
	acquireScope = AQLFenceScope((header >> AQLHeaderAcquireFenceShift) & AQLHeaderAcquireFenceMask)
	releaseScope = AQLFenceScope((header >> AQLHeaderReleaseFenceShift) & AQLHeaderReleaseFenceMask)
	return
}

// AQLKernelDispatchPacket matches the AMD HSA ABI hsa_kernel_dispatch_packet_t layout.
// It is exactly 64 bytes and must be 64-byte aligned in user-space queue memory.
type AQLKernelDispatchPacket struct {
	Header             uint16 // Packet type, barrier bit, acquire/release fence scopes
	Setup              uint16 // Dimensions (1, 2, or 3) in bits 0..1
	WorkgroupSizeX     uint16 // Workgroup size X dimension (1..65535)
	WorkgroupSizeY     uint16 // Workgroup size Y dimension (1..65535)
	WorkgroupSizeZ     uint16 // Workgroup size Z dimension (1..65535)
	Reserved0          uint16 // Reserved for alignment / future expansion
	GridSizeX          uint32 // Grid size X dimension in workitems
	GridSizeY          uint32 // Grid size Y dimension in workitems
	GridSizeZ          uint32 // Grid size Z dimension in workitems
	PrivateSegmentSize uint32 // Per-workitem private (scratch) memory in bytes
	GroupSegmentSize   uint32 // Per-workgroup group (LDS) memory in bytes
	KernelObject       uint64 // GPU virtual address of kernel code / descriptor
	KernargAddress     uint64 // GPU virtual address of kernel arguments buffer
	Reserved1          uint64 // Reserved for runtime
	CompletionSignal   uint64 // Completion signal handle (hsa_signal_t), 0 if none
}

// Ensure struct size is statically 64 bytes.
var _ [64]byte = [unsafe.Sizeof(AQLKernelDispatchPacket{})]byte{}

// MarshalBinary encodes the kernel dispatch packet into a 64-byte slice in little-endian format.
func (p *AQLKernelDispatchPacket) MarshalBinary() ([]byte, error) {
	buf := make([]byte, AQLPacketSize)
	p.Serialize(buf)
	return buf, nil
}

// Marshal64 encodes the kernel dispatch packet into a fixed 64-byte array in little-endian format.
func (p *AQLKernelDispatchPacket) Marshal64() [64]byte {
	var buf [64]byte
	p.Serialize(buf[:])
	return buf
}

// Serialize writes the packet directly into the destination buffer (at least 64 bytes).
func (p *AQLKernelDispatchPacket) Serialize(buf []byte) {
	if len(buf) < AQLPacketSize {
		panic(fmt.Sprintf("amdgpu: destination buffer too small (%d < 64)", len(buf)))
	}
	binary.LittleEndian.PutUint16(buf[0:2], p.Header)
	binary.LittleEndian.PutUint16(buf[2:4], p.Setup)
	binary.LittleEndian.PutUint16(buf[4:6], p.WorkgroupSizeX)
	binary.LittleEndian.PutUint16(buf[6:8], p.WorkgroupSizeY)
	binary.LittleEndian.PutUint16(buf[8:10], p.WorkgroupSizeZ)
	binary.LittleEndian.PutUint16(buf[10:12], p.Reserved0)
	binary.LittleEndian.PutUint32(buf[12:16], p.GridSizeX)
	binary.LittleEndian.PutUint32(buf[16:20], p.GridSizeY)
	binary.LittleEndian.PutUint32(buf[20:24], p.GridSizeZ)
	binary.LittleEndian.PutUint32(buf[24:28], p.PrivateSegmentSize)
	binary.LittleEndian.PutUint32(buf[28:32], p.GroupSegmentSize)
	binary.LittleEndian.PutUint64(buf[32:40], p.KernelObject)
	binary.LittleEndian.PutUint64(buf[40:48], p.KernargAddress)
	binary.LittleEndian.PutUint64(buf[48:56], p.Reserved1)
	binary.LittleEndian.PutUint64(buf[56:64], p.CompletionSignal)
}

// UnmarshalBinary decodes a 64-byte little-endian slice into the packet.
func (p *AQLKernelDispatchPacket) UnmarshalBinary(data []byte) error {
	if len(data) < AQLPacketSize {
		return fmt.Errorf("amdgpu: invalid AQL packet length %d (expected %d)", len(data), AQLPacketSize)
	}
	p.Header = binary.LittleEndian.Uint16(data[0:2])
	p.Setup = binary.LittleEndian.Uint16(data[2:4])
	p.WorkgroupSizeX = binary.LittleEndian.Uint16(data[4:6])
	p.WorkgroupSizeY = binary.LittleEndian.Uint16(data[6:8])
	p.WorkgroupSizeZ = binary.LittleEndian.Uint16(data[8:10])
	p.Reserved0 = binary.LittleEndian.Uint16(data[10:12])
	p.GridSizeX = binary.LittleEndian.Uint32(data[12:16])
	p.GridSizeY = binary.LittleEndian.Uint32(data[16:20])
	p.GridSizeZ = binary.LittleEndian.Uint32(data[20:24])
	p.PrivateSegmentSize = binary.LittleEndian.Uint32(data[24:28])
	p.GroupSegmentSize = binary.LittleEndian.Uint32(data[28:32])
	p.KernelObject = binary.LittleEndian.Uint64(data[32:40])
	p.KernargAddress = binary.LittleEndian.Uint64(data[40:48])
	p.Reserved1 = binary.LittleEndian.Uint64(data[48:56])
	p.CompletionSignal = binary.LittleEndian.Uint64(data[56:64])
	return nil
}

// AQLBarrierAndPacket matches the AMD HSA ABI hsa_barrier_and_packet_t layout.
// It synchronizes completion of up to 5 dependent signals before unblocking downstream dispatch.
type AQLBarrierAndPacket struct {
	Header           uint16    // Packet type (AQLPacketTypeBarrierAnd), barrier flag, fences
	Reserved0        uint16    // Reserved
	Reserved1        uint32    // Reserved
	DepSignals       [5]uint64 // Up to 5 input dependency signals to wait on
	Reserved2        uint64    // Reserved
	CompletionSignal uint64    // Signal to notify upon barrier resolution
}

// Ensure struct size is statically 64 bytes.
var _ [64]byte = [unsafe.Sizeof(AQLBarrierAndPacket{})]byte{}

// MarshalBinary encodes the Barrier-AND packet into a 64-byte slice in little-endian format.
func (p *AQLBarrierAndPacket) MarshalBinary() ([]byte, error) {
	buf := make([]byte, AQLPacketSize)
	p.Serialize(buf)
	return buf, nil
}

// Marshal64 encodes the Barrier-AND packet into a fixed 64-byte array.
func (p *AQLBarrierAndPacket) Marshal64() [64]byte {
	var buf [64]byte
	p.Serialize(buf[:])
	return buf
}

// Serialize writes the Barrier-AND packet directly into destination buffer.
func (p *AQLBarrierAndPacket) Serialize(buf []byte) {
	if len(buf) < AQLPacketSize {
		panic(fmt.Sprintf("amdgpu: destination buffer too small (%d < 64)", len(buf)))
	}
	binary.LittleEndian.PutUint16(buf[0:2], p.Header)
	binary.LittleEndian.PutUint16(buf[2:4], p.Reserved0)
	binary.LittleEndian.PutUint32(buf[4:8], p.Reserved1)
	for i := 0; i < 5; i++ {
		offset := 8 + i*8
		binary.LittleEndian.PutUint64(buf[offset:offset+8], p.DepSignals[i])
	}
	binary.LittleEndian.PutUint64(buf[48:56], p.Reserved2)
	binary.LittleEndian.PutUint64(buf[56:64], p.CompletionSignal)
}

// UnmarshalBinary decodes a 64-byte little-endian slice into the Barrier-AND packet.
func (p *AQLBarrierAndPacket) UnmarshalBinary(data []byte) error {
	if len(data) < AQLPacketSize {
		return fmt.Errorf("amdgpu: invalid AQL packet length %d (expected %d)", len(data), AQLPacketSize)
	}
	p.Header = binary.LittleEndian.Uint16(data[0:2])
	p.Reserved0 = binary.LittleEndian.Uint16(data[2:4])
	p.Reserved1 = binary.LittleEndian.Uint32(data[4:8])
	for i := 0; i < 5; i++ {
		offset := 8 + i*8
		p.DepSignals[i] = binary.LittleEndian.Uint64(data[offset : offset+8])
	}
	p.Reserved2 = binary.LittleEndian.Uint64(data[48:56])
	p.CompletionSignal = binary.LittleEndian.Uint64(data[56:64])
	return nil
}

// AQLBarrierOrPacket matches the AMD HSA ABI hsa_barrier_or_packet_t layout.
// It resolves as soon as ANY of the active dependent signals fires.
type AQLBarrierOrPacket struct {
	Header           uint16    // Packet type (AQLPacketTypeBarrierOr), barrier flag, fences
	Reserved0        uint16    // Reserved
	Reserved1        uint32    // Reserved
	DepSignals       [5]uint64 // Up to 5 input dependency signals
	Reserved2        uint64    // Reserved
	CompletionSignal uint64    // Signal to notify upon barrier resolution
}

// Ensure struct size is statically 64 bytes.
var _ [64]byte = [unsafe.Sizeof(AQLBarrierOrPacket{})]byte{}

// MarshalBinary encodes the Barrier-OR packet into a 64-byte slice in little-endian format.
func (p *AQLBarrierOrPacket) MarshalBinary() ([]byte, error) {
	buf := make([]byte, AQLPacketSize)
	p.Serialize(buf)
	return buf, nil
}

// Marshal64 encodes the Barrier-OR packet into a fixed 64-byte array.
func (p *AQLBarrierOrPacket) Marshal64() [64]byte {
	var buf [64]byte
	p.Serialize(buf[:])
	return buf
}

// Serialize writes the Barrier-OR packet directly into destination buffer.
func (p *AQLBarrierOrPacket) Serialize(buf []byte) {
	if len(buf) < AQLPacketSize {
		panic(fmt.Sprintf("amdgpu: destination buffer too small (%d < 64)", len(buf)))
	}
	binary.LittleEndian.PutUint16(buf[0:2], p.Header)
	binary.LittleEndian.PutUint16(buf[2:4], p.Reserved0)
	binary.LittleEndian.PutUint32(buf[4:8], p.Reserved1)
	for i := 0; i < 5; i++ {
		offset := 8 + i*8
		binary.LittleEndian.PutUint64(buf[offset:offset+8], p.DepSignals[i])
	}
	binary.LittleEndian.PutUint64(buf[48:56], p.Reserved2)
	binary.LittleEndian.PutUint64(buf[56:64], p.CompletionSignal)
}

// UnmarshalBinary decodes a 64-byte little-endian slice into the Barrier-OR packet.
func (p *AQLBarrierOrPacket) UnmarshalBinary(data []byte) error {
	if len(data) < AQLPacketSize {
		return fmt.Errorf("amdgpu: invalid AQL packet length %d (expected %d)", len(data), AQLPacketSize)
	}
	p.Header = binary.LittleEndian.Uint16(data[0:2])
	p.Reserved0 = binary.LittleEndian.Uint16(data[2:4])
	p.Reserved1 = binary.LittleEndian.Uint32(data[4:8])
	for i := 0; i < 5; i++ {
		offset := 8 + i*8
		p.DepSignals[i] = binary.LittleEndian.Uint64(data[offset : offset+8])
	}
	p.Reserved2 = binary.LittleEndian.Uint64(data[48:56])
	p.CompletionSignal = binary.LittleEndian.Uint64(data[56:64])
	return nil
}

// AQLQueue is an in-memory ring buffer serializer writing contiguous 64-byte aligned packets
// directly into user-space HSA queue memory.
type AQLQueue struct {
	mu         sync.RWMutex
	capacity   uint64 // Number of packet slots (must be a power of 2)
	ring       []byte // Backing memory buffer of size capacity * AQLPacketSize
	writeIndex uint64 // Monotonically increasing submission index
	readIndex  uint64 // Monotonically increasing consumer index
}

// NewAQLQueue allocates an in-memory AQL queue ring buffer with the specified packet capacity.
// Capacity must be a non-zero power of 2 (e.g. 64, 128, 512, 1024).
func NewAQLQueue(capacity uint64) (*AQLQueue, error) {
	if capacity == 0 || (capacity&(capacity-1)) != 0 {
		return nil, fmt.Errorf("amdgpu: queue capacity (%d) must be a non-zero power of 2", capacity)
	}
	return &AQLQueue{
		capacity: capacity,
		ring:     make([]byte, capacity*AQLPacketSize),
	}, nil
}

// Capacity returns the total number of 64-byte packet slots in the queue.
func (q *AQLQueue) Capacity() uint64 {
	return q.capacity
}

// Size returns the count of unconsumed packets currently residing in the queue.
func (q *AQLQueue) Size() uint64 {
	q.mu.RLock()
	defer q.mu.RUnlock()
	w := atomic.LoadUint64(&q.writeIndex)
	r := atomic.LoadUint64(&q.readIndex)
	if w >= r {
		return w - r
	}
	return 0
}

// WriteIndex returns the current monotonic write index.
func (q *AQLQueue) WriteIndex() uint64 {
	return atomic.LoadUint64(&q.writeIndex)
}

// ReadIndex returns the current monotonic read index.
func (q *AQLQueue) ReadIndex() uint64 {
	return atomic.LoadUint64(&q.readIndex)
}

// SetReadIndex updates the consumer read index.
func (q *AQLQueue) SetReadIndex(idx uint64) {
	atomic.StoreUint64(&q.readIndex, idx)
}

// AdvanceReadIndex increments the consumer read index by n.
func (q *AQLQueue) AdvanceReadIndex(n uint64) {
	atomic.AddUint64(&q.readIndex, n)
}

// SubmitPacket writes a 64-byte packet to the next available ring slot.
// Following the HSA ring buffer protocol, payload bytes (2..63) are written
// before the header (0..1) so the hardware command processor never sees a
// partially written packet.
func (q *AQLQueue) SubmitPacket(packetData []byte) (uint64, error) {
	if len(packetData) != AQLPacketSize {
		return 0, fmt.Errorf("amdgpu: packet data length (%d) does not match AQL packet size (64)", len(packetData))
	}
	q.mu.Lock()
	defer q.mu.Unlock()

	w := atomic.LoadUint64(&q.writeIndex)
	r := atomic.LoadUint64(&q.readIndex)
	if w-r >= q.capacity {
		return 0, errors.New("amdgpu: AQL queue ring buffer is full")
	}

	slot := w & (q.capacity - 1)
	slotOffset := slot * AQLPacketSize

	// Step 1: Write packet payload (bytes 2..63)
	copy(q.ring[slotOffset+2:slotOffset+AQLPacketSize], packetData[2:AQLPacketSize])

	// Step 2: Write packet header (bytes 0..1) as the commit action
	binary.LittleEndian.PutUint16(q.ring[slotOffset:slotOffset+2], binary.LittleEndian.Uint16(packetData[0:2]))

	atomic.AddUint64(&q.writeIndex, 1)
	return w, nil
}

// SubmitKernelDispatch writes an AQLKernelDispatchPacket into the queue.
func (q *AQLQueue) SubmitKernelDispatch(pkt AQLKernelDispatchPacket) (uint64, error) {
	data := pkt.Marshal64()
	return q.SubmitPacket(data[:])
}

// SubmitBarrierAnd writes an AQLBarrierAndPacket into the queue.
func (q *AQLQueue) SubmitBarrierAnd(pkt AQLBarrierAndPacket) (uint64, error) {
	data := pkt.Marshal64()
	return q.SubmitPacket(data[:])
}

// SubmitBarrierOr writes an AQLBarrierOrPacket into the queue.
func (q *AQLQueue) SubmitBarrierOr(pkt AQLBarrierOrPacket) (uint64, error) {
	data := pkt.Marshal64()
	return q.SubmitPacket(data[:])
}

// PacketAt returns a copy of the 64-byte packet at the given slot index.
func (q *AQLQueue) PacketAt(index uint64) ([]byte, error) {
	q.mu.RLock()
	defer q.mu.RUnlock()

	slot := index & (q.capacity - 1)
	slotOffset := slot * AQLPacketSize
	pkt := make([]byte, AQLPacketSize)
	copy(pkt, q.ring[slotOffset:slotOffset+AQLPacketSize])
	return pkt, nil
}

// RingBytes returns the underlying ring buffer byte slice.
func (q *AQLQueue) RingBytes() []byte {
	q.mu.RLock()
	defer q.mu.RUnlock()
	return q.ring
}

// Reset clears the queue and resets the write and read indices to zero.
func (q *AQLQueue) Reset() {
	q.mu.Lock()
	defer q.mu.Unlock()
	for i := range q.ring {
		q.ring[i] = 0
	}
	atomic.StoreUint64(&q.writeIndex, 0)
	atomic.StoreUint64(&q.readIndex, 0)
}

// PM4 (Packet Microcode 4) Command Processor Packet Specifications.
// PM4 Type-3 packets are written directly to AMD GPU Command Processor (CP) ring buffers.
const (
	// PM4Type3HeaderMask is the bitmask identifying a Type-3 PM4 packet (bits 31:30 = 0b11).
	PM4Type3HeaderMask uint32 = 0xC0000000

	// Common AMD PM4 Type-3 IT Opcodes.
	IT_NOP               uint8 = 0x10 // No-operation
	IT_SET_SH_REG        uint8 = 0x76 // Set shader registers (118)
	IT_SET_CONFIG_REG    uint8 = 0x68 // Set config registers (104)
	IT_SET_CONTEXT_REG   uint8 = 0x69 // Set context registers (105)
	IT_DISPATCH_DIRECT   uint8 = 0x15 // Direct compute dispatch (21)
	IT_DISPATCH_INDIRECT uint8 = 0x16 // Indirect compute dispatch (22)
	IT_WAIT_REG_MEM      uint8 = 0x3C // Wait on register or memory condition (60)
	IT_WRITE_DATA        uint8 = 0x37 // Write data to memory or register (55)
	IT_EVENT_WRITE       uint8 = 0x46 // Send event to hardware block (70)
	IT_EVENT_WRITE_EOP   uint8 = 0x47 // End-of-pipe event write (71)
	IT_RELEASE_MEM       uint8 = 0x49 // Release memory fence/flush (73)
	IT_ACQUIRE_MEM       uint8 = 0x58 // Acquire memory fence/invalidate (88)
)

// IT_WAIT_REG_MEM function conditions.
const (
	WaitRegMemFuncAlways       uint32 = 0
	WaitRegMemFuncLessThan     uint32 = 1
	WaitRegMemFuncLessEqual    uint32 = 2
	WaitRegMemFuncEqual        uint32 = 3
	WaitRegMemFuncNotEqual     uint32 = 4
	WaitRegMemFuncGreaterEqual uint32 = 5
	WaitRegMemFuncGreaterThan  uint32 = 6

	WaitRegMemMemSpaceReg uint32 = 0 // Polling register
	WaitRegMemMemSpaceMem uint32 = 1 // Polling memory location

	WaitRegMemEngineME  uint32 = 0 // Micro-Engine
	WaitRegMemEnginePFP uint32 = 1 // Prefetch Parser
)

// Common hardware event types for IT_EVENT_WRITE.
const (
	EventSampleStreamoutstats uint32 = 0x01
	EventSamplePipelinestats  uint32 = 0x02
	EventCacheFlushTS         uint32 = 0x04
	EventCSPartialFlush       uint32 = 0x07
	EventBottomOfPipeTS       uint32 = 0x14
	EventCacheFlushAndInvTS   uint32 = 0x14
)

// PM4Type3Header builds a 32-bit Type-3 packet header according to the AMD PM4 specification:
// 0xC0000000 | ((count & 0x3FFF) << 16) | ((it_opcode & 0xFF) << 8).
// In PM4 Type-3, count represents the number of DWORDs following the header MINUS 1.
func PM4Type3Header(itOpcode uint8, count uint16) uint32 {
	return PM4Type3HeaderMask | ((uint32(count) & 0x3FFF) << 16) | ((uint32(itOpcode) & 0xFF) << 8)
}

// PM4Builder constructs contiguous PM4 command streams for GPU Command Processor submission.
type PM4Builder struct {
	dwords []uint32
}

// NewPM4Builder initializes an empty PM4 command packet builder.
func NewPM4Builder() *PM4Builder {
	return &PM4Builder{
		dwords: make([]uint32, 0, 64),
	}
}

// EmitRaw appends raw DWORDs to the command stream.
func (b *PM4Builder) EmitRaw(dwords ...uint32) *PM4Builder {
	b.dwords = append(b.dwords, dwords...)
	return b
}

// EmitPacket appends a complete Type-3 PM4 packet with the given opcode and body DWORDs.
func (b *PM4Builder) EmitPacket(opcode uint8, body ...uint32) *PM4Builder {
	if len(body) == 0 {
		return b
	}
	count := uint16(len(body) - 1)
	header := PM4Type3Header(opcode, count)
	b.dwords = append(b.dwords, header)
	b.dwords = append(b.dwords, body...)
	return b
}

// SetShReg builds an IT_SET_SH_REG packet to configure shader registers.
// regOffset is the target register address offset, followed by one or more register values.
func (b *PM4Builder) SetShReg(regOffset uint32, values ...uint32) *PM4Builder {
	if len(values) == 0 {
		return b
	}
	body := make([]uint32, 1+len(values))
	body[0] = regOffset
	copy(body[1:], values)
	return b.EmitPacket(IT_SET_SH_REG, body...)
}

// DispatchDirect builds an IT_DISPATCH_DIRECT packet for launching a 3D grid of workgroups.
// dimX, dimY, dimZ are the grid dimensions; initiator provides CP dispatch control flags.
func (b *PM4Builder) DispatchDirect(dimX, dimY, dimZ uint32, initiator uint32) *PM4Builder {
	return b.EmitPacket(IT_DISPATCH_DIRECT, dimX, dimY, dimZ, initiator)
}

// WaitRegMem builds an IT_WAIT_REG_MEM packet to pause CP execution until a memory or register
// condition is satisfied.
func (b *PM4Builder) WaitRegMem(engine, memSpace, function uint32, addr uint64, ref, mask, pollInterval uint32) *PM4Builder {
	dw0 := (function & 0x07) | ((memSpace & 0x01) << 4) | ((engine & 0x01) << 8)
	addrLo := uint32(addr & 0xFFFFFFFF)
	addrHi := uint32((addr >> 32) & 0xFFFFFFFF)
	return b.EmitPacket(IT_WAIT_REG_MEM, dw0, addrLo, addrHi, ref, mask, pollInterval)
}

// EventWrite builds an IT_EVENT_WRITE packet to trigger hardware cache flushes, timestamps, or pipe sync.
func (b *PM4Builder) EventWrite(eventType, eventIndex uint32) *PM4Builder {
	dw0 := (eventType & 0xFF) | ((eventIndex & 0x0F) << 8)
	return b.EmitPacket(IT_EVENT_WRITE, dw0)
}

// Dwords returns the constructed PM4 command stream as a slice of 32-bit DWORDs.
func (b *PM4Builder) Dwords() []uint32 {
	out := make([]uint32, len(b.dwords))
	copy(out, b.dwords)
	return out
}

// Bytes returns the command stream encoded as little-endian bytes.
func (b *PM4Builder) Bytes() []byte {
	buf := make([]byte, len(b.dwords)*4)
	for i, dw := range b.dwords {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], dw)
	}
	return buf
}

// Len returns the current length of the command stream in 32-bit DWORDs.
func (b *PM4Builder) Len() int {
	return len(b.dwords)
}

// Reset clears the builder buffer for reuse.
func (b *PM4Builder) Reset() {
	b.dwords = b.dwords[:0]
}

// PM4Packet represents a parsed PM4 Type-3 packet extracted from a command stream.
type PM4Packet struct {
	Header  uint32   // Raw Type-3 header DWORD
	Opcode  uint8    // IT opcode (bits 8..15)
	Count   uint16   // DWORD count minus 1 (bits 16..29)
	Payload []uint32 // Body DWORDs following the header
}

// DecodePM4 parses a slice of 32-bit DWORDs into individual PM4 Type-3 packets.
func DecodePM4(dwords []uint32) ([]PM4Packet, error) {
	var packets []PM4Packet
	idx := 0
	for idx < len(dwords) {
		hdr := dwords[idx]
		if (hdr & PM4Type3HeaderMask) != PM4Type3HeaderMask {
			return nil, fmt.Errorf("amdgpu: invalid PM4 Type-3 header 0x%08X at dword index %d", hdr, idx)
		}
		opcode := uint8((hdr >> 8) & 0xFF)
		count := uint16((hdr >> 16) & 0x3FFF)
		payloadLen := int(count) + 1

		if idx+1+payloadLen > len(dwords) {
			return nil, fmt.Errorf("amdgpu: truncated PM4 packet at index %d (needs %d dwords, available %d)",
				idx, 1+payloadLen, len(dwords)-idx)
		}

		payload := make([]uint32, payloadLen)
		copy(payload, dwords[idx+1:idx+1+payloadLen])

		packets = append(packets, PM4Packet{
			Header:  hdr,
			Opcode:  opcode,
			Count:   count,
			Payload: payload,
		})
		idx += 1 + payloadLen
	}
	return packets, nil
}

// DecodePM4Bytes parses a little-endian byte slice into individual PM4 Type-3 packets.
func DecodePM4Bytes(data []byte) ([]PM4Packet, error) {
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("amdgpu: PM4 byte stream length (%d) is not a multiple of 4", len(data))
	}
	dwords := make([]uint32, len(data)/4)
	for i := 0; i < len(dwords); i++ {
		dwords[i] = binary.LittleEndian.Uint32(data[i*4 : (i+1)*4])
	}
	return DecodePM4(dwords)
}
