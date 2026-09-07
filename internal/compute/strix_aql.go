// Package compute implements hardware abstraction, tensor computation, memory slab management,
// and zero-copy device interconnect acceleration for the fak agent kernel.
package compute

import (
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"
)

// HSA Kernel Dispatch Packet and AQL Constants.
const (
	// HSAPacketBytes is the exact 64-byte size of an architected AQL packet.
	HSAPacketBytes = 64

	// HSA Default and Maximum Queue Sizes for Strix Halo.
	DefaultStrixAQLQueueSize uint32 = 1024
	MaxStrixAQLQueueSize     uint32 = 65536

	// HSA Packet Types (bits [7:0] of Header).
	HSAPacketTypeVendorSpecific uint16 = 0
	HSAPacketTypeInvalid        uint16 = 1
	HSAPacketTypeKernelDispatch uint16 = 2
	HSAPacketTypeBarrierAnd     uint16 = 3
	HSAPacketTypeAgentDispatch  uint16 = 4
	HSAPacketTypeBarrierOr      uint16 = 5

	// HSA Header bitfield masks and shifts.
	HSAHeaderTypeShift    = 0
	HSAHeaderTypeMask     = 0xFF
	HSAHeaderBarrierBit   = 1 << 8
	HSAHeaderAcquireShift = 9
	HSAHeaderAcquireMask  = 0x3
	HSAHeaderReleaseShift = 11
	HSAHeaderReleaseMask  = 0x3

	// HSA Memory Fence Scopes.
	HSAFenceScopeNone   uint16 = 0
	HSAFenceScopeAgent  uint16 = 1
	HSAFenceScopeSystem uint16 = 2
)

// Strix AQL Queue errors.
var (
	// ErrAQLQueueFull indicates the AQL ring buffer capacity is exhausted.
	ErrAQLQueueFull = errors.New("compute/strix_aql: AQL user-mode queue ring buffer full")

	// ErrInvalidPacketSize indicates provided byte slice is not 64 bytes.
	ErrInvalidPacketSize = errors.New("compute/strix_aql: packet data must be at least 64 bytes")

	// ErrInvalidQueueSize indicates queue capacity is not a non-zero power of two.
	ErrInvalidQueueSize = errors.New("compute/strix_aql: queue size must be a non-zero power of two")
)

// HSAKernelDispatchPacket represents the architected 64-byte HSA kernel dispatch packet
// matching hsa_kernel_dispatch_packet_t for AMD RDNA 3.5 (gfx1151 / Strix Halo).
// Layout is cache-line aligned to exactly 64 bytes (Zen 5 CPU cache line and RDNA 3.5 L2 cache line).
type HSAKernelDispatchPacket struct {
	Header             uint16 // offset 0:  bits [7:0] type, bit 8 barrier, bits [10:9] acquire fence, bits [12:11] release fence
	Setup              uint16 // offset 2:  dimensions (1, 2, or 3)
	WorkgroupSizeX     uint16 // offset 4:  workgroup size X dimension
	WorkgroupSizeY     uint16 // offset 6:  workgroup size Y dimension
	WorkgroupSizeZ     uint16 // offset 8:  workgroup size Z dimension
	Reserved0          uint16 // offset 10: reserved padding / alignment
	GridSizeX          uint32 // offset 12: grid size X dimension
	GridSizeY          uint32 // offset 16: grid size Y dimension
	GridSizeZ          uint32 // offset 20: grid size Z dimension
	PrivateSegmentSize uint32 // offset 24: per-work-item private segment size (bytes)
	GroupSegmentSize   uint32 // offset 28: group segment / LDS size in bytes
	KernelObject       uint64 // offset 32: GPU virtual address of kernel machine code
	KernargAddress     uint64 // offset 40: GPU virtual address of kernel arguments buffer
	Reserved2          uint64 // offset 48: reserved padding
	CompletionSignal   uint64 // offset 56: 64-bit completion signal handle
}

// Compile-time assertion: HSAKernelDispatchPacket must be exactly 64 bytes.
var _ = [1]struct{}{}[unsafe.Sizeof(HSAKernelDispatchPacket{})-HSAPacketBytes]

// EncodeHSAHeader constructs the 16-bit AQL packet header with bitfield packing.
func EncodeHSAHeader(packetType uint16, barrier bool, acquireScope, releaseScope uint16) uint16 {
	h := packetType & HSAHeaderTypeMask
	if barrier {
		h |= HSAHeaderBarrierBit
	}
	h |= (acquireScope & HSAHeaderAcquireMask) << HSAHeaderAcquireShift
	h |= (releaseScope & HSAHeaderReleaseMask) << HSAHeaderReleaseShift
	return h
}

// NewKernelDispatchPacket instantiates an HSAKernelDispatchPacket with standard agent-level
// fence semantics and initializes dimension and buffer parameters.
func NewKernelDispatchPacket(
	dimensions uint16,
	wgX, wgY, wgZ uint16,
	gridX, gridY, gridZ uint32,
	kernelObj, kernargAddr uint64,
	groupSegSize, privateSegSize uint32,
	completionSignal uint64,
) HSAKernelDispatchPacket {
	header := EncodeHSAHeader(HSAPacketTypeKernelDispatch, false, HSAFenceScopeAgent, HSAFenceScopeAgent)
	return HSAKernelDispatchPacket{
		Header:             header,
		Setup:              dimensions,
		WorkgroupSizeX:     wgX,
		WorkgroupSizeY:     wgY,
		WorkgroupSizeZ:     wgZ,
		Reserved0:          0,
		GridSizeX:          gridX,
		GridSizeY:          gridY,
		GridSizeZ:          gridZ,
		PrivateSegmentSize: privateSegSize,
		GroupSegmentSize:   groupSegSize,
		KernelObject:       kernelObj,
		KernargAddress:     kernargAddr,
		Reserved2:          0,
		CompletionSignal:   completionSignal,
	}
}

// Serialize encodes the HSAKernelDispatchPacket into a 64-byte little-endian array.
func (p *HSAKernelDispatchPacket) Serialize() [64]byte {
	var buf [64]byte
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
	binary.LittleEndian.PutUint64(buf[48:56], p.Reserved2)
	binary.LittleEndian.PutUint64(buf[56:64], p.CompletionSignal)
	return buf
}

// DeserializeKernelDispatchPacket parses a 64-byte slice into an HSAKernelDispatchPacket.
func DeserializeKernelDispatchPacket(data []byte) (HSAKernelDispatchPacket, error) {
	if len(data) < HSAPacketBytes {
		return HSAKernelDispatchPacket{}, fmt.Errorf("%w: got %d bytes", ErrInvalidPacketSize, len(data))
	}
	return HSAKernelDispatchPacket{
		Header:             binary.LittleEndian.Uint16(data[0:2]),
		Setup:              binary.LittleEndian.Uint16(data[2:4]),
		WorkgroupSizeX:     binary.LittleEndian.Uint16(data[4:6]),
		WorkgroupSizeY:     binary.LittleEndian.Uint16(data[6:8]),
		WorkgroupSizeZ:     binary.LittleEndian.Uint16(data[8:10]),
		Reserved0:          binary.LittleEndian.Uint16(data[10:12]),
		GridSizeX:          binary.LittleEndian.Uint32(data[12:16]),
		GridSizeY:          binary.LittleEndian.Uint32(data[16:20]),
		GridSizeZ:          binary.LittleEndian.Uint32(data[20:24]),
		PrivateSegmentSize: binary.LittleEndian.Uint32(data[24:28]),
		GroupSegmentSize:   binary.LittleEndian.Uint32(data[28:32]),
		KernelObject:       binary.LittleEndian.Uint64(data[32:40]),
		KernargAddress:     binary.LittleEndian.Uint64(data[40:48]),
		Reserved2:          binary.LittleEndian.Uint64(data[48:56]),
		CompletionSignal:   binary.LittleEndian.Uint64(data[56:64]),
	}, nil
}

// PM4 Type-3 Command Processor Constants for RDNA 3.5 (gfx1151).
const (
	// PM4 Type 3 Packet Opcodes.
	PKT3_DISPATCH_DIRECT uint8 = 0x15 // Compute grid dispatch without HSA descriptors
	PKT3_SET_SH_REG      uint8 = 0x76 // Direct shader register configuration
	PKT3_ACQUIRE_MEM     uint8 = 0x58 // Hardware cache invalidation/write-back
	PKT3_RELEASE_MEM     uint8 = 0x49 // Memory fence and event signaling

	// SH Register base and offsets for RDNA 3.5 compute pipe.
	RegSHConfigBase    uint32 = 0x2C00
	RegComputePgmRsrc1 uint32 = 0x2E12
	RegComputePgmRsrc2 uint32 = 0x2E13
	RegComputePgmRsrc3 uint32 = 0x2E14

	// COMPUTE_PGM_RSRC1 bitfields for RDNA 3.5 (gfx1151).
	ComputePgmRsrc1Wave32Bit = 1 << 19 // Wave32 execution mode on RDNA 3.5 (WGP mode)
	ComputePgmRsrc1WgpMode   = 1 << 19 // WGP (Work Group Processor) mode

	// Cache Flush / Invalidation flags for PKT3_ACQUIRE_MEM and PKT3_RELEASE_MEM.
	CPCoherCNTLCoherInvL1   uint32 = 1 << 0 // Invalidate TCP (L1 vector cache)
	CPCoherCNTLCoherInvL2   uint32 = 1 << 1 // Invalidate L2 cache
	CPCoherCNTLCoherWbL2    uint32 = 1 << 2 // Write-back L2 cache
	CPCoherCNTLCoherInvK    uint32 = 1 << 3 // Invalidate SQ-K$ (Scalar cache)
	CPCoherCNTLCoherInvI    uint32 = 1 << 4 // Invalidate SQ-I$ (Instruction cache)
	CPCoherCNTLCoherInvMALL uint32 = 1 << 5 // Invalidate MALL (Infinity Cache)
	CPCoherCNTLCoherWbMALL  uint32 = 1 << 6 // Write-back MALL (Infinity Cache)
	CPCoherCNTLMALLFlushAll uint32 = CPCoherCNTLCoherInvMALL | CPCoherCNTLCoherWbMALL
)

// PM4Packet3Header constructs a 32-bit PM4 Type-3 packet header.
// Formula: (3 << 30) | ((count & 0x3FFF) << 16) | (opcode << 8)
func PM4Packet3Header(opcode uint8, count uint16) uint32 {
	return (3 << 30) | ((uint32(count) & 0x3FFF) << 16) | (uint32(opcode) << 8)
}

// PM4CommandBuffer is a fluent builder to assemble and serialize PM4 Type-3 packets
// for direct submission to the RDNA 3.5 hardware command processor.
type PM4CommandBuffer struct {
	dwords []uint32
}

// NewPM4CommandBuffer creates a new PM4CommandBuffer with default preallocated capacity.
func NewPM4CommandBuffer() *PM4CommandBuffer {
	return &PM4CommandBuffer{
		dwords: make([]uint32, 0, 128),
	}
}

// NewPM4CommandBufferCapacity creates a PM4CommandBuffer with the given initial capacity.
func NewPM4CommandBufferCapacity(cap int) *PM4CommandBuffer {
	return &PM4CommandBuffer{
		dwords: make([]uint32, 0, cap),
	}
}

// Reset clears the buffer for reuse without reallocating underlying storage.
func (b *PM4CommandBuffer) Reset() *PM4CommandBuffer {
	b.dwords = b.dwords[:0]
	return b
}

// Len returns the number of DWORDs currently in the command buffer.
func (b *PM4CommandBuffer) Len() int {
	return len(b.dwords)
}

// Dwords returns a copy of the DWORD stream.
func (b *PM4CommandBuffer) Dwords() []uint32 {
	res := make([]uint32, len(b.dwords))
	copy(res, b.dwords)
	return res
}

// Bytes serializes the DWORDs into a little-endian byte slice.
func (b *PM4CommandBuffer) Bytes() []byte {
	out := make([]byte, len(b.dwords)*4)
	for i, dw := range b.dwords {
		binary.LittleEndian.PutUint32(out[i*4:(i+1)*4], dw)
	}
	return out
}

// EmitDispatchDirect appends a PKT3_DISPATCH_DIRECT packet (opcode 0x15).
// Parameters:
//   - gridX, gridY, gridZ: Grid dimensions in work-items
//   - flags: Initiator flags (e.g. COMPUTE_DISPATCH_INITIATOR)
func (b *PM4CommandBuffer) EmitDispatchDirect(gridX, gridY, gridZ uint32, flags uint32) *PM4CommandBuffer {
	// Payload: 4 DWORDs -> count = 4 - 1 = 3
	header := PM4Packet3Header(PKT3_DISPATCH_DIRECT, 3)
	b.dwords = append(b.dwords, header, gridX, gridY, gridZ, flags)
	return b
}

// EmitSetSHReg appends a PKT3_SET_SH_REG packet (opcode 0x76).
// Parameters:
//   - reg: Destination shader register offset (or absolute address >= RegSHConfigBase)
//   - values: One or more consecutive register values
func (b *PM4CommandBuffer) EmitSetSHReg(reg uint32, values ...uint32) *PM4CommandBuffer {
	if len(values) == 0 {
		return b
	}
	regOffset := reg
	if regOffset >= RegSHConfigBase {
		regOffset -= RegSHConfigBase
	}
	// Payload: 1 DWORD (reg offset) + len(values) -> count = (1 + len) - 1 = len(values)
	header := PM4Packet3Header(PKT3_SET_SH_REG, uint16(len(values)))
	b.dwords = append(b.dwords, header, regOffset)
	b.dwords = append(b.dwords, values...)
	return b
}

// EmitSetComputePgmRsrc configures COMPUTE_PGM_RSRC1 and COMPUTE_PGM_RSRC2 registers.
// Parameters:
//   - wave32: Set Wave32 execution mode bit on RDNA 3.5
//   - vgprs: Vector general-purpose registers per thread
//   - sgprs: Scalar general-purpose registers per wavefront
//   - ldsBytes: Local Data Share allocation size in bytes
func (b *PM4CommandBuffer) EmitSetComputePgmRsrc(wave32 bool, vgprs, sgprs, ldsBytes uint32) *PM4CommandBuffer {
	// COMPUTE_PGM_RSRC1: bits [5:0] VGPRS granularity ((vgprs-1)/8 on RDNA), bits [9:6] SGPRS
	var vgprGran uint32
	if vgprs > 0 {
		vgprGran = (vgprs - 1) / 8
	}
	var sgprGran uint32
	if sgprs > 0 {
		sgprGran = (sgprs - 1) / 8
	}
	rsrc1 := (vgprGran & 0x3F) | ((sgprGran & 0x0F) << 6)
	if wave32 {
		rsrc1 |= ComputePgmRsrc1Wave32Bit
	}

	// COMPUTE_PGM_RSRC2: bits [23:15] LDS size in 512-byte blocks, bits [4:0] User SGPRs
	ldsBlocks := (ldsBytes + 511) / 512
	rsrc2 := ((ldsBlocks & 0x1FF) << 15) | 2 // User SGPR count default 2

	return b.EmitSetSHReg(RegComputePgmRsrc1, rsrc1, rsrc2)
}

// EmitAcquireMem appends a PKT3_ACQUIRE_MEM packet (opcode 0x58) for cache invalidation
// and write-back across L1/L2 and MALL (Infinity Cache).
// Parameters:
//   - flags: Bitmask of CPCoherCNTL flags (e.g. CPCoherCNTLCoherInvL1 | CPCoherCNTLMALLFlushAll)
func (b *PM4CommandBuffer) EmitAcquireMem(flags uint32) *PM4CommandBuffer {
	// Payload: 6 DWORDs -> count = 6 - 1 = 5
	// Dword 0: flags (coherence control)
	// Dword 1: cp_coher_size (0xFFFFFFFF = full range)
	// Dword 2: cp_coher_size_hi
	// Dword 3: cp_coher_base
	// Dword 4: cp_coher_base_hi
	// Dword 5: poll_interval (10 cycles)
	header := PM4Packet3Header(PKT3_ACQUIRE_MEM, 5)
	b.dwords = append(b.dwords, header, flags, 0xFFFFFFFF, 0, 0, 0, 10)
	return b
}

// EmitReleaseMem appends a PKT3_RELEASE_MEM packet (opcode 0x49) for memory fence,
// cache flush completion, and event signaling.
// Optional extra arguments:
//   - extra[0] = dataSel uint32, extra[1] = addr uint64, extra[2] = data uint64
//   - OR extra[0] = addr uint64, extra[1] = data uint64
func (b *PM4CommandBuffer) EmitReleaseMem(event uint32, extra ...any) *PM4CommandBuffer {
	var dataSel uint32 = 1 // default: write 64-bit int
	var addr uint64
	var data uint64
	var intSel uint32

	if len(extra) >= 3 {
		if ds, ok := extra[0].(uint32); ok {
			dataSel = ds
		}
		if a, ok := extra[1].(uint64); ok {
			addr = a
		}
		if d, ok := extra[2].(uint64); ok {
			data = d
		}
	} else if len(extra) == 2 {
		if a, ok := extra[0].(uint64); ok {
			addr = a
		}
		if d, ok := extra[1].(uint64); ok {
			data = d
		}
	}

	addrLo := uint32(addr & 0xFFFFFFFF)
	addrHi := uint32((addr >> 32) & 0xFFFFFFFF)
	dataLo := uint32(data & 0xFFFFFFFF)
	dataHi := uint32((data >> 32) & 0xFFFFFFFF)

	// Payload: 7 DWORDs -> count = 7 - 1 = 6
	header := PM4Packet3Header(PKT3_RELEASE_MEM, 6)
	b.dwords = append(b.dwords, header, event, dataSel, addrLo, addrHi, dataLo, dataHi, intSel)
	return b
}

// AQLDoorbell defines the interface for ringing an AQL queue doorbell aperture.
type AQLDoorbell interface {
	Ring(writePtr uint64)
}

// StrixAQLQueue represents a user-space AQL circular ring buffer in coherent memory
// mapped directly to the AMD RDNA 3.5 hardware command processor.
type StrixAQLQueue struct {
	QueueID     uint32
	Size        uint32
	Mask        uint32
	RingBuffer  []HSAKernelDispatchPacket
	WritePtr    uint64      // 64-bit atomic queue write-pointer index
	ReadPtr     uint64      // 64-bit atomic queue read-pointer index
	Doorbell    AQLDoorbell // hardware or simulated HSA doorbell (see amd_gpudirect.go)
	DoorbellPtr *uint64     // optional direct atomic pointer to MMIO aperture register
	mu          sync.Mutex
}

// NewStrixAQLQueue allocates an in-memory AQL circular ring buffer for the specified capacity.
func NewStrixAQLQueue(queueID uint32, size uint32, doorbell AQLDoorbell) *StrixAQLQueue {
	if size == 0 || (size&(size-1)) != 0 {
		size = DefaultStrixAQLQueueSize
	}
	if size > MaxStrixAQLQueueSize {
		size = MaxStrixAQLQueueSize
	}

	return &StrixAQLQueue{
		QueueID:    queueID,
		Size:       size,
		Mask:       size - 1,
		RingBuffer: make([]HSAKernelDispatchPacket, size),
		Doorbell:   doorbell,
	}
}

// SetDoorbellPtr binds a direct atomic pointer to the hardware MMIO doorbell register.
func (q *StrixAQLQueue) SetDoorbellPtr(ptr *uint64) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.DoorbellPtr = ptr
}

// LoadWritePtr atomically reads the current queue write-pointer index.
func (q *StrixAQLQueue) LoadWritePtr() uint64 {
	return atomic.LoadUint64(&q.WritePtr)
}

// LoadReadPtr atomically reads the current queue read-pointer index.
func (q *StrixAQLQueue) LoadReadPtr() uint64 {
	return atomic.LoadUint64(&q.ReadPtr)
}

// AdvanceReadPtr advances the queue read-pointer index by count, simulating GPU consumption.
func (q *StrixAQLQueue) AdvanceReadPtr(count uint64) {
	atomic.AddUint64(&q.ReadPtr, count)
}

// SetReadPtr explicitly updates the queue read-pointer index.
func (q *StrixAQLQueue) SetReadPtr(idx uint64) {
	atomic.StoreUint64(&q.ReadPtr, idx)
}

// SubmitPacket enqueues a 64-byte HSAKernelDispatchPacket into the circular ring buffer,
// advances the write-pointer index with release semantics, and rings the hardware doorbell.
// Returns the new write pointer index or ErrAQLQueueFull if the queue is saturated.
func (q *StrixAQLQueue) SubmitPacket(pkt HSAKernelDispatchPacket) (uint64, error) {
	q.mu.Lock()
	defer q.mu.Unlock()

	writeIdx := atomic.LoadUint64(&q.WritePtr)
	readIdx := atomic.LoadUint64(&q.ReadPtr)

	// Check for ring buffer saturation
	if writeIdx-readIdx >= uint64(q.Size) {
		return writeIdx, ErrAQLQueueFull
	}

	slotIdx := writeIdx & uint64(q.Mask)
	q.RingBuffer[slotIdx] = pkt

	nextIdx := writeIdx + 1
	if nextIdx == 0 {
		nextIdx = 1
	}

	// Release memory barrier: ensure packet contents are visible before write pointer update
	atomic.StoreUint64(&q.WritePtr, nextIdx)

	// Ring hardware doorbell MMIO aperture
	if q.DoorbellPtr != nil {
		atomic.StoreUint64(q.DoorbellPtr, nextIdx)
	}
	if q.Doorbell != nil {
		q.Doorbell.Ring(nextIdx)
	}

	return nextIdx, nil
}
