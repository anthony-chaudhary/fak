// Package strix implements direct userspace AQL queue management,
// cacheline-aligned 64-byte packet formatting, and low-overhead MMIO hardware
// doorbell signaling for sub-50µs tool-yield preemption and resumption on AMD
// Strix Halo (Ryzen AI Max+ 395 / GFX1151).
package strix

import (
	"fmt"
	"unsafe"
)

// AQLPacketSizeBytes is the architected size of an HSA / AQL packet (64 bytes).
// Every packet occupies exactly one CPU cache line on Zen 5 and RDNA 3.5.
const AQLPacketSizeBytes = 64

// AQLPacketType identifies the purpose and format of an AQL packet (bits 0-7 of header).
type AQLPacketType uint8

const (
	// AQLPacketTypeVendorSpecific denotes AMD/vendor-specific packet formats.
	AQLPacketTypeVendorSpecific AQLPacketType = 0

	// AQLPacketTypeInvalid marks an uninitialized or consumed packet slot.
	AQLPacketTypeInvalid AQLPacketType = 1

	// AQLPacketTypeKernelDispatch specifies a standard compute kernel dispatch.
	AQLPacketTypeKernelDispatch AQLPacketType = 2

	// AQLPacketTypeBarrierAnd specifies an AND barrier waiting on all dependent signals.
	AQLPacketTypeBarrierAnd AQLPacketType = 3

	// AQLPacketTypeAgentDispatch specifies an agent-level command packet.
	AQLPacketTypeAgentDispatch AQLPacketType = 4

	// AQLPacketTypeBarrierOr specifies an OR barrier waiting on any dependent signal.
	AQLPacketTypeBarrierOr AQLPacketType = 5
)

// String returns the human-readable name of the AQL packet type.
func (t AQLPacketType) String() string {
	switch t {
	case AQLPacketTypeVendorSpecific:
		return "VENDOR_SPECIFIC"
	case AQLPacketTypeInvalid:
		return "INVALID"
	case AQLPacketTypeKernelDispatch:
		return "KERNEL_DISPATCH"
	case AQLPacketTypeBarrierAnd:
		return "BARRIER_AND"
	case AQLPacketTypeAgentDispatch:
		return "AGENT_DISPATCH"
	case AQLPacketTypeBarrierOr:
		return "BARRIER_OR"
	default:
		return fmt.Sprintf("UNKNOWN_AQL_TYPE(%d)", uint8(t))
	}
}

// AQLFenceScope specifies memory synchronization domains for acquire and release operations.
type AQLFenceScope uint8

const (
	// AQLFenceScopeNone applies no memory fence before or after dispatch.
	AQLFenceScopeNone AQLFenceScope = 0

	// AQLFenceScopeComponent restricts fence synchronization to the local compute unit / GPU agent.
	AQLFenceScopeComponent AQLFenceScope = 1

	// AQLFenceScopeSystem enforces full coherent UMA memory fence across Zen 5 CPU and RDNA 3.5 GPU.
	AQLFenceScopeSystem AQLFenceScope = 2
)

// String returns the human-readable name of the fence scope.
func (s AQLFenceScope) String() string {
	switch s {
	case AQLFenceScopeNone:
		return "NONE"
	case AQLFenceScopeComponent:
		return "COMPONENT"
	case AQLFenceScopeSystem:
		return "SYSTEM"
	default:
		return fmt.Sprintf("UNKNOWN_FENCE_SCOPE(%d)", uint8(s))
	}
}

// Header bitmasks and bit shifts for the 16-bit AQL packet header bitfield.
const (
	AQLPacketHeaderTypeMask          uint16 = 0x00FF
	AQLPacketHeaderTypeShift         uint16 = 0
	AQLPacketHeaderBarrierBit        uint16 = 1 << 8
	AQLPacketHeaderAcquireScopeMask  uint16 = 0x03 << 9
	AQLPacketHeaderAcquireScopeShift uint16 = 9
	AQLPacketHeaderReleaseScopeMask  uint16 = 0x03 << 11
	AQLPacketHeaderReleaseScopeShift uint16 = 11
)

// AQLPacketHeader represents the 16-bit header at offset 0 of every AQL packet.
//
// Layout:
//
//	Bits 0-7:   Packet Type (AQLPacketType)
//	Bit 8:      Barrier Flag (1 = wait for all prior packets before starting)
//	Bits 9-10:  Acquire Memory Fence Scope (AQLFenceScope)
//	Bits 11-12: Release Memory Fence Scope (AQLFenceScope)
//	Bits 13-15: Reserved (must be 0)
type AQLPacketHeader uint16

// NewPacketHeader encodes type, barrier flag, acquire fence scope, and release
// fence scope into a 16-bit AQLPacketHeader.
func NewPacketHeader(pktType AQLPacketType, barrier bool, acquire, release AQLFenceScope) AQLPacketHeader {
	var val uint16
	val |= uint16(pktType) & AQLPacketHeaderTypeMask
	if barrier {
		val |= AQLPacketHeaderBarrierBit
	}
	val |= (uint16(acquire) & 0x03) << AQLPacketHeaderAcquireScopeShift
	val |= (uint16(release) & 0x03) << AQLPacketHeaderReleaseScopeShift
	return AQLPacketHeader(val)
}

// Type extracts the packet type from bits 0-7.
func (h AQLPacketHeader) Type() AQLPacketType {
	return AQLPacketType(uint16(h) & AQLPacketHeaderTypeMask)
}

// Barrier reports whether the barrier flag (bit 8) is set.
func (h AQLPacketHeader) Barrier() bool {
	return (uint16(h) & AQLPacketHeaderBarrierBit) != 0
}

// AcquireScope extracts the acquire fence scope from bits 9-10.
func (h AQLPacketHeader) AcquireScope() AQLFenceScope {
	return AQLFenceScope((uint16(h) & AQLPacketHeaderAcquireScopeMask) >> AQLPacketHeaderAcquireScopeShift)
}

// ReleaseScope extracts the release fence scope from bits 11-12.
func (h AQLPacketHeader) ReleaseScope() AQLFenceScope {
	return AQLFenceScope((uint16(h) & AQLPacketHeaderReleaseScopeMask) >> AQLPacketHeaderReleaseScopeShift)
}

// Raw returns the underlying uint16 representation of the header.
func (h AQLPacketHeader) Raw() uint16 {
	return uint16(h)
}

// WithBarrier returns a copy of the header with the barrier bit set or cleared.
func (h AQLPacketHeader) WithBarrier(barrier bool) AQLPacketHeader {
	if barrier {
		return AQLPacketHeader(uint16(h) | AQLPacketHeaderBarrierBit)
	}
	return AQLPacketHeader(uint16(h) &^ AQLPacketHeaderBarrierBit)
}

// WithFenceScope returns a copy of the header with updated acquire and release scopes.
func (h AQLPacketHeader) WithFenceScope(acquire, release AQLFenceScope) AQLPacketHeader {
	val := uint16(h) &^ (AQLPacketHeaderAcquireScopeMask | AQLPacketHeaderReleaseScopeMask)
	val |= (uint16(acquire) & 0x03) << AQLPacketHeaderAcquireScopeShift
	val |= (uint16(release) & 0x03) << AQLPacketHeaderReleaseScopeShift
	return AQLPacketHeader(val)
}

// String returns a compact representation of the packet header.
func (h AQLPacketHeader) String() string {
	return fmt.Sprintf("AQLHeader(type=%s, barrier=%t, acq=%s, rel=%s)",
		h.Type(), h.Barrier(), h.AcquireScope(), h.ReleaseScope())
}

// AQLPacket defines the common interface for 64-byte HSA / AQL packets.
type AQLPacket interface {
	PacketHeader() AQLPacketHeader
	Bytes() [64]byte
}

// AQLKernelDispatchPacket represents an architected 64-byte AQL kernel dispatch packet
// for GFX1151 / RDNA 3.5 command processor user-mode queues.
type AQLKernelDispatchPacket struct {
	Header             AQLPacketHeader `json:"header"`               // offset 0: packet type, barrier, fences (2 bytes)
	Dimensions         uint16          `json:"dimensions"`           // offset 2: number of grid dimensions (1, 2, or 3) (2 bytes)
	WorkgroupSizeX     uint16          `json:"workgroup_size_x"`     // offset 4: workgroup X size (2 bytes)
	WorkgroupSizeY     uint16          `json:"workgroup_size_y"`     // offset 6: workgroup Y size (2 bytes)
	WorkgroupSizeZ     uint16          `json:"workgroup_size_z"`     // offset 8: workgroup Z size (2 bytes)
	Reserved0          uint16          `json:"reserved0,omitempty"`  // offset 10: reserved padding (2 bytes)
	GridSizeX          uint32          `json:"grid_size_x"`          // offset 12: grid X size (4 bytes)
	GridSizeY          uint32          `json:"grid_size_y"`          // offset 16: grid Y size (4 bytes)
	GridSizeZ          uint32          `json:"grid_size_z"`          // offset 20: grid Z size (4 bytes)
	PrivateSegmentSize uint32          `json:"private_segment_size"` // offset 24: private scratch size per work-item (4 bytes)
	GroupSegmentSize   uint32          `json:"group_segment_size"`   // offset 28: LDS group segment size per workgroup (4 bytes)
	KernelObject       uint64          `json:"kernel_object"`        // offset 32: GPU virtual address of kernel object (8 bytes)
	KernargAddress     uint64          `json:"kernarg_address"`      // offset 40: GPU virtual address of kernarg buffer (8 bytes)
	Reserved1          uint64          `json:"reserved1,omitempty"`  // offset 48: reserved padding (8 bytes)
	CompletionSignal   uint64          `json:"completion_signal"`    // offset 56: completion signal handle (8 bytes)
}

// PacketHeader returns the packet's 16-bit header.
func (p AQLKernelDispatchPacket) PacketHeader() AQLPacketHeader {
	return p.Header
}

// Bytes serializes the kernel dispatch packet into a raw 64-byte cache line array.
func (p AQLKernelDispatchPacket) Bytes() [64]byte {
	return *(*[64]byte)(unsafe.Pointer(&p))
}

// AQLBarrierPacket represents an architected 64-byte AQL barrier packet (BARRIER_AND or BARRIER_OR).
// It enforces ordering and waits for up to 5 dependent completion signals before downstream work runs.
type AQLBarrierPacket struct {
	Header           AQLPacketHeader `json:"header"`              // offset 0: packet type, barrier, fences (2 bytes)
	Reserved0        uint16          `json:"reserved0,omitempty"` // offset 2: reserved padding (2 bytes)
	Reserved1        uint32          `json:"reserved1,omitempty"` // offset 4: reserved padding (4 bytes)
	DepSignals       [5]uint64       `json:"dep_signals"`         // offset 8: up to 5 dependent completion signals (40 bytes)
	Reserved2        uint64          `json:"reserved2,omitempty"` // offset 48: reserved padding (8 bytes)
	CompletionSignal uint64          `json:"completion_signal"`   // offset 56: completion signal handle (8 bytes)
}

// PacketHeader returns the barrier packet's 16-bit header.
func (p AQLBarrierPacket) PacketHeader() AQLPacketHeader {
	return p.Header
}

// Bytes serializes the barrier packet into a raw 64-byte cache line array.
func (p AQLBarrierPacket) Bytes() [64]byte {
	return *(*[64]byte)(unsafe.Pointer(&p))
}

// KernelDispatchFromBytes reconstructs an AQLKernelDispatchPacket from a 64-byte array.
func KernelDispatchFromBytes(raw [64]byte) AQLKernelDispatchPacket {
	return *(*AQLKernelDispatchPacket)(unsafe.Pointer(&raw))
}

// BarrierFromBytes reconstructs an AQLBarrierPacket from a 64-byte array.
func BarrierFromBytes(raw [64]byte) AQLBarrierPacket {
	return *(*AQLBarrierPacket)(unsafe.Pointer(&raw))
}

// Compile-time assertions: every AQL packet type must be strictly 64 bytes in size.
var (
	_ = [1]struct{}{}[unsafe.Sizeof(AQLKernelDispatchPacket{})-AQLPacketSizeBytes]
	_ = [1]struct{}{}[AQLPacketSizeBytes-unsafe.Sizeof(AQLKernelDispatchPacket{})]
	_ = [1]struct{}{}[unsafe.Sizeof(AQLBarrierPacket{})-AQLPacketSizeBytes]
	_ = [1]struct{}{}[AQLPacketSizeBytes-unsafe.Sizeof(AQLBarrierPacket{})]
)
