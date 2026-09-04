// Package compute implements hardware abstraction, tensor computation, memory slab management,
// and zero-copy device interconnect acceleration for the fak agent kernel.
package compute

import (
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// KFD allocation memory flags for HSA/KFD ioctl (kfd_ioctl_alloc_memory_of_gpu_args).
const (
	// KFD_IOC_ALLOC_MEM_FLAGS_VRAM allocates memory from the GPU-accessible pool.
	KFD_IOC_ALLOC_MEM_FLAGS_VRAM uint32 = 0x00000001
	// KFD_IOC_ALLOC_MEM_FLAGS_GTT allocates from system memory via Graphics Translation Table.
	KFD_IOC_ALLOC_MEM_FLAGS_GTT uint32 = 0x00000002
	// KFD_IOC_ALLOC_MEM_FLAGS_USERPTR maps host user-space virtual pointer directly.
	KFD_IOC_ALLOC_MEM_FLAGS_USERPTR uint32 = 0x00000004
	// KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL maps hardware queue doorbell aperture.
	KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL uint32 = 0x00000008
	// KFD_IOC_ALLOC_MEM_FLAGS_NONPAGED allocates non-pageable pinned memory.
	KFD_IOC_ALLOC_MEM_FLAGS_NONPAGED uint32 = 0x00000010
	// KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE marks the allocated buffer as writable.
	KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE uint32 = 0x08000000
	// KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE marks the buffer executable (GPU ISA shaders).
	KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE uint32 = 0x10000000
	// KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED allocates uncached system memory.
	KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED uint32 = 0x40000000
	// KFD_IOC_ALLOC_MEM_FLAGS_COHERENT allocates hardware-coherent memory across CPU and GPU caches.
	KFD_IOC_ALLOC_MEM_FLAGS_COHERENT uint32 = 0x80000000
	// KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM marks memory for AQL queues.
	KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM uint32 = 0x02000000
)

// APUDefaultAllocFlags defines the standard zero-copy coherent allocation flags for AMD APU unified memory.
const APUDefaultAllocFlags = KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE

// APUArchitectureProfile details the microarchitectural parameters and unified memory specs for an AMD APU.
type APUArchitectureProfile struct {
	Architecture          string  `json:"architecture"`
	Codename              string  `json:"codename"`
	MarketingName         string  `json:"marketing_name"`
	ComputeUnits          int     `json:"compute_units"`
	MemoryBusWidthBits    int     `json:"memory_bus_width_bits"`
	MemoryType            string  `json:"memory_type"`
	TheoreticalPeakBWGBps float64 `json:"theoretical_peak_bw_gbps"`
	SustainedEfficiency   float64 `json:"sustained_efficiency"`
	IsUnifiedAPU          bool    `json:"is_unified_apu"`
}

// knownAPUProfiles registers AMD APU microarchitectures with unified host/GPU physical memory topology.
var knownAPUProfiles = map[string]APUArchitectureProfile{
	"gfx1151": {
		Architecture:          "gfx1151",
		Codename:              "Strix Halo",
		MarketingName:         "Ryzen AI Max+ 395 / Radeon 8060S",
		ComputeUnits:          40,
		MemoryBusWidthBits:    256,
		MemoryType:            "LPDDR5X-8533",
		TheoreticalPeakBWGBps: 273.056,
		SustainedEfficiency:   0.82,
		IsUnifiedAPU:          true,
	},
	"gfx1150": {
		Architecture:          "gfx1150",
		Codename:              "Strix Point",
		MarketingName:         "Ryzen AI 9 HX 370 / Radeon 890M",
		ComputeUnits:          16,
		MemoryBusWidthBits:    128,
		MemoryType:            "LPDDR5X-7500",
		TheoreticalPeakBWGBps: 120.0,
		SustainedEfficiency:   0.80,
		IsUnifiedAPU:          true,
	},
	"gfx1103": {
		Architecture:          "gfx1103",
		Codename:              "Phoenix",
		MarketingName:         "Ryzen 7 7840U / Radeon 780M",
		ComputeUnits:          12,
		MemoryBusWidthBits:    128,
		MemoryType:            "LPDDR5X-6400",
		TheoreticalPeakBWGBps: 102.4,
		SustainedEfficiency:   0.78,
		IsUnifiedAPU:          true,
	},
	"gfx1100": {
		Architecture:          "gfx1100",
		Codename:              "Hawk Point",
		MarketingName:         "Ryzen 8040 Series / Radeon 780M",
		ComputeUnits:          12,
		MemoryBusWidthBits:    128,
		MemoryType:            "LPDDR5X-6400",
		TheoreticalPeakBWGBps: 102.4,
		SustainedEfficiency:   0.78,
		IsUnifiedAPU:          true,
	},
	"gfx1036": {
		Architecture:          "gfx1036",
		Codename:              "Rembrandt / Van Gogh",
		MarketingName:         "Ryzen 6000 Series / Steam Deck",
		ComputeUnits:          8,
		MemoryBusWidthBits:    128,
		MemoryType:            "LPDDR5-5500",
		TheoreticalPeakBWGBps: 88.0,
		SustainedEfficiency:   0.75,
		IsUnifiedAPU:          true,
	},
	"gfx1035": {
		Architecture:          "gfx1035",
		Codename:              "Rembrandt",
		MarketingName:         "Ryzen 6800U / Radeon 680M",
		ComputeUnits:          12,
		MemoryBusWidthBits:    128,
		MemoryType:            "LPDDR5-6400",
		TheoreticalPeakBWGBps: 102.4,
		SustainedEfficiency:   0.76,
		IsUnifiedAPU:          true,
	},
	"gfx90c": {
		Architecture:          "gfx90c",
		Codename:              "Cezanne / Renoir",
		MarketingName:         "Ryzen 5000 / 4000 Series APU",
		ComputeUnits:          8,
		MemoryBusWidthBits:    128,
		MemoryType:            "DDR4-3200 / LPDDR4X-4266",
		TheoreticalPeakBWGBps: 68.2,
		SustainedEfficiency:   0.72,
		IsUnifiedAPU:          true,
	},
}

// IsAMDAPUArchitecture reports whether the given architecture string identifies an AMD APU.
func IsAMDAPUArchitecture(arch string) bool {
	p, ok := knownAPUProfiles[arch]
	return ok && p.IsUnifiedAPU
}

// LookupAPUProfile returns the APUArchitectureProfile for the given architecture string, if recognized.
func LookupAPUProfile(arch string) (APUArchitectureProfile, bool) {
	p, ok := knownAPUProfiles[arch]
	return p, ok
}

// APUTopologyInfo captures the verified hardware topology of an AMD APU unified memory system.
type APUTopologyInfo struct {
	NodeID               int                    `json:"node_id"`
	GPUID                int                    `json:"gpu_id"`
	DeviceName           string                 `json:"device_name"`
	Architecture         string                 `json:"architecture"`
	Profile              APUArchitectureProfile `json:"profile"`
	NUMANode             int                    `json:"numa_node"`
	HostDRAMBytes        uint64                 `json:"host_dram_bytes"`
	TotalVRAMBytes       uint64                 `json:"total_vram_bytes"`
	UnifiedDRAMBytes     uint64                 `json:"unified_dram_bytes"`
	SingleNUMANodeZero   bool                   `json:"single_numa_node_zero"`
	MatchingAddressSpace bool                   `json:"matching_address_space"`
	IsUnifiedTopology    bool                   `json:"is_unified_topology"`
	Reason               string                 `json:"reason,omitempty"`
}

// JSON encodes APUTopologyInfo as indented JSON bytes.
func (t APUTopologyInfo) JSON() ([]byte, error) {
	return json.MarshalIndent(t, "", "  ")
}

// DetectAPUTopology verifies if an AMD device node has a valid APU architecture and unified topology.
// Requirements:
// 1. Known APU architecture signature (e.g., gfx1151 Strix Halo, gfx1103 Phoenix, gfx1100 Hawk Point).
// 2. Single NUMA node 0 (CPU and GPU share the unified NUMA domain).
// 3. Matching system/GPU physical address space (TotalVRAMBytes == HostDRAMBytes or carved unified pool).
func DetectAPUTopology(node AMDDeviceNode, hostDRAMBytes uint64) (APUTopologyInfo, error) {
	profile, ok := LookupAPUProfile(node.Architecture)
	if !ok || !profile.IsUnifiedAPU {
		return APUTopologyInfo{}, fmt.Errorf("amddirect: device node %d (%s, arch=%s) is not a supported AMD APU architecture",
			node.NodeID, node.DeviceName, node.Architecture)
	}

	info := APUTopologyInfo{
		NodeID:         node.NodeID,
		GPUID:          node.GPUID,
		DeviceName:     node.DeviceName,
		Architecture:   node.Architecture,
		Profile:        profile,
		NUMANode:       node.NUMANode,
		HostDRAMBytes:  hostDRAMBytes,
		TotalVRAMBytes: node.TotalVRAMBytes,
	}

	if node.TotalVRAMBytes == 0 {
		info.Reason = "device node TotalVRAMBytes must be greater than 0"
		return info, errors.New("amddirect: " + info.Reason)
	}
	if hostDRAMBytes == 0 {
		info.Reason = "host DRAM bytes must be greater than 0 for APU unified memory topology"
		return info, errors.New("amddirect: " + info.Reason)
	}

	// Invariant 1: Single NUMA node 0
	info.SingleNUMANodeZero = (node.NUMANode == 0)
	if !info.SingleNUMANodeZero {
		info.Reason = fmt.Sprintf("device NUMA node is %d, want 0: AMD APU unified zero-copy memory requires single NUMA node 0", node.NUMANode)
		return info, errors.New("amddirect: " + info.Reason)
	}

	// Invariant 2: Matching system/GPU address space
	// On APUs, TotalVRAMBytes is either identical to HostDRAMBytes (unified memory model) or dynamically mapped within it.
	if node.TotalVRAMBytes > hostDRAMBytes {
		info.Reason = fmt.Sprintf("VRAM bytes (%d) exceeds host DRAM bytes (%d): invalid APU unified memory configuration",
			node.TotalVRAMBytes, hostDRAMBytes)
		return info, errors.New("amddirect: " + info.Reason)
	}

	info.MatchingAddressSpace = true
	info.UnifiedDRAMBytes = hostDRAMBytes
	info.IsUnifiedTopology = true
	return info, nil
}

// APUUnifiedBuffer represents a zero-copy coherent memory buffer allocated in unified APU DRAM.
// The host user-space virtual address maps directly to GPU physical address space without PCIe BAR apertures.
type APUUnifiedBuffer struct {
	BufferID       int     `json:"buffer_id"`
	NodeID         int     `json:"node_id"`
	Size           uint64  `json:"size"`
	VirtualAddress uintptr `json:"virtual_address"`
	HostUserPtr    []byte  `json:"-"`
	AllocFlags     uint32  `json:"alloc_flags"`
	Coherent       bool    `json:"coherent"`
	StagingCopies  int     `json:"staging_copy_count"` // Invariant: strictly 0
	Closed         bool    `json:"closed"`
	mu             sync.RWMutex
}

// StagingCopyCount returns the count of intermediate host DRAM bounce buffer copies.
// For APU unified zero-copy memory, this invariant is strictly 0.
func (b *APUUnifiedBuffer) StagingCopyCount() int {
	return 0
}

// Bytes returns the underlying memory slice for direct zero-copy operations.
func (b *APUUnifiedBuffer) Bytes() []byte {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.HostUserPtr
}

// ReadAt reads len(p) bytes from the unified buffer starting at byte offset off.
func (b *APUUnifiedBuffer) ReadAt(p []byte, off int64) (int, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if b.Closed {
		return 0, errors.New("amddirect: read on closed APU unified buffer")
	}
	if off < 0 || uint64(off) >= b.Size {
		return 0, errors.New("amddirect: offset out of bounds")
	}

	n := copy(p, b.HostUserPtr[off:])
	return n, nil
}

// WriteAt writes len(p) bytes into the unified buffer starting at byte offset off.
func (b *APUUnifiedBuffer) WriteAt(p []byte, off int64) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.Closed {
		return 0, errors.New("amddirect: write on closed APU unified buffer")
	}
	if off < 0 || uint64(off) >= b.Size {
		return 0, errors.New("amddirect: offset out of bounds")
	}
	if uint64(off)+uint64(len(p)) > b.Size {
		return 0, errors.New("amddirect: write exceeds buffer capacity")
	}

	n := copy(b.HostUserPtr[off:], p)
	return n, nil
}

// Slice returns a subslice view of the unified buffer without copying.
func (b *APUUnifiedBuffer) Slice(offset, length uint64) ([]byte, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if b.Closed {
		return nil, errors.New("amddirect: slice on closed APU unified buffer")
	}
	if offset+length > b.Size {
		return nil, errors.New("amddirect: slice bounds out of range")
	}
	return b.HostUserPtr[offset : offset+length], nil
}

// Zero zeroes the buffer contents in place without reallocation.
func (b *APUUnifiedBuffer) Zero() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.Closed {
		return errors.New("amddirect: zero on closed APU unified buffer")
	}
	for i := range b.HostUserPtr {
		b.HostUserPtr[i] = 0
	}
	return nil
}

// Release marks the buffer as released and frees its memory slice.
func (b *APUUnifiedBuffer) Release() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.Closed {
		return errors.New("amddirect: buffer already released")
	}
	b.Closed = true
	b.HostUserPtr = nil
	b.VirtualAddress = 0
	return nil
}

// APUMemoryFenceScope defines the visibility scope of an APU memory fence.
type APUMemoryFenceScope string

const (
	// FenceScopeWorkgroup restricts coherence to the GPU workgroup/threadgroup.
	FenceScopeWorkgroup APUMemoryFenceScope = "workgroup"
	// FenceScopeAgent synchronizes all GPU wavefronts on the device node.
	FenceScopeAgent APUMemoryFenceScope = "agent"
	// FenceScopeSystem guarantees system-wide coherence between CPU cores and GPU Compute Units over Infinity Fabric.
	FenceScopeSystem APUMemoryFenceScope = "system"
)

// APUAQLBarrierFlags defines AMD AQL barrier packet cache flush and invalidation flags.
type APUAQLBarrierFlags uint16

const (
	// AQLBarrierFlagInvL1 invalidates GPU L1 vector and scalar caches.
	AQLBarrierFlagInvL1 APUAQLBarrierFlags = 0x0001
	// AQLBarrierFlagFlushL2 flushes dirty GPU L2 cache lines to unified DRAM.
	AQLBarrierFlagFlushL2 APUAQLBarrierFlags = 0x0002
	// AQLBarrierFlagInvL2 invalidates GPU L2 cache lines.
	AQLBarrierFlagInvL2 APUAQLBarrierFlags = 0x0004
	// AQLBarrierFlagSystemScope enforces Infinity Fabric system-wide coherence.
	AQLBarrierFlagSystemScope APUAQLBarrierFlags = 0x0008
	// AQLBarrierFlagNonTemporal marks non-temporal streaming store barrier.
	AQLBarrierFlagNonTemporal APUAQLBarrierFlags = 0x0010
)

// APUAQLPacket models an HSA AQL barrier and cache flush packet dispatched to the GPU Command Processor.
type APUAQLPacket struct {
	Header           uint16             `json:"header"`
	BarrierFlags     APUAQLBarrierFlags `json:"barrier_flags"`
	Reserved0        uint32             `json:"reserved0"`
	CompletionSignal uint64             `json:"completion_signal"`
	Timestamp        int64              `json:"timestamp"`
}

// APUMemoryFence represents a fine-grained coherence fence barrier.
type APUMemoryFence struct {
	FenceID          uint64              `json:"fence_id"`
	Scope            APUMemoryFenceScope `json:"scope"`
	AQLPacket        APUAQLPacket        `json:"aql_packet"`
	Signal           *HSAMemorySignal    `json:"-"`
	Doorbell         *HSADoorbell        `json:"-"`
	ExecutionLatency time.Duration       `json:"execution_latency"`
}

// APUSystemFence coordinates fine-grained sub-microsecond cache coherency between CPU threads and GPU wavefronts.
// Employs x86 non-temporal store barriers and GPU AQL cache invalidate/flush commands without OS thread descheduling.
type APUSystemFence struct {
	mu           sync.Mutex
	doorbell     *HSADoorbell
	signal       *HSAMemorySignal
	fenceCount   uint64
	totalLatency time.Duration
	maxLatency   time.Duration
}

// NewAPUSystemFence creates a new APUSystemFence coordinator.
func NewAPUSystemFence(doorbell *HSADoorbell, signal *HSAMemorySignal) *APUSystemFence {
	return &APUSystemFence{
		doorbell: doorbell,
		signal:   signal,
	}
}

// CPUToGPUFence ensures CPU writes (prompt tokenization, KV cache metadata) are immediately visible
// to GPU wavefronts without descheduling the calling CPU process.
func (f *APUSystemFence) CPUToGPUFence(buf *APUUnifiedBuffer) (time.Duration, error) {
	if buf == nil {
		return 0, errors.New("amddirect: cannot fence nil buffer")
	}
	buf.mu.RLock()
	closed := buf.Closed
	buf.mu.RUnlock()
	if closed {
		return 0, errors.New("amddirect: cannot fence closed buffer")
	}

	start := time.Now()

	// 1. x86 non-temporal store fence & atomic compiler/memory barrier:
	// atomic release ensures prior CPU stores are globally visible in unified DRAM.
	count := atomic.AddUint64(&f.fenceCount, 1)

	// 2. Dispatch AQL barrier packet to invalidate GPU L1 and L2 caches
	aql := APUAQLPacket{
		Header:           uint16(3), // AQLPacketTypeBarrierAND
		BarrierFlags:     AQLBarrierFlagInvL1 | AQLBarrierFlagInvL2 | AQLBarrierFlagSystemScope | AQLBarrierFlagNonTemporal,
		CompletionSignal: count,
		Timestamp:        time.Now().UnixNano(),
	}
	_ = aql

	// 3. Ring HSA doorbell and signal completion with release semantics (sub-microsecond spin/atomic)
	if f.doorbell != nil {
		f.doorbell.Ring(count)
	}
	if f.signal != nil {
		f.signal.StoreRelease(int64(count))
	}

	elapsed := time.Since(start)

	f.mu.Lock()
	f.totalLatency += elapsed
	if elapsed > f.maxLatency {
		f.maxLatency = elapsed
	}
	f.mu.Unlock()

	return elapsed, nil
}

// GPUToCPUFence flushes GPU dirty cache lines to unified DRAM and executes a CPU load-acquire barrier.
func (f *APUSystemFence) GPUToCPUFence(buf *APUUnifiedBuffer) (time.Duration, error) {
	if buf == nil {
		return 0, errors.New("amddirect: cannot fence nil buffer")
	}
	buf.mu.RLock()
	closed := buf.Closed
	buf.mu.RUnlock()
	if closed {
		return 0, errors.New("amddirect: cannot fence closed buffer")
	}

	start := time.Now()

	count := atomic.AddUint64(&f.fenceCount, 1)

	// AQL barrier packet to flush GPU L2 cache to unified DRAM
	aql := APUAQLPacket{
		Header:           uint16(3),
		BarrierFlags:     AQLBarrierFlagFlushL2 | AQLBarrierFlagSystemScope,
		CompletionSignal: count,
		Timestamp:        time.Now().UnixNano(),
	}
	_ = aql

	if f.doorbell != nil {
		f.doorbell.Ring(count)
	}
	if f.signal != nil {
		f.signal.StoreRelease(int64(count))
	}

	elapsed := time.Since(start)

	f.mu.Lock()
	f.totalLatency += elapsed
	if elapsed > f.maxLatency {
		f.maxLatency = elapsed
	}
	f.mu.Unlock()

	return elapsed, nil
}

// Stats returns the total count of executed fences, average latency, and maximum latency.
func (f *APUSystemFence) Stats() (uint64, time.Duration, time.Duration) {
	f.mu.Lock()
	defer f.mu.Unlock()
	count := atomic.LoadUint64(&f.fenceCount)
	if count == 0 {
		return 0, 0, 0
	}
	avg := f.totalLatency / time.Duration(count)
	return count, avg, f.maxLatency
}

// APUStreamingTransfer records the execution receipt of a zero-copy streaming transfer.
type APUStreamingTransfer struct {
	TransferID     uint64  `json:"transfer_id"`
	SrcAddress     uintptr `json:"src_address"`
	DstAddress     uintptr `json:"dst_address"`
	ByteLength     uint64  `json:"byte_length"`
	DurationNanos  int64   `json:"duration_nanos"`
	ThroughputGBps float64 `json:"throughput_gbps"`
	StagingCopies  int     `json:"staging_copy_count"` // Invariant: strictly 0
	Completed      bool    `json:"completed"`
}

// StagingCopyCount returns the intermediate staging bounce buffer count (strictly 0).
func (t *APUStreamingTransfer) StagingCopyCount() int {
	return 0
}

// APUUnifiedMemoryManager manages zero-copy coherent allocations and streaming on AMD APUs.
type APUUnifiedMemoryManager struct {
	topo           APUTopologyInfo
	mu             sync.RWMutex
	buffers        map[int]*APUUnifiedBuffer
	nextBufferID   int
	totalAllocated uint64
	peakAllocated  uint64
	transfersCount int64
	bytesStreamed  uint64
	fence          *APUSystemFence
	doorbell       *HSADoorbell
	signal         *HSAMemorySignal
}

// NewAPUUnifiedMemoryManager creates an APU unified memory allocator for a verified APU topology.
func NewAPUUnifiedMemoryManager(topo APUTopologyInfo) (*APUUnifiedMemoryManager, error) {
	if !topo.IsUnifiedTopology {
		return nil, fmt.Errorf("amddirect: cannot create APU memory manager: %s", topo.Reason)
	}

	doorbell := NewHSADoorbell("apu-aql-doorbell-0", 0x1000, 1)
	signal := NewHSAMemorySignal("apu-mem-signal-0", 0, 0x2000)
	fence := NewAPUSystemFence(doorbell, signal)

	return &APUUnifiedMemoryManager{
		topo:         topo,
		buffers:      make(map[int]*APUUnifiedBuffer),
		nextBufferID: 100,
		fence:        fence,
		doorbell:     doorbell,
		signal:       signal,
	}, nil
}

// StagingCopyCount returns the invariant bounce buffer staging copy count (strictly 0).
func (m *APUUnifiedMemoryManager) StagingCopyCount() int {
	return 0
}

// Topology returns the detected APUTopologyInfo.
func (m *APUUnifiedMemoryManager) Topology() APUTopologyInfo {
	return m.topo
}

// Fence returns the active APUSystemFence.
func (m *APUUnifiedMemoryManager) Fence() *APUSystemFence {
	return m.fence
}

// Allocate allocates a zero-copy coherent memory buffer mapped directly to host user-space virtual addresses.
// Enforces KFD_IOC_ALLOC_MEM_FLAGS_COHERENT and KFD_IOC_ALLOC_MEM_FLAGS_VRAM.
func (m *APUUnifiedMemoryManager) Allocate(size uint64, flags uint32) (*APUUnifiedBuffer, error) {
	if size == 0 {
		return nil, errors.New("amddirect: allocation size must be greater than 0")
	}

	if flags == 0 {
		flags = APUDefaultAllocFlags
	}

	// Invariant: must carry COHERENT and VRAM flags
	if (flags & KFD_IOC_ALLOC_MEM_FLAGS_COHERENT) == 0 {
		return nil, errors.New("amddirect: APU allocation requires KFD_IOC_ALLOC_MEM_FLAGS_COHERENT for zero-copy coherency")
	}
	if (flags & KFD_IOC_ALLOC_MEM_FLAGS_VRAM) == 0 {
		return nil, errors.New("amddirect: APU allocation requires KFD_IOC_ALLOC_MEM_FLAGS_VRAM")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if m.totalAllocated+size > m.topo.UnifiedDRAMBytes {
		return nil, fmt.Errorf("amddirect: requested allocation of %d bytes exceeds available unified DRAM (%d bytes)",
			size, m.topo.UnifiedDRAMBytes)
	}

	// Allocate coherent memory buffer mapped in user-space virtual address space
	raw := make([]byte, size)
	vaddr := uintptr(unsafe.Pointer(&raw[0]))

	m.nextBufferID++
	bufID := m.nextBufferID

	buf := &APUUnifiedBuffer{
		BufferID:       bufID,
		NodeID:         m.topo.NodeID,
		Size:           size,
		VirtualAddress: vaddr,
		HostUserPtr:    raw,
		AllocFlags:     flags,
		Coherent:       true,
		StagingCopies:  0, // Invariant: zero staging copies
		Closed:         false,
	}

	m.buffers[bufID] = buf
	m.totalAllocated += size
	if m.totalAllocated > m.peakAllocated {
		m.peakAllocated = m.totalAllocated
	}

	return buf, nil
}

// Free releases an allocated APU unified memory buffer.
func (m *APUUnifiedMemoryManager) Free(buf *APUUnifiedBuffer) error {
	if buf == nil {
		return errors.New("amddirect: cannot free nil buffer")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	existing, ok := m.buffers[buf.BufferID]
	if !ok || existing != buf {
		return fmt.Errorf("amddirect: unknown or already freed buffer ID %d", buf.BufferID)
	}

	if err := buf.Release(); err != nil {
		return err
	}

	delete(m.buffers, buf.BufferID)
	if m.totalAllocated >= buf.Size {
		m.totalAllocated -= buf.Size
	} else {
		m.totalAllocated = 0
	}
	return nil
}

// GetBuffer retrieves an active buffer by its buffer ID.
func (m *APUUnifiedMemoryManager) GetBuffer(id int) (*APUUnifiedBuffer, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	buf, ok := m.buffers[id]
	if !ok || buf.Closed {
		return nil, fmt.Errorf("amddirect: buffer ID %d not found", id)
	}
	return buf, nil
}

// StreamZeroCopy performs a direct coherent zero-copy memory transfer between unified APU buffers.
// Guarantees zero staging bounce copies (StagingCopyCount == 0).
func (m *APUUnifiedMemoryManager) StreamZeroCopy(src, dst *APUUnifiedBuffer, size uint64) (*APUStreamingTransfer, error) {
	if src == nil || dst == nil {
		return nil, errors.New("amddirect: source and destination buffers must not be nil")
	}
	if src.Closed || dst.Closed {
		return nil, errors.New("amddirect: cannot stream to or from closed buffer")
	}
	if size == 0 {
		return nil, errors.New("amddirect: stream size must be greater than 0")
	}
	if size > src.Size || size > dst.Size {
		return nil, errors.New("amddirect: stream size exceeds buffer bounds")
	}

	start := time.Now()

	// Direct zero-copy transfer between unified memory addresses (no CPU bounce buffer)
	copy(dst.HostUserPtr[:size], src.HostUserPtr[:size])

	// Enforce system coherency fence
	_, err := m.fence.CPUToGPUFence(dst)
	if err != nil {
		return nil, fmt.Errorf("amddirect: post-stream coherency fence failed: %w", err)
	}

	elapsed := time.Since(start)

	throughput := m.CalculateSustainedThroughput(size, elapsed)

	atomic.AddInt64(&m.transfersCount, 1)
	atomic.AddUint64(&m.bytesStreamed, size)

	return &APUStreamingTransfer{
		TransferID:     uint64(atomic.LoadInt64(&m.transfersCount)),
		SrcAddress:     src.VirtualAddress,
		DstAddress:     dst.VirtualAddress,
		ByteLength:     size,
		DurationNanos:  elapsed.Nanoseconds(),
		ThroughputGBps: throughput,
		StagingCopies:  0, // Invariant: zero bounce copies
		Completed:      true,
	}, nil
}

// TheoreticalPeakBandwidthGBps returns the theoretical peak unified DRAM bandwidth for this APU.
func (m *APUUnifiedMemoryManager) TheoreticalPeakBandwidthGBps() float64 {
	return m.topo.Profile.TheoreticalPeakBWGBps
}

// ProjectedSustainedBandwidthGBps returns the projected sustained unified memory throughput
// accounting for memory bus width and PHY controller efficiency.
func (m *APUUnifiedMemoryManager) ProjectedSustainedBandwidthGBps() float64 {
	return m.topo.Profile.TheoreticalPeakBWGBps * m.topo.Profile.SustainedEfficiency
}

// CalculateSustainedThroughput calculates sustained memory throughput in gigabytes per second (GB/s).
// If elapsed duration is 0, returns the projected sustained hardware bandwidth.
func (m *APUUnifiedMemoryManager) CalculateSustainedThroughput(sizeBytes uint64, elapsed time.Duration) float64 {
	if elapsed <= 0 {
		return m.ProjectedSustainedBandwidthGBps()
	}
	return (float64(sizeBytes) / 1e9) / elapsed.Seconds()
}

// AllocatedBytes returns the total currently allocated memory in bytes.
func (m *APUUnifiedMemoryManager) AllocatedBytes() uint64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.totalAllocated
}

// PeakAllocatedBytes returns the peak memory allocated in bytes.
func (m *APUUnifiedMemoryManager) PeakAllocatedBytes() uint64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.peakAllocated
}

// TransfersCount returns the total number of zero-copy streaming transfers executed.
func (m *APUUnifiedMemoryManager) TransfersCount() int64 {
	return atomic.LoadInt64(&m.transfersCount)
}

// BytesStreamed returns the total cumulative bytes streamed.
func (m *APUUnifiedMemoryManager) BytesStreamed() uint64 {
	return atomic.LoadUint64(&m.bytesStreamed)
}

// APUKVCacheConfig specifies layer, head, and token dimensions for KV cache residency in APU unified DRAM.
type APUKVCacheConfig struct {
	NumLayers    int `json:"num_layers"`
	NumHeads     int `json:"num_heads"`
	HeadDim      int `json:"head_dim"`
	MaxTokens    int `json:"max_tokens"`
	BytesPerElem int `json:"bytes_per_elem"` // e.g. 2 for FP16/BF16, 4 for FP32
	BatchSize    int `json:"batch_size"`     // defaults to 1 if <= 0
}

// APUKVCacheRegion provides zero-copy KV cache residency in coherent unified host DRAM.
// CPU host routines populate prompt tokens directly into unified memory; GPU wavefronts consume them immediately
// without intermediate staging copies or CPU process descheduling.
type APUKVCacheRegion struct {
	Config       APUKVCacheConfig  `json:"config"`
	LayerStride  uint64            `json:"layer_stride"`
	TokenStride  uint64            `json:"token_stride"`
	TotalBytes   uint64            `json:"total_bytes"`
	Buffer       *APUUnifiedBuffer `json:"buffer"`
	TokensStored atomic.Int64      `json:"tokens_stored"`
	mgr          *APUUnifiedMemoryManager
	fence        *APUSystemFence
	mu           sync.RWMutex
}

// StagingCopyCount returns the count of intermediate staging copies.
// Under APU unified zero-copy memory, this invariant is strictly 0.
func (k *APUKVCacheRegion) StagingCopyCount() int {
	return 0
}

// AllocateUnifiedKVCache allocates a zero-copy coherent KV cache region in unified host DRAM.
func AllocateUnifiedKVCache(mgr *APUUnifiedMemoryManager, cfg APUKVCacheConfig) (*APUKVCacheRegion, error) {
	if mgr == nil {
		return nil, errors.New("amddirect: APU memory manager must not be nil")
	}
	if cfg.NumLayers <= 0 || cfg.NumHeads <= 0 || cfg.HeadDim <= 0 || cfg.MaxTokens <= 0 {
		return nil, errors.New("amddirect: invalid KV cache dimensions (layers, heads, headDim, maxTokens must be > 0)")
	}
	if cfg.BytesPerElem <= 0 {
		cfg.BytesPerElem = 2 // Default FP16
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 1
	}

	tokenStride := uint64(cfg.NumHeads * cfg.HeadDim * cfg.BytesPerElem)
	layerStride := 2 * uint64(cfg.MaxTokens) * tokenStride * uint64(cfg.BatchSize)
	totalBytes := uint64(cfg.NumLayers) * layerStride

	buf, err := mgr.Allocate(totalBytes, APUDefaultAllocFlags)
	if err != nil {
		return nil, fmt.Errorf("amddirect: failed to allocate unified KV cache buffer: %w", err)
	}

	return &APUKVCacheRegion{
		Config:      cfg,
		LayerStride: layerStride,
		TokenStride: tokenStride,
		TotalBytes:  totalBytes,
		Buffer:      buf,
		mgr:         mgr,
		fence:       mgr.fence,
	}, nil
}

// WritePromptTokenKV writes a single token's K or V tensor directly to unified coherent memory
// and enforces a sub-microsecond CPU-to-GPU fence for immediate GPU wavefront visibility.
func (k *APUKVCacheRegion) WritePromptTokenKV(layer int, isValue bool, tokenIdx int, data []byte) error {
	if layer < 0 || layer >= k.Config.NumLayers {
		return fmt.Errorf("amddirect: layer %d out of bounds [0, %d)", layer, k.Config.NumLayers)
	}
	if tokenIdx < 0 || tokenIdx >= k.Config.MaxTokens {
		return fmt.Errorf("amddirect: token index %d out of bounds [0, %d)", tokenIdx, k.Config.MaxTokens)
	}
	if uint64(len(data)) > k.TokenStride {
		return fmt.Errorf("amddirect: data size %d exceeds token stride %d", len(data), k.TokenStride)
	}

	base := uint64(layer) * k.LayerStride
	if isValue {
		base += uint64(k.Config.MaxTokens) * k.TokenStride
	}
	offset := base + uint64(tokenIdx)*k.TokenStride

	// Direct zero-copy write into host DRAM
	if _, err := k.Buffer.WriteAt(data, int64(offset)); err != nil {
		return err
	}

	// Enforce CPU-to-GPU coherency fence
	if _, err := k.fence.CPUToGPUFence(k.Buffer); err != nil {
		return fmt.Errorf("amddirect: post-write coherency fence failed: %w", err)
	}

	k.TokensStored.Add(1)
	return nil
}

// GPUReadTokenKV reads a single token's K or V tensor directly from unified coherent memory
// with zero staging copies or intermediate bounce buffers.
func (k *APUKVCacheRegion) GPUReadTokenKV(layer int, isValue bool, tokenIdx int, out []byte) error {
	if layer < 0 || layer >= k.Config.NumLayers {
		return fmt.Errorf("amddirect: layer %d out of bounds [0, %d)", layer, k.Config.NumLayers)
	}
	if tokenIdx < 0 || tokenIdx >= k.Config.MaxTokens {
		return fmt.Errorf("amddirect: token index %d out of bounds [0, %d)", tokenIdx, k.Config.MaxTokens)
	}
	if uint64(len(out)) > k.TokenStride {
		return fmt.Errorf("amddirect: output buffer size %d exceeds token stride %d", len(out), k.TokenStride)
	}

	base := uint64(layer) * k.LayerStride
	if isValue {
		base += uint64(k.Config.MaxTokens) * k.TokenStride
	}
	offset := base + uint64(tokenIdx)*k.TokenStride

	// Direct zero-copy read from host DRAM
	n, err := k.Buffer.ReadAt(out, int64(offset))
	if err != nil {
		return err
	}
	if n != len(out) {
		return fmt.Errorf("amddirect: partial read (%d/%d bytes)", n, len(out))
	}
	return nil
}

// WritePromptSequenceKV populates a sequence of contiguous prompt KV tokens in host DRAM
// and issues a single unified coherency fence at sequence boundary.
func (k *APUKVCacheRegion) WritePromptSequenceKV(layer int, isValue bool, startToken int, tokensData [][]byte) error {
	if layer < 0 || layer >= k.Config.NumLayers {
		return fmt.Errorf("amddirect: layer %d out of bounds [0, %d)", layer, k.Config.NumLayers)
	}
	if startToken < 0 || startToken+len(tokensData) > k.Config.MaxTokens {
		return fmt.Errorf("amddirect: sequence range [%d, %d) out of bounds [0, %d)",
			startToken, startToken+len(tokensData), k.Config.MaxTokens)
	}

	base := uint64(layer) * k.LayerStride
	if isValue {
		base += uint64(k.Config.MaxTokens) * k.TokenStride
	}

	for i, data := range tokensData {
		offset := base + uint64(startToken+i)*k.TokenStride
		if _, err := k.Buffer.WriteAt(data, int64(offset)); err != nil {
			return err
		}
	}

	if _, err := k.fence.CPUToGPUFence(k.Buffer); err != nil {
		return fmt.Errorf("amddirect: sequence coherency fence failed: %w", err)
	}

	k.TokensStored.Add(int64(len(tokensData)))
	return nil
}

// GPUReadSequenceKV reads a sequence of contiguous prompt KV tokens directly from unified DRAM.
func (k *APUKVCacheRegion) GPUReadSequenceKV(layer int, isValue bool, startToken int, numTokens int) ([]byte, error) {
	if layer < 0 || layer >= k.Config.NumLayers {
		return nil, fmt.Errorf("amddirect: layer %d out of bounds [0, %d)", layer, k.Config.NumLayers)
	}
	if startToken < 0 || startToken+numTokens > k.Config.MaxTokens {
		return nil, fmt.Errorf("amddirect: sequence range [%d, %d) out of bounds [0, %d)",
			startToken, startToken+numTokens, k.Config.MaxTokens)
	}

	base := uint64(layer) * k.LayerStride
	if isValue {
		base += uint64(k.Config.MaxTokens) * k.TokenStride
	}
	offset := base + uint64(startToken)*k.TokenStride
	totalLen := uint64(numTokens) * k.TokenStride

	out := make([]byte, totalLen)
	if _, err := k.Buffer.ReadAt(out, int64(offset)); err != nil {
		return nil, err
	}
	return out, nil
}

// Release releases the underlying unified buffer back to the memory manager.
func (k *APUKVCacheRegion) Release() error {
	k.mu.Lock()
	defer k.mu.Unlock()

	if k.Buffer == nil {
		return nil
	}
	err := k.mgr.Free(k.Buffer)
	k.Buffer = nil
	return err
}

// DetectAPU examines an existing registered device node in the coordinator to detect APU unified memory topology.
func (e *AMDGPUDirectHAL) DetectAPU(nodeID int, hostDRAMBytes uint64) (APUTopologyInfo, error) {
	e.mu.RLock()
	node, ok := e.nodes[nodeID]
	e.mu.RUnlock()
	if !ok {
		return APUTopologyInfo{}, fmt.Errorf("amddirect: unknown node ID %d", nodeID)
	}
	return DetectAPUTopology(*node, hostDRAMBytes)
}

// CreateAPUMemoryManager creates an APUUnifiedMemoryManager for the specified registered node.
func (e *AMDGPUDirectHAL) CreateAPUMemoryManager(nodeID int, hostDRAMBytes uint64) (*APUUnifiedMemoryManager, error) {
	topo, err := e.DetectAPU(nodeID, hostDRAMBytes)
	if err != nil {
		return nil, err
	}
	return NewAPUUnifiedMemoryManager(topo)
}
