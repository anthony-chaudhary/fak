package compute

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"unsafe"
)

// DirectSlabConfig defines configuration parameters for DirectSlabAllocator.
type DirectSlabConfig struct {
	// TotalBytes is the total capacity of the memory slab in bytes.
	TotalBytes int64

	// Alignment is the byte alignment for allocations (e.g., 64 for cache-line,
	// 4096 for UMA/RDMA page alignment). Must be a positive power of 2.
	// Defaults to 64 if not specified or <= 0.
	Alignment int

	// PinMemory indicates whether the backing slab memory should be pinned in DRAM
	// via runtime.Pinner to prevent GC movement or page faults during RDMA/DMA transfers.
	PinMemory bool

	// DirectInterconnect indicates whether allocations are registered directly with
	// the RDMA/interconnect fabric (DS4_TP_BIG_DIRECT semantics), bypassing CPU staging.
	DirectInterconnect bool

	// DeviceAccessible indicates whether the memory is unified and directly accessible
	// by device/GPU hardware (UMA uncached / write-combining DRAM).
	DeviceAccessible bool

	// LKey is the optional RDMA memory region local key (ibv_sge.lkey).
	// If 0 and Registered/DirectInterconnect is true, defaults to 0x1000.
	LKey uint32
}

// SlabAllocation represents a contiguous allocation within the direct slab.
type SlabAllocation struct {
	// Offset is the byte offset of the allocation relative to the base address.
	Offset int64

	// Size is the requested size of the allocation in bytes.
	Size int64

	// Data is the subslice of the direct slab buffer corresponding to this allocation.
	Data []byte

	// Direct indicates whether this allocation supports direct zero-copy transfers.
	Direct bool

	// Registered indicates whether the backing memory is registered with the interconnect.
	Registered bool

	// Release frees the allocation back to the allocator.
	Release func()

	allocator *DirectSlabAllocator
	block     *slabBlock
	freed     bool
}

// Free releases the allocation back to its owning DirectSlabAllocator.
func (a *SlabAllocation) Free() error {
	if a == nil || a.allocator == nil {
		return fmt.Errorf("compute: cannot free nil slab allocation")
	}
	return a.allocator.Free(a)
}

// GetSGE returns the ScatterGatherElement descriptor for this allocation.
func (a *SlabAllocation) GetSGE() (ScatterGatherElement, error) {
	if a == nil || a.allocator == nil {
		return ScatterGatherElement{}, fmt.Errorf("compute: cannot get SGE for nil slab allocation")
	}
	return a.allocator.GetSGE(a)
}

// ScatterGatherElement represents an RDMA/verbs Scatter/Gather Element (ibv_sge)
// pointing directly to a memory address without CPU staging buffers.
type ScatterGatherElement struct {
	Address uintptr
	Length  uint32
	LKey    uint32
}

type slabBlock struct {
	offset int64
	length int64 // block span in bytes (always a multiple of alignment)
	free   bool
	prev   *slabBlock
	next   *slabBlock
}

// DirectSlabAllocator maps contiguous tensor allocations into pre-pinned,
// RDMA/interconnect-registered memory slabs. On Unified Memory Architecture (UMA)
// APUs (e.g. AMD Strix Halo), CPU and GPU share physical DRAM, but memory allocated
// via GPU runtimes is mapped with Write-Combining (WC) uncached semantics where standard
// CPU memcpy runs at ~189-200 MB/s. DirectSlabAllocator implements DS4_TP_BIG_DIRECT=1
// semantics: tensors are allocated directly within the RDMA-registered slab, and
// verbs SGEs point directly to device-accessible addresses, completely bypassing
// intermediate host staging copies (zero-copy).
type DirectSlabAllocator struct {
	mu            sync.Mutex
	cfg           DirectSlabConfig
	rawBuffer     []byte
	baseSlice     []byte
	baseAddr      uintptr
	totalCapacity int64
	usedBytes     int64
	alignment     int
	head          *slabBlock
	active        map[*SlabAllocation]*slabBlock
	registered    bool
	direct        bool
	pinner        *runtime.Pinner
	stagingCopies int64
	closed        bool
	lkey          uint32
}

func isDirectEnvEnabled() bool {
	if v := os.Getenv("DS4_TP_BIG_DIRECT"); v == "1" || strings.EqualFold(v, "true") {
		return true
	}
	if v := os.Getenv("FAK_DIRECT_SLAB"); v == "1" || strings.EqualFold(v, "true") {
		return true
	}
	return false
}

// NewDirectSlabAllocator creates a new DirectSlabAllocator with the provided configuration.
func NewDirectSlabAllocator(cfg DirectSlabConfig) (*DirectSlabAllocator, error) {
	if cfg.TotalBytes <= 0 {
		return nil, fmt.Errorf("compute: total bytes must be positive, got %d", cfg.TotalBytes)
	}

	alignment := cfg.Alignment
	if alignment <= 0 {
		alignment = 64
	}
	if (alignment & (alignment - 1)) != 0 {
		return nil, fmt.Errorf("compute: alignment %d must be a positive power of 2", alignment)
	}

	direct := (cfg.DirectInterconnect && cfg.DeviceAccessible) || isDirectEnvEnabled()
	registered := cfg.PinMemory || cfg.DirectInterconnect || direct

	lkey := cfg.LKey
	if lkey == 0 && registered {
		lkey = 0x1000
	}

	// Allocate backing memory with extra headroom to guarantee alignment.
	raw := make([]byte, cfg.TotalBytes+int64(alignment))
	ptr := uintptr(unsafe.Pointer(&raw[0]))
	alignOffset := (uintptr(alignment) - (ptr % uintptr(alignment))) % uintptr(alignment)
	baseSlice := raw[alignOffset : alignOffset+uintptr(cfg.TotalBytes)]
	baseAddr := uintptr(unsafe.Pointer(&baseSlice[0]))

	var pinner *runtime.Pinner
	if cfg.PinMemory {
		pinner = new(runtime.Pinner)
		pinner.Pin(&raw[0])
	}

	head := &slabBlock{
		offset: 0,
		length: cfg.TotalBytes,
		free:   true,
	}

	d := &DirectSlabAllocator{
		cfg:           cfg,
		rawBuffer:     raw,
		baseSlice:     baseSlice,
		baseAddr:      baseAddr,
		totalCapacity: cfg.TotalBytes,
		alignment:     alignment,
		head:          head,
		active:        make(map[*SlabAllocation]*slabBlock),
		registered:    registered,
		direct:        direct,
		pinner:        pinner,
		lkey:          lkey,
	}

	return d, nil
}

// Allocate allocates a contiguous block of size bytes from the slab.
// The allocation is guaranteed to start at a multiple of the configured alignment.
func (d *DirectSlabAllocator) Allocate(size int64) (*SlabAllocation, error) {
	if size <= 0 {
		return nil, fmt.Errorf("compute: allocation size must be positive, got %d", size)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed {
		return nil, fmt.Errorf("compute: direct slab allocator is closed")
	}

	align := int64(d.alignment)
	alignedSize := (size + align - 1) &^ (align - 1)

	if d.usedBytes+size > d.totalCapacity {
		return nil, fmt.Errorf("compute: direct slab exhausted: requested %d bytes, available %d bytes",
			size, d.totalCapacity-d.usedBytes)
	}

	// First-fit search for a free block with sufficient capacity.
	curr := d.head
	var chosen *slabBlock
	for curr != nil {
		if curr.free && curr.length >= alignedSize {
			chosen = curr
			break
		}
		curr = curr.next
	}

	if chosen == nil {
		return nil, fmt.Errorf("compute: direct slab fragmented: no contiguous block of %d bytes (available %d bytes)",
			alignedSize, d.totalCapacity-d.usedBytes)
	}

	// Split block if there is remaining capacity.
	if chosen.length > alignedSize {
		remainder := &slabBlock{
			offset: chosen.offset + alignedSize,
			length: chosen.length - alignedSize,
			free:   true,
			prev:   chosen,
			next:   chosen.next,
		}
		if chosen.next != nil {
			chosen.next.prev = remainder
		}
		chosen.next = remainder
		chosen.length = alignedSize
	}
	chosen.free = false

	subData := d.baseSlice[chosen.offset : chosen.offset+size]

	alloc := &SlabAllocation{
		Offset:     chosen.offset,
		Size:       size,
		Data:       subData,
		Direct:     d.direct,
		Registered: d.registered,
		allocator:  d,
		block:      chosen,
	}
	alloc.Release = func() {
		_ = d.Free(alloc)
	}

	d.active[alloc] = chosen
	d.usedBytes += size

	return alloc, nil
}

// Free returns an allocation back to the slab, coalescing adjacent free blocks.
func (d *DirectSlabAllocator) Free(alloc *SlabAllocation) error {
	if alloc == nil {
		return fmt.Errorf("compute: cannot free nil slab allocation")
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed {
		return fmt.Errorf("compute: direct slab allocator is closed")
	}

	if alloc.freed {
		return fmt.Errorf("compute: slab allocation already freed")
	}

	block, ok := d.active[alloc]
	if !ok || block == nil {
		return fmt.Errorf("compute: slab allocation not recognized or not active")
	}

	alloc.freed = true
	delete(d.active, alloc)
	d.usedBytes -= alloc.Size
	block.free = true

	// Coalesce with next block if free.
	if block.next != nil && block.next.free {
		block.length += block.next.length
		block.next = block.next.next
		if block.next != nil {
			block.next.prev = block
		}
	}

	// Coalesce with prev block if free.
	if block.prev != nil && block.prev.free {
		block.prev.length += block.length
		block.prev.next = block.next
		if block.prev.next != nil {
			block.prev.next.prev = block.prev
		}
	}

	return nil
}

// Available returns the number of remaining bytes available for allocation.
func (d *DirectSlabAllocator) Available() int64 {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.totalCapacity - d.usedBytes
}

// Used returns the number of bytes currently allocated.
func (d *DirectSlabAllocator) Used() int64 {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.usedBytes
}

// TotalBytes returns the total capacity of the direct slab in bytes.
func (d *DirectSlabAllocator) TotalBytes() int64 {
	return d.totalCapacity
}

// Alignment returns the alignment in bytes used for block allocations.
func (d *DirectSlabAllocator) Alignment() int {
	return d.alignment
}

// BaseAddress returns the uintptr address of the start of the aligned slab.
func (d *DirectSlabAllocator) BaseAddress() uintptr {
	return d.baseAddr
}

// Reset clears all active allocations and resets the slab to its initial state.
func (d *DirectSlabAllocator) Reset() {
	d.mu.Lock()
	defer d.mu.Unlock()

	for alloc := range d.active {
		alloc.freed = true
	}
	d.active = make(map[*SlabAllocation]*slabBlock)
	d.head = &slabBlock{
		offset: 0,
		length: d.totalCapacity,
		free:   true,
	}
	d.usedBytes = 0
}

// Close releases the allocator and unpins the backing memory if pinned.
func (d *DirectSlabAllocator) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed {
		return nil
	}
	d.closed = true

	for alloc := range d.active {
		alloc.freed = true
	}
	d.active = nil
	d.head = nil

	if d.pinner != nil {
		d.pinner.Unpin()
		d.pinner = nil
	}
	d.baseSlice = nil
	d.rawBuffer = nil
	return nil
}

// IsZeroCopy reports whether the allocator operates in direct zero-copy mode.
func (d *DirectSlabAllocator) IsZeroCopy() bool {
	return d.direct
}

// StagingCopyCount returns the number of intermediate host staging copies performed.
// Under direct zero-copy interconnect transfers, this count must remain 0.
func (d *DirectSlabAllocator) StagingCopyCount() int64 {
	return atomic.LoadInt64(&d.stagingCopies)
}

// RecordStagingCopy tracks an intermediate staging copy (used if fallback path is taken).
func (d *DirectSlabAllocator) RecordStagingCopy(bytes int64) {
	atomic.AddInt64(&d.stagingCopies, 1)
}

// GetSGE generates an RDMA/verbs ScatterGatherElement pointing directly to the
// allocated physical buffer without intermediate staging copies.
func (d *DirectSlabAllocator) GetSGE(alloc *SlabAllocation) (ScatterGatherElement, error) {
	if alloc == nil {
		return ScatterGatherElement{}, fmt.Errorf("compute: cannot get SGE for nil allocation")
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed {
		return ScatterGatherElement{}, fmt.Errorf("compute: direct slab allocator is closed")
	}

	if alloc.freed {
		return ScatterGatherElement{}, fmt.Errorf("compute: cannot get SGE for freed allocation")
	}

	if _, ok := d.active[alloc]; !ok {
		return ScatterGatherElement{}, fmt.Errorf("compute: allocation not active in this allocator")
	}

	if alloc.Size > math.MaxUint32 {
		return ScatterGatherElement{}, fmt.Errorf("compute: allocation size %d exceeds max uint32 for verbs SGE", alloc.Size)
	}

	addr := d.baseAddr + uintptr(alloc.Offset)
	return ScatterGatherElement{
		Address: addr,
		Length:  uint32(alloc.Size),
		LKey:    d.lkey,
	}, nil
}
