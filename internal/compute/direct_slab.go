package compute

import (
	"fmt"
	"math"
	"runtime"
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

	// PinMemory pins the Go object against GC movement. It does not lock pages,
	// register memory with a transport, or map memory into a device.
	PinMemory bool

	// DirectInterconnect requests direct transport policy, not registration evidence.
	DirectInterconnect bool

	// DeviceAccessible requests device access, not evidence of a device mapping.
	DeviceAccessible bool

	// LKey is a legacy requested key. Only a provider receipt can supply the
	// actual transport key; this value does not establish registration.
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

	// Direct snapshots provider-attested device eligibility at allocation time.
	// GetSGE revalidates the live receipt; this is not a transfer measurement.
	Direct bool

	// Registered snapshots provider registration at allocation time.
	// It is not independent authority to use an address after Free.
	Registered bool

	// Release frees the allocation back to the allocator.
	Release func()

	allocator *DirectSlabAllocator
	block     *slabBlock
	freed     bool
	size      int64 // immutable requested length; exported metadata is not authority
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

// DirectSlabAllocator suballocates aligned Go heap storage. Physical transport
// registration and same-address device access require a live provider receipt;
// configuration flags, environment variables and GC pinning are not evidence.
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
	pinner        *runtime.Pinner
	stagingCopies int64
	closed        bool
	registration  directSlabRegistrationFacts
	owner         directSlabRegistrationOwner
	closeErr      error
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

	// Both the backing slice length and its alignment headroom must fit in int.
	maxInt := int64(int(^uint(0) >> 1))
	if cfg.TotalBytes > maxInt-int64(alignment) {
		return nil, fmt.Errorf("compute: slab capacity %d plus alignment %d exceeds addressable size", cfg.TotalBytes, alignment)
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
		pinner:        pinner,
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

	// Check by subtraction before rounding, so oversized requests cannot wrap
	// into a negative block span and mutate the free list.
	if size > d.totalCapacity-d.usedBytes {
		return nil, fmt.Errorf("compute: direct slab exhausted: requested %d bytes, available %d bytes",
			size, d.totalCapacity-d.usedBytes)
	}

	align := int64(d.alignment)
	alignedSize := (size + align - 1) &^ (align - 1)

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
	registered := d.registrationLiveLocked()

	alloc := &SlabAllocation{
		Offset:     chosen.offset,
		Size:       size,
		size:       size,
		Data:       subData,
		Direct:     registered && d.registration.deviceAccessible,
		Registered: registered,
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
	d.usedBytes -= alloc.size
	alloc.Direct = false
	alloc.Registered = false
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

// Reset clears heap allocations. Provider-backed users must be freed first.
func (d *DirectSlabAllocator) Reset() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed {
		return fmt.Errorf("compute: direct slab allocator is closed")
	}
	if d.owner != nil && len(d.active) != 0 {
		return fmt.Errorf("compute: registered slab has live allocations")
	}
	for alloc := range d.active {
		alloc.freed = true
		alloc.Direct = false
		alloc.Registered = false
	}
	d.active = make(map[*SlabAllocation]*slabBlock)
	d.head = &slabBlock{
		offset: 0,
		length: d.totalCapacity,
		free:   true,
	}
	d.usedBytes = 0
	return nil
}

// Close refuses to retire a registration with live allocations. Provider release
// is attempted once. On release failure the owner and backing remain retained,
// physical access is disabled, and repeated Close returns the same error.
func (d *DirectSlabAllocator) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.closed {
		return d.closeErr
	}
	if d.owner != nil && len(d.active) != 0 {
		return fmt.Errorf("compute: registered slab has live allocations")
	}
	d.closed = true
	if d.owner != nil {
		d.closeErr = d.owner.release()
		if d.closeErr != nil {
			return d.closeErr
		}
		d.owner = nil
		d.registration = directSlabRegistrationFacts{}
	}

	for alloc := range d.active {
		alloc.freed = true
		alloc.Direct = false
		alloc.Registered = false
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

// IsZeroCopy reports live provider-attested same-address device eligibility.
// It does not measure transfers or establish device qualification.
func (d *DirectSlabAllocator) IsZeroCopy() bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.registrationLiveLocked() && d.registration.deviceAccessible
}

// StagingCopyCount returns the number of intermediate host staging copies performed.
// A zero count alone establishes no registration or device-access capability.
func (d *DirectSlabAllocator) StagingCopyCount() int64 {
	return atomic.LoadInt64(&d.stagingCopies)
}

// RecordStagingCopy tracks an intermediate staging copy (used if fallback path is taken).
func (d *DirectSlabAllocator) RecordStagingCopy(bytes int64) {
	atomic.AddInt64(&d.stagingCopies, 1)
}

// GetSGE borrows a provider-registered descriptor until alloc.Free.
// Callers must retain the allocation until all descriptor users have completed.
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

	block, ok := d.active[alloc]
	if !ok || block == nil {
		return ScatterGatherElement{}, fmt.Errorf("compute: allocation not active in this allocator")
	}

	if alloc.Offset != block.offset || alloc.Size != alloc.size || alloc.Size <= 0 ||
		alloc.Offset < 0 || alloc.Offset > d.totalCapacity || alloc.Size > d.totalCapacity-alloc.Offset {
		return ScatterGatherElement{}, fmt.Errorf("compute: slab allocation bounds do not match its owner")
	}
	if !d.registrationLiveLocked() {
		return ScatterGatherElement{}, fmt.Errorf("compute: slab has no live provider registration")
	}

	if alloc.Size > math.MaxUint32 {
		return ScatterGatherElement{}, fmt.Errorf("compute: allocation size %d exceeds max uint32 for verbs SGE", alloc.Size)
	}

	addr := d.baseAddr + uintptr(alloc.Offset)
	return ScatterGatherElement{
		Address: addr,
		Length:  uint32(alloc.Size),
		LKey:    d.registration.lkey,
	}, nil
}
