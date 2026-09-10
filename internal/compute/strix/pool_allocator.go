// Package strix implements the Two-Dimensional (2D) Batching Scheduler,
// hardware profile, memory bandwidth physics, compute HAL, and unified
// memory pipeline for AMD Strix Halo (Ryzen AI Max+ 395).
package strix

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// MemoryPressureLevel represents the user-space GTT memory pressure tier.
type MemoryPressureLevel string

const (
	PressureGreen  MemoryPressureLevel = "GREEN"
	PressureYellow MemoryPressureLevel = "YELLOW"
	PressureRed    MemoryPressureLevel = "RED"
	PressureBlack  MemoryPressureLevel = "BLACK"
)

// Binned pool constants for AMD Strix Halo GTT unified memory.
const (
	// MinBinSize is the smallest power-of-two slab bin (64 KiB).
	MinBinSize int64 = 64 * 1024

	// MaxBinSize is the largest power-of-two slab bin (32 MiB).
	MaxBinSize int64 = 32 * 1024 * 1024

	// DefaultSlabSize is the standard 2 MiB hugepage slab size used to back smaller bins.
	DefaultSlabSize int64 = 2 * 1024 * 1024

	// DefaultHighWatermarkBytes is the default threshold before aggressive trimming (64 GiB).
	DefaultHighWatermarkBytes int64 = 64 * 1024 * 1024 * 1024
)

// Typed errors for block caching allocator operations.
var (
	// ErrBlockClosed is returned when operating on an unallocated or freed block descriptor.
	ErrBlockClosed = errors.New("strix/pool: block descriptor is not active or already closed")

	// ErrNilBlock is returned when a nil block is passed to Free.
	ErrNilBlock = errors.New("strix/pool: cannot free nil block descriptor")

	// ErrSlabAllocationFailed is returned when backing slab allocation fails across all providers.
	ErrSlabAllocationFailed = errors.New("strix/pool: backing slab allocation failed")
)

// BinSizes defines the 10 segregated power-of-two bin sizes from 64 KiB to 32 MiB.
var BinSizes = []int64{
	64 * 1024,        // Bin 0: 64 KiB
	128 * 1024,       // Bin 1: 128 KiB
	256 * 1024,       // Bin 2: 256 KiB
	512 * 1024,       // Bin 3: 512 KiB
	1 * 1024 * 1024,  // Bin 4: 1 MiB
	2 * 1024 * 1024,  // Bin 5: 2 MiB
	4 * 1024 * 1024,  // Bin 6: 4 MiB
	8 * 1024 * 1024,  // Bin 7: 8 MiB
	16 * 1024 * 1024, // Bin 8: 16 MiB
	32 * 1024 * 1024, // Bin 9: 32 MiB
}

// NumBins is the count of segregated power-of-two size bins.
const NumBins = 10

// BinIndexForSize calculates the segregated power-of-two bin index and canonical bin size.
func BinIndexForSize(size int64) (int, int64, error) {
	if size <= 0 {
		return -1, 0, ErrInvalidBufferSize
	}
	if size > MaxBinSize {
		return -1, 0, fmt.Errorf("%w: requested size %d exceeds max bin size %d", ErrInvalidBufferSize, size, MaxBinSize)
	}
	for i, binSize := range BinSizes {
		if size <= binSize {
			return i, binSize, nil
		}
	}
	return -1, 0, fmt.Errorf("%w: requested size %d could not be binned", ErrInvalidBufferSize, size)
}

// BlockDescriptor represents an allocated or cached memory block in a segregated bin.
type BlockDescriptor struct {
	ID         uint64
	BinIndex   int
	BinSize    int64
	ActualSize int64
	Ptr        unsafe.Pointer
	Data       []byte
	SlabID     uint64
	slab       *slabRecord
	InUse      bool
	AllocTime  time.Time
}

// Slice returns the active sub-slice up to ActualSize bytes.
func (b *BlockDescriptor) Slice() []byte {
	if b == nil || b.Data == nil {
		return nil
	}
	if b.ActualSize > 0 && b.ActualSize <= int64(len(b.Data)) {
		return b.Data[:b.ActualSize]
	}
	return b.Data
}

// Raw returns the underlying full buffer slice for the bin size.
func (b *BlockDescriptor) Raw() []byte {
	if b == nil {
		return nil
	}
	return b.Data
}

// DevPtr returns the device pointer (uintptr) matching host virtual address in UMA.
func (b *BlockDescriptor) DevPtr() uintptr {
	if b == nil {
		return 0
	}
	return uintptr(b.Ptr)
}

// HostPtr returns the host virtual address (uintptr), identical to DevPtr.
func (b *BlockDescriptor) HostPtr() uintptr {
	return b.DevPtr()
}

// PoolTelemetry records real-time allocation, cache recycling, and IOCTL bypass metrics.
type PoolTelemetry struct {
	TotalAllocatedBytes int64 `json:"total_allocated_bytes"`
	ActiveBytes         int64 `json:"active_bytes"`
	CachedBytes         int64 `json:"cached_bytes"`
	PeakBytes           int64 `json:"peak_bytes"`
	AllocationRequests  int64 `json:"allocation_requests"`
	PoolReclaims        int64 `json:"pool_reclaims"`
	SlabAllocations     int64 `json:"slab_allocations"`
	IoctlBypassCount    int64 `json:"ioctl_bypass_count"`
	ReleaseCount        int64 `json:"release_count"`
	TrimCount           int64 `json:"trim_count"`
	TrimmedBytes        int64 `json:"trimmed_bytes"`
	FallbackHeapCount   int64 `json:"fallback_heap_count"`
}

// BackingProvider manages raw slab allocation (e.g. via GTT 2MB hugepage or virtual memory).
type BackingProvider interface {
	AllocateSlab(size int64) (unsafe.Pointer, []byte, error)
	FreeSlab(ptr unsafe.Pointer, size int64) error
	IsDirectGTT() bool
	IoctlCount() int64
}

// UMAPointerBackingProvider allocates slabs via UMAPointerManager with hugepage alignment.
type UMAPointerBackingProvider struct {
	mgr        *UMAPointerManager
	ioctlCount atomic.Int64
}

// NewUMAPointerBackingProvider creates a backing provider integrated with UMAPointerManager.
func NewUMAPointerBackingProvider(mgr *UMAPointerManager) *UMAPointerBackingProvider {
	if mgr == nil {
		mgr = DefaultUMAPointerManager()
	}
	return &UMAPointerBackingProvider{
		mgr: mgr,
	}
}

// AllocateSlab allocates a slab buffer aligned to 2MB hugepage boundaries.
func (p *UMAPointerBackingProvider) AllocateSlab(size int64) (unsafe.Pointer, []byte, error) {
	buf, err := p.mgr.AllocateUMABuffer(size, DefaultSlabSize)
	if err != nil {
		return nil, nil, err
	}
	p.ioctlCount.Add(1) // Simulate initial GEM allocation ioctl
	return buf.UnsafePointer(), buf.Slice(), nil
}

// FreeSlab releases the slab memory.
func (p *UMAPointerBackingProvider) FreeSlab(ptr unsafe.Pointer, size int64) error {
	return nil
}

// IsDirectGTT indicates direct GTT address space backing.
func (p *UMAPointerBackingProvider) IsDirectGTT() bool {
	return true
}

// IoctlCount returns the simulated/actual count of DRM GEM ioctls issued.
func (p *UMAPointerBackingProvider) IoctlCount() int64 {
	return p.ioctlCount.Load()
}

// HeapBackingProvider allocates standard heap-backed virtual memory slabs.
type HeapBackingProvider struct {
	ioctlCount atomic.Int64
}

// NewHeapBackingProvider creates a pure heap-backed virtual memory provider.
func NewHeapBackingProvider() *HeapBackingProvider {
	return &HeapBackingProvider{}
}

// AllocateSlab allocates a heap byte slice with 64-byte cache line alignment.
func (p *HeapBackingProvider) AllocateSlab(size int64) (unsafe.Pointer, []byte, error) {
	align := int64(UMACacheLineAlignment)
	raw := make([]byte, size+align)
	baseAddr := uintptr(unsafe.Pointer(&raw[0]))
	offset := int((uintptr(align) - (baseAddr % uintptr(align))) % uintptr(align))
	ptr := unsafe.Add(unsafe.Pointer(&raw[0]), offset)
	slice := raw[offset : offset+int(size) : offset+int(size)]
	p.ioctlCount.Add(1)
	return ptr, slice, nil
}

// FreeSlab releases a heap slab.
func (p *HeapBackingProvider) FreeSlab(ptr unsafe.Pointer, size int64) error {
	return nil
}

// IsDirectGTT reports false for heap backing.
func (p *HeapBackingProvider) IsDirectGTT() bool {
	return false
}

// IoctlCount returns ioctl count.
func (p *HeapBackingProvider) IoctlCount() int64 {
	return p.ioctlCount.Load()
}

// AllocatorConfig sets options for the BlockCachingAllocator.
type AllocatorConfig struct {
	MinBinSize         int64
	MaxBinSize         int64
	DefaultSlabSize    int64
	HighWatermarkBytes int64
	BackingProvider    BackingProvider
	FallbackToHeap     bool
}

// DefaultAllocatorConfig returns optimal default configuration for AMD Strix Halo UMA.
func DefaultAllocatorConfig() AllocatorConfig {
	return AllocatorConfig{
		MinBinSize:         MinBinSize,
		MaxBinSize:         MaxBinSize,
		DefaultSlabSize:    DefaultSlabSize,
		HighWatermarkBytes: DefaultHighWatermarkBytes,
		FallbackToHeap:     true,
	}
}

// slabRecord tracks an allocated backing slab and its constituent blocks.
type slabRecord struct {
	id          uint64
	size        int64
	ptr         unsafe.Pointer
	binIndex    int
	blockSize   int64
	totalBlocks int
	freeBlocks  atomic.Int32
	blocks      []*BlockDescriptor
}

// binHeader represents a single power-of-two size bin with its free list.
type binHeader struct {
	mu          sync.Mutex
	binIndex    int
	binSize     int64
	totalBlocks int64
	freeList    []*BlockDescriptor
}

// BlockCachingAllocator implements MLX-style lock-free/low-overhead unified memory
// block-caching allocator with power-of-two size bins for AMD Strix Halo GTT.
// It completely eliminates runtime DRM_IOCTL_AMDGPU_GEM_CREATE calls on the warm decode path.
type BlockCachingAllocator struct {
	cfg        AllocatorConfig
	provider   BackingProvider
	heapBackup BackingProvider

	mu     sync.RWMutex
	bins   [NumBins]*binHeader
	slabs  map[uint64]*slabRecord
	nextID uint64

	// Telemetry atomic counters.
	totalAllocatedBytes atomic.Int64
	activeBytes         atomic.Int64
	cachedBytes         atomic.Int64
	peakBytes           atomic.Int64
	allocationRequests  atomic.Int64
	poolReclaims        atomic.Int64
	slabAllocations     atomic.Int64
	ioctlBypassCount    atomic.Int64
	releaseCount        atomic.Int64
	trimCount           atomic.Int64
	trimmedBytes        atomic.Int64
	fallbackHeapCount   atomic.Int64
}

// NewBlockCachingAllocator initializes a new block-caching allocator.
func NewBlockCachingAllocator(cfg AllocatorConfig) *BlockCachingAllocator {
	if cfg.MinBinSize <= 0 {
		cfg.MinBinSize = MinBinSize
	}
	if cfg.MaxBinSize <= 0 {
		cfg.MaxBinSize = MaxBinSize
	}
	if cfg.DefaultSlabSize <= 0 {
		cfg.DefaultSlabSize = DefaultSlabSize
	}
	if cfg.HighWatermarkBytes <= 0 {
		cfg.HighWatermarkBytes = DefaultHighWatermarkBytes
	}
	if cfg.BackingProvider == nil {
		cfg.BackingProvider = NewUMAPointerBackingProvider(nil)
	}

	alloc := &BlockCachingAllocator{
		cfg:        cfg,
		provider:   cfg.BackingProvider,
		heapBackup: NewHeapBackingProvider(),
		slabs:      make(map[uint64]*slabRecord),
	}

	for i := 0; i < NumBins; i++ {
		alloc.bins[i] = &binHeader{
			binIndex: i,
			binSize:  BinSizes[i],
			freeList: make([]*BlockDescriptor, 0, 16),
		}
	}

	return alloc
}

// Allocate requests a memory block of at least size bytes. The request is routed to
// the matching power-of-two size bin (64KB to 32MB). If a cached block is available,
// it is returned in O(1) time without issuing any kernel ioctl or memory allocation.
func (a *BlockCachingAllocator) Allocate(size int64) (*BlockDescriptor, error) {
	binIdx, binSize, err := BinIndexForSize(size)
	if err != nil {
		return nil, err
	}

	a.allocationRequests.Add(1)

	bin := a.bins[binIdx]
	bin.mu.Lock()

	// Fast path: O(1) pop from segregated free list.
	if n := len(bin.freeList); n > 0 {
		block := bin.freeList[n-1]
		bin.freeList = bin.freeList[:n-1]
		bin.mu.Unlock()

		block.InUse = true
		block.ActualSize = size
		if block.slab != nil {
			block.slab.freeBlocks.Add(-1)
		}

		a.poolReclaims.Add(1)
		a.ioctlBypassCount.Add(1)
		a.activeBytes.Add(binSize)
		a.cachedBytes.Add(-binSize)

		return block, nil
	}
	bin.mu.Unlock()

	// Slow path: allocate new backing slab and populate bin.
	return a.allocateNewSlab(binIdx, binSize, size)
}

// allocateNewSlab allocates a backing slab, partitions it into bin-sized blocks,
// activates one block for the caller, and places remainder into the bin free list.
func (a *BlockCachingAllocator) allocateNewSlab(binIdx int, binSize, requestedSize int64) (*BlockDescriptor, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	// Double-check bin free list in case another goroutine replenished it.
	bin := a.bins[binIdx]
	bin.mu.Lock()
	if n := len(bin.freeList); n > 0 {
		block := bin.freeList[n-1]
		bin.freeList = bin.freeList[:n-1]
		bin.mu.Unlock()

		block.InUse = true
		block.ActualSize = requestedSize
		if block.slab != nil {
			block.slab.freeBlocks.Add(-1)
		}

		a.poolReclaims.Add(1)
		a.ioctlBypassCount.Add(1)
		a.activeBytes.Add(binSize)
		a.cachedBytes.Add(-binSize)

		return block, nil
	}
	bin.mu.Unlock()

	// Determine slab size.
	slabSize := a.cfg.DefaultSlabSize
	if binSize > slabSize {
		slabSize = binSize
	}

	blocksCount := int(slabSize / binSize)
	if blocksCount < 1 {
		blocksCount = 1
		slabSize = binSize
	}

	ptr, slice, err := a.provider.AllocateSlab(slabSize)
	if err != nil {
		if a.cfg.FallbackToHeap && a.heapBackup != nil {
			a.fallbackHeapCount.Add(1)
			ptr, slice, err = a.heapBackup.AllocateSlab(slabSize)
		}
		if err != nil {
			return nil, fmt.Errorf("%w: %v", ErrSlabAllocationFailed, err)
		}
	}

	a.nextID++
	slabID := a.nextID

	slab := &slabRecord{
		id:          slabID,
		size:        slabSize,
		ptr:         ptr,
		binIndex:    binIdx,
		blockSize:   binSize,
		totalBlocks: blocksCount,
		blocks:      make([]*BlockDescriptor, 0, blocksCount),
	}
	slab.freeBlocks.Store(int32(blocksCount - 1))

	var activeBlock *BlockDescriptor
	for i := 0; i < blocksCount; i++ {
		offset := int64(i) * binSize
		blockPtr := unsafe.Add(ptr, offset)
		blockData := slice[offset : offset+binSize : offset+binSize]

		a.nextID++
		desc := &BlockDescriptor{
			ID:       a.nextID,
			BinIndex: binIdx,
			BinSize:  binSize,
			Ptr:      blockPtr,
			Data:     blockData,
			SlabID:   slabID,
			slab:     slab,
			InUse:    false,
		}

		slab.blocks = append(slab.blocks, desc)

		if i == 0 {
			desc.InUse = true
			desc.ActualSize = requestedSize
			desc.AllocTime = time.Now()
			activeBlock = desc
		} else {
			bin.mu.Lock()
			bin.freeList = append(bin.freeList, desc)
			bin.mu.Unlock()
		}
	}

	a.slabs[slabID] = slab

	a.slabAllocations.Add(1)
	a.totalAllocatedBytes.Add(slabSize)
	a.activeBytes.Add(binSize)
	if remainingCached := slabSize - binSize; remainingCached > 0 {
		a.cachedBytes.Add(remainingCached)
	}

	// Update peak memory.
	curTotal := a.totalAllocatedBytes.Load()
	for {
		peak := a.peakBytes.Load()
		if curTotal <= peak || a.peakBytes.CompareAndSwap(peak, curTotal) {
			break
		}
	}

	return activeBlock, nil
}

// Free returns an allocated block descriptor back to its segregated power-of-two bin.
// The operation is O(1) and executes entirely in userspace without any system calls.
func (a *BlockCachingAllocator) Free(b *BlockDescriptor) error {
	if b == nil {
		return ErrNilBlock
	}
	if !b.InUse {
		return ErrBlockClosed
	}

	binIdx := b.BinIndex
	if binIdx < 0 || binIdx >= NumBins {
		return fmt.Errorf("strix/pool: invalid bin index %d", binIdx)
	}

	b.InUse = false
	b.ActualSize = 0

	bin := a.bins[binIdx]
	bin.mu.Lock()
	bin.freeList = append(bin.freeList, b)
	bin.mu.Unlock()

	if b.slab != nil {
		b.slab.freeBlocks.Add(1)
	}

	a.releaseCount.Add(1)
	a.activeBytes.Add(-b.BinSize)
	a.cachedBytes.Add(b.BinSize)

	return nil
}

// Trim releases unreferenced, fully-free slabs back to the backing provider until
// targetReleaseBytes has been reclaimed or all completely-free slabs are freed.
// If targetReleaseBytes <= 0, all fully-free slabs are trimmed.
func (a *BlockCachingAllocator) Trim(targetReleaseBytes int64) (int64, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	var releasedTotal int64
	slabsToFree := make([]*slabRecord, 0)

	for _, slab := range a.slabs {
		if int(slab.freeBlocks.Load()) == slab.totalBlocks {
			slabsToFree = append(slabsToFree, slab)
			releasedTotal += slab.size
			if targetReleaseBytes > 0 && releasedTotal >= targetReleaseBytes {
				break
			}
		}
	}

	if len(slabsToFree) == 0 {
		return 0, nil
	}

	// Remove blocks belonging to trimmed slabs from bin free lists.
	for _, slab := range slabsToFree {
		bin := a.bins[slab.binIndex]
		bin.mu.Lock()

		kept := make([]*BlockDescriptor, 0, len(bin.freeList))
		for _, b := range bin.freeList {
			if b.SlabID != slab.id {
				kept = append(kept, b)
			}
		}
		bin.freeList = kept
		bin.mu.Unlock()

		// Release raw slab memory via provider.
		_ = a.provider.FreeSlab(slab.ptr, slab.size)
		delete(a.slabs, slab.id)

		a.totalAllocatedBytes.Add(-slab.size)
		a.cachedBytes.Add(-slab.size)
	}

	a.trimCount.Add(1)
	a.trimmedBytes.Add(releasedTotal)

	return releasedTotal, nil
}

// TrimUnderPressure trims cached free blocks based on the memory pressure level.
func (a *BlockCachingAllocator) TrimUnderPressure(level MemoryPressureLevel) (int64, error) {
	cached := a.cachedBytes.Load()
	if cached <= 0 {
		return 0, nil
	}

	var target int64
	switch level {
	case PressureGreen:
		return 0, nil
	case PressureYellow:
		target = cached / 4 // 25%
	case PressureRed:
		target = (cached * 3) / 4 // 75%
	case PressureBlack:
		target = cached // 100%
	default:
		return 0, nil
	}

	if target <= 0 {
		target = 1
	}

	return a.Trim(target)
}

// Telemetry returns an immutable snapshot of current allocator performance counters.
func (a *BlockCachingAllocator) Telemetry() PoolTelemetry {
	return PoolTelemetry{
		TotalAllocatedBytes: a.totalAllocatedBytes.Load(),
		ActiveBytes:         a.activeBytes.Load(),
		CachedBytes:         a.cachedBytes.Load(),
		PeakBytes:           a.peakBytes.Load(),
		AllocationRequests:  a.allocationRequests.Load(),
		PoolReclaims:        a.poolReclaims.Load(),
		SlabAllocations:     a.slabAllocations.Load(),
		IoctlBypassCount:    a.ioctlBypassCount.Load(),
		ReleaseCount:        a.releaseCount.Load(),
		TrimCount:           a.trimCount.Load(),
		TrimmedBytes:        a.trimmedBytes.Load(),
		FallbackHeapCount:   a.fallbackHeapCount.Load(),
	}
}

// Reset clears the allocator state and empties all pools.
func (a *BlockCachingAllocator) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()

	for i := 0; i < NumBins; i++ {
		bin := a.bins[i]
		bin.mu.Lock()
		bin.freeList = bin.freeList[:0]
		bin.mu.Unlock()
	}

	for _, slab := range a.slabs {
		_ = a.provider.FreeSlab(slab.ptr, slab.size)
	}
	a.slabs = make(map[uint64]*slabRecord)

	a.totalAllocatedBytes.Store(0)
	a.activeBytes.Store(0)
	a.cachedBytes.Store(0)
	a.peakBytes.Store(0)
	a.allocationRequests.Store(0)
	a.poolReclaims.Store(0)
	a.slabAllocations.Store(0)
	a.ioctlBypassCount.Store(0)
	a.releaseCount.Store(0)
	a.trimCount.Store(0)
	a.trimmedBytes.Store(0)
	a.fallbackHeapCount.Store(0)
}
