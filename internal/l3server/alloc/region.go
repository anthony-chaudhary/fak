package alloc

import (
	"fmt"
	"sync/atomic"
	"unsafe"
)

// Region represents a contiguous memory region (mmap'd on Linux, heap-allocated elsewhere).
type Region struct {
	data           []byte
	size           uint64
	useHuge        bool   // derived: hugePageSizeKB > 0
	hugePageSizeKB int    // 0 = disabled, 2048 = 2MB, 1048576 = 1GB
	isMapped       bool   // true if mmap'd, false if heap-allocated
	gotHuge        bool   // true if MAP_HUGETLB succeeded
	gotHugeSizeKB  int    // actual hugepage size that succeeded (2048 or 1048576)
	thpHinted      bool   // true if madvise(MADV_HUGEPAGE) was applied
	pinned         bool   // true if pages are locked in RAM via mlock
	devdaxFd       int    // -1 for anonymous mmap, >=0 for devdax
	devdaxOffset   uint64 // offset within device
	devdaxPath     string // "/dev/dax0.0" (for client advertisement)
}

// NewRegion allocates a memory region of the given size.
func NewRegion(size uint64, hugePageSizeKB int) (*Region, error) {
	if size == 0 {
		return nil, fmt.Errorf("region size must be > 0")
	}
	r := &Region{
		size:           size,
		hugePageSizeKB: hugePageSizeKB,
		useHuge:        hugePageSizeKB > 0,
		devdaxFd:       -1,
	}
	if err := r.allocate(); err != nil {
		return nil, err
	}
	return r, nil
}

// Data returns the raw byte slice for this region.
func (r *Region) Data() []byte {
	return r.data
}

// Size returns the region size in bytes.
func (r *Region) Size() uint64 {
	return r.size
}

// DataPtr returns the base pointer of the region's backing memory.
func (r *Region) DataPtr() unsafe.Pointer {
	if len(r.data) == 0 {
		return nil
	}
	return unsafe.Pointer(&r.data[0])
}

// Close releases the region memory.
func (r *Region) Close() error {
	if r.isMapped {
		return r.munmap()
	}
	r.data = nil
	return nil
}

// GotHugePages returns true if MAP_HUGETLB succeeded for this region.
func (r *Region) GotHugePages() bool {
	return r.gotHuge
}

// GotHugePageSizeKB returns the actual hugepage size that succeeded (2048 or 1048576).
func (r *Region) GotHugePageSizeKB() int {
	return r.gotHugeSizeKB
}

// HugePageSizeKB returns the requested hugepage size in KB (0 if disabled).
func (r *Region) HugePageSizeKB() int {
	return r.hugePageSizeKB
}

// THPHinted returns true if madvise(MADV_HUGEPAGE) was applied to this region.
func (r *Region) THPHinted() bool {
	return r.thpHinted
}

// Pinned returns true if the region's pages are locked in RAM via mlock.
func (r *Region) Pinned() bool {
	return r.pinned
}

// DevdaxPath returns the device path if this is a devdax-backed region, or "" otherwise.
func (r *Region) DevdaxPath() string { return r.devdaxPath }

// DevdaxOffset returns the byte offset within the devdax device for this region.
func (r *Region) DevdaxOffset() uint64 { return r.devdaxOffset }

// IsDevdax returns true if this region is backed by a devdax device.
func (r *Region) IsDevdax() bool { return r.devdaxPath != "" }

// QueryDevdaxCapacity reads the capacity of a devdax device from sysfs.
func QueryDevdaxCapacity(devPath string) (uint64, error) {
	return queryDevdaxCapacityImpl(devPath)
}

// BitmapAllocator manages allocations within a Region using a bitmap free list.
type BitmapAllocator struct {
	region    *Region
	slotSize  uint64
	numSlots  uint64
	bitmap    []uint64
	freeCount atomic.Int64
}

// NewBitmapAllocator creates a bitmap allocator for fixed-size slots within a region.
func NewBitmapAllocator(region *Region, slotSize uint64) *BitmapAllocator {
	numSlots := region.Size() / slotSize
	bitmapLen := (numSlots + 63) / 64
	bitmap := make([]uint64, bitmapLen)

	// Reserve slot 0
	bitmap[0] |= 1

	ba := &BitmapAllocator{
		region:   region,
		slotSize: slotSize,
		numSlots: numSlots,
		bitmap:   bitmap,
	}
	ba.freeCount.Store(int64(numSlots - 1))
	return ba
}

func (ba *BitmapAllocator) Alloc() (uint64, bool) {
	for i := range ba.bitmap {
		word := ba.bitmap[i]
		if word == ^uint64(0) {
			continue
		}
		bit := firstZeroBit(word)
		slotIdx := uint64(i)*64 + uint64(bit)
		if slotIdx >= ba.numSlots {
			return 0, false
		}
		ba.bitmap[i] |= 1 << bit
		ba.freeCount.Add(-1)
		return slotIdx * ba.slotSize, true
	}
	return 0, false
}

func (ba *BitmapAllocator) Free(offset uint64) {
	slotIdx := offset / ba.slotSize
	wordIdx := slotIdx / 64
	bitIdx := slotIdx % 64
	ba.bitmap[wordIdx] &^= 1 << bitIdx
	ba.freeCount.Add(1)
}

func (ba *BitmapAllocator) FreeCount() int64 {
	return ba.freeCount.Load()
}

func (ba *BitmapAllocator) NumSlots() uint64 {
	return ba.numSlots
}

func (ba *BitmapAllocator) SlotSize() uint64 {
	return ba.slotSize
}

func (ba *BitmapAllocator) SlotData(offset uint64) []byte {
	return ba.region.Data()[offset : offset+ba.slotSize]
}

func firstZeroBit(x uint64) uint {
	return uint(countTrailingOnes(x))
}

func countTrailingOnes(x uint64) int {
	return trailingZeros64(^x)
}

func trailingZeros64(x uint64) int {
	if x == 0 {
		return 64
	}
	n := 0
	if x&0xFFFFFFFF == 0 {
		n += 32
		x >>= 32
	}
	if x&0xFFFF == 0 {
		n += 16
		x >>= 16
	}
	if x&0xFF == 0 {
		n += 8
		x >>= 8
	}
	if x&0xF == 0 {
		n += 4
		x >>= 4
	}
	if x&0x3 == 0 {
		n += 2
		x >>= 2
	}
	if x&0x1 == 0 {
		n += 1
	}
	return n
}
