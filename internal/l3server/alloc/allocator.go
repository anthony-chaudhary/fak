package alloc

// Allocator is the interface that both SlabAllocator and OffsetAllocator satisfy.
// It provides memory allocation, read/write, and metrics for a shard's value store.
type Allocator interface {
	// Alloc allocates memory for the given size.
	Alloc(size uint64) (Allocation, error)
	// AllocWithPromotion tries best-fit, then walks up to larger classes.
	// maxClassIdx caps how far up to search (-1 = no cap).
	// Returns (allocation, promoted bool, error).
	AllocWithPromotion(size uint64, maxClassIdx int) (Allocation, bool, error)
	// Free releases an allocation.
	Free(a Allocation)
	// Write copies data into an allocation's memory.
	Write(a Allocation, data []byte)
	// Read returns the byte slice backing an allocation.
	Read(a Allocation) []byte

	// AllocatedBytes returns total currently allocated bytes.
	AllocatedBytes() int64
	// SlotUtilization returns the slot utilization for a given request size.
	// Returns avgRequestBytes / classSize (1.0 = perfect fit, lower = more waste).
	SlotUtilization(requestSize uint64) float64

	// Regions returns all memory regions (for RDMA MR registration).
	Regions() []RegionInfo
	// HugepageSummary returns counts by hugepage backing type.
	HugepageSummary() (gotHuge, thpHinted, regular int)

	// ClassUtilizations returns per-class usage statistics.
	ClassUtilizations() []ClassUtilization
	// ResetCounters zeroes allocation tracking counters.
	ResetCounters()
	// NumClasses returns the number of size classes (1 for offset allocator).
	NumClasses() int
	// ClassSize returns the slot size for class i.
	ClassSize(i int) uint64
	// FindClass returns the index of the smallest class that fits size, or -1.
	FindClass(size uint64) int
	// ModelClassCapacity returns capacity info for the given value size.
	ModelClassCapacity(valueSize uint64) (totalSlots, classSize uint64)

	// CurrentWeights returns the class weight map used at construction time.
	// Keyed by class size in bytes.
	CurrentWeights() map[uint64]float64

	// Close releases all resources.
	Close() error
}

// OffsetAllocatorConfig configures the offset allocator.
type OffsetAllocatorConfig struct {
	MaxMemoryBytes uint64
	UseHugePages   bool // deprecated: use HugePageSizeKB instead (kept for backward compat)
	HugePageSizeKB int  // 0 = disabled, 2048 = 2MB, 1048576 = 1GB (takes precedence over UseHugePages)
	MaxAllocations uint32 // determines node pool size; 0 = auto
	DevdaxPath     string // if set, back region with this devdax device
}

// resolvedHugePageSizeKB returns the effective hugepage size for the config.
// HugePageSizeKB takes precedence; falls back to UseHugePages bool → 2048.
func (c OffsetAllocatorConfig) resolvedHugePageSizeKB() int {
	if c.HugePageSizeKB > 0 {
		return c.HugePageSizeKB
	}
	if c.UseHugePages {
		return 2048
	}
	return 0
}

// NewAllocator creates an Allocator based on the mode string.
// mode "slab" (default) uses SlabAllocator; mode "offset" uses OffsetAllocator.
func NewAllocator(mode string, slabCfg SlabConfig, offsetCfg OffsetAllocatorConfig) (Allocator, error) {
	switch mode {
	case "offset":
		return NewOffsetAllocator(offsetCfg)
	default:
		return NewSlabAllocator(slabCfg)
	}
}
