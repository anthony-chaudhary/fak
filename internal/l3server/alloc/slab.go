package alloc

import (
	"fmt"
	"sort"
	"sync/atomic"
)

// Default size classes in bytes
var DefaultSizeClasses = []uint64{
	64, 128, 256, 512,
	1024, 2048, 4096, 8192, 16384, 32768, 65536, // 1K-64K
	131072, 262144, 524288, // 128K-512K
	655360, 786432, // 640K, 768K
	1048576, 1310720, 1572864, // 1M, 1.25M, 1.5M
	2097152, 2621440, 3145728, 4194304, // 2M-4M
	5242880, 6291456, 8388608, 10485760, // 5.25M-10M
	16777216, 33554432, // 16M, 32M
}

// SlabClass represents a single size class with its bitmap allocator.
type SlabClass struct {
	Size              uint64
	Allocator         *BitmapAllocator
	Region            *Region
	AllocCount        atomic.Int64
	FreeCount_        atomic.Int64
	TotalRequestBytes atomic.Int64
}

var _ Allocator = (*SlabAllocator)(nil)

type SlabAllocator struct {
	classes        []SlabClass
	maxMemory      uint64
	allocated      atomic.Int64
	useHuge        bool
	hugePageSizeKB int
	classWeights   map[uint64]float64
}

type SlabConfig struct {
	MaxMemoryBytes  uint64
	UseHugePages    bool
	HugePageSizeKB  int
	ModelPageBytes  uint64
	CustomClasses   []uint64
	MaxKeysPerClass uint64
	ClassWeights    map[uint64]float64
	DevdaxPath      string
	DevdaxCapacity  uint64
	Dedicated       bool
}

func (c SlabConfig) resolvedHugePageSizeKB() int {
	if c.HugePageSizeKB > 0 {
		return c.HugePageSizeKB
	}
	if c.UseHugePages {
		return 2048
	}
	return 0
}

func (c SlabConfig) regionHugePageSizeKB(regionSize uint64) int {
	resolved := c.resolvedHugePageSizeKB()
	if resolved == 1048576 && regionSize < 1024*1024*1024 {
		return 2048
	}
	return resolved
}

type Allocation struct {
	ClassIdx int
	Offset   uint64
	Size     uint64
}

func NewSlabAllocator(cfg SlabConfig) (*SlabAllocator, error) {
	classes := buildSizeClasses(cfg)
	if len(classes) == 0 {
		return nil, fmt.Errorf("no size classes configured")
	}
	sort.Slice(classes, func(i, j int) bool { return classes[i] < classes[j] })

	deduped := []uint64{classes[0]}
	for i := 1; i < len(classes); i++ {
		if classes[i] != classes[i-1] {
			deduped = append(deduped, classes[i])
		}
	}

	var totalWeight float64
	weights := make([]float64, len(deduped))

	if cfg.Dedicated && cfg.ClassWeights == nil {
		for i, sz := range deduped {
			if sz == cfg.ModelPageBytes {
				weights[i] = 95.0
			} else {
				weights[i] = 5.0
			}
			totalWeight += weights[i]
		}
	} else {
		modelWeights := map[uint64]float64{}
		if cfg.ModelPageBytes > 0 {
			mpb := cfg.ModelPageBytes
			modelWeights[mpb] = 100.0
			if v := mpb / 2; v > 0 {
				modelWeights[v] = 4.0
			}
			if v := mpb / 4; v > 0 {
				modelWeights[v] = 4.0
			}
			if v := mpb / 8; v > 0 {
				modelWeights[v] = 4.0
			}

			resolved := make(map[uint64]float64, len(modelWeights))
			for k, v := range modelWeights {
				resolved[k] = v
			}
			for exactSize, w := range modelWeights {
				belowIdx := sort.Search(len(deduped), func(i int) bool {
					return deduped[i] >= exactSize
				}) - 1
				if belowIdx >= 0 && deduped[belowIdx] >= exactSize/2 {
					belowWeight := w * 0.25
					if existing := resolved[deduped[belowIdx]]; belowWeight > existing {
						resolved[deduped[belowIdx]] = belowWeight
					}
				}
			}
			modelWeights = resolved
		}

		for i, sz := range deduped {
			w := 1.0
			if cfg.ClassWeights != nil {
				if cw, ok := cfg.ClassWeights[sz]; ok {
					w = cw
				} else {
					w = 0.5
				}
			} else if mw, ok := modelWeights[sz]; ok {
				w = mw
			} else if sz >= 524288 {
				if cfg.ModelPageBytes > 0 {
					w = 1.0
				} else {
					w = 8.0
				}
			} else if sz >= 65536 {
				w = 2.0
			}
			weights[i] = w
			totalWeight += w
		}
	}

	minUsefulSlots := uint64(16)
	if cfg.MaxMemoryBytes < 128*1024*1024 {
		minUsefulSlots = 2
	}
	if len(deduped) > 1 {
		largestClass := deduped[len(deduped)-1]
		var keptDeduped []uint64
		var keptWeights []float64
		for i, sz := range deduped {
			memFraction := weights[i] / totalWeight
			regionSize := uint64(float64(cfg.MaxMemoryBytes) * memFraction)
			naturalSlots := regionSize / sz
			isExplicitWeight := false
			if cfg.ClassWeights != nil {
				_, isExplicitWeight = cfg.ClassWeights[sz]
			}
			if naturalSlots >= minUsefulSlots || sz == largestClass || sz == cfg.ModelPageBytes || isExplicitWeight {
				keptDeduped = append(keptDeduped, sz)
				keptWeights = append(keptWeights, weights[i])
			}
		}
		if len(keptDeduped) > 0 && len(keptDeduped) < len(deduped) {
			deduped = keptDeduped
			weights = keptWeights
			totalWeight = 0
			for _, w := range weights {
				totalWeight += w
			}
		}
	}

	finalWeights := make(map[uint64]float64, len(deduped))
	for i, sz := range deduped {
		finalWeights[sz] = weights[i]
	}

	var devdaxOffset uint64

	sa := &SlabAllocator{
		maxMemory:      cfg.MaxMemoryBytes,
		useHuge:        cfg.resolvedHugePageSizeKB() > 0,
		hugePageSizeKB: cfg.resolvedHugePageSizeKB(),
		classWeights:   finalWeights,
	}

	for i, sz := range deduped {
		memFraction := weights[i] / totalWeight
		regionSize := uint64(float64(cfg.MaxMemoryBytes) * memFraction)
		minRegion := sz * 4
		if regionSize < minRegion {
			regionSize = minRegion
		}
		regionSize = (regionSize / sz) * sz

		if cfg.MaxKeysPerClass > 0 {
			maxRegion := cfg.MaxKeysPerClass * sz
			if regionSize > maxRegion {
				regionSize = maxRegion
			}
		}

		var region *Region
		var err error
		if cfg.DevdaxPath != "" {
			const align2MB uint64 = 2 * 1024 * 1024
			regionSize = ((regionSize + align2MB - 1) / align2MB) * align2MB
			region, err = NewDevdaxRegion(cfg.DevdaxPath, devdaxOffset, regionSize)
			devdaxOffset += regionSize
			if err == nil && cfg.DevdaxCapacity > 0 && devdaxOffset > cfg.DevdaxCapacity {
				region.Close()
				for j := 0; j < len(sa.classes); j++ {
					sa.classes[j].Region.Close()
				}
				return nil, fmt.Errorf("devdax capacity exceeded: need %d bytes but device is %d bytes",
					devdaxOffset, cfg.DevdaxCapacity)
			}
		} else {
			region, err = NewRegion(regionSize, cfg.regionHugePageSizeKB(regionSize))
		}
		if err != nil {
			for j := 0; j < len(sa.classes); j++ {
				sa.classes[j].Region.Close()
			}
			return nil, fmt.Errorf("failed to allocate region for class %d: %w", sz, err)
		}

		alloc := NewBitmapAllocator(region, sz)
		sa.classes = append(sa.classes, SlabClass{
			Size:      sz,
			Allocator: alloc,
			Region:    region,
		})
	}

	return sa, nil
}

func (sa *SlabAllocator) allocFromClass(classIdx int, requestSize uint64) (Allocation, error) {
	cls := &sa.classes[classIdx]
	offset, ok := cls.Allocator.Alloc()
	if !ok {
		return Allocation{}, fmt.Errorf("size class %d exhausted (%d/%d slots used)",
			cls.Size, cls.Allocator.NumSlots()-uint64(cls.Allocator.FreeCount()), cls.Allocator.NumSlots())
	}

	cls.AllocCount.Add(1)
	cls.TotalRequestBytes.Add(int64(requestSize))
	sa.allocated.Add(int64(cls.Size))
	return Allocation{
		ClassIdx: classIdx,
		Offset:   offset,
		Size:     cls.Size,
	}, nil
}

func (sa *SlabAllocator) Alloc(size uint64) (Allocation, error) {
	classIdx := sa.findClass(size)
	if classIdx < 0 {
		return Allocation{}, fmt.Errorf("no size class for %d bytes (max class: %d)", size, sa.classes[len(sa.classes)-1].Size)
	}
	return sa.allocFromClass(classIdx, size)
}

func (sa *SlabAllocator) AllocWithPromotion(size uint64, maxClassIdx int) (Allocation, bool, error) {
	bestFit := sa.findClass(size)
	if bestFit < 0 {
		return Allocation{}, false, fmt.Errorf("no size class for %d bytes", size)
	}
	if a, err := sa.allocFromClass(bestFit, size); err == nil {
		return a, false, nil
	}
	limit := len(sa.classes)
	if maxClassIdx >= 0 && maxClassIdx+1 < limit {
		limit = maxClassIdx + 1
	}
	for ci := bestFit + 1; ci < limit; ci++ {
		if a, err := sa.allocFromClass(ci, size); err == nil {
			return a, true, nil
		}
	}
	return Allocation{}, false, fmt.Errorf("all classes exhausted for %d bytes", size)
}

func (sa *SlabAllocator) Free(a Allocation) {
	sa.classes[a.ClassIdx].FreeCount_.Add(1)
	sa.classes[a.ClassIdx].Allocator.Free(a.Offset)
	sa.allocated.Add(-int64(a.Size))
}

func (sa *SlabAllocator) Write(a Allocation, data []byte) {
	slot := sa.classes[a.ClassIdx].Allocator.SlotData(a.Offset)
	copy(slot, data)
}

func (sa *SlabAllocator) Read(a Allocation) []byte {
	return sa.classes[a.ClassIdx].Allocator.SlotData(a.Offset)
}

func (sa *SlabAllocator) AllocatedBytes() int64 {
	return sa.allocated.Load()
}

func (sa *SlabAllocator) SlotUtilization(requestSize uint64) float64 {
	classIdx := sa.findClass(requestSize)
	if classIdx < 0 {
		return 0.0
	}
	classSize := sa.classes[classIdx].Size
	return float64(requestSize) / float64(classSize)
}

func (sa *SlabAllocator) NumClasses() int {
	return len(sa.classes)
}

func (sa *SlabAllocator) ClassInfo(i int) *SlabClass {
	return &sa.classes[i]
}

func (sa *SlabAllocator) ClassSize(i int) uint64 {
	return sa.classes[i].Size
}

type RegionInfo struct {
	ClassIdx int
	SlotSize uint64
	Region   *Region
}

func (sa *SlabAllocator) Regions() []RegionInfo {
	regions := make([]RegionInfo, len(sa.classes))
	for i := range sa.classes {
		regions[i] = RegionInfo{
			ClassIdx: i,
			SlotSize: sa.classes[i].Size,
			Region:   sa.classes[i].Region,
		}
	}
	return regions
}

func (sa *SlabAllocator) HugepageSummary() (gotHuge, thpHinted, regular int) {
	for i := range sa.classes {
		r := sa.classes[i].Region
		switch {
		case r.GotHugePages():
			gotHuge++
		case r.THPHinted():
			thpHinted++
		default:
			regular++
		}
	}
	return
}

type ClassUtilization struct {
	Size            uint64
	TotalSlots      uint64
	UsedSlots       uint64
	AllocCount      int64
	FreeCount       int64
	AvgRequestBytes float64
	SlotUtilization float64
}

func (sa *SlabAllocator) ClassUtilizations() []ClassUtilization {
	result := make([]ClassUtilization, len(sa.classes))
	for i := range sa.classes {
		cls := &sa.classes[i]
		totalSlots := cls.Allocator.NumSlots()
		freeSlots := uint64(cls.Allocator.FreeCount())
		usedSlots := totalSlots - freeSlots
		allocCount := cls.AllocCount.Load()
		freeCount := cls.FreeCount_.Load()
		totalReqBytes := cls.TotalRequestBytes.Load()

		var avgReq float64
		var slotUtil float64
		if allocCount > 0 {
			avgReq = float64(totalReqBytes) / float64(allocCount)
			slotUtil = avgReq / float64(cls.Size)
			if slotUtil > 1.0 {
				slotUtil = 1.0
			}
		}

		result[i] = ClassUtilization{
			Size:            cls.Size,
			TotalSlots:      totalSlots,
			UsedSlots:       usedSlots,
			AllocCount:      allocCount,
			FreeCount:       freeCount,
			AvgRequestBytes: avgReq,
			SlotUtilization: slotUtil,
		}
	}
	return result
}

func (sa *SlabAllocator) ResetCounters() {
	for i := range sa.classes {
		sa.classes[i].AllocCount.Store(0)
		sa.classes[i].FreeCount_.Store(0)
		sa.classes[i].TotalRequestBytes.Store(0)
	}
}

func (sa *SlabAllocator) Close() error {
	for i := range sa.classes {
		if err := sa.classes[i].Region.Close(); err != nil {
			return err
		}
	}
	return nil
}

func (sa *SlabAllocator) FindClass(size uint64) int {
	return sa.findClass(size)
}

func (sa *SlabAllocator) findClass(size uint64) int {
	idx := sort.Search(len(sa.classes), func(i int) bool {
		return sa.classes[i].Size >= size
	})
	if idx >= len(sa.classes) {
		return -1
	}
	return idx
}

func (sa *SlabAllocator) ModelClassCapacity(valueSize uint64) (totalSlots uint64, classSize uint64) {
	idx := sa.findClass(valueSize)
	if idx < 0 {
		return 0, 0
	}
	return sa.classes[idx].Allocator.NumSlots(), sa.classes[idx].Size
}

func (sa *SlabAllocator) CurrentWeights() map[uint64]float64 {
	out := make(map[uint64]float64, len(sa.classWeights))
	for k, v := range sa.classWeights {
		out[k] = v
	}
	return out
}

func buildSizeClasses(cfg SlabConfig) []uint64 {
	if cfg.Dedicated && cfg.ModelPageBytes > 0 {
		return []uint64{256, cfg.ModelPageBytes}
	}

	classes := make([]uint64, len(DefaultSizeClasses))
	copy(classes, DefaultSizeClasses)

	if cfg.ModelPageBytes > 0 {
		mpb := cfg.ModelPageBytes
		classes = append(classes, mpb)
		if mpb/8 > 0 {
			classes = append(classes, mpb/8)
		}
		if mpb/4 > 0 {
			classes = append(classes, mpb/4)
		}
		if mpb/2 > 0 {
			classes = append(classes, mpb/2)
		}
	}

	classes = append(classes, cfg.CustomClasses...)
	return classes
}
