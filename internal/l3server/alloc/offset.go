package alloc

import (
	"fmt"
	"math/bits"
	"sync/atomic"
)

// Constants for the 256-bin floating-point distribution.
const (
	mantissaBits  = 3
	mantissaValue = 1 << mantissaBits // 8
	mantissaMask  = mantissaValue - 1 // 0x7
	numBins       = 256
	numTopBins    = numBins / 8 // 32
	noNode        = ^uint32(0)  // sentinel for "no node"
)

type offsetNode struct {
	dataOffset   uint64
	dataSize     uint64
	binListPrev  uint32
	binListNext  uint32
	neighborPrev uint32
	neighborNext uint32
	used         bool
}

type OffsetAllocator struct {
	region     *Region
	regionSize uint64

	nodes        []offsetNode
	freeNodes    []uint32          // stack of unused node pool indices
	offsetToNode map[uint64]uint32 // byte offset -> node index

	binListHeads [numBins]uint32
	usedBinsTop  uint32            // 1 bit per top-level group (32 groups)
	usedBins     [numTopBins]uint8 // 1 bit per leaf bin within each group

	allocated     atomic.Int64
	allocCount    atomic.Int64
	freeCount_    atomic.Int64
	totalReqBytes atomic.Int64
}

var _ Allocator = (*OffsetAllocator)(nil)

func NewOffsetAllocator(cfg OffsetAllocatorConfig) (*OffsetAllocator, error) {
	if cfg.MaxMemoryBytes == 0 {
		return nil, fmt.Errorf("MaxMemoryBytes must be > 0")
	}

	region, err := NewRegion(cfg.MaxMemoryBytes, cfg.resolvedHugePageSizeKB())
	if err != nil {
		return nil, fmt.Errorf("failed to allocate region: %w", err)
	}

	maxAllocs := cfg.MaxAllocations
	if maxAllocs == 0 {
		maxAllocs = uint32(cfg.MaxMemoryBytes / 64)
		if maxAllocs > 4194304 {
			maxAllocs = 4194304
		}
	}
	maxNodes := maxAllocs*2 + uint32(numBins)

	oa := &OffsetAllocator{
		region:       region,
		regionSize:   cfg.MaxMemoryBytes,
		nodes:        make([]offsetNode, maxNodes),
		freeNodes:    make([]uint32, 0, maxNodes-1),
		offsetToNode: make(map[uint64]uint32, maxAllocs),
	}

	for i := range oa.binListHeads {
		oa.binListHeads[i] = noNode
	}

	for i := maxNodes - 1; i >= 1; i-- {
		oa.freeNodes = append(oa.freeNodes, i)
	}

	const reservedOffset uint64 = 64
	freeStart := reservedOffset
	freeSize := cfg.MaxMemoryBytes - reservedOffset

	oa.nodes[0] = offsetNode{
		dataOffset:   freeStart,
		dataSize:     freeSize,
		binListPrev:  noNode,
		binListNext:  noNode,
		neighborPrev: noNode,
		neighborNext: noNode,
		used:         false,
	}
	oa.offsetToNode[freeStart] = 0

	bin := uintToFloatRoundDown(freeSize)
	oa.insertIntoBin(0, bin)

	return oa, nil
}

func uintToFloatRoundUp(size uint64) uint32 {
	if size == 0 {
		return 0
	}
	if size < mantissaValue {
		return uint32(size)
	}
	highestBit := uint32(bits.Len64(size) - 1)
	mantissaStartBit := highestBit - mantissaBits
	exp := mantissaStartBit + 1

	rawMantissa := (size + (1 << mantissaStartBit) - 1) >> mantissaStartBit

	if rawMantissa >= 2*uint64(mantissaValue) {
		exp++
		rawMantissa >>= 1
	}

	mantissa := uint32(rawMantissa) & mantissaMask
	bin := (exp << mantissaBits) | mantissa
	if bin >= numBins {
		return numBins - 1
	}
	return bin
}

func uintToFloatRoundDown(size uint64) uint32 {
	if size == 0 {
		return 0
	}
	if size < mantissaValue {
		return uint32(size)
	}
	highestBit := uint32(bits.Len64(size) - 1)
	mantissaStartBit := highestBit - mantissaBits
	exp := mantissaStartBit + 1
	mantissa := uint32(size>>mantissaStartBit) & mantissaMask

	bin := (exp << mantissaBits) | mantissa
	if bin >= numBins {
		return numBins - 1
	}
	return bin
}

func binSize(bin uint32) uint64 {
	exp := bin >> mantissaBits
	mantissa := bin & mantissaMask
	if exp == 0 {
		return uint64(mantissa)
	}
	return uint64(mantissa|mantissaValue) << (exp - 1)
}

func (oa *OffsetAllocator) insertIntoBin(nodeIdx uint32, bin uint32) {
	node := &oa.nodes[nodeIdx]
	head := oa.binListHeads[bin]

	node.binListPrev = noNode
	node.binListNext = head
	if head != noNode {
		oa.nodes[head].binListPrev = nodeIdx
	}
	oa.binListHeads[bin] = nodeIdx

	topBin := bin / 8
	leafBin := bin % 8
	oa.usedBins[topBin] |= 1 << leafBin
	oa.usedBinsTop |= 1 << topBin
}

func (oa *OffsetAllocator) removeFromBin(nodeIdx uint32, bin uint32) {
	node := &oa.nodes[nodeIdx]

	if node.binListPrev != noNode {
		oa.nodes[node.binListPrev].binListNext = node.binListNext
	} else {
		oa.binListHeads[bin] = node.binListNext
	}
	if node.binListNext != noNode {
		oa.nodes[node.binListNext].binListPrev = node.binListPrev
	}
	node.binListPrev = noNode
	node.binListNext = noNode

	if oa.binListHeads[bin] == noNode {
		topBin := bin / 8
		leafBin := bin % 8
		oa.usedBins[topBin] &^= 1 << leafBin
		if oa.usedBins[topBin] == 0 {
			oa.usedBinsTop &^= 1 << topBin
		}
	}
}

func (oa *OffsetAllocator) findBin(minBin uint32) (uint32, bool) {
	topBin := minBin / 8
	leafBin := minBin % 8

	topMask := oa.usedBins[topBin] >> leafBin
	if topMask != 0 {
		return topBin*8 + leafBin + uint32(bits.TrailingZeros8(topMask)), true
	}

	if topBin+1 >= numTopBins {
		return 0, false
	}
	topBitMask := oa.usedBinsTop >> (topBin + 1)
	if topBitMask == 0 {
		return 0, false
	}
	nextTopBin := topBin + 1 + uint32(bits.TrailingZeros32(topBitMask))
	if nextTopBin >= numTopBins {
		return 0, false
	}
	leaf := uint32(bits.TrailingZeros8(oa.usedBins[nextTopBin]))
	return nextTopBin*8 + leaf, true
}

func (oa *OffsetAllocator) allocNode() (uint32, bool) {
	n := len(oa.freeNodes)
	if n == 0 {
		return noNode, false
	}
	idx := oa.freeNodes[n-1]
	oa.freeNodes = oa.freeNodes[:n-1]
	return idx, true
}

func (oa *OffsetAllocator) releaseNode(idx uint32) {
	oa.nodes[idx] = offsetNode{
		binListPrev:  noNode,
		binListNext:  noNode,
		neighborPrev: noNode,
		neighborNext: noNode,
	}
	oa.freeNodes = append(oa.freeNodes, idx)
}

func (oa *OffsetAllocator) AllocWithPromotion(size uint64, maxClassIdx int) (Allocation, bool, error) {
	a, err := oa.Alloc(size)
	return a, false, err
}

func (oa *OffsetAllocator) Alloc(size uint64) (Allocation, error) {
	if size == 0 {
		return Allocation{}, fmt.Errorf("cannot allocate 0 bytes")
	}

	bin := uintToFloatRoundUp(size)
	allocSize := binSize(bin)
	if allocSize < size {
		allocSize = size
	}

	foundBin, ok := oa.findBin(bin)
	if !ok {
		return Allocation{}, fmt.Errorf("offset allocator exhausted: no free block for %d bytes (allocated: %d / %d)",
			size, oa.allocated.Load(), oa.regionSize)
	}

	nodeIdx := oa.binListHeads[foundBin]
	if nodeIdx == noNode {
		return Allocation{}, fmt.Errorf("offset allocator: internal error — empty bin %d has set bit", foundBin)
	}
	oa.removeFromBin(nodeIdx, foundBin)

	node := &oa.nodes[nodeIdx]
	node.used = true

	remainder := node.dataSize - allocSize
	if remainder > 0 {
		remIdx, ok := oa.allocNode()
		if !ok {
			allocSize = node.dataSize
		} else {
			remNode := &oa.nodes[remIdx]
			remNode.dataOffset = node.dataOffset + allocSize
			remNode.dataSize = remainder
			remNode.used = false
			remNode.neighborPrev = nodeIdx
			remNode.neighborNext = node.neighborNext

			if node.neighborNext != noNode {
				oa.nodes[node.neighborNext].neighborPrev = remIdx
			}
			node.neighborNext = remIdx
			node.dataSize = allocSize

			remBin := uintToFloatRoundDown(remainder)
			oa.insertIntoBin(remIdx, remBin)
			oa.offsetToNode[remNode.dataOffset] = remIdx
		}
	}

	oa.allocated.Add(int64(allocSize))
	oa.allocCount.Add(1)
	oa.totalReqBytes.Add(int64(size))

	return Allocation{
		ClassIdx: 0,
		Offset:   node.dataOffset,
		Size:     allocSize,
	}, nil
}

func (oa *OffsetAllocator) Free(a Allocation) {
	nodeIdx, ok := oa.offsetToNode[a.Offset]
	if !ok {
		return
	}

	node := &oa.nodes[nodeIdx]
	if !node.used {
		return
	}
	node.used = false
	oa.allocated.Add(-int64(node.dataSize))
	oa.freeCount_.Add(1)

	// Coalesce with right neighbor
	rightIdx := node.neighborNext
	if rightIdx != noNode && !oa.nodes[rightIdx].used {
		right := &oa.nodes[rightIdx]
		rightBin := uintToFloatRoundDown(right.dataSize)
		oa.removeFromBin(rightIdx, rightBin)

		node.dataSize += right.dataSize
		delete(oa.offsetToNode, right.dataOffset)

		nextNext := right.neighborNext
		node.neighborNext = nextNext
		if nextNext != noNode {
			oa.nodes[nextNext].neighborPrev = nodeIdx
		}
		oa.releaseNode(rightIdx)
	}

	// Coalesce with left neighbor
	leftIdx := node.neighborPrev
	if leftIdx != noNode && !oa.nodes[leftIdx].used {
		left := &oa.nodes[leftIdx]
		leftBin := uintToFloatRoundDown(left.dataSize)
		oa.removeFromBin(leftIdx, leftBin)

		left.dataSize += node.dataSize
		delete(oa.offsetToNode, node.dataOffset)

		rightRight := node.neighborNext
		left.neighborNext = rightRight
		if rightRight != noNode {
			oa.nodes[rightRight].neighborPrev = leftIdx
		}
		oa.releaseNode(nodeIdx)
		nodeIdx = leftIdx
	}

	mergedNode := &oa.nodes[nodeIdx]
	mergeBin := uintToFloatRoundDown(mergedNode.dataSize)
	oa.insertIntoBin(nodeIdx, mergeBin)
}

func (oa *OffsetAllocator) Write(a Allocation, data []byte) {
	copy(oa.region.Data()[a.Offset:], data)
}

func (oa *OffsetAllocator) Read(a Allocation) []byte {
	nodeIdx, ok := oa.offsetToNode[a.Offset]
	if !ok {
		end := a.Offset + a.Size
		if end > oa.regionSize {
			end = oa.regionSize
		}
		return oa.region.Data()[a.Offset:end]
	}
	node := &oa.nodes[nodeIdx]
	return oa.region.Data()[node.dataOffset : node.dataOffset+node.dataSize]
}

func (oa *OffsetAllocator) AllocatedBytes() int64 {
	return oa.allocated.Load()
}

func (oa *OffsetAllocator) SlotUtilization(requestSize uint64) float64 {
	if requestSize == 0 {
		return 0
	}
	bin := uintToFloatRoundUp(requestSize)
	bs := binSize(bin)
	if bs == 0 {
		return 0
	}
	return float64(requestSize) / float64(bs)
}

func (oa *OffsetAllocator) Regions() []RegionInfo {
	return []RegionInfo{{
		ClassIdx: 0,
		SlotSize: oa.regionSize,
		Region:   oa.region,
	}}
}

func (oa *OffsetAllocator) HugepageSummary() (gotHuge, thpHinted, regular int) {
	switch {
	case oa.region.GotHugePages():
		return 1, 0, 0
	case oa.region.THPHinted():
		return 0, 1, 0
	default:
		return 0, 0, 1
	}
}

func (oa *OffsetAllocator) ClassUtilizations() []ClassUtilization {
	ac := oa.allocCount.Load()
	fc := oa.freeCount_.Load()
	trb := oa.totalReqBytes.Load()
	allocBytes := oa.allocated.Load()

	var avgReq, slotUtil float64
	if ac > 0 {
		avgReq = float64(trb) / float64(ac)
		roundedAvg := float64(binSize(uintToFloatRoundUp(uint64(avgReq))))
		if roundedAvg > 0 {
			slotUtil = avgReq / roundedAvg
		}
	}

	return []ClassUtilization{{
		Size:            oa.regionSize,
		TotalSlots:      oa.regionSize,
		UsedSlots:       uint64(allocBytes),
		AllocCount:      ac,
		FreeCount:       fc,
		AvgRequestBytes: avgReq,
		SlotUtilization: slotUtil,
	}}
}

func (oa *OffsetAllocator) ResetCounters() {
	oa.allocCount.Store(0)
	oa.freeCount_.Store(0)
	oa.totalReqBytes.Store(0)
}

func (oa *OffsetAllocator) NumClasses() int {
	return 1
}

func (oa *OffsetAllocator) ClassSize(i int) uint64 {
	return oa.regionSize
}

func (oa *OffsetAllocator) FindClass(size uint64) int {
	if size <= oa.regionSize {
		return 0
	}
	return -1
}

func (oa *OffsetAllocator) ModelClassCapacity(valueSize uint64) (totalSlots, classSize uint64) {
	binned := binSize(uintToFloatRoundUp(valueSize))
	if binned == 0 || binned > oa.regionSize {
		return 0, 0
	}
	return oa.regionSize / binned, binned
}

func (oa *OffsetAllocator) CurrentWeights() map[uint64]float64 {
	return map[uint64]float64{oa.regionSize: 1.0}
}

func (oa *OffsetAllocator) Close() error {
	return oa.region.Close()
}
