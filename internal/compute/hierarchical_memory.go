// Package compute implements hardware abstraction, tensor computation, memory slab management,
// and zero-copy device interconnect acceleration for the fak agent kernel.
package compute

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// MemoryTier designates a physical storage/memory tier in the workstation hierarchy.
type MemoryTier int

const (
	// Tier0VRAM is GPU VRAM (32 GB GDDR7 @ 1,792 GB/s, RTX 5090 FE, sm_120).
	Tier0VRAM MemoryTier = 0

	// Tier1HostDRAM is Host Pinned DRAM (128 GB DDR4 @ ~50 GB/s, Ryzen 5950X, PCIe 4.0 x16 ~14.5 GB/s bidirectional DMA via CUDA async streams / UVA).
	Tier1HostDRAM MemoryTier = 1

	// Tier2NVMeDirect is Direct NVMe Storage (PCIe 4.0 x4 @ ~7.1 GB/s via BaM P2PDMA in M2A_CPU slot).
	Tier2NVMeDirect MemoryTier = 2
)

// String returns a human-readable representation of the MemoryTier.
func (t MemoryTier) String() string {
	switch t {
	case Tier0VRAM:
		return "Tier0_VRAM"
	case Tier1HostDRAM:
		return "Tier1_HostDRAM"
	case Tier2NVMeDirect:
		return "Tier2_NVMeDirect"
	default:
		return fmt.Sprintf("Tier_Unknown(%d)", int(t))
	}
}

// Workstation architecture default constants.
const (
	// DefaultTier0VRAMCapacityBytes is 32 GB for NVIDIA RTX 5090 FE.
	DefaultTier0VRAMCapacityBytes uint64 = 32 * 1024 * 1024 * 1024

	// DefaultTier1HostDRAMCapacityBytes is 128 GB for AMD Ryzen 5950X system DDR4.
	DefaultTier1HostDRAMCapacityBytes uint64 = 128 * 1024 * 1024 * 1024

	// DefaultTier2NVMeCapacityBytes is 2 TB for PCIe 4.0 x4 M2A_CPU direct storage.
	DefaultTier2NVMeCapacityBytes uint64 = 2 * 1024 * 1024 * 1024 * 1024

	// DefaultTier0HighWatermark triggers eviction when Tier 0 usage reaches 85%.
	DefaultTier0HighWatermark float64 = 0.85

	// DefaultTier0LowWatermark is the target threshold (70%) to evict down to in Tier 0.
	DefaultTier0LowWatermark float64 = 0.70

	// DefaultTier1HighWatermark triggers eviction when Tier 1 usage reaches 90%.
	DefaultTier1HighWatermark float64 = 0.90

	// DefaultTier1LowWatermark is the target threshold (75%) to evict down to in Tier 1.
	DefaultTier1LowWatermark float64 = 0.75

	// Bandwidth ratings for the user's workstation profile.
	Tier0BandwidthGBps float64 = 1792.0 // GDDR7 RTX 5090 FE
	Tier1BandwidthGBps float64 = 50.0   // DDR4 Host DRAM / PCIe 4.0 x16
	Tier2BandwidthGBps float64 = 7.1    // NVMe M2A_CPU slot PCIe 4.0 x4
)

// HierarchicalMemoryConfig defines capacity boundaries and watermark thresholds across tiers.
type HierarchicalMemoryConfig struct {
	Tier0CapacityBytes uint64  `json:"tier0_capacity_bytes"`
	Tier1CapacityBytes uint64  `json:"tier1_capacity_bytes"`
	Tier2CapacityBytes uint64  `json:"tier2_capacity_bytes"`
	Tier0HighWatermark float64 `json:"tier0_high_watermark"`
	Tier0LowWatermark  float64 `json:"tier0_low_watermark"`
	Tier1HighWatermark float64 `json:"tier1_high_watermark"`
	Tier1LowWatermark  float64 `json:"tier1_low_watermark"`
}

// HierarchicalBlock represents a managed unit of memory situated in a specific tier.
type HierarchicalBlock struct {
	BlockID        string     `json:"block_id"`
	CurrentTier    MemoryTier `json:"current_tier"`
	SizeBytes      uint64     `json:"size_bytes"`
	LastAccessNano int64      `json:"last_access_nano"`
	AccessCount    uint64     `json:"access_count"`
	Pinned         bool       `json:"pinned"`
	NVMeLBA        uint64     `json:"nvme_lba"`
	Data           []byte     `json:"data,omitempty"`
	VRAMAddress    uintptr    `json:"vram_address,omitempty"`

	accessSeq uint64
	isDirty   bool
}

// HierarchicalMemoryStats provides operational metrics and accounting for the memory hierarchy.
type HierarchicalMemoryStats struct {
	Tier0UsageBytes    uint64  `json:"tier0_usage_bytes"`
	Tier1UsageBytes    uint64  `json:"tier1_usage_bytes"`
	Tier2UsageBytes    uint64  `json:"tier2_usage_bytes"`
	Tier0BlockCount    int     `json:"tier0_block_count"`
	Tier1BlockCount    int     `json:"tier1_block_count"`
	Tier2BlockCount    int     `json:"tier2_block_count"`
	TotalBlocks        int     `json:"total_blocks"`
	PinnedBlockCount   int     `json:"pinned_block_count"`
	PromotedCount      uint64  `json:"promoted_count"`
	DemotedCount       uint64  `json:"demoted_count"`
	EvictionCount      uint64  `json:"eviction_count"`
	ReadHits           uint64  `json:"read_hits"`
	WriteCount         uint64  `json:"write_count"`
	Tier0BandwidthGBps float64 `json:"tier0_bandwidth_gbps"`
	Tier1BandwidthGBps float64 `json:"tier1_bandwidth_gbps"`
	Tier2BandwidthGBps float64 `json:"tier2_bandwidth_gbps"`
}

// HierarchicalMemoryManager coordinates allocation, LRU watermark eviction, promotion,
// and demotion across GPU VRAM (Tier 0), Host DRAM (Tier 1), and direct NVMe storage (Tier 2).
type HierarchicalMemoryManager struct {
	cfg        HierarchicalMemoryConfig
	bamSlab    *CUDADirectStorageMemorySlab
	mu         sync.RWMutex
	blocks     map[string]*HierarchicalBlock
	tierUsage  [3]uint64
	lbaCounter atomic.Uint64
	seqCounter atomic.Uint64
	prefetchWG sync.WaitGroup
	stats      HierarchicalMemoryStats
}

// NewHierarchicalMemoryManager instantiates a 3-tier memory manager tailored to the workstation profile.
func NewHierarchicalMemoryManager(cfg HierarchicalMemoryConfig, bamSlab *CUDADirectStorageMemorySlab) (*HierarchicalMemoryManager, error) {
	if cfg.Tier0CapacityBytes == 0 {
		cfg.Tier0CapacityBytes = DefaultTier0VRAMCapacityBytes
	}
	if cfg.Tier1CapacityBytes == 0 {
		cfg.Tier1CapacityBytes = DefaultTier1HostDRAMCapacityBytes
	}
	if cfg.Tier2CapacityBytes == 0 {
		cfg.Tier2CapacityBytes = DefaultTier2NVMeCapacityBytes
	}

	if cfg.Tier0HighWatermark <= 0 || cfg.Tier0HighWatermark > 1.0 {
		cfg.Tier0HighWatermark = DefaultTier0HighWatermark
	}
	if cfg.Tier0LowWatermark <= 0 || cfg.Tier0LowWatermark >= cfg.Tier0HighWatermark {
		cfg.Tier0LowWatermark = DefaultTier0LowWatermark
	}
	if cfg.Tier1HighWatermark <= 0 || cfg.Tier1HighWatermark > 1.0 {
		cfg.Tier1HighWatermark = DefaultTier1HighWatermark
	}
	if cfg.Tier1LowWatermark <= 0 || cfg.Tier1LowWatermark >= cfg.Tier1HighWatermark {
		cfg.Tier1LowWatermark = DefaultTier1LowWatermark
	}

	if bamSlab == nil {
		slabCfg := CUDADirectStorageConfig{
			NodeID:          0,
			BlockSize:       64 * 1024,
			TotalBlocks:     1024,
			BaseAddress:     0x200000000,
			Arch:            CUDABlackwellArch,
			DeviceName:      CUDARTX5090DeviceName,
			QueueCapacity:   256,
			DoorbellAddress: 0xD0000000,
		}
		var err error
		bamSlab, err = NewCUDADirectStorageMemorySlab(slabCfg)
		if err != nil {
			return nil, fmt.Errorf("hierarchical: failed to initialize default CUDA direct storage slab: %w", err)
		}
	}

	mgr := &HierarchicalMemoryManager{
		cfg:     cfg,
		bamSlab: bamSlab,
		blocks:  make(map[string]*HierarchicalBlock),
	}

	return mgr, nil
}

func (m *HierarchicalMemoryManager) tierCapacity(tier MemoryTier) uint64 {
	switch tier {
	case Tier0VRAM:
		return m.cfg.Tier0CapacityBytes
	case Tier1HostDRAM:
		return m.cfg.Tier1CapacityBytes
	case Tier2NVMeDirect:
		return m.cfg.Tier2CapacityBytes
	default:
		return 0
	}
}

func (m *HierarchicalMemoryManager) highWatermarkBytes(tier MemoryTier) uint64 {
	switch tier {
	case Tier0VRAM:
		return uint64(float64(m.cfg.Tier0CapacityBytes) * m.cfg.Tier0HighWatermark)
	case Tier1HostDRAM:
		return uint64(float64(m.cfg.Tier1CapacityBytes) * m.cfg.Tier1HighWatermark)
	default:
		return m.cfg.Tier2CapacityBytes
	}
}

func (m *HierarchicalMemoryManager) lowWatermarkBytes(tier MemoryTier) uint64 {
	switch tier {
	case Tier0VRAM:
		return uint64(float64(m.cfg.Tier0CapacityBytes) * m.cfg.Tier0LowWatermark)
	case Tier1HostDRAM:
		return uint64(float64(m.cfg.Tier1CapacityBytes) * m.cfg.Tier1LowWatermark)
	default:
		return 0
	}
}

// Allocate provisions a new block in the specified preferred tier, executing watermark evictions as necessary.
func (m *HierarchicalMemoryManager) Allocate(blockID string, sizeBytes uint64, preferredTier MemoryTier, pinned bool) (*HierarchicalBlock, error) {
	if blockID == "" {
		return nil, errors.New("hierarchical: empty block ID")
	}
	if preferredTier < Tier0VRAM || preferredTier > Tier2NVMeDirect {
		return nil, fmt.Errorf("hierarchical: invalid preferred tier %v", preferredTier)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.blocks[blockID]; exists {
		return nil, fmt.Errorf("hierarchical: block %q already exists", blockID)
	}

	targetTier := preferredTier

	// Attempt placement in targetTier, cascading down if capacity is exceeded
	for t := targetTier; t <= Tier2NVMeDirect; t++ {
		// If adding sizeBytes exceeds high watermark, trigger eviction on this tier
		if t < Tier2NVMeDirect && m.tierUsage[t]+sizeBytes > m.highWatermarkBytes(t) {
			_, _ = m.evictFromTierLocked(t)
		}

		// Check if it fits within capacity
		if m.tierUsage[t]+sizeBytes <= m.tierCapacity(t) {
			targetTier = t
			break
		}

		if t == Tier2NVMeDirect {
			return nil, fmt.Errorf("hierarchical: unable to allocate block %q (%d bytes): insufficient capacity across all tiers", blockID, sizeBytes)
		}
	}

	lba := m.lbaCounter.Add(1)
	blk := &HierarchicalBlock{
		BlockID:        blockID,
		CurrentTier:    targetTier,
		SizeBytes:      sizeBytes,
		LastAccessNano: time.Now().UnixNano(),
		AccessCount:    0,
		Pinned:         pinned,
		NVMeLBA:        lba,
		Data:           make([]byte, sizeBytes),
		accessSeq:      m.seqCounter.Add(1),
	}

	if targetTier == Tier0VRAM && m.bamSlab != nil {
		blk.VRAMAddress = m.bamSlab.baseAddress + uintptr((lba%uint64(m.bamSlab.totalBlocks))*m.bamSlab.blockSize)
	}

	m.tierUsage[targetTier] += sizeBytes
	m.blocks[blockID] = blk

	return blk, nil
}

// Write writes payload data to the specified block, adjusting size accounting and triggering evictions if needed.
func (m *HierarchicalMemoryManager) Write(blockID string, data []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	blk, ok := m.blocks[blockID]
	if !ok {
		return fmt.Errorf("hierarchical: block %q not found", blockID)
	}

	newSize := uint64(len(data))
	oldSize := blk.SizeBytes

	if newSize != oldSize {
		if newSize > oldSize {
			diff := newSize - oldSize
			if blk.CurrentTier < Tier2NVMeDirect && m.tierUsage[blk.CurrentTier]+diff > m.highWatermarkBytes(blk.CurrentTier) {
				_, _ = m.evictFromTierLocked(blk.CurrentTier)
			}
			if m.tierUsage[blk.CurrentTier]+diff > m.tierCapacity(blk.CurrentTier) {
				return fmt.Errorf("hierarchical: tier %v capacity exceeded during write", blk.CurrentTier)
			}
			m.tierUsage[blk.CurrentTier] += diff
		} else {
			diff := oldSize - newSize
			if diff > m.tierUsage[blk.CurrentTier] {
				m.tierUsage[blk.CurrentTier] = 0
			} else {
				m.tierUsage[blk.CurrentTier] -= diff
			}
		}
		blk.SizeBytes = newSize
	}

	blk.Data = make([]byte, len(data))
	copy(blk.Data, data)
	blk.LastAccessNano = time.Now().UnixNano()
	blk.AccessCount++
	blk.accessSeq = m.seqCounter.Add(1)
	blk.isDirty = true
	m.stats.WriteCount++

	if blk.CurrentTier == Tier0VRAM && m.bamSlab != nil {
		if blk.NVMeLBA == 0 {
			blk.NVMeLBA = m.lbaCounter.Add(1)
		}
		_ = m.bamSlab.WriteBlock(blk.NVMeLBA, data)
		blk.VRAMAddress = m.bamSlab.baseAddress + uintptr((blk.NVMeLBA%uint64(m.bamSlab.totalBlocks))*m.bamSlab.blockSize)
	}

	return nil
}

// Read retrieves a block's data, optionally staging or migrating it into destTier.
func (m *HierarchicalMemoryManager) Read(blockID string, destTier MemoryTier) ([]byte, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	blk, ok := m.blocks[blockID]
	if !ok {
		return nil, fmt.Errorf("hierarchical: block %q not found", blockID)
	}

	if destTier < Tier0VRAM || destTier > Tier2NVMeDirect {
		return nil, fmt.Errorf("hierarchical: invalid destination tier %v", destTier)
	}

	blk.LastAccessNano = time.Now().UnixNano()
	blk.AccessCount++
	blk.accessSeq = m.seqCounter.Add(1)
	m.stats.ReadHits++

	if destTier != blk.CurrentTier {
		if destTier < blk.CurrentTier {
			if err := m.promoteLocked(blockID, destTier); err != nil {
				return nil, fmt.Errorf("hierarchical: read promotion to %v failed: %w", destTier, err)
			}
		} else {
			if err := m.demoteLocked(blockID, destTier); err != nil {
				return nil, fmt.Errorf("hierarchical: read demotion to %v failed: %w", destTier, err)
			}
		}
	}

	out := make([]byte, len(blk.Data))
	copy(out, blk.Data)
	return out, nil
}

// Promote elevates a block from a slower tier to a faster tier.
func (m *HierarchicalMemoryManager) Promote(blockID string, targetTier MemoryTier) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.promoteLocked(blockID, targetTier)
}

func (m *HierarchicalMemoryManager) promoteLocked(blockID string, targetTier MemoryTier) error {
	blk, ok := m.blocks[blockID]
	if !ok {
		return fmt.Errorf("hierarchical: block %q not found", blockID)
	}

	if targetTier > blk.CurrentTier {
		return fmt.Errorf("hierarchical: cannot promote block %q from %v to slower tier %v; use Demote", blockID, blk.CurrentTier, targetTier)
	}
	if targetTier == blk.CurrentTier {
		return nil
	}

	if targetTier < Tier2NVMeDirect && m.tierUsage[targetTier]+blk.SizeBytes > m.highWatermarkBytes(targetTier) {
		_, _ = m.evictFromTierLocked(targetTier)
	}
	if m.tierUsage[targetTier]+blk.SizeBytes > m.tierCapacity(targetTier) {
		return fmt.Errorf("hierarchical: target tier %v capacity exceeded for promote", targetTier)
	}

	oldTier := blk.CurrentTier
	m.tierUsage[oldTier] -= blk.SizeBytes
	m.tierUsage[targetTier] += blk.SizeBytes
	blk.CurrentTier = targetTier
	blk.LastAccessNano = time.Now().UnixNano()
	blk.accessSeq = m.seqCounter.Add(1)
	m.stats.PromotedCount++

	if targetTier == Tier0VRAM {
		if m.bamSlab != nil {
			if blk.NVMeLBA == 0 {
				blk.NVMeLBA = m.lbaCounter.Add(1)
			}
			if len(blk.Data) > 0 {
				_ = m.bamSlab.WriteBlock(blk.NVMeLBA, blk.Data)
			}
			blk.VRAMAddress = m.bamSlab.baseAddress + uintptr((blk.NVMeLBA%uint64(m.bamSlab.totalBlocks))*m.bamSlab.blockSize)
			if oldTier == Tier2NVMeDirect {
				_ = m.bamSlab.DirectNVMeSwapIn(&CUDAStorageMemoryBlock{
					BlockID:     blk.NVMeLBA,
					SizeBytes:   blk.SizeBytes,
					NVMeLBA:     blk.NVMeLBA,
					VRAMAddress: blk.VRAMAddress,
				}, 1)
			}
		}
	} else {
		blk.VRAMAddress = 0
	}

	return nil
}

// Demote moves a block from a faster tier to a slower tier.
func (m *HierarchicalMemoryManager) Demote(blockID string, targetTier MemoryTier) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.demoteLocked(blockID, targetTier)
}

func (m *HierarchicalMemoryManager) demoteLocked(blockID string, targetTier MemoryTier) error {
	blk, ok := m.blocks[blockID]
	if !ok {
		return fmt.Errorf("hierarchical: block %q not found", blockID)
	}

	if targetTier < blk.CurrentTier {
		return fmt.Errorf("hierarchical: cannot demote block %q from %v to faster tier %v; use Promote", blockID, blk.CurrentTier, targetTier)
	}
	if targetTier == blk.CurrentTier {
		return nil
	}

	if targetTier < Tier2NVMeDirect && m.tierUsage[targetTier]+blk.SizeBytes > m.highWatermarkBytes(targetTier) {
		_, _ = m.evictFromTierLocked(targetTier)
	}
	if m.tierUsage[targetTier]+blk.SizeBytes > m.tierCapacity(targetTier) {
		return fmt.Errorf("hierarchical: target tier %v capacity exceeded for demote", targetTier)
	}

	oldTier := blk.CurrentTier
	m.tierUsage[oldTier] -= blk.SizeBytes
	m.tierUsage[targetTier] += blk.SizeBytes
	blk.CurrentTier = targetTier
	blk.LastAccessNano = time.Now().UnixNano()
	blk.accessSeq = m.seqCounter.Add(1)
	blk.VRAMAddress = 0
	m.stats.DemotedCount++

	if targetTier == Tier2NVMeDirect && m.bamSlab != nil {
		if blk.NVMeLBA == 0 {
			blk.NVMeLBA = m.lbaCounter.Add(1)
		}
		if oldTier == Tier0VRAM {
			_ = m.bamSlab.DirectNVMeSwapOut(&CUDAStorageMemoryBlock{
				BlockID:   blk.NVMeLBA,
				SizeBytes: blk.SizeBytes,
				NVMeLBA:   blk.NVMeLBA,
			}, 1)
		}
	}

	return nil
}

// EvictFromTier demotes LRU unpinned blocks from the specified tier down to the next tier until low watermark is met.
func (m *HierarchicalMemoryManager) EvictFromTier(tier MemoryTier) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.evictFromTierLocked(tier)
}

func (m *HierarchicalMemoryManager) evictFromTierLocked(tier MemoryTier) (int, error) {
	if tier == Tier2NVMeDirect {
		// Tier 2 is bottom direct storage: purge oldest unpinned block if capacity reclamation is needed
		var oldest *HierarchicalBlock
		for _, blk := range m.blocks {
			if blk.CurrentTier != tier || blk.Pinned {
				continue
			}
			if oldest == nil || blk.LastAccessNano < oldest.LastAccessNano ||
				(blk.LastAccessNano == oldest.LastAccessNano && blk.accessSeq < oldest.accessSeq) {
				oldest = blk
			}
		}
		if oldest == nil {
			return 0, nil
		}
		delete(m.blocks, oldest.BlockID)
		m.tierUsage[tier] -= oldest.SizeBytes
		m.stats.EvictionCount++
		return 1, nil
	}

	targetLow := m.lowWatermarkBytes(tier)
	evictedCount := 0

	for {
		var oldest *HierarchicalBlock
		for _, blk := range m.blocks {
			if blk.CurrentTier != tier || blk.Pinned {
				continue
			}
			if oldest == nil || blk.LastAccessNano < oldest.LastAccessNano ||
				(blk.LastAccessNano == oldest.LastAccessNano && blk.accessSeq < oldest.accessSeq) {
				oldest = blk
			}
		}

		if oldest == nil {
			break
		}

		if evictedCount > 0 && m.tierUsage[tier] <= targetLow {
			break
		}

		nextTier := tier + 1
		if m.tierUsage[nextTier]+oldest.SizeBytes > m.tierCapacity(nextTier) {
			if _, err := m.evictFromTierLocked(nextTier); err != nil {
				return evictedCount, err
			}
			if m.tierUsage[nextTier]+oldest.SizeBytes > m.tierCapacity(nextTier) {
				return evictedCount, fmt.Errorf("hierarchical: next tier %v full during eviction", nextTier)
			}
		}

		m.tierUsage[tier] -= oldest.SizeBytes
		m.tierUsage[nextTier] += oldest.SizeBytes
		oldest.CurrentTier = nextTier
		oldest.LastAccessNano = time.Now().UnixNano()
		oldest.accessSeq = m.seqCounter.Add(1)
		oldest.VRAMAddress = 0
		m.stats.DemotedCount++
		m.stats.EvictionCount++
		evictedCount++

		// Cascading watermark check: if next tier (e.g. Tier 1) now exceeds its HighWatermark, trigger eviction in next tier
		if nextTier == Tier1HostDRAM && m.tierUsage[Tier1HostDRAM] > m.highWatermarkBytes(Tier1HostDRAM) {
			if _, err := m.evictFromTierLocked(Tier1HostDRAM); err != nil {
				return evictedCount, err
			}
		}

		if m.tierUsage[tier] <= targetLow {
			break
		}
	}

	return evictedCount, nil
}

// Prefetch triggers asynchronous non-blocking prefetching of the specified blockIDs into targetTier.
func (m *HierarchicalMemoryManager) Prefetch(blockIDs []string, targetTier MemoryTier) error {
	if len(blockIDs) == 0 {
		return errors.New("hierarchical: empty block list for prefetch")
	}
	if targetTier < Tier0VRAM || targetTier > Tier2NVMeDirect {
		return fmt.Errorf("hierarchical: invalid target tier %v for prefetch", targetTier)
	}

	m.mu.RLock()
	for _, id := range blockIDs {
		if _, ok := m.blocks[id]; !ok {
			m.mu.RUnlock()
			return fmt.Errorf("hierarchical: block %q not found for prefetch", id)
		}
	}
	m.mu.RUnlock()

	m.prefetchWG.Add(1)
	go func(ids []string, tier MemoryTier) {
		defer m.prefetchWG.Done()
		for _, id := range ids {
			m.mu.Lock()
			blk, ok := m.blocks[id]
			if ok {
				if blk.CurrentTier > tier {
					_ = m.promoteLocked(id, tier)
				} else if blk.CurrentTier < tier {
					_ = m.demoteLocked(id, tier)
				}
			}
			m.mu.Unlock()
		}
	}(blockIDs, targetTier)

	return nil
}

// WaitForPrefetch blocks until all outstanding background prefetch tasks finish.
func (m *HierarchicalMemoryManager) WaitForPrefetch() {
	m.prefetchWG.Wait()
}

// GetBlock returns a pointer to the block descriptor for inspection.
func (m *HierarchicalMemoryManager) GetBlock(blockID string) (*HierarchicalBlock, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	blk, ok := m.blocks[blockID]
	if !ok {
		return nil, fmt.Errorf("hierarchical: block %q not found", blockID)
	}
	return blk, nil
}

// SetPinned sets the pinned status for a block. Pinned blocks cannot be demoted by watermark evictions.
func (m *HierarchicalMemoryManager) SetPinned(blockID string, pinned bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	blk, ok := m.blocks[blockID]
	if !ok {
		return fmt.Errorf("hierarchical: block %q not found", blockID)
	}
	blk.Pinned = pinned
	return nil
}

// Stats returns a snapshot of memory usage, block counts, and eviction telemetry across all tiers.
func (m *HierarchicalMemoryManager) Stats() HierarchicalMemoryStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := m.stats
	stats.Tier0UsageBytes = m.tierUsage[Tier0VRAM]
	stats.Tier1UsageBytes = m.tierUsage[Tier1HostDRAM]
	stats.Tier2UsageBytes = m.tierUsage[Tier2NVMeDirect]

	var t0Count, t1Count, t2Count, pinnedCount int
	for _, blk := range m.blocks {
		switch blk.CurrentTier {
		case Tier0VRAM:
			t0Count++
		case Tier1HostDRAM:
			t1Count++
		case Tier2NVMeDirect:
			t2Count++
		}
		if blk.Pinned {
			pinnedCount++
		}
	}

	stats.Tier0BlockCount = t0Count
	stats.Tier1BlockCount = t1Count
	stats.Tier2BlockCount = t2Count
	stats.TotalBlocks = len(m.blocks)
	stats.PinnedBlockCount = pinnedCount

	// Workstation hardware profile bandwidth metrics
	stats.Tier0BandwidthGBps = Tier0BandwidthGBps
	stats.Tier1BandwidthGBps = Tier1BandwidthGBps
	stats.Tier2BandwidthGBps = Tier2BandwidthGBps

	return stats
}
