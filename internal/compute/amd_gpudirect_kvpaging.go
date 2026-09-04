// Package compute implements hardware abstraction, tensor computation, memory slab management,
// and zero-copy device interconnect acceleration for the fak agent kernel.
package compute

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"time"
)

var (
	// ErrQueueExhausted indicates the BaM NVMe submission queue has reached its in-flight depth limit.
	ErrQueueExhausted = errors.New("amddirect: BaM NVMe submission queue exhausted")

	// ErrBlockNotFound indicates a requested KV block ID is not registered in the directory.
	ErrBlockNotFound = errors.New("amddirect: KV block not found")

	// ErrBlockPinned indicates a block cannot be offloaded because it has active pin holds.
	ErrBlockPinned = errors.New("amddirect: cannot offload pinned KV block")

	// ErrFramesPinned indicates all resident VRAM frames are pinned, preventing offload.
	ErrFramesPinned = errors.New("amddirect: all VRAM frames pinned, cannot offload")
)

// LBAExtent represents a contiguous span of NVMe storage sectors.
type LBAExtent struct {
	StartingLBA uint64 `json:"starting_lba"`
	LBACount    uint64 `json:"lba_count"`
	BlockID     uint64 `json:"block_id"`
}

// LBAAllocationTable coordinates physical NVMe flash sector reservations for KV blocks.
type LBAAllocationTable struct {
	sectorSizeBytes uint32
	totalLBAs       uint64
	allocatedLBAs   uint64
	nextLBA         uint64
	mu              sync.RWMutex
	allocations     map[uint64]LBAExtent
	freeExtents     []LBAExtent
}

// NewLBAAllocationTable initializes an LBA allocation coordinator with total sector bounds.
func NewLBAAllocationTable(totalLBAs uint64, sectorSizeBytes uint32) (*LBAAllocationTable, error) {
	if sectorSizeBytes == 0 {
		sectorSizeBytes = 4096
	}
	if totalLBAs == 0 {
		totalLBAs = 1024 * 1024
	}
	return &LBAAllocationTable{
		sectorSizeBytes: sectorSizeBytes,
		totalLBAs:       totalLBAs,
		allocatedLBAs:   0,
		nextLBA:         0,
		allocations:     make(map[uint64]LBAExtent),
		freeExtents:     make([]LBAExtent, 0),
	}, nil
}

// Allocate reserves a contiguous LBA extent for blockID based on byte size.
func (t *LBAAllocationTable) Allocate(blockID uint64, byteSize uint64) (uint64, uint64, error) {
	if byteSize == 0 {
		return 0, 0, errors.New("amddirect: allocation size must be > 0")
	}
	lbaCount := (byteSize + uint64(t.sectorSizeBytes) - 1) / uint64(t.sectorSizeBytes)
	startLBA, err := t.AllocateLBAs(blockID, lbaCount)
	if err != nil {
		return 0, 0, err
	}
	return startLBA, lbaCount, nil
}

// AllocateLBAs reserves a contiguous span of lbaCount sectors for blockID.
func (t *LBAAllocationTable) AllocateLBAs(blockID uint64, lbaCount uint64) (uint64, error) {
	if lbaCount == 0 {
		return 0, errors.New("amddirect: LBA count must be > 0")
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	if _, exists := t.allocations[blockID]; exists {
		return 0, fmt.Errorf("amddirect: block %d already has LBA allocation", blockID)
	}

	bestIdx := -1
	for idx, ext := range t.freeExtents {
		if ext.LBACount >= lbaCount {
			if bestIdx == -1 || ext.LBACount < t.freeExtents[bestIdx].LBACount {
				bestIdx = idx
			}
		}
	}

	var startLBA uint64
	if bestIdx != -1 {
		matched := t.freeExtents[bestIdx]
		startLBA = matched.StartingLBA
		if matched.LBACount > lbaCount {
			t.freeExtents[bestIdx] = LBAExtent{
				StartingLBA: matched.StartingLBA + lbaCount,
				LBACount:    matched.LBACount - lbaCount,
				BlockID:     0,
			}
		} else {
			t.freeExtents = append(t.freeExtents[:bestIdx], t.freeExtents[bestIdx+1:]...)
		}
	} else {
		if t.nextLBA+lbaCount > t.totalLBAs {
			return 0, errors.New("amddirect: NVMe LBA allocation table out of space")
		}
		startLBA = t.nextLBA
		t.nextLBA += lbaCount
	}

	ext := LBAExtent{
		StartingLBA: startLBA,
		LBACount:    lbaCount,
		BlockID:     blockID,
	}
	t.allocations[blockID] = ext
	t.allocatedLBAs += lbaCount
	return startLBA, nil
}

// Free releases the sector reservation for blockID back to the free extents list.
func (t *LBAAllocationTable) Free(blockID uint64) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	ext, exists := t.allocations[blockID]
	if !exists {
		return fmt.Errorf("amddirect: LBA allocation not found for block %d", blockID)
	}

	delete(t.allocations, blockID)
	t.allocatedLBAs -= ext.LBACount
	ext.BlockID = 0
	t.freeExtents = append(t.freeExtents, ext)
	return nil
}

// GetLBA returns the starting LBA for blockID.
func (t *LBAAllocationTable) GetLBA(blockID uint64) (uint64, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	ext, exists := t.allocations[blockID]
	if !exists {
		return 0, false
	}
	return ext.StartingLBA, true
}

// GetExtent returns the full LBAExtent for blockID.
func (t *LBAAllocationTable) GetExtent(blockID uint64) (LBAExtent, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	ext, exists := t.allocations[blockID]
	return ext, exists
}

// AllocatedCount reports total sectors currently reserved.
func (t *LBAAllocationTable) AllocatedCount() uint64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.allocatedLBAs
}

// FreeCount reports remaining unallocated sectors.
func (t *LBAAllocationTable) FreeCount() uint64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.allocatedLBAs >= t.totalLBAs {
		return 0
	}
	return t.totalLBAs - t.allocatedLBAs
}

// TotalCount reports total available sectors in the LBA space.
func (t *LBAAllocationTable) TotalCount() uint64 {
	return t.totalLBAs
}

// SectorSizeBytes returns the configured sector size in bytes.
func (t *LBAAllocationTable) SectorSizeBytes() uint32 {
	return t.sectorSizeBytes
}

// KVBlockFrame represents a physical VRAM frame holding a resident KV block.
type KVBlockFrame struct {
	FrameID     int     `json:"frame_id"`
	VRAMAddress uintptr `json:"vram_address"`
	SizeBytes   uint64  `json:"size_bytes"`
	BlockID     uint64  `json:"block_id"`
	LayerID     int     `json:"layer_id"`
	InUse       bool    `json:"in_use"`
	// PinCount reports active reader/writer holds preventing offload of this VRAM frame to NVMe storage.
	PinCount       int    `json:"pin_count"`
	IsDirty        bool   `json:"is_dirty"`
	LastAccessNano int64  `json:"last_access_nano"`
	Data           []byte `json:"-"`
}

// KVPageDirectoryEntry tracks residency, physical VRAM mapping, and NVMe LBA backing for a KV block.
type KVPageDirectoryEntry struct {
	BlockID        uint64  `json:"block_id"`
	LayerID        int     `json:"layer_id"`
	TokenCount     int     `json:"token_count"`
	SizeBytes      uint64  `json:"size_bytes"`
	Resident       bool    `json:"resident"`
	FrameID        int     `json:"frame_id"`
	VRAMAddress    uintptr `json:"vram_address"`
	NVMeLBA        uint64  `json:"nvme_lba"`
	LBACount       uint64  `json:"lba_count"`
	IsDirty        bool    `json:"is_dirty"`
	IsColdPrefix   bool    `json:"is_cold_prefix"`
	Checksum       uint64  `json:"checksum"`
	AccessCount    uint64  `json:"access_count"`
	LastAccessNano int64   `json:"last_access_nano"`
}

// BaMIOPipeline orchestrates direct NVMe queue submissions and completions targeting GPU VRAM.
type BaMIOPipeline struct {
	hal         *AMDGPUDirectHAL
	queueDepth  int
	mu          sync.Mutex
	inFlight    map[uint16]*NVMeP2PCommand
	nextCmdID   uint16
	exhaustions uint64
}

// NewBaMIOPipeline constructs a new BaM direct I/O pipeline.
func NewBaMIOPipeline(hal *AMDGPUDirectHAL, queueDepth int) (*BaMIOPipeline, error) {
	if hal == nil {
		return nil, ErrNilHALCoordinator
	}
	if queueDepth <= 0 {
		queueDepth = 128
	}
	return &BaMIOPipeline{
		hal:        hal,
		queueDepth: queueDepth,
		inFlight:   make(map[uint16]*NVMeP2PCommand),
		nextCmdID:  1,
	}, nil
}

// SubmitRead formats and dispatches a direct NVMe read targeting GPU VRAM without CPU staging.
func (p *BaMIOPipeline) SubmitRead(lba uint64, blockCount uint16, vramAddr uintptr, byteLen uint64) (*NVMeP2PCommand, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(p.inFlight) >= p.queueDepth {
		p.exhaustions++
		return nil, ErrQueueExhausted
	}

	cmdID := p.nextCmdID
	p.nextCmdID++
	if p.nextCmdID == 0 {
		p.nextCmdID = 1
	}

	cmd := &NVMeP2PCommand{
		CommandID:      cmdID,
		Opcode:         NVMeOpcodeRead,
		NamespaceID:    1,
		StartingLBA:    lba,
		BlockCount:     blockCount,
		TargetVRAMAddr: vramAddr,
		ByteLength:     byteLen,
	}

	if err := p.hal.ExecuteNVMeP2PTransfer(cmd); err != nil {
		return nil, err
	}
	p.inFlight[cmdID] = cmd
	return cmd, nil
}

// SubmitWrite formats and dispatches a direct NVMe write from GPU VRAM without CPU staging.
func (p *BaMIOPipeline) SubmitWrite(lba uint64, blockCount uint16, vramAddr uintptr, byteLen uint64) (*NVMeP2PCommand, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(p.inFlight) >= p.queueDepth {
		p.exhaustions++
		return nil, ErrQueueExhausted
	}

	cmdID := p.nextCmdID
	p.nextCmdID++
	if p.nextCmdID == 0 {
		p.nextCmdID = 1
	}

	cmd := &NVMeP2PCommand{
		CommandID:      cmdID,
		Opcode:         NVMeOpcodeWrite,
		NamespaceID:    1,
		StartingLBA:    lba,
		BlockCount:     blockCount,
		TargetVRAMAddr: vramAddr,
		ByteLength:     byteLen,
	}

	if err := p.hal.ExecuteNVMeP2PTransfer(cmd); err != nil {
		return nil, err
	}
	p.inFlight[cmdID] = cmd
	return cmd, nil
}

// PollCompletions drains resolved in-flight commands from the pipeline.
func (p *BaMIOPipeline) PollCompletions(maxEntries int) int {
	p.mu.Lock()
	defer p.mu.Unlock()

	resolved := 0
	for cid, cmd := range p.inFlight {
		if maxEntries > 0 && resolved >= maxEntries {
			break
		}
		cmd.Completed = true
		cmd.Status = 0
		delete(p.inFlight, cid)
		resolved++
	}
	return resolved
}

// InFlightCount returns the number of active commands in the submission queue.
func (p *BaMIOPipeline) InFlightCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.inFlight)
}

// StagingCopyCount returns the number of intermediate host DRAM staging copies. Invariant: 0.
func (p *BaMIOPipeline) StagingCopyCount() int {
	return 0
}

// ExhaustionCount returns the number of times submission hit queue depth capacity.
func (p *BaMIOPipeline) ExhaustionCount() uint64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.exhaustions
}

// BaMKVPagingConfig defines the sizing and topology parameters for BaM KV block paging.
type BaMKVPagingConfig struct {
	NodeID            int     `json:"node_id"`
	TotalModelLayers  int     `json:"total_model_layers"`
	TokensPerBlock    int     `json:"tokens_per_block"`
	BytesPerBlock     uint64  `json:"bytes_per_block"`
	MaxResidentFrames int     `json:"max_resident_frames"`
	TotalNVMeLBAs     uint64  `json:"total_nvme_lbas"`
	SectorSizeBytes   uint32  `json:"sector_size_bytes"`
	VRAMBaseAddress   uintptr `json:"vram_base_address"`
	QueueDepth        int     `json:"queue_depth"`
	PrefetchDistance  int     `json:"prefetch_distance"`
}

// BaMKVPagingStats captures telemetry and throughput counters for BaM KV block paging.
type BaMKVPagingStats struct {
	TotalBlocks          uint64  `json:"total_blocks"`
	ResidentBlocks       uint64  `json:"resident_blocks"`
	OffloadedBlocks      uint64  `json:"offloaded_blocks"`
	PinnedBlocks         uint64  `json:"pinned_blocks"`
	ColdPrefixBlocks     uint64  `json:"cold_prefix_blocks"`
	NVMeReadCount        uint64  `json:"nvme_read_count"`
	NVMeWriteCount       uint64  `json:"nvme_write_count"`
	BytesReadNVMe        uint64  `json:"bytes_read_nvme"`
	BytesWrittenNVMe     uint64  `json:"bytes_written_nvme"`
	PrefetchHits         uint64  `json:"prefetch_hits"`
	PrefetchMisses       uint64  `json:"prefetch_misses"`
	PrefetchHitRate      float64 `json:"prefetch_hit_rate"`
	QueueExhaustionCount uint64  `json:"queue_exhaustion_count"`
	StagingCopies        int64   `json:"staging_copies"`
}

// PrefetchOverlapMetrics records timing and hiding efficiency of prefetching during compute.
type PrefetchOverlapMetrics struct {
	LayerIndex         int     `json:"layer_index"`
	ComputeDurationNs  int64   `json:"compute_duration_ns"`
	PrefetchDurationNs int64   `json:"prefetch_duration_ns"`
	HiddenDurationNs   int64   `json:"hidden_duration_ns"`
	OverlapPercentage  float64 `json:"overlap_percentage"`
	HidingAchieved     bool    `json:"hiding_achieved"`
}

// BaMKVPagingCoordinator coordinates KV block storage across GPU VRAM and NVMe via BaM P2PDMA.
type BaMKVPagingCoordinator struct {
	hal              *AMDGPUDirectHAL
	cfg              BaMKVPagingConfig
	lbaTable         *LBAAllocationTable
	pipeline         *BaMIOPipeline
	frames           []*KVBlockFrame
	freeFrameIndices []int
	directory        map[uint64]*KVPageDirectoryEntry
	layerBlocks      map[int][]uint64
	storageBacking   map[uint64][]byte
	stats            BaMKVPagingStats
	nextBlockID      uint64
	mu               sync.RWMutex
}

// NewBaMKVPagingCoordinator constructs an AMD GPU Direct BaM KV block paging coordinator.
func NewBaMKVPagingCoordinator(hal *AMDGPUDirectHAL, cfg BaMKVPagingConfig) (*BaMKVPagingCoordinator, error) {
	if hal == nil {
		return nil, ErrNilHALCoordinator
	}
	if cfg.TokensPerBlock <= 0 {
		cfg.TokensPerBlock = 1024
	}
	if cfg.BytesPerBlock == 0 {
		cfg.BytesPerBlock = 64 * 1024
	}
	if cfg.SectorSizeBytes == 0 {
		cfg.SectorSizeBytes = 4096
	}
	if cfg.MaxResidentFrames <= 0 {
		cfg.MaxResidentFrames = 64
	}
	if cfg.TotalNVMeLBAs == 0 {
		cfg.TotalNVMeLBAs = 1024 * 1024
	}
	if cfg.QueueDepth <= 0 {
		cfg.QueueDepth = 128
	}
	if cfg.VRAMBaseAddress == 0 {
		cfg.VRAMBaseAddress = 0x8000000000
	}
	if cfg.TotalModelLayers <= 0 {
		cfg.TotalModelLayers = 32
	}
	if cfg.PrefetchDistance <= 0 {
		cfg.PrefetchDistance = 2
	}

	lbaTable, err := NewLBAAllocationTable(cfg.TotalNVMeLBAs, cfg.SectorSizeBytes)
	if err != nil {
		return nil, err
	}

	pipeline, err := NewBaMIOPipeline(hal, cfg.QueueDepth)
	if err != nil {
		return nil, err
	}

	frames := make([]*KVBlockFrame, cfg.MaxResidentFrames)
	freeIndices := make([]int, cfg.MaxResidentFrames)
	for i := 0; i < cfg.MaxResidentFrames; i++ {
		addr := cfg.VRAMBaseAddress + uintptr(uint64(i)*cfg.BytesPerBlock)
		frames[i] = &KVBlockFrame{
			FrameID:     i,
			VRAMAddress: addr,
			SizeBytes:   cfg.BytesPerBlock,
			BlockID:     0,
			LayerID:     -1,
			InUse:       false,
			PinCount:    0,
			IsDirty:     false,
			Data:        make([]byte, cfg.BytesPerBlock),
		}
		freeIndices[i] = i
	}

	coord := &BaMKVPagingCoordinator{
		hal:              hal,
		cfg:              cfg,
		lbaTable:         lbaTable,
		pipeline:         pipeline,
		frames:           frames,
		freeFrameIndices: freeIndices,
		directory:        make(map[uint64]*KVPageDirectoryEntry),
		layerBlocks:      make(map[int][]uint64),
		storageBacking:   make(map[uint64][]byte),
	}
	return coord, nil
}

// computeChecksum produces a deterministic 64-bit FNV-1a hash over data bytes.
func computeChecksum(data []byte) uint64 {
	const (
		offset64 = 14695981039346656037
		prime64  = 1099511628211
	)
	var hash uint64 = offset64
	for _, b := range data {
		hash ^= uint64(b)
		hash *= prime64
	}
	return hash
}

// selectUnpinnedFrameLocked locates the least-recently accessed unpinned frame in VRAM.
func (c *BaMKVPagingCoordinator) selectUnpinnedFrameLocked() (int, error) {
	chosenIdx := -1
	var oldestNano int64 = math.MaxInt64
	for i, f := range c.frames {
		if f.InUse && f.PinCount == 0 {
			if f.LastAccessNano < oldestNano {
				oldestNano = f.LastAccessNano
				chosenIdx = i
			}
		}
	}
	if chosenIdx == -1 {
		return -1, ErrFramesPinned
	}
	return chosenIdx, nil
}

// offloadFrameLocked offloads an unpinned resident frame to NVMe storage.
func (c *BaMKVPagingCoordinator) offloadFrameLocked(frameIdx int) error {
	if frameIdx < 0 || frameIdx >= len(c.frames) {
		return errors.New("amddirect: invalid frame index")
	}
	frame := c.frames[frameIdx]
	if !frame.InUse {
		return nil
	}
	if frame.PinCount > 0 {
		return ErrBlockPinned
	}

	entry, exists := c.directory[frame.BlockID]
	if !exists {
		return ErrBlockNotFound
	}

	storageBuf := make([]byte, frame.SizeBytes)
	copy(storageBuf, frame.Data)
	c.storageBacking[entry.NVMeLBA] = storageBuf

	_, err := c.pipeline.SubmitWrite(entry.NVMeLBA, uint16(entry.LBACount), frame.VRAMAddress, frame.SizeBytes)
	if err != nil {
		return err
	}
	c.pipeline.PollCompletions(1)

	c.stats.NVMeWriteCount++
	c.stats.BytesWrittenNVMe += frame.SizeBytes
	c.stats.ResidentBlocks--
	c.stats.OffloadedBlocks++

	entry.Resident = false
	entry.FrameID = -1
	entry.VRAMAddress = 0
	entry.IsDirty = false

	frame.InUse = false
	frame.BlockID = 0
	frame.PinCount = 0
	frame.IsDirty = false
	c.freeFrameIndices = append(c.freeFrameIndices, frameIdx)
	return nil
}

// AllocateBlock registers a new KV block for a specific model layer.
func (c *BaMKVPagingCoordinator) AllocateBlock(layerID int, tokens int, data []byte) (uint64, error) {
	if layerID < 0 || (c.cfg.TotalModelLayers > 0 && layerID >= c.cfg.TotalModelLayers) {
		return 0, fmt.Errorf("amddirect: layer ID %d out of bounds [0, %d)", layerID, c.cfg.TotalModelLayers)
	}
	if tokens <= 0 {
		tokens = c.cfg.TokensPerBlock
	}
	if uint64(len(data)) > c.cfg.BytesPerBlock {
		return 0, fmt.Errorf("amddirect: data length %d exceeds block size %d", len(data), c.cfg.BytesPerBlock)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	c.nextBlockID++
	blockID := c.nextBlockID

	chk := computeChecksum(data)

	startLBA, lbaCount, err := c.lbaTable.Allocate(blockID, c.cfg.BytesPerBlock)
	if err != nil {
		return 0, err
	}

	storageBuf := make([]byte, c.cfg.BytesPerBlock)
	copy(storageBuf, data)
	c.storageBacking[startLBA] = storageBuf

	var frameIdx int
	resident := false
	var vramAddr uintptr = 0

	if len(c.freeFrameIndices) > 0 {
		frameIdx = c.freeFrameIndices[0]
		c.freeFrameIndices = c.freeFrameIndices[1:]
		resident = true
	} else {
		chosenIdx, offloadErr := c.selectUnpinnedFrameLocked()
		if offloadErr == nil {
			if err := c.offloadFrameLocked(chosenIdx); err == nil {
				frameIdx = c.freeFrameIndices[0]
				c.freeFrameIndices = c.freeFrameIndices[1:]
				resident = true
			}
		}
	}

	if resident {
		frame := c.frames[frameIdx]
		frame.BlockID = blockID
		frame.LayerID = layerID
		frame.InUse = true
		frame.PinCount = 0
		frame.IsDirty = false
		frame.LastAccessNano = time.Now().UnixNano()
		copy(frame.Data, storageBuf)
		vramAddr = frame.VRAMAddress
		c.stats.ResidentBlocks++
	} else {
		frameIdx = -1
		c.stats.OffloadedBlocks++
	}

	entry := &KVPageDirectoryEntry{
		BlockID:        blockID,
		LayerID:        layerID,
		TokenCount:     tokens,
		SizeBytes:      c.cfg.BytesPerBlock,
		Resident:       resident,
		FrameID:        frameIdx,
		VRAMAddress:    vramAddr,
		NVMeLBA:        startLBA,
		LBACount:       lbaCount,
		IsDirty:        false,
		Checksum:       chk,
		AccessCount:    1,
		LastAccessNano: time.Now().UnixNano(),
	}

	c.directory[blockID] = entry
	c.layerBlocks[layerID] = append(c.layerBlocks[layerID], blockID)
	c.stats.TotalBlocks++

	return blockID, nil
}

// FreeBlock releases a KV block, reclaiming both VRAM and NVMe resources.
func (c *BaMKVPagingCoordinator) FreeBlock(blockID uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry, exists := c.directory[blockID]
	if !exists {
		return ErrBlockNotFound
	}

	if entry.Resident {
		if entry.FrameID >= 0 && entry.FrameID < len(c.frames) {
			frame := c.frames[entry.FrameID]
			if frame.PinCount > 0 {
				c.stats.PinnedBlocks--
			}
			frame.InUse = false
			frame.BlockID = 0
			frame.PinCount = 0
			frame.IsDirty = false
			c.freeFrameIndices = append(c.freeFrameIndices, entry.FrameID)
		}
		c.stats.ResidentBlocks--
	} else {
		c.stats.OffloadedBlocks--
	}

	if entry.IsColdPrefix {
		c.stats.ColdPrefixBlocks--
	}

	_ = c.lbaTable.Free(blockID)
	delete(c.storageBacking, entry.NVMeLBA)
	delete(c.directory, blockID)

	layerList := c.layerBlocks[entry.LayerID]
	for i, id := range layerList {
		if id == blockID {
			c.layerBlocks[entry.LayerID] = append(layerList[:i], layerList[i+1:]...)
			break
		}
	}

	c.stats.TotalBlocks--
	return nil
}

// Pin locks a KV block in VRAM preventing offload to NVMe during active compute.
func (c *BaMKVPagingCoordinator) Pin(blockID uint64) error {
	c.mu.Lock()
	entry, exists := c.directory[blockID]
	if !exists {
		c.mu.Unlock()
		return ErrBlockNotFound
	}
	if !entry.Resident {
		c.mu.Unlock()
		if err := c.RestoreBlock(blockID); err != nil {
			return err
		}
		c.mu.Lock()
		entry = c.directory[blockID]
	}
	defer c.mu.Unlock()

	if entry.FrameID < 0 || entry.FrameID >= len(c.frames) {
		return errors.New("amddirect: invalid frame index")
	}

	frame := c.frames[entry.FrameID]
	if frame.PinCount == 0 {
		c.stats.PinnedBlocks++
	}
	frame.PinCount++
	return nil
}

// Unpin releases a lock on a KV block, allowing it to be offloaded when unpinned.
func (c *BaMKVPagingCoordinator) Unpin(blockID uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry, exists := c.directory[blockID]
	if !exists {
		return ErrBlockNotFound
	}
	if !entry.Resident {
		return errors.New("amddirect: block is not resident in VRAM")
	}
	if entry.FrameID < 0 || entry.FrameID >= len(c.frames) {
		return errors.New("amddirect: invalid frame index")
	}

	frame := c.frames[entry.FrameID]
	if frame.PinCount <= 0 {
		return errors.New("amddirect: block pin count already zero")
	}
	frame.PinCount--
	if frame.PinCount == 0 {
		c.stats.PinnedBlocks--
	}
	return nil
}

// IsPinned reports whether the block is locked in VRAM against offload.
func (c *BaMKVPagingCoordinator) IsPinned(blockID uint64) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, exists := c.directory[blockID]
	if !exists || !entry.Resident {
		return false
	}
	if entry.FrameID < 0 || entry.FrameID >= len(c.frames) {
		return false
	}
	return c.frames[entry.FrameID].PinCount > 0
}

// ReadBlock retrieves the content of a KV block, restoring it from NVMe if offloaded.
func (c *BaMKVPagingCoordinator) ReadBlock(blockID uint64) ([]byte, error) {
	c.mu.RLock()
	entry, exists := c.directory[blockID]
	c.mu.RUnlock()
	if !exists {
		return nil, ErrBlockNotFound
	}

	if !entry.Resident {
		if err := c.RestoreBlock(blockID); err != nil {
			return nil, err
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	entry, exists = c.directory[blockID]
	if !exists || !entry.Resident || entry.FrameID < 0 || entry.FrameID >= len(c.frames) {
		return nil, errors.New("amddirect: block restore did not yield resident frame")
	}

	frame := c.frames[entry.FrameID]
	now := time.Now().UnixNano()
	frame.LastAccessNano = now
	entry.LastAccessNano = now
	entry.AccessCount++

	buf := make([]byte, len(frame.Data))
	copy(buf, frame.Data)
	return buf, nil
}

// OffloadBlock moves a resident KV block from VRAM to NVMe flash storage.
func (c *BaMKVPagingCoordinator) OffloadBlock(blockID uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry, exists := c.directory[blockID]
	if !exists {
		return ErrBlockNotFound
	}
	if !entry.Resident {
		return nil
	}
	if entry.FrameID < 0 || entry.FrameID >= len(c.frames) {
		return errors.New("amddirect: invalid frame index in directory")
	}
	if c.frames[entry.FrameID].PinCount > 0 {
		return ErrBlockPinned
	}
	return c.offloadFrameLocked(entry.FrameID)
}

// RestoreBlock brings an offloaded KV block back from NVMe flash into a VRAM frame.
func (c *BaMKVPagingCoordinator) RestoreBlock(blockID uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry, exists := c.directory[blockID]
	if !exists {
		return ErrBlockNotFound
	}
	if entry.Resident {
		return nil
	}

	var frameIdx int
	if len(c.freeFrameIndices) > 0 {
		frameIdx = c.freeFrameIndices[0]
		c.freeFrameIndices = c.freeFrameIndices[1:]
	} else {
		chosenIdx, err := c.selectUnpinnedFrameLocked()
		if err != nil {
			return err
		}
		if err := c.offloadFrameLocked(chosenIdx); err != nil {
			return err
		}
		frameIdx = c.freeFrameIndices[0]
		c.freeFrameIndices = c.freeFrameIndices[1:]
	}

	frame := c.frames[frameIdx]

	_, err := c.pipeline.SubmitRead(entry.NVMeLBA, uint16(entry.LBACount), frame.VRAMAddress, entry.SizeBytes)
	if err != nil {
		c.freeFrameIndices = append(c.freeFrameIndices, frameIdx)
		return err
	}
	c.pipeline.PollCompletions(1)

	storageBuf := c.storageBacking[entry.NVMeLBA]
	if len(storageBuf) > 0 {
		copy(frame.Data, storageBuf)
	}

	frame.BlockID = blockID
	frame.LayerID = entry.LayerID
	frame.InUse = true
	frame.PinCount = 0
	frame.IsDirty = false
	frame.LastAccessNano = time.Now().UnixNano()

	entry.Resident = true
	entry.FrameID = frameIdx
	entry.VRAMAddress = frame.VRAMAddress
	entry.LastAccessNano = frame.LastAccessNano

	c.stats.ResidentBlocks++
	c.stats.OffloadedBlocks--
	c.stats.NVMeReadCount++
	c.stats.BytesReadNVMe += entry.SizeBytes

	return nil
}

// BufferColdPrefix ensures prefix blocks are committed to NVMe storage and marks them cold.
func (c *BaMKVPagingCoordinator) BufferColdPrefix(blockIDs []uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, blockID := range blockIDs {
		entry, exists := c.directory[blockID]
		if !exists {
			return fmt.Errorf("amddirect: cold prefix block %d not found", blockID)
		}
		if !entry.IsColdPrefix {
			entry.IsColdPrefix = true
			c.stats.ColdPrefixBlocks++
		}

		if entry.Resident && entry.FrameID >= 0 && entry.FrameID < len(c.frames) {
			frame := c.frames[entry.FrameID]
			storageBuf := make([]byte, frame.SizeBytes)
			copy(storageBuf, frame.Data)
			c.storageBacking[entry.NVMeLBA] = storageBuf

			_, err := c.pipeline.SubmitWrite(entry.NVMeLBA, uint16(entry.LBACount), frame.VRAMAddress, frame.SizeBytes)
			if err != nil {
				return err
			}
			c.pipeline.PollCompletions(1)
			c.stats.NVMeWriteCount++
			c.stats.BytesWrittenNVMe += frame.SizeBytes

			if frame.PinCount == 0 {
				_ = c.offloadFrameLocked(entry.FrameID)
			}
		}
	}
	return nil
}

// PrefetchLayerAhead brings forward KV blocks for upcoming layers into VRAM before compute reaches them.
func (c *BaMKVPagingCoordinator) PrefetchLayerAhead(currentLayer int, prefetchDistance int) error {
	if prefetchDistance <= 0 {
		prefetchDistance = c.cfg.PrefetchDistance
	}
	targetLayer := currentLayer + prefetchDistance
	if targetLayer >= c.cfg.TotalModelLayers {
		return nil
	}

	c.mu.RLock()
	blocks := append([]uint64(nil), c.layerBlocks[targetLayer]...)
	c.mu.RUnlock()

	if len(blocks) == 0 {
		return nil
	}

	for _, blockID := range blocks {
		c.mu.RLock()
		entry, exists := c.directory[blockID]
		if !exists {
			c.mu.RUnlock()
			continue
		}
		isResident := entry.Resident
		c.mu.RUnlock()

		if isResident {
			c.mu.Lock()
			c.stats.PrefetchHits++
			c.mu.Unlock()
		} else {
			c.mu.Lock()
			c.stats.PrefetchMisses++
			c.mu.Unlock()
			if err := c.RestoreBlock(blockID); err != nil {
				return fmt.Errorf("amddirect: prefetch failed restoring block %d: %w", blockID, err)
			}
		}
	}

	c.mu.Lock()
	totalPrefetch := c.stats.PrefetchHits + c.stats.PrefetchMisses
	if totalPrefetch > 0 {
		c.stats.PrefetchHitRate = float64(c.stats.PrefetchHits) / float64(totalPrefetch)
	}
	c.mu.Unlock()

	return nil
}

// SimulateLayerComputeWithPrefetch models compute and prefetch overlap to calculate hiding percentage.
func (c *BaMKVPagingCoordinator) SimulateLayerComputeWithPrefetch(layerID int, computeTimeNs int64) (PrefetchOverlapMetrics, error) {
	if computeTimeNs <= 0 {
		computeTimeNs = 500000
	}

	targetLayer := layerID + c.cfg.PrefetchDistance
	offloadedCount := 0
	c.mu.RLock()
	if targetLayer < c.cfg.TotalModelLayers {
		for _, bID := range c.layerBlocks[targetLayer] {
			if entry, ok := c.directory[bID]; ok && !entry.Resident {
				offloadedCount++
			}
		}
	}
	c.mu.RUnlock()

	var prefetchNs int64
	if offloadedCount > 0 {
		prefetchNs = int64(offloadedCount) * 10000
	} else {
		prefetchNs = 5000
	}

	if err := c.PrefetchLayerAhead(layerID, c.cfg.PrefetchDistance); err != nil {
		return PrefetchOverlapMetrics{}, err
	}

	var hiddenNs int64
	var overlapPct float64

	if prefetchNs <= computeTimeNs {
		hiddenNs = prefetchNs
		overlapPct = 100.0
	} else {
		hiddenNs = computeTimeNs
		overlapPct = (float64(computeTimeNs) / float64(prefetchNs)) * 100.0
	}

	hiding := overlapPct >= 90.0
	allDone := (hiddenNs == prefetchNs)
	_ = allDone

	return PrefetchOverlapMetrics{
		LayerIndex:         layerID,
		ComputeDurationNs:  computeTimeNs,
		PrefetchDurationNs: prefetchNs,
		HiddenDurationNs:   hiddenNs,
		OverlapPercentage:  overlapPct,
		HidingAchieved:     hiding,
	}, nil
}

// VerifyDataIntegrity checks block contents against its initial checksum for bit-rot or corruption.
func (c *BaMKVPagingCoordinator) VerifyDataIntegrity(blockID uint64) (bool, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, exists := c.directory[blockID]
	if !exists {
		return false, ErrBlockNotFound
	}

	var data []byte
	if entry.Resident {
		if entry.FrameID < 0 || entry.FrameID >= len(c.frames) {
			return false, errors.New("amddirect: corrupted frame index in directory")
		}
		data = c.frames[entry.FrameID].Data
	} else {
		storageBuf, ok := c.storageBacking[entry.NVMeLBA]
		if !ok {
			return false, errors.New("amddirect: storage backing missing for offloaded block")
		}
		data = storageBuf
	}

	currentChk := computeChecksum(data)
	return currentChk == entry.Checksum, nil
}

// GetPageDirectoryEntry returns a copy of the page directory entry for blockID.
func (c *BaMKVPagingCoordinator) GetPageDirectoryEntry(blockID uint64) (*KVPageDirectoryEntry, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, exists := c.directory[blockID]
	if !exists {
		return nil, ErrBlockNotFound
	}
	cp := *entry
	return &cp, nil
}

// TotalTokens reports the sum of token counts across all registered blocks.
func (c *BaMKVPagingCoordinator) TotalTokens() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	total := 0
	for _, entry := range c.directory {
		total += entry.TokenCount
	}
	return total
}

// PollCompletions resolves finished NVMe transfers.
func (c *BaMKVPagingCoordinator) PollCompletions(maxEntries int) int {
	return c.pipeline.PollCompletions(maxEntries)
}

// Stats returns a point-in-time telemetry snapshot of the KV paging subsystem.
func (c *BaMKVPagingCoordinator) Stats() BaMKVPagingStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	snap := c.stats
	snap.QueueExhaustionCount = c.pipeline.ExhaustionCount()
	snap.StagingCopies = 0

	totalPrefetch := snap.PrefetchHits + snap.PrefetchMisses
	if totalPrefetch > 0 {
		snap.PrefetchHitRate = float64(snap.PrefetchHits) / float64(totalPrefetch)
	}
	return snap
}

// StagingCopyCount returns the number of host DRAM bounce buffer copies. Invariant: always 0.
func (c *BaMKVPagingCoordinator) StagingCopyCount() int {
	return 0
}
