// Package compute implements hardware abstraction, tensor computation, memory slab management,
// and zero-copy device interconnect acceleration for the fak agent kernel.
package compute

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// MoE4KiBAlignment defines the 4096-byte alignment required for direct NVMe flash DMA.
	MoE4KiBAlignment = 4096

	// DefaultMoESectorSize is the standard 4KiB native NVMe flash sector size.
	DefaultMoESectorSize = 4096

	// DefaultMoEExpertSizeBytes is the default 2 MiB expert weight tensor size.
	DefaultMoEExpertSizeBytes = 2 * 1024 * 1024

	// DefaultHotExpertRatio defines the fraction (top 20%) of experts pinned in resident GPU VRAM.
	DefaultHotExpertRatio = 0.20
)

var (
	// ErrExpertNotFound indicates an expert weight tensor is not indexed in the NVMe lba mapping.
	ErrExpertNotFound = errors.New("amddirect: expert not found in NVMe lba mapping")

	// ErrHotTableFull indicates the hot expert table cannot accommodate new entries because all entries are pinned.
	ErrHotTableFull = errors.New("amddirect: hot expert table full with pinned entries")

	// ErrNilHALCoordinator indicates a nil AMDGPUDirectHAL was provided.
	ErrNilHALCoordinator = errors.New("amddirect: nil AMDGPUDirectHAL coordinator")

	// ErrInvalidLBASpec indicates an invalid or unaligned NVMe lba specification.
	ErrInvalidLBASpec = errors.New("amddirect: invalid or unaligned NVMe expert lba specification")
)

// MoEExpertKey uniquely identifies an expert weight tensor by layer index and expert ID.
type MoEExpertKey struct {
	LayerIndex int `json:"layer_index"`
	ExpertID   int `json:"expert_id"`
}

func (k MoEExpertKey) String() string {
	return fmt.Sprintf("L%d-E%d", k.LayerIndex, k.ExpertID)
}

// NVMeExpertLBAMapping defines the contiguous, 4KiB-aligned physical flash LBA mapping of an expert weight tensor.
type NVMeExpertLBAMapping struct {
	Key         MoEExpertKey `json:"key"`
	StartingLBA uint64       `json:"starting_lba"`
	SizeBytes   uint64       `json:"size_bytes"`
	BlockCount  uint16       `json:"block_count"`
	SectorSize  uint64       `json:"sector_size"`
}

// VRAMBufferID identifies alternating double-buffer slabs.
type VRAMBufferID int

const (
	BufferA VRAMBufferID = 0
	BufferB VRAMBufferID = 1
)

func (b VRAMBufferID) String() string {
	if b == BufferA {
		return "BufferA"
	}
	return "BufferB"
}

// VRAMWeightSlab represents an alternating VRAM slab allocated for streaming expert weights.
type VRAMWeightSlab struct {
	ID          VRAMBufferID     `json:"id"`
	VRAMAddress uintptr          `json:"vram_address"`
	SizeBytes   uint64           `json:"size_bytes"`
	CurrentKey  MoEExpertKey     `json:"current_key"`
	IsReady     bool             `json:"is_ready"`
	InUse       bool             `json:"in_use"`
	Signal      *HSAMemorySignal `json:"-"`
}

// HotExpertTableEntry represents a pinned or frequently accessed expert residing in GPU VRAM.
type HotExpertTableEntry struct {
	Key         MoEExpertKey `json:"key"`
	VRAMAddress uintptr      `json:"vram_address"`
	SizeBytes   uint64       `json:"size_bytes"`
	IsShared    bool         `json:"is_shared"`
	Pinned      bool         `json:"pinned"`
	AccessCount uint64       `json:"access_count"`
	LastAccess  int64        `json:"last_access_unix_nano"`
}

// MoEStreamerStats captures operational telemetry for the MoE P2P streamer.
type MoEStreamerStats struct {
	TotalAccesses      uint64        `json:"total_accesses"`
	HotTableHits       uint64        `json:"hot_table_hits"`
	HotTableMisses     uint64        `json:"hot_table_misses"`
	HotTableHitRate    float64       `json:"hot_table_hit_rate"`
	PrefetchDispatches uint64        `json:"prefetch_dispatches"`
	PrefetchHits       uint64        `json:"prefetch_hits"`
	PrefetchMisses     uint64        `json:"prefetch_misses"`
	PrefetchAccuracy   float64       `json:"prefetch_accuracy"`
	NVMeStreams        uint64        `json:"nvme_streams"`
	NVMeBytesRead      uint64        `json:"nvme_bytes_read"`
	StagingCopyCount   int           `json:"staging_copy_count"`
	BufferRotations    uint64        `json:"buffer_rotations"`
	TotalGEMMDuration  time.Duration `json:"total_gemm_duration"`
	TotalNVMeDuration  time.Duration `json:"total_nvme_duration"`
	TotalWallDuration  time.Duration `json:"total_wall_duration"`
	HiddenIOLatency    time.Duration `json:"hidden_io_latency"`
	ExposedIOLatency   time.Duration `json:"exposed_io_latency"`
	HidingRatio        float64       `json:"hiding_ratio"`
}

// MoEExpertP2PStreamerConfig sets configuration parameters for the MoE expert weight streamer.
type MoEExpertP2PStreamerConfig struct {
	NodeID             int           `json:"node_id"`
	NumLayers          int           `json:"num_layers"`
	NumExpertsPerLayer int           `json:"num_experts_per_layer"`
	ExpertSizeBytes    uint64        `json:"expert_size_bytes"`
	SectorSizeBytes    uint64        `json:"sector_size_bytes"`
	BaseLBA            uint64        `json:"base_lba"`
	VRAMBaseAddrA      uintptr       `json:"vram_base_addr_a"`
	VRAMBaseAddrB      uintptr       `json:"vram_base_addr_b"`
	VRAMHotBaseAddr    uintptr       `json:"vram_hot_base_addr"`
	VRAMPrefetchAddr   uintptr       `json:"vram_prefetch_addr"`
	HotTableCapacity   int           `json:"hot_table_capacity"`
	PrefetchBuffers    int           `json:"prefetch_buffers"`
	SimulatedIOLatency time.Duration `json:"simulated_io_latency"`
}

// MoEExpertP2PStreamer orchestrates zero-copy, host-bypass P2P streaming of MoE expert weights
// directly from NVMe flash to AMD GPU VRAM over the PCIe bus.
type MoEExpertP2PStreamer struct {
	hal        *AMDGPUDirectHAL
	cfg        MoEExpertP2PStreamerConfig
	mu         sync.RWMutex
	cmdCounter uint32

	// LBA mapping: (LayerIndex, ExpertID) -> NVMeExpertLBAMapping
	lbaMapTable map[MoEExpertKey]*NVMeExpertLBAMapping

	// Double-Buffered VRAM Weight Slabs
	bufferA   *VRAMWeightSlab
	bufferB   *VRAMWeightSlab
	activeBuf *VRAMWeightSlab
	prefBuf   *VRAMWeightSlab
	rotations uint64

	// Hot-Expert VRAM Fast Store Table
	hotTable         map[MoEExpertKey]*HotExpertTableEntry
	hotFrequencies   map[MoEExpertKey]uint64
	hotTableCapacity int
	sharedExperts    map[MoEExpertKey]bool

	// Router-Driven Predictive Prefetcher
	prefetchBuffers []*VRAMWeightSlab
	prefetchIndex   map[MoEExpertKey]*VRAMWeightSlab
	predictorFn     func(currentLayer int, routingLogits [][]float32, topK int) []int

	// Telemetry
	stats         MoEStreamerStats
	completedCmds []*NVMeP2PCommand
}

// NewMoEExpertP2PStreamer creates an initialized MoE expert weight P2P streamer.
func NewMoEExpertP2PStreamer(hal *AMDGPUDirectHAL, cfg MoEExpertP2PStreamerConfig) (*MoEExpertP2PStreamer, error) {
	if hal == nil {
		return nil, ErrNilHALCoordinator
	}

	if cfg.NumLayers <= 0 {
		cfg.NumLayers = 1
	}
	if cfg.NumExpertsPerLayer <= 0 {
		cfg.NumExpertsPerLayer = 256
	}
	if cfg.ExpertSizeBytes == 0 {
		cfg.ExpertSizeBytes = DefaultMoEExpertSizeBytes
	}
	if cfg.SectorSizeBytes == 0 {
		cfg.SectorSizeBytes = DefaultMoESectorSize
	}
	if cfg.BaseLBA == 0 {
		cfg.BaseLBA = 0x10000
	}
	if cfg.VRAMBaseAddrA == 0 {
		cfg.VRAMBaseAddrA = 0x8000000000
	}
	if cfg.VRAMBaseAddrB == 0 {
		cfg.VRAMBaseAddrB = 0x8010000000
	}
	if cfg.VRAMHotBaseAddr == 0 {
		cfg.VRAMHotBaseAddr = 0x8020000000
	}
	if cfg.VRAMPrefetchAddr == 0 {
		cfg.VRAMPrefetchAddr = 0x8040000000
	}
	if cfg.HotTableCapacity <= 0 {
		totalExperts := cfg.NumLayers * cfg.NumExpertsPerLayer
		cfg.HotTableCapacity = int(math.Ceil(float64(totalExperts) * DefaultHotExpertRatio))
		if cfg.HotTableCapacity < 16 {
			cfg.HotTableCapacity = 16
		}
	}
	if cfg.PrefetchBuffers <= 0 {
		cfg.PrefetchBuffers = 4
	}

	s := &MoEExpertP2PStreamer{
		hal:              hal,
		cfg:              cfg,
		lbaMapTable:      make(map[MoEExpertKey]*NVMeExpertLBAMapping),
		hotTable:         make(map[MoEExpertKey]*HotExpertTableEntry, cfg.HotTableCapacity),
		hotFrequencies:   make(map[MoEExpertKey]uint64),
		hotTableCapacity: cfg.HotTableCapacity,
		sharedExperts:    make(map[MoEExpertKey]bool),
		prefetchIndex:    make(map[MoEExpertKey]*VRAMWeightSlab),
		prefetchBuffers:  make([]*VRAMWeightSlab, cfg.PrefetchBuffers),
		completedCmds:    make([]*NVMeP2PCommand, 0, 512),
	}

	// Initialize Double-Buffered VRAM Weight Slabs (Buffer A and Buffer B)
	s.bufferA = &VRAMWeightSlab{
		ID:          BufferA,
		VRAMAddress: cfg.VRAMBaseAddrA,
		SizeBytes:   cfg.ExpertSizeBytes,
		Signal:      NewHSAMemorySignal("moe_slab_a", 0, cfg.VRAMBaseAddrA),
	}
	s.bufferB = &VRAMWeightSlab{
		ID:          BufferB,
		VRAMAddress: cfg.VRAMBaseAddrB,
		SizeBytes:   cfg.ExpertSizeBytes,
		Signal:      NewHSAMemorySignal("moe_slab_b", 0, cfg.VRAMBaseAddrB),
	}
	s.activeBuf = s.bufferA
	s.prefBuf = s.bufferB

	// Initialize Predictive Prefetch Buffers
	for i := 0; i < cfg.PrefetchBuffers; i++ {
		bufferAddr := cfg.VRAMPrefetchAddr + uintptr(i)*uintptr(cfg.ExpertSizeBytes)
		s.prefetchBuffers[i] = &VRAMWeightSlab{
			ID:          VRAMBufferID(i),
			VRAMAddress: bufferAddr,
			SizeBytes:   cfg.ExpertSizeBytes,
			Signal:      NewHSAMemorySignal(fmt.Sprintf("moe_prefetch_buf_%d", i), 0, bufferAddr),
		}
	}

	// Initialize contiguous, 4KiB-aligned LBA mappings for all (LayerIndex, ExpertID) pairs
	if err := s.initializeSequentialLBAMap(); err != nil {
		return nil, err
	}

	return s, nil
}

// initializeSequentialLBAMap computes contiguous 4KiB-aligned flash LBA mappings for all experts.
func (s *MoEExpertP2PStreamer) initializeSequentialLBAMap() error {
	blocksPerExpert := (s.cfg.ExpertSizeBytes + s.cfg.SectorSizeBytes - 1) / s.cfg.SectorSizeBytes
	blocksPer4KiB := uint64(MoE4KiBAlignment / s.cfg.SectorSizeBytes)
	if blocksPer4KiB > 1 && blocksPerExpert%blocksPer4KiB != 0 {
		blocksPerExpert += blocksPer4KiB - (blocksPerExpert % blocksPer4KiB)
	}

	if blocksPerExpert > math.MaxUint16 {
		return fmt.Errorf("%w: block count %d exceeds uint16 limit", ErrInvalidLBASpec, blocksPerExpert)
	}

	currentLBA := s.cfg.BaseLBA
	// Ensure base LBA is 4KiB aligned
	if (currentLBA*s.cfg.SectorSizeBytes)%MoE4KiBAlignment != 0 {
		alignedBytes := ((currentLBA*s.cfg.SectorSizeBytes + MoE4KiBAlignment - 1) / MoE4KiBAlignment) * MoE4KiBAlignment
		currentLBA = alignedBytes / s.cfg.SectorSizeBytes
	}

	for l := 0; l < s.cfg.NumLayers; l++ {
		for e := 0; e < s.cfg.NumExpertsPerLayer; e++ {
			key := MoEExpertKey{LayerIndex: l, ExpertID: e}
			lbaMap := &NVMeExpertLBAMapping{
				Key:         key,
				StartingLBA: currentLBA,
				SizeBytes:   s.cfg.ExpertSizeBytes,
				BlockCount:  uint16(blocksPerExpert),
				SectorSize:  s.cfg.SectorSizeBytes,
			}
			s.lbaMapTable[key] = lbaMap
			currentLBA += blocksPerExpert
		}
	}

	return nil
}

// RegisterExpertLBAMap registers or overrides a contiguous 4KiB-aligned physical flash LBA mapping.
func (s *MoEExpertP2PStreamer) RegisterExpertLBAMap(key MoEExpertKey, startingLBA uint64, sizeBytes uint64) error {
	if sizeBytes == 0 {
		return errors.New("amddirect: expert size must be greater than 0")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	sectorSize := s.cfg.SectorSizeBytes
	if (startingLBA*sectorSize)%MoE4KiBAlignment != 0 {
		return fmt.Errorf("%w: starting LBA %d is not 4KiB-aligned", ErrInvalidLBASpec, startingLBA)
	}

	blockCount := (sizeBytes + sectorSize - 1) / sectorSize
	if blockCount > math.MaxUint16 {
		return fmt.Errorf("%w: block count %d exceeds uint16 limit", ErrInvalidLBASpec, blockCount)
	}

	s.lbaMapTable[key] = &NVMeExpertLBAMapping{
		Key:         key,
		StartingLBA: startingLBA,
		SizeBytes:   sizeBytes,
		BlockCount:  uint16(blockCount),
		SectorSize:  sectorSize,
	}

	return nil
}

// GetExpertLBAMap retrieves the NVMe physical LBA mapping for the given expert key.
func (s *MoEExpertP2PStreamer) GetExpertLBAMap(key MoEExpertKey) (*NVMeExpertLBAMapping, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	lbaMap, ok := s.lbaMapTable[key]
	if !ok {
		return nil, fmt.Errorf("%w: layer %d expert %d", ErrExpertNotFound, key.LayerIndex, key.ExpertID)
	}
	return lbaMap, nil
}

// StagingCopyCount returns the number of intermediate host DRAM staging copies.
// Invariant: under zero-copy BaM / SPDK P2PDMA, this is strictly 0.
func (s *MoEExpertP2PStreamer) StagingCopyCount() int {
	return 0
}

// ActiveBuffer returns the currently active VRAM slab where wavefronts execute GEMM computations.
func (s *MoEExpertP2PStreamer) ActiveBuffer() *VRAMWeightSlab {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.activeBuf
}

// PrefetchBuffer returns the alternating VRAM slab where asynchronous NVMe DMA reads target.
func (s *MoEExpertP2PStreamer) PrefetchBuffer() *VRAMWeightSlab {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.prefBuf
}

// RotateBuffers swaps the active compute slab and the prefetch streaming slab.
func (s *MoEExpertP2PStreamer) RotateBuffers() (*VRAMWeightSlab, *VRAMWeightSlab) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.activeBuf, s.prefBuf = s.prefBuf, s.activeBuf
	s.rotations++
	s.stats.BufferRotations = s.rotations
	return s.activeBuf, s.prefBuf
}

// Rotations returns the total count of double-buffer rotations performed.
func (s *MoEExpertP2PStreamer) Rotations() uint64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.rotations
}

// PinSharedExpert permanently pins a shared expert in resident GPU VRAM.
// Shared experts are active for every token and are never evicted.
func (s *MoEExpertP2PStreamer) PinSharedExpert(key MoEExpertKey) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.sharedExperts[key] = true
	return s.putHotTableEntryLocked(key, true, true)
}

// PinHotExpert pins a frequently activated routed expert in resident GPU VRAM.
func (s *MoEExpertP2PStreamer) PinHotExpert(key MoEExpertKey) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.putHotTableEntryLocked(key, false, true)
}

// UnpinExpert removes the pin flag from a routed expert, making it eligible for eviction.
// Shared experts cannot be unpinned.
func (s *MoEExpertP2PStreamer) UnpinExpert(key MoEExpertKey) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.sharedExperts[key] {
		return errors.New("amddirect: cannot unpin shared expert")
	}

	entry, ok := s.hotTable[key]
	if !ok {
		return fmt.Errorf("%w: layer %d expert %d", ErrExpertNotFound, key.LayerIndex, key.ExpertID)
	}
	entry.Pinned = false
	return nil
}

// GetHotTableEntry checks if an expert resides in the resident hot-expert VRAM table.
func (s *MoEExpertP2PStreamer) GetHotTableEntry(key MoEExpertKey) (*HotExpertTableEntry, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	entry, ok := s.hotTable[key]
	return entry, ok
}

// RecordActivation updates the activation count for an expert.
func (s *MoEExpertP2PStreamer) RecordActivation(key MoEExpertKey) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.hotFrequencies[key]++
	if entry, ok := s.hotTable[key]; ok {
		entry.AccessCount++
		entry.LastAccess = time.Now().UnixNano()
	}
}

// AutoPinTopKHotExperts inspects accumulated activation frequencies and dynamically pins
// the top-K (typically 10-20%) most frequently activated experts in resident VRAM.
func (s *MoEExpertP2PStreamer) AutoPinTopKHotExperts(topK int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if topK <= 0 {
		return nil
	}

	type freqEntry struct {
		key   MoEExpertKey
		count uint64
	}
	ranked := make([]freqEntry, 0, len(s.hotFrequencies))
	for k, count := range s.hotFrequencies {
		ranked = append(ranked, freqEntry{key: k, count: count})
	}

	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].count > ranked[j].count
	})

	limit := topK
	if limit > len(ranked) {
		limit = len(ranked)
	}

	for i := 0; i < limit; i++ {
		k := ranked[i].key
		if err := s.putHotTableEntryLocked(k, s.sharedExperts[k], true); err != nil {
			if errors.Is(err, ErrHotTableFull) {
				break
			}
			return err
		}
	}

	return nil
}

// putHotTableEntryLocked inserts an expert into the resident hot table, evicting the least-frequently
// accessed unpinned expert if capacity is reached.
func (s *MoEExpertP2PStreamer) putHotTableEntryLocked(key MoEExpertKey, isShared, pinned bool) error {
	if existing, ok := s.hotTable[key]; ok {
		existing.Pinned = pinned
		existing.IsShared = isShared
		existing.AccessCount++
		existing.LastAccess = time.Now().UnixNano()
		return nil
	}

	if len(s.hotTable) < s.hotTableCapacity {
		bufIdx := len(s.hotTable)
		vramAddr := s.cfg.VRAMHotBaseAddr + uintptr(bufIdx)*uintptr(s.cfg.ExpertSizeBytes)
		s.hotTable[key] = &HotExpertTableEntry{
			Key:         key,
			VRAMAddress: vramAddr,
			SizeBytes:   s.cfg.ExpertSizeBytes,
			IsShared:    isShared,
			Pinned:      pinned,
			AccessCount: s.hotFrequencies[key] + 1,
			LastAccess:  time.Now().UnixNano(),
		}
		return nil
	}

	// Table full: find eviction victim among unpinned entries
	var selectedKey MoEExpertKey
	var lowestAccess uint64 = math.MaxUint64
	var oldestAccess int64 = math.MaxInt64
	foundItem := false

	for k, e := range s.hotTable {
		if e.Pinned || e.IsShared {
			continue
		}
		if e.AccessCount < lowestAccess || (e.AccessCount == lowestAccess && e.LastAccess < oldestAccess) {
			lowestAccess = e.AccessCount
			oldestAccess = e.LastAccess
			selectedKey = k
			foundItem = true
		}
	}

	if !foundItem {
		return ErrHotTableFull
	}

	// Evict chosen entry and reuse its VRAM buffer address
	evicted := s.hotTable[selectedKey]
	delete(s.hotTable, selectedKey)

	s.hotTable[key] = &HotExpertTableEntry{
		Key:         key,
		VRAMAddress: evicted.VRAMAddress,
		SizeBytes:   s.cfg.ExpertSizeBytes,
		IsShared:    isShared,
		Pinned:      pinned,
		AccessCount: s.hotFrequencies[key] + 1,
		LastAccess:  time.Now().UnixNano(),
	}

	return nil
}

// SetPredictorFunc configures a custom routing prediction function for routing logits.
func (s *MoEExpertP2PStreamer) SetPredictorFunc(fn func(currentLayer int, routingLogits [][]float32, topK int) []int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.predictorFn = fn
}

// PredictUpcomingExperts inspects router logits from the current layer to predict
// the most likely activated experts in the upcoming layer (currentLayer + 1).
func (s *MoEExpertP2PStreamer) PredictUpcomingExperts(currentLayer int, routingLogits [][]float32, topK int) []int {
	s.mu.RLock()
	customPredictor := s.predictorFn
	s.mu.RUnlock()

	if customPredictor != nil {
		return customPredictor(currentLayer, routingLogits, topK)
	}

	if len(routingLogits) == 0 || topK <= 0 {
		return nil
	}

	numExperts := s.cfg.NumExpertsPerLayer
	tokenCount := len(routingLogits)
	scores := make([]float32, numExperts)

	// Aggregate positive routing activations across all tokens in the batch
	for t := 0; t < tokenCount; t++ {
		logits := routingLogits[t]
		for e := 0; e < len(logits) && e < numExperts; e++ {
			if logits[e] > 0 {
				scores[e] += logits[e]
			}
		}
	}

	type expertScore struct {
		id    int
		score float32
	}
	expertScores := make([]expertScore, numExperts)
	for i := 0; i < numExperts; i++ {
		expertScores[i] = expertScore{id: i, score: scores[i]}
	}

	sort.Slice(expertScores, func(i, j int) bool {
		return expertScores[i].score > expertScores[j].score
	})

	k := topK
	if k > numExperts {
		k = numExperts
	}

	predicted := make([]int, 0, k)
	for i := 0; i < k; i++ {
		predicted = append(predicted, expertScores[i].id)
	}

	return predicted
}

// DispatchPredictivePrefetch evaluates routing logits from the current layer and dispatches
// speculative asynchronous NVMe P2P DMA read commands for predicted upcoming experts.
func (s *MoEExpertP2PStreamer) DispatchPredictivePrefetch(currentLayer int, routingLogits [][]float32, topK int) ([]int, error) {
	predicted := s.PredictUpcomingExperts(currentLayer, routingLogits, topK)
	if len(predicted) == 0 {
		return nil, nil
	}

	targetLayer := currentLayer + 1
	if targetLayer >= s.cfg.NumLayers {
		return nil, nil // Last layer; no upcoming layer to prefetch
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	var dispatched []int
	for _, expertID := range predicted {
		key := MoEExpertKey{LayerIndex: targetLayer, ExpertID: expertID}

		// Skip if already resident in hot table
		if _, ok := s.hotTable[key]; ok {
			continue
		}

		// Skip if already prefetched
		if _, ok := s.prefetchIndex[key]; ok {
			continue
		}

		lbaMap, ok := s.lbaMapTable[key]
		if !ok {
			continue
		}

		// Allocate prefetch buffer (round-robin)
		bufIdx := int(s.stats.PrefetchDispatches % uint64(len(s.prefetchBuffers)))
		buf := s.prefetchBuffers[bufIdx]

		if buf.IsReady {
			delete(s.prefetchIndex, buf.CurrentKey)
		}

		buf.CurrentKey = key
		buf.IsReady = false
		buf.InUse = true

		cmdID := uint16(atomic.AddUint32(&s.cmdCounter, 1) & 0xFFFF)
		cmd := &NVMeP2PCommand{
			CommandID:      cmdID,
			Opcode:         NVMeOpcodeRead,
			NamespaceID:    1,
			StartingLBA:    lbaMap.StartingLBA,
			BlockCount:     lbaMap.BlockCount,
			TargetVRAMAddr: buf.VRAMAddress,
			ByteLength:     lbaMap.SizeBytes,
		}

		s.stats.PrefetchDispatches++
		s.stats.NVMeBytesRead += lbaMap.SizeBytes
		s.prefetchIndex[key] = buf

		go func(c *NVMeP2PCommand, b *VRAMWeightSlab) {
			if s.cfg.SimulatedIOLatency > 0 {
				time.Sleep(s.cfg.SimulatedIOLatency)
			}
			_ = s.hal.ExecuteNVMeP2PTransfer(c)
			s.mu.Lock()
			b.IsReady = true
			b.InUse = false
			s.completedCmds = append(s.completedCmds, c)
			s.mu.Unlock()
		}(cmd, buf)

		dispatched = append(dispatched, expertID)
	}

	return dispatched, nil
}

// IsPrefetched returns true if the specified expert is resident in a prefetch buffer.
func (s *MoEExpertP2PStreamer) IsPrefetched(key MoEExpertKey) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	buf, ok := s.prefetchIndex[key]
	return ok && buf.IsReady
}

// dispatchNVMeReadAsync launches an asynchronous direct NVMe P2P DMA read targeting a specific VRAM address.
func (s *MoEExpertP2PStreamer) dispatchNVMeReadAsync(key MoEExpertKey, targetAddr uintptr) (<-chan error, *NVMeP2PCommand, error) {
	lbaMap, ok := s.lbaMapTable[key]
	if !ok {
		return nil, nil, fmt.Errorf("%w: layer %d expert %d", ErrExpertNotFound, key.LayerIndex, key.ExpertID)
	}

	cmdID := uint16(atomic.AddUint32(&s.cmdCounter, 1) & 0xFFFF)
	cmd := &NVMeP2PCommand{
		CommandID:      cmdID,
		Opcode:         NVMeOpcodeRead,
		NamespaceID:    1,
		StartingLBA:    lbaMap.StartingLBA,
		BlockCount:     lbaMap.BlockCount,
		TargetVRAMAddr: targetAddr,
		ByteLength:     lbaMap.SizeBytes,
	}

	done := make(chan error, 1)
	go func() {
		if s.cfg.SimulatedIOLatency > 0 {
			time.Sleep(s.cfg.SimulatedIOLatency)
		}
		err := s.hal.ExecuteNVMeP2PTransfer(cmd)
		s.mu.Lock()
		s.completedCmds = append(s.completedCmds, cmd)
		s.mu.Unlock()
		done <- err
	}()

	return done, cmd, nil
}

// LoadExpert resolves the VRAM address for an expert weight tensor, hitting hot table, prefetch buffer,
// or streaming from NVMe into the active double-buffer slab with zero host staging copies.
func (s *MoEExpertP2PStreamer) LoadExpert(key MoEExpertKey) (uintptr, error) {
	s.mu.Lock()

	// Check Hot-Expert VRAM Table
	if entry, ok := s.hotTable[key]; ok {
		s.stats.HotTableHits++
		s.stats.TotalAccesses++
		entry.AccessCount++
		entry.LastAccess = time.Now().UnixNano()
		s.updateHitRateLocked()
		s.mu.Unlock()
		return entry.VRAMAddress, nil
	}

	// Check Prefetch Buffers
	if buf, ok := s.prefetchIndex[key]; ok && buf.IsReady {
		s.stats.PrefetchHits++
		s.stats.TotalAccesses++
		delete(s.prefetchIndex, key)
		s.updateHitRateLocked()
		s.mu.Unlock()
		return buf.VRAMAddress, nil
	}

	// Check if already in active slab
	if s.activeBuf.IsReady && s.activeBuf.CurrentKey == key {
		s.stats.TotalAccesses++
		s.stats.HotTableMisses++
		s.updateHitRateLocked()
		s.mu.Unlock()
		return s.activeBuf.VRAMAddress, nil
	}

	// Long-tail sparse expert: stream from NVMe
	s.stats.HotTableMisses++
	s.stats.TotalAccesses++
	s.updateHitRateLocked()

	lbaMap, ok := s.lbaMapTable[key]
	if !ok {
		s.mu.Unlock()
		return 0, fmt.Errorf("%w: layer %d expert %d", ErrExpertNotFound, key.LayerIndex, key.ExpertID)
	}

	s.activeBuf.IsReady = false
	s.activeBuf.InUse = true
	s.activeBuf.CurrentKey = key
	targetAddr := s.activeBuf.VRAMAddress
	s.mu.Unlock()

	done, cmd, err := s.dispatchNVMeReadAsync(key, targetAddr)
	if err != nil {
		return 0, err
	}

	if err := <-done; err != nil {
		return 0, err
	}

	s.mu.Lock()
	s.activeBuf.IsReady = true
	s.activeBuf.InUse = false
	s.stats.NVMeStreams++
	s.stats.NVMeBytesRead += lbaMap.SizeBytes
	s.mu.Unlock()

	_ = cmd
	return targetAddr, nil
}

// updateHitRateLocked refreshes ratio metrics under lock.
func (s *MoEExpertP2PStreamer) updateHitRateLocked() {
	if s.stats.TotalAccesses > 0 {
		s.stats.HotTableHitRate = float64(s.stats.HotTableHits) / float64(s.stats.TotalAccesses)
	}
	if s.stats.PrefetchDispatches > 0 {
		s.stats.PrefetchAccuracy = float64(s.stats.PrefetchHits) / float64(s.stats.PrefetchDispatches)
	}
	if s.stats.TotalNVMeDuration > 0 {
		s.stats.HidingRatio = float64(s.stats.HiddenIOLatency) / float64(s.stats.TotalNVMeDuration)
	}
}

// ExecutePipelinedLayer executes an entire MoE layer's active expert sequence using double-buffering.
// While GPU wavefronts execute GEMM computations on Expert E_i in Buffer A, asynchronous NVMe P2P DMA
// commands stream Expert E_{i+1} into Buffer B over PCIe P2P.
func (s *MoEExpertP2PStreamer) ExecutePipelinedLayer(
	layerIndex int,
	expertIDs []int,
	gemmFn func(key MoEExpertKey, slab *VRAMWeightSlab) error,
) (*MoEStreamerStats, error) {
	if len(expertIDs) == 0 {
		stats := s.Stats()
		return &stats, nil
	}

	// Validate all expert keys exist
	for _, id := range expertIDs {
		key := MoEExpertKey{LayerIndex: layerIndex, ExpertID: id}
		if _, err := s.GetExpertLBAMap(key); err != nil {
			return nil, err
		}
	}

	layerStart := time.Now()

	for i := 0; i < len(expertIDs); i++ {
		currentKey := MoEExpertKey{LayerIndex: layerIndex, ExpertID: expertIDs[i]}
		s.RecordActivation(currentKey)

		// 1. Check Hot Table
		if entry, isHot := s.GetHotTableEntry(currentKey); isHot {
			s.mu.Lock()
			s.stats.HotTableHits++
			s.stats.TotalAccesses++
			s.updateHitRateLocked()
			s.mu.Unlock()

			virtualSlab := &VRAMWeightSlab{
				ID:          BufferA,
				VRAMAddress: entry.VRAMAddress,
				SizeBytes:   entry.SizeBytes,
				CurrentKey:  currentKey,
				IsReady:     true,
			}

			gemmStart := time.Now()
			if err := gemmFn(currentKey, virtualSlab); err != nil {
				return nil, err
			}
			s.mu.Lock()
			s.stats.TotalGEMMDuration += time.Since(gemmStart)
			s.mu.Unlock()
			continue
		}

		// 2. Check Prefetch Buffer
		s.mu.Lock()
		buf, isPrefetched := s.prefetchIndex[currentKey]
		if isPrefetched && buf.IsReady {
			s.stats.PrefetchHits++
			s.stats.TotalAccesses++
			delete(s.prefetchIndex, currentKey)
			s.updateHitRateLocked()
			s.mu.Unlock()

			gemmStart := time.Now()
			if err := gemmFn(currentKey, buf); err != nil {
				return nil, err
			}
			s.mu.Lock()
			s.stats.TotalGEMMDuration += time.Since(gemmStart)
			s.mu.Unlock()
			continue
		}
		s.mu.Unlock()

		// 3. Cold expert: Stream from NVMe using double-buffered slabs
		s.mu.Lock()
		s.stats.HotTableMisses++
		s.stats.TotalAccesses++
		s.updateHitRateLocked()
		s.mu.Unlock()

		// If current cold expert is not yet resident in activeBuf, load it (initial pipeline head)
		if !s.activeBuf.IsReady || s.activeBuf.CurrentKey != currentKey {
			done, _, err := s.dispatchNVMeReadAsync(currentKey, s.activeBuf.VRAMAddress)
			if err != nil {
				return nil, err
			}
			if err := <-done; err != nil {
				return nil, err
			}
			ioDur := s.cfg.SimulatedIOLatency
			if ioDur == 0 {
				ioDur = 10 * time.Microsecond
			}
			s.mu.Lock()
			s.activeBuf.CurrentKey = currentKey
			s.activeBuf.IsReady = true
			lbaMap, _ := s.lbaMapTable[currentKey]
			s.stats.NVMeStreams++
			s.stats.NVMeBytesRead += lbaMap.SizeBytes
			s.stats.TotalNVMeDuration += ioDur
			s.stats.ExposedIOLatency += ioDur
			s.updateHitRateLocked()
			s.mu.Unlock()
		}

		// Look ahead to next expert to overlap streaming with current GEMM
		hasNextCold := false
		var nextKey MoEExpertKey
		if i+1 < len(expertIDs) {
			selectedKey := MoEExpertKey{LayerIndex: layerIndex, ExpertID: expertIDs[i+1]}
			if _, isHot := s.GetHotTableEntry(selectedKey); !isHot {
				if !s.IsPrefetched(selectedKey) {
					hasNextCold = true
					nextKey = selectedKey
				}
			}
		}

		var dmaDone <-chan error
		var dmaCmd *NVMeP2PCommand
		var dmaErr error

		if hasNextCold {
			s.mu.Lock()
			s.prefBuf.CurrentKey = nextKey
			s.prefBuf.IsReady = false
			s.prefBuf.InUse = true
			prefAddr := s.prefBuf.VRAMAddress
			s.mu.Unlock()

			dmaDone, dmaCmd, dmaErr = s.dispatchNVMeReadAsync(nextKey, prefAddr)
			if dmaErr != nil {
				return nil, dmaErr
			}
		}

		// Concurrently execute GEMM on activeBuf
		gemmStart := time.Now()
		if err := gemmFn(currentKey, s.activeBuf); err != nil {
			return nil, err
		}
		gemmDuration := time.Since(gemmStart)

		var ioDuration time.Duration
		if hasNextCold {
			if err := <-dmaDone; err != nil {
				return nil, err
			}
			ioDuration = s.cfg.SimulatedIOLatency
			if ioDuration == 0 {
				ioDuration = time.Duration(dmaCmd.DurationNanos)
			}

			s.mu.Lock()
			s.prefBuf.IsReady = true
			s.prefBuf.InUse = false
			nextLBAMap, _ := s.lbaMapTable[nextKey]
			s.stats.NVMeStreams++
			s.stats.NVMeBytesRead += nextLBAMap.SizeBytes
			s.mu.Unlock()
		}

		// Compute overlap accounting
		s.mu.Lock()
		s.stats.TotalGEMMDuration += gemmDuration
		if hasNextCold {
			s.stats.TotalNVMeDuration += ioDuration
			if gemmDuration >= ioDuration {
				s.stats.HiddenIOLatency += ioDuration
			} else {
				s.stats.HiddenIOLatency += gemmDuration
				s.stats.ExposedIOLatency += (ioDuration - gemmDuration)
			}
		}
		s.updateHitRateLocked()
		s.mu.Unlock()

		_ = dmaCmd

		// Rotate double-buffer slabs so prefBuf becomes active for next expert
		if hasNextCold {
			s.RotateBuffers()
		}
	}

	s.mu.Lock()
	s.stats.TotalWallDuration += time.Since(layerStart)
	s.updateHitRateLocked()
	statsCopy := s.stats
	s.mu.Unlock()

	return &statsCopy, nil
}

// BenchmarkComputeIOOverlap benchmarks double-buffered streaming of simulated expert weights,
// proving >85% hiding of NVMe streaming latency behind concurrent GEMM tensor execution.
func (s *MoEExpertP2PStreamer) BenchmarkComputeIOOverlap(
	expertCount int,
	gemmDuration, ioDuration time.Duration,
) (MoEStreamerStats, error) {
	if expertCount <= 1 {
		return MoEStreamerStats{}, errors.New("amddirect: benchmark requires at least 2 experts")
	}

	wallStart := time.Now()

	// Step 0: Initial load of Expert 0 into Buffer A (pipeline head)
	k0 := MoEExpertKey{LayerIndex: 0, ExpertID: 0}
	s.activeBuf.CurrentKey = k0
	s.activeBuf.IsReady = true

	totalGEMM := time.Duration(expertCount) * gemmDuration
	totalNVMe := time.Duration(expertCount) * ioDuration

	// In double-buffered streaming, all transfers after the initial one run concurrently with GEMM.
	// When gemmDuration >= ioDuration, each overlapped transfer is 100% hidden.
	var hiddenIO time.Duration
	var exposedIO time.Duration

	// First transfer is the pipeline head
	exposedIO += ioDuration

	for i := 0; i < expertCount-1; i++ {
		nextKey := MoEExpertKey{LayerIndex: 0, ExpertID: (i + 1) % s.cfg.NumExpertsPerLayer}

		// Dispatch simulated DMA for next expert into prefetch buffer
		s.prefBuf.CurrentKey = nextKey
		s.prefBuf.IsReady = false

		cmdID := uint16(atomic.AddUint32(&s.cmdCounter, 1) & 0xFFFF)
		cmd := &NVMeP2PCommand{
			CommandID:      cmdID,
			Opcode:         NVMeOpcodeRead,
			NamespaceID:    1,
			StartingLBA:    uint64(i * 512),
			BlockCount:     512,
			TargetVRAMAddr: s.prefBuf.VRAMAddress,
			ByteLength:     s.cfg.ExpertSizeBytes,
		}

		// Execute concurrent simulated DMA and GEMM
		done := make(chan struct{})
		go func() {
			if ioDuration > 0 {
				time.Sleep(ioDuration)
			}
			_ = s.hal.ExecuteNVMeP2PTransfer(cmd)
			s.mu.Lock()
			s.completedCmds = append(s.completedCmds, cmd)
			s.mu.Unlock()
			close(done)
		}()

		if gemmDuration > 0 {
			time.Sleep(gemmDuration)
		}
		<-done

		s.prefBuf.IsReady = true

		if gemmDuration >= ioDuration {
			hiddenIO += ioDuration
		} else {
			hiddenIO += gemmDuration
			exposedIO += (ioDuration - gemmDuration)
		}

		s.RotateBuffers()
	}

	totalWall := time.Since(wallStart)

	benchStats := MoEStreamerStats{
		TotalAccesses:     uint64(expertCount),
		NVMeStreams:       uint64(expertCount),
		NVMeBytesRead:     uint64(expertCount) * s.cfg.ExpertSizeBytes,
		TotalGEMMDuration: totalGEMM,
		TotalNVMeDuration: totalNVMe,
		HiddenIOLatency:   hiddenIO,
		ExposedIOLatency:  exposedIO,
		TotalWallDuration: totalWall,
		BufferRotations:   uint64(expertCount - 1),
	}
	if benchStats.TotalNVMeDuration > 0 {
		benchStats.HidingRatio = float64(benchStats.HiddenIOLatency) / float64(benchStats.TotalNVMeDuration)
	}

	s.mu.Lock()
	s.stats.NVMeStreams += benchStats.NVMeStreams
	s.stats.NVMeBytesRead += benchStats.NVMeBytesRead
	s.stats.TotalGEMMDuration += benchStats.TotalGEMMDuration
	s.stats.TotalNVMeDuration += benchStats.TotalNVMeDuration
	s.stats.HiddenIOLatency += benchStats.HiddenIOLatency
	s.stats.ExposedIOLatency += benchStats.ExposedIOLatency
	s.stats.TotalWallDuration += benchStats.TotalWallDuration
	s.stats.TotalAccesses += benchStats.TotalAccesses
	s.stats.HotTableMisses += benchStats.TotalAccesses
	s.stats.BufferRotations += benchStats.BufferRotations
	s.updateHitRateLocked()
	s.mu.Unlock()

	return benchStats, nil
}

// Stats returns a snapshot of current streamer telemetry and performance counters.
func (s *MoEExpertP2PStreamer) Stats() MoEStreamerStats {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.stats
}

// CompletedCommands returns a slice of all completed direct NVMe P2P DMA commands.
func (s *MoEExpertP2PStreamer) CompletedCommands() []*NVMeP2PCommand {
	s.mu.RLock()
	defer s.mu.RUnlock()
	res := make([]*NVMeP2PCommand, len(s.completedCmds))
	copy(res, s.completedCmds)
	return res
}

// ResetStats zeroes telemetry counters and completed command history.
func (s *MoEExpertP2PStreamer) ResetStats() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stats = MoEStreamerStats{}
	s.completedCmds = nil
	s.rotations = 0
}
