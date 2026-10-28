package slotstream

import (
	"fmt"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const (
	// ExpertChunkBytes is the serialized NVFP4/Q4_K expert record size (~2.76 MB).
	ExpertChunkBytes = 2894064

	// DefaultDenseTrunkBytes is the pinned dense trunk size (~3.8 GB).
	DefaultDenseTrunkBytes int64 = 3800000000

	// DefaultVRAMBudgetBytes is the 24 GB consumer GPU VRAM cap. It is the
	// FALLBACK budget for devices whose reported VRAM is unknown; a device
	// that reports its ceiling gets VRAMBudgetForDevice instead.
	DefaultVRAMBudgetBytes int64 = 24 * 1024 * 1024 * 1024

	// vramHeadroomRatio is the fraction of a device's reported VRAM that the
	// slotstream budget does NOT cover. The 24 GiB default equals the full
	// reported capacity of a nominal 24 GB consumer card (CUDA totalGlobalMem
	// = 25,769,803,776 B == 24 GiB), so the default implies zero reserved
	// headroom: a known device budgets its full reported VRAM, and a 24 GB
	// device keeps today's exact behavior.
	vramHeadroomRatio float64 = 0

	// DefaultGLM5NextSlotsPerLayer is the preallocated resident expert-slot
	// count per GLM5Next sparse MoE layer. 288 routed experts are served from
	// 54 hot slots, with cold tail experts streamed in on demand.
	DefaultGLM5NextSlotsPerLayer = 54

	// GLM5NextNumExperts is the routed-expert topology size for GLM-5.3-Flash.
	GLM5NextNumExperts = 288
)

// ExpertSlot represents a preallocated device buffer slot holding one expert's parameters.
type ExpertSlot struct {
	LayerID    int
	SlotID     int
	ExpertID   int
	Resident   bool
	Data       []byte
	LastAccess int64
}

// MultiTensorSlotPool manages preallocated VRAM slots for massive MoE models.
type MultiTensorSlotPool struct {
	mu              sync.Mutex
	SlotsPerLayer   int
	NumLayers       int
	Slots           map[int][]*ExpertSlot // layerID -> slots
	DenseTrunkBytes int64
	TotalVRAMBytes  int64
}

// VRAMBudgetForDevice derives the slotstream VRAM budget from a device's
// reported total VRAM. A non-positive deviceVRAMBytes means "unknown" and
// falls back to DefaultVRAMBudgetBytes (the 24 GiB consumer cap), preserving
// pre-lift behavior for devices that cannot report their ceiling. A known
// device budgets its full reported VRAM under vramHeadroomRatio, applied with
// the same arithmetic as the HAL fit checks (compute.BudgetAfterHeadroom).
func VRAMBudgetForDevice(deviceVRAMBytes int64) int64 {
	if deviceVRAMBytes <= 0 {
		return DefaultVRAMBudgetBytes
	}
	return compute.BudgetAfterHeadroom(deviceVRAMBytes, vramHeadroomRatio)
}

// NewMultiTensorSlotPool initializes preallocated slots for all layers within
// the VRAM budget. Devices that cannot report their VRAM get the 24 GiB
// consumer cap (DefaultVRAMBudgetBytes); pass a device's reported ceiling to
// NewMultiTensorSlotPoolForDevice instead.
func NewMultiTensorSlotPool(numLayers, slotsPerLayer int) (*MultiTensorSlotPool, error) {
	return newMultiTensorSlotPoolWithBudget(numLayers, slotsPerLayer, DefaultVRAMBudgetBytes)
}

// NewMultiTensorSlotPoolForDevice is NewMultiTensorSlotPool sized against a
// device's reported total VRAM: the budget is derived from deviceVRAMBytes
// (VRAMBudgetForDevice) rather than the 24 GiB fallback, so a larger card
// (e.g. the owned 32 GB RTX 5090) is no longer refused or mis-sized on the
// expert-slot path. A non-positive deviceVRAMBytes keeps the fallback.
func NewMultiTensorSlotPoolForDevice(numLayers, slotsPerLayer int, deviceVRAMBytes int64) (*MultiTensorSlotPool, error) {
	return newMultiTensorSlotPoolWithBudget(numLayers, slotsPerLayer, VRAMBudgetForDevice(deviceVRAMBytes))
}

func newMultiTensorSlotPoolWithBudget(numLayers, slotsPerLayer int, budget int64) (*MultiTensorSlotPool, error) {
	if numLayers <= 0 || slotsPerLayer <= 0 {
		return nil, fmt.Errorf("slotstream: invalid geometry: layers=%d, slots=%d", numLayers, slotsPerLayer)
	}

	slotBytes := int64(numLayers) * int64(slotsPerLayer) * ExpertChunkBytes
	totalVRAM := DefaultDenseTrunkBytes + slotBytes

	if totalVRAM > budget {
		return nil, fmt.Errorf("slotstream: requested VRAM %d bytes exceeds budget %d bytes", totalVRAM, budget)
	}

	pool := &MultiTensorSlotPool{
		SlotsPerLayer:   slotsPerLayer,
		NumLayers:       numLayers,
		Slots:           make(map[int][]*ExpertSlot, numLayers),
		DenseTrunkBytes: DefaultDenseTrunkBytes,
		TotalVRAMBytes:  totalVRAM,
	}

	for l := 0; l < numLayers; l++ {
		layerSlots := make([]*ExpertSlot, slotsPerLayer)
		for s := 0; s < slotsPerLayer; s++ {
			layerSlots[s] = &ExpertSlot{
				LayerID:  l,
				SlotID:   s,
				ExpertID: -1,
				Resident: false,
				Data:     make([]byte, ExpertChunkBytes),
			}
		}
		pool.Slots[l] = layerSlots
	}

	return pool, nil
}

// AcquireSlot returns an existing slot for expertID if resident, or evicts the LRU slot.
func (p *MultiTensorSlotPool) AcquireSlot(layerID, expertID int, now int64) (*ExpertSlot, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()

	slots, ok := p.Slots[layerID]
	if !ok {
		return nil, false
	}

	// 1. Check if already resident (cache hit)
	for _, s := range slots {
		if s.Resident && s.ExpertID == expertID {
			s.LastAccess = now
			return s, true
		}
	}

	// 2. Find empty slot or LRU eviction candidate
	var victim *ExpertSlot
	oldest := int64(1<<62 - 1)

	for _, s := range slots {
		if !s.Resident {
			victim = s
			break
		}
		if s.LastAccess < oldest {
			oldest = s.LastAccess
			victim = s
		}
	}

	if victim == nil {
		victim = slots[0]
	}

	victim.Resident = false
	victim.ExpertID = expertID
	victim.LastAccess = now
	return victim, false
}

// StreamEngine coordinates asynchronous streaming of expert weights into the slot pool.
type StreamEngine struct {
	Pool         *MultiTensorSlotPool
	SourceStream func(expertID int, dst []byte) error
}

// NewStreamEngine creates an expert streaming engine.
func NewStreamEngine(pool *MultiTensorSlotPool) *StreamEngine {
	return &StreamEngine{
		Pool: pool,
		SourceStream: func(expertID int, dst []byte) error {
			// Fast simulated DirectStorage / io_uring direct-I/O streaming
			for i := 0; i < len(dst); i += 4096 {
				dst[i] = byte(expertID ^ (i & 0xFF))
			}
			return nil
		},
	}
}

// StreamPicks streams the required expert picks for a layer into the preallocated slot pool.
func (e *StreamEngine) StreamPicks(layerID int, expertIDs []int, now int64) (hits, misses int, duration time.Duration, err error) {
	start := time.Now()
	for _, expID := range expertIDs {
		slot, hit := e.Pool.AcquireSlot(layerID, expID, now)
		if slot == nil {
			return hits, misses, time.Since(start), fmt.Errorf("failed to acquire slot for layer %d expert %d", layerID, expID)
		}
		if hit {
			hits++
			continue
		}
		misses++
		if err := e.SourceStream(expID, slot.Data); err != nil {
			return hits, misses, time.Since(start), err
		}
		slot.Resident = true
	}
	return hits, misses, time.Since(start), nil
}
