package slotstream

import (
	"fmt"
	"sync"
	"time"
)

const (
	// ExpertChunkBytes is the serialized NVFP4/Q4_K expert record size (~2.76 MB).
	ExpertChunkBytes = 2894064

	// DefaultDenseTrunkBytes is the pinned dense trunk size (~3.8 GB).
	DefaultDenseTrunkBytes int64 = 3800000000

	// DefaultVRAMBudgetBytes is the 24 GB consumer GPU VRAM cap.
	DefaultVRAMBudgetBytes int64 = 24 * 1024 * 1024 * 1024
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
	mu             sync.Mutex
	SlotsPerLayer  int
	NumLayers      int
	Slots          map[int][]*ExpertSlot // layerID -> slots
	DenseTrunkBytes int64
	TotalVRAMBytes int64
}

// NewMultiTensorSlotPool initializes preallocated slots for all layers within the VRAM budget.
func NewMultiTensorSlotPool(numLayers, slotsPerLayer int) (*MultiTensorSlotPool, error) {
	if numLayers <= 0 || slotsPerLayer <= 0 {
		return nil, fmt.Errorf("slotstream: invalid geometry: layers=%d, slots=%d", numLayers, slotsPerLayer)
	}

	slotBytes := int64(numLayers) * int64(slotsPerLayer) * ExpertChunkBytes
	totalVRAM := DefaultDenseTrunkBytes + slotBytes

	if totalVRAM > DefaultVRAMBudgetBytes {
		return nil, fmt.Errorf("slotstream: requested VRAM %d bytes exceeds budget %d bytes", totalVRAM, DefaultVRAMBudgetBytes)
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
