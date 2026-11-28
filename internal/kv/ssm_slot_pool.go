// Copyright (c) 2026 fak contributors
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from state-spaces/mamba (mamba_ssm/modules/mamba2.py:345@e9594ce1c732):
// fixed-size SSM state-slot pool with dense per-layer conv_state + ssm_state
// buffers and decode-side state_batch_indices (negative = pad).
package kv

import (
	"errors"
	"fmt"
	"sync"
)

var (
	// ErrSSMStatePoolFull is returned when all state slots in the pool are occupied.
	ErrSSMStatePoolFull = errors.New("kv: ssm state slot pool exhausted")
	// ErrInvalidSSMShape is returned when an SSM state shape has non-positive dimensions.
	ErrInvalidSSMShape = errors.New("kv: invalid ssm state shape")
)

// SSMStateShape defines the dense per-layer conv_state and ssm_state geometry
// for a recurrent/SSM model managed alongside paged KV.
type SSMStateShape struct {
	Layers    int
	ConvDim   int
	DConv     int
	NumHeads  int
	HeadDim   int
	DState    int
	ElemBytes int // bytes per scalar element (default 4 for f32)
}

func (s SSMStateShape) elemBytes() int {
	if s.ElemBytes <= 0 {
		return 4
	}
	return s.ElemBytes
}

// Validate checks that every required tensor dimension is positive.
func (s SSMStateShape) Validate() error {
	if s.Layers <= 0 || s.ConvDim <= 0 || s.DConv <= 0 || s.NumHeads <= 0 || s.HeadDim <= 0 || s.DState <= 0 {
		return fmt.Errorf("%w: %+v", ErrInvalidSSMShape, s)
	}
	return nil
}

// ConvStateBytesPerLayer returns the byte size of one layer's conv_state (conv_dim * d_conv).
func (s SSMStateShape) ConvStateBytesPerLayer() int64 {
	return int64(s.ConvDim) * int64(s.DConv) * int64(s.elemBytes())
}

// SSMStateBytesPerLayer returns the byte size of one layer's ssm_state (nheads * headdim * dstate).
func (s SSMStateShape) SSMStateBytesPerLayer() int64 {
	return int64(s.NumHeads) * int64(s.HeadDim) * int64(s.DState) * int64(s.elemBytes())
}

// SlotBytes returns total bytes per sequence slot across all layers.
func (s SSMStateShape) SlotBytes() int64 {
	return int64(s.Layers) * (s.ConvStateBytesPerLayer() + s.SSMStateBytesPerLayer())
}

// SSMStateSlotPoolStats snapshots pool occupancy and byte footprint.
type SSMStateSlotPoolStats struct {
	MaxSlots       int
	UsedSlots      int
	FreeSlots      int
	BytesPerSlot   int64
	TotalPoolBytes int64
}

// SSMStateSlotPool manages a fixed-capacity pool of dense SSM state slots and
// maps active sequence IDs to decode-side batch slot indices.
type SSMStateSlotPool struct {
	mu       sync.Mutex
	shape    SSMStateShape
	maxSlots int
	inUse    []bool
	seqSlot  map[string]int
	convBuf  []byte
	ssmBuf   []byte
}

// NewSSMStateSlotPool constructs a fixed-size SSM state-slot pool with maxSlots slots.
func NewSSMStateSlotPool(shape SSMStateShape, maxSlots int) (*SSMStateSlotPool, error) {
	if err := shape.Validate(); err != nil {
		return nil, err
	}
	if maxSlots <= 0 {
		return nil, fmt.Errorf("%w: maxSlots=%d must be positive", ErrInvalidSSMShape, maxSlots)
	}
	convSlotBytes := int64(shape.Layers) * shape.ConvStateBytesPerLayer()
	ssmSlotBytes := int64(shape.Layers) * shape.SSMStateBytesPerLayer()
	return &SSMStateSlotPool{
		shape:    shape,
		maxSlots: maxSlots,
		inUse:    make([]bool, maxSlots),
		seqSlot:  make(map[string]int, maxSlots),
		convBuf:  make([]byte, int64(maxSlots)*convSlotBytes),
		ssmBuf:   make([]byte, int64(maxSlots)*ssmSlotBytes),
	}, nil
}

// Allocate reserves a state slot for seqID (idempotent if already allocated) and
// zeroes its backing conv/ssm buffers on fresh assignment.
func (p *SSMStateSlotPool) Allocate(seqID string) (int, error) {
	if seqID == "" {
		return -1, fmt.Errorf("kv: empty seqID")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if slot, ok := p.seqSlot[seqID]; ok {
		return slot, nil
	}
	for slot, used := range p.inUse {
		if !used {
			p.inUse[slot] = true
			p.seqSlot[seqID] = slot
			p.clearSlotLocked(slot)
			return slot, nil
		}
	}
	return -1, ErrSSMStatePoolFull
}

func (p *SSMStateSlotPool) clearSlotLocked(slot int) {
	convSlot := int(int64(p.shape.Layers) * p.shape.ConvStateBytesPerLayer())
	ssmSlot := int(int64(p.shape.Layers) * p.shape.SSMStateBytesPerLayer())
	clear(p.convBuf[slot*convSlot : (slot+1)*convSlot])
	clear(p.ssmBuf[slot*ssmSlot : (slot+1)*ssmSlot])
}

// Release frees the state slot held by seqID.
func (p *SSMStateSlotPool) Release(seqID string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	slot, ok := p.seqSlot[seqID]
	if !ok {
		return false
	}
	delete(p.seqSlot, seqID)
	p.inUse[slot] = false
	return true
}

// SlotOf returns the allocated slot index for seqID, or (-1, false) if absent.
func (p *SSMStateSlotPool) SlotOf(seqID string) (int, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	slot, ok := p.seqSlot[seqID]
	return slot, ok
}

// BatchIndices builds decode-side state_batch_indices for seqIDs: allocated
// sequences map to their non-negative slot index, while empty or unallocated
// entries map to -1 (pad).
func (p *SSMStateSlotPool) BatchIndices(seqIDs []string) []int32 {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]int32, len(seqIDs))
	for i, id := range seqIDs {
		if id == "" {
			out[i] = -1
			continue
		}
		if slot, ok := p.seqSlot[id]; ok {
			out[i] = int32(slot)
		} else {
			out[i] = -1
		}
	}
	return out
}

// Stats returns current pool occupancy and byte accounting.
func (p *SSMStateSlotPool) Stats() SSMStateSlotPoolStats {
	p.mu.Lock()
	defer p.mu.Unlock()
	used := len(p.seqSlot)
	perSlot := p.shape.SlotBytes()
	return SSMStateSlotPoolStats{
		MaxSlots:       p.maxSlots,
		UsedSlots:      used,
		FreeSlots:      p.maxSlots - used,
		BytesPerSlot:   perSlot,
		TotalPoolBytes: int64(p.maxSlots) * perSlot,
	}
}
