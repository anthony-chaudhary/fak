package model

import (
	"fmt"
	"sync"
)

// Qwen35StateBankReceipt records preallocation and runtime reuse metrics.
type Qwen35StateBankReceipt struct {
	WarmupAllocations  int   `json:"warmup_allocations"`
	RuntimeAllocations int   `json:"runtime_allocations"`
	MaxUnits           int   `json:"max_units"`
	NumLinearLayers    int   `json:"num_linear_layers"`
	TotalStateBytes    int64 `json:"total_state_bytes"`
}

// Qwen35LayerStateUnit stores preallocated conv and recurrent state buffers for one unit at one layer.
type Qwen35LayerStateUnit struct {
	UnitID          int       `json:"unit_id"`
	Layer           int       `json:"layer"`
	ConvBuffer      []float32 `json:"conv_buffer"`
	RecurrentBuffer []float32 `json:"recurrent_buffer"`
}

// Qwen35PreallocatedStateBank preallocates fixed-address state buffers across layers at warmup,
// completely eliminating per-session state allocations on the hot path.
type Qwen35PreallocatedStateBank struct {
	mu                 sync.Mutex
	cfg                Config
	maxUnits           int
	numLinearLayers    int
	units              [][]Qwen35LayerStateUnit // [unitID][layerIdx]
	occupied           map[int]bool
	warmupAllocations  int
	runtimeAllocations int
	totalBytes         int64
}

// NewQwen35PreallocatedStateBank preallocates all layer conv and recurrent state buffers for maxUnits.
func NewQwen35PreallocatedStateBank(cfg Config, maxUnits int) (*Qwen35PreallocatedStateBank, error) {
	if maxUnits <= 0 {
		return nil, fmt.Errorf("maxUnits must be positive, got %d", maxUnits)
	}

	linearLayers := 0
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			linearLayers++
		}
	}
	if linearLayers == 0 {
		if cfg.IsQwen35Hybrid() {
			linearLayers = cfg.NumLayers
		} else {
			linearLayers = 1
		}
	}

	nK, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	kernel := cfg.LinearConvKernelDim
	if kernel <= 1 {
		kernel = 4
	}

	convElements := (kernel - 1) * convDim
	recurrentElements := nV * kHd * vHd
	if recurrentElements == 0 && nK > 0 {
		recurrentElements = nK * kHd * vHd
	}

	allocCount := 0
	var totalBytes int64

	units := make([][]Qwen35LayerStateUnit, maxUnits)
	for u := 0; u < maxUnits; u++ {
		units[u] = make([]Qwen35LayerStateUnit, linearLayers)
		for l := 0; l < linearLayers; l++ {
			cBuf := make([]float32, convElements)
			rBuf := make([]float32, recurrentElements)
			allocCount += 2
			totalBytes += int64((convElements + recurrentElements) * 4)

			units[u][l] = Qwen35LayerStateUnit{
				UnitID:          u,
				Layer:           l,
				ConvBuffer:      cBuf,
				RecurrentBuffer: rBuf,
			}
		}
	}

	return &Qwen35PreallocatedStateBank{
		cfg:                cfg,
		maxUnits:           maxUnits,
		numLinearLayers:    linearLayers,
		units:              units,
		occupied:           make(map[int]bool, maxUnits),
		warmupAllocations:  allocCount,
		runtimeAllocations: 0,
		totalBytes:         totalBytes,
	}, nil
}

// Receipt returns the deterministic metrics.
func (p *Qwen35PreallocatedStateBank) Receipt() Qwen35StateBankReceipt {
	p.mu.Lock()
	defer p.mu.Unlock()
	return Qwen35StateBankReceipt{
		WarmupAllocations:  p.warmupAllocations,
		RuntimeAllocations: p.runtimeAllocations,
		MaxUnits:           p.maxUnits,
		NumLinearLayers:    p.numLinearLayers,
		TotalStateBytes:    p.totalBytes,
	}
}

// Acquire reserves an active unit and returns its preallocated layer buffers without allocating.
func (p *Qwen35PreallocatedStateBank) Acquire() (int, []Qwen35LayerStateUnit, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	for u := 0; u < p.maxUnits; u++ {
		if !p.occupied[u] {
			p.occupied[u] = true
			return u, p.units[u], nil
		}
	}

	return -1, nil, fmt.Errorf("state bank capacity exhausted: all %d units occupied", p.maxUnits)
}

// Release zeroes and returns the preallocated layer buffers for unitID.
func (p *Qwen35PreallocatedStateBank) Release(unitID int) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if unitID < 0 || unitID >= p.maxUnits {
		return fmt.Errorf("unit ID %d out of range [0, %d)", unitID, p.maxUnits)
	}
	if !p.occupied[unitID] {
		return fmt.Errorf("unit ID %d is not occupied", unitID)
	}

	for l := range p.units[unitID] {
		for i := range p.units[unitID][l].ConvBuffer {
			p.units[unitID][l].ConvBuffer[i] = 0
		}
		for i := range p.units[unitID][l].RecurrentBuffer {
			p.units[unitID][l].RecurrentBuffer[i] = 0
		}
	}

	delete(p.occupied, unitID)
	return nil
}
