package model

import (
	"fmt"
	"sync"
)

// Qwen35RecurrentPricing reports the exact conv + recurrent state memory footprint per request unit.
type Qwen35RecurrentPricing struct {
	NumLinearLayers        int   `json:"num_linear_layers"`
	ConvDim                int   `json:"conv_dim"`
	ConvKernel             int   `json:"conv_kernel"`
	ConvBytesPerLayer      int64 `json:"conv_bytes_per_layer"`
	RecurrentBytesPerLayer int64 `json:"recurrent_bytes_per_layer"`
	BytesPerLayer          int64 `json:"bytes_per_layer"`
	BytesPerUnit           int64 `json:"bytes_per_unit"`
	MaxUnits               int   `json:"max_units"`
	TotalCapacityBytes     int64 `json:"total_capacity_bytes"`
}

// ComputeQwen35RecurrentPricing derives the per-unit recurrent + conv state bytes from model geometry.
func ComputeQwen35RecurrentPricing(cfg Config, maxUnits int) (Qwen35RecurrentPricing, error) {
	if maxUnits <= 0 {
		return Qwen35RecurrentPricing{}, fmt.Errorf("maxUnits must be positive, got %d", maxUnits)
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
	convBytesPerLayer := int64(convElements * 4)

	recurrentElements := nV * kHd * vHd
	if recurrentElements == 0 && nK > 0 {
		recurrentElements = nK * kHd * vHd
	}
	recurrentBytesPerLayer := int64(recurrentElements * 4)

	bytesPerLayer := convBytesPerLayer + recurrentBytesPerLayer
	bytesPerUnit := bytesPerLayer * int64(linearLayers)
	totalCapacity := bytesPerUnit * int64(maxUnits)

	return Qwen35RecurrentPricing{
		NumLinearLayers:        linearLayers,
		ConvDim:                convDim,
		ConvKernel:             kernel,
		ConvBytesPerLayer:      convBytesPerLayer,
		RecurrentBytesPerLayer: recurrentBytesPerLayer,
		BytesPerLayer:          bytesPerLayer,
		BytesPerUnit:           bytesPerUnit,
		MaxUnits:               maxUnits,
		TotalCapacityBytes:     totalCapacity,
	}, nil
}

// Qwen35RecurrentUnitManager tracks active request units and enforces capacity before allocation.
type Qwen35RecurrentUnitManager struct {
	mu          sync.Mutex
	pricing     Qwen35RecurrentPricing
	occupied    map[int]bool
	activeCount int
}

// NewQwen35RecurrentUnitManager constructs a capacity manager for recurrent state units.
func NewQwen35RecurrentUnitManager(cfg Config, maxUnits int) (*Qwen35RecurrentUnitManager, error) {
	pricing, err := ComputeQwen35RecurrentPricing(cfg, maxUnits)
	if err != nil {
		return nil, err
	}
	return &Qwen35RecurrentUnitManager{
		pricing:  pricing,
		occupied: make(map[int]bool, maxUnits),
	}, nil
}

// Pricing returns the immutable pricing structure for this manager.
func (m *Qwen35RecurrentUnitManager) Pricing() Qwen35RecurrentPricing {
	return m.pricing
}

// ActiveUnits returns the current number of admitted units.
func (m *Qwen35RecurrentUnitManager) ActiveUnits() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.activeCount
}

// FreeUnits returns remaining unadmitted units.
func (m *Qwen35RecurrentUnitManager) FreeUnits() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.pricing.MaxUnits - m.activeCount
}

// Admit reserves a recurrent state unit if under capacity, or returns an explicit refusal.
func (m *Qwen35RecurrentUnitManager) Admit() (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.activeCount >= m.pricing.MaxUnits {
		return -1, fmt.Errorf("recurrent capacity exhausted: %d/%d active units, %d bytes committed (cannot admit %d bytes for unit %d)",
			m.activeCount, m.pricing.MaxUnits, m.pricing.TotalCapacityBytes, m.pricing.BytesPerUnit, m.activeCount)
	}

	for unit := 0; unit < m.pricing.MaxUnits; unit++ {
		if !m.occupied[unit] {
			m.occupied[unit] = true
			m.activeCount++
			return unit, nil
		}
	}

	return -1, fmt.Errorf("no free units available")
}

// Release returns a previously admitted unit to the pool.
func (m *Qwen35RecurrentUnitManager) Release(unitID int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if unitID < 0 || unitID >= m.pricing.MaxUnits {
		return fmt.Errorf("unit ID %d out of bounds [0, %d)", unitID, m.pricing.MaxUnits)
	}
	if !m.occupied[unitID] {
		return fmt.Errorf("unit ID %d is not currently admitted", unitID)
	}

	delete(m.occupied, unitID)
	m.activeCount--
	return nil
}
