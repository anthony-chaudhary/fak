package model

import (
	"fmt"
	"strings"
	"sync"
)

// Qwen35RecurrentPricing reports the exact conv + recurrent state memory footprint per request unit.
type Qwen35RecurrentPricing struct {
	StateDtype             string `json:"state_dtype"`
	ElementBytes           int    `json:"element_bytes"`
	NumLinearLayers        int    `json:"num_linear_layers"`
	ConvDim                int    `json:"conv_dim"`
	ConvKernel             int    `json:"conv_kernel"`
	ConvBytesPerLayer      int64  `json:"conv_bytes_per_layer"`
	RecurrentBytesPerLayer int64  `json:"recurrent_bytes_per_layer"`
	BytesPerLayer          int64  `json:"bytes_per_layer"`
	BytesPerUnit           int64  `json:"bytes_per_unit"`
	MaxUnits               int    `json:"max_units"`
	TotalCapacityBytes     int64  `json:"total_capacity_bytes"`
}

// ComputeQwen35RecurrentPricing derives the per-unit recurrent + conv state bytes from model geometry defaulting to f32.
func ComputeQwen35RecurrentPricing(cfg Config, maxUnits int) (Qwen35RecurrentPricing, error) {
	return ComputeQwen35RecurrentPricingWithDtype(cfg, maxUnits, "f32")
}

// ComputeQwen35RecurrentPricingWithDtype derives the per-unit recurrent + conv state bytes from model geometry
// for a specified state dtype ("f32", "f16", "bf16"). Empty stateDtype defaults to "f32".
func ComputeQwen35RecurrentPricingWithDtype(cfg Config, maxUnits int, stateDtype string) (Qwen35RecurrentPricing, error) {
	if maxUnits <= 0 {
		return Qwen35RecurrentPricing{}, fmt.Errorf("maxUnits must be positive, got %d", maxUnits)
	}

	normalized := strings.ToLower(strings.TrimSpace(stateDtype))
	if normalized == "" {
		normalized = "f32"
	}

	var elemBytes int
	switch normalized {
	case "f32":
		elemBytes = 4
	case "f16", "bf16":
		elemBytes = 2
	default:
		return Qwen35RecurrentPricing{}, fmt.Errorf("unsupported state dtype %q: must be f32, f16, or bf16", stateDtype)
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
	convBytesPerLayer := int64(convElements * elemBytes)

	recurrentElements := nV * kHd * vHd
	if recurrentElements == 0 && nK > 0 {
		recurrentElements = nK * kHd * vHd
	}
	recurrentBytesPerLayer := int64(recurrentElements * elemBytes)

	bytesPerLayer := convBytesPerLayer + recurrentBytesPerLayer
	bytesPerUnit := bytesPerLayer * int64(linearLayers)
	totalCapacity := bytesPerUnit * int64(maxUnits)

	return Qwen35RecurrentPricing{
		StateDtype:             normalized,
		ElementBytes:           elemBytes,
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

// Qwen35RecurrentUnitState stores allocated conv and recurrent state buffers for one request unit.
type Qwen35RecurrentUnitState struct {
	UnitID           int      `json:"unit_id"`
	StateDtype       string   `json:"state_dtype"`
	ElementBytes     int      `json:"element_bytes"`
	LinearLayers     int      `json:"linear_layers"`
	ConvBytes        int64    `json:"conv_bytes"`
	RecurrentBytes   int64    `json:"recurrent_bytes"`
	TotalBytes       int64    `json:"total_bytes"`
	ConvBuffers      [][]byte `json:"-"`
	RecurrentBuffers [][]byte `json:"-"`
}

// Qwen35RecurrentUnitManager tracks active request units and enforces capacity before allocation.
type Qwen35RecurrentUnitManager struct {
	mu          sync.Mutex
	pricing     Qwen35RecurrentPricing
	occupied    map[int]bool
	activeCount int
}

// NewQwen35RecurrentUnitManager constructs a capacity manager for recurrent state units defaulting to f32.
func NewQwen35RecurrentUnitManager(cfg Config, maxUnits int) (*Qwen35RecurrentUnitManager, error) {
	return NewQwen35RecurrentUnitManagerWithDtype(cfg, maxUnits, "f32")
}

// NewQwen35RecurrentUnitManagerWithDtype constructs a capacity manager for recurrent state units with a given state dtype.
func NewQwen35RecurrentUnitManagerWithDtype(cfg Config, maxUnits int, stateDtype string) (*Qwen35RecurrentUnitManager, error) {
	pricing, err := ComputeQwen35RecurrentPricingWithDtype(cfg, maxUnits, stateDtype)
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

// StateDtype returns the state dtype format used by this manager.
func (m *Qwen35RecurrentUnitManager) StateDtype() string {
	return m.pricing.StateDtype
}

// ElementBytes returns the per-element byte size used by this manager.
func (m *Qwen35RecurrentUnitManager) ElementBytes() int {
	return m.pricing.ElementBytes
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

// CommittedBytes returns total bytes committed to currently active units.
func (m *Qwen35RecurrentUnitManager) CommittedBytes() int64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return int64(m.activeCount) * m.pricing.BytesPerUnit
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

// AllocateState allocates zeroed state buffers for an admitted unit according to manager pricing.
func (m *Qwen35RecurrentUnitManager) AllocateState(unitID int) (*Qwen35RecurrentUnitState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if unitID < 0 || unitID >= m.pricing.MaxUnits {
		return nil, fmt.Errorf("unit ID %d out of bounds [0, %d)", unitID, m.pricing.MaxUnits)
	}
	if !m.occupied[unitID] {
		return nil, fmt.Errorf("unit ID %d is not currently admitted", unitID)
	}

	convBuffers := make([][]byte, m.pricing.NumLinearLayers)
	recurrentBuffers := make([][]byte, m.pricing.NumLinearLayers)
	for l := 0; l < m.pricing.NumLinearLayers; l++ {
		convBuffers[l] = make([]byte, m.pricing.ConvBytesPerLayer)
		recurrentBuffers[l] = make([]byte, m.pricing.RecurrentBytesPerLayer)
	}

	return &Qwen35RecurrentUnitState{
		UnitID:           unitID,
		StateDtype:       m.pricing.StateDtype,
		ElementBytes:     m.pricing.ElementBytes,
		LinearLayers:     m.pricing.NumLinearLayers,
		ConvBytes:        m.pricing.ConvBytesPerLayer * int64(m.pricing.NumLinearLayers),
		RecurrentBytes:   m.pricing.RecurrentBytesPerLayer * int64(m.pricing.NumLinearLayers),
		TotalBytes:       m.pricing.BytesPerUnit,
		ConvBuffers:      convBuffers,
		RecurrentBuffers: recurrentBuffers,
	}, nil
}

// AdmitWithState admits a unit and immediately allocates its state buffers.
func (m *Qwen35RecurrentUnitManager) AdmitWithState() (int, *Qwen35RecurrentUnitState, error) {
	unit, err := m.Admit()
	if err != nil {
		return -1, nil, err
	}
	state, err := m.AllocateState(unit)
	if err != nil {
		_ = m.Release(unit)
		return -1, nil, err
	}
	return unit, state, nil
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
