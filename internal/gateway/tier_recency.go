package gateway

import (
	"fmt"
	"sync"
	"time"
)

// MultiTierBlockState tracks a block's presence and recency across device and external tiers.
type MultiTierBlockState struct {
	ID                 string    `json:"id"`
	DeviceResident     bool      `json:"device_resident"`
	ExternalResident   bool      `json:"external_resident"`
	DeviceHits         int       `json:"device_hits"`
	ExternalLastAccess time.Time `json:"external_last_access"`
}

// MultiTierRecencyManager ensures device-tier hits refresh external-tier (host/disk) LRU recency,
// preventing cold evictions of hot prefixes when external tiers face memory pressure.
type MultiTierRecencyManager struct {
	mu            sync.Mutex
	blocks        map[string]*MultiTierBlockState
	externalOrder []string // least recently used first
}

// NewMultiTierRecencyManager constructs a multi-tier recency manager.
func NewMultiTierRecencyManager() *MultiTierRecencyManager {
	return &MultiTierRecencyManager{
		blocks: make(map[string]*MultiTierBlockState),
	}
}

// RegisterBlock adds a block to the manager with specified tier residency.
func (m *MultiTierRecencyManager) RegisterBlock(id string, device, external bool, initialTime time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()

	b := &MultiTierBlockState{
		ID:                 id,
		DeviceResident:     device,
		ExternalResident:   external,
		ExternalLastAccess: initialTime,
	}
	m.blocks[id] = b
	if external {
		m.externalOrder = append(m.externalOrder, id)
	}
}

// RecordDeviceHit records a serving hit on the device tier and refreshes external-tier recency to now.
func (m *MultiTierRecencyManager) RecordDeviceHit(id string, now time.Time) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	b, ok := m.blocks[id]
	if !ok {
		return fmt.Errorf("unknown block %q", id)
	}
	if !b.DeviceResident {
		return fmt.Errorf("block %q is not device resident", id)
	}

	b.DeviceHits++
	b.ExternalLastAccess = now

	// Refresh position in external LRU order (move to most recently used tail)
	if b.ExternalResident {
		newOrder := make([]string, 0, len(m.externalOrder))
		for _, blk := range m.externalOrder {
			if blk != id {
				newOrder = append(newOrder, blk)
			}
		}
		newOrder = append(newOrder, id)
		m.externalOrder = newOrder
	}

	return nil
}

// ReclaimExternalUnderPressure reclaims oldest external-tier blocks until remaining count <= targetCapacity.
// Returns the list of reclaimed block IDs.
func (m *MultiTierRecencyManager) ReclaimExternalUnderPressure(targetCapacity int) []string {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.externalOrder) <= targetCapacity {
		return nil
	}

	toReclaimCount := len(m.externalOrder) - targetCapacity
	reclaimed := make([]string, toReclaimCount)
	copy(reclaimed, m.externalOrder[:toReclaimCount])

	m.externalOrder = m.externalOrder[toReclaimCount:]
	for _, id := range reclaimed {
		if b, ok := m.blocks[id]; ok {
			b.ExternalResident = false
		}
	}

	return reclaimed
}

// IsExternalResident checks whether a block remains resident in the external tier.
func (m *MultiTierRecencyManager) IsExternalResident(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	b, ok := m.blocks[id]
	return ok && b.ExternalResident
}
