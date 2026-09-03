package gateway

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// ErrDecodeCordoned is emitted when a decode step attempts to read a block whose onload is still in flight.
var ErrDecodeCordoned = errors.New("gateway: decode blocked by active KV onload cordon")

// ErrRecycleCordoned is emitted when a cancelled session attempts to recycle blocks while transfer is still in flight.
var ErrRecycleCordoned = errors.New("gateway: block recycling blocked by active in-flight onload")

// CordonState tracks the lifecycle of an onload cordon.
type CordonState string

const (
	CordonActive    CordonState = "active"
	CordonCancelled CordonState = "cancelled"
	CordonReleased  CordonState = "released"
)

// KVOnloadCordon guards device blocks during asynchronous H2D prefix restoration.
type KVOnloadCordon struct {
	mu          sync.Mutex
	RequestID   string            `json:"request_id"`
	Blocks      []int             `json:"blocks"`
	State       CordonState       `json:"state"`
	Handle      *KVTransferHandle `json:"handle"`
	CancelledAt time.Time         `json:"cancelled_at,omitempty"`
}

// KVCordonManager enforces decode cordons and prevents premature cancellation recycling.
type KVCordonManager struct {
	mu       sync.Mutex
	cordons  map[string]*KVOnloadCordon
	blockMap map[int]*KVOnloadCordon
}

// NewKVCordonManager constructs a cordon manager.
func NewKVCordonManager() *KVCordonManager {
	return &KVCordonManager{
		cordons:  make(map[string]*KVOnloadCordon),
		blockMap: make(map[int]*KVOnloadCordon),
	}
}

// CordonOnload registers an active cordon for blocks being onloaded by handle.
func (m *KVCordonManager) CordonOnload(requestID string, blocks []int, handle *KVTransferHandle) (*KVOnloadCordon, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if requestID == "" {
		return nil, fmt.Errorf("requestID must not be empty")
	}
	if len(blocks) == 0 {
		return nil, fmt.Errorf("blocks must not be empty")
	}
	if handle == nil {
		return nil, fmt.Errorf("handle must not be nil")
	}

	for _, b := range blocks {
		if existing, exists := m.blockMap[b]; exists && existing.State != CordonReleased {
			return nil, fmt.Errorf("block %d is already cordoned by request %q", b, existing.RequestID)
		}
	}

	cordon := &KVOnloadCordon{
		RequestID: requestID,
		Blocks:    append([]int(nil), blocks...),
		State:     CordonActive,
		Handle:    handle,
	}

	m.cordons[requestID] = cordon
	for _, b := range blocks {
		m.blockMap[b] = cordon
	}

	return cordon, nil
}

// CanDecode verifies that the request's onload has completed before allowing decode execution.
func (m *KVCordonManager) CanDecode(requestID string) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	cordon, ok := m.cordons[requestID]
	if !ok {
		return true, nil // not cordoned
	}

	cordon.mu.Lock()
	defer cordon.mu.Unlock()

	if cordon.State == CordonCancelled {
		return false, fmt.Errorf("request %q was cancelled", requestID)
	}

	if !cordon.Handle.Completed {
		return false, fmt.Errorf("%w for request %q (transfer %s in-flight)",
			ErrDecodeCordoned, requestID, cordon.Handle.ID)
	}

	return true, nil
}

// CancelRequest cancels the request while preserving the block quarantine until transfer finishes.
func (m *KVCordonManager) CancelRequest(requestID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	cordon, ok := m.cordons[requestID]
	if !ok {
		return fmt.Errorf("unknown cordon request %q", requestID)
	}

	cordon.mu.Lock()
	defer cordon.mu.Unlock()

	if cordon.State != CordonReleased {
		cordon.State = CordonCancelled
		cordon.CancelledAt = time.Now()
	}

	return nil
}

// CanRecycleBlock checks whether a block can be safely recycled.
// Blocks under active OR cancelled cordons whose transfer is still in-flight are strictly protected.
func (m *KVCordonManager) CanRecycleBlock(blockID int) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	cordon, ok := m.blockMap[blockID]
	if !ok {
		return true, nil
	}

	cordon.mu.Lock()
	defer cordon.mu.Unlock()

	if cordon.State == CordonReleased {
		return true, nil
	}

	if !cordon.Handle.Completed {
		return false, fmt.Errorf("%w: block %d held by in-flight onload %s (state=%s)",
			ErrRecycleCordoned, blockID, cordon.Handle.ID, cordon.State)
	}

	return true, nil
}

// ReleaseOnload releases the cordon after transfer completion, making blocks available for decode or reuse.
func (m *KVCordonManager) ReleaseOnload(requestID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	cordon, ok := m.cordons[requestID]
	if !ok {
		return fmt.Errorf("unknown cordon request %q", requestID)
	}

	cordon.mu.Lock()
	defer cordon.mu.Unlock()

	if !cordon.Handle.Completed {
		return fmt.Errorf("cannot release cordon while transfer %s is incomplete", cordon.Handle.ID)
	}

	cordon.State = CordonReleased
	for _, b := range cordon.Blocks {
		delete(m.blockMap, b)
	}
	delete(m.cordons, requestID)

	return nil
}
