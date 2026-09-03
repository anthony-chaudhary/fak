package gateway

import (
	"fmt"
	"sync"
	"time"
)

// KVTransferDirection identifies whether KV blocks are being onloaded to device or offloaded to host.
type KVTransferDirection string

const (
	KVTransferOnload  KVTransferDirection = "onload"
	KVTransferOffload KVTransferDirection = "offload"
)

// KVTransferHandle reports in-flight direction, exact touched device blocks, and completion status.
type KVTransferHandle struct {
	mu           sync.Mutex
	ID           string              `json:"id"`
	Direction    KVTransferDirection `json:"direction"`
	DeviceBlocks []int               `json:"device_blocks"`
	CreatedAt    time.Time           `json:"created_at"`
	Completed    bool                `json:"completed"`
	CompletedAt  time.Time           `json:"completed_at,omitempty"`
	Err          error               `json:"-"`
	done         chan struct{}
}

// Complete marks the transfer finished and unblocks any synchronized waiters.
func (h *KVTransferHandle) Complete(err error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.Completed {
		return
	}
	h.Completed = true
	h.CompletedAt = time.Now()
	h.Err = err
	close(h.done)
}

// Done returns a channel that closes when the transfer operation completes.
func (h *KVTransferHandle) Done() <-chan struct{} {
	return h.done
}

// KVTransferManager coordinates in-flight transfers and guards touched blocks against concurrent eviction.
type KVTransferManager struct {
	mu       sync.Mutex
	seq      int
	handles  map[string]*KVTransferHandle
	inFlight map[int]int // blockID -> count of active handles touching this block
}

// NewKVTransferManager builds an initialized transfer manager.
func NewKVTransferManager() *KVTransferManager {
	return &KVTransferManager{
		handles:  make(map[string]*KVTransferHandle),
		inFlight: make(map[int]int),
	}
}

// StartTransfer registers an asynchronous transfer and locks all touched device blocks against eviction.
func (m *KVTransferManager) StartTransfer(dir KVTransferDirection, blocks []int) (*KVTransferHandle, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if dir != KVTransferOnload && dir != KVTransferOffload {
		return nil, fmt.Errorf("invalid transfer direction: %q", dir)
	}
	if len(blocks) == 0 {
		return nil, fmt.Errorf("blocks must not be empty")
	}

	m.seq++
	handleID := fmt.Sprintf("transfer-%s-%d", dir, m.seq)
	copiedBlocks := append([]int(nil), blocks...)

	handle := &KVTransferHandle{
		ID:           handleID,
		Direction:    dir,
		DeviceBlocks: copiedBlocks,
		CreatedAt:    time.Now(),
		done:         make(chan struct{}),
	}

	m.handles[handleID] = handle
	for _, b := range copiedBlocks {
		m.inFlight[b]++
	}

	return handle, nil
}

// FinishTransfer releases the in-flight hold on all touched device blocks.
func (m *KVTransferManager) FinishTransfer(handle *KVTransferHandle, err error) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	h, ok := m.handles[handle.ID]
	if !ok {
		return fmt.Errorf("unknown transfer handle %q", handle.ID)
	}

	h.Complete(err)
	delete(m.handles, handle.ID)

	for _, b := range h.DeviceBlocks {
		m.inFlight[b]--
		if m.inFlight[b] <= 0 {
			delete(m.inFlight, b)
		}
	}

	return nil
}

// CanReclaimBlock checks whether a device block is free of in-flight transfer locks.
// If an in-flight transfer touches the block, reclamation/eviction is strictly refused.
func (m *KVTransferManager) CanReclaimBlock(blockID int) (bool, string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	count := m.inFlight[blockID]
	if count > 0 {
		return false, fmt.Sprintf("block %d is locked by %d in-flight KV transfer(s)", blockID, count)
	}
	return true, ""
}
