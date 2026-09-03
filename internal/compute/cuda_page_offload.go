package compute

import (
	"fmt"
	"sync"
)

// KVPageBundle bundles KV cache state for a contiguous block across all model layers (#10722, epic #2236).
// In transactional page-granular CPU offloading:
// 1. KV page bundles (e.g. 2MB chunks) are copied from GPU VRAM to pinned host DRAM.
// 2. The host copy is committed before releasing or unmapping GPU physical pages.
// 3. On restore, GPU physical memory is reallocated/mapped and populated from host DRAM.
type KVPageBundle struct {
	BlockID    CUDABlockID
	LayerCount int
	Tokens     int
	HostData   []byte
	Committed  bool
}

// PageOffloadManager coordinates transactional page-granular offloading between device and host.
type PageOffloadManager struct {
	mu           sync.Mutex
	bundles      map[CUDABlockID]*KVPageBundle
	offloadedCnt int64
	restoredCnt  int64
}

// NewPageOffloadManager builds a page offload manager.
func NewPageOffloadManager() *PageOffloadManager {
	return &PageOffloadManager{
		bundles: make(map[CUDABlockID]*KVPageBundle),
	}
}

// OffloadBundle commits a KV page bundle to host memory before releasing device pages (#10722).
func (m *PageOffloadManager) OffloadBundle(blockID CUDABlockID, layers, tokens int, data []byte) (*KVPageBundle, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("cannot offload empty page bundle for block %d", blockID)
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	bundle := &KVPageBundle{
		BlockID:    blockID,
		LayerCount: layers,
		Tokens:     tokens,
		HostData:   append([]byte(nil), data...),
		Committed:  true,
	}
	m.bundles[blockID] = bundle
	m.offloadedCnt++
	return bundle, nil
}

// RestoreBundle retrieves the committed host page bundle for restoring back to device memory.
func (m *PageOffloadManager) RestoreBundle(blockID CUDABlockID) (*KVPageBundle, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	b, ok := m.bundles[blockID]
	if ok && b.Committed {
		m.restoredCnt++
		return b, true
	}
	return nil, false
}

// RemoveHostBundle removes the host-side copy when the block is permanently reclaimed or freed.
func (m *PageOffloadManager) RemoveHostBundle(blockID CUDABlockID) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.bundles, blockID)
}

// OffloadedBlocks returns the count of currently offloaded page bundles.
func (m *PageOffloadManager) OffloadedBlocks() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.bundles)
}
