package model

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"
)

// ErrRollbackExceedsJournal is returned when a rollback request exceeds the
// available recorded depth of the rolling reversion journal.
var ErrRollbackExceedsJournal = errors.New("model: rollback tokens exceeds journal depth")

const (
	// DefaultMaxRollback is the standard 16-token speculative decode reversion depth.
	DefaultMaxRollback = 16

	// DefaultCacheCapacity is the standard contiguous pool token capacity (1024 tokens),
	// matching the local sliding window for DeepSeek Sparse Attention (DSA) and MLA.
	DefaultCacheCapacity = 1024
)

// journalEntry tracks the prior state of a circular buffer slot for reversion.
type journalEntry struct {
	pos     int
	slot    int
	hadPrev bool
	prevPos int
}

// CompactLatentCache is a compact circular latent cache for validated NoPE
// (no-position-embedding) keys and values in DSA / MLA architectures.
// It uses a contiguous flat pool buffer to prevent VRAM memory fragmentation
// and allocation spikes during long context generation, paired with a rolling
// 16-token reversion journal that supports speculative decode rollback without
// retaining uncommitted activation histories across the full sequence.
type CompactLatentCache struct {
	mu          sync.RWMutex
	dim         int
	capacity    int
	maxRollback int

	// Contiguous pool buffer for validated NoPE latents: [capacity * dim]float32.
	pool      []float32
	positions []int
	occupied  []bool

	// Rolling reversion journal for speculative decode rollback up to maxRollback tokens.
	journalPool  []float32      // preallocated backup vectors: [maxRollback * dim]float32
	journal      []journalEntry // ring buffer of rolling journal entries
	journalHead  int            // next insertion index in journal ring buffer
	journalCount int            // active rollback tokens available (0 <= journalCount <= maxRollback)
}

// NewCompactLatentCache creates a new compact circular latent cache.
// dim is the latent vector dimension (e.g. 512 for DeepSeek MLA kv_lora_rank).
// maxRollback is the maximum rollback depth (defaults to 16 if <= 0).
// An optional capacity can be provided to override DefaultCacheCapacity (1024).
func NewCompactLatentCache(dim, maxRollback int, capacity ...int) *CompactLatentCache {
	if dim <= 0 {
		panic(fmt.Sprintf("model: CompactLatentCache non-positive dim %d", dim))
	}
	if maxRollback <= 0 {
		maxRollback = DefaultMaxRollback
	}
	cap := DefaultCacheCapacity
	if len(capacity) > 0 && capacity[0] > 0 {
		cap = capacity[0]
	}
	if cap < maxRollback {
		cap = maxRollback
	}

	positions := make([]int, cap)
	for i := range positions {
		positions[i] = -1
	}

	return &CompactLatentCache{
		dim:         dim,
		capacity:    cap,
		maxRollback: maxRollback,
		pool:        make([]float32, cap*dim),
		positions:   positions,
		occupied:    make([]bool, cap),
		journalPool: make([]float32, maxRollback*dim),
		journal:     make([]journalEntry, maxRollback),
	}
}

// Dim returns the latent vector dimension.
func (c *CompactLatentCache) Dim() int {
	return c.dim
}

// Capacity returns the circular pool token capacity.
func (c *CompactLatentCache) Capacity() int {
	return c.capacity
}

// MaxRollback returns the configured maximum rollback depth.
func (c *CompactLatentCache) MaxRollback() int {
	return c.maxRollback
}

// JournalDepth returns the current number of reversible rollback tokens recorded.
func (c *CompactLatentCache) JournalDepth() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.journalCount
}

// Append stores a latent vector for sequence position pos into the circular pool buffer,
// recording the prior slot state in the rolling reversion journal.
func (c *CompactLatentCache) Append(pos int, vec []float32) {
	if len(vec) != c.dim {
		panic(fmt.Sprintf("model: CompactLatentCache vector dimension %d, want %d", len(vec), c.dim))
	}
	if pos < 0 {
		panic(fmt.Sprintf("model: CompactLatentCache negative position %d", pos))
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	slot := pos % c.capacity

	// Record prior slot state in the rolling reversion journal.
	entryIdx := c.journalHead
	entry := &c.journal[entryIdx]
	entry.pos = pos
	entry.slot = slot
	entry.hadPrev = c.occupied[slot]
	if entry.hadPrev {
		entry.prevPos = c.positions[slot]
		copy(c.journalPool[entryIdx*c.dim:(entryIdx+1)*c.dim], c.pool[slot*c.dim:(slot+1)*c.dim])
	} else {
		entry.prevPos = -1
	}

	c.journalHead = (c.journalHead + 1) % c.maxRollback
	if c.journalCount < c.maxRollback {
		c.journalCount++
	}

	// Copy new latent vector into contiguous pool buffer.
	copy(c.pool[slot*c.dim:(slot+1)*c.dim], vec)
	c.positions[slot] = pos
	c.occupied[slot] = true
}

// Rollback restores state by undoing the most recent tokens appended within the
// journal depth. If tokens exceeds the available journal depth, the operation is
// refused with an error and the cache state remains unchanged.
func (c *CompactLatentCache) Rollback(tokens int) error {
	if tokens < 0 {
		return fmt.Errorf("model: negative rollback tokens %d", tokens)
	}
	if tokens == 0 {
		return nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if tokens > c.journalCount {
		return fmt.Errorf("%w: requested %d tokens, journal depth is %d", ErrRollbackExceedsJournal, tokens, c.journalCount)
	}

	for i := 0; i < tokens; i++ {
		c.journalHead = (c.journalHead - 1 + c.maxRollback) % c.maxRollback
		entry := c.journal[c.journalHead]
		slot := entry.slot

		if entry.hadPrev {
			c.positions[slot] = entry.prevPos
			c.occupied[slot] = true
			copy(c.pool[slot*c.dim:(slot+1)*c.dim], c.journalPool[c.journalHead*c.dim:(c.journalHead+1)*c.dim])
		} else {
			c.positions[slot] = -1
			c.occupied[slot] = false
		}
		c.journalCount--
	}

	return nil
}

// Get retrieves the latent vector at absolute sequence position pos.
// If the position has been overwritten by the circular buffer or never stored,
// it returns (nil, false).
func (c *CompactLatentCache) Get(pos int) ([]float32, bool) {
	if pos < 0 {
		return nil, false
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	slot := pos % c.capacity
	if !c.occupied[slot] || c.positions[slot] != pos {
		return nil, false
	}

	out := make([]float32, c.dim)
	copy(out, c.pool[slot*c.dim:(slot+1)*c.dim])
	return out, true
}

// MemoryBytes returns the total memory allocated by the compact latent cache in bytes.
func (c *CompactLatentCache) MemoryBytes() int64 {
	c.mu.RLock()
	defer c.mu.RUnlock()

	poolBytes := int64(len(c.pool)) * 4
	posBytes := int64(len(c.positions)) * 8
	occBytes := int64(len(c.occupied))
	journalPoolBytes := int64(len(c.journalPool)) * 4
	journalMetaBytes := int64(len(c.journal)) * int64(unsafe.Sizeof(journalEntry{}))
	structBytes := int64(unsafe.Sizeof(*c))

	return poolBytes + posBytes + occBytes + journalPoolBytes + journalMetaBytes + structBytes
}
