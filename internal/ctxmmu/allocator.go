package ctxmmu

import (
	"errors"
	"math/bits"
	"os"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// Default constants for GTT attention page allocations on Strix Halo APU architectures.
// By default, 524,288 tokens (512K context) in FP8 GQA require 64 KiB per token (32 GiB KV cache),
// mapped into a 35.0 GiB contiguous slab of GTT memory with headroom for page tables and guard pages.
const (
	DefaultMaxTokens         = 524288
	DefaultPoolBytesPerToken = int64(65536)                   // 64 KiB per token for FP8 GQA
	DefaultTotalGTTBytes     = int64(35) * 1024 * 1024 * 1024 // 35.0 GiB contiguous slab
	EnvHalogenKVPoolFit      = "HALOGEN_KV_POOL_FIT"          // Env var for dynamic downscale ratio
)

// Standard typed errors for token pool operations.
var (
	// ErrPoolExhausted is returned when a reservation cannot be satisfied because either
	// the effective token capacity or the total GTT byte budget would be exceeded.
	ErrPoolExhausted = errors.New("ctxmmu: token pool exhausted")

	// ErrInvalidTokens is returned when a requested token count is zero or negative.
	ErrInvalidTokens = errors.New("ctxmmu: token count must be greater than zero")

	// ErrEmptyStreamID is returned when an empty stream or session ID is provided.
	ErrEmptyStreamID = errors.New("ctxmmu: stream ID cannot be empty")

	// ErrExceedsReservation is returned when attempting to commit more tokens than currently reserved.
	ErrExceedsReservation = errors.New("ctxmmu: committed tokens exceed reserved headroom")

	// ErrStreamNotFound is returned when attempting to commit or release headroom for an unknown stream.
	ErrStreamNotFound = errors.New("ctxmmu: stream reservation not found")
)

// SharedTokenPool represents a contiguous slab of GTT attention pages for public core runtime
// context-MMU token management. It arbitrates multi-session KV-cache allocations with dynamic
// capacity admission, two-phase reservation/commit cycles, and immediate headroom reclamation.
type SharedTokenPool struct {
	mu              sync.RWMutex
	maxTokens       int            // Default 524,288 tokens
	committedTokens int            // Sum of tokens committed across all active streams
	reservedTokens  int            // Sum of uncommitted output headroom reserved across active streams
	bytesPerToken   int64          // Default 65,536 bytes/token (FP8 GQA)
	totalGTTBytes   int64          // Default 35 GiB aperture
	downscaleFactor float64        // Default 1.0, or derived from HALOGEN_KV_POOL_FIT
	reservations    map[string]int // session/stream ID -> reserved uncommitted output headroom
	committed       map[string]int // session/stream ID -> committed tokens
}

// NewSharedTokenPool initializes a SharedTokenPool with default Strix Halo GTT parameters:
// 524,288 tokens, 35.0 GiB total GTT aperture, and 65,536 bytes/tok.
// If the HALOGEN_KV_POOL_FIT environment variable is set to a valid positive float,
// downscaleFactor is initialized from it (defaulting to 1.0 otherwise).
func NewSharedTokenPool() *SharedTokenPool {
	factor := 1.0
	if val := os.Getenv(EnvHalogenKVPoolFit); val != "" {
		if f, err := strconv.ParseFloat(val, 64); err == nil && f > 0 {
			factor = f
		}
	}
	return &SharedTokenPool{
		maxTokens:       DefaultMaxTokens,
		bytesPerToken:   DefaultPoolBytesPerToken,
		totalGTTBytes:   DefaultTotalGTTBytes,
		downscaleFactor: factor,
		reservations:    make(map[string]int),
		committed:       make(map[string]int),
	}
}

// NewSharedTokenPoolWithConfig creates a SharedTokenPool with explicit capacity parameters.
// If downscaleFactor <= 0, it falls back to 1.0 (or HALOGEN_KV_POOL_FIT if set).
func NewSharedTokenPoolWithConfig(maxTokens int, totalGTTBytes int64, bytesPerToken int64, downscaleFactor float64) *SharedTokenPool {
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTokens
	}
	if totalGTTBytes <= 0 {
		totalGTTBytes = DefaultTotalGTTBytes
	}
	if bytesPerToken <= 0 {
		bytesPerToken = DefaultPoolBytesPerToken
	}
	if downscaleFactor <= 0 {
		downscaleFactor = 1.0
		if val := os.Getenv(EnvHalogenKVPoolFit); val != "" {
			if f, err := strconv.ParseFloat(val, 64); err == nil && f > 0 {
				downscaleFactor = f
			}
		}
	}
	return &SharedTokenPool{
		maxTokens:       maxTokens,
		bytesPerToken:   bytesPerToken,
		totalGTTBytes:   totalGTTBytes,
		downscaleFactor: downscaleFactor,
		reservations:    make(map[string]int),
		committed:       make(map[string]int),
	}
}

// effectiveMaxLocked calculates the dynamic effective token capacity of the pool,
// scaling maxTokens by downscaleFactor. Must be called while holding mu.
func (p *SharedTokenPool) effectiveMaxLocked() int {
	eff := int(float64(p.maxTokens) * p.downscaleFactor)
	if eff < 0 {
		return 0
	}
	return eff
}

// Reserve performs dynamic capacity admission and reserves requested uncommitted output headroom
// for streamID. Capacity admission enforces two invariants:
// 1. committed_tokens + reserved_tokens + requested_tokens <= effectiveMax
// 2. (committed + reserved + requested) * bytesPerToken <= totalGTTBytes
// Returns ErrPoolExhausted if capacity is exceeded.
func (p *SharedTokenPool) Reserve(streamID string, requested int) error {
	if streamID == "" {
		return ErrEmptyStreamID
	}
	if requested <= 0 {
		return ErrInvalidTokens
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	effectiveMax := p.effectiveMaxLocked()
	projectedTokens := p.committedTokens + p.reservedTokens + requested

	// Check 1: Token limit against effectiveMax
	if projectedTokens > effectiveMax {
		return ErrPoolExhausted
	}

	// Check 2: Byte limit against total GTT capacity
	if p.bytesPerToken > 0 && int64(projectedTokens)*p.bytesPerToken > p.totalGTTBytes {
		return ErrPoolExhausted
	}

	p.reservedTokens += requested
	p.reservations[streamID] += requested
	return nil
}

// Commit shifts tokens from reserved output headroom to committed tokens for streamID.
// This is called as tokens are generated or upon turn completion.
// Returns ErrStreamNotFound if streamID has no reservation, or ErrExceedsReservation
// if tokens exceeds the reserved headroom.
func (p *SharedTokenPool) Commit(streamID string, tokens int) error {
	if streamID == "" {
		return ErrEmptyStreamID
	}
	if tokens <= 0 {
		return ErrInvalidTokens
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	reserved, ok := p.reservations[streamID]
	if !ok || reserved == 0 {
		return ErrStreamNotFound
	}
	if tokens > reserved {
		return ErrExceedsReservation
	}

	// Shift from reserved to committed
	p.reservations[streamID] -= tokens
	if p.reservations[streamID] == 0 {
		delete(p.reservations, streamID)
	}
	p.reservedTokens -= tokens

	p.committed[streamID] += tokens
	p.committedTokens += tokens
	return nil
}

// Release releases both committed and reserved tokens for streamID, returning all
// associated GTT pages back to the shared pool.
func (p *SharedTokenPool) Release(streamID string) {
	if streamID == "" {
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if res, ok := p.reservations[streamID]; ok {
		p.reservedTokens -= res
		delete(p.reservations, streamID)
	}
	if com, ok := p.committed[streamID]; ok {
		p.committedTokens -= com
		delete(p.committed, streamID)
	}
}

// ReleaseHeadroom immediately returns unused reserved output headroom for streamID
// upon terminal stop or finish reason, freeing the headroom for other streams while
// preserving the stream's committed tokens.
func (p *SharedTokenPool) ReleaseHeadroom(streamID string) {
	if streamID == "" {
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if res, ok := p.reservations[streamID]; ok {
		p.reservedTokens -= res
		delete(p.reservations, streamID)
	}
}

// FitPoolToAvailableMemory dynamically adjusts the pool to available GTT memory,
// clamping maxTokens to what availableGTTBytes can support per the HALOGEN_KV_POOL_FIT pattern.
// If HALOGEN_KV_POOL_FIT is set in the environment, downscaleFactor is refreshed.
// It returns the new effective maximum token capacity.
func (p *SharedTokenPool) FitPoolToAvailableMemory(availableGTTBytes int64) int {
	p.mu.Lock()
	defer p.mu.Unlock()

	if val := os.Getenv(EnvHalogenKVPoolFit); val != "" {
		if f, err := strconv.ParseFloat(val, 64); err == nil && f > 0 {
			p.downscaleFactor = f
		}
	}

	p.totalGTTBytes = availableGTTBytes
	if p.bytesPerToken > 0 {
		memTokens := int(availableGTTBytes / p.bytesPerToken)
		if memTokens < 0 {
			memTokens = 0
		}
		if memTokens < p.maxTokens {
			p.maxTokens = memTokens
		}
	}
	return p.effectiveMaxLocked()
}

// MaxTokens returns the configured maximum token capacity of the pool.
func (p *SharedTokenPool) MaxTokens() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.maxTokens
}

// CommittedTokens returns the total number of tokens currently committed across all streams.
func (p *SharedTokenPool) CommittedTokens() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.committedTokens
}

// ReservedTokens returns the total number of tokens currently reserved as headroom across all streams.
func (p *SharedTokenPool) ReservedTokens() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.reservedTokens
}

// FreeTokens returns the remaining unallocated token capacity that can be admitted
// before either the effectiveMax or totalGTTBytes limit is reached.
func (p *SharedTokenPool) FreeTokens() int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	effectiveMax := p.effectiveMaxLocked()
	used := p.committedTokens + p.reservedTokens
	freeTokens := effectiveMax - used

	if p.bytesPerToken > 0 {
		byteFree := int(p.totalGTTBytes/p.bytesPerToken) - used
		if byteFree < freeTokens {
			freeTokens = byteFree
		}
	}

	if freeTokens < 0 {
		return 0
	}
	return freeTokens
}

// AllocatedBytes returns the total GTT bytes currently tied up by committed and reserved tokens.
func (p *SharedTokenPool) AllocatedBytes() int64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return int64(p.committedTokens+p.reservedTokens) * p.bytesPerToken
}

// ActiveStreams returns the number of unique streams that have active reservations or committed tokens.
func (p *SharedTokenPool) ActiveStreams() int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	active := make(map[string]struct{})
	for k, v := range p.reservations {
		if v > 0 {
			active[k] = struct{}{}
		}
	}
	for k, v := range p.committed {
		if v > 0 {
			active[k] = struct{}{}
		}
	}
	return len(active)
}

// EffectiveMaxTokens returns the dynamic admission ceiling after applying downscaleFactor.
func (p *SharedTokenPool) EffectiveMaxTokens() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.effectiveMaxLocked()
}

// DownscaleFactor returns the current downscale factor.
func (p *SharedTokenPool) DownscaleFactor() float64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.downscaleFactor
}

// TotalGTTBytes returns the configured GTT memory slab size in bytes.
func (p *SharedTokenPool) TotalGTTBytes() int64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.totalGTTBytes
}

// BytesPerToken returns the memory cost per token in bytes.
func (p *SharedTokenPool) BytesPerToken() int64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.bytesPerToken
}

// StreamUsage returns the committed and reserved token counts for a specific stream.
func (p *SharedTokenPool) StreamUsage(streamID string) (committed int, reserved int) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.committed[streamID], p.reservations[streamID]
}

// -----------------------------------------------------------------------------
// Lock-Free Physical Block Allocator (Issue #12191)
// -----------------------------------------------------------------------------

var (
	// ErrNoFreeBlocks is returned when no physical blocks are available for allocation.
	ErrNoFreeBlocks = errors.New("ctxmmu: no free physical blocks available")

	// ErrInvalidBlockID is returned when an invalid physical block ID is requested.
	ErrInvalidBlockID = errors.New("ctxmmu: invalid physical block ID")

	// ErrBlockNotFound is returned when a requested physical block is not currently allocated.
	ErrBlockNotFound = errors.New("ctxmmu: physical block not found")

	// ErrBlockIDExhausted is returned before allocation state changes when no new
	// allocation-incarnation ID can be represented by int.
	ErrBlockIDExhausted = errors.New("ctxmmu: physical block allocation ID exhausted")
)

// PhysicalBlock represents an atomic physical memory page in unified memory for KV caches.
type PhysicalBlock struct {
	// ID identifies this allocation incarnation. Physical slots may be reused,
	// but an ID is never reused by an allocator.
	ID int
	// Data belongs only to this allocation incarnation. A replacement allocation
	// receives fresh backing so a retired block cannot overwrite it.
	Data []byte

	allocationID int
	slot         int
	refCount     atomic.Int32
	lastAccess   atomic.Int64
	swapped      atomic.Bool
}

const maxPhysicalBlockRefCount = int32(^uint32(0) >> 1)

// PhysicalSlot returns the bounded physical slot used by device page tables.
// Unlike ID, the slot may be reused after this allocation is retired.
func (b *PhysicalBlock) PhysicalSlot() int {
	if b == nil {
		return -1
	}
	return b.slot
}

// Retain increments a live block's reference count and updates last access.
// Retired blocks stay at zero and cannot be resurrected; overflow saturates.
func (b *PhysicalBlock) Retain() int32 {
	if b == nil {
		return 0
	}
	for {
		current := b.refCount.Load()
		if current <= 0 {
			return 0
		}
		if current == maxPhysicalBlockRefCount {
			return current
		}
		if b.refCount.CompareAndSwap(current, current+1) {
			b.Touch()
			return current + 1
		}
	}
}

// Release decrements a live block's reference count and updates last access.
// Releasing a retired or already-unreferenced block is idempotent at zero.
func (b *PhysicalBlock) Release() int32 {
	if b == nil {
		return 0
	}
	for {
		current := b.refCount.Load()
		if current <= 0 {
			return 0
		}
		if b.refCount.CompareAndSwap(current, current-1) {
			b.Touch()
			return current - 1
		}
	}
}

// RefCount returns the current atomic reference count.
func (b *PhysicalBlock) RefCount() int32 {
	return b.refCount.Load()
}

// Touch updates the last access timestamp in nanoseconds.
func (b *PhysicalBlock) Touch() {
	b.lastAccess.Store(time.Now().UnixNano())
}

// LastAccess returns the last access timestamp in nanoseconds.
func (b *PhysicalBlock) LastAccess() int64 {
	return b.lastAccess.Load()
}

// Swapped returns whether the block is currently swapped out.
func (b *PhysicalBlock) Swapped() bool {
	return b.swapped.Load()
}

// SetSwapped updates the swapped state of the block.
func (b *PhysicalBlock) SetSwapped(val bool) {
	b.swapped.Store(val)
}

// PhysicalBlockAllocator manages reusable physical slots. Allocation IDs are
// opaque, monotonic incarnation handles; blocksByID is the authoritative live
// allocation registry and blocks indexes the current occupant of each slot.
type PhysicalBlockAllocator struct {
	mu             sync.RWMutex
	totalBlocks    int
	blockSizeBytes int
	bitmask        []atomic.Uint64
	blocks         []*PhysicalBlock
	blocksByID     map[int]*PhysicalBlock
	nextID         uint64
	allocatedCount atomic.Int64
	allocHint      atomic.Uint64
}

// NewPhysicalBlockAllocator creates a PhysicalBlockAllocator with reusable
// physical slots backed by an atomic bitmask.
func NewPhysicalBlockAllocator(totalBlocks int, blockSizeBytes int) *PhysicalBlockAllocator {
	if totalBlocks <= 0 {
		totalBlocks = 1024
	}
	if blockSizeBytes < 0 {
		blockSizeBytes = 0
	}
	numWords := (totalBlocks + 63) / 64
	bitmask := make([]atomic.Uint64, numWords)
	if rem := totalBlocks % 64; rem != 0 {
		var unusedMask uint64 = ^((uint64(1) << rem) - 1)
		bitmask[numWords-1].Store(unusedMask)
	}

	return &PhysicalBlockAllocator{
		totalBlocks:    totalBlocks,
		blockSizeBytes: blockSizeBytes,
		bitmask:        bitmask,
		blocks:         make([]*PhysicalBlock, totalBlocks),
		blocksByID:     make(map[int]*PhysicalBlock, totalBlocks),
	}
}

func maxPhysicalBlockID() uint64 {
	return uint64(^uint(0) >> 1)
}

// Allocate claims a free physical slot and gives it a unique allocation ID.
func (a *PhysicalBlockAllocator) Allocate() (*PhysicalBlock, error) {
	if a == nil || a.totalBlocks == 0 {
		return nil, ErrNoFreeBlocks
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	numWords := len(a.bitmask)
	if numWords == 0 {
		return nil, ErrNoFreeBlocks
	}
	if a.nextID > maxPhysicalBlockID() {
		return nil, ErrBlockIDExhausted
	}

	start := int(a.allocHint.Load() % uint64(numWords))
	for offset := 0; offset < numWords; offset++ {
		w := (start + offset) % numWords
		word := a.bitmask[w].Load()
		inv := ^word
		if inv == 0 {
			continue
		}
		bit := bits.TrailingZeros64(inv)
		if bit >= 64 {
			continue
		}
		slot := w*64 + bit
		if slot >= a.totalBlocks {
			continue
		}

		var data []byte
		if a.blockSizeBytes > 0 {
			data = make([]byte, a.blockSizeBytes)
		}
		id := int(a.nextID)
		block := &PhysicalBlock{
			ID:           id,
			Data:         data,
			allocationID: id,
			slot:         slot,
		}
		block.refCount.Store(1)
		block.Touch()

		a.nextID++
		a.bitmask[w].Store(word | uint64(1)<<bit)
		a.blocks[slot] = block
		a.blocksByID[id] = block
		a.allocatedCount.Add(1)
		a.allocHint.Store(uint64(w))
		return block, nil
	}
	return nil, ErrNoFreeBlocks
}

// Free retires one exact allocation incarnation and releases its physical slot.
// It returns true if the block was previously allocated and successfully freed.
func (a *PhysicalBlockAllocator) Free(id int) (bool, error) {
	if a == nil || id < 0 {
		return false, ErrInvalidBlockID
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	return a.freeLocked(id, false)
}

func (a *PhysicalBlockAllocator) freeLocked(id int, requireUnreferenced bool) (bool, error) {
	block, ok := a.blocksByID[id]
	if !ok {
		if uint64(id) >= a.nextID {
			return false, ErrInvalidBlockID
		}
		return false, nil
	}
	if block.allocationID != id {
		return false, ErrBlockNotFound
	}
	slot := block.slot
	if slot < 0 || slot >= a.totalBlocks || a.blocks[slot] != block {
		return false, ErrBlockNotFound
	}
	if requireUnreferenced && block.RefCount() > 0 {
		return false, nil
	}
	w := slot / 64
	bit := slot % 64
	mask := uint64(1) << bit
	word := a.bitmask[w].Load()
	if word&mask == 0 {
		return false, ErrBlockNotFound
	}

	delete(a.blocksByID, id)
	a.blocks[slot] = nil
	a.bitmask[w].Store(word &^ mask)
	block.refCount.Store(0)
	block.swapped.Store(false)
	a.allocatedCount.Add(-1)
	return true, nil
}

// GetBlock retrieves the physical block by ID if it is currently allocated.
func (a *PhysicalBlockAllocator) GetBlock(id int) (*PhysicalBlock, error) {
	if a == nil || id < 0 {
		return nil, ErrInvalidBlockID
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	block, ok := a.blocksByID[id]
	if !ok {
		if uint64(id) >= a.nextID {
			return nil, ErrInvalidBlockID
		}
		return nil, ErrBlockNotFound
	}
	if block.allocationID != id || block.slot < 0 || block.slot >= a.totalBlocks || a.blocks[block.slot] != block {
		return nil, ErrBlockNotFound
	}
	return block, nil
}

// Retain increments the reference count of the physical block by ID.
func (a *PhysicalBlockAllocator) Retain(id int) error {
	if a == nil || id < 0 {
		return ErrInvalidBlockID
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	block, ok := a.blocksByID[id]
	if !ok {
		if uint64(id) >= a.nextID {
			return ErrInvalidBlockID
		}
		return ErrBlockNotFound
	}
	if block.allocationID != id || block.slot < 0 || block.slot >= a.totalBlocks || a.blocks[block.slot] != block {
		return ErrBlockNotFound
	}
	if block.Retain() <= 0 {
		return ErrBlockNotFound
	}
	return nil
}

// AllocatedCount returns the number of physical blocks currently allocated.
func (a *PhysicalBlockAllocator) AllocatedCount() int {
	if a == nil {
		return 0
	}
	c := int(a.allocatedCount.Load())
	if c < 0 {
		return 0
	}
	return c
}

// TotalCount returns the total number of physical blocks in the pool.
func (a *PhysicalBlockAllocator) TotalCount() int {
	if a == nil {
		return 0
	}
	return a.totalBlocks
}

// FreeCount returns the number of available physical blocks.
func (a *PhysicalBlockAllocator) FreeCount() int {
	if a == nil {
		return 0
	}
	f := a.totalBlocks - a.AllocatedCount()
	if f < 0 {
		return 0
	}
	return f
}

// BlockSize returns the configured byte size of each physical block.
func (a *PhysicalBlockAllocator) BlockSize() int {
	if a == nil {
		return 0
	}
	return a.blockSizeBytes
}

type lruCandidate struct {
	id         int
	refCount   int32
	lastAccess int64
}

// RecycleLRU reclaims up to reclaimCount least-recently-used physical blocks,
// prioritizing unreferenced blocks (refCount <= 0) and oldest access times.
func (a *PhysicalBlockAllocator) RecycleLRU(reclaimCount int) ([]int, error) {
	if a == nil || reclaimCount <= 0 {
		return nil, nil
	}

	a.mu.RLock()
	var candidates []lruCandidate
	for slot, b := range a.blocks {
		if b != nil {
			w := slot / 64
			bit := slot % 64
			mask := uint64(1) << bit
			if a.bitmask[w].Load()&mask == 0 {
				continue
			}
			candidates = append(candidates, lruCandidate{
				id:         b.allocationID,
				refCount:   b.RefCount(),
				lastAccess: b.LastAccess(),
			})
		}
	}
	a.mu.RUnlock()

	if len(candidates) == 0 {
		return nil, nil
	}

	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].refCount != candidates[j].refCount {
			return candidates[i].refCount < candidates[j].refCount
		}
		if candidates[i].lastAccess != candidates[j].lastAccess {
			return candidates[i].lastAccess < candidates[j].lastAccess
		}
		return candidates[i].id < candidates[j].id
	})

	var reclaimed []int
	for _, c := range candidates {
		if len(reclaimed) >= reclaimCount {
			break
		}
		if c.refCount > 0 {
			break // only recycle unreferenced blocks; active resident blocks must be evicted via swap
		}
		a.mu.Lock()
		ok, freeErr := a.freeLocked(c.id, true)
		a.mu.Unlock()
		if freeErr == nil && ok {
			reclaimed = append(reclaimed, c.id)
		}
	}

	return reclaimed, nil
}
