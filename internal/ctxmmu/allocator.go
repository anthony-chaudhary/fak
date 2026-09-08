package ctxmmu

import (
	"errors"
	"math/bits"
	"os"
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
	DefaultPoolBytesPerToken = int64(65536)                    // 64 KiB per token for FP8 GQA
	DefaultTotalGTTBytes     = int64(35) * 1024 * 1024 * 1024 // 35.0 GiB contiguous slab
	EnvHalogenKVPoolFit      = "HALOGEN_KV_POOL_FIT"           // Env var for dynamic downscale ratio
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

// Standard typed errors for physical block allocator and page table operations.
var (
	// ErrPhysicalMemoryExhausted is returned when no free physical pages remain in the block allocator.
	ErrPhysicalMemoryExhausted = errors.New("ctxmmu: physical page pool exhausted")

	// ErrInvalidPageID is returned when an out-of-bounds or invalid physical page ID is requested.
	ErrInvalidPageID = errors.New("ctxmmu: invalid physical page ID")

	// ErrPageAlreadyFree is returned if an attempt is made to release a page whose reference count is already 0.
	ErrPageAlreadyFree = errors.New("ctxmmu: physical page already free")

	// ErrPageInUse is returned when attempting an illegal operation on an active physical page.
	ErrPageInUse = errors.New("ctxmmu: physical page is in active use")
)

// PhysicalPageID identifies a distinct physical memory block allocated in unified memory.
type PhysicalPageID int

// PhysicalPage represents a single physical memory page block of fixed byte size in unified DRAM.
// Shared pages across Copy-on-Write forks are reference-counted and immutable until unshared.
type PhysicalPage struct {
	ID         PhysicalPageID
	refCount   atomic.Int32
	lastAccess atomic.Int64 // Unix nanoseconds for LRU tracking
	data       []byte       // Physical memory slab buffer for KV cache entries
}

// RefCount returns the current reference count of this physical page.
func (p *PhysicalPage) RefCount() int32 {
	return p.refCount.Load()
}

// Retain atomically increments the reference count of this physical page.
func (p *PhysicalPage) Retain() int32 {
	return p.refCount.Add(1)
}

// Release atomically decrements the reference count of this physical page.
func (p *PhysicalPage) Release() int32 {
	return p.refCount.Add(-1)
}

// IsShared reports whether more than one session holds a reference to this page.
func (p *PhysicalPage) IsShared() bool {
	return p.refCount.Load() > 1
}

// Touch updates the last-access timestamp to current wall-clock time for LRU tracking.
func (p *PhysicalPage) Touch() {
	p.lastAccess.Store(time.Now().UnixNano())
}

// LastAccess returns the last access timestamp in Unix nanoseconds.
func (p *PhysicalPage) LastAccess() int64 {
	return p.lastAccess.Load()
}

// Data returns the underlying byte slice representing the physical memory buffer.
func (p *PhysicalPage) Data() []byte {
	return p.data
}

// BlockAllocator manages a pool of fixed-size physical memory pages in unified memory.
// It provides lock-free O(1) page allocation and deallocation using a free-list bitmask,
// atomic reference counting for Copy-on-Write sharing, and LRU page tracking.
type BlockAllocator struct {
	numPages       int
	pageSizeBytes  int64
	pages          []*PhysicalPage
	bitmask        []atomic.Uint64 // Bitmask free-list: 0 = free, 1 = allocated
	searchHint     atomic.Uint64   // Round-robin hint index into bitmask words
	allocatedCount atomic.Int64
	cowClones      atomic.Int64
}

// NewBlockAllocator initializes a BlockAllocator with the specified number of physical pages
// and page byte size.
func NewBlockAllocator(numPages int, pageSizeBytes int64) *BlockAllocator {
	if numPages <= 0 {
		numPages = 1024
	}
	if pageSizeBytes <= 0 {
		pageSizeBytes = 65536
	}

	numWords := (numPages + 63) / 64
	bitmask := make([]atomic.Uint64, numWords)

	// If numPages is not an exact multiple of 64, mask out trailing out-of-bounds bits as 1 (allocated)
	// so they are never handed out during bitmask scans.
	rem := numPages % 64
	if rem != 0 {
		invalidMask := ^uint64(0) << rem
		bitmask[numWords-1].Store(invalidMask)
	}

	pages := make([]*PhysicalPage, numPages)
	for i := 0; i < numPages; i++ {
		pages[i] = &PhysicalPage{
			ID:   PhysicalPageID(i),
			data: make([]byte, pageSizeBytes),
		}
	}

	return &BlockAllocator{
		numPages:      numPages,
		pageSizeBytes: pageSizeBytes,
		pages:         pages,
		bitmask:       bitmask,
	}
}

// Allocate provides lock-free O(1) allocation of a physical page using the free-list bitmask.
// Returns ErrPhysicalMemoryExhausted when all pages in the pool are allocated.
func (a *BlockAllocator) Allocate() (PhysicalPageID, error) {
	numWords := len(a.bitmask)
	if numWords == 0 {
		return -1, ErrPhysicalMemoryExhausted
	}

	start := a.searchHint.Load() % uint64(numWords)
	for i := 0; i < numWords; i++ {
		w := int((start + uint64(i)) % uint64(numWords))
		for {
			val := a.bitmask[w].Load()
			if ^val == 0 {
				// All 64 bits in this word are allocated
				break
			}

			bit := bits.TrailingZeros64(^val)
			pageID := w*64 + bit
			if pageID >= a.numPages {
				break
			}

			newVal := val | (uint64(1) << bit)
			if a.bitmask[w].CompareAndSwap(val, newVal) {
				// Successfully claimed the bit
				a.searchHint.Store(uint64(w))
				a.allocatedCount.Add(1)

				page := a.pages[pageID]
				page.refCount.Store(1)
				page.Touch()
				return PhysicalPageID(pageID), nil
			}
			// CAS contention: retry with fresh val in the same word
		}
	}

	return -1, ErrPhysicalMemoryExhausted
}

// AllocateBatch allocates multiple physical pages in O(count) time.
// If not enough pages are available, all allocated pages in the batch are freed and an error is returned.
func (a *BlockAllocator) AllocateBatch(count int) ([]PhysicalPageID, error) {
	if count <= 0 {
		return nil, nil
	}
	res := make([]PhysicalPageID, 0, count)
	for i := 0; i < count; i++ {
		p, err := a.Allocate()
		if err != nil {
			// Rollback allocated pages
			for _, allocated := range res {
				_, _ = a.Release(allocated)
			}
			return nil, err
		}
		res = append(res, p)
	}
	return res, nil
}

// Retain increments the reference count for a physical page.
// Returns ErrInvalidPageID or ErrPageAlreadyFree if the page is invalid or unallocated.
func (a *BlockAllocator) Retain(pageID PhysicalPageID) (int32, error) {
	id := int(pageID)
	if id < 0 || id >= a.numPages {
		return 0, ErrInvalidPageID
	}
	page := a.pages[id]
	for {
		cur := page.refCount.Load()
		if cur <= 0 {
			return 0, ErrPageAlreadyFree
		}
		if page.refCount.CompareAndSwap(cur, cur+1) {
			page.Touch()
			return cur + 1, nil
		}
	}
}

// Release decrements the reference count of a physical page.
// If the reference count drops to 0, the page is recycled back to the free-list bitmask in O(1) time.
func (a *BlockAllocator) Release(pageID PhysicalPageID) (int32, error) {
	id := int(pageID)
	if id < 0 || id >= a.numPages {
		return 0, ErrInvalidPageID
	}
	page := a.pages[id]
	newRef := page.Release()
	if newRef > 0 {
		return newRef, nil
	}
	if newRef < 0 {
		page.refCount.Store(0)
		return 0, ErrPageAlreadyFree
	}

	// newRef == 0: Recycle page back to the bitmask free list in O(1)
	w := id / 64
	bit := id % 64
	mask := ^(uint64(1) << bit)
	for {
		oldVal := a.bitmask[w].Load()
		newVal := oldVal & mask
		if a.bitmask[w].CompareAndSwap(oldVal, newVal) {
			break
		}
	}

	a.allocatedCount.Add(-1)
	a.searchHint.Store(uint64(w))
	return 0, nil
}

// ClonePage performs Copy-on-Write by allocating a new physical page and copying
// the source page's buffer contents.
func (a *BlockAllocator) ClonePage(srcID PhysicalPageID) (PhysicalPageID, error) {
	src, err := a.Page(srcID)
	if err != nil {
		return -1, err
	}
	dstID, err := a.Allocate()
	if err != nil {
		return -1, err
	}
	dst := a.pages[dstID]
	copy(dst.data, src.data)
	a.cowClones.Add(1)
	return dstID, nil
}

// Page returns the PhysicalPage pointer for the given ID.
func (a *BlockAllocator) Page(pageID PhysicalPageID) (*PhysicalPage, error) {
	id := int(pageID)
	if id < 0 || id >= a.numPages {
		return nil, ErrInvalidPageID
	}
	return a.pages[id], nil
}

// PageData returns the backing byte buffer for the given physical page.
func (a *BlockAllocator) PageData(pageID PhysicalPageID) ([]byte, error) {
	p, err := a.Page(pageID)
	if err != nil {
		return nil, err
	}
	return p.Data(), nil
}

// IsAllocated checks whether the specified physical page is currently allocated in the bitmask.
func (a *BlockAllocator) IsAllocated(pageID PhysicalPageID) bool {
	id := int(pageID)
	if id < 0 || id >= a.numPages {
		return false
	}
	w := id / 64
	bit := id % 64
	return (a.bitmask[w].Load() & (uint64(1) << bit)) != 0
}

// TotalPages returns the total number of physical pages managed by this allocator.
func (a *BlockAllocator) TotalPages() int {
	return a.numPages
}

// AllocatedPages returns the number of physical pages currently allocated.
func (a *BlockAllocator) AllocatedPages() int {
	return int(a.allocatedCount.Load())
}

// FreePages returns the number of free physical pages available for allocation.
func (a *BlockAllocator) FreePages() int {
	free := a.numPages - a.AllocatedPages()
	if free < 0 {
		return 0
	}
	return free
}

// PageSizeBytes returns the size of each physical page buffer in bytes.
func (a *BlockAllocator) PageSizeBytes() int64 {
	return a.pageSizeBytes
}

// TotalBytes returns the total unified memory capacity managed by this allocator in bytes.
func (a *BlockAllocator) TotalBytes() int64 {
	return int64(a.numPages) * a.pageSizeBytes
}

// COWClones returns the total number of Copy-on-Write page clones performed.
func (a *BlockAllocator) COWClones() int64 {
	return a.cowClones.Load()
}

// FindLRUEvictable finds the allocated physical page with the oldest access timestamp
// matching the provided filter function (e.g. refCount == 1).
func (a *BlockAllocator) FindLRUEvictable(filter func(p *PhysicalPage) bool) (PhysicalPageID, error) {
	var bestID PhysicalPageID = -1
	var oldestAccess int64 = -1

	for i, p := range a.pages {
		if !a.IsAllocated(PhysicalPageID(i)) {
			continue
		}
		if filter != nil && !filter(p) {
			continue
		}
		acc := p.LastAccess()
		if oldestAccess == -1 || acc < oldestAccess {
			oldestAccess = acc
			bestID = PhysicalPageID(i)
		}
	}

	if bestID == -1 {
		return -1, errors.New("ctxmmu: no evictable physical page found")
	}
	return bestID, nil
}

// VirtualPageTable maps logical sequence page indices to physical memory page IDs.
// It supports zero-copy Copy-on-Write branching via shallow reference duplication.
type VirtualPageTable struct {
	mu      sync.RWMutex
	entries []PhysicalPageID
}

// NewVirtualPageTable creates an empty virtual page table.
func NewVirtualPageTable() *VirtualPageTable {
	return &VirtualPageTable{
		entries: make([]PhysicalPageID, 0),
	}
}

// PageCount returns the number of mapped logical pages in the page table.
func (t *VirtualPageTable) PageCount() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return len(t.entries)
}

// GetPhysicalPage returns the physical page ID mapped to the given logical page index.
func (t *VirtualPageTable) GetPhysicalPage(logicalIdx int) (PhysicalPageID, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if logicalIdx < 0 || logicalIdx >= len(t.entries) {
		return -1, false
	}
	physID := t.entries[logicalIdx]
	if physID < 0 {
		return -1, false
	}
	return physID, true
}

// MapPage maps a logical page index to a physical page ID.
func (t *VirtualPageTable) MapPage(logicalIdx int, physID PhysicalPageID) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if logicalIdx >= len(t.entries) {
		newEntries := make([]PhysicalPageID, logicalIdx+1)
		copy(newEntries, t.entries)
		for i := len(t.entries); i < logicalIdx; i++ {
			newEntries[i] = -1
		}
		t.entries = newEntries
	}
	t.entries[logicalIdx] = physID
}

// AppendPage appends a physical page mapping as the next logical page index.
func (t *VirtualPageTable) AppendPage(physID PhysicalPageID) int {
	t.mu.Lock()
	defer t.mu.Unlock()
	idx := len(t.entries)
	t.entries = append(t.entries, physID)
	return idx
}

// PhysicalPages returns a snapshot copy of all mapped physical page IDs.
func (t *VirtualPageTable) PhysicalPages() []PhysicalPageID {
	t.mu.RLock()
	defer t.mu.RUnlock()
	res := make([]PhysicalPageID, len(t.entries))
	copy(res, t.entries)
	return res
}

// Clone creates a shallow copy of the page table for zero-copy Copy-on-Write branching.
func (t *VirtualPageTable) Clone() *VirtualPageTable {
	t.mu.RLock()
	defer t.mu.RUnlock()
	cloned := make([]PhysicalPageID, len(t.entries))
	copy(cloned, t.entries)
	return &VirtualPageTable{
		entries: cloned,
	}
}
