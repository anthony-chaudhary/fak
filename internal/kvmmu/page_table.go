package kvmmu

import (
	"sync"
	"sync/atomic"
)

// PageTokens defines the fixed number of tokens per physical KV page block (16 tokens).
const PageTokens = 16

// PhysicalPage represents a single fixed-size physical page block of KV cache.
type PhysicalPage struct {
	ID        int64
	Tokens    [PageTokens]int
	NumTokens int
	RefCount  atomic.Int32
	Pinned    bool
}

// PagePool manages an allocation pool of fixed-size physical pages.
type PagePool struct {
	pages   map[int64]*PhysicalPage
	freeIDs []int64
	nextID  atomic.Int64
	mu      sync.RWMutex
}

// NewPagePool initializes and returns an empty physical page pool.
func NewPagePool() *PagePool {
	return &PagePool{
		pages:   make(map[int64]*PhysicalPage),
		freeIDs: make([]int64, 0),
	}
}

// AllocatePage allocates a physical page from the pool with RefCount initialized to 1.
// If reusable pages exist in the free list, one is recycled; otherwise a new page is allocated.
func (p *PagePool) AllocatePage() *PhysicalPage {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(p.freeIDs) > 0 {
		lastIdx := len(p.freeIDs) - 1
		id := p.freeIDs[lastIdx]
		p.freeIDs = p.freeIDs[:lastIdx]

		page := p.pages[id]
		page.Tokens = [PageTokens]int{}
		page.NumTokens = 0
		page.RefCount.Store(1)
		page.Pinned = false
		return page
	}

	id := p.nextID.Add(1)
	page := &PhysicalPage{
		ID: id,
	}
	page.RefCount.Store(1)
	p.pages[id] = page
	return page
}

// FreePage returns a page back to the free list when its reference count reaches zero.
func (p *PagePool) FreePage(id int64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	page, exists := p.pages[id]
	if !exists {
		return
	}

	// Guard against duplicate frees
	for _, fid := range p.freeIDs {
		if fid == id {
			return
		}
	}

	page.RefCount.Store(0)
	page.NumTokens = 0
	page.Tokens = [PageTokens]int{}
	page.Pinned = false
	p.freeIDs = append(p.freeIDs, id)
}

// ActivePageCount returns the number of physical pages currently in use.
func (p *PagePool) ActivePageCount() int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return len(p.pages) - len(p.freeIDs)
}

// Page retrieves a physical page by ID from the pool, or nil if not found.
func (p *PagePool) Page(id int64) *PhysicalPage {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.pages[id]
}

// GetPage is an alias for Page to look up a physical page by ID.
func (p *PagePool) GetPage(id int64) *PhysicalPage {
	return p.Page(id)
}

// V2PTable represents a virtual-to-physical paged KV mapping table for speculative branches.
type V2PTable struct {
	pool          *PagePool
	PageIDs       []int64
	TotalTokens   int
	BranchID      string
	ParentID      string
	Authoritative bool
	mu            sync.RWMutex
}

// Pool returns the underlying physical page pool.
func (t *V2PTable) Pool() *PagePool {
	return t.pool
}
