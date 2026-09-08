package kvmmu

import (
	"errors"
)

// Common errors for KV MMU operations.
var (
	ErrNilTable = errors.New("kvmmu: nil V2PTable")
	ErrNilPool  = errors.New("kvmmu: nil PagePool")
)

// NewV2PTable creates a new virtual page table associated with a physical page pool.
func NewV2PTable(pool *PagePool, branchID string) *V2PTable {
	return &V2PTable{
		pool:     pool,
		BranchID: branchID,
		PageIDs:  make([]int64, 0),
	}
}

// AppendTokens appends tokens to the virtual page table.
// If the last page has space and RefCount == 1, it appends directly.
// If the last page has space and RefCount > 1, it performs Copy-On-Write:
//   - Allocates a new physical page from pool.
//   - Copies tokens from the old shared page.
//   - Decrements refcount on the old shared page.
//   - Appends new tokens to the fresh page with RefCount = 1.
//   - Updates PageIDs[last] = newPage.ID.
//
// It allocates fresh pages from pool as needed for remaining tokens, setting RefCount = 1.
func (t *V2PTable) AppendTokens(tokens []int) error {
	if t == nil {
		return ErrNilTable
	}
	if len(tokens) == 0 {
		return nil
	}
	if t.pool == nil {
		return ErrNilPool
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	remaining := tokens

	// If the last page has space, attempt append or COW
	if len(t.PageIDs) > 0 {
		lastID := t.PageIDs[len(t.PageIDs)-1]
		lastPage := t.pool.Page(lastID)
		if lastPage != nil && lastPage.NumTokens < PageTokens {
			space := PageTokens - lastPage.NumTokens
			toAppend := len(remaining)
			if toAppend > space {
				toAppend = space
			}

			if lastPage.RefCount.Load() == 1 {
				// Direct append to non-shared page
				for i := 0; i < toAppend; i++ {
					lastPage.Tokens[lastPage.NumTokens+i] = remaining[i]
				}
				lastPage.NumTokens += toAppend
				t.TotalTokens += toAppend
				remaining = remaining[toAppend:]
			} else if lastPage.RefCount.Load() > 1 {
				// Copy-On-Write for shared page:
				newPage := t.pool.AllocatePage()
				if newPage == nil {
					return errors.New("kvmmu: failed to allocate page from pool")
				}
				for i := 0; i < lastPage.NumTokens; i++ {
					newPage.Tokens[i] = lastPage.Tokens[i]
				}
				newPage.NumTokens = lastPage.NumTokens

				if oldRefs := lastPage.RefCount.Add(-1); oldRefs <= 0 {
					t.pool.FreePage(lastPage.ID)
				}

				for i := 0; i < toAppend; i++ {
					newPage.Tokens[newPage.NumTokens+i] = remaining[i]
				}
				newPage.NumTokens += toAppend
				newPage.RefCount.Store(1)

				t.PageIDs[len(t.PageIDs)-1] = newPage.ID
				t.TotalTokens += toAppend
				remaining = remaining[toAppend:]
			}
		}
	}

	// Allocates fresh pages from pool as needed for remaining tokens, setting RefCount = 1
	for len(remaining) > 0 {
		newPage := t.pool.AllocatePage()
		if newPage == nil {
			return errors.New("kvmmu: failed to allocate page from pool")
		}
		newPage.RefCount.Store(1)

		toAppend := len(remaining)
		if toAppend > PageTokens {
			toAppend = PageTokens
		}
		for i := 0; i < toAppend; i++ {
			newPage.Tokens[i] = remaining[i]
		}
		newPage.NumTokens = toAppend

		t.PageIDs = append(t.PageIDs, newPage.ID)
		t.TotalTokens += toAppend
		remaining = remaining[toAppend:]
	}

	return nil
}

// ForkBranch creates a zero-copy speculative child branch sharing existing physical pages.
// It copies the PageIDs slice and atomically increments RefCount on every page in PageIDs.
func (t *V2PTable) ForkBranch(childBranchID string) (*V2PTable, error) {
	if t == nil {
		return nil, ErrNilTable
	}
	if t.pool == nil {
		return nil, ErrNilPool
	}

	t.mu.RLock()
	defer t.mu.RUnlock()

	pageIDs := make([]int64, len(t.PageIDs))
	copy(pageIDs, t.PageIDs)

	for _, id := range pageIDs {
		page := t.pool.Page(id)
		if page != nil {
			page.RefCount.Add(1)
		}
	}

	child := &V2PTable{
		pool:          t.pool,
		PageIDs:       pageIDs,
		TotalTokens:   t.TotalTokens,
		BranchID:      childBranchID,
		ParentID:      t.BranchID,
		Authoritative: false,
	}
	return child, nil
}

// Commit marks this branch as authoritative.
func (t *V2PTable) Commit() error {
	if t == nil {
		return ErrNilTable
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	t.Authoritative = true
	return nil
}

// Squash immediately decrements RefCount on all held PageIDs.
// Releases pages whose RefCount drops to 0 back to pool.FreePage.
// Empties PageIDs and resets TotalTokens = 0.
func (t *V2PTable) Squash() error {
	if t == nil {
		return ErrNilTable
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	for _, id := range t.PageIDs {
		page := t.pool.Page(id)
		if page != nil {
			if remaining := page.RefCount.Add(-1); remaining <= 0 {
				t.pool.FreePage(id)
			}
		}
	}

	t.PageIDs = nil
	t.TotalTokens = 0
	return nil
}

// ReadTokens reads all tokens across all physical pages in logical order.
func (t *V2PTable) ReadTokens() []int {
	if t == nil {
		return nil
	}
	t.mu.RLock()
	defer t.mu.RUnlock()

	tokens := make([]int, 0, t.TotalTokens)
	for _, id := range t.PageIDs {
		page := t.pool.Page(id)
		if page == nil {
			continue
		}
		for i := 0; i < page.NumTokens; i++ {
			tokens = append(tokens, page.Tokens[i])
		}
	}
	return tokens
}

// UpdateFrom updates this table to adopt the winning speculative branch's pages and tokens.
// Old held pages are released and replaced with the winning branch's pages.
func (t *V2PTable) UpdateFrom(winning *V2PTable) error {
	if t == nil {
		return ErrNilTable
	}
	if winning == nil {
		return errors.New("kvmmu: nil winning V2PTable")
	}

	winning.mu.RLock()
	winPages := make([]int64, len(winning.PageIDs))
	copy(winPages, winning.PageIDs)
	winTokens := winning.TotalTokens
	winning.mu.RUnlock()

	// Increment refcount on winning pages to retain them
	for _, id := range winPages {
		if page := t.pool.Page(id); page != nil {
			page.RefCount.Add(1)
		}
	}

	// Release previous pages
	_ = t.Squash()

	t.mu.Lock()
	defer t.mu.Unlock()

	t.PageIDs = winPages
	t.TotalTokens = winTokens
	t.Authoritative = true
	return nil
}
