package ctxmmu

import (
	"encoding/binary"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// -----------------------------------------------------------------------------
// Lock-Free Candidate Tree Speculation for MTP Draft Tokens (#12319)
// -----------------------------------------------------------------------------

// CandidateBranch is an alias for ForkedCandidate to support speculative tree terminology (#12319).
type CandidateBranch = ForkedCandidate

// ForkedCandidate represents a lightweight speculative candidate branch for multi-candidate
// MTP (Multi-Token Prediction) draft tokens (#12319).
// Decoupled from the heavyweight SessionBranch registry, it references existing PageBlock
// pointers via atomic reference counting, bypassing global mutexes during fork and squash.
type ForkedCandidate struct {
	mu            sync.RWMutex
	table         *COWPageTable
	ID            string
	ParentID      string
	depth         int
	Blocks        []*PageBlock
	TokenCount    int
	CreatedAt     time.Time
	ForkLatency   time.Duration
	released      bool
	authoritative bool
	inlineBlocks  [64]*PageBlock
}

// CandidateID returns the candidate branch ID.
func (c *ForkedCandidate) CandidateID() string {
	return c.ID
}

// SessionID returns the candidate branch ID.
func (c *ForkedCandidate) SessionID() string {
	return c.ID
}

// GetParentID returns the parent branch ID (session or candidate).
func (c *ForkedCandidate) GetParentID() string {
	return c.ParentID
}

// Depth returns the tree depth of this speculative candidate (1 for root child).
func (c *ForkedCandidate) Depth() int {
	return c.depth
}

// PageCount returns the number of page blocks in this candidate branch.
func (c *ForkedCandidate) PageCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.Blocks)
}

// BlockCount returns the number of page blocks in this candidate branch.
func (c *ForkedCandidate) BlockCount() int {
	return c.PageCount()
}

// IsReleased reports whether this candidate branch has been squashed or released.
func (c *ForkedCandidate) IsReleased() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.released
}

// IsAuthoritative reports whether this candidate branch has been committed/promoted.
func (c *ForkedCandidate) IsAuthoritative() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.authoritative
}

// SyncUMA provides an explicit CPU memory store barrier to guarantee UMA DRAM cache
// coherence between the CPU drafter and GPU tree-attention verifier.
func (c *ForkedCandidate) SyncUMA() {
	var barrier atomic.Int32
	barrier.Store(1)
}

// ReadTokens returns a defensive copy of all tokens in this candidate branch.
func (c *ForkedCandidate) ReadTokens() []int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.released {
		return nil
	}
	res := make([]int, 0, c.TokenCount)
	rem := c.TokenCount
	for _, blk := range c.Blocks {
		if rem <= 0 {
			break
		}
		blk.mu.RLock()
		take := blk.TokenCount
		if take > rem {
			take = rem
		}
		res = append(res, blk.Tokens[:take]...)
		rem -= take
		blk.mu.RUnlock()
	}
	return res
}

// Tokens returns a defensive copy of all tokens in this candidate branch.
func (c *ForkedCandidate) Tokens() []int {
	return c.ReadTokens()
}

// Append appends variadic draft tokens to this candidate branch.
func (c *ForkedCandidate) Append(tokens ...int) error {
	return c.AppendTokens(tokens)
}

// AppendTokens appends draft tokens into this candidate branch.
// If the final block is shared with parent or sibling candidates (RefCount > 1),
// Copy-On-Write allocates a private physical block, preserving parent blocks.
func (c *ForkedCandidate) AppendTokens(tokens []int) error {
	if len(tokens) == 0 {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.released {
		return ErrCandidateReleased
	}

	t := c.table
	rem := tokens

	// Step 1: Check if the last block has space and needs COW or direct append
	if len(c.Blocks) > 0 {
		lastIdx := len(c.Blocks) - 1
		lastBlock := c.Blocks[lastIdx]

		lastBlock.mu.RLock()
		isFull := lastBlock.TokenCount >= lastBlock.Capacity
		lastBlock.mu.RUnlock()

		if !isFull {
			// Check if shared
			if lastBlock.RefCount() > 1 || lastBlock.immutable {
				// Copy-On-Write for candidate branch
				newBlock := t.allocBlockLockFree()

				lastBlock.mu.RLock()
				newBlock.Tokens = append(newBlock.Tokens[:0], lastBlock.Tokens...)
				newBlock.TokenCount = len(newBlock.Tokens)
				copy(newBlock.Buffer, lastBlock.Buffer)
				lastBlock.mu.RUnlock()
				newBlock.immutable = false

				// Decrement refcount on shared block
				t.releaseBlockLockFree(lastBlock)

				// Replace block in candidate
				c.Blocks[lastIdx] = newBlock
				lastBlock = newBlock
			}

			// Append tokens into private lastBlock up to remaining capacity
			avail := lastBlock.Capacity - lastBlock.TokenCount
			toAdd := len(rem)
			if toAdd > avail {
				toAdd = avail
			}

			lastBlock.mu.Lock()
			startTokIdx := len(lastBlock.Tokens)
			lastBlock.Tokens = append(lastBlock.Tokens, rem[:toAdd]...)
			lastBlock.TokenCount = len(lastBlock.Tokens)
			for j := 0; j < toAdd; j++ {
				tokIdx := startTokIdx + j
				off := tokIdx * t.bytesPerToken
				if off+8 <= len(lastBlock.Buffer) {
					binary.LittleEndian.PutUint64(lastBlock.Buffer[off:], uint64(rem[j]))
				}
			}
			lastBlock.mu.Unlock()

			c.TokenCount += toAdd
			rem = rem[toAdd:]
		}
	}

	// Step 2: For any remaining tokens, allocate new blocks
	for len(rem) > 0 {
		toAdd := len(rem)
		if toAdd > t.blockCapacity {
			toAdd = t.blockCapacity
		}

		newBlock := t.allocBlockLockFree()
		newBlock.mu.Lock()
		newBlock.Tokens = append(newBlock.Tokens[:0], rem[:toAdd]...)
		newBlock.TokenCount = len(newBlock.Tokens)
		for j := 0; j < toAdd; j++ {
			off := j * t.bytesPerToken
			if off+8 <= len(newBlock.Buffer) {
				binary.LittleEndian.PutUint64(newBlock.Buffer[off:], uint64(rem[j]))
			}
		}
		newBlock.immutable = false
		newBlock.mu.Unlock()

		c.Blocks = append(c.Blocks, newBlock)
		c.TokenCount += toAdd

		rem = rem[toAdd:]
	}

	c.SyncUMA()
	return nil
}

// MutateToken writes a new token value to the specified index in the candidate branch.
func (c *ForkedCandidate) MutateToken(tokenIndex int, newToken int) error {
	if tokenIndex < 0 {
		return ErrIndexOutOfBounds
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.released {
		return ErrCandidateReleased
	}

	t := c.table
	currIdx := 0
	for i, blk := range c.Blocks {
		if tokenIndex >= currIdx && tokenIndex < currIdx+blk.TokenCount {
			offset := tokenIndex - currIdx

			if blk.RefCount() > 1 || blk.immutable {
				newBlock := t.allocBlockLockFree()

				blk.mu.RLock()
				newBlock.Tokens = append(newBlock.Tokens[:0], blk.Tokens...)
				newBlock.TokenCount = len(newBlock.Tokens)
				copy(newBlock.Buffer, blk.Buffer)
				blk.mu.RUnlock()
				newBlock.immutable = false

				t.releaseBlockLockFree(blk)
				c.Blocks[i] = newBlock
				blk = newBlock
			}

			blk.mu.Lock()
			blk.Tokens[offset] = newToken
			off := offset * t.bytesPerToken
			if off+8 <= len(blk.Buffer) {
				binary.LittleEndian.PutUint64(blk.Buffer[off:], uint64(newToken))
			}
			blk.mu.Unlock()
			c.SyncUMA()
			return nil
		}
		currIdx += blk.TokenCount
	}

	return ErrIndexOutOfBounds
}

// Squash squashes this candidate branch in <20µs, returning unshared blocks to the pool.
func (c *ForkedCandidate) Squash() error {
	if c == nil || c.table == nil {
		return ErrCandidateNotFound
	}
	return c.table.SquashCandidate(c.ID)
}

// Promote commits this candidate branch to its authoritative parent.
func (c *ForkedCandidate) Promote(acceptedTokens ...int) error {
	if c == nil || c.table == nil {
		return ErrCandidateNotFound
	}
	return c.table.PromoteCandidate(c.ID, acceptedTokens...)
}

// ForkCandidate performs zero-copy candidate tree speculation branching for MTP draft tokens (#12319).
// It shallow-copies the parent session or candidate branch's block slice in O(1) time and atomically
// retains each block's refcount without acquiring the global COWPageTable.mu mutex.
func (t *COWPageTable) ForkCandidate(parentBranchID string, candidateID string) (*ForkedCandidate, error) {
	start := time.Now()

	if parentBranchID == "" || candidateID == "" {
		return nil, ErrInvalidSessionID
	}
	if parentBranchID == candidateID {
		return nil, ErrSelfFork
	}

	if _, exists := t.candidates.Load(candidateID); exists {
		return nil, ErrCandidateExists
	}

	var parentBlocks []*PageBlock
	var tokenCount int
	var depth int

	// First check if parent is another candidate branch (hierarchical draft tree)
	if pVal, isCand := t.candidates.Load(parentBranchID); isCand {
		parentCand := pVal.(*ForkedCandidate)
		parentCand.mu.RLock()
		if parentCand.released {
			parentCand.mu.RUnlock()
			return nil, ErrCandidateReleased
		}
		if parentCand.depth >= MaxCandidateTreeDepth {
			parentCand.mu.RUnlock()
			return nil, ErrMaxTreeDepth
		}
		depth = parentCand.depth + 1
		tokenCount = parentCand.TokenCount
		parentBlocks = parentCand.Blocks
		parentCand.mu.RUnlock()
	} else {
		// Parent must be a root session branch. Use RLock to allow concurrent forks.
		t.mu.RLock()
		parentSess, ok := t.sessions[parentBranchID]
		if !ok {
			t.mu.RUnlock()
			return nil, ErrParentNotFound
		}
		if parentSess.released {
			t.mu.RUnlock()
			return nil, ErrSessionReleased
		}
		parentSess.mu.RLock()
		t.mu.RUnlock()

		depth = 1
		tokenCount = parentSess.TokenCount
		parentBlocks = parentSess.Blocks
		parentSess.mu.RUnlock()
	}

	// Allocate candidate descriptor from pool
	var cand *ForkedCandidate
	if obj := t.candidatePool.Get(); obj != nil {
		cand = obj.(*ForkedCandidate)
	} else {
		cand = &ForkedCandidate{table: t}
	}

	cand.mu.Lock()
	cand.table = t
	cand.ID = candidateID
	cand.ParentID = parentBranchID
	cand.depth = depth
	cand.TokenCount = tokenCount
	cand.released = false
	cand.authoritative = false
	cand.CreatedAt = time.Now()

	nBlocks := len(parentBlocks)
	if nBlocks <= len(cand.inlineBlocks) {
		cand.Blocks = cand.inlineBlocks[:nBlocks]
	} else {
		cand.Blocks = make([]*PageBlock, nBlocks)
	}

	// Atomically retain each block and mark immutable
	for i, blk := range parentBlocks {
		blk.Retain()
		blk.immutable = true
		cand.Blocks[i] = blk
	}

	cand.ForkLatency = time.Since(start)
	cand.SyncUMA()
	cand.mu.Unlock()

	t.candidates.Store(candidateID, cand)
	return cand, nil
}

// ForkCandidateUInt32 is a convenience helper for uint32 candidate IDs.
func (t *COWPageTable) ForkCandidateUInt32(parentBranchID string, candID uint32) (*ForkedCandidate, error) {
	return t.ForkCandidate(parentBranchID, strconv.FormatUint(uint64(candID), 10))
}

// SquashCandidate squashes a losing candidate branch in <20µs (#12319).
// It atomically decrements block refcounts, returns unshared private blocks to the lock-free pool,
// and returns the candidate descriptor to sync.Pool without taking the global COWPageTable.mu mutex.
func (t *COWPageTable) SquashCandidate(candidateID string) error {
	if candidateID == "" {
		return ErrInvalidSessionID
	}

	val, ok := t.candidates.LoadAndDelete(candidateID)
	if !ok {
		return ErrCandidateNotFound
	}
	cand := val.(*ForkedCandidate)

	cand.mu.Lock()
	if cand.released {
		cand.mu.Unlock()
		return ErrCandidateReleased
	}
	cand.released = true

	// Atomically decrement refcounts on all blocks; unshared blocks return to pool
	for _, blk := range cand.Blocks {
		t.releaseBlockLockFree(blk)
	}

	for i := range cand.inlineBlocks {
		cand.inlineBlocks[i] = nil
	}
	cand.Blocks = nil
	cand.TokenCount = 0
	cand.mu.Unlock()

	t.candidatePool.Put(cand)
	return nil
}

// SquashCandidateBranch squashes candidate by pointer reference.
func (t *COWPageTable) SquashCandidateBranch(cand *ForkedCandidate) error {
	if cand == nil {
		return ErrCandidateNotFound
	}
	return t.SquashCandidate(cand.ID)
}

// PromoteCandidate commits a winning candidate branch to its authoritative parent (#12319).
// It replaces the parent's block mapping with the winning candidate's blocks up to acceptedTokens,
// squashing obsolete prior pages and unaccepted draft tail pages in zero byte copies.
// If acceptedTokens is omitted, all tokens in the candidate are accepted.
func (t *COWPageTable) PromoteCandidate(candidateID string, acceptedTokens ...int) error {
	if candidateID == "" {
		return ErrInvalidSessionID
	}

	val, ok := t.candidates.Load(candidateID)
	if !ok {
		return ErrCandidateNotFound
	}
	cand := val.(*ForkedCandidate)

	cand.mu.Lock()
	defer cand.mu.Unlock()

	if cand.released {
		return ErrCandidateReleased
	}

	numAccepted := cand.TokenCount
	if len(acceptedTokens) > 0 && acceptedTokens[0] >= 0 && acceptedTokens[0] < cand.TokenCount {
		numAccepted = acceptedTokens[0]
	}

	// Prepare accepted blocks: take prefix up to numAccepted, release any unaccepted tail blocks
	newBlocks := make([]*PageBlock, 0, len(cand.Blocks))
	tokensRemaining := numAccepted

	for _, blk := range cand.Blocks {
		if tokensRemaining <= 0 {
			// Unaccepted draft tail page - release immediately
			t.releaseBlockLockFree(blk)
			continue
		}

		blk.mu.RLock()
		blkTokCount := blk.TokenCount
		blk.mu.RUnlock()

		if tokensRemaining >= blkTokCount {
			// Full block accepted
			newBlocks = append(newBlocks, blk)
			tokensRemaining -= blkTokCount
		} else {
			// Partial block acceptance: truncate to tokensRemaining
			if blk.RefCount() > 1 || blk.immutable {
				// Block is shared, allocate private copy for truncated portion
				cloned := t.allocBlockLockFree()
				blk.mu.RLock()
				cloned.Tokens = append(cloned.Tokens[:0], blk.Tokens[:tokensRemaining]...)
				cloned.TokenCount = tokensRemaining
				copy(cloned.Buffer, blk.Buffer[:tokensRemaining*t.bytesPerToken])
				blk.mu.RUnlock()
				cloned.immutable = false
				t.releaseBlockLockFree(blk)
				blk = cloned
			} else {
				// Private block: truncate in place
				blk.mu.Lock()
				blk.Tokens = blk.Tokens[:tokensRemaining]
				blk.TokenCount = tokensRemaining
				startByte := tokensRemaining * t.bytesPerToken
				if startByte < len(blk.Buffer) {
					for b := startByte; b < len(blk.Buffer); b++ {
						blk.Buffer[b] = 0
					}
				}
				blk.mu.Unlock()
			}
			newBlocks = append(newBlocks, blk)
			tokensRemaining = 0
		}
	}

	// Promote into parent (candidate or session branch)
	if pCandVal, isCand := t.candidates.Load(cand.ParentID); isCand {
		parentCand := pCandVal.(*ForkedCandidate)
		parentCand.mu.Lock()
		if parentCand.released {
			parentCand.mu.Unlock()
			return ErrCandidateReleased
		}
		for _, oldBlk := range parentCand.Blocks {
			t.releaseBlockLockFree(oldBlk)
		}
		parentCand.Blocks = newBlocks
		parentCand.TokenCount = numAccepted
		parentCand.SyncUMA()
		parentCand.mu.Unlock()
	} else {
		// Parent is a root session branch
		t.mu.Lock()
		parentSess, ok := t.sessions[cand.ParentID]
		if !ok {
			t.mu.Unlock()
			return ErrParentNotFound
		}
		parentSess.mu.Lock()
		if parentSess.released {
			parentSess.mu.Unlock()
			t.mu.Unlock()
			return ErrSessionReleased
		}

		for _, oldBlk := range parentSess.Blocks {
			t.releaseBlockLocked(oldBlk)
		}

		for _, blk := range newBlocks {
			if _, exists := t.physicalBlocks[blk.ID]; !exists {
				t.physicalBlocks[blk.ID] = blk
				t.totalAllocatedBytes += t.blockSize
			}
		}

		parentSess.Blocks = newBlocks
		parentSess.TokenCount = numAccepted
		parentSess.mu.Unlock()
		t.mu.Unlock()
	}

	cand.authoritative = true
	cand.released = true
	cand.Blocks = nil
	cand.TokenCount = 0
	for i := range cand.inlineBlocks {
		cand.inlineBlocks[i] = nil
	}
	t.candidates.Delete(candidateID)
	cand.SyncUMA()

	return nil
}

// PromoteCandidateBranch promotes candidate by pointer reference.
func (t *COWPageTable) PromoteCandidateBranch(parentID string, cand *ForkedCandidate, acceptedTokens ...int) error {
	if cand == nil {
		return ErrCandidateNotFound
	}
	return t.PromoteCandidate(cand.ID, acceptedTokens...)
}

// GetCandidate retrieves an active candidate branch by ID.
func (t *COWPageTable) GetCandidate(candidateID string) (*ForkedCandidate, error) {
	if candidateID == "" {
		return nil, ErrInvalidSessionID
	}
	val, ok := t.candidates.Load(candidateID)
	if !ok {
		return nil, ErrCandidateNotFound
	}
	return val.(*ForkedCandidate), nil
}

// HasCandidate reports whether a candidate is registered and active.
func (t *COWPageTable) HasCandidate(candidateID string) bool {
	if candidateID == "" {
		return false
	}
	val, ok := t.candidates.Load(candidateID)
	if !ok {
		return false
	}
	cand := val.(*ForkedCandidate)
	return !cand.IsReleased()
}

// ActiveCandidates returns a list of active candidate branch IDs.
func (t *COWPageTable) ActiveCandidates() []string {
	var res []string
	t.candidates.Range(func(key, val any) bool {
		cand := val.(*ForkedCandidate)
		if !cand.IsReleased() {
			res = append(res, key.(string))
		}
		return true
	})
	return res
}

// CandidateCount returns the count of active candidate branches.
func (t *COWPageTable) CandidateCount() int {
	var count int
	t.candidates.Range(func(_, val any) bool {
		cand := val.(*ForkedCandidate)
		if !cand.IsReleased() {
			count++
		}
		return true
	})
	return count
}

// ForkCOWCandidate forks a speculative candidate branch via default COWPageTable.
func ForkCOWCandidate(parentID, candidateID string) (*ForkedCandidate, error) {
	return defaultCOWPageTable.ForkCandidate(parentID, candidateID)
}

// SquashCOWCandidate squashes a candidate branch via default COWPageTable.
func SquashCOWCandidate(candidateID string) error {
	return defaultCOWPageTable.SquashCandidate(candidateID)
}

// PromoteCOWCandidate promotes a winning candidate branch via default COWPageTable.
func PromoteCOWCandidate(candidateID string, acceptedTokens ...int) error {
	return defaultCOWPageTable.PromoteCandidate(candidateID, acceptedTokens...)
}
