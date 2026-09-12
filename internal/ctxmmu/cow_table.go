package ctxmmu

import (
	"encoding/binary"
	"sync"
	"sync/atomic"
	"time"
)

// COWConfig specifies configuration parameters for COWPageTable.
type COWConfig struct {
	BlockCapacity int
	BytesPerToken int
}

// COWOption configures a COWPageTable.
type COWOption func(*COWConfig)

// WithBlockCapacity sets token capacity per page block.
func WithBlockCapacity(cap int) COWOption {
	return func(c *COWConfig) {
		if cap > 0 {
			c.BlockCapacity = cap
		}
	}
}

// WithBytesPerToken sets byte footprint per token.
func WithBytesPerToken(bytes int) COWOption {
	return func(c *COWConfig) {
		if bytes > 0 {
			c.BytesPerToken = bytes
		}
	}
}

// COWPageTable coordinates zero-copy subagent prefix branching and copy-on-write page tables.
type COWPageTable struct {
	mu                  sync.RWMutex
	blockCapacity       int
	bytesPerToken       int
	blockSize           int64
	nextBlockID         atomic.Int64
	sessions            map[string]*SessionBranch
	physicalBlocks      map[int64]*PageBlock
	freeBlocks          []*PageBlock
	totalAllocatedBytes int64
	totalLogicalBytes   int64

	// Lock-free candidate tree speculation fields (#12319)
	candidates    sync.Map  // map[string]*ForkedCandidate
	candidatePool sync.Pool // pool of *ForkedCandidate
	blockPool     sync.Pool // lock-free pool of *PageBlock
}

// NewCOWPageTable constructs a new copy-on-write page table manager.
func NewCOWPageTable(opts ...COWOption) *COWPageTable {
	cfg := COWConfig{
		BlockCapacity: DefaultCOWBlockCapacity,
		BytesPerToken: DefaultCOWBytesPerToken,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	blockSize := int64(cfg.BlockCapacity * cfg.BytesPerToken)
	t := &COWPageTable{
		blockCapacity:  cfg.BlockCapacity,
		bytesPerToken:  cfg.BytesPerToken,
		blockSize:      blockSize,
		sessions:       make(map[string]*SessionBranch),
		physicalBlocks: make(map[int64]*PageBlock),
	}
	t.candidatePool.New = func() any {
		return &ForkedCandidate{
			table: t,
		}
	}
	t.blockPool.New = func() any {
		id := t.nextBlockID.Add(1)
		blk := &PageBlock{
			ID:        id,
			Capacity:  t.blockCapacity,
			Buffer:    make([]byte, t.blockSize),
			Tokens:    make([]int, 0, t.blockCapacity),
			immutable: false,
		}
		blk.refCount.Store(1)
		return blk
	}
	return t
}

func (t *COWPageTable) allocBlockLocked() *PageBlock {
	if len(t.freeBlocks) > 0 {
		blk := t.freeBlocks[len(t.freeBlocks)-1]
		t.freeBlocks = t.freeBlocks[:len(t.freeBlocks)-1]
		blk.Tokens = blk.Tokens[:0]
		blk.TokenCount = 0
		blk.immutable = false
		blk.refCount.Store(1)
		for i := range blk.Buffer {
			blk.Buffer[i] = 0
		}
		t.physicalBlocks[blk.ID] = blk
		t.totalAllocatedBytes += t.blockSize
		return blk
	}

	obj := t.blockPool.Get()
	if obj != nil {
		blk := obj.(*PageBlock)
		blk.Tokens = blk.Tokens[:0]
		blk.TokenCount = 0
		blk.immutable = false
		blk.refCount.Store(1)
		for i := range blk.Buffer {
			blk.Buffer[i] = 0
		}
		t.physicalBlocks[blk.ID] = blk
		t.totalAllocatedBytes += t.blockSize
		return blk
	}

	id := t.nextBlockID.Add(1)
	blk := &PageBlock{
		ID:         id,
		TokenCount: 0,
		Capacity:   t.blockCapacity,
		Buffer:     make([]byte, t.blockSize),
		Tokens:     make([]int, 0, t.blockCapacity),
		immutable:  false,
	}
	blk.refCount.Store(1)
	t.physicalBlocks[id] = blk
	t.totalAllocatedBytes += t.blockSize
	return blk
}

func (t *COWPageTable) allocBlockLockFree() *PageBlock {
	obj := t.blockPool.Get()
	if obj != nil {
		blk := obj.(*PageBlock)
		blk.refCount.Store(1)
		blk.TokenCount = 0
		blk.Tokens = blk.Tokens[:0]
		blk.immutable = false
		for i := range blk.Buffer {
			blk.Buffer[i] = 0
		}
		return blk
	}

	id := t.nextBlockID.Add(1)
	cap := t.blockCapacity
	if cap <= 0 {
		cap = DefaultCOWBlockCapacity
	}
	bSize := t.blockSize
	if bSize <= 0 {
		bSize = int64(cap * DefaultCOWBytesPerToken)
	}
	blk := &PageBlock{
		ID:         id,
		TokenCount: 0,
		Capacity:   cap,
		Buffer:     make([]byte, bSize),
		Tokens:     make([]int, 0, cap),
		immutable:  false,
	}
	blk.refCount.Store(1)
	return blk
}

func (t *COWPageTable) releaseBlockLockFree(blk *PageBlock) int32 {
	if blk == nil {
		return 0
	}
	newRef := blk.Release()
	if newRef <= 0 {
		blk.refCount.Store(0)
		blk.Tokens = blk.Tokens[:0]
		blk.TokenCount = 0
		blk.immutable = false
		t.blockPool.Put(blk)
	}
	return newRef
}

func (t *COWPageTable) releaseBlockLocked(blk *PageBlock) {
	if blk == nil {
		return
	}
	newRef := blk.Release()
	if newRef <= 0 {
		blk.refCount.Store(0)
		delete(t.physicalBlocks, blk.ID)
		t.totalAllocatedBytes -= t.blockSize
		blk.Tokens = blk.Tokens[:0]
		blk.TokenCount = 0
		blk.immutable = false
		for i := range blk.Buffer {
			blk.Buffer[i] = 0
		}
		t.freeBlocks = append(t.freeBlocks, blk)
	}
}

// CreateSession initializes an empty session branch with the given session ID.
func (t *COWPageTable) CreateSession(sessionID string) (*SessionBranch, error) {
	if sessionID == "" {
		return nil, ErrInvalidSessionID
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	if _, exists := t.sessions[sessionID]; exists {
		return nil, ErrSessionExists
	}
	sess := &SessionBranch{
		table:         t,
		ID:            sessionID,
		Blocks:        make([]*PageBlock, 0),
		CreatedAt:     time.Now(),
		PrefixHitRate: 1.0,
	}
	t.sessions[sessionID] = sess
	return sess, nil
}

// RegisterSession is an alias for CreateSession.
func (t *COWPageTable) RegisterSession(sessionID string) (*SessionBranch, error) {
	return t.CreateSession(sessionID)
}

// GetSession retrieves an active session branch by its ID.
func (t *COWPageTable) GetSession(sessionID string) (*SessionBranch, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	sess, ok := t.sessions[sessionID]
	if !ok {
		return nil, ErrSessionNotFound
	}
	return sess, nil
}

// HasSession reports whether a session is registered and active.
func (t *COWPageTable) HasSession(sessionID string) bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	_, ok := t.sessions[sessionID]
	return ok
}

// ActiveSessions returns a list of active session branch IDs.
func (t *COWPageTable) ActiveSessions() []string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	ids := make([]string, 0, len(t.sessions))
	for id := range t.sessions {
		ids = append(ids, id)
	}
	return ids
}

// ForkSession performs zero-copy subagent prefix branching via copy-on-write page tables.
// Page block pointers are shallow copied in O(1) time (<1 ms latency), refcounts are incremented,
// and 0 duplicate physical pages are allocated.
func (t *COWPageTable) ForkSession(parentSessionID string, childSessionID string) (*SessionBranch, error) {
	start := time.Now()

	if parentSessionID == "" || childSessionID == "" {
		return nil, ErrInvalidSessionID
	}
	if parentSessionID == childSessionID {
		return nil, ErrSelfFork
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	parent, ok := t.sessions[parentSessionID]
	if !ok {
		return nil, ErrParentNotFound
	}
	if parent.released {
		return nil, ErrSessionReleased
	}
	if _, exists := t.sessions[childSessionID]; exists {
		return nil, ErrSessionExists
	}

	parent.mu.Lock()
	defer parent.mu.Unlock()

	nBlocks := len(parent.Blocks)
	childBlocks := make([]*PageBlock, nBlocks)

	// Shallow copy page block pointers and retain refcount
	for i, blk := range parent.Blocks {
		blk.Retain()
		blk.immutable = true
		childBlocks[i] = blk
	}

	t.totalLogicalBytes += int64(nBlocks) * t.blockSize
	forkLatency := time.Since(start)

	child := &SessionBranch{
		table:         t,
		ID:            childSessionID,
		ParentID:      parentSessionID,
		Blocks:        childBlocks,
		TokenCount:    parent.TokenCount,
		PrefixTokens:  parent.TokenCount,
		PrefixBlocks:  nBlocks,
		PrefixHitRate: 1.0,
		CreatedAt:     time.Now(),
		ForkLatency:   forkLatency,
	}
	t.sessions[childSessionID] = child
	return child, nil
}

// AppendTokens appends tokens into the session branch.
// If writing to a block with RefCount > 1, copy-on-write allocates a new private physical block,
// copies existing data, decrements the parent block refcount, and assigns the new block to the session.
// Parent blocks remain immutable and untouched.
func (t *COWPageTable) AppendTokens(sessionID string, tokens []int) error {
	if len(tokens) == 0 {
		return nil
	}
	if sessionID == "" {
		return ErrInvalidSessionID
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	sess, ok := t.sessions[sessionID]
	if !ok {
		// Auto-register session if not already registered
		sess = &SessionBranch{
			table:         t,
			ID:            sessionID,
			Blocks:        make([]*PageBlock, 0),
			CreatedAt:     time.Now(),
			PrefixHitRate: 1.0,
		}
		t.sessions[sessionID] = sess
	}
	if sess.released {
		return ErrSessionReleased
	}

	sess.mu.Lock()
	defer sess.mu.Unlock()

	rem := tokens

	// Step 1: Check if the last block has space and needs COW or direct append
	if len(sess.Blocks) > 0 {
		lastIdx := len(sess.Blocks) - 1
		lastBlock := sess.Blocks[lastIdx]

		lastBlock.mu.RLock()
		isFull := lastBlock.TokenCount >= lastBlock.Capacity
		lastBlock.mu.RUnlock()

		if !isFull {
			// Check if shared: RefCount > 1 or immutable
			if lastBlock.RefCount() > 1 || lastBlock.immutable {
				// Copy-On-Write:
				// Allocate a new private physical block, copy data, decrement parent block refcount,
				// and assign new block to this session.
				newBlock := t.allocBlockLocked()

				lastBlock.mu.RLock()
				newBlock.Tokens = append(newBlock.Tokens[:0], lastBlock.Tokens...)
				newBlock.TokenCount = len(newBlock.Tokens)
				copy(newBlock.Buffer, lastBlock.Buffer)
				lastBlock.mu.RUnlock()
				newBlock.immutable = false

				// Decrement refcount on parent/shared block
				t.releaseBlockLocked(lastBlock)

				// Assign new block to this session
				sess.Blocks[lastIdx] = newBlock
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

			sess.TokenCount += toAdd
			rem = rem[toAdd:]
		}
	}

	// Step 2: For any remaining tokens, allocate new blocks
	for len(rem) > 0 {
		toAdd := len(rem)
		if toAdd > t.blockCapacity {
			toAdd = t.blockCapacity
		}

		newBlock := t.allocBlockLocked()
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

		sess.Blocks = append(sess.Blocks, newBlock)
		sess.TokenCount += toAdd
		t.totalLogicalBytes += t.blockSize

		rem = rem[toAdd:]
	}

	return nil
}

// MutateToken writes a new token value to the specified index in the session context.
// If the target block is shared (RefCount > 1), Copy-On-Write is triggered so that
// other sessions holding the block remain unmodified.
func (t *COWPageTable) MutateToken(sessionID string, tokenIndex int, newToken int) error {
	if tokenIndex < 0 {
		return ErrIndexOutOfBounds
	}
	if sessionID == "" {
		return ErrInvalidSessionID
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	sess, ok := t.sessions[sessionID]
	if !ok {
		return ErrSessionNotFound
	}
	if sess.released {
		return ErrSessionReleased
	}

	sess.mu.Lock()
	defer sess.mu.Unlock()

	currIdx := 0
	for i, blk := range sess.Blocks {
		if tokenIndex >= currIdx && tokenIndex < currIdx+blk.TokenCount {
			offset := tokenIndex - currIdx

			// If block is shared, trigger Copy-On-Write
			if blk.RefCount() > 1 || blk.immutable {
				newBlock := t.allocBlockLocked()

				blk.mu.RLock()
				newBlock.Tokens = append(newBlock.Tokens[:0], blk.Tokens...)
				newBlock.TokenCount = len(newBlock.Tokens)
				copy(newBlock.Buffer, blk.Buffer)
				blk.mu.RUnlock()
				newBlock.immutable = false

				t.releaseBlockLocked(blk)
				sess.Blocks[i] = newBlock
				blk = newBlock
			}

			// Mutate token in private block
			blk.mu.Lock()
			blk.Tokens[offset] = newToken
			off := offset * t.bytesPerToken
			if off+8 <= len(blk.Buffer) {
				binary.LittleEndian.PutUint64(blk.Buffer[off:], uint64(newToken))
			}
			blk.mu.Unlock()
			return nil
		}
		currIdx += blk.TokenCount
	}

	return ErrIndexOutOfBounds
}

// ReleaseSession releases a session and decrements refcounts on all blocks owned by the session,
// freeing physical blocks that reach refcount 0.
func (t *COWPageTable) ReleaseSession(sessionID string) error {
	if sessionID == "" {
		return ErrInvalidSessionID
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	sess, ok := t.sessions[sessionID]
	if !ok {
		return ErrSessionNotFound
	}
	if sess.released {
		return ErrSessionReleased
	}

	sess.mu.Lock()
	sess.released = true

	for _, blk := range sess.Blocks {
		t.releaseBlockLocked(blk)
	}
	t.totalLogicalBytes -= int64(len(sess.Blocks)) * t.blockSize
	sess.Blocks = nil
	sess.TokenCount = 0
	sess.mu.Unlock()

	delete(t.sessions, sessionID)
	return nil
}
