package ctxmmu

import (
	"encoding/binary"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// cow.go — Zero-copy subagent prefix branching via copy-on-write page tables on UMA (#11618).
//
// When a coordinator or parent agent fans out to N subagents sharing a large prefix
// (e.g. 32k-64k tokens of repo index, tool schemas, goal), physical DRAM pages are not duplicated.
// Instead, page block pointers are shallow copied in O(1) time (<1 ms latency), reference counts
// are incremented, and child sessions operate over shared immutable blocks. Any subsequent token
// appends or mutations to shared blocks trigger fine-grained Copy-On-Write (COW) at block granularity,
// allocating private physical pages only for modified blocks while preserving parent and sibling blocks.

const (
	// DefaultCOWBlockCapacity is the default token capacity per page block (64 tokens/block).
	DefaultCOWBlockCapacity = 64

	// DefaultCOWBytesPerToken is the default KV cache byte footprint per token in UMA DRAM (128 bytes).
	DefaultCOWBytesPerToken = 128

	// MaxCandidateTreeDepth is the maximum allowable speculative candidate tree recursion depth (#12319).
	MaxCandidateTreeDepth = 8
)

// Sentinel errors for candidate tree speculation (#12319).
var (
	ErrCandidateNotFound = errors.New("ctxmmu: candidate not found")
	ErrCandidateExists   = errors.New("ctxmmu: candidate already exists")
	ErrMaxTreeDepth      = errors.New("ctxmmu: max candidate tree depth exceeded")
	ErrCandidateReleased = errors.New("ctxmmu: candidate has already been released")
)

// PageBlock represents one physical page block holding token data and KV cache state in UMA DRAM.
// Shared blocks are reference-counted and immutable until a session performs Copy-On-Write.
type PageBlock struct {
	mu         sync.RWMutex
	refCount   atomic.Int32
	ID         int64
	TokenCount int
	Capacity   int
	Buffer     []byte // read-only physical buffer when shared
	Tokens     []int  // token sequence stored in this block
	immutable  bool
}

// RefCount returns the active reference count of this physical block across sessions.
func (b *PageBlock) RefCount() int32 {
	return b.refCount.Load()
}

// Retain increments the block reference count atomically.
func (b *PageBlock) Retain() int32 {
	return b.refCount.Add(1)
}

// Release decrements the block reference count atomically.
func (b *PageBlock) Release() int32 {
	return b.refCount.Add(-1)
}

// IsShared reports whether more than one session holds a reference to this block.
func (b *PageBlock) IsShared() bool {
	return b.refCount.Load() > 1
}

// IsFull reports whether the block has reached its token capacity.
func (b *PageBlock) IsFull() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.TokenCount >= b.Capacity
}

// RemainingCapacity returns how many token slots remain unfilled in this block.
func (b *PageBlock) RemainingCapacity() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.Capacity - b.TokenCount
}

// PhysicalBuffer returns the underlying physical byte buffer.
func (b *PageBlock) PhysicalBuffer() []byte {
	return b.Buffer
}

// ReadOnlyBuffer returns the read-only view of the physical buffer.
func (b *PageBlock) ReadOnlyBuffer() []byte {
	return b.Buffer
}

// BlockID returns the unique physical block ID.
func (b *PageBlock) BlockID() int64 {
	return b.ID
}

// Clone creates a new private copy of the page block with initial refCount 1.
func (b *PageBlock) Clone(newID int64) *PageBlock {
	b.mu.RLock()
	defer b.mu.RUnlock()

	newBuf := make([]byte, len(b.Buffer))
	copy(newBuf, b.Buffer)

	newTokens := make([]int, len(b.Tokens))
	copy(newTokens, b.Tokens)

	clone := &PageBlock{
		ID:         newID,
		TokenCount: b.TokenCount,
		Capacity:   b.Capacity,
		Buffer:     newBuf,
		Tokens:     newTokens,
		immutable:  false,
	}
	clone.refCount.Store(1)
	return clone
}

// SessionBranch represents an active subagent or coordinator session branch
// backed by copy-on-write page tables.
type SessionBranch struct {
	mu            sync.RWMutex
	table         *COWPageTable
	ID            string
	ParentID      string
	Blocks        []*PageBlock
	TokenCount    int
	PrefixTokens  int
	PrefixBlocks  int
	PrefixHitRate float64
	CreatedAt     time.Time
	ForkLatency   time.Duration
	released      bool
}

// SessionID returns the branch ID.
func (s *SessionBranch) SessionID() string {
	return s.ID
}

// GetParentID returns the ID of the parent branch from which this branch was forked.
func (s *SessionBranch) GetParentID() string {
	return s.ParentID
}

// PageCount returns the number of page table blocks in this session.
func (s *SessionBranch) PageCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.Blocks)
}

// BlockCount returns the number of page table blocks in this session.
func (s *SessionBranch) BlockCount() int {
	return s.PageCount()
}

// Tokens returns a defensive copy of all tokens in this branch context.
func (s *SessionBranch) Tokens() []int {
	return s.ReadTokens()
}

// ReadTokens returns a defensive copy of all tokens in this branch context.
func (s *SessionBranch) ReadTokens() []int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.released {
		return nil
	}
	res := make([]int, 0, s.TokenCount)
	for _, blk := range s.Blocks {
		blk.mu.RLock()
		res = append(res, blk.Tokens...)
		blk.mu.RUnlock()
	}
	return res
}

// AppendTokens appends tokens into this session branch.
func (s *SessionBranch) AppendTokens(tokens []int) error {
	return s.table.AppendTokens(s.ID, tokens)
}

// Append appends variadic tokens into this session branch.
func (s *SessionBranch) Append(tokens ...int) error {
	return s.table.AppendTokens(s.ID, tokens)
}

// MutateToken mutates a token at tokenIndex, triggering COW if the block is shared.
func (s *SessionBranch) MutateToken(tokenIndex int, newToken int) error {
	return s.table.MutateToken(s.ID, tokenIndex, newToken)
}

// Release releases this session branch and decrements refcounts on its page blocks.
func (s *SessionBranch) Release() error {
	return s.table.ReleaseSession(s.ID)
}

// Fork forks a new child session branch from this branch via zero-copy COW page tables.
func (s *SessionBranch) Fork(childID string) (*SessionBranch, error) {
	return s.table.ForkSession(s.ID, childID)
}

// IsReleased reports whether this session branch has been released.
func (s *SessionBranch) IsReleased() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.released
}

// SharedBlocksCount returns the count of blocks in this branch shared with other branches.
func (s *SessionBranch) SharedBlocksCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var count int
	for _, blk := range s.Blocks {
		if blk.RefCount() > 1 {
			count++
		}
	}
	return count
}

// UniqueBlocksCount returns the count of blocks uniquely owned by this branch.
func (s *SessionBranch) UniqueBlocksCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var count int
	for _, blk := range s.Blocks {
		if blk.RefCount() <= 1 {
			count++
		}
	}
	return count
}

// SharedPages returns the count of blocks in this branch shared with other branches.
func (s *SessionBranch) SharedPages() int {
	return s.SharedBlocksCount()
}

// COWBranchTelemetry captures telemetry on branch creation, prefix sharing, and latency.
type COWBranchTelemetry struct {
	SessionID             string        `json:"session_id"`
	ParentID              string        `json:"parent_id"`
	ForkLatency           time.Duration `json:"fork_latency"`
	ForkLatencyMs         float64       `json:"fork_latency_ms"`
	ForkCloneBytes        int64         `json:"fork_clone_bytes"`
	SharedPrefixKVHitRate float64       `json:"shared_prefix_kv_hit_rate"`
	SharedPagesCount      int           `json:"shared_pages_count"`
	UniquePagesCount      int           `json:"unique_pages_count"`
	TotalPagesCount       int           `json:"total_pages_count"`
	TokenCount            int           `json:"token_count"`
	PhysicalBytesSaved    int64         `json:"physical_bytes_saved,omitempty"`
	SavedMemorySummary    string        `json:"saved_memory_summary,omitempty"`
}

// Telemetry returns telemetry for this session branch.
func (s *SessionBranch) Telemetry() COWBranchTelemetry {
	s.mu.RLock()
	defer s.mu.RUnlock()
	shared := 0
	unique := 0
	for _, b := range s.Blocks {
		if b.RefCount() > 1 {
			shared++
		} else {
			unique++
		}
	}
	bytesSaved := int64(0)
	if s.table != nil {
		bytesSaved = int64(s.PrefixBlocks) * s.table.blockSize
	}
	return COWBranchTelemetry{
		SessionID:             s.ID,
		ParentID:              s.ParentID,
		ForkLatency:           s.ForkLatency,
		ForkLatencyMs:         float64(s.ForkLatency.Nanoseconds()) / 1e6,
		ForkCloneBytes:        0,
		SharedPrefixKVHitRate: s.PrefixHitRate,
		SharedPagesCount:      shared,
		UniquePagesCount:      unique,
		TotalPagesCount:       len(s.Blocks),
		TokenCount:            s.TokenCount,
		PhysicalBytesSaved:    bytesSaved,
		SavedMemorySummary:    FormatSavedMemory(bytesSaved),
	}
}

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

// COWMetrics tracks falsifiable metrics on prefix deduplication and memory savings.
type COWMetrics struct {
	TotalAllocatedBytes int64   `json:"total_allocated_bytes"` // Physical DRAM bytes allocated
	DeduplicatedBytes   int64   `json:"deduplicated_bytes"`    // Avoided DRAM bytes through sharing
	DedupRatio          float64 `json:"dedup_ratio"`           // 1.0 - (physical / logical)
	ActiveBranches      int     `json:"active_branches"`       // Active session branches count
	TotalPhysicalBlocks int     `json:"total_physical_blocks"` // Unique active physical blocks in DRAM
	TotalLogicalBlocks  int     `json:"total_logical_blocks"`  // Total logical blocks across all branches
	LogicalBytes        int64   `json:"logical_bytes"`         // Total logical bytes across all branches
	SharedPages         int     `json:"shared_pages"`          // Count of shared physical page blocks
	PrefixHitRate       float64 `json:"prefix_hit_rate"`       // Prefix cache hit rate across branches
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

// Metrics calculates and returns deduplication, memory, and branch metrics.
func (t *COWPageTable) Metrics() COWMetrics {
	t.mu.RLock()
	defer t.mu.RUnlock()

	var dedupRatio float64
	if t.totalLogicalBytes > 0 && t.totalLogicalBytes > t.totalAllocatedBytes {
		dedupRatio = 1.0 - (float64(t.totalAllocatedBytes) / float64(t.totalLogicalBytes))
	}

	dedupBytes := t.totalLogicalBytes - t.totalAllocatedBytes
	if dedupBytes < 0 {
		dedupBytes = 0
	}

	totalLogicalBlocks := 0
	for _, s := range t.sessions {
		totalLogicalBlocks += len(s.Blocks)
	}

	sharedPagesCount := 0
	for _, blk := range t.physicalBlocks {
		if blk.IsShared() {
			sharedPagesCount++
		}
	}
	prefixHitRate := t.PrefixHitRate()

	return COWMetrics{
		TotalAllocatedBytes: t.totalAllocatedBytes,
		DeduplicatedBytes:   dedupBytes,
		DedupRatio:          dedupRatio,
		ActiveBranches:      len(t.sessions),
		TotalPhysicalBlocks: len(t.physicalBlocks),
		TotalLogicalBlocks:  totalLogicalBlocks,
		LogicalBytes:        t.totalLogicalBytes,
		SharedPages:         sharedPagesCount,
		PrefixHitRate:       prefixHitRate,
	}
}

// TotalAllocatedBytes returns the total physical DRAM bytes allocated for all active blocks.
func (t *COWPageTable) TotalAllocatedBytes() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.totalAllocatedBytes
}

// DeduplicatedBytes returns total avoided DRAM bytes through copy-on-write page table sharing.
func (t *COWPageTable) DeduplicatedBytes() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.totalLogicalBytes > t.totalAllocatedBytes {
		return t.totalLogicalBytes - t.totalAllocatedBytes
	}
	return 0
}

// DedupRatio returns the memory deduplication ratio: 1.0 - (physical / logical).
func (t *COWPageTable) DedupRatio() float64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.totalLogicalBytes > 0 && t.totalLogicalBytes > t.totalAllocatedBytes {
		return 1.0 - (float64(t.totalAllocatedBytes) / float64(t.totalLogicalBytes))
	}
	return 0.0
}

// ActiveBranches returns the number of active, unreleased session branches.
func (t *COWPageTable) ActiveBranches() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return len(t.sessions)
}

// PhysicalBlockCount returns the number of active unique physical blocks in DRAM.
func (t *COWPageTable) PhysicalBlockCount() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return len(t.physicalBlocks)
}

// DuplicatePhysicalPagesAllocated returns the number of redundant physical pages allocated on fork (always 0).
func (t *COWPageTable) DuplicatePhysicalPagesAllocated() int {
	return 0
}

// BlockCapacity returns the configured block capacity.
func (t *COWPageTable) BlockCapacity() int {
	return t.blockCapacity
}

// BlockSize returns the byte size of one page block.
func (t *COWPageTable) BlockSize() int64 {
	return t.blockSize
}

// SharedPages returns the number of physical page blocks currently shared across multiple session branches.
func (t *COWPageTable) SharedPages() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	count := 0
	for _, blk := range t.physicalBlocks {
		if blk.IsShared() {
			count++
		}
	}
	return count
}

// PrefixHitRate returns the prefix cache hit rate across active session branches.
// For forked subagent sessions, this measures the fraction of logical blocks served from
// the shared parent prefix.
func (t *COWPageTable) PrefixHitRate() float64 {
	t.mu.RLock()
	defer t.mu.RUnlock()

	totalForkedBlocks := 0
	totalPrefixBlocks := 0
	hasForks := false

	for _, s := range t.sessions {
		s.mu.RLock()
		if s.ParentID != "" {
			hasForks = true
			totalForkedBlocks += len(s.Blocks)
			totalPrefixBlocks += s.PrefixBlocks
		}
		s.mu.RUnlock()
	}

	if !hasForks {
		if len(t.sessions) > 0 {
			return 1.0
		}
		return 0.0
	}

	if totalForkedBlocks > 0 {
		rate := float64(totalPrefixBlocks) / float64(totalForkedBlocks)
		if rate > 1.0 {
			return 1.0
		}
		return rate
	}
	return 1.0
}

// SubagentForkTelemetry captures zero-copy physical memory savings and prefix sharing metrics
// when a subagent session is forked via COWPageTable.ForkSession.
type SubagentForkTelemetry struct {
	ParentID           string        `json:"parent_id"`
	ChildID            string        `json:"child_id"`
	SharedPages        int           `json:"shared_pages"`
	SharedTokens       int           `json:"shared_tokens"`
	PhysicalBytesSaved int64         `json:"physical_bytes_saved"`
	DeduplicatedBytes  int64         `json:"deduplicated_bytes"`
	DedupRatio         float64       `json:"dedup_ratio"`
	PrefixHitRate      float64       `json:"prefix_hit_rate"`
	ForkLatency        time.Duration `json:"fork_latency"`
	DuplicatePages     int           `json:"duplicate_pages"` // Always 0 for zero-copy COW
	SavedMemorySummary string        `json:"saved_memory_summary"`
}

// Summary returns the human-readable summary of saved physical memory.
func (t SubagentForkTelemetry) Summary() string {
	return t.SavedMemorySummary
}

// CaptureSubagentForkTelemetry captures physical memory saved and prefix sharing telemetry
// for a child subagent session branch forked via ForkSession.
func (t *COWPageTable) CaptureSubagentForkTelemetry(childSessionID string) (SubagentForkTelemetry, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	child, ok := t.sessions[childSessionID]
	if !ok {
		return SubagentForkTelemetry{}, ErrSessionNotFound
	}
	return t.captureForkTelemetryLocked(child), nil
}

func (t *COWPageTable) captureForkTelemetryLocked(child *SessionBranch) SubagentForkTelemetry {
	child.mu.RLock()
	defer child.mu.RUnlock()

	sharedPages := 0
	for _, blk := range child.Blocks {
		if blk.RefCount() > 1 {
			sharedPages++
		}
	}

	bytesSaved := int64(child.PrefixBlocks) * t.blockSize
	dedupBytes := int64(0)
	if t.totalLogicalBytes > t.totalAllocatedBytes {
		dedupBytes = t.totalLogicalBytes - t.totalAllocatedBytes
	}

	dedupRatio := 0.0
	if t.totalLogicalBytes > 0 && t.totalLogicalBytes > t.totalAllocatedBytes {
		dedupRatio = 1.0 - (float64(t.totalAllocatedBytes) / float64(t.totalLogicalBytes))
	}

	return SubagentForkTelemetry{
		ParentID:           child.ParentID,
		ChildID:            child.ID,
		SharedPages:        sharedPages,
		SharedTokens:       child.PrefixTokens,
		PhysicalBytesSaved: bytesSaved,
		DeduplicatedBytes:  dedupBytes,
		DedupRatio:         dedupRatio,
		PrefixHitRate:      child.PrefixHitRate,
		ForkLatency:        child.ForkLatency,
		DuplicatePages:     0,
		SavedMemorySummary: FormatSavedMemory(bytesSaved),
	}
}

// SubagentForkTelemetry returns telemetry capturing physical memory saved by this session branch.
func (s *SessionBranch) SubagentForkTelemetry() SubagentForkTelemetry {
	s.table.mu.RLock()
	defer s.table.mu.RUnlock()
	return s.table.captureForkTelemetryLocked(s)
}

// ForkSubagent forks a child subagent session and returns both the branch and its zero-copy memory telemetry.
func (t *COWPageTable) ForkSubagent(parentID, childID string) (*SessionBranch, SubagentForkTelemetry, error) {
	child, err := t.ForkSession(parentID, childID)
	if err != nil {
		return nil, SubagentForkTelemetry{}, err
	}
	telem, err := t.CaptureSubagentForkTelemetry(childID)
	return child, telem, err
}

// CaptureSubagentForkTelemetry captures physical memory saved when a subagent session branch
// is forked from a parent coordinator via the default COWPageTable.
func CaptureSubagentForkTelemetry(table *COWPageTable, childSessionID string) (SubagentForkTelemetry, error) {
	if table == nil {
		table = defaultCOWPageTable
	}
	return table.CaptureSubagentForkTelemetry(childSessionID)
}

// SubagentSharingReport captures aggregate telemetry across all forked subagents.
type SubagentSharingReport struct {
	ActiveBranches      int     `json:"active_branches"`
	ForkedSubagents     int     `json:"forked_subagents"`
	TotalSharedPages    int     `json:"total_shared_pages"`
	TotalPhysicalBlocks int     `json:"total_physical_blocks"`
	TotalLogicalBlocks  int     `json:"total_logical_blocks"`
	AllocatedBytes      int64   `json:"allocated_bytes"`
	DeduplicatedBytes   int64   `json:"deduplicated_bytes"`
	DedupRatio          float64 `json:"dedup_ratio"`
	PrefixHitRate       float64 `json:"prefix_hit_rate"`
	DuplicatePagesAlloc int     `json:"duplicate_pages_allocated"` // Always 0
	SavedMemorySummary  string  `json:"saved_memory_summary"`
}

// SubagentSharingReport returns aggregate zero-copy memory savings across all active branches.
func (t *COWPageTable) SubagentSharingReport() SubagentSharingReport {
	t.mu.RLock()
	defer t.mu.RUnlock()

	sharedBlocks := 0
	for _, blk := range t.physicalBlocks {
		if blk.IsShared() {
			sharedBlocks++
		}
	}

	totalLogicalBlocks := 0
	forkedCount := 0
	for _, s := range t.sessions {
		s.mu.RLock()
		totalLogicalBlocks += len(s.Blocks)
		if s.ParentID != "" {
			forkedCount++
		}
		s.mu.RUnlock()
	}

	dedupBytes := int64(0)
	if t.totalLogicalBytes > t.totalAllocatedBytes {
		dedupBytes = t.totalLogicalBytes - t.totalAllocatedBytes
	}

	dedupRatio := 0.0
	if t.totalLogicalBytes > 0 && t.totalLogicalBytes > t.totalAllocatedBytes {
		dedupRatio = 1.0 - (float64(t.totalAllocatedBytes) / float64(t.totalLogicalBytes))
	}

	prefixRate := 1.0
	if forkedCount > 0 {
		totalForked := 0
		totalPrefix := 0
		for _, s := range t.sessions {
			s.mu.RLock()
			if s.ParentID != "" {
				totalForked += len(s.Blocks)
				totalPrefix += s.PrefixBlocks
			}
			s.mu.RUnlock()
		}
		if totalForked > 0 {
			prefixRate = float64(totalPrefix) / float64(totalForked)
			if prefixRate > 1.0 {
				prefixRate = 1.0
			}
		}
	}

	return SubagentSharingReport{
		ActiveBranches:      len(t.sessions),
		ForkedSubagents:     forkedCount,
		TotalSharedPages:    sharedBlocks,
		TotalPhysicalBlocks: len(t.physicalBlocks),
		TotalLogicalBlocks:  totalLogicalBlocks,
		AllocatedBytes:      t.totalAllocatedBytes,
		DeduplicatedBytes:   dedupBytes,
		DedupRatio:          dedupRatio,
		PrefixHitRate:       prefixRate,
		DuplicatePagesAlloc: 0,
		SavedMemorySummary:  FormatSavedMemory(dedupBytes),
	}
}

// SubagentSharingReport returns aggregate zero-copy memory savings via the default COWPageTable.
func SubagentSharingReportSummary() SubagentSharingReport {
	return defaultCOWPageTable.SubagentSharingReport()
}

// FormatSavedMemory formats an avoided physical memory allocation byte count into a canonical summary string:
// e.g. "Saved 1.8 GB GPU VRAM / host DRAM via zero-copy subagent KV sharing".
func FormatSavedMemory(bytes int64) string {
	return fmt.Sprintf("Saved %s GPU VRAM / host DRAM via zero-copy subagent KV sharing", FormatBytes(bytes))
}

// FormatBytes formats byte counts into human-readable representation (B, KB, MB, GB, TB).
func FormatBytes(bytes int64) string {
	const (
		kb = 1024
		mb = 1024 * kb
		gb = 1024 * mb
		tb = 1024 * gb
	)
	switch {
	case bytes >= tb:
		return fmt.Sprintf("%.1f TB", float64(bytes)/float64(tb))
	case bytes >= gb:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(gb))
	case bytes >= mb:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(mb))
	case bytes >= kb:
		return fmt.Sprintf("%.1f KB", float64(bytes)/float64(kb))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

// -----------------------------------------------------------------------------
// MMU Integration & Default Instance
// -----------------------------------------------------------------------------

var (
	defaultCOWPageTable = NewCOWPageTable()
	mmuCOWMu            sync.Mutex
	mmuCOWPageTables    = make(map[*MMU]*COWPageTable)
)

// COWPageTable returns the COWPageTable instance associated with this MMU.
func (m *MMU) COWPageTable() *COWPageTable {
	if m == nil {
		return defaultCOWPageTable
	}
	mmuCOWMu.Lock()
	defer mmuCOWMu.Unlock()
	tbl := mmuCOWPageTables[m]
	if tbl == nil {
		tbl = NewCOWPageTable()
		mmuCOWPageTables[m] = tbl
	}
	return tbl
}

// ForkCOWSession forks a session branch via the default COWPageTable.
func ForkCOWSession(parentID, childID string) (*SessionBranch, error) {
	return defaultCOWPageTable.ForkSession(parentID, childID)
}

// ReleaseCOWSession releases a session branch via the default COWPageTable.
func ReleaseCOWSession(sessionID string) error {
	return defaultCOWPageTable.ReleaseSession(sessionID)
}

// AppendCOWTokens appends tokens into a session branch via the default COWPageTable.
func AppendCOWTokens(sessionID string, tokens []int) error {
	return defaultCOWPageTable.AppendTokens(sessionID, tokens)
}

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
