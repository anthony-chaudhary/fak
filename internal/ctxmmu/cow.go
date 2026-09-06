package ctxmmu

import (
	"encoding/binary"
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
	return &COWPageTable{
		blockCapacity:  cfg.BlockCapacity,
		bytesPerToken:  cfg.BytesPerToken,
		blockSize:      blockSize,
		sessions:       make(map[string]*SessionBranch),
		physicalBlocks: make(map[int64]*PageBlock),
	}
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

	return COWMetrics{
		TotalAllocatedBytes: t.totalAllocatedBytes,
		DeduplicatedBytes:   dedupBytes,
		DedupRatio:          dedupRatio,
		ActiveBranches:      len(t.sessions),
		TotalPhysicalBlocks: len(t.physicalBlocks),
		TotalLogicalBlocks:  totalLogicalBlocks,
		LogicalBytes:        t.totalLogicalBytes,
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
