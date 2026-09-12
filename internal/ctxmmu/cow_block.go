package ctxmmu

import (
	"sync"
	"sync/atomic"
	"time"
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
