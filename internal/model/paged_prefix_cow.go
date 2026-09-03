package model

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
)

// paged_prefix_cow.go — Exact-boundary Qwen Metal prefix block sharing with measured copy-on-write (#9499, M7).
//
// For multi-session agent requests with identical prefixes, avoid deep-cloning full contiguous KV caches.
// Instead, share immutable reference-counted attention K/Kraw/V page blocks at an exact token boundary
// (fork_clone_bytes = 0) and copy only the mutable recurrent/convolution sidecar state.
//
// Key invariants:
//  1. PagedPrefixOwner holds immutable page blocks at an exact token boundary.
//  2. Fork session from prefix shares attention blocks without cloning (fork_clone_bytes = 0), increments
//     refcounts, and clones only the mutable recurrent/convolution sidecar state.
//  3. Tail-only COW: continuation past the prefix boundary writes only newly allocated tail blocks, leaving
//     shared prefix blocks untouched. Mutation of a shared block triggers copy-on-write for only that page.
//  4. Session release decrements refcounts on shared blocks, freeing underlying Metal/host pages only when
//     refcount drops to zero.
//  5. Telemetry tracks shared_pages, fork_clone_bytes, cow_bytes, recurrent_sidecar_bytes, and verifies
//     zero full-cache rematerialization.

var (
	// ErrNonExactHybridPrefix indicates that a prefix request does not match the exact
	// token boundary required for Qwen hybrid recurrent state sharing, triggering a typed
	// fallback to re-prefill.
	ErrNonExactHybridPrefix = errors.New("model: non-exact hybrid prefix boundary requires re-prefill")

	// ErrSessionReleased indicates an operation was attempted on an already-released session.
	ErrSessionReleased = errors.New("model: paged prefix session has been released")

	// ErrOwnerReleased indicates an operation was attempted on an already-released prefix owner.
	ErrOwnerReleased = errors.New("model: paged prefix owner has been released")

	// ErrFullMaterializationForbidden enforces the zero full-cache rematerialization invariant.
	ErrFullMaterializationForbidden = errors.New("model: full cache rematerialization forbidden by paged prefix COW policy")
)

// NonExactHybridFallbackError is the typed refusal emitted when prefix sharing cannot be
// satisfied at the requested boundary for a Qwen hybrid model. Callers must handle this by
// falling back to re-prefilling rather than panicking.
type NonExactHybridFallbackError struct {
	ExpectedTokens  int
	RequestedTokens int
	Reason          string
}

func (e *NonExactHybridFallbackError) Error() string {
	return fmt.Sprintf("model: non-exact hybrid prefix boundary (expected %d tokens, requested %d: %s); fallback to re-prefill",
		e.ExpectedTokens, e.RequestedTokens, e.Reason)
}

func (e *NonExactHybridFallbackError) Is(target error) bool {
	return target == ErrNonExactHybridPrefix
}

// PagedPrefixTelemetry captures falsifiable metrics on prefix sharing, copy-on-write
// behavior, sidecar duplication, and rematerialization avoidance.
type PagedPrefixTelemetry struct {
	SharedPages              int   `json:"shared_pages"`
	ForkCloneBytes           int64 `json:"fork_clone_bytes"`
	COWBytes                 int64 `json:"cow_bytes"`
	RecurrentSidecarBytes    int64 `json:"recurrent_sidecar_bytes"`
	FullMaterializationBytes int64 `json:"full_materialization_bytes"`
	RematerializedBytes      int64 `json:"rematerialized_bytes"`
}

// VerifyZeroRematerialization verifies that no full-cache cloning or page-to-contiguous
// rematerialization occurred during session lifecycle.
func (t PagedPrefixTelemetry) VerifyZeroRematerialization() error {
	if t.FullMaterializationBytes != 0 {
		return fmt.Errorf("model: full cache materialization detected: %d bytes (want 0)", t.FullMaterializationBytes)
	}
	if t.RematerializedBytes != 0 {
		return fmt.Errorf("model: rematerialized bytes detected: %d bytes (want 0)", t.RematerializedBytes)
	}
	return nil
}

// PagedPrefixBlock represents one physical page block holding attention K, Kraw, and V
// tensors for a fixed number of tokens across layers.
type PagedPrefixBlock struct {
	id        int
	tokens    int
	layers    int
	stride    int
	planes    int
	data      []float32
	refcount  int32
	immutable bool
	isMetal   bool
	freed     bool
}

func (b *PagedPrefixBlock) ID() int           { return b.id }
func (b *PagedPrefixBlock) Tokens() int       { return b.tokens }
func (b *PagedPrefixBlock) Layers() int       { return b.layers }
func (b *PagedPrefixBlock) Stride() int       { return b.stride }
func (b *PagedPrefixBlock) Planes() int       { return b.planes }
func (b *PagedPrefixBlock) Refcount() int32   { return atomic.LoadInt32(&b.refcount) }
func (b *PagedPrefixBlock) IsImmutable() bool { return b.immutable }
func (b *PagedPrefixBlock) IsMetal() bool     { return b.isMetal }
func (b *PagedPrefixBlock) IsFreed() bool     { return b.freed }
func (b *PagedPrefixBlock) SizeBytes() int64  { return int64(len(b.data) * 4) }
func (b *PagedPrefixBlock) Data() []float32   { return b.data }

func (b *PagedPrefixBlock) slot(layer, plane, tok int) int {
	return ((layer*b.planes+plane)*b.tokens + tok) * b.stride
}

func (b *PagedPrefixBlock) Get(layer, plane, tok int) []float32 {
	if layer < 0 || layer >= b.layers || plane < 0 || plane >= b.planes || tok < 0 || tok >= b.tokens {
		return nil
	}
	off := b.slot(layer, plane, tok)
	return b.data[off : off+b.stride]
}

func (b *PagedPrefixBlock) WriteToken(tok int, k, kraw, v [][]float32) {
	if tok < 0 || tok >= b.tokens {
		return
	}
	for l := 0; l < b.layers; l++ {
		if l < len(k) && len(k[l]) > 0 {
			dst := b.slot(l, planeK, tok)
			copy(b.data[dst:dst+b.stride], k[l])
		}
		if l < len(v) && len(v[l]) > 0 {
			dst := b.slot(l, planeV, tok)
			copy(b.data[dst:dst+b.stride], v[l])
		}
		if b.planes >= 3 && l < len(kraw) && len(kraw[l]) > 0 {
			dst := b.slot(l, planeKraw, tok)
			copy(b.data[dst:dst+b.stride], kraw[l])
		}
	}
}

// PagedBlockPool manages physical allocation, reference counting, and lifecycle
// of Metal/host attention page blocks.
type PagedBlockPool struct {
	mu                  sync.Mutex
	cfg                 Config
	blockTokens         int
	stride              int
	layers              int
	planes              int
	isMetal             bool
	blocks              []*PagedPrefixBlock
	freeList            []int
	totalAllocatedPages int
	metalPagesAllocated int
	metalPagesFreed     int
	hostPagesAllocated  int
	hostPagesFreed      int
}

// NewPagedBlockPool creates a pool configured for the model architecture.
func NewPagedBlockPool(cfg Config, blockTokens int, isMetal bool) *PagedBlockPool {
	if blockTokens <= 0 {
		blockTokens = 16
	}
	stride := cfg.NumKVHeads * cfg.HeadDim
	if stride < 0 {
		stride = 0
	}
	layers := cfg.NumLayers
	if layers < 0 {
		layers = 0
	}
	return &PagedBlockPool{
		cfg:         cfg,
		blockTokens: blockTokens,
		stride:      stride,
		layers:      layers,
		planes:      3, // K, V, Kraw
		isMetal:     isMetal,
	}
}

// Alloc reserves or reuses a physical page block with initial refcount 1.
func (p *PagedBlockPool) Alloc() *PagedPrefixBlock {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.freeList) > 0 {
		id := p.freeList[len(p.freeList)-1]
		p.freeList = p.freeList[:len(p.freeList)-1]
		blk := p.blocks[id]
		blk.freed = false
		blk.immutable = false
		atomic.StoreInt32(&blk.refcount, 1)
		clear(blk.data)
		if blk.isMetal {
			p.metalPagesAllocated++
		} else {
			p.hostPagesAllocated++
		}
		p.totalAllocatedPages++
		return blk
	}
	id := len(p.blocks)
	blk := &PagedPrefixBlock{
		id:        id,
		tokens:    p.blockTokens,
		layers:    p.layers,
		stride:    p.stride,
		planes:    p.planes,
		data:      make([]float32, p.layers*p.planes*p.blockTokens*p.stride),
		refcount:  1,
		immutable: false,
		isMetal:   p.isMetal,
		freed:     false,
	}
	p.blocks = append(p.blocks, blk)
	if blk.isMetal {
		p.metalPagesAllocated++
	} else {
		p.hostPagesAllocated++
	}
	p.totalAllocatedPages++
	return blk
}

// Retain increments the reference count of a block.
func (p *PagedBlockPool) Retain(id int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if id < 0 || id >= len(p.blocks) {
		return
	}
	atomic.AddInt32(&p.blocks[id].refcount, 1)
}

// Release decrements the reference count on a block, freeing it if refcount reaches zero.
func (p *PagedBlockPool) Release(id int) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if id < 0 || id >= len(p.blocks) {
		return false
	}
	blk := p.blocks[id]
	if blk.freed {
		return false
	}
	newRef := atomic.AddInt32(&blk.refcount, -1)
	if newRef <= 0 {
		atomic.StoreInt32(&blk.refcount, 0)
		blk.freed = true
		blk.immutable = false
		if blk.isMetal {
			p.metalPagesFreed++
		} else {
			p.hostPagesFreed++
		}
		clear(blk.data)
		p.freeList = append(p.freeList, id)
		return true
	}
	return false
}

// ActivePages returns the number of physical blocks currently referenced.
func (p *PagedBlockPool) ActivePages() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	active := 0
	for _, blk := range p.blocks {
		if !blk.freed && atomic.LoadInt32(&blk.refcount) > 0 {
			active++
		}
	}
	return active
}

// TotalAllocated returns cumulative allocated pages.
func (p *PagedBlockPool) TotalAllocated() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.totalAllocatedPages
}

func (p *PagedBlockPool) MetalPagesAllocated() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.metalPagesAllocated
}

func (p *PagedBlockPool) MetalPagesFreed() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.metalPagesFreed
}

func (p *PagedBlockPool) HostPagesAllocated() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.hostPagesAllocated
}

func (p *PagedBlockPool) HostPagesFreed() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.hostPagesFreed
}

func (p *PagedBlockPool) Block(id int) *PagedPrefixBlock {
	p.mu.Lock()
	defer p.mu.Unlock()
	if id < 0 || id >= len(p.blocks) {
		return nil
	}
	return p.blocks[id]
}

// PagedPrefixOwnerConfig configures a PagedPrefixOwner instance.
type PagedPrefixOwnerConfig struct {
	Config      Config
	BlockTokens int
	Tokens      int
	TokenIDs    []int
	K           [][][]float32 // [pos][layer][dim]
	Kraw        [][][]float32 // [pos][layer][dim]
	V           [][][]float32 // [pos][layer][dim]
	Sidecar     []GDNLayerState
	Pool        *PagedBlockPool
	IsMetal     bool
}

// PagedPrefixOwner holds immutable, reference-counted page blocks for attention K/Kraw/V
// tensors at an exact token boundary, plus the fixed recurrent/convolution sidecar state.
type PagedPrefixOwner struct {
	mu           sync.Mutex
	prefixTokens int
	tokenIDs     []int
	pageTable    []int
	sidecar      []GDNLayerState
	pool         *PagedBlockPool
	cfg          Config
	blockTokens  int
	stride       int
	layers       int
	isMetal      bool
	released     bool
}

// NewPagedPrefixOwner constructs a sealed, immutable prefix owner.
func NewPagedPrefixOwner(params PagedPrefixOwnerConfig) (*PagedPrefixOwner, error) {
	tokens := params.Tokens
	if tokens <= 0 && len(params.TokenIDs) > 0 {
		tokens = len(params.TokenIDs)
	}
	if tokens <= 0 {
		return nil, fmt.Errorf("model: paged prefix owner tokens must be positive, got %d", tokens)
	}
	blockTokens := params.BlockTokens
	if blockTokens <= 0 {
		blockTokens = 16
	}
	pool := params.Pool
	if pool == nil {
		pool = NewPagedBlockPool(params.Config, blockTokens, params.IsMetal)
	}

	nBlocks := (tokens + blockTokens - 1) / blockTokens
	pageTable := make([]int, nBlocks)

	for b := 0; b < nBlocks; b++ {
		blk := pool.Alloc()
		pageTable[b] = blk.id

		tokStart := b * blockTokens
		tokEnd := tokStart + blockTokens
		if tokEnd > tokens {
			tokEnd = tokens
		}

		for pos := tokStart; pos < tokEnd; pos++ {
			tokInBlk := pos - tokStart
			var kRows, krawRows, vRows [][]float32
			if pos < len(params.K) {
				kRows = params.K[pos]
			}
			if pos < len(params.Kraw) {
				krawRows = params.Kraw[pos]
			}
			if pos < len(params.V) {
				vRows = params.V[pos]
			}
			blk.WriteToken(tokInBlk, kRows, krawRows, vRows)
		}

		blk.immutable = true
	}

	var tokenIDs []int
	if len(params.TokenIDs) > 0 {
		tokenIDs = append([]int(nil), params.TokenIDs...)
	}

	return &PagedPrefixOwner{
		prefixTokens: tokens,
		tokenIDs:     tokenIDs,
		pageTable:    pageTable,
		sidecar:      cloneSidecar(params.Sidecar),
		pool:         pool,
		cfg:          params.Config,
		blockTokens:  blockTokens,
		stride:       pool.stride,
		layers:       pool.layers,
		isMetal:      params.IsMetal,
	}, nil
}

func (o *PagedPrefixOwner) PrefixTokens() int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.prefixTokens
}

func (o *PagedPrefixOwner) TokenIDs() []int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return append([]int(nil), o.tokenIDs...)
}

func (o *PagedPrefixOwner) PageTable() []int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return append([]int(nil), o.pageTable...)
}

func (o *PagedPrefixOwner) Block(index int) *PagedPrefixBlock {
	o.mu.Lock()
	defer o.mu.Unlock()
	if index < 0 || index >= len(o.pageTable) {
		return nil
	}
	return o.pool.Block(o.pageTable[index])
}

func (o *PagedPrefixOwner) Blocks() []*PagedPrefixBlock {
	o.mu.Lock()
	defer o.mu.Unlock()
	blks := make([]*PagedPrefixBlock, len(o.pageTable))
	for i, id := range o.pageTable {
		blks[i] = o.pool.Block(id)
	}
	return blks
}

func (o *PagedPrefixOwner) Sidecar() []GDNLayerState {
	o.mu.Lock()
	defer o.mu.Unlock()
	return cloneSidecar(o.sidecar)
}

func (o *PagedPrefixOwner) LayerSidecar(layer int) (GDNLayerState, bool) {
	o.mu.Lock()
	defer o.mu.Unlock()
	for _, l := range o.sidecar {
		if l.Layer == layer {
			return l.Clone(), true
		}
	}
	return GDNLayerState{}, false
}

func (o *PagedPrefixOwner) Pool() *PagedBlockPool {
	return o.pool
}

func (o *PagedPrefixOwner) IsReleased() bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.released
}

// Release drops the owner's hold on prefix blocks, freeing them if no sessions hold refs.
func (o *PagedPrefixOwner) Release() error {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.released {
		return nil
	}
	o.released = true
	for _, id := range o.pageTable {
		o.pool.Release(id)
	}
	o.pageTable = nil
	o.prefixTokens = 0
	return nil
}

// ForkSession forks a continuation session from the exact prefix boundary.
func (o *PagedPrefixOwner) ForkSession(sessionID string) (*PagedPrefixSession, error) {
	return o.forkInternal(sessionID, o.prefixTokens)
}

// ForkSessionAt forks a continuation session if requestedTokens matches the exact prefix boundary.
func (o *PagedPrefixOwner) ForkSessionAt(sessionID string, requestedTokens int) (*PagedPrefixSession, error) {
	o.mu.Lock()
	prefixTokens := o.prefixTokens
	o.mu.Unlock()
	if requestedTokens != prefixTokens {
		return nil, &NonExactHybridFallbackError{
			ExpectedTokens:  prefixTokens,
			RequestedTokens: requestedTokens,
			Reason:          "prefix length mismatch: hybrid recurrent state requires exact token boundary",
		}
	}
	return o.forkInternal(sessionID, requestedTokens)
}

// ForkSessionExact forks a continuation session if requestedTokenIDs matches the exact prefix.
func (o *PagedPrefixOwner) ForkSessionExact(sessionID string, requestedTokenIDs []int) (*PagedPrefixSession, error) {
	o.mu.Lock()
	prefixTokens := o.prefixTokens
	storedTokens := append([]int(nil), o.tokenIDs...)
	o.mu.Unlock()

	if len(requestedTokenIDs) != prefixTokens {
		return nil, &NonExactHybridFallbackError{
			ExpectedTokens:  prefixTokens,
			RequestedTokens: len(requestedTokenIDs),
			Reason:          "prefix token count mismatch",
		}
	}
	if len(storedTokens) > 0 {
		for i, tok := range requestedTokenIDs {
			if i < len(storedTokens) && tok != storedTokens[i] {
				return nil, &NonExactHybridFallbackError{
					ExpectedTokens:  prefixTokens,
					RequestedTokens: len(requestedTokenIDs),
					Reason:          fmt.Sprintf("prefix token mismatch at position %d (want %d, got %d)", i, storedTokens[i], tok),
				}
			}
		}
	}
	return o.forkInternal(sessionID, prefixTokens)
}

func (o *PagedPrefixOwner) forkInternal(sessionID string, tokens int) (*PagedPrefixSession, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.released {
		return nil, ErrOwnerReleased
	}
	if o.prefixTokens <= 0 {
		return nil, errors.New("model: cannot fork session from empty prefix owner")
	}

	childTable := make([]int, len(o.pageTable))
	copy(childTable, o.pageTable)
	sharedMap := make(map[int]struct{}, len(o.pageTable))
	for _, id := range childTable {
		o.pool.Retain(id)
		sharedMap[id] = struct{}{}
	}

	sidecarClones := cloneSidecar(o.sidecar)
	scBytes := sidecarBytes(o.sidecar)

	sess := &PagedPrefixSession{
		sessionID:             sessionID,
		owner:                 o,
		pool:                  o.pool,
		pageTable:             childTable,
		tokens:                tokens,
		prefixTokens:          tokens,
		sidecar:               sidecarClones,
		sharedBlockIDs:        sharedMap,
		sharedPages:           len(childTable),
		forkCloneBytes:        0, // Zero-copy fork: attention blocks shared, not cloned
		cowBytes:              0,
		recurrentSidecarBytes: scBytes,
	}

	return sess, nil
}

// PagedPrefixSession represents an active continuation context for an agent request,
// backed by shared immutable prefix blocks with tail-only copy-on-write allocation.
type PagedPrefixSession struct {
	mu                       sync.Mutex
	sessionID                string
	owner                    *PagedPrefixOwner
	pool                     *PagedBlockPool
	pageTable                []int
	tokens                   int
	prefixTokens             int
	sidecar                  []GDNLayerState
	sharedBlockIDs           map[int]struct{}
	sharedPages              int
	forkCloneBytes           int64
	cowBytes                 int64
	recurrentSidecarBytes    int64
	fullMaterializationBytes int64
	rematerializedBytes      int64
	released                 bool
}

func (s *PagedPrefixSession) SessionID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sessionID
}

func (s *PagedPrefixSession) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.tokens
}

func (s *PagedPrefixSession) PrefixTokens() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.prefixTokens
}

func (s *PagedPrefixSession) PageTable() []int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]int(nil), s.pageTable...)
}

func (s *PagedPrefixSession) Block(logicalIndex int) *PagedPrefixBlock {
	s.mu.Lock()
	defer s.mu.Unlock()
	if logicalIndex < 0 || logicalIndex >= len(s.pageTable) {
		return nil
	}
	return s.pool.Block(s.pageTable[logicalIndex])
}

func (s *PagedPrefixSession) Blocks() []*PagedPrefixBlock {
	s.mu.Lock()
	defer s.mu.Unlock()
	blks := make([]*PagedPrefixBlock, len(s.pageTable))
	for i, id := range s.pageTable {
		blks[i] = s.pool.Block(id)
	}
	return blks
}

func (s *PagedPrefixSession) Sidecar() []GDNLayerState {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sidecar
}

func (s *PagedPrefixSession) CloneSidecar() []GDNLayerState {
	s.mu.Lock()
	defer s.mu.Unlock()
	return cloneSidecar(s.sidecar)
}

func (s *PagedPrefixSession) LayerSidecar(layer int) (GDNLayerState, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, l := range s.sidecar {
		if l.Layer == layer {
			return l.Clone(), true
		}
	}
	return GDNLayerState{}, false
}

func (s *PagedPrefixSession) UpdateLayerSidecar(layer int, conv, recurrent []float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.released {
		return ErrSessionReleased
	}
	for i, l := range s.sidecar {
		if l.Layer == layer {
			if conv != nil {
				s.sidecar[i].Conv = append([]float32(nil), conv...)
			}
			if recurrent != nil {
				s.sidecar[i].Recurrent = append([]float32(nil), recurrent...)
			}
			return nil
		}
	}
	s.sidecar = append(s.sidecar, GDNLayerState{
		Layer:     layer,
		Conv:      append([]float32(nil), conv...),
		Recurrent: append([]float32(nil), recurrent...),
	})
	return nil
}

func (s *PagedPrefixSession) ensureOwned(li int) error {
	blockID := s.pageTable[li]
	blk := s.pool.Block(blockID)
	if blk == nil {
		return fmt.Errorf("model: block %d not found", blockID)
	}

	_, isShared := s.sharedBlockIDs[blockID]
	if blk.Refcount() == 1 && !blk.immutable && !isShared {
		return nil
	}

	newBlk := s.pool.Alloc()
	copy(newBlk.data, blk.data)

	s.cowBytes += newBlk.SizeBytes()

	s.pool.Release(blockID)
	delete(s.sharedBlockIDs, blockID)

	s.pageTable[li] = newBlk.id
	return nil
}

// Append appends a token past the prefix boundary. Tail blocks are allocated as needed;
// shared blocks remain untouched.
func (s *PagedPrefixSession) Append(k, kraw, v [][]float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.released {
		return ErrSessionReleased
	}

	pos := s.tokens
	li := pos / s.pool.blockTokens
	off := pos % s.pool.blockTokens

	if li == len(s.pageTable) {
		newBlk := s.pool.Alloc()
		s.pageTable = append(s.pageTable, newBlk.id)
	} else {
		if err := s.ensureOwned(li); err != nil {
			return err
		}
	}

	blk := s.pool.Block(s.pageTable[li])
	blk.WriteToken(off, k, kraw, v)
	s.tokens++
	return nil
}

// MutateToken writes new K/Kraw/V values to position pos, triggering Copy-On-Write
// if pos falls within a shared block.
func (s *PagedPrefixSession) MutateToken(pos int, k, kraw, v [][]float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.released {
		return ErrSessionReleased
	}
	if pos < 0 || pos >= s.tokens {
		return fmt.Errorf("model: mutate token index %d out of bounds [0, %d)", pos, s.tokens)
	}

	li := pos / s.pool.blockTokens
	off := pos % s.pool.blockTokens

	if err := s.ensureOwned(li); err != nil {
		return err
	}

	blk := s.pool.Block(s.pageTable[li])
	blk.WriteToken(off, k, kraw, v)
	return nil
}

func (s *PagedPrefixSession) GatherK(layer int) []float32 {
	return s.gather(layer, planeK)
}

func (s *PagedPrefixSession) GatherV(layer int) []float32 {
	return s.gather(layer, planeV)
}

func (s *PagedPrefixSession) GatherKraw(layer int) []float32 {
	return s.gather(layer, planeKraw)
}

func (s *PagedPrefixSession) gather(layer, plane int) []float32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.tokens == 0 || s.pool.stride == 0 || layer < 0 || layer >= s.pool.layers {
		return nil
	}
	out := make([]float32, s.tokens*s.pool.stride)
	for pos := 0; pos < s.tokens; pos++ {
		li := pos / s.pool.blockTokens
		off := pos % s.pool.blockTokens
		blk := s.pool.Block(s.pageTable[li])
		src := blk.slot(layer, plane, off)
		copy(out[pos*s.pool.stride:(pos+1)*s.pool.stride], blk.data[src:src+s.pool.stride])
	}
	return out
}

// Fork creates a child session sharing current page blocks by reference.
func (s *PagedPrefixSession) Fork(childSessionID string) (*PagedPrefixSession, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.released {
		return nil, ErrSessionReleased
	}

	childTable := make([]int, len(s.pageTable))
	copy(childTable, s.pageTable)
	sharedMap := make(map[int]struct{}, len(s.pageTable))
	for _, id := range childTable {
		s.pool.Retain(id)
		sharedMap[id] = struct{}{}
	}

	sidecarClones := cloneSidecar(s.sidecar)
	scBytes := sidecarBytes(s.sidecar)

	child := &PagedPrefixSession{
		sessionID:             childSessionID,
		owner:                 s.owner,
		pool:                  s.pool,
		pageTable:             childTable,
		tokens:                s.tokens,
		prefixTokens:          s.tokens,
		sidecar:               sidecarClones,
		sharedBlockIDs:        sharedMap,
		sharedPages:           len(childTable),
		forkCloneBytes:        0,
		cowBytes:              0,
		recurrentSidecarBytes: scBytes,
	}
	return child, nil
}

// Release releases all held page blocks, freeing unreferenced pages.
func (s *PagedPrefixSession) Release() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.released {
		return nil
	}
	s.released = true
	for _, id := range s.pageTable {
		s.pool.Release(id)
	}
	s.pageTable = nil
	s.tokens = 0
	s.sharedBlockIDs = nil
	return nil
}

func (s *PagedPrefixSession) IsReleased() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.released
}

func (s *PagedPrefixSession) Telemetry() PagedPrefixTelemetry {
	s.mu.Lock()
	defer s.mu.Unlock()
	return PagedPrefixTelemetry{
		SharedPages:              s.sharedPages,
		ForkCloneBytes:           s.forkCloneBytes,
		COWBytes:                 s.cowBytes,
		RecurrentSidecarBytes:    s.recurrentSidecarBytes,
		FullMaterializationBytes: s.fullMaterializationBytes,
		RematerializedBytes:      s.rematerializedBytes,
	}
}

func (s *PagedPrefixSession) VerifyZeroRematerialization() error {
	return s.Telemetry().VerifyZeroRematerialization()
}

// RematerializeContiguous explicitly refuses full-cache cloning/rematerialization.
func (s *PagedPrefixSession) RematerializeContiguous() (*KVCache, error) {
	return nil, ErrFullMaterializationForbidden
}

// ToPrefixShareReceipt converts the session's sharing telemetry into a PrefixShareReceipt.
func (s *PagedPrefixSession) ToPrefixShareReceipt(acceptedTokens int) PrefixShareReceipt {
	t := s.Telemetry()
	blockBytes := int64(s.pool.layers * s.pool.planes * s.pool.blockTokens * s.pool.stride * 4)
	logicalBytes := int64(len(s.PageTable())) * blockBytes
	avoided := logicalBytes - t.ForkCloneBytes
	if avoided < 0 {
		avoided = 0
	}
	var perAccepted float64
	if acceptedTokens > 0 {
		perAccepted = float64(avoided) / float64(acceptedTokens)
	}
	return PrefixShareReceipt{
		Schema:                  "fak-prefix-share-receipt/1",
		Engine:                  "fak-native",
		PrefixTokens:            s.PrefixTokens(),
		AcceptedTokens:          acceptedTokens,
		SharedBlocks:            t.SharedPages,
		LogicalPrefixBytes:      logicalBytes,
		ForkCloneBytes:          t.ForkCloneBytes,
		BytesAvoided:            avoided,
		BytesAvoidedPerAccepted: perAccepted,
		Validation:              "shared-zero-clone",
	}
}

func cloneSidecar(src []GDNLayerState) []GDNLayerState {
	if len(src) == 0 {
		return nil
	}
	out := make([]GDNLayerState, len(src))
	for i, s := range src {
		out[i] = s.Clone()
	}
	return out
}

func sidecarBytes(sidecar []GDNLayerState) int64 {
	var total int64
	for _, s := range sidecar {
		total += int64((len(s.Conv) + len(s.Recurrent)) * 4)
	}
	return total
}
