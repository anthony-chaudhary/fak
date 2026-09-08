package ctxmmu

import (
	"encoding/binary"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// kv_pool.go — Model-agnostic decoupled unified memory KV cache pooling and paging substrate (#12191).
//
// Modern agent inference architectures suffer 60–80% internal memory fragmentation when KV caches
// are statically pre-allocated to maximum sequence contexts (e.g. 32k–128k tokens). This static
// over-allocation bounds multi-agent serving throughput and prevents prompt prefix sharing.
//
// KVPool implements a virtual memory manager that decouples logical sequence token positions from
// non-contiguous physical memory pages in unified memory. It supports:
//   1. Model-agnostic KV geometries across dense MHA/GQA (LLaMA, Mistral, Qwen) and latent attention
//      topologies (DeepSeek Multi-head Latent Attention - MLA).
//   2. Lock-free O(1) physical page allocation and deallocation with bitmask free-list tracking.
//   3. Zero-copy Copy-on-Write (CoW) prompt prefix branching across agent sessions.
//   4. Context MMU memory pressure triggers and LRU page eviction hooks.

// Standard typed errors for KVPool operations.
var (
	// ErrSequenceNotFound is returned when a requested sequence context ID does not exist in the pool.
	ErrSequenceNotFound = errors.New("ctxmmu: sequence context not found")

	// ErrSequenceAlreadyExists is returned when attempting to create a sequence with an ID already in use.
	ErrSequenceAlreadyExists = errors.New("ctxmmu: sequence context already exists")

	// ErrInvalidGeometry is returned when KVGeometry parameters violate dimension invariants.
	ErrInvalidGeometry = errors.New("ctxmmu: invalid KV cache geometry parameters")

	// ErrTokenIndexOutOfBounds is returned when accessing a token index outside the sequence bounds.
	ErrTokenIndexOutOfBounds = errors.New("ctxmmu: token index out of bounds")

	// ErrEmptySequenceID is returned when an empty string is provided as sequence ID.
	ErrEmptySequenceID = errors.New("ctxmmu: sequence ID cannot be empty")
)

// Architecture distinguishes standard multi-head/grouped-query attention from latent attention.
type Architecture string

const (
	// ArchDense represents standard MHA / GQA topologies (LLaMA, Mistral, Qwen)
	// where Key and Value projections are stored per layer and head.
	ArchDense Architecture = "dense"

	// ArchMLA represents DeepSeek Multi-Head Latent Attention (MLA)
	// where KV states are compressed into a low-rank latent vector plus decoupled RoPE key.
	ArchMLA Architecture = "mla"
)

// DType represents the numeric precision of KV cache entries.
type DType string

const (
	DTypeFP32 DType = "fp32" // 4 bytes per element
	DTypeFP16 DType = "fp16" // 2 bytes per element
	DTypeBF16 DType = "bf16" // 2 bytes per element
	DTypeFP8  DType = "fp8"  // 1 byte per element
)

// Bytes returns the number of bytes per scalar element for this DType.
func (d DType) Bytes() int {
	switch d {
	case DTypeFP32:
		return 4
	case DTypeFP16, DTypeBF16:
		return 2
	case DTypeFP8:
		return 1
	default:
		return 2
	}
}

// String returns the string representation of DType.
func (d DType) String() string {
	return string(d)
}

// KVGeometry defines the dimension parameters and layout geometry for KV cache pages.
type KVGeometry struct {
	Architecture  Architecture `json:"architecture"`
	TokensPerPage int          `json:"tokens_per_page"`
	NumLayers     int          `json:"num_layers"`
	NumKVHeads    int          `json:"num_kv_heads"`
	HeadDim       int          `json:"head_dim"`
	LatentDim     int          `json:"latent_dim"` // For MLA: compressed KV dimension + decoupled RoPE key
	DType         DType        `json:"dtype"`
}

// Validate checks that the geometry parameters are positive and well-formed.
func (g KVGeometry) Validate() error {
	if g.TokensPerPage <= 0 {
		return fmt.Errorf("%w: tokens_per_page must be > 0", ErrInvalidGeometry)
	}
	if g.NumLayers <= 0 {
		return fmt.Errorf("%w: num_layers must be > 0", ErrInvalidGeometry)
	}
	if g.Architecture == ArchMLA {
		if g.LatentDim <= 0 && (g.NumKVHeads <= 0 || g.HeadDim <= 0) {
			return fmt.Errorf("%w: mla latent_dim must be > 0", ErrInvalidGeometry)
		}
	} else {
		if g.NumKVHeads <= 0 || g.HeadDim <= 0 {
			return fmt.Errorf("%w: dense num_kv_heads and head_dim must be > 0", ErrInvalidGeometry)
		}
	}
	return nil
}

// ElementsPerToken returns the total number of scalar elements required to represent
// one token's KV state across all layers.
func (g KVGeometry) ElementsPerToken() int {
	if g.Architecture == ArchMLA {
		dim := g.LatentDim
		if dim <= 0 {
			dim = g.NumKVHeads * g.HeadDim
		}
		return g.NumLayers * dim
	}
	// Dense (MHA/GQA): 2 * NumLayers * NumKVHeads * HeadDim (factor of 2 for Key and Value)
	return 2 * g.NumLayers * g.NumKVHeads * g.HeadDim
}

// BytesPerToken calculates the memory byte footprint per token.
func (g KVGeometry) BytesPerToken() int {
	return g.ElementsPerToken() * g.DType.Bytes()
}

// PageSizeBytes returns the byte size of a single physical page holding TokensPerPage tokens.
func (g KVGeometry) PageSizeBytes() int64 {
	return int64(g.BytesPerToken() * g.TokensPerPage)
}

// NewDenseGeometry creates a KVGeometry for standard MHA/GQA architectures (e.g. LLaMA, Qwen, Mistral).
func NewDenseGeometry(tokensPerPage, numLayers, numKVHeads, headDim int, dtype DType) KVGeometry {
	return KVGeometry{
		Architecture:  ArchDense,
		TokensPerPage: tokensPerPage,
		NumLayers:     numLayers,
		NumKVHeads:    numKVHeads,
		HeadDim:       headDim,
		DType:         dtype,
	}
}

// NewMLAGeometry creates a KVGeometry for DeepSeek Multi-Head Latent Attention (MLA) architectures.
func NewMLAGeometry(tokensPerPage, numLayers, latentDim int, dtype DType) KVGeometry {
	return KVGeometry{
		Architecture:  ArchMLA,
		TokensPerPage: tokensPerPage,
		NumLayers:     numLayers,
		LatentDim:     latentDim,
		DType:         dtype,
	}
}

// KVPoolConfig supplies configuration for instantiating a KVPool.
type KVPoolConfig struct {
	Geometry          KVGeometry
	NumPhysicalPages  int
	EvictionWatermark float64 // Fraction (0.0–1.0) above which memory pressure triggers (default 0.90)
	OnMemoryPressure  func(pool *KVPool, stats KVPoolStats)
}

// KVPoolStats reports telemetry on memory utilization, paging activity, and fragmentation.
type KVPoolStats struct {
	TotalPages             int     `json:"total_pages"`
	AllocatedPages         int     `json:"allocated_pages"`
	FreePages              int     `json:"free_pages"`
	SharedPages            int     `json:"shared_pages"`
	ActiveSequences        int     `json:"active_sequences"`
	TokensPerPage          int     `json:"tokens_per_page"`
	BytesPerToken          int     `json:"bytes_per_token"`
	PageSizeBytes          int64   `json:"page_size_bytes"`
	TotalCapacityBytes     int64   `json:"total_capacity_bytes"`
	UsedBytes              int64   `json:"used_bytes"`
	COWBranches            int64   `json:"cow_branches"`
	PrefixHits             int64   `json:"prefix_hits"`
	TotalAppendedTokens    int64   `json:"total_appended_tokens"`
	InternalFragmentation  float64 `json:"internal_fragmentation"`  // Fraction of unused token slots in allocated pages
	PagingEfficiencyRatio  float64 `json:"paging_efficiency_ratio"` // Capacity multiplier over static 32k pre-allocation
}

// SequenceContext represents an active agent session's logical sequence of tokens,
// backed by a VirtualPageTable pointing to non-contiguous physical pages in KVPool.
type SequenceContext struct {
	mu         sync.RWMutex
	id         string
	parentID   string
	pool       *KVPool
	pageTable  *VirtualPageTable
	tokens     []int
	lastAccess time.Time
	isFork     bool
}

// ID returns the unique sequence context identifier.
func (s *SequenceContext) ID() string {
	return s.id
}

// ParentID returns the ID of the parent sequence if forked via CoW.
func (s *SequenceContext) ParentID() string {
	return s.parentID
}

// TokenCount returns the total number of tokens currently stored in this sequence.
func (s *SequenceContext) TokenCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.tokens)
}

// PageCount returns the number of physical pages mapped to this sequence.
func (s *SequenceContext) PageCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.pageTable.PageCount()
}

// Tokens returns a defensive copy of all token IDs stored in this sequence.
func (s *SequenceContext) Tokens() []int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	res := make([]int, len(s.tokens))
	copy(res, s.tokens)
	return res
}

// PhysicalPages returns a list of physical page IDs currently mapped in this sequence's page table.
func (s *SequenceContext) PhysicalPages() []PhysicalPageID {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.pageTable.PhysicalPages()
}

// LastAccess returns the timestamp of the most recent read or write to this sequence.
func (s *SequenceContext) LastAccess() time.Time {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lastAccess
}

// Touch marks the sequence as active at the current time.
func (s *SequenceContext) Touch() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastAccess = time.Now()
}

// Append appends tokens and optional KV tensor bytes to this sequence context.
func (s *SequenceContext) Append(tokens []int, kvBytes ...[]byte) error {
	return s.pool.AppendTokens(s.id, tokens, kvBytes...)
}

// ReadToken retrieves the token ID and KV tensor data for the given token index.
func (s *SequenceContext) ReadToken(tokenIndex int) (int, []byte, error) {
	return s.pool.ReadToken(s.id, tokenIndex)
}

// MutateToken overwrites an existing token and KV data at tokenIndex, triggering CoW if shared.
func (s *SequenceContext) MutateToken(tokenIndex int, newToken int, newKV []byte) error {
	return s.pool.MutateToken(s.id, tokenIndex, newToken, newKV)
}

// Fork creates a zero-copy child sequence branch sharing all existing physical pages.
func (s *SequenceContext) Fork(childID string) (*SequenceContext, error) {
	return s.pool.ForkSequence(s.id, childID)
}

// Release releases all physical pages held by this sequence and removes it from the pool.
func (s *SequenceContext) Release() error {
	return s.pool.ReleaseSequence(s.id)
}

// KVPool coordinates model-agnostic decoupled virtual memory paging and pooling
// for KV caches across multi-agent serving sessions.
type KVPool struct {
	mu                  sync.RWMutex
	geometry            KVGeometry
	allocator           *BlockAllocator
	sequences           map[string]*SequenceContext
	evictionWatermark   float64
	pressureHook        func(pool *KVPool, stats KVPoolStats)
	cowBranches         atomic.Int64
	prefixHits          atomic.Int64
	totalAppendedTokens atomic.Int64
}

// NewKVPool instantiates a KVPool with the specified configuration.
func NewKVPool(config KVPoolConfig) (*KVPool, error) {
	if err := config.Geometry.Validate(); err != nil {
		return nil, err
	}
	if config.NumPhysicalPages <= 0 {
		config.NumPhysicalPages = 1024
	}
	watermark := config.EvictionWatermark
	if watermark <= 0 || watermark > 1.0 {
		watermark = 0.90
	}

	pageSizeBytes := config.Geometry.PageSizeBytes()
	allocator := NewBlockAllocator(config.NumPhysicalPages, pageSizeBytes)

	return &KVPool{
		geometry:          config.Geometry,
		allocator:         allocator,
		sequences:         make(map[string]*SequenceContext),
		evictionWatermark: watermark,
		pressureHook:      config.OnMemoryPressure,
	}, nil
}

// Geometry returns the configured KV cache geometry.
func (p *KVPool) Geometry() KVGeometry {
	return p.geometry
}

// Allocator returns the underlying physical block allocator.
func (p *KVPool) Allocator() *BlockAllocator {
	return p.allocator
}

// CreateSequence initializes an empty sequence context in the pool.
func (p *KVPool) CreateSequence(seqID string) (*SequenceContext, error) {
	if seqID == "" {
		return nil, ErrEmptySequenceID
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if _, exists := p.sequences[seqID]; exists {
		return nil, ErrSequenceAlreadyExists
	}

	seq := &SequenceContext{
		id:         seqID,
		pool:       p,
		pageTable:  NewVirtualPageTable(),
		tokens:     make([]int, 0),
		lastAccess: time.Now(),
	}
	p.sequences[seqID] = seq
	return seq, nil
}

// GetSequence returns the SequenceContext for seqID if present.
func (p *KVPool) GetSequence(seqID string) (*SequenceContext, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	seq, ok := p.sequences[seqID]
	return seq, ok
}

// ForkSequence performs zero-copy Copy-on-Write prefix cache branching from parentID into childID.
// All physical pages currently mapped by parentID are retained with incremented reference counts;
// no duplicate physical memory pages are allocated.
func (p *KVPool) ForkSequence(parentID, childID string) (*SequenceContext, error) {
	if parentID == "" || childID == "" {
		return nil, ErrEmptySequenceID
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	parent, ok := p.sequences[parentID]
	if !ok {
		return nil, ErrSequenceNotFound
	}
	if _, exists := p.sequences[childID]; exists {
		return nil, ErrSequenceAlreadyExists
	}

	parent.mu.RLock()
	childTable := parent.pageTable.Clone()
	childTokens := make([]int, len(parent.tokens))
	copy(childTokens, parent.tokens)
	parent.mu.RUnlock()

	// Retain each shared physical page in the child's page table
	physPages := childTable.PhysicalPages()
	for _, physID := range physPages {
		if _, err := p.allocator.Retain(physID); err != nil {
			// Rollback retained pages on failure
			for _, rolled := range physPages {
				if rolled == physID {
					break
				}
				_, _ = p.allocator.Release(rolled)
			}
			return nil, err
		}
	}

	child := &SequenceContext{
		id:         childID,
		parentID:   parentID,
		pool:       p,
		pageTable:  childTable,
		tokens:     childTokens,
		lastAccess: time.Now(),
		isFork:     true,
	}

	p.sequences[childID] = child
	p.prefixHits.Add(int64(len(physPages)))
	return child, nil
}

// ReleaseSequence frees all physical page references held by seqID and deregisters the sequence.
func (p *KVPool) ReleaseSequence(seqID string) error {
	if seqID == "" {
		return ErrEmptySequenceID
	}

	p.mu.Lock()
	seq, ok := p.sequences[seqID]
	if !ok {
		p.mu.Unlock()
		return ErrSequenceNotFound
	}
	delete(p.sequences, seqID)
	p.mu.Unlock()

	seq.mu.Lock()
	defer seq.mu.Unlock()

	physPages := seq.pageTable.PhysicalPages()
	for _, physID := range physPages {
		_, _ = p.allocator.Release(physID)
	}
	seq.tokens = nil
	return nil
}

// AppendTokens appends tokens and optional KV tensor data to seqID.
// Physical pages are allocated dynamically on-demand. If an appended token targets
// a partially filled physical page that is currently shared across forked sessions,
// Copy-on-Write clones the page into a private physical block before writing.
func (p *KVPool) AppendTokens(seqID string, tokens []int, kvBytes ...[]byte) error {
	if seqID == "" {
		return ErrEmptySequenceID
	}
	if len(tokens) == 0 {
		return nil
	}

	p.mu.RLock()
	seq, ok := p.sequences[seqID]
	p.mu.RUnlock()
	if !ok {
		return ErrSequenceNotFound
	}

	needPressureCheck := false
	err := func() error {
		seq.mu.Lock()
		defer seq.mu.Unlock()

		tokensPerPage := p.geometry.TokensPerPage
		bytesPerToken := p.geometry.BytesPerToken()

		for i, tok := range tokens {
			logicalIdx := len(seq.tokens)
			logicalPage := logicalIdx / tokensPerPage
			slotInPage := logicalIdx % tokensPerPage
			byteOffset := slotInPage * bytesPerToken

			physID, exists := seq.pageTable.GetPhysicalPage(logicalPage)
			if !exists || physID < 0 {
				// Allocate a brand new physical page on demand
				newPhysID, err := p.allocator.Allocate()
				if err != nil {
					needPressureCheck = true
					return err
				}
				seq.pageTable.MapPage(logicalPage, newPhysID)
				physID = newPhysID
			} else {
				// Page exists: check if CoW is required due to sharing
				page, err := p.allocator.Page(physID)
				if err != nil {
					return err
				}
				if page.IsShared() {
					// CoW branch: clone page into private block
					newPhysID, err := p.allocator.ClonePage(physID)
					if err != nil {
						needPressureCheck = true
						return err
					}
					_, _ = p.allocator.Release(physID)
					seq.pageTable.MapPage(logicalPage, newPhysID)
					physID = newPhysID
					p.cowBranches.Add(1)
				}
			}

			// Write token ID and KV tensor bytes into physical page buffer
			page, err := p.allocator.Page(physID)
			if err != nil {
				return err
			}
			data := page.Data()

			// Slot header: store token ID in first 4 bytes
			binary.LittleEndian.PutUint32(data[byteOffset:byteOffset+4], uint32(tok))

			// Write KV tensor payload if provided
			if i < len(kvBytes) && len(kvBytes[i]) > 0 {
				copyLen := len(kvBytes[i])
				maxPayload := bytesPerToken - 4
				if copyLen > maxPayload {
					copyLen = maxPayload
				}
				copy(data[byteOffset+4:byteOffset+4+copyLen], kvBytes[i][:copyLen])
			}

			page.Touch()
			seq.tokens = append(seq.tokens, tok)
		}

		seq.lastAccess = time.Now()
		needPressureCheck = true
		return nil
	}()

	if needPressureCheck {
		p.totalAppendedTokens.Add(int64(len(tokens)))
		p.checkMemoryPressure()
	}
	return err
}

// ReadToken retrieves the stored token ID and KV payload at tokenIndex.
func (p *KVPool) ReadToken(seqID string, tokenIndex int) (int, []byte, error) {
	if seqID == "" {
		return 0, nil, ErrEmptySequenceID
	}

	p.mu.RLock()
	seq, ok := p.sequences[seqID]
	p.mu.RUnlock()
	if !ok {
		return 0, nil, ErrSequenceNotFound
	}

	seq.mu.RLock()
	defer seq.mu.RUnlock()

	if tokenIndex < 0 || tokenIndex >= len(seq.tokens) {
		return 0, nil, ErrTokenIndexOutOfBounds
	}

	tokensPerPage := p.geometry.TokensPerPage
	bytesPerToken := p.geometry.BytesPerToken()

	logicalPage := tokenIndex / tokensPerPage
	slotInPage := tokenIndex % tokensPerPage
	byteOffset := slotInPage * bytesPerToken

	physID, exists := seq.pageTable.GetPhysicalPage(logicalPage)
	if !exists {
		return 0, nil, fmt.Errorf("ctxmmu: logical page %d not mapped", logicalPage)
	}

	page, err := p.allocator.Page(physID)
	if err != nil {
		return 0, nil, err
	}
	page.Touch()

	data := page.Data()
	tokID := int(binary.LittleEndian.Uint32(data[byteOffset : byteOffset+4]))

	payload := make([]byte, bytesPerToken-4)
	copy(payload, data[byteOffset+4:byteOffset+bytesPerToken])

	return tokID, payload, nil
}

// MutateToken modifies an existing token and KV payload at tokenIndex,
// triggering Copy-on-Write if the underlying physical page is shared.
func (p *KVPool) MutateToken(seqID string, tokenIndex int, newToken int, newKV []byte) error {
	if seqID == "" {
		return ErrEmptySequenceID
	}

	p.mu.RLock()
	seq, ok := p.sequences[seqID]
	p.mu.RUnlock()
	if !ok {
		return ErrSequenceNotFound
	}

	seq.mu.Lock()
	defer seq.mu.Unlock()

	if tokenIndex < 0 || tokenIndex >= len(seq.tokens) {
		return ErrTokenIndexOutOfBounds
	}

	tokensPerPage := p.geometry.TokensPerPage
	bytesPerToken := p.geometry.BytesPerToken()

	logicalPage := tokenIndex / tokensPerPage
	slotInPage := tokenIndex % tokensPerPage
	byteOffset := slotInPage * bytesPerToken

	physID, exists := seq.pageTable.GetPhysicalPage(logicalPage)
	if !exists {
		return fmt.Errorf("ctxmmu: logical page %d not mapped", logicalPage)
	}

	page, err := p.allocator.Page(physID)
	if err != nil {
		return err
	}

	if page.IsShared() {
		// CoW clone
		newPhysID, err := p.allocator.ClonePage(physID)
		if err != nil {
			return err
		}
		_, _ = p.allocator.Release(physID)
		seq.pageTable.MapPage(logicalPage, newPhysID)
		physID = newPhysID
		page, _ = p.allocator.Page(physID)
		p.cowBranches.Add(1)
	}

	data := page.Data()
	binary.LittleEndian.PutUint32(data[byteOffset:byteOffset+4], uint32(newToken))

	if len(newKV) > 0 {
		copyLen := len(newKV)
		maxPayload := bytesPerToken - 4
		if copyLen > maxPayload {
			copyLen = maxPayload
		}
		copy(data[byteOffset+4:byteOffset+4+copyLen], newKV[:copyLen])
	}

	page.Touch()
	seq.tokens[tokenIndex] = newToken
	seq.lastAccess = time.Now()
	return nil
}

// checkMemoryPressure inspects allocated page levels and invokes pressureHook if above watermark.
func (p *KVPool) checkMemoryPressure() {
	if p.pressureHook == nil {
		return
	}
	total := p.allocator.TotalPages()
	if total == 0 {
		return
	}
	allocated := p.allocator.AllocatedPages()
	ratio := float64(allocated) / float64(total)
	if ratio >= p.evictionWatermark {
		stats := p.Stats()
		p.pressureHook(p, stats)
	}
}

// EvictLRU evicts least-recently accessed sequences until targetPages are reclaimed
// or no more evictable sequences remain.
func (p *KVPool) EvictLRU(targetPages int) (int, error) {
	if targetPages <= 0 {
		return 0, nil
	}

	p.mu.Lock()
	type seqInfo struct {
		id         string
		lastAccess time.Time
		pageCount  int
	}

	seqs := make([]seqInfo, 0, len(p.sequences))
	for id, seq := range p.sequences {
		seq.mu.RLock()
		seqs = append(seqs, seqInfo{
			id:         id,
			lastAccess: seq.lastAccess,
			pageCount:  seq.pageTable.PageCount(),
		})
		seq.mu.RUnlock()
	}
	p.mu.Unlock()

	// Sort oldest first
	sort.Slice(seqs, func(i, j int) bool {
		return seqs[i].lastAccess.Before(seqs[j].lastAccess)
	})

	reclaimed := 0
	for _, s := range seqs {
		if reclaimed >= targetPages {
			break
		}
		before := p.allocator.AllocatedPages()
		if err := p.ReleaseSequence(s.id); err != nil {
			continue
		}
		after := p.allocator.AllocatedPages()
		reclaimed += (before - after)
	}

	return reclaimed, nil
}

// Stats compiles comprehensive memory utilization and paging metrics.
func (p *KVPool) Stats() KVPoolStats {
	p.mu.RLock()
	defer p.mu.RUnlock()

	totalPages := p.allocator.TotalPages()
	allocatedPages := p.allocator.AllocatedPages()
	freePages := p.allocator.FreePages()
	pageSizeBytes := p.geometry.PageSizeBytes()
	tokensPerPage := p.geometry.TokensPerPage

	// Count shared pages and calculate internal fragmentation across active sequences
	sharedCount := 0
	totalAllocatedSlots := 0
	totalUsedSlots := 0

	for i := 0; i < totalPages; i++ {
		if p.allocator.IsAllocated(PhysicalPageID(i)) {
			if page, err := p.allocator.Page(PhysicalPageID(i)); err == nil {
				if page.IsShared() {
					sharedCount++
				}
			}
		}
	}

	for _, seq := range p.sequences {
		seq.mu.RLock()
		pCount := seq.pageTable.PageCount()
		tCount := len(seq.tokens)
		seq.mu.RUnlock()

		totalAllocatedSlots += pCount * tokensPerPage
		totalUsedSlots += tCount
	}

	frag := 0.0
	if totalAllocatedSlots > 0 {
		unused := totalAllocatedSlots - totalUsedSlots
		frag = float64(unused) / float64(totalAllocatedSlots)
	}

	// PagingEfficiencyRatio: Multiplier of concurrent capacity versus static 32k pre-allocation
	efficiency := 1.0
	if allocatedPages > 0 && len(p.sequences) > 0 {
		staticTokensPerSeq := 32768
		staticTotalSlots := len(p.sequences) * staticTokensPerSeq
		actualSlots := allocatedPages * tokensPerPage
		if actualSlots > 0 {
			efficiency = float64(staticTotalSlots) / float64(actualSlots)
		}
	}

	return KVPoolStats{
		TotalPages:            totalPages,
		AllocatedPages:        allocatedPages,
		FreePages:             freePages,
		SharedPages:           sharedCount,
		ActiveSequences:       len(p.sequences),
		TokensPerPage:         tokensPerPage,
		BytesPerToken:         p.geometry.BytesPerToken(),
		PageSizeBytes:         pageSizeBytes,
		TotalCapacityBytes:    int64(totalPages) * pageSizeBytes,
		UsedBytes:             int64(allocatedPages) * pageSizeBytes,
		COWBranches:           p.cowBranches.Load(),
		PrefixHits:            p.prefixHits.Load(),
		TotalAppendedTokens:   p.totalAppendedTokens.Load(),
		InternalFragmentation: frag,
		PagingEfficiencyRatio: efficiency,
	}
}
