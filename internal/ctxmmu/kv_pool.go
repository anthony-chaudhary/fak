package ctxmmu

import (
	"errors"
	"fmt"
	"sort"
	"sync"
)

// Typed errors for KV pool and sequence operations.
var (
	// ErrSequenceNotFound is returned when a sequence ID cannot be located in the pool.
	ErrSequenceNotFound = errors.New("ctxmmu: sequence not found")

	// ErrSequenceExists is returned when attempting to create or fork a sequence ID that already exists.
	ErrSequenceExists = errors.New("ctxmmu: sequence already exists")

	// ErrTokenOutOfBounds is returned when a requested token position exceeds the sequence length.
	ErrTokenOutOfBounds = errors.New("ctxmmu: token position out of bounds")

	// ErrPageNotAllocated is returned when translating or accessing an unallocated page table entry.
	ErrPageNotAllocated = errors.New("ctxmmu: page not allocated")

	// ErrInvalidLayer is returned when a requested layer index is out of bounds [0, NumLayers).
	ErrInvalidLayer = errors.New("ctxmmu: invalid layer index")

	// ErrBlockSwapped is returned when reading or writing directly to a block currently swapped out.
	ErrBlockSwapped = errors.New("ctxmmu: block is currently swapped out")

	// ErrBlockShared is returned when attempting to swap a physical block currently shared across sequences.
	ErrBlockShared = errors.New("ctxmmu: cannot swap shared physical block")

	// ErrSwapNotFound is returned when attempting to swap in page data that was not previously saved.
	ErrSwapNotFound = errors.New("ctxmmu: swapped page data not found")
)

// KVDType represents the numerical precision of KV cache activations.
type KVDType string

const (
	KVDTypeFP32 KVDType = "fp32"
	KVDTypeFP16 KVDType = "fp16"
	KVDTypeBF16 KVDType = "bf16"
	KVDTypeFP8  KVDType = "fp8"
)

// BytesPerElement returns the byte width of the KV data type.
func (d KVDType) BytesPerElement() int {
	switch d {
	case KVDTypeFP32:
		return 4
	case KVDTypeFP16, KVDTypeBF16:
		return 2
	case KVDTypeFP8:
		return 1
	default:
		return 2
	}
}

// KVTopology represents the attention architecture geometry.
type KVTopology string

const (
	KVTopologyGQA KVTopology = "gqa"
	KVTopologyMLA KVTopology = "mla"
)

// KVPoolConfig parameterizes the paged attention KV memory pool.
type KVPoolConfig struct {
	TokensPerPage int        `json:"tokens_per_page"`
	NumLayers     int        `json:"num_layers"`
	NumKVHeads    int        `json:"num_kv_heads"`     // for GQA/MHA
	HeadDim       int        `json:"head_dim"`          // for GQA/MHA
	KVLoraRank    int        `json:"kv_lora_rank"`      // for MLA (e.g. 512)
	QKRopeHeadDim int        `json:"qk_rope_head_dim"`  // for MLA (e.g. 64)
	DType         KVDType    `json:"dtype"`
	Topology      KVTopology `json:"topology"`
	TotalPages    int        `json:"total_pages"`
}

// PageBytes calculates the byte footprint of a physical page according to attention geometry:
// - For GQA: TokensPerPage * NumLayers * 2 * NumKVHeads * HeadDim * DTypeBytes
// - For MLA: TokensPerPage * NumLayers * (KVLoraRank + QKRopeHeadDim) * DTypeBytes
func (cfg KVPoolConfig) PageBytes() int {
	dBytes := cfg.DType.BytesPerElement()
	tokens := cfg.TokensPerPage
	if tokens <= 0 {
		tokens = 16
	}
	layers := cfg.NumLayers
	if layers <= 0 {
		layers = 32
	}

	if cfg.Topology == KVTopologyMLA {
		rank := cfg.KVLoraRank
		if rank <= 0 {
			rank = 512
		}
		ropeDim := cfg.QKRopeHeadDim
		if ropeDim <= 0 {
			ropeDim = 64
		}
		return tokens * layers * (rank + ropeDim) * dBytes
	}

	// GQA (default)
	heads := cfg.NumKVHeads
	if heads <= 0 {
		heads = 8
	}
	dim := cfg.HeadDim
	if dim <= 0 {
		dim = 128
	}
	return tokens * layers * 2 * heads * dim * dBytes
}

// KVSequence represents a logical sequence with paged, non-contiguous physical KV backing.
type KVSequence struct {
	mu           sync.RWMutex
	ID           string
	TokenCount   int
	PageTable    []int // maps logical page index to physical block ID
	PagedKV      bool  // whether paged KV management is active
	swappedPages map[int][]byte
}

// Pages returns a point-in-time copy of the sequence's page table.
func (s *KVSequence) Pages() []int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]int, len(s.PageTable))
	copy(out, s.PageTable)
	return out
}

// Tokens returns the current token count of the sequence.
func (s *KVSequence) Tokens() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.TokenCount
}

// Len returns the current token count of the sequence.
func (s *KVSequence) Len() int {
	return s.Tokens()
}

// KVPool coordinates virtual memory management mapping logical sequence tokens
// to non-contiguous physical pages across multiple concurrent generation streams.
type KVPool struct {
	mu            sync.RWMutex
	cfg           KVPoolConfig
	allocator     *PhysicalBlockAllocator
	sequences     map[string]*KVSequence
	swapMu        sync.Mutex
	swapStore     map[string][]byte
	pressureHooks []func(float64)
}

// NewKVPool initializes a unified memory KV cache pool with physical block allocator.
func NewKVPool(cfg KVPoolConfig) (*KVPool, error) {
	if cfg.TokensPerPage <= 0 {
		cfg.TokensPerPage = 16
	}
	if cfg.TotalPages <= 0 {
		cfg.TotalPages = 1024
	}
	if cfg.NumLayers <= 0 {
		cfg.NumLayers = 32
	}
	if cfg.DType == "" {
		cfg.DType = KVDTypeFP16
	}
	if cfg.Topology == "" {
		cfg.Topology = KVTopologyGQA
	}
	if cfg.Topology == KVTopologyGQA {
		if cfg.NumKVHeads <= 0 {
			cfg.NumKVHeads = 8
		}
		if cfg.HeadDim <= 0 {
			cfg.HeadDim = 128
		}
	} else if cfg.Topology == KVTopologyMLA {
		if cfg.KVLoraRank <= 0 {
			cfg.KVLoraRank = 512
		}
		if cfg.QKRopeHeadDim <= 0 {
			cfg.QKRopeHeadDim = 64
		}
	}

	pageBytes := cfg.PageBytes()
	if pageBytes <= 0 {
		return nil, errors.New("ctxmmu: invalid page size calculation")
	}

	alloc := NewPhysicalBlockAllocator(cfg.TotalPages, pageBytes)

	return &KVPool{
		cfg:           cfg,
		allocator:     alloc,
		sequences:     make(map[string]*KVSequence),
		swapStore:     make(map[string][]byte),
		pressureHooks: nil,
	}, nil
}

// CreateSequence registers a new logical KV sequence in the pool.
func (p *KVPool) CreateSequence(seqID string) (*KVSequence, error) {
	if seqID == "" {
		return nil, ErrEmptyStreamID
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if _, exists := p.sequences[seqID]; exists {
		return nil, ErrSequenceExists
	}

	seq := &KVSequence{
		ID:           seqID,
		TokenCount:   0,
		PageTable:    make([]int, 0),
		PagedKV:      true,
		swappedPages: make(map[int][]byte),
	}
	p.sequences[seqID] = seq
	return seq, nil
}

// GetSequence looks up an existing KV sequence by ID.
func (p *KVPool) GetSequence(seqID string) (*KVSequence, bool) {
	if seqID == "" {
		return nil, false
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	seq, ok := p.sequences[seqID]
	return seq, ok
}

// ReleaseSequence removes a sequence and releases its physical pages back to the allocator.
func (p *KVPool) ReleaseSequence(seqID string) error {
	if seqID == "" {
		return ErrEmptyStreamID
	}

	p.mu.Lock()
	seq, ok := p.sequences[seqID]
	if !ok {
		p.mu.Unlock()
		return ErrSequenceNotFound
	}
	delete(p.sequences, seqID)

	// Clean up any swapped entries for this sequence
	prefix := seqID + ":"
	p.swapMu.Lock()
	for k := range p.swapStore {
		if len(k) >= len(prefix) && k[:len(prefix)] == prefix {
			delete(p.swapStore, k)
		}
	}
	p.swapMu.Unlock()
	p.mu.Unlock()

	seq.mu.Lock()
	defer seq.mu.Unlock()

	for _, blockID := range seq.PageTable {
		if blockID >= 0 {
			if block, err := p.allocator.GetBlock(blockID); err == nil {
				if block.Release() <= 0 {
					p.allocator.Free(blockID)
				}
			}
		}
	}
	seq.PageTable = nil
	seq.TokenCount = 0
	return nil
}

// AppendTokens allocates physical pages on demand for additional tokens. If appending to
// a shared page with refCount > 1, it executes Copy-on-Write so only diverging pages are copied.
func (p *KVPool) AppendTokens(seqID string, count int) error {
	if count < 0 {
		return ErrInvalidTokens
	}
	if count == 0 {
		return nil
	}

	p.mu.RLock()
	seq, ok := p.sequences[seqID]
	p.mu.RUnlock()
	if !ok {
		return ErrSequenceNotFound
	}

	seq.mu.Lock()
	defer seq.mu.Unlock()

	oldTokens := seq.TokenCount
	newTokens := oldTokens + count
	tokensPerPage := p.cfg.TokensPerPage

	// 1. If oldTokens > 0 and the last page is partially filled, check if it was shared.
	if oldTokens > 0 && (oldTokens%tokensPerPage != 0) && len(seq.PageTable) > 0 {
		lastPageIdx := (oldTokens - 1) / tokensPerPage
		if lastPageIdx < len(seq.PageTable) {
			lastBlockID := seq.PageTable[lastPageIdx]
			if lastBlockID >= 0 {
				block, err := p.allocator.GetBlock(lastBlockID)
				if err == nil && block.RefCount() > 1 {
					// CoW: copy the shared block so this sequence has its own divergent copy
					newBlock, err := p.allocator.Allocate()
					if err != nil {
						return err
					}
					copy(newBlock.Data, block.Data)
					newBlock.Touch()
					seq.PageTable[lastPageIdx] = newBlock.ID
					if block.Release() <= 0 {
						p.allocator.Free(lastBlockID)
					}
				}
			}
		}
	}

	// 2. Calculate how many total pages are needed
	neededPages := (newTokens + tokensPerPage - 1) / tokensPerPage
	currentPages := len(seq.PageTable)

	if neededPages > currentPages {
		newBlocksAllocated := make([]int, 0, neededPages-currentPages)
		for i := currentPages; i < neededPages; i++ {
			b, err := p.allocator.Allocate()
			if err != nil {
				// Rollback allocated blocks on failure
				for _, bID := range newBlocksAllocated {
					p.allocator.Free(bID)
				}
				return err
			}
			newBlocksAllocated = append(newBlocksAllocated, b.ID)
		}
		seq.PageTable = append(seq.PageTable, newBlocksAllocated...)
	}

	seq.TokenCount = newTokens
	return nil
}

// Translate maps a logical sequence token position to its backing physical block ID and page offset.
func (p *KVPool) Translate(seqID string, tokenPos int) (blockID int, pageOffset int, err error) {
	if tokenPos < 0 {
		return -1, -1, ErrTokenOutOfBounds
	}
	p.mu.RLock()
	seq, ok := p.sequences[seqID]
	p.mu.RUnlock()
	if !ok {
		return -1, -1, ErrSequenceNotFound
	}

	seq.mu.RLock()
	defer seq.mu.RUnlock()

	if tokenPos >= seq.TokenCount {
		return -1, -1, ErrTokenOutOfBounds
	}

	tokensPerPage := p.cfg.TokensPerPage
	pageIdx := tokenPos / tokensPerPage
	pageOffset = tokenPos % tokensPerPage

	if pageIdx >= len(seq.PageTable) {
		return -1, -1, ErrPageNotAllocated
	}

	blockID = seq.PageTable[pageIdx]
	return blockID, pageOffset, nil
}

// WriteTokenKV writes Key and Value data for a specific sequence token at a given layer.
// If the physical block backing the token is shared (refCount > 1), Copy-on-Write is performed.
func (p *KVPool) WriteTokenKV(seqID string, tokenPos int, layer int, keyData, valData []byte) error {
	if layer < 0 || layer >= p.cfg.NumLayers {
		return ErrInvalidLayer
	}

	p.mu.RLock()
	seq, ok := p.sequences[seqID]
	p.mu.RUnlock()
	if !ok {
		return ErrSequenceNotFound
	}

	seq.mu.Lock()
	defer seq.mu.Unlock()

	if tokenPos < 0 || tokenPos >= seq.TokenCount {
		return ErrTokenOutOfBounds
	}

	tokensPerPage := p.cfg.TokensPerPage
	pageIdx := tokenPos / tokensPerPage
	pageOffset := tokenPos % tokensPerPage

	if pageIdx >= len(seq.PageTable) {
		return ErrPageNotAllocated
	}

	blockID := seq.PageTable[pageIdx]
	block, err := p.allocator.GetBlock(blockID)
	if err != nil {
		return err
	}

	// If block is shared, perform Copy-on-Write
	if block.RefCount() > 1 {
		newBlock, err := p.allocator.Allocate()
		if err != nil {
			return err
		}
		copy(newBlock.Data, block.Data)
		newBlock.Touch()
		seq.PageTable[pageIdx] = newBlock.ID
		if block.Release() <= 0 {
			p.allocator.Free(blockID)
		}
		block = newBlock
		blockID = newBlock.ID
	}

	if block.Swapped() {
		return ErrBlockSwapped
	}

	block.Touch()

	dBytes := p.cfg.DType.BytesPerElement()
	if p.cfg.Topology == KVTopologyMLA {
		rankBytes := p.cfg.KVLoraRank * dBytes
		ropeBytes := p.cfg.QKRopeHeadDim * dBytes
		stridePerLayer := rankBytes + ropeBytes
		tokenStride := p.cfg.NumLayers * stridePerLayer
		offset := pageOffset*tokenStride + layer*stridePerLayer

		if len(valData) == 0 && len(keyData) > rankBytes {
			limit := len(keyData)
			if limit > stridePerLayer {
				limit = stridePerLayer
			}
			copy(block.Data[offset:], keyData[:limit])
		} else {
			if len(keyData) > 0 {
				limit := len(keyData)
				if limit > rankBytes {
					limit = rankBytes
				}
				copy(block.Data[offset:], keyData[:limit])
			}
			if len(valData) > 0 {
				limit := len(valData)
				if limit > ropeBytes {
					limit = ropeBytes
				}
				copy(block.Data[offset+rankBytes:], valData[:limit])
			}
		}
	} else {
		// GQA
		kvStride := p.cfg.NumKVHeads * p.cfg.HeadDim * dBytes
		layerStride := 2 * kvStride
		tokenStride := p.cfg.NumLayers * layerStride

		offsetK := pageOffset*tokenStride + layer*layerStride
		offsetV := offsetK + kvStride

		if len(keyData) > 0 {
			limit := len(keyData)
			if limit > kvStride {
				limit = kvStride
			}
			copy(block.Data[offsetK:], keyData[:limit])
		}
		if len(valData) > 0 {
			limit := len(valData)
			if limit > kvStride {
				limit = kvStride
			}
			copy(block.Data[offsetV:], valData[:limit])
		}
	}

	return nil
}

// ReadTokenKV reads Key and Value data for a specific sequence token at a given layer.
func (p *KVPool) ReadTokenKV(seqID string, tokenPos int, layer int) (keyData, valData []byte, err error) {
	if layer < 0 || layer >= p.cfg.NumLayers {
		return nil, nil, ErrInvalidLayer
	}

	p.mu.RLock()
	seq, ok := p.sequences[seqID]
	p.mu.RUnlock()
	if !ok {
		return nil, nil, ErrSequenceNotFound
	}

	seq.mu.RLock()
	defer seq.mu.RUnlock()

	if tokenPos < 0 || tokenPos >= seq.TokenCount {
		return nil, nil, ErrTokenOutOfBounds
	}

	tokensPerPage := p.cfg.TokensPerPage
	pageIdx := tokenPos / tokensPerPage
	pageOffset := tokenPos % tokensPerPage

	if pageIdx >= len(seq.PageTable) {
		return nil, nil, ErrPageNotAllocated
	}

	blockID := seq.PageTable[pageIdx]
	block, err := p.allocator.GetBlock(blockID)
	if err != nil {
		return nil, nil, err
	}

	if block.Swapped() {
		return nil, nil, ErrBlockSwapped
	}

	block.Touch()

	dBytes := p.cfg.DType.BytesPerElement()
	if p.cfg.Topology == KVTopologyMLA {
		rankBytes := p.cfg.KVLoraRank * dBytes
		ropeBytes := p.cfg.QKRopeHeadDim * dBytes
		stridePerLayer := rankBytes + ropeBytes
		tokenStride := p.cfg.NumLayers * stridePerLayer
		offset := pageOffset*tokenStride + layer*stridePerLayer

		keyData = make([]byte, rankBytes)
		copy(keyData, block.Data[offset:offset+rankBytes])

		valData = make([]byte, ropeBytes)
		copy(valData, block.Data[offset+rankBytes:offset+stridePerLayer])

		return keyData, valData, nil
	}

	// GQA
	kvStride := p.cfg.NumKVHeads * p.cfg.HeadDim * dBytes
	layerStride := 2 * kvStride
	tokenStride := p.cfg.NumLayers * layerStride

	offsetK := pageOffset*tokenStride + layer*layerStride
	offsetV := offsetK + kvStride

	keyData = make([]byte, kvStride)
	copy(keyData, block.Data[offsetK:offsetK+kvStride])

	valData = make([]byte, kvStride)
	copy(valData, block.Data[offsetV:offsetV+kvStride])

	return keyData, valData, nil
}

// ForkSequence creates a zero-copy CoW child sequence sharing all physical blocks of the parent.
// RefCounts of existing physical blocks are incremented; 0 new physical blocks are allocated.
func (p *KVPool) ForkSequence(parentID, childID string) (*KVSequence, error) {
	if parentID == "" || childID == "" {
		return nil, ErrEmptyStreamID
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	parent, ok := p.sequences[parentID]
	if !ok {
		return nil, ErrSequenceNotFound
	}
	if _, exists := p.sequences[childID]; exists {
		return nil, ErrSequenceExists
	}

	parent.mu.RLock()
	defer parent.mu.RUnlock()

	pageCount := len(parent.PageTable)
	childPageTable := make([]int, pageCount)
	for i, bID := range parent.PageTable {
		if err := p.allocator.Retain(bID); err != nil {
			// Rollback retained blocks on error
			for j := 0; j < i; j++ {
				if b, getErr := p.allocator.GetBlock(childPageTable[j]); getErr == nil {
					b.Release()
				}
			}
			return nil, err
		}
		childPageTable[i] = bID
	}

	child := &KVSequence{
		ID:           childID,
		TokenCount:   parent.TokenCount,
		PageTable:    childPageTable,
		PagedKV:      parent.PagedKV,
		swappedPages: make(map[int][]byte),
	}
	p.sequences[childID] = child
	return child, nil
}

// RegisterPressureHook registers a callback for memory pressure notifications.
func (p *KVPool) RegisterPressureHook(hook func(pressure float64)) {
	if hook == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pressureHooks = append(p.pressureHooks, hook)
}

// TriggerPressure dispatches a memory pressure notification to all registered hooks.
func (p *KVPool) TriggerPressure(pressure float64) {
	p.mu.RLock()
	hooks := make([]func(float64), len(p.pressureHooks))
	copy(hooks, p.pressureHooks)
	p.mu.RUnlock()

	for _, h := range hooks {
		if h != nil {
			h(pressure)
		}
	}
}

// SwapBlock swaps out a sequence's physical page to the swap store and marks it swapped.
func (p *KVPool) SwapBlock(seqID string, pageIdx int) error {
	p.mu.RLock()
	seq, ok := p.sequences[seqID]
	p.mu.RUnlock()
	if !ok {
		return ErrSequenceNotFound
	}

	seq.mu.Lock()
	defer seq.mu.Unlock()

	if pageIdx < 0 || pageIdx >= len(seq.PageTable) {
		return ErrPageNotAllocated
	}

	blockID := seq.PageTable[pageIdx]
	if blockID < 0 {
		return nil
	}

	block, err := p.allocator.GetBlock(blockID)
	if err != nil {
		return err
	}

	if block.RefCount() > 1 {
		return ErrBlockShared
	}

	key := fmt.Sprintf("%s:%d", seqID, pageIdx)
	dataCopy := make([]byte, len(block.Data))
	copy(dataCopy, block.Data)

	p.swapMu.Lock()
	p.swapStore[key] = dataCopy
	p.swapMu.Unlock()

	block.SetSwapped(true)
	return nil
}

// SwapInBlock restores a previously swapped-out page from the swap store.
func (p *KVPool) SwapInBlock(seqID string, pageIdx int) error {
	p.mu.RLock()
	seq, ok := p.sequences[seqID]
	p.mu.RUnlock()
	if !ok {
		return ErrSequenceNotFound
	}

	seq.mu.Lock()
	defer seq.mu.Unlock()

	if pageIdx < 0 || pageIdx >= len(seq.PageTable) {
		return ErrPageNotAllocated
	}

	key := fmt.Sprintf("%s:%d", seqID, pageIdx)
	p.swapMu.Lock()
	swappedData, exists := p.swapStore[key]
	if exists {
		delete(p.swapStore, key)
	}
	p.swapMu.Unlock()

	if !exists {
		return ErrSwapNotFound
	}

	blockID := seq.PageTable[pageIdx]
	block, err := p.allocator.GetBlock(blockID)
	if err != nil || block == nil {
		newBlock, allocErr := p.allocator.Allocate()
		if allocErr != nil {
			return allocErr
		}
		copy(newBlock.Data, swappedData)
		newBlock.SetSwapped(false)
		newBlock.Touch()
		seq.PageTable[pageIdx] = newBlock.ID
		return nil
	}

	copy(block.Data, swappedData)
	block.SetSwapped(false)
	block.Touch()
	return nil
}

// EvictLRU reclaims up to targetBlocks physical blocks under memory pressure.
// It first recycles unreferenced allocator blocks, then swaps out active resident pages in LRU order.
func (p *KVPool) EvictLRU(targetBlocks int) (int, error) {
	if targetBlocks <= 0 {
		return 0, nil
	}

	reclaimed, err := p.allocator.RecycleLRU(targetBlocks)
	if err != nil {
		return 0, err
	}
	if len(reclaimed) >= targetBlocks {
		return len(reclaimed), nil
	}

	evicted := len(reclaimed)

	type pageRef struct {
		seqID      string
		pageIdx    int
		blockID    int
		lastAccess int64
	}
	var candidates []pageRef

	p.mu.RLock()
	for sID, seq := range p.sequences {
		seq.mu.RLock()
		for pIdx, bID := range seq.PageTable {
			if bID >= 0 {
				if b, bErr := p.allocator.GetBlock(bID); bErr == nil && !b.Swapped() && b.RefCount() == 1 {
					candidates = append(candidates, pageRef{
						seqID:      sID,
						pageIdx:    pIdx,
						blockID:    bID,
						lastAccess: b.LastAccess(),
					})
				}
			}
		}
		seq.mu.RUnlock()
	}
	p.mu.RUnlock()

	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].lastAccess != candidates[j].lastAccess {
			return candidates[i].lastAccess < candidates[j].lastAccess
		}
		if candidates[i].seqID != candidates[j].seqID {
			return candidates[i].seqID < candidates[j].seqID
		}
		return candidates[i].pageIdx < candidates[j].pageIdx
	})

	for _, cand := range candidates {
		if evicted >= targetBlocks {
			break
		}
		if swapErr := p.SwapBlock(cand.seqID, cand.pageIdx); swapErr == nil {
			evicted++
		}
	}

	return evicted, nil
}

// PhysicalBlocksAllocated returns the count of physical blocks currently claimed in DRAM.
func (p *KVPool) PhysicalBlocksAllocated() int {
	if p == nil || p.allocator == nil {
		return 0
	}
	return p.allocator.AllocatedCount()
}

// LogicalTokensAllocated returns the total number of logical tokens managed across all sequences.
func (p *KVPool) LogicalTokensAllocated() int {
	if p == nil {
		return 0
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	sum := 0
	for _, s := range p.sequences {
		sum += s.Tokens()
	}
	return sum
}

// DuplicatePhysicalPagesAllocated returns the number of redundant physical pages allocated on fork (always 0).
func (p *KVPool) DuplicatePhysicalPagesAllocated() int {
	return 0
}

// CapacityGainRatio computes the memory capacity expansion ratio over naive static allocation:
// baseline tokens / actual physical tokens.
func (p *KVPool) CapacityGainRatio(maxSeqLen int) float64 {
	if p == nil {
		return 1.0
	}
	p.mu.RLock()
	defer p.mu.RUnlock()

	numSeqs := len(p.sequences)
	if numSeqs == 0 {
		return 1.0
	}

	tokensPerPage := p.cfg.TokensPerPage
	if tokensPerPage <= 0 {
		tokensPerPage = 16
	}

	var baselineTokens int
	if maxSeqLen > 0 {
		baselineTokens = numSeqs * maxSeqLen
	} else {
		for _, s := range p.sequences {
			baselineTokens += s.Tokens()
		}
	}

	if baselineTokens == 0 {
		return 1.0
	}

	physBlocks := p.allocator.AllocatedCount()
	if physBlocks <= 0 {
		return 1.0
	}

	physTokens := physBlocks * tokensPerPage
	return float64(baselineTokens) / float64(physTokens)
}

// Allocator returns the underlying physical block allocator.
func (p *KVPool) Allocator() *PhysicalBlockAllocator {
	if p == nil {
		return nil
	}
	return p.allocator
}

// Config returns the pool configuration.
func (p *KVPool) Config() KVPoolConfig {
	if p == nil {
		return KVPoolConfig{}
	}
	return p.cfg
}

// -----------------------------------------------------------------------------
// MMU Integration
// -----------------------------------------------------------------------------

var (
	defaultKVPoolMu sync.Mutex
	defaultKVPool   *KVPool
	mmuKVPoolMu     sync.Mutex
	mmuKVPools      = make(map[*MMU]*KVPool)
)

func getDefaultKVPool() *KVPool {
	defaultKVPoolMu.Lock()
	defer defaultKVPoolMu.Unlock()
	if defaultKVPool == nil {
		defaultKVPool, _ = NewKVPool(KVPoolConfig{TotalPages: 1024})
	}
	return defaultKVPool
}

// KVPool returns a KVPool instance associated with this MMU.
func (m *MMU) KVPool(cfg ...KVPoolConfig) *KVPool {
	if m == nil {
		if len(cfg) > 0 {
			p, _ := NewKVPool(cfg[0])
			return p
		}
		return getDefaultKVPool()
	}
	mmuKVPoolMu.Lock()
	defer mmuKVPoolMu.Unlock()
	p := mmuKVPools[m]
	if p == nil || len(cfg) > 0 {
		var c KVPoolConfig
		if len(cfg) > 0 {
			c = cfg[0]
		} else {
			c = KVPoolConfig{TotalPages: 1024}
		}
		newP, err := NewKVPool(c)
		if err == nil {
			p = newP
			mmuKVPools[m] = p
		}
	}
	return p
}
