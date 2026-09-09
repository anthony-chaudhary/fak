package model

import (
	"errors"
	"fmt"
	"sync"
)

// gdn_align_cache.go — Align-mode recurrent state checkpointing and prefix caching
// for Qwen3.5/3.8 GDN (Gated DeltaNet) linear attention layers.
// Provenance: vllm-project/vllm-metal:PR#584-align-state (#12588).

// RecurrentStateSlab holds an immutable snapshot of recurrent/convolution state
// aligned to a block boundary.
type RecurrentStateSlab struct {
	BlockIndex int             `json:"block_index"`
	TokenCount int             `json:"token_count"`
	TokenIDs   []int           `json:"token_ids,omitempty"`
	Layers     []GDNLayerState `json:"layers,omitempty"`
	SizeBytes  int64           `json:"size_bytes"`
	IsFrozen   bool            `json:"is_frozen"`
}

// computeSlabSizeBytes calculates total memory footprint in bytes for token IDs and layer states.
func computeSlabSizeBytes(tokenIDs []int, layers []GDNLayerState) int64 {
	var size int64
	size += int64(len(tokenIDs) * 8)
	for _, l := range layers {
		size += int64((len(l.Conv) + len(l.Recurrent)) * 4)
	}
	return size
}

// NewRecurrentStateSlab constructs a new RecurrentStateSlab with cloned layers and token IDs.
func NewRecurrentStateSlab(blockIndex int, tokenCount int, tokenIDs []int, layers []GDNLayerState) *RecurrentStateSlab {
	var tokCopy []int
	if len(tokenIDs) > 0 {
		tokCopy = append([]int(nil), tokenIDs...)
	}
	layerClones := cloneSidecar(layers)
	return &RecurrentStateSlab{
		BlockIndex: blockIndex,
		TokenCount: tokenCount,
		TokenIDs:   tokCopy,
		Layers:     layerClones,
		SizeBytes:  computeSlabSizeBytes(tokCopy, layerClones),
		IsFrozen:   false,
	}
}

// Clone creates an independent deep copy of the recurrent state slab.
func (s *RecurrentStateSlab) Clone() *RecurrentStateSlab {
	if s == nil {
		return nil
	}
	var tokCopy []int
	if len(s.TokenIDs) > 0 {
		tokCopy = append([]int(nil), s.TokenIDs...)
	}
	return &RecurrentStateSlab{
		BlockIndex: s.BlockIndex,
		TokenCount: s.TokenCount,
		TokenIDs:   tokCopy,
		Layers:     cloneSidecar(s.Layers),
		SizeBytes:  s.SizeBytes,
		IsFrozen:   s.IsFrozen,
	}
}

// Freeze marks the slab as immutable.
func (s *RecurrentStateSlab) Freeze() {
	if s != nil {
		s.IsFrozen = true
	}
}

// ToSidecar returns a cloned slice of layer states suitable for use as a session sidecar.
func (s *RecurrentStateSlab) ToSidecar() []GDNLayerState {
	if s == nil {
		return nil
	}
	return cloneSidecar(s.Layers)
}

// AlignStateStats captures summary statistics for align-mode state checkpoints.
type AlignStateStats struct {
	BlockSize        int   `json:"block_size"`
	CheckpointsCount int   `json:"checkpoints_count"`
	TotalTokens      int   `json:"total_tokens"`
	TotalBytes       int64 `json:"total_bytes"`
}

// AlignStateManager manages block-aligned recurrent state checkpoints for GDN layers.
type AlignStateManager struct {
	mu           sync.RWMutex
	blockSize    int
	slabs        map[int]*RecurrentStateSlab
	tokenCount   int
	tokenIDs     []int
	activeLayers []GDNLayerState
}

// NewAlignStateManager creates a manager with the specified block size (default 16 if <= 0).
func NewAlignStateManager(blockSize int) *AlignStateManager {
	if blockSize <= 0 {
		blockSize = 16
	}
	return &AlignStateManager{
		blockSize: blockSize,
		slabs:     make(map[int]*RecurrentStateSlab),
	}
}

// BlockSize returns the configured block size in tokens.
func (m *AlignStateManager) BlockSize() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.blockSize
}

// TokenCount returns the current cumulative token count.
func (m *AlignStateManager) TokenCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.tokenCount
}

// ActiveLayers returns a deep copy of the active mutable recurrent layers.
func (m *AlignStateManager) ActiveLayers() []GDNLayerState {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return cloneSidecar(m.activeLayers)
}

// Step advances the token counter and updates active state. On block boundaries (T > 0 && T % blockSize == 0),
// it freezes the current state into slab B = (T/blockSize) - 1, inserts it into the slabs map,
// and performs copy-forward so the active state is ready for block B+1.
func (m *AlignStateManager) Step(tokensDelta int, currentLayers []GDNLayerState, tokenIDs []int) (checkpointed bool, blockIdx int, err error) {
	if tokensDelta < 0 {
		return false, -1, fmt.Errorf("model: tokensDelta must be non-negative, got %d", tokensDelta)
	}
	if tokensDelta == 0 {
		return false, -1, nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	m.tokenCount += tokensDelta
	T := m.tokenCount

	if len(tokenIDs) == T {
		m.tokenIDs = append([]int(nil), tokenIDs...)
	} else if len(tokenIDs) == tokensDelta {
		m.tokenIDs = append(m.tokenIDs, tokenIDs...)
	} else if len(tokenIDs) > 0 {
		m.tokenIDs = append([]int(nil), tokenIDs...)
	}

	if T > 0 && T%m.blockSize == 0 {
		B := (T / m.blockSize) - 1
		var slabToks []int
		if len(m.tokenIDs) >= T {
			slabToks = append([]int(nil), m.tokenIDs[:T]...)
		} else if len(tokenIDs) > 0 {
			slabToks = append([]int(nil), tokenIDs...)
		}

		slab := NewRecurrentStateSlab(B, T, slabToks, currentLayers)
		slab.Freeze()
		m.slabs[B] = slab

		// Copy-forward: active state gets an independent mutable deep copy ready for block B+1.
		m.activeLayers = slab.ToSidecar()
		return true, B, nil
	}

	m.activeLayers = cloneSidecar(currentLayers)
	return false, -1, nil
}

// CheckpointBlock explicitly checkpoints a block with specified index, tokens, token IDs, and layers.
func (m *AlignStateManager) CheckpointBlock(blockIndex int, tokenCount int, tokenIDs []int, layers []GDNLayerState) (*RecurrentStateSlab, error) {
	if blockIndex < 0 {
		return nil, fmt.Errorf("model: invalid block index %d", blockIndex)
	}
	if tokenCount <= 0 {
		tokenCount = (blockIndex + 1) * m.blockSize
	}
	m.mu.Lock()
	defer m.mu.Unlock()

	var tokCopy []int
	if len(tokenIDs) > 0 {
		tokCopy = append([]int(nil), tokenIDs...)
	}
	slab := NewRecurrentStateSlab(blockIndex, tokenCount, tokCopy, layers)
	slab.Freeze()
	m.slabs[blockIndex] = slab

	if tokenCount > m.tokenCount {
		m.tokenCount = tokenCount
		if len(tokCopy) > 0 {
			m.tokenIDs = append([]int(nil), tokCopy...)
		}
	}
	m.activeLayers = slab.ToSidecar()
	return slab, nil
}

// HasCheckpoint reports whether a checkpoint slab exists for the given block index.
func (m *AlignStateManager) HasCheckpoint(blockIndex int) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, ok := m.slabs[blockIndex]
	return ok
}

// GetSlab returns an independent deep copy of the checkpoint slab for the block index in O(1) time.
func (m *AlignStateManager) GetSlab(blockIndex int) (*RecurrentStateSlab, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	slab, ok := m.slabs[blockIndex]
	if !ok {
		return nil, false
	}
	return slab.Clone(), true
}

// RestoreAtTokens returns a deep copy of the recurrent layers checkpointed at tokenCount.
func (m *AlignStateManager) RestoreAtTokens(tokenCount int) ([]GDNLayerState, error) {
	if tokenCount <= 0 {
		return nil, fmt.Errorf("model: token count must be positive, got %d", tokenCount)
	}
	m.mu.RLock()
	defer m.mu.RUnlock()

	if tokenCount%m.blockSize != 0 {
		return nil, fmt.Errorf("model: token count %d is not aligned to block size %d", tokenCount, m.blockSize)
	}
	blockIdx := (tokenCount / m.blockSize) - 1
	slab, ok := m.slabs[blockIdx]
	if !ok {
		return nil, fmt.Errorf("model: checkpoint slab for block %d (tokens %d) not found", blockIdx, tokenCount)
	}
	return slab.ToSidecar(), nil
}

// TotalBytes returns the cumulative memory footprint in bytes across all stored checkpoint slabs.
func (m *AlignStateManager) TotalBytes() int64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var total int64
	for _, slab := range m.slabs {
		total += slab.SizeBytes
	}
	return total
}

// Stats returns a summary of the state manager's checkpoints and memory.
func (m *AlignStateManager) Stats() AlignStateStats {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var totalBytes int64
	for _, slab := range m.slabs {
		totalBytes += slab.SizeBytes
	}
	return AlignStateStats{
		BlockSize:        m.blockSize,
		CheckpointsCount: len(m.slabs),
		TotalTokens:      m.tokenCount,
		TotalBytes:       totalBytes,
	}
}

// AlignRadixTree is a prefix tree for block-aligned token sequences with recurrent state checkpoints.
type AlignRadixTree struct {
	mu        sync.RWMutex
	blockSize int
	root      *alignRadixNode
}

type alignRadixNode struct {
	tokens   []int
	slab     *RecurrentStateSlab
	children map[string]*alignRadixNode
}

// NewAlignRadixTree creates a new radix tree with the given block size.
func NewAlignRadixTree(blockSize int) *AlignRadixTree {
	if blockSize <= 0 {
		blockSize = 16
	}
	return &AlignRadixTree{
		blockSize: blockSize,
		root: &alignRadixNode{
			children: make(map[string]*alignRadixNode),
		},
	}
}

// BlockSize returns the configured block size in tokens.
func (r *AlignRadixTree) BlockSize() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.blockSize
}

func encodeBlockKey(tokens []int) string {
	b := make([]byte, len(tokens)*8)
	for i, v := range tokens {
		u := uint64(v)
		b[i*8] = byte(u)
		b[i*8+1] = byte(u >> 8)
		b[i*8+2] = byte(u >> 16)
		b[i*8+3] = byte(u >> 24)
		b[i*8+4] = byte(u >> 32)
		b[i*8+5] = byte(u >> 40)
		b[i*8+6] = byte(u >> 48)
		b[i*8+7] = byte(u >> 56)
	}
	return string(b)
}

// InsertSequence inserts a block-aligned token sequence and its corresponding recurrent slabs into the radix tree.
func (r *AlignRadixTree) InsertSequence(tokenIDs []int, slabs []*RecurrentStateSlab) error {
	if len(tokenIDs) == 0 {
		return errors.New("model: token sequence cannot be empty")
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	nBlocks := len(tokenIDs) / r.blockSize
	if nBlocks == 0 {
		return fmt.Errorf("model: token sequence length %d shorter than block size %d", len(tokenIDs), r.blockSize)
	}
	if len(slabs) < nBlocks {
		return fmt.Errorf("model: expected at least %d slabs for %d tokens (block size %d), got %d", nBlocks, len(tokenIDs), r.blockSize, len(slabs))
	}

	curr := r.root
	for b := 0; b < nBlocks; b++ {
		slab := slabs[b]
		if slab == nil {
			return fmt.Errorf("model: slab at index %d is nil", b)
		}
		toks := tokenIDs[b*r.blockSize : (b+1)*r.blockSize]
		key := encodeBlockKey(toks)

		child, ok := curr.children[key]
		if !ok {
			child = &alignRadixNode{
				tokens:   append([]int(nil), toks...),
				slab:     slab.Clone(),
				children: make(map[string]*alignRadixNode),
			}
			curr.children[key] = child
		} else {
			child.slab = slab.Clone()
		}
		curr = child
	}
	return nil
}

// MatchLongestPrefix matches the longest block-aligned common prefix in the tree for the given token sequence.
// Returns the number of matched blocks, the corresponding token count, and the slab at that block boundary.
func (r *AlignRadixTree) MatchLongestPrefix(tokenIDs []int) (matchedBlocks int, matchedTokens int, slab *RecurrentStateSlab) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	nBlocks := len(tokenIDs) / r.blockSize
	if nBlocks == 0 {
		return 0, 0, nil
	}

	curr := r.root
	var lastSlab *RecurrentStateSlab
	matched := 0

	for b := 0; b < nBlocks; b++ {
		toks := tokenIDs[b*r.blockSize : (b+1)*r.blockSize]
		key := encodeBlockKey(toks)

		child, ok := curr.children[key]
		if !ok {
			break
		}
		matched++
		lastSlab = child.slab
		curr = child
	}

	return matched, matched * r.blockSize, lastSlab.Clone()
}
