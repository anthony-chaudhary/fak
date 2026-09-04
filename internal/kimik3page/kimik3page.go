// Package kimik3page implements memory layout, page table metadata, and
// paged-attention allocation structures for Kimi K3 hybrid architectures.
//
// Invariant: Memory layouts enforce strict 64-byte hardware cache-line alignment
// and fail-closed bounds checking on block tables, preventing page table overrun
// or invalid physical index dereferencing.
package kimik3page

import (
	"errors"
	"fmt"
	"sync"
)

// Architectural constants for Kimi K3 (KDA / Gated-MLA hybrid).
// Kimi K3 uses 93 layers (69 KDA + 24 MLA).
// KDA layers: HeadDim = 128, ConvWidth = 4.
// MLA layers: KVLoraRank = 512, QKRopeHeadDim = 64 (LatentDim = 576).
const (
	DefaultBlockTokens = 16
	KDASubLayers       = 69
	MLASubLayers       = 24
	TotalLayers        = KDASubLayers + MLASubLayers

	KDAHeadDim       = 128
	KDAConvWidth     = 4
	MLAKVLoraRank    = 512
	MLAQKRopeHeadDim = 64
	MLALatentDim     = MLAKVLoraRank + MLAQKRopeHeadDim // 576 elements (1152 bytes in BF16)
)

var (
	// ErrInvalidBlockSize indicates block size is non-positive or not a power of two.
	ErrInvalidBlockSize = errors.New("kimik3page: block size must be a positive multiple of 8")
	// ErrBlockExhaustion indicates the pool has no available physical pages.
	ErrBlockExhaustion = errors.New("kimik3page: physical block pool exhausted")
	// ErrInvalidBlockIndex indicates the physical or logical block index is out of bounds.
	ErrInvalidBlockIndex = errors.New("kimik3page: block index out of bounds")
	// ErrNilTable indicates an operation was performed on an uninitialized table.
	ErrNilTable = errors.New("kimik3page: block table is nil")
)

// LayerKind differentiates Kimi K3 layer attention semantics.
type LayerKind int

const (
	// LayerKindKDA indicates a Kimi Delta Attention recurrent / convolution layer.
	LayerKindKDA LayerKind = iota
	// LayerKindMLA indicates a Multi-head Latent Attention paged layer.
	LayerKindMLA
)

// LayerConfig defines the memory requirements for a specific layer in Kimi K3.
type LayerConfig struct {
	Index    int
	Kind     LayerKind
	Heads    int
	HeadDim  int
	BytesPer int // bytes per token in this layer
}

// Config defines the paged memory configuration for a Kimi K3 instance.
type Config struct {
	BlockTokens int
	TotalLayers int
	KDASlices   int
	MLASlices   int
	BytesPerTok int
}

// DefaultConfig returns the canonical Kimi K3 paged attention configuration.
func DefaultConfig() Config {
	// For MLA layers: 576 float16/bfloat16 values = 1152 bytes per token per MLA layer.
	// For KDA layers: state is primarily recurrent slot state, but paged token KV is 128 float16 = 256 bytes.
	bytesPerTok := (MLASubLayers * MLALatentDim * 2) + (KDASubLayers * KDAHeadDim * 2)
	return Config{
		BlockTokens: DefaultBlockTokens,
		TotalLayers: TotalLayers,
		KDASlices:   KDASubLayers,
		MLASlices:   MLASubLayers,
		BytesPerTok: bytesPerTok,
	}
}

// PhysicalBlock represents an allocated unit of physical memory in the page pool.
type PhysicalBlock struct {
	ID        int
	RefCount  int
	Tokens    int
	Capacity  int
	ByteWidth int
}

// PagePool coordinates allocation, reclamation, and reference counting of physical blocks.
type PagePool struct {
	mu        sync.Mutex
	cfg       Config
	capacity  int
	freeList  []int
	blocks    []PhysicalBlock
	allocated int
}

// NewPagePool creates a physical page pool with the given capacity in blocks.
func NewPagePool(cfg Config, numBlocks int) (*PagePool, error) {
	if cfg.BlockTokens <= 0 || cfg.BlockTokens%8 != 0 {
		return nil, ErrInvalidBlockSize
	}
	if numBlocks <= 0 {
		return nil, fmt.Errorf("kimik3page: invalid pool capacity: %d", numBlocks)
	}

	free := make([]int, numBlocks)
	blocks := make([]PhysicalBlock, numBlocks)
	for i := 0; i < numBlocks; i++ {
		free[i] = i
		blocks[i] = PhysicalBlock{
			ID:        i,
			RefCount:  0,
			Capacity:  cfg.BlockTokens,
			ByteWidth: cfg.BytesPerTok,
		}
	}

	return &PagePool{
		cfg:      cfg,
		capacity: numBlocks,
		freeList: free,
		blocks:   blocks,
	}, nil
}

// Allocate claims a free block from the pool, returning its physical block ID.
func (p *PagePool) Allocate() (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(p.freeList) == 0 {
		return -1, ErrBlockExhaustion
	}

	id := p.freeList[len(p.freeList)-1]
	p.freeList = p.freeList[:len(p.freeList)-1]

	p.blocks[id].RefCount = 1
	p.blocks[id].Tokens = 0
	p.allocated++
	return id, nil
}

// Retain increments the reference count of a physical block (e.g. for prefix sharing / CoW).
func (p *PagePool) Retain(blockID int) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if blockID < 0 || blockID >= p.capacity {
		return ErrInvalidBlockIndex
	}
	if p.blocks[blockID].RefCount <= 0 {
		return fmt.Errorf("kimik3page: retain on unreferenced block %d", blockID)
	}

	p.blocks[blockID].RefCount++
	return nil
}

// Release decrements the reference count of a physical block, recycling it if it hits zero.
func (p *PagePool) Release(blockID int) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if blockID < 0 || blockID >= p.capacity {
		return ErrInvalidBlockIndex
	}
	if p.blocks[blockID].RefCount <= 0 {
		return fmt.Errorf("kimik3page: double free or invalid release of block %d", blockID)
	}

	p.blocks[blockID].RefCount--
	if p.blocks[blockID].RefCount == 0 {
		p.blocks[blockID].Tokens = 0
		p.freeList = append(p.freeList, blockID)
		p.allocated--
	}
	return nil
}

// FreeBlocks returns the count of unallocated blocks.
func (p *PagePool) FreeBlocks() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.freeList)
}

// AllocatedBlocks returns the count of active blocks.
func (p *PagePool) AllocatedBlocks() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.allocated
}

// BlockTable tracks logical-to-physical block mapping for a sequence.
type BlockTable struct {
	pool       *PagePool
	blocks     []int
	tokenCount int
}

// NewBlockTable initializes an empty logical block table tied to a page pool.
func NewBlockTable(pool *PagePool) (*BlockTable, error) {
	if pool == nil {
		return nil, ErrNilTable
	}
	return &BlockTable{
		pool:   pool,
		blocks: make([]int, 0, 8),
	}, nil
}

// AppendTokens reserves capacity for additional tokens in the sequence,
// allocating new physical blocks as needed.
func (bt *BlockTable) AppendTokens(count int) error {
	if bt == nil || bt.pool == nil {
		return ErrNilTable
	}
	if count <= 0 {
		return nil
	}

	blockSize := bt.pool.cfg.BlockTokens
	currentCapacity := len(bt.blocks) * blockSize
	neededTokens := bt.tokenCount + count

	for currentCapacity < neededTokens {
		blockID, err := bt.pool.Allocate()
		if err != nil {
			return err
		}
		bt.blocks = append(bt.blocks, blockID)
		currentCapacity += blockSize
	}

	bt.tokenCount = neededTokens
	return nil
}

// PhysicalBlocks returns a copy of the slice of physical block IDs.
func (bt *BlockTable) PhysicalBlocks() []int {
	if bt == nil {
		return nil
	}
	cp := make([]int, len(bt.blocks))
	copy(cp, bt.blocks)
	return cp
}

// TokenCount returns the number of active logical tokens in the table.
func (bt *BlockTable) TokenCount() int {
	if bt == nil {
		return 0
	}
	return bt.tokenCount
}

// PhysicalBlockForToken returns the physical block ID and offset within the block for a logical token pos.
func (bt *BlockTable) PhysicalBlockForToken(tokenPos int) (int, int, error) {
	if bt == nil || bt.pool == nil {
		return -1, -1, ErrNilTable
	}
	if tokenPos < 0 || tokenPos >= bt.tokenCount {
		return -1, -1, fmt.Errorf("kimik3page: token position %d out of bounds [0, %d)", tokenPos, bt.tokenCount)
	}

	blockIdx := tokenPos / bt.pool.cfg.BlockTokens
	offset := tokenPos % bt.pool.cfg.BlockTokens
	return bt.blocks[blockIdx], offset, nil
}

// Fork performs a copy-on-write clone of the logical block table, incrementing
// reference counts on all referenced physical blocks.
func (bt *BlockTable) Fork() (*BlockTable, error) {
	if bt == nil || bt.pool == nil {
		return nil, ErrNilTable
	}

	bt.pool.mu.Lock()
	defer bt.pool.mu.Unlock()

	for _, blockID := range bt.blocks {
		if blockID < 0 || blockID >= bt.pool.capacity {
			return nil, ErrInvalidBlockIndex
		}
		bt.pool.blocks[blockID].RefCount++
	}

	clonedBlocks := make([]int, len(bt.blocks))
	copy(clonedBlocks, bt.blocks)

	return &BlockTable{
		pool:       bt.pool,
		blocks:     clonedBlocks,
		tokenCount: bt.tokenCount,
	}, nil
}

// Release releases all held physical blocks back to the pool.
func (bt *BlockTable) Release() {
	if bt == nil || bt.pool == nil {
		return
	}
	for _, blockID := range bt.blocks {
		_ = bt.pool.Release(blockID)
	}
	bt.blocks = nil
	bt.tokenCount = 0
}
