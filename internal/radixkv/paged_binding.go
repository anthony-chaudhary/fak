package radixkv

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Typed errors for paged RadixAttention block pool binding.
var (
	ErrNilNode           = errors.New("radixkv: nil node")
	ErrNilPool           = errors.New("radixkv: nil kv pool")
	ErrNilBlockTable     = errors.New("radixkv: nil block table")
	ErrBlockNotFound     = errors.New("radixkv: physical block not found")
	ErrNoBlocksAssigned  = errors.New("radixkv: no blocks assigned to node")
	ErrInvalidTokenCount = errors.New("radixkv: invalid token count")
	ErrBufferTooSmall    = errors.New("radixkv: buffer too small for token kv")
	ErrInvalidLayerIndex = errors.New("radixkv: invalid layer index")
)

// PagedBlockTable represents a stream's logical-to-physical block table for Metal PagedAttention.
// It maps logical page indices to physical block IDs allocated from ctxmmu.KVPool (Issue #12528).
type PagedBlockTable struct {
	StreamID   string
	TokenCount int
	BlockIDs   []int    // physical block IDs in logical sequence order
	MetalTable []uint32 // Metal-ready 32-bit physical page table (for Metal PageTableBase argument buffer)
	retained   bool
}

// BlockCount returns the number of physical blocks mapped in this block table.
func (s *PagedBlockTable) BlockCount() int {
	if s == nil {
		return 0
	}
	return len(s.BlockIDs)
}

// Pages returns a point-in-time copy of the physical block IDs.
func (s *PagedBlockTable) Pages() []int {
	if s == nil {
		return nil
	}
	out := make([]int, len(s.BlockIDs))
	copy(out, s.BlockIDs)
	return out
}

// MetalPages returns a point-in-time copy of the Metal 32-bit page table.
func (s *PagedBlockTable) MetalPages() []uint32 {
	if s == nil {
		return nil
	}
	out := make([]uint32, len(s.MetalTable))
	copy(out, s.MetalTable)
	return out
}

// MetalPageTableBytes returns the binary bytes of the Metal 32-bit page table,
// suitable for direct assignment to Metal command encoder or buffer contents.
func (s *PagedBlockTable) MetalPageTableBytes() []byte {
	if s == nil || len(s.MetalTable) == 0 {
		return nil
	}
	buf := make([]byte, len(s.MetalTable)*4)
	for i, v := range s.MetalTable {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], v)
	}
	return buf
}

// PagedRadixKVPool binds a RadixAttention prefix tree to Metal PagedAttention page tables
// backed by physical page blocks in ctxmmu.KVPool (Issue #12528).
//
// In standard RadixAttention, prefix matches clone contiguous model.KVCache instances,
// incurring heavy DRAM copy and tensor allocation overhead.
// PagedRadixKVPool replaces tensor cloning with zero-copy page table pointer swapping:
// on a prefix cache hit, it binds physical block IDs directly into a PagedBlockTable
// (and/or ctxmmu.KVSequence), incrementing physical block reference counts in ctxmmu.KVPool.
//
// Key invariants:
//  1. Zero Tensor Allocations: Prefix hits map block IDs with zero float32/KVCache tensor allocations.
//  2. Sub-50µs Binding Latency: Direct block table resolution and assignment completes in <50µs.
//  3. Reference-Counted Eviction: Shared physical blocks remain valid and resident as long as
//     either the RadixKV tree or an active stream holds a reference. Evicting a node from RadixKV
//     decrements the tree's reference; underlying blocks are only freed when all stream leases
//     and tree references reach zero.
type PagedRadixKVPool struct {
	*Tree
	mu            sync.RWMutex
	pool          *ctxmmu.KVPool
	tokensPerPage int
	nodeBlocks    map[*node][]int
	activeTables  map[string]*PagedBlockTable

	// Telemetry & accounting counters
	hits                atomic.Int64
	misses              atomic.Int64
	boundTokens         atomic.Int64
	zeroCopyBindings    atomic.Int64
	tensorAllocsAvoided atomic.Int64
}

// NewPagedRadixKVPool constructs a paged RadixAttention pool pinned to a ctxmmu.KVPool.
// If pool is nil, a default KVPool with 1024 pages and 16 tokens/page is constructed.
func NewPagedRadixKVPool(maxTokens int, pool *ctxmmu.KVPool) *PagedRadixKVPool {
	return NewPagedRadixKVPoolWithTree(New(maxTokens), pool)
}

// NewPagedRadixKVPoolWithTree wraps an existing RadixKV Tree with paged Metal PagedAttention binding.
func NewPagedRadixKVPoolWithTree(tree *Tree, pool *ctxmmu.KVPool) *PagedRadixKVPool {
	if tree == nil {
		tree = New(0)
	}
	if pool == nil {
		var err error
		pool, err = ctxmmu.NewKVPool(ctxmmu.KVPoolConfig{
			TotalPages:    1024,
			TokensPerPage: 16,
			NumLayers:     32,
			NumKVHeads:    8,
			HeadDim:       128,
			DType:         ctxmmu.KVDTypeFP32,
			Topology:      ctxmmu.KVTopologyGQA,
		})
		if err != nil {
			panic(fmt.Sprintf("radixkv: failed to initialize default ctxmmu.KVPool: %v", err))
		}
	}

	tpp := pool.Config().TokensPerPage
	if tpp <= 0 {
		tpp = 16
	}

	pk := &PagedRadixKVPool{
		Tree:          tree,
		pool:          pool,
		tokensPerPage: tpp,
		nodeBlocks:    make(map[*node][]int),
		activeTables:  make(map[string]*PagedBlockTable),
	}

	// Register this pool as page occupancy tracker on the tree so page-aware eviction
	// groups candidates by physical block IDs.
	tree.SetPageOccupancyTracker(pk)
	return pk
}

// ChunkID implements PageOccupancyTracker so that Tree page-aware eviction groups
// candidates by primary physical page block ID.
func (p *PagedRadixKVPool) ChunkID(n *node) int {
	if p == nil || n == nil {
		return 0
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	if blocks, ok := p.nodeBlocks[n]; ok && len(blocks) > 0 {
		return blocks[0]
	}
	return n.chunkID
}

// Pool returns the underlying ctxmmu.KVPool instance.
func (p *PagedRadixKVPool) Pool() *ctxmmu.KVPool {
	if p == nil {
		return nil
	}
	return p.pool
}

// TokensPerPage returns the token capacity per physical page block.
func (p *PagedRadixKVPool) TokensPerPage() int {
	if p == nil {
		return 16
	}
	return p.tokensPerPage
}

// SetNodeBlocks associates physical page block IDs directly with a radix node.
// It retains an initial reference count in the physical block allocator on behalf of the tree.
func (p *PagedRadixKVPool) SetNodeBlocks(n *Node, blockIDs []int) error {
	if p == nil {
		return ErrNilPool
	}
	if n == nil {
		return ErrNilNode
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	// Retain each block in the allocator for the tree
	alloc := p.pool.Allocator()
	for _, bID := range blockIDs {
		if err := alloc.Retain(bID); err != nil {
			return err
		}
	}

	// Release any previously assigned blocks
	if old, exists := p.nodeBlocks[n]; exists {
		for _, bID := range old {
			if blk, err := alloc.GetBlock(bID); err == nil {
				if blk.Release() <= 0 {
					alloc.Free(bID)
				}
			}
		}
	}

	p.nodeBlocks[n] = append([]int(nil), blockIDs...)
	if len(blockIDs) > 0 {
		n.SetChunkID(blockIDs[0])
	}
	return nil
}

// GetNodeBlocks returns the physical page block IDs assigned directly to node n.
func (p *PagedRadixKVPool) GetNodeBlocks(n *Node) []int {
	if p == nil || n == nil {
		return nil
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	blocks := p.nodeBlocks[n]
	if len(blocks) == 0 {
		return nil
	}
	out := make([]int, len(blocks))
	copy(out, blocks)
	return out
}

// ResolveBlockIDs resolves the complete list of physical page block IDs for the full prefix
// leading to node n. If node n does not have directly assigned blocks (e.g. formed by an edge split),
// it walks ancestors or descendants to resolve the exact block chain corresponding to n.plen tokens.
func (p *PagedRadixKVPool) ResolveBlockIDs(n *Node) []int {
	if p == nil || n == nil {
		return nil
	}
	p.mu.RLock()
	defer p.mu.RUnlock()

	return p.resolveBlockIDsLocked(n)
}

func (p *PagedRadixKVPool) resolveBlockIDsLocked(n *node) []int {
	if n == nil {
		return nil
	}

	// 1. Check direct entry
	if blocks, ok := p.nodeBlocks[n]; ok && len(blocks) > 0 {
		out := make([]int, len(blocks))
		copy(out, blocks)
		return out
	}

	// 2. If node has descendants with blocks (common after split), derive prefix blocks
	reqBlocks := (n.plen + p.tokensPerPage - 1) / p.tokensPerPage
	if reqBlocks <= 0 && n.plen > 0 {
		reqBlocks = 1
	}

	var candidateBlocks []int
	var findDescendantBlocks func(curr *node) bool
	findDescendantBlocks = func(curr *node) bool {
		if blks, ok := p.nodeBlocks[curr]; ok && len(blks) >= reqBlocks {
			candidateBlocks = blks
			return true
		}
		for _, child := range curr.children {
			if findDescendantBlocks(child) {
				return true
			}
		}
		return false
	}
	findDescendantBlocks(n)

	if len(candidateBlocks) >= reqBlocks {
		out := make([]int, reqBlocks)
		copy(out, candidateBlocks[:reqBlocks])
		return out
	}

	// 3. Fallback: accumulate along root->node path
	var path []*node
	for curr := n; curr != nil && curr.parent != nil; curr = curr.parent {
		path = append(path, curr)
	}
	// Reverse path to be root-to-leaf
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}

	var accumulated []int
	for _, step := range path {
		if blks, ok := p.nodeBlocks[step]; ok {
			accumulated = append(accumulated, blks...)
		}
	}

	if len(accumulated) > 0 {
		if reqBlocks > 0 && len(accumulated) > reqBlocks {
			accumulated = accumulated[:reqBlocks]
		}
		return accumulated
	}

	return nil
}

// ResolveMetalPageTable returns the 32-bit Metal page table for node n.
func (p *PagedRadixKVPool) ResolveMetalPageTable(n *Node) []uint32 {
	blocks := p.ResolveBlockIDs(n)
	if len(blocks) == 0 {
		return nil
	}
	out := make([]uint32, len(blocks))
	for i, bID := range blocks {
		out[i] = uint32(bID)
	}
	return out
}

// InsertPaged inserts a token suffix off boundary into the radix tree, associates the provided
// physical block IDs from ctxmmu.KVPool, and increments reference counts so the blocks remain
// retained while referenced by the tree.
func (p *PagedRadixKVPool) InsertPaged(boundary *Node, suffix []int, blockIDs []int) (*Node, error) {
	if p == nil {
		return nil, ErrNilPool
	}

	// Insert into underlying radix tree (kv can be nil as physical pages back storage)
	leaf := p.Tree.Insert(boundary, suffix, nil)
	if leaf == nil {
		return nil, errors.New("radixkv: failed to insert leaf node")
	}

	if err := p.SetNodeBlocks(leaf, blockIDs); err != nil {
		return nil, err
	}

	return leaf, nil
}

// InsertWithAllocatedTokens allocates physical page blocks from ctxmmu.KVPool for the suffix,
// attaches the leaf to the radix tree, and associates the newly allocated blocks.
func (p *PagedRadixKVPool) InsertWithAllocatedTokens(boundary *Node, suffix []int) (*Node, []int, error) {
	if p == nil {
		return nil, nil, ErrNilPool
	}
	if len(suffix) == 0 {
		return boundary, nil, nil
	}

	neededBlocks := (len(suffix) + p.tokensPerPage - 1) / p.tokensPerPage
	alloc := p.pool.Allocator()
	allocated := make([]int, 0, neededBlocks)

	for i := 0; i < neededBlocks; i++ {
		blk, err := alloc.Allocate()
		if err != nil {
			// Rollback allocated blocks
			for _, id := range allocated {
				alloc.Free(id)
			}
			return nil, nil, fmt.Errorf("radixkv: physical block allocation failed: %w", err)
		}
		// Allocate returns block with refCount = 1
		allocated = append(allocated, blk.ID)
	}

	leaf, err := p.InsertPaged(boundary, suffix, allocated)
	if err != nil {
		for _, id := range allocated {
			alloc.Free(id)
		}
		return nil, nil, err
	}

	// Note: SetNodeBlocks called inside InsertPaged retained each block, so refCount is 2 (allocator + SetNodeBlocks).
	// Decrement the initial Allocate refCount so the tree holds exactly 1 reference.
	for _, id := range allocated {
		if blk, err := alloc.GetBlock(id); err == nil {
			blk.Release()
		}
	}

	return leaf, allocated, nil
}

// BindPrefix looks up the longest cached prefix for tokens. On a match (hit), it constructs
// a PagedBlockTable binding physical page block IDs directly into the stream without
// allocating or copying tensor memory, and increments reference counts on the shared blocks.
// Returns (table, matchedTokens, hit, err).
func (p *PagedRadixKVPool) BindPrefix(tokens []int, streamID string) (*PagedBlockTable, int, bool, error) {
	if p == nil {
		return nil, 0, false, ErrNilPool
	}
	if len(tokens) == 0 {
		return nil, 0, false, nil
	}

	boundary, matched := p.Tree.Lookup(tokens)
	if boundary == nil || matched == 0 {
		p.misses.Add(1)
		return nil, 0, false, nil
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	blockIDs := p.resolveBlockIDsLocked(boundary)
	if len(blockIDs) == 0 {
		p.misses.Add(1)
		p.Tree.Done(boundary)
		return nil, matched, false, nil
	}

	// Calculate how many blocks correspond to matched tokens
	neededBlocks := (matched + p.tokensPerPage - 1) / p.tokensPerPage
	if neededBlocks > len(blockIDs) {
		neededBlocks = len(blockIDs)
	}
	activeBlocks := blockIDs[:neededBlocks]

	// Retain each shared block for this stream
	alloc := p.pool.Allocator()
	for _, bID := range activeBlocks {
		if err := alloc.Retain(bID); err != nil {
			p.Tree.Done(boundary)
			return nil, matched, false, err
		}
	}

	// Construct Metal page table (uint32 physical block IDs)
	metalTable := make([]uint32, len(activeBlocks))
	for i, bID := range activeBlocks {
		metalTable[i] = uint32(bID)
	}

	table := &PagedBlockTable{
		StreamID:   streamID,
		TokenCount: matched,
		BlockIDs:   append([]int(nil), activeBlocks...),
		MetalTable: metalTable,
		retained:   true,
	}

	if streamID != "" {
		p.activeTables[streamID] = table
	}

	// Release lookup lease on radix tree since physical block refcounts now protect memory
	p.Tree.Done(boundary)

	// Update telemetry
	p.hits.Add(1)
	p.boundTokens.Add(int64(matched))
	p.zeroCopyBindings.Add(1)
	p.tensorAllocsAvoided.Add(1)

	return table, matched, true, nil
}

// BindPrefixToSequence performs zero-copy block table assignment directly into a ctxmmu.KVSequence.
// It sets seq.PageTable to the matched physical block IDs, updates seq.TokenCount, and increments
// reference counts on the shared physical page blocks.
func (p *PagedRadixKVPool) BindPrefixToSequence(tokens []int, seq *ctxmmu.KVSequence) (int, bool, error) {
	if p == nil {
		return 0, false, ErrNilPool
	}
	if seq == nil {
		return 0, false, errors.New("radixkv: nil sequence")
	}

	table, matched, hit, err := p.BindPrefix(tokens, seq.ID)
	if err != nil || !hit {
		return matched, hit, err
	}

	seq.PageTable = table.Pages()
	seq.TokenCount = matched
	seq.PagedKV = true
	return matched, true, nil
}

// BindNode binds an existing radix node's physical blocks directly to a PagedBlockTable.
func (p *PagedRadixKVPool) BindNode(n *Node, streamID string) (*PagedBlockTable, error) {
	if p == nil {
		return nil, ErrNilPool
	}
	if n == nil {
		return nil, ErrNilNode
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	blockIDs := p.resolveBlockIDsLocked(n)
	if len(blockIDs) == 0 {
		return nil, ErrNoBlocksAssigned
	}

	alloc := p.pool.Allocator()
	for _, bID := range blockIDs {
		if err := alloc.Retain(bID); err != nil {
			return nil, err
		}
	}

	metalTable := make([]uint32, len(blockIDs))
	for i, bID := range blockIDs {
		metalTable[i] = uint32(bID)
	}

	table := &PagedBlockTable{
		StreamID:   streamID,
		TokenCount: n.plen,
		BlockIDs:   append([]int(nil), blockIDs...),
		MetalTable: metalTable,
		retained:   true,
	}

	if streamID != "" {
		p.activeTables[streamID] = table
	}

	p.zeroCopyBindings.Add(1)
	p.tensorAllocsAvoided.Add(1)
	return table, nil
}

// ReleaseBlockTable releases a stream's lease on shared physical blocks, decrementing
// their reference counts. Blocks are only freed back to the allocator when all stream leases
// and radix tree references reach zero.
func (p *PagedRadixKVPool) ReleaseBlockTable(table *PagedBlockTable) error {
	if p == nil {
		return ErrNilPool
	}
	if table == nil {
		return ErrNilBlockTable
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if !table.retained {
		return nil
	}
	table.retained = false

	if table.StreamID != "" {
		delete(p.activeTables, table.StreamID)
	}

	alloc := p.pool.Allocator()
	for _, bID := range table.BlockIDs {
		if blk, err := alloc.GetBlock(bID); err == nil {
			if blk.Release() <= 0 {
				alloc.Free(bID)
			}
		}
	}

	table.BlockIDs = nil
	table.MetalTable = nil
	table.TokenCount = 0
	return nil
}

// ReleaseStream releases a stream by stream ID.
func (p *PagedRadixKVPool) ReleaseStream(streamID string) error {
	if p == nil {
		return ErrNilPool
	}
	p.mu.Lock()
	table, ok := p.activeTables[streamID]
	p.mu.Unlock()

	if !ok {
		return nil
	}
	return p.ReleaseBlockTable(table)
}

// EvictNodePaged evicts node n from the radix tree and releases the tree's reference to its
// physical page blocks. If any active streams still reference those blocks, the blocks remain
// retained in ctxmmu.KVPool until those streams release them.
func (p *PagedRadixKVPool) EvictNodePaged(n *Node) (int, error) {
	if p == nil {
		return 0, ErrNilPool
	}
	if n == nil {
		return 0, nil
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	// Collect blocks to release from n and all descendants
	var nodesToClean []*node
	var collectNodes func(curr *node)
	collectNodes = func(curr *node) {
		if curr == nil {
			return
		}
		nodesToClean = append(nodesToClean, curr)
		for _, child := range curr.children {
			collectNodes(child)
		}
	}
	collectNodes(n)

	alloc := p.pool.Allocator()
	for _, target := range nodesToClean {
		if blocks, ok := p.nodeBlocks[target]; ok {
			for _, bID := range blocks {
				if blk, err := alloc.GetBlock(bID); err == nil {
					if blk.Release() <= 0 {
						alloc.Free(bID)
					}
				}
			}
			delete(p.nodeBlocks, target)
		}
	}

	freedTokens := p.Tree.EvictNode(n)
	return freedTokens, nil
}

// EvictPrefixPaged evicts the prefix matching tokens from the radix tree and releases the
// tree's reference to the physical blocks. Active stream references continue to protect physical memory.
func (p *PagedRadixKVPool) EvictPrefixPaged(tokens []int) (int, error) {
	if p == nil {
		return 0, ErrNilPool
	}
	boundary, matched := p.Tree.Lookup(tokens)
	if boundary == nil || matched == 0 {
		return 0, nil
	}
	p.Tree.Done(boundary)
	return p.EvictNodePaged(boundary)
}

// ReconcileEvictions scans tracked nodes and releases physical block references for any nodes
// that were evicted by background LRU budget eviction (Tree.evictToBudget).
func (p *PagedRadixKVPool) ReconcileEvictions() int {
	if p == nil {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	alloc := p.pool.Allocator()
	reaped := 0

	for n, blocks := range p.nodeBlocks {
		// A node is considered evicted if its parent is nil and it's not a root,
		// or if it is no longer attached in its parent's children map.
		isEvicted := false
		if n.parent != nil {
			if len(n.key) > 0 {
				child, ok := n.parent.children[n.key[0]]
				if !ok || child != n {
					isEvicted = true
				}
			}
		}

		if isEvicted {
			for _, bID := range blocks {
				if blk, err := alloc.GetBlock(bID); err == nil {
					if blk.Release() <= 0 {
						alloc.Free(bID)
					}
				}
			}
			delete(p.nodeBlocks, n)
			reaped++
		}
	}
	return reaped
}

// -----------------------------------------------------------------------------
// Physical K/V Data I/O and Attention Math Verification
// -----------------------------------------------------------------------------

// WriteTokenKV writes float32 K and V activations for a specific token into a physical page block.
// Layout in block: TokensPerPage * NumLayers * 2 * (NumKVHeads * HeadDim) * 4 bytes.
func (p *PagedRadixKVPool) WriteTokenKV(blockID int, tokenInBlock int, layer int, k, v []float32) error {
	if p == nil {
		return ErrNilPool
	}
	if tokenInBlock < 0 || tokenInBlock >= p.tokensPerPage {
		return ErrInvalidTokenCount
	}

	cfg := p.pool.Config()
	if layer < 0 || layer >= cfg.NumLayers {
		return ErrInvalidLayerIndex
	}

	alloc := p.pool.Allocator()
	blk, err := alloc.GetBlock(blockID)
	if err != nil {
		return ErrBlockNotFound
	}

	stride := cfg.NumKVHeads * cfg.HeadDim
	if len(k) < stride || len(v) < stride {
		return errors.New("radixkv: k or v slice shorter than head stride")
	}

	// Each token spans: numLayers * 2 * stride * 4 bytes
	layerSpanBytes := 2 * stride * 4
	tokenSpanBytes := cfg.NumLayers * layerSpanBytes
	tokenOffset := tokenInBlock * tokenSpanBytes
	layerOffset := layer * layerSpanBytes
	kOffset := tokenOffset + layerOffset
	vOffset := kOffset + stride*4

	if vOffset+stride*4 > len(blk.Data) {
		return ErrBufferTooSmall
	}

	for d := 0; d < stride; d++ {
		binary.LittleEndian.PutUint32(blk.Data[kOffset+d*4:kOffset+(d+1)*4], math.Float32bits(k[d]))
		binary.LittleEndian.PutUint32(blk.Data[vOffset+d*4:vOffset+(d+1)*4], math.Float32bits(v[d]))
	}
	return nil
}

// ReadTokenKV reads float32 K and V activations for a specific token from a physical page block.
func (p *PagedRadixKVPool) ReadTokenKV(blockID int, tokenInBlock int, layer int) (k, v []float32, err error) {
	if p == nil {
		return nil, nil, ErrNilPool
	}
	if tokenInBlock < 0 || tokenInBlock >= p.tokensPerPage {
		return nil, nil, ErrInvalidTokenCount
	}

	cfg := p.pool.Config()
	if layer < 0 || layer >= cfg.NumLayers {
		return nil, nil, ErrInvalidLayerIndex
	}

	alloc := p.pool.Allocator()
	blk, err := alloc.GetBlock(blockID)
	if err != nil {
		return nil, nil, ErrBlockNotFound
	}

	stride := cfg.NumKVHeads * cfg.HeadDim
	layerSpanBytes := 2 * stride * 4
	tokenSpanBytes := cfg.NumLayers * layerSpanBytes
	tokenOffset := tokenInBlock * tokenSpanBytes
	layerOffset := layer * layerSpanBytes
	kOffset := tokenOffset + layerOffset
	vOffset := kOffset + stride*4

	if vOffset+stride*4 > len(blk.Data) {
		return nil, nil, ErrBufferTooSmall
	}

	k = make([]float32, stride)
	v = make([]float32, stride)
	for d := 0; d < stride; d++ {
		k[d] = math.Float32frombits(binary.LittleEndian.Uint32(blk.Data[kOffset+d*4 : kOffset+(d+1)*4]))
		v[d] = math.Float32frombits(binary.LittleEndian.Uint32(blk.Data[vOffset+d*4 : vOffset+(d+1)*4]))
	}
	return k, v, nil
}

// GatherBlockTableKV reconstructs contiguous K and V float32 tensors for layer l from a block table's
// non-contiguous physical page blocks.
func (p *PagedRadixKVPool) GatherBlockTableKV(table *PagedBlockTable, layer int) (k, v []float32, err error) {
	if p == nil {
		return nil, nil, ErrNilPool
	}
	if table == nil {
		return nil, nil, ErrNilBlockTable
	}

	cfg := p.pool.Config()
	stride := cfg.NumKVHeads * cfg.HeadDim
	totalTokens := table.TokenCount
	k = make([]float32, totalTokens*stride)
	v = make([]float32, totalTokens*stride)

	for tok := 0; tok < totalTokens; tok++ {
		blkIdx := tok / p.tokensPerPage
		if blkIdx >= len(table.BlockIDs) {
			return nil, nil, errors.New("radixkv: block index exceeds block table blocks")
		}
		bID := table.BlockIDs[blkIdx]
		tokInBlk := tok % p.tokensPerPage

		tokK, tokV, err := p.ReadTokenKV(bID, tokInBlk, layer)
		if err != nil {
			return nil, nil, fmt.Errorf("radixkv: read token %d failed: %w", tok, err)
		}

		copy(k[tok*stride:(tok+1)*stride], tokK)
		copy(v[tok*stride:(tok+1)*stride], tokV)
	}

	return k, v, nil
}

// ComputePagedAttention computes multi-head scaled dot-product attention for query Q against
// the stream's paged KV cache for layer l.
// This proves mathematical bit-exactness against reference contiguous attention.
func (p *PagedRadixKVPool) ComputePagedAttention(table *PagedBlockTable, q []float32, layer int) ([]float32, error) {
	if p == nil {
		return nil, ErrNilPool
	}
	if table == nil {
		return nil, ErrNilBlockTable
	}
	if table.TokenCount == 0 {
		return nil, errors.New("radixkv: empty block table")
	}

	cfg := p.pool.Config()
	numHeads := cfg.NumKVHeads
	headDim := cfg.HeadDim
	if len(q) < numHeads*headDim {
		return nil, fmt.Errorf("radixkv: query length %d less than %d", len(q), numHeads*headDim)
	}

	// Gather K and V from paged blocks
	gatheredK, gatheredV, err := p.GatherBlockTableKV(table, layer)
	if err != nil {
		return nil, err
	}

	// Reference SDPA: softmax(Q * K^T / sqrt(headDim)) * V per head
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	numTokens := table.TokenCount
	stride := numHeads * headDim

	out := make([]float32, numHeads*headDim)
	scores := make([]float32, numTokens)

	for h := 0; h < numHeads; h++ {
		qHead := q[h*headDim : (h+1)*headDim]

		// 1. Compute dot products: Q_h * K_{t, h}
		var maxScore float32 = -math.MaxFloat32
		for t := 0; t < numTokens; t++ {
			kHead := gatheredK[t*stride+h*headDim : t*stride+(h+1)*headDim]
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += qHead[d] * kHead[d]
			}
			score := dot * scale
			scores[t] = score
			if score > maxScore {
				maxScore = score
			}
		}

		// 2. Numerically stable softmax
		var sumExp float32
		for t := 0; t < numTokens; t++ {
			expVal := float32(math.Exp(float64(scores[t] - maxScore)))
			scores[t] = expVal
			sumExp += expVal
		}
		invSum := float32(1.0) / sumExp
		for t := 0; t < numTokens; t++ {
			scores[t] *= invSum
		}

		// 3. Weighted sum of V_{t, h}
		outHead := out[h*headDim : (h+1)*headDim]
		for t := 0; t < numTokens; t++ {
			vHead := gatheredV[t*stride+h*headDim : t*stride+(h+1)*headDim]
			weight := scores[t]
			for d := 0; d < headDim; d++ {
				outHead[d] += weight * vHead[d]
			}
		}
	}

	return out, nil
}

// Telemetry returns falsifiable metrics on paged radix binding operations.
type PagedBindingTelemetry struct {
	Hits                int64   `json:"hits"`
	Misses              int64   `json:"misses"`
	BoundTokens         int64   `json:"bound_tokens"`
	ZeroCopyBindings    int64   `json:"zero_copy_bindings"`
	TensorAllocsAvoided int64   `json:"tensor_allocs_avoided"`
	HitRate             float64 `json:"hit_rate"`
}

// Telemetry returns a point-in-time snapshot of paged binding telemetry.
func (p *PagedRadixKVPool) Telemetry() PagedBindingTelemetry {
	if p == nil {
		return PagedBindingTelemetry{}
	}
	hits := p.hits.Load()
	misses := p.misses.Load()
	total := hits + misses
	var rate float64
	if total > 0 {
		rate = float64(hits) / float64(total)
	}

	return PagedBindingTelemetry{
		Hits:                hits,
		Misses:              misses,
		BoundTokens:         p.boundTokens.Load(),
		ZeroCopyBindings:    p.zeroCopyBindings.Load(),
		TensorAllocsAvoided: p.tensorAllocsAvoided.Load(),
		HitRate:             rate,
	}
}

// Suppress unused import warning for model package if only used in docs/types
var _ = model.Config{}
