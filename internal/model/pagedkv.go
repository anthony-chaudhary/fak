package model

// pagedkv.go — PagedAttention KV block allocator (#277, B-006, Track B · Performance).
//
// The kernel-owned KVCache (kvcache.go) stores each layer's K/V as ONE contiguous,
// per-sequence slice that grows on append. That layout is bit-exact and simple, but it
// pays vLLM-PagedAttention's two costs: a long-lived sequence must keep its KV in one
// growing run (so reserving the worst-case length wastes memory, and growing re-copies),
// and two sequences that share a common prefix cannot share the bytes — RadixAttention's
// tree shares a prefix only by holding a FULL-prefix KVCache *copy* per node (see
// docs/benchmarks/RADIXATTENTION-RESULTS.md: "a paged share would not" copy).
//
// PagedAttention is the orthogonal allocation discipline: cut KV into fixed-size physical
// blocks drawn from a shared pool, and give each sequence a PAGE TABLE — a logical→physical
// block map — instead of one contiguous run. That buys three things this file implements:
//
//   - Page-based KV allocation. A sequence allocates a block only when it crosses a block
//     boundary, so a variable-length sequence costs ceil(len/blockTokens) blocks with at
//     most ONE partial tail block of internal fragmentation — never the worst-case length.
//   - Page-table management. The per-sequence []int table addresses blocks by logical
//     order; GatherK/GatherV walk it to reconstruct the contiguous K/V a kernel reads.
//   - Copy-on-write sharing. Fork shares every block by reference count; a later write into
//     a shared block copies just THAT block (one page), so a common prefix is shared with
//     zero byte copies and divergence costs one block, not a whole-prefix clone.
//
// Scope (honest, in the paging.go tradition): this is the STANDALONE allocator + page
// table + COW primitive, proven on real float32 KV bytes. It is NOT a GPU paged-attention
// kernel (no block-sparse gather kernel here — GatherK/GatherV materialize contiguously on
// the host), it is NOT yet wired onto the live serve/decode path, and it does NOT replace
// KVCache or internal/radixkv.Tree. Wiring the COW share into radixkv (so the prefix tree
// shares blocks instead of cloning a KVCache) and a device-side paged gather are the next
// steps. What lands here is the data structure those steps build on, with its memory and
// sharing properties measured (pagedkv_test.go).

// PagedKVPool is a shared pool of fixed-size physical KV blocks. Every PagedKV sequence
// minted from one pool draws blocks from the same backing store, so a forked sequence can
// SHARE a physical block with its parent (reference-counted) until one of them writes it.
//
// A block holds blockTokens token-slots of K and V for every layer, laid out flat as
// [layer][k|v][tokenInBlock][stride] where stride = NumKVHeads*HeadDim. Block ids are dense
// indices into blocks/ref; a freed block returns to free for reuse, so a steady-state pool
// allocates no new backing memory.
type PagedKVPool struct {
	blockTokens int         // page size: tokens per physical block
	stride      int         // float32 per token, per (K or V), per layer = NumKVHeads*HeadDim
	nLayers     int         //
	blocks      [][]float32 // physical storage, indexed by block id (len == blockFloats())
	ref         []int       // reference count per block id; 0 == free
	free        []int       // free list of reusable block ids
}

// NewPagedKVPool builds a pool sized to a model config with blockTokens tokens per block.
// A non-positive blockTokens falls back to 16 (the vLLM default page size); a degenerate
// config (no layers / zero stride) is allowed — such a pool simply allocates empty blocks.
func NewPagedKVPool(cfg Config, blockTokens int) *PagedKVPool {
	if blockTokens <= 0 {
		blockTokens = 16
	}
	stride := cfg.NumKVHeads * cfg.HeadDim
	if stride < 0 {
		stride = 0
	}
	nLayers := cfg.NumLayers
	if nLayers < 0 {
		nLayers = 0
	}
	return &PagedKVPool{blockTokens: blockTokens, stride: stride, nLayers: nLayers}
}

// blockFloats is the float32 length of one physical block: K and V, every layer, every
// token-slot in the block.
func (p *PagedKVPool) blockFloats() int { return p.nLayers * 2 * p.blockTokens * p.stride }

// slot is the float32 offset of (layer, isV, tokenInBlock) within a block. isV is 0 for K,
// 1 for V. The slice [slot : slot+stride] is that token's K (or V) for that layer.
func (p *PagedKVPool) slot(layer, isV, tok int) int {
	return ((layer*2+isV)*p.blockTokens + tok) * p.stride
}

// alloc returns an owned (ref==1) block id, reusing a freed block if one is available and
// zeroing it so a reused block never leaks a previous sequence's KV.
func (p *PagedKVPool) alloc() int {
	if n := len(p.free); n > 0 {
		id := p.free[n-1]
		p.free = p.free[:n-1]
		p.ref[id] = 1
		clear(p.blocks[id])
		return id
	}
	id := len(p.blocks)
	p.blocks = append(p.blocks, make([]float32, p.blockFloats()))
	p.ref = append(p.ref, 1)
	return id
}

// release drops one reference to a block; the last reference returns it to the free list.
func (p *PagedKVPool) release(id int) {
	if id < 0 || id >= len(p.ref) || p.ref[id] == 0 {
		return
	}
	p.ref[id]--
	if p.ref[id] == 0 {
		p.free = append(p.free, id)
	}
}

// PhysicalBlocks is the number of distinct blocks the pool currently holds live (ref>0).
// Two sequences sharing a forked prefix count their shared blocks ONCE here — the metric
// that makes copy-on-write sharing observable (vs the sum of per-sequence Blocks()).
func (p *PagedKVPool) PhysicalBlocks() int {
	n := 0
	for _, r := range p.ref {
		if r > 0 {
			n++
		}
	}
	return n
}

// BlockTokens is the pool's page size (tokens per physical block).
func (p *PagedKVPool) BlockTokens() int { return p.blockTokens }

// PagedKV is one sequence's view onto a PagedKVPool: a page table (logical block order →
// physical block id) plus the live token count. Multiple PagedKV may reference the same
// physical block after Fork until one writes it (copy-on-write).
type PagedKV struct {
	pool    *PagedKVPool
	table   []int // page table: table[i] is the physical block id for logical block i
	nTokens int   // live token count; <= len(table)*blockTokens
}

// NewSequence mints an empty sequence backed by this pool.
func (p *PagedKVPool) NewSequence() *PagedKV { return &PagedKV{pool: p} }

// Len is the number of tokens written to this sequence.
func (s *PagedKV) Len() int { return s.nTokens }

// Blocks is the number of logical blocks in this sequence's page table (its KV footprint
// if it owned every block outright — ceil(Len/blockTokens)).
func (s *PagedKV) Blocks() int { return len(s.table) }

// ensureOwned makes logical block li privately owned before a write: if it is shared
// (ref>1) it is copied to a fresh block (the copy-on-write), so the write cannot corrupt a
// sequence that shares the original. A block already owned (ref==1) is left in place.
func (s *PagedKV) ensureOwned(li int) {
	old := s.table[li]
	if s.pool.ref[old] == 1 {
		return
	}
	nb := s.pool.alloc()
	copy(s.pool.blocks[nb], s.pool.blocks[old])
	s.pool.release(old) // this sequence no longer references the shared original
	s.table[li] = nb
}

// Append writes one token's K and V into the sequence. k and v each carry nLayers rows of
// stride float32 (k[layer] is that layer's key for this token). The token lands at the
// current tail: a fresh logical block is allocated on a block boundary, otherwise the tail
// block is made owned (copy-on-write) before the in-place write. Rows shorter than stride
// are written as far as they go; extra is ignored.
func (s *PagedKV) Append(k, v [][]float32) {
	p := s.pool
	li := s.nTokens / p.blockTokens
	off := s.nTokens % p.blockTokens
	if li == len(s.table) {
		s.table = append(s.table, p.alloc())
	} else {
		s.ensureOwned(li)
	}
	blk := p.blocks[s.table[li]]
	for l := 0; l < p.nLayers; l++ {
		if l < len(k) {
			copy(blk[p.slot(l, 0, off):p.slot(l, 0, off)+p.stride], k[l])
		}
		if l < len(v) {
			copy(blk[p.slot(l, 1, off):p.slot(l, 1, off)+p.stride], v[l])
		}
	}
	s.nTokens++
}

// Fork returns a new sequence that SHARES every block of this one by reference count — no
// KV bytes are copied. It is the cache-sharing primitive: a common prefix is shared until
// one branch writes, and only the written block is then copied (see ensureOwned). This is
// strictly cheaper than RadixAttention's per-node full-prefix KVCache clone, which copies
// the whole prefix up front.
func (s *PagedKV) Fork() *PagedKV {
	n := &PagedKV{pool: s.pool, table: append([]int(nil), s.table...), nTokens: s.nTokens}
	for _, b := range n.table {
		s.pool.ref[b]++
	}
	return n
}

// gather walks the page table and copies this sequence's K (isV==0) or V (isV==1) for one
// layer into a fresh contiguous [nTokens*stride] slice, in logical token order — the host
// equivalent of a paged-attention gather kernel. The result is identical to the run a
// contiguous KVCache would hold, which is what makes the paged layout a drop-in.
func (s *PagedKV) gather(layer, isV int) []float32 {
	p := s.pool
	out := make([]float32, s.nTokens*p.stride)
	for pos := 0; pos < s.nTokens; pos++ {
		blk := p.blocks[s.table[pos/p.blockTokens]]
		off := pos % p.blockTokens
		src := p.slot(layer, isV, off)
		copy(out[pos*p.stride:(pos+1)*p.stride], blk[src:src+p.stride])
	}
	return out
}

// GatherK reconstructs the contiguous K run for one layer (paged → contiguous).
func (s *PagedKV) GatherK(layer int) []float32 { return s.gather(layer, 0) }

// GatherV reconstructs the contiguous V run for one layer (paged → contiguous).
func (s *PagedKV) GatherV(layer int) []float32 { return s.gather(layer, 1) }

// Free releases every block this sequence holds (decrementing shared refs) and empties its
// page table. Blocks still shared by another sequence survive; blocks it last-held return
// to the pool free list.
func (s *PagedKV) Free() {
	for _, b := range s.table {
		s.pool.release(b)
	}
	s.table = nil
	s.nTokens = 0
}

// OverheadRatio is this sequence's internal-fragmentation ratio: the unused token-slots in
// its (partial) tail block divided by its live tokens. This is PagedAttention's whole
// memory cost — slack in the last block, bounded by (blockTokens-1)/Len — and the figure
// the issue's "memory overhead ≤ 20%" acceptance is measured against. An empty sequence
// reports 0.
func (s *PagedKV) OverheadRatio() float64 {
	if s.nTokens == 0 {
		return 0
	}
	allocatedSlots := len(s.table) * s.pool.blockTokens
	return float64(allocatedSlots-s.nTokens) / float64(s.nTokens)
}
