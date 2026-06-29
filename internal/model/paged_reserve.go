package model

// paged_reserve.go — #34 increment: carry the contiguous KVCache's Clone / CloneWithReserve /
// Reserve REUSE semantics (kvcache.go:181-234) onto the paged/block KV allocator (pagedkv.go,
// #277), so the paged path preserves the same reserve-then-grow and deep-copy contracts the
// serve/radixkv reuse path relies on — and keeps that reuse bit-exact.
//
// What the contiguous path offers and why each twin exists here:
//
//   - Reserve(extra) on the flat cache grows each layer's slice CAPACITY (not Len) so the
//     planned decode/result tail can be appended without re-copying the prefix. The paged
//     analogue is NOT "grow a capacity" — a paged sequence never re-copies on growth (each
//     physical block is fixed-size and written in place). The cost a paged append still pays
//     is the pool ALLOCATION at every block boundary. So paged Reserve pre-pays exactly that:
//     it pre-allocates the owned blocks a future `extra`-token growth will cross into, so the
//     decode hot-path draws them from the page table instead of minting them mid-step. Live
//     content (Len, and what GatherK/V return) is unchanged — reserved blocks sit past the
//     live tail, so reuse stays bit-exact.
//
//   - Clone() on the flat cache is an eager full deep copy. PagedKV already has Fork() — the
//     COPY-ON-WRITE share that is strictly cheaper (no bytes copied until a write). Clone() is
//     kept as the EAGER twin that matches the contiguous semantics exactly: a fully private,
//     independent sequence whose blocks share nothing, so neither side ever pays a later COW
//     split. Use Fork for cheap prefix sharing; use Clone where the contiguous Clone was used
//     (an up-front independent snapshot). Both are observably independent and bit-identical to
//     the source; Clone just front-loads the copy.
//
//   - CloneWithReserve(extra) is Clone + Reserve, the paged twin of kvcache.go's
//     CloneWithReserve — clone a prefix and reserve room for the continuation in one call.
//
// All three operate at the BLOCK level (whole-block copy / whole-block alloc), below the
// plane abstraction, so they are plane-count-agnostic: on a 3-plane NewPagedKVPoolWithRaw pool
// (paged_evict.go) they carry the Kraw plane forward bit-exact for free, with no Kraw-specific
// code here. Scope (honest): this is the allocator-level reuse primitive set, proven bit-exact
// on real float32 KV with no GPU (paged_reserve_test.go). Live opt-in HAL gather and GLM-DSA's
// separate paged-row witness live in sibling files; this file only owns reuse semantics.

// blocksForTokens is the number of fixed-size physical blocks a sequence of n tokens occupies:
// ceil(n / blockTokens). It is the page-table length Reserve grows toward.
func (p *PagedKVPool) blocksForTokens(n int) int {
	if p.blockTokens <= 0 || n <= 0 {
		return 0
	}
	return (n + p.blockTokens - 1) / p.blockTokens
}

// Reserve pre-allocates the owned physical blocks a future growth of extraTokens more tokens
// will cross into, so subsequent Append/AppendRaw calls draw them from this sequence's page
// table instead of minting fresh blocks from the pool on the decode hot-path. It does NOT
// change Len(): reserved blocks sit past the live tail, are owned (ref==1) and zeroed, and are
// never read by GatherK/GatherV (which stop at Len), so the sequence's live content stays
// bit-exact. Blocks() and OverheadRatio() do grow to reflect the reserved capacity — the paged
// analogue of the contiguous cache's cap > len. A non-positive extraTokens is a no-op.
func (s *PagedKV) Reserve(extraTokens int) {
	if extraTokens <= 0 {
		return
	}
	p := s.pool
	want := p.blocksForTokens(s.nTokens + extraTokens)
	for len(s.table) < want {
		s.table = append(s.table, p.alloc())
	}
}

// Clone returns an EAGER deep copy of this sequence: every physical block is copied to a fresh
// owned block, so the clone shares nothing with the source (contrast Fork, which shares blocks
// copy-on-write). The clone's GatherK/GatherV/GatherKraw are byte-for-byte identical to the
// source, and a later write to either side cannot affect the other — the paged twin of the
// contiguous KVCache.Clone (kvcache.go). Because it copies whole blocks it carries every plane
// (K, V, and Kraw on a 3-plane pool) forward bit-exact.
func (s *PagedKV) Clone() *PagedKV {
	p := s.pool
	n := &PagedKV{pool: p, table: make([]int, len(s.table)), nTokens: s.nTokens}
	for i, src := range s.table {
		nb := p.alloc()
		copy(p.blocks[nb], p.blocks[src])
		n.table[i] = nb
	}
	return n
}

// CloneWithReserve is Clone plus Reserve(extraTokens): an independent deep copy of the prefix
// with room pre-allocated for extraTokens more tokens, so the cloned sequence can be grown to
// its planned length without minting blocks mid-decode — the paged twin of the contiguous
// KVCache.CloneWithReserve (kvcache.go).
func (s *PagedKV) CloneWithReserve(extraTokens int) *PagedKV {
	n := s.Clone()
	n.Reserve(extraTokens)
	return n
}
