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

// canReserveBlocks reports whether the pool can satisfy an n-block batch reservation in full:
// the free list plus, when MaxBlocks > 0, the remaining growth budget (MaxBlocks - len(blocks),
// clamped at zero in case MaxBlocks was set below the already-grown backing store). With
// MaxBlocks == 0 the pool is unbounded and any batch is satisfiable — the pre-#3386 contract.
// Pure capacity arithmetic: it mints nothing and mutates nothing.
func (p *PagedKVPool) canReserveBlocks(n int) bool {
	if n <= 0 {
		return true
	}
	if p.MaxBlocks <= 0 {
		return true
	}
	budget := p.MaxBlocks - len(p.blocks)
	if budget < 0 {
		budget = 0
	}
	return len(p.free)+budget >= n
}

// TryReserveBlocks is the capacity-aware, ALL-OR-NOTHING batch reserve (#3386): it appends
// exactly n owned (ref==1), zeroed physical blocks to this sequence's page table and returns
// true — but only after checking, BEFORE minting any block, that the pool's free list plus its
// remaining growth budget (MaxBlocks - len(blocks), when MaxBlocks > 0) can satisfy the whole
// batch. On a shortfall it returns false and mutates NOTHING — no block is minted, the free
// list, backing store, and page table are untouched — so there is no partial commit and no
// rollback path. A non-positive n is trivially satisfied (true, nothing minted). With
// MaxBlocks == 0 the pool is unbounded and TryReserveBlocks always succeeds, preserving the
// pre-#3386 alloc-never-fails behavior.
func (s *PagedKV) TryReserveBlocks(n int) bool {
	p := s.pool
	if !p.canReserveBlocks(n) {
		return false
	}
	for i := 0; i < n; i++ {
		s.table = append(s.table, p.alloc())
	}
	return true
}

// Reserve pre-allocates the owned physical blocks a future growth of extraTokens more tokens
// will cross into, so subsequent Append/AppendRaw calls draw them from this sequence's page
// table instead of minting fresh blocks from the pool on the decode hot-path. It does NOT
// change Len(): reserved blocks sit past the live tail, are owned (ref==1) and zeroed, and are
// never read by GatherK/GatherV (which stop at Len), so the sequence's live content stays
// bit-exact. Blocks() and OverheadRatio() do grow to reflect the reserved capacity — the paged
// analogue of the contiguous cache's cap > len. A non-positive extraTokens is a no-op.
//
// Since #3386 Reserve is backed by TryReserveBlocks, so it is capacity-aware and all-or-nothing:
// on a pool with MaxBlocks > 0, a reservation the free list plus growth budget cannot FULLY
// satisfy fails cleanly (no block minted, nothing mutated) instead of growing the pool
// unbounded. With MaxBlocks == 0 (the default) every reservation succeeds exactly as before, so
// existing callers are unaffected.
func (s *PagedKV) Reserve(extraTokens int) {
	if extraTokens <= 0 {
		return
	}
	want := s.pool.blocksForTokens(s.nTokens + extraTokens)
	s.TryReserveBlocks(want - len(s.table))
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
