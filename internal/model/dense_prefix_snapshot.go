// SPDX-License-Identifier: Apache-2.0
// Copyright 2023-2024 SGLang Team
// Dense retained-prefix ownership adapted from SGLang radix_cache.py at
// 181380682b50f4606d1a09df3cbbbfc21de931e3. Modified for FAK's existing
// PagedKV allocator and request-private contiguous Session cache.
package model

import (
	"math"
	"reflect"
	"strconv"
)

// DensePrefixSnapshot is an immutable dense-FP32 cache owner. It shares physical
// pages only with snapshots derived from the same admitted prefix. Materialize
// returns a private contiguous cache, not a zero-copy Session. All methods,
// including materialization and Free, require the owning tree's external lock.
type DensePrefixSnapshot struct {
	seq     *PagedKV
	cfg     Config
	prec    KVPrecision
	pos     []int
	lineage tokenLineage
}

// DensePrefixCapture is a synchronous, pre-allocation plan. The source cache
// and prefix must remain unchanged under the same external lock until Capture.
// It owns no pages before Capture and must not be queued for asynchronous use.
type DensePrefixCapture struct {
	source        *KVCache
	prefix        *DensePrefixSnapshot
	pool          *PagedKVPool
	blocks, slots int
	additional    int64
}

const densePrefixPageTokens = 16

func densePrefixAdd(a, b int64) int64 {
	if a < 0 || b < 0 || a > math.MaxInt64-b {
		return math.MaxInt64
	}
	return a + b
}

func densePrefixBytes(n, width int) int64 {
	if n < 0 || width < 0 || (width != 0 && uint64(n) > uint64(math.MaxInt64)/uint64(width)) {
		return math.MaxInt64
	}
	return int64(n) * int64(width)
}

func densePrefixMetadataBytes(tokens, blocks int) int64 {
	return densePrefixAdd(densePrefixBytes(tokens, strconv.IntSize/8+4), densePrefixBytes(blocks, strconv.IntSize/8))
}

func densePrefixEligible(c *KVCache) bool {
	if c == nil || c.prec != KVPrecisionFP32 || c.linear != nil || c.glm != nil || c.msa != nil ||
		len(c.kQ8) != 0 || len(c.vQ8) != 0 || c.cfg.usesMLAMoELayout() ||
		c.cfg.isMiniMaxSparseAttn() || c.cfg.IsQwen35Hybrid() ||
		len(c.cfg.HeadDimPerLayer) != 0 || len(c.cfg.NumKVHeadsPerLayer) != 0 ||
		c.cfg.NumLayers <= 0 || c.cfg.NumKVHeads <= 0 || c.cfg.HeadDim <= 0 ||
		len(c.lineage.ids) != c.Len() {
		return false
	}
	width, ok := qwenSwapSizeProduct(c.cfg.NumKVHeads, c.cfg.HeadDim)
	if !ok {
		return false
	}
	rows, ok := qwenSwapSizeProduct(c.Len(), width)
	if !ok || len(c.K) != c.cfg.NumLayers || len(c.Kraw) != c.cfg.NumLayers || len(c.V) != c.cfg.NumLayers {
		return false
	}
	for l := 0; l < c.cfg.NumLayers; l++ {
		if len(c.K[l]) != rows || len(c.Kraw[l]) != rows || len(c.V[l]) != rows {
			return false
		}
	}
	_, ok = qwenSwapSizeProduct(c.cfg.NumLayers, 3, densePrefixPageTokens, width, 4)
	return ok
}

// PlanDensePrefixCapture refuses unsupported or malformed layouts before any
// tree eviction, page allocation or reference change. A nonmatching prefix is
// not reused; token names alone never authorize sharing actual model bytes.
func PlanDensePrefixCapture(c *KVCache, prefix *DensePrefixSnapshot) (*DensePrefixCapture, bool) {
	if !densePrefixEligible(c) {
		return nil, false
	}
	if prefix != nil && !prefix.matches(c) {
		prefix = nil
	}
	pool := (*PagedKVPool)(nil)
	start := 0
	if prefix != nil {
		pool, start = prefix.seq.pool, prefix.seq.nTokens
	} else {
		pool = NewPagedKVPoolWithRaw(c.cfg, densePrefixPageTokens)
		pool.reclaimFreeBacking = true
	}
	blocks := qwenSwapCeilDiv(c.Len(), pool.blockTokens)
	newBlocks := blocks
	if prefix != nil {
		newBlocks = 0
		if c.Len() > start {
			newBlocks = blocks - start/pool.blockTokens
		}
	}
	maxInt := int(^uint(0) >> 1)
	if newBlocks > maxInt-len(pool.blocks) {
		return nil, false
	}
	// Ignore currently free IDs when pricing metadata growth: intervening
	// budget evictions can only make this synchronous upper bound smaller.
	slots := len(pool.blocks) + newBlocks
	additional := densePrefixMetadataBytes(c.Len(), blocks)
	floats, ok := qwenSwapSizeProduct(newBlocks, pool.blockFloats())
	if !ok {
		return nil, false
	}
	additional = densePrefixAdd(additional, densePrefixBytes(floats, 4))
	for _, capacity := range []int{cap(pool.ref), cap(pool.free)} {
		if slots > capacity {
			additional = densePrefixAdd(additional, densePrefixBytes(slots-capacity, strconv.IntSize/8))
		}
	}
	if additional == math.MaxInt64 {
		return nil, false
	}
	return &DensePrefixCapture{source: c, prefix: prefix, pool: pool, blocks: blocks, slots: slots, additional: additional}, true
}

// AdditionalBytes bounds the increase in retained numeric backing, including
// page tables, pool ref/free arrays, positions and exact token lineage. It does
// not price transient conversion views, Go headers, allocator rounding or RSS.
func (p *DensePrefixCapture) AdditionalBytes() int64 {
	if p == nil {
		return math.MaxInt64
	}
	return p.additional
}

func (s *DensePrefixSnapshot) matches(c *KVCache) bool {
	if s == nil || s.seq == nil || s.seq.nTokens > c.Len() || !reflect.DeepEqual(s.cfg, c.cfg) ||
		s.prec != c.prec || s.lineage.fault != c.lineage.fault {
		return false
	}
	for i := range s.pos {
		if s.pos[i] != c.pos[i] || s.lineage.ids[i] != c.lineage.ids[i] {
			return false
		}
	}
	p := s.seq.pool
	for l := 0; l < p.nLayers; l++ {
		for plane, rows := range [][][]float32{c.K, c.V, c.Kraw} {
			for token := 0; token < s.seq.nTokens; token++ {
				block := p.blocks[s.seq.table[token/p.blockTokens]]
				off := p.slot(l, plane, token%p.blockTokens)
				for j := 0; j < p.stride; j++ {
					if math.Float32bits(block[off+j]) != math.Float32bits(rows[l][token*p.stride+j]) {
						return false
					}
				}
			}
		}
	}
	return true
}

func densePrefixReserveMetadata(p *PagedKVPool, slots int) {
	if cap(p.blocks) < slots {
		x := make([][]float32, len(p.blocks), slots)
		copy(x, p.blocks)
		p.blocks = x
	}
	if cap(p.ref) < slots {
		x := make([]int, len(p.ref), slots)
		copy(x, p.ref)
		p.ref = x
	}
	if cap(p.free) < slots {
		x := make([]int, len(p.free), slots)
		copy(x, p.free)
		p.free = x
	}
}

// Capture consumes one plan synchronously. The existing Fork/AppendRaw COW
// and Free operations remain the only physical-page ownership mechanisms.
// Call once after successful retained-byte admission under the same lock.
func (p *DensePrefixCapture) Capture() *DensePrefixSnapshot {
	if p == nil || p.source == nil {
		return nil
	}
	c := p.source
	p.source = nil // prevent consuming the same plan twice
	densePrefixReserveMetadata(p.pool, p.slots)
	seq := p.pool.NewSequence()
	if p.prefix != nil {
		seq = p.prefix.seq.Fork()
	}
	// Exact table capacity makes the retained-byte preflight independent of
	// Go append growth. Fork's short-lived table is not retained by the owner.
	table := make([]int, len(seq.table), p.blocks)
	copy(table, seq.table)
	seq.table = table
	k, raw, v := make([][]float32, p.pool.nLayers), make([][]float32, p.pool.nLayers), make([][]float32, p.pool.nLayers)
	for token := seq.nTokens; token < c.Len(); token++ {
		a, b := token*p.pool.stride, (token+1)*p.pool.stride
		for l := range k {
			k[l], raw[l], v[l] = c.K[l][a:b:b], c.Kraw[l][a:b:b], c.V[l][a:b:b]
		}
		seq.AppendRaw(k, raw, v)
	}
	pos := make([]int, len(c.pos))
	copy(pos, c.pos)
	ids := make([]uint32, len(c.lineage.ids))
	copy(ids, c.lineage.ids)
	return &DensePrefixSnapshot{seq: seq, cfg: c.cfg, prec: c.prec, pos: pos, lineage: tokenLineage{ids: ids, fault: c.lineage.fault}}
}

// PrefixMetadataBytes prices an independently owned prefix view before Fork.
// No physical page is copied until a later branch writes a shared partial tail.
func (s *DensePrefixSnapshot) PrefixMetadataBytes(n int) (int64, bool) {
	if s == nil || s.seq == nil || n < 0 || n > s.seq.nTokens {
		return 0, false
	}
	return densePrefixMetadataBytes(n, qwenSwapCeilDiv(n, s.seq.pool.blockTokens)), true
}

func (s *DensePrefixSnapshot) ForkPrefix(n int) *DensePrefixSnapshot {
	if _, ok := s.PrefixMetadataBytes(n); !ok {
		return nil
	}
	seq := s.seq.Fork()
	blocks := qwenSwapCeilDiv(n, seq.pool.blockTokens)
	for _, id := range seq.table[blocks:] {
		seq.pool.release(id)
	}
	table := make([]int, blocks)
	copy(table, seq.table[:blocks])
	seq.table, seq.nTokens = table, n
	pos := make([]int, n)
	copy(pos, s.pos[:n])
	ids := make([]uint32, n)
	copy(ids, s.lineage.ids[:n])
	return &DensePrefixSnapshot{seq: seq, cfg: s.cfg, prec: s.prec, pos: pos, lineage: tokenLineage{ids: ids, fault: s.lineage.fault}}
}

// Materialize preserves absolute positions and lineage. The older ToKVCache
// helper intentionally relabels positions; it is not this snapshot's contract.
func (s *DensePrefixSnapshot) Materialize() *KVCache {
	if s == nil || s.seq == nil {
		return nil
	}
	c := NewKVCacheWithPrecision(s.cfg, s.prec)
	for l := range c.K {
		c.K[l], c.Kraw[l], c.V[l] = s.seq.GatherK(l), s.seq.GatherKraw(l), s.seq.GatherV(l)
	}
	c.pos = make([]int, len(s.pos))
	copy(c.pos, s.pos)
	c.lineage = s.lineage.clone(0)
	return c
}

// Free is idempotent. Only this store's pools detach zero-reference backing;
// unrelated legacy paged pools keep their existing warm free-list behavior.
func (s *DensePrefixSnapshot) Free() {
	if s == nil || s.seq == nil {
		return
	}
	s.seq.Free()
	s.seq = nil
	s.pos = nil
	s.lineage = tokenLineage{}
	s.cfg = Config{}
}

func (s *DensePrefixSnapshot) MetadataBytes() int64 {
	if s == nil || s.seq == nil {
		return 0
	}
	return densePrefixAdd(densePrefixAdd(densePrefixBytes(cap(s.seq.table), strconv.IntSize/8),
		densePrefixBytes(cap(s.pos), strconv.IntSize/8)), densePrefixBytes(cap(s.lineage.ids), 4))
}

// DensePrefixAccounting counts each shared pool once, including retained free
// slots and numeric allocator metadata. Zero refs do not by themselves imply
// reclaimed backing. The zero value starts one externally serialized census.
type DensePrefixAccounting struct{ seen map[*PagedKVPool]bool }

func (a *DensePrefixAccounting) Add(s *DensePrefixSnapshot) int64 {
	if s == nil || s.seq == nil {
		return 0
	}
	total := s.MetadataBytes()
	p := s.seq.pool
	if a.seen == nil {
		a.seen = make(map[*PagedKVPool]bool)
	}
	if a.seen[p] {
		return total
	}
	a.seen[p] = true
	total = densePrefixAdd(total, densePrefixBytes(cap(p.ref), strconv.IntSize/8))
	total = densePrefixAdd(total, densePrefixBytes(cap(p.free), strconv.IntSize/8))
	for _, block := range p.blocks {
		total = densePrefixAdd(total, densePrefixBytes(cap(block), 4))
	}
	return total
}
