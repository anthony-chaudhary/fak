package model

// paged_evict.go — #33 design-gate prototype: carrying the bit-exact middle-span Evict
// value-add (kvcache.go) onto the paged/block KV layout (pagedkv.go, #277).
//
// THE TENSION (#33). The contiguous KVCache stores each layer's K/V as one flat slice
// indexed by position, so Evict(from,n) is a slice splice plus a single-rotation re-RoPE of
// every shifted survivor's K from its pre-RoPE Kraw at its NEW position — proven byte-for-byte
// `== never-saw-it` (evict_test.go TestKVQuarantineEqualsNeverSaw). The PagedKV allocator
// stores positions in NON-CONTIGUOUS physical blocks addressed by a page table, and — the gap
// this file closes — it stores only K and V, no Kraw, and has no Evict. Without a place to
// keep the pre-RoPE K, a paged middle-span evict would have to compose two rotations to move a
// survivor, which drifts ~1e-6 and can flip a greedy token — exactly the bug the contiguous
// path's single-rotation-from-Kraw design avoids.
//
// THE RESOLUTION proven here. The single-rotation re-RoPE invariant depends only on (a) the
// survivor's pre-RoPE Kraw and (b) its NEW LOGICAL position — NOT on physical contiguity. So
// it survives paging cleanly IF the layout carries a per-token Kraw plane (NewPagedKVPoolWithRaw
// below) that travels with the token wherever it physically lands. Evict then re-derives each
// shifted survivor's K in one rotation at its new logical index, identical to the contiguous
// path. The one genuinely new constraint paging adds is copy-on-write: a re-RoPE MUTATES K, so
// a survivor sitting in a block shared (COW) with a sequence that did NOT evict must trigger a
// block copy first. This prototype gets that for free by rebuilding the evicting sequence into
// fresh private blocks — a forked parent keeps its own refs and is left byte-for-byte unchanged
// (paged_evict_test.go TestPagedEvictCOWLeavesForkedParentUnchanged).
//
// Scope: the dense softmax-K/V/Kraw PagedKV path. GLM-DSA's separate attention/index cache
// has different row geometry and is covered by paged_glmdsa.go instead.

// Plane indices within a block (the slot()/appendPlanes() plane argument): K and V are the
// 2-plane pool's planes; Kraw (pre-RoPE K) is the third plane a NewPagedKVPoolWithRaw pool adds
// so exact-span eviction can re-rotate a shifted survivor in a single rotation.
const (
	planeK    = 0
	planeV    = 1
	planeKraw = 2
)

// NewPagedKVPoolWithRaw builds a pool whose blocks carry a third per-token plane — Kraw, the
// pre-RoPE K — alongside K and V. That plane is what lets PagedKV.Evict re-derive a shifted
// survivor's post-RoPE K in a single rotation at its new logical position, the bit-exact
// reposition the contiguous KVCache.Evict performs. A pool built this way costs 1.5× the K/V
// bytes per block; a plain NewPagedKVPool (K/V only) cannot carry exact-span eviction.
func NewPagedKVPoolWithRaw(cfg Config, blockTokens int) *PagedKVPool {
	p := NewPagedKVPool(cfg, blockTokens)
	p.planes = 3
	return p
}

// SupportsRaw reports whether this pool carries the Kraw plane (and so can serve AppendRaw /
// Evict). A 2-plane K/V pool returns false.
func (p *PagedKVPool) SupportsRaw() bool { return p.planes >= 3 }

// AppendRaw writes one token's per-layer K, pre-RoPE Kraw, and V into the sequence tail. It is
// Append plus the Kraw plane, and requires a pool built by NewPagedKVPoolWithRaw; on a 2-plane
// pool it panics rather than silently dropping the Kraw a later Evict depends on.
func (s *PagedKV) AppendRaw(k, kraw, v [][]float32) {
	if !s.pool.SupportsRaw() {
		panic("model: PagedKV.AppendRaw requires a NewPagedKVPoolWithRaw pool (no Kraw plane)")
	}
	s.appendPlanes([][][]float32{k, v, kraw})
}

// AppendLayerRaw writes one layer's K, pre-RoPE Kraw, and V for the current token. It is the
// layer-incremental form of AppendRaw used by the compute HAL, where each transformer layer
// appends its K/V immediately before running attention for that layer.
func (s *PagedKV) AppendLayerRaw(layer int, k, kraw, v []float32) {
	if !s.pool.SupportsRaw() {
		panic("model: PagedKV.AppendLayerRaw requires a NewPagedKVPoolWithRaw pool (no Kraw plane)")
	}
	s.appendLayerPlanes(layer, [][]float32{k, v, kraw})
}

// GatherKraw reconstructs the contiguous pre-RoPE Kraw run for one layer (paged → contiguous),
// the Kraw counterpart of GatherK/GatherV.
func (s *PagedKV) GatherKraw(layer int) []float32 { return s.gather(layer, planeKraw) }

// Evict removes a contiguous span [from, from+n) of LOGICAL positions from this paged
// sequence and re-indexes every survivor to a new contiguous logical position, so the result
// is byte-for-byte what a paged sequence that never saw the span would hold — the paged twin
// of KVCache.Evict (kvcache.go). It returns the number of positions removed.
//
// Shifted survivors (new logical index i below their original position) have their post-RoPE K
// re-derived from the pre-RoPE Kraw plane in a SINGLE rotation at position i (Alibi: the
// pre-RoPE row IS the K, copied verbatim) — physical block placement is irrelevant to that
// arithmetic, which is the whole point of #33. V and Kraw move verbatim (V is never rotated;
// Kraw is position-independent). The sequence is rebuilt into fresh private blocks, so a
// fork()ed sibling sharing the old blocks (copy-on-write) is left untouched.
//
// Requires a NewPagedKVPoolWithRaw pool; cfg must be the model config whose RoPE the cache was
// filled with. A non-positive or out-of-range span is a no-op (returns 0).
func (s *PagedKV) Evict(from, n int, cfg Config) int {
	if !s.pool.SupportsRaw() {
		panic("model: PagedKV.Evict requires a NewPagedKVPoolWithRaw pool (no Kraw plane to re-rotate from)")
	}
	if from < 0 || n <= 0 || from >= s.nTokens {
		return 0
	}
	end := from + n
	if end > s.nTokens {
		end = s.nTokens
	}
	nL, stride := s.pool.nLayers, s.pool.stride

	// Gather the whole sequence out first (read-only — safe even if blocks are COW-shared).
	K := make([][]float32, nL)
	Kraw := make([][]float32, nL)
	V := make([][]float32, nL)
	for l := 0; l < nL; l++ {
		K[l] = s.GatherK(l)
		Kraw[l] = s.GatherKraw(l)
		V[l] = s.GatherV(l)
	}

	// Survivor logical positions, in order: [0,from) then [end,nTokens).
	survivors := make([]int, 0, s.nTokens-(end-from))
	for p := 0; p < from; p++ {
		survivors = append(survivors, p)
	}
	for p := end; p < s.nTokens; p++ {
		survivors = append(survivors, p)
	}

	// Rebuild into fresh private blocks (this is the copy-on-write split: a forked parent
	// keeps its refs and its bytes; only THIS sequence's view is rewritten).
	s.Free()
	row := func(src []float32, pos int) []float32 {
		return append([]float32(nil), src[pos*stride:(pos+1)*stride]...)
	}
	for i, op := range survivors {
		k := make([][]float32, nL)
		kraw := make([][]float32, nL)
		v := make([][]float32, nL)
		for l := 0; l < nL; l++ {
			kraw[l] = row(Kraw[l], op)
			v[l] = row(V[l], op)
			if op == i {
				// Unmoved survivor (before the span): keep its original post-RoPE K verbatim,
				// matching the contiguous path, which only re-rotates where pos[i] != i.
				k[l] = row(K[l], op)
			} else {
				// Shifted survivor: re-derive K from pre-RoPE Kraw in ONE rotation at new pos i.
				k[l] = reropeRowFromRaw(cfg, l, i, kraw[l])
			}
		}
		s.AppendRaw(k, kraw, v)
	}
	return end - from
}

// reropeRowFromRaw re-derives layer l's post-RoPE K row for logical position pos from its
// pre-RoPE Kraw, in a SINGLE rotation per KV head — the exact reposition arithmetic the
// contiguous KVCache.Evict uses (kvcache.go), reusing the same ropeRowForLayer/applyRopeRow so
// the result is bit-identical, not merely close. Alibi layers carry no RoPE, so the pre-RoPE
// row IS the stored K and is returned as a verbatim copy (again matching the contiguous path).
func reropeRowFromRaw(cfg Config, l, pos int, kraw []float32) []float32 {
	out := append([]float32(nil), kraw...)
	if cfg.Alibi {
		return out
	}
	hd, nKV := cfg.HeadDim, cfg.NumKVHeads
	cos, sin := ropeRowForLayer(cfg, l, pos)
	for h := 0; h < nKV; h++ {
		applyRopeRow(out[h*hd:(h+1)*hd], cos, sin)
	}
	return out
}
