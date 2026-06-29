package model

import "testing"

// paged_evict_test.go — the #33 proof gate: a mid-span Evict on the paged/block KV layout
// (paged_evict.go) is BIT-IDENTICAL to the contiguous KVCache.Evict (kvcache.go), and the
// re-RoPE of shifted survivors does not corrupt a copy-on-write-shared sibling. All on real
// float32 KV with no GPU, mirroring evict_test.go's contiguous bit-exactness proof.

// pagedEvictCfg is the layer-specific-RoPE-theta config evict_test.go uses to stress the
// reposition: two layers with different theta means a survivor moved to a new position must be
// re-rotated per layer, so a composed-rotation bug (the thing the Kraw plane prevents) would
// show up as a non-zero bit diff.
func pagedEvictCfg() Config {
	return Config{
		HiddenSize:        32,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         97,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		RopeThetaPerLayer: []float64{10000, 1000000},
		BlockTopology:     SandwichNorm,
	}
}

// snapshotCacheToPaged copies a contiguous KVCache verbatim into a 3-plane paged sequence,
// token by token (K, pre-RoPE Kraw, V), so the paged copy starts byte-identical to the
// contiguous one — the fair starting point for proving Evict agrees on both layouts.
func snapshotCacheToPaged(pool *PagedKVPool, c *KVCache) *PagedKV {
	nL, w := pool.nLayers, c.kvStride()
	s := pool.NewSequence()
	for pos := 0; pos < c.Len(); pos++ {
		k := make([][]float32, nL)
		kraw := make([][]float32, nL)
		v := make([][]float32, nL)
		for l := 0; l < nL; l++ {
			k[l] = append([]float32(nil), c.K[l][pos*w:(pos+1)*w]...)
			kraw[l] = append([]float32(nil), c.Kraw[l][pos*w:(pos+1)*w]...)
			v[l] = append([]float32(nil), c.V[l][pos*w:(pos+1)*w]...)
		}
		s.AppendRaw(k, kraw, v)
	}
	return s
}

// churnNonContiguous allocates then frees alternating decoy blocks so the pool's free list
// hands the next sequence physically NON-CONTIGUOUS, out-of-logical-order block ids — the
// property that makes "physical position != logical position" real instead of an accident of a
// fresh pool. Returns the live decoy ids the caller leaves held.
func churnNonContiguous(pool *PagedKVPool) []int {
	var decoy []int
	for i := 0; i < 6; i++ {
		decoy = append(decoy, pool.alloc())
	}
	live := decoy[:0:0]
	for i, id := range decoy {
		if i%2 == 0 {
			pool.release(id) // free the even ids; the LIFO free list pops them in reverse
		} else {
			live = append(live, id)
		}
	}
	return live
}

func tableAscending(table []int) bool {
	for i := 1; i < len(table); i++ {
		if table[i] < table[i-1] {
			return false
		}
	}
	return true
}

// TestPagedEvictBitIdenticalToContiguous is the #33 acceptance witness: a mid-span evict on a
// NON-CONTIGUOUS paged layout produces a cache (K, Kraw, V) and next-token logits that are
// byte-for-byte identical to the contiguous KVCache.Evict on the same data. The span is
// deliberately not block-aligned (blockTokens=4, evict positions [2,5)) so survivors cross
// block boundaries and must be re-RoPE'd at new logical positions.
func TestPagedEvictBitIdenticalToContiguous(t *testing.T) {
	cfg := pagedEvictCfg()
	m := NewSynthetic(cfg)

	all := []int{3, 17, 5, 23, 41, 2, 19} // 7 tokens
	const from, n = 2, 3                   // middle span [2,5); survivors {0,1,5,6} -> {0,1,2,3}

	// Contiguous reference.
	cs := m.NewSession()
	cs.Prefill(all)

	// Paged copy on a churned (non-contiguous) pool.
	pool := NewPagedKVPoolWithRaw(cfg, 4)
	churnNonContiguous(pool)
	seq := snapshotCacheToPaged(pool, cs.Cache)
	if tableAscending(seq.table) {
		t.Fatalf("weak setup: page table %v is physically contiguous; the proof must exercise non-contiguous blocks", seq.table)
	}

	// Sanity: the snapshot is faithful BEFORE eviction.
	for l := 0; l < cfg.NumLayers; l++ {
		assertFloat32BitsEqual(t, "pre-evict K l"+itoa(l), cs.Cache.K[l], seq.GatherK(l))
		assertFloat32BitsEqual(t, "pre-evict Kraw l"+itoa(l), cs.Cache.Kraw[l], seq.GatherKraw(l))
		assertFloat32BitsEqual(t, "pre-evict V l"+itoa(l), cs.Cache.V[l], seq.GatherV(l))
	}

	// Evict the same middle span on both layouts.
	rc := cs.Cache.Evict(from, n)
	rp := seq.Evict(from, n, cfg)
	if rc != rp || rc != n {
		t.Fatalf("removed mismatch: contiguous=%d paged=%d want=%d", rc, rp, n)
	}
	if seq.Len() != cs.Cache.Len() {
		t.Fatalf("post-evict len mismatch: contiguous=%d paged=%d", cs.Cache.Len(), seq.Len())
	}

	// The load-bearing claim: paged Evict == contiguous Evict, bit-for-bit, every layer.
	for l := 0; l < cfg.NumLayers; l++ {
		assertFloat32BitsEqual(t, "paged-evict K l"+itoa(l), cs.Cache.K[l], seq.GatherK(l))
		assertFloat32BitsEqual(t, "paged-evict Kraw l"+itoa(l), cs.Cache.Kraw[l], seq.GatherKraw(l))
		assertFloat32BitsEqual(t, "paged-evict V l"+itoa(l), cs.Cache.V[l], seq.GatherV(l))
	}

	// End-to-end: decode the same next token from each evicted cache; logits must be identical.
	rebuilt := NewKVCache(cfg)
	for l := 0; l < cfg.NumLayers; l++ {
		rebuilt.K[l] = seq.GatherK(l)
		rebuilt.Kraw[l] = seq.GatherKraw(l)
		rebuilt.V[l] = seq.GatherV(l)
	}
	rebuilt.pos = make([]int, seq.Len())
	for i := range rebuilt.pos {
		rebuilt.pos[i] = i
	}
	ps := &Session{M: m, Cache: rebuilt}
	const nextTok = 7
	wantLogits := cs.Step(nextTok)
	gotLogits := ps.Step(nextTok)
	assertFloat32BitsEqual(t, "post-evict next-token logits", wantLogits, gotLogits)
}

// TestPagedEvictCOWLeavesForkedParentUnchanged proves the one constraint paging adds over the
// contiguous layout: the survivor re-RoPE MUTATES K, so a sequence that Fork()ed (and so shares
// physical blocks copy-on-write) and did NOT evict must be left byte-for-byte unchanged. The
// evicting fork rebuilds into fresh private blocks; the parent keeps its own bytes.
func TestPagedEvictCOWLeavesForkedParentUnchanged(t *testing.T) {
	cfg := pagedEvictCfg()
	m := NewSynthetic(cfg)
	all := []int{3, 17, 5, 23, 41, 2, 19}
	const from, n = 2, 3

	cs := m.NewSession()
	cs.Prefill(all)

	pool := NewPagedKVPoolWithRaw(cfg, 4)
	parent := snapshotCacheToPaged(pool, cs.Cache)
	child := parent.Fork()
	if pb := pool.PhysicalBlocks(); pb != parent.Blocks() {
		t.Fatalf("Fork copied bytes: PhysicalBlocks=%d, want %d (shared)", pb, parent.Blocks())
	}

	// Snapshot the parent's bytes before the child evicts.
	parentK := make([][]float32, cfg.NumLayers)
	parentKraw := make([][]float32, cfg.NumLayers)
	parentV := make([][]float32, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		parentK[l] = parent.GatherK(l)
		parentKraw[l] = parent.GatherKraw(l)
		parentV[l] = parent.GatherV(l)
	}

	child.Evict(from, n, cfg)
	cs.Cache.Evict(from, n)

	// Parent untouched by the child's re-RoPE (copy-on-write held).
	for l := 0; l < cfg.NumLayers; l++ {
		assertFloat32BitsEqual(t, "parent K unchanged l"+itoa(l), parentK[l], parent.GatherK(l))
		assertFloat32BitsEqual(t, "parent Kraw unchanged l"+itoa(l), parentKraw[l], parent.GatherKraw(l))
		assertFloat32BitsEqual(t, "parent V unchanged l"+itoa(l), parentV[l], parent.GatherV(l))
	}
	if parent.Len() != len(all) {
		t.Fatalf("parent length changed by child evict: %d != %d", parent.Len(), len(all))
	}

	// Child still equals the contiguous evicted cache.
	for l := 0; l < cfg.NumLayers; l++ {
		assertFloat32BitsEqual(t, "child K == contiguous l"+itoa(l), cs.Cache.K[l], child.GatherK(l))
		assertFloat32BitsEqual(t, "child V == contiguous l"+itoa(l), cs.Cache.V[l], child.GatherV(l))
	}
}
