package model

import (
	"reflect"
	"testing"
)

func treeCompactionCache(cfg Config, n int) *KVCache {
	c := NewKVCache(cfg)
	w := c.kvStride()
	for l := range c.K {
		c.K[l] = make([]float32, n*w)
		c.Kraw[l] = make([]float32, n*w)
		c.V[l] = make([]float32, n*w)
		for p := 0; p < n; p++ {
			for x := 0; x < w; x++ {
				i := p*w + x
				c.K[l][i] = float32(100000*l + 1000*p + x)
				c.Kraw[l][i] = float32(200000*l + 1000*p + x)
				c.V[l][i] = float32(300000*l + 1000*p + x)
			}
		}
	}
	for p := 0; p < n; p++ {
		c.appendPosition(p, 500+p)
	}
	return c
}

func TestKVTreeCompactionContiguousRetainsExactRowsAndLineage(t *testing.T) {
	cfg := cfgV(32, 2, 2, 1, 16, 64)
	const prefix, tree = 3, 7
	c := treeCompactionCache(cfg, prefix+tree)
	before := c.Clone()
	path := []int{0, 2, 5}
	for dst, rel := range path {
		c.pos[prefix+rel] = prefix + dst // VerifyForward tree rows carry path depth, not panel index.
		before.pos[prefix+rel] = prefix + dst
	}
	if err := PruneAndCompactTreeKV(c, prefix, path, tree); err != nil {
		t.Fatal(err)
	}
	if got, want := c.Len(), prefix+len(path); got != want {
		t.Fatalf("Len=%d, want %d", got, want)
	}
	w := c.kvStride()
	for l := range c.K {
		for dst, rel := range path {
			srcPos, dstPos := prefix+rel, prefix+dst
			for name, pair := range map[string][2][]float32{"K": {before.K[l], c.K[l]}, "Kraw": {before.Kraw[l], c.Kraw[l]}, "V": {before.V[l], c.V[l]}} {
				want := pair[0][srcPos*w : (srcPos+1)*w]
				got := pair[1][dstPos*w : (dstPos+1)*w]
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("layer %d %s accepted node %d changed bits", l, name, rel)
				}
			}
		}
	}
	for i, pos := range c.pos {
		if pos != i {
			t.Fatalf("pos[%d]=%d, want compact linear position", i, pos)
		}
	}
	wantTokens := []int{500, 501, 502, 503, 505, 508}
	if _, err := c.lineage.verify(c.pos, wantTokens); err != nil {
		t.Fatalf("lineage did not follow accepted rows: %v", err)
	}
}

func TestKVTreeCompactionInvalidInputIsAtomic(t *testing.T) {
	cfg := cfgV(32, 2, 2, 1, 16, 64)
	cases := []struct {
		name string
		path []int
	}{
		{"negative", []int{-1}},
		{"out_of_range", []int{0, 6}},
		{"duplicate", []int{0, 0}},
		{"descending", []int{2, 1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := treeCompactionCache(cfg, 8)
			before := c.Clone()
			if err := PruneAndCompactTreeKV(c, 2, tc.path, 6); err == nil {
				t.Fatal("invalid path succeeded")
			}
			if !reflect.DeepEqual(c, before) {
				t.Fatal("cache mutated before rejecting invalid path")
			}
		})
	}
}

func TestKVTreeCompactionPagedBoundaryRecyclesPages(t *testing.T) {
	cfg := cfgV(16, 2, 2, 1, 8, 32)
	pool := NewPagedKVPoolWithRaw(cfg, 4)
	s := pool.NewSequence()
	for p := 0; p < 12; p++ {
		appendPagedTreeRow(s, cfg, p)
	}
	if err := PruneAndCompactPagedTreeKV(s, 3, []int{0, 3, 7}, 9); err != nil {
		t.Fatal(err)
	}
	if s.Len() != 6 || s.Blocks() != 2 {
		t.Fatalf("paged shape len=%d blocks=%d, want 6/2", s.Len(), s.Blocks())
	}
	if pool.FreeBlocks() != 1 {
		t.Fatalf("free pages=%d, want 1 recycled page", pool.FreeBlocks())
	}
	wantPos := []int{0, 1, 2, 3, 6, 10}
	assertPagedRows(t, s, cfg, wantPos)
}

func TestKVTreeCompactionPagedCOWPreservesFork(t *testing.T) {
	cfg := cfgV(16, 1, 2, 1, 8, 32)
	pool := NewPagedKVPoolWithRaw(cfg, 4)
	parent := pool.NewSequence()
	for p := 0; p < 8; p++ {
		appendPagedTreeRow(parent, cfg, p)
	}
	child := parent.Fork()
	beforeK, beforeR, beforeV := parent.GatherK(0), parent.GatherKraw(0), parent.GatherV(0)
	// Pre-grow one recyclable block so the COW split need not grow backing storage.
	spare := pool.Alloc()
	pool.Release(spare)
	if err := PruneAndCompactPagedTreeKV(child, 2, []int{0, 3}, 6); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(parent.GatherK(0), beforeK) || !reflect.DeepEqual(parent.GatherKraw(0), beforeR) || !reflect.DeepEqual(parent.GatherV(0), beforeV) {
		t.Fatal("compacting fork mutated shared parent")
	}
	assertPagedRows(t, child, cfg, []int{0, 1, 2, 5})
}

func TestKVTreeCompactionPagedCapacityRefusalIsAtomic(t *testing.T) {
	cfg := cfgV(16, 1, 2, 1, 8, 32)
	pool := NewPagedKVPoolWithRaw(cfg, 4)
	parent := pool.NewSequence()
	for p := 0; p < 8; p++ {
		appendPagedTreeRow(parent, cfg, p)
	}
	child := parent.Fork()
	pool.MaxBlocks = pool.TotalBlocks() // no free or growth capacity for a destination-page COW split
	beforeTable := append([]int(nil), child.table...)
	beforeBlocks := make([][]float32, len(pool.blocks))
	for i := range pool.blocks {
		beforeBlocks[i] = append([]float32(nil), pool.blocks[i]...)
	}
	beforeRef := append([]int(nil), pool.ref...)
	beforeFree := make([]int, len(pool.free)) // preserve non-nil empty allocator state
	copy(beforeFree, pool.free)
	if err := PruneAndCompactPagedTreeKV(child, 2, []int{0, 3}, 6); err == nil {
		t.Fatal("capacity-constrained COW compaction succeeded")
	}
	if child.nTokens != 8 {
		t.Errorf("capacity refusal token count=%d, want 8", child.nTokens)
	}
	if !reflect.DeepEqual(child.table, beforeTable) {
		t.Errorf("capacity refusal table=%v, want %v", child.table, beforeTable)
	}
	if !reflect.DeepEqual(pool.blocks, beforeBlocks) {
		t.Error("capacity refusal mutated physical block bytes")
	}
	if !reflect.DeepEqual(pool.ref, beforeRef) {
		t.Errorf("capacity refusal refs=%v, want %v", pool.ref, beforeRef)
	}
	if !reflect.DeepEqual(pool.free, beforeFree) {
		t.Errorf("capacity refusal free=%v, want %v", pool.free, beforeFree)
	}
}

func TestKVTreeCompactionPagedMalformedLayoutIsAtomic(t *testing.T) {
	cfg := cfgV(16, 1, 2, 1, 8, 32)
	pool := NewPagedKVPool(cfg, 4) // two planes cannot retain Kraw
	s := pool.NewSequence()
	for p := 0; p < 4; p++ {
		w := cfg.NumKVHeads * cfg.HeadDim
		s.Append([][]float32{make([]float32, w)}, [][]float32{make([]float32, w)})
	}
	beforeTable, beforeRef := append([]int(nil), s.table...), append([]int(nil), pool.ref...)
	if err := PruneAndCompactPagedTreeKV(s, 1, []int{0}, 3); err == nil {
		t.Fatal("two-plane paged cache accepted tree compaction")
	}
	if s.nTokens != 4 || !reflect.DeepEqual(s.table, beforeTable) || !reflect.DeepEqual(pool.ref, beforeRef) {
		t.Fatal("malformed layout refusal mutated paged cache")
	}
}

func TestKVTreeCompactionZeroAllocations(t *testing.T) {
	cfg := cfgV(64, 4, 4, 2, 16, 128)
	c := treeCompactionCache(cfg, 40)
	path := []int{0, 5, 17, 31}
	allocs := testing.AllocsPerRun(100, func() {
		for l := range c.K {
			c.K[l] = c.K[l][:40*c.kvStride()]
			c.Kraw[l] = c.Kraw[l][:40*c.kvStride()]
			c.V[l] = c.V[l][:40*c.kvStride()]
		}
		c.pos = c.pos[:40]
		c.lineage.ids = c.lineage.ids[:40]
		for dst, rel := range path {
			c.pos[8+rel] = 8 + dst
		}
		if err := PruneAndCompactTreeKV(c, 8, path, 32); err != nil {
			panic(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocations/run=%v, want 0", allocs)
	}
}

func TestKVTreeCompactionPagedFirstCallZeroAllocations(t *testing.T) {
	cfg := cfgV(16, 1, 2, 1, 8, 32)
	seqs := make([]*PagedKV, 101) // AllocsPerRun performs one warm-up call plus 100 measured calls.
	for i := range seqs {
		pool := NewPagedKVPoolWithRaw(cfg, 4)
		seqs[i] = pool.NewSequence()
		for p := 0; p < 8; p++ {
			appendPagedTreeRow(seqs[i], cfg, p)
		}
	}
	next := 0
	allocs := testing.AllocsPerRun(100, func() {
		s := seqs[next]
		next++
		if err := PruneAndCompactPagedTreeKV(s, 2, []int{0, 3}, 6); err != nil {
			panic(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("first-call paged allocations/run=%v, want 0", allocs)
	}
}

func BenchmarkKVTreeCompaction(b *testing.B) {
	// Realistic GQA cache geometry; this is a CPU memory-compaction receipt, not a hardware latency claim.
	cfg := cfgV(4096, 32, 32, 8, 128, 11008)
	const prefix, tree = 128, 32
	c := treeCompactionCache(cfg, prefix+tree)
	path := []int{0, 5, 17, 31}
	w := c.kvStride()
	b.ReportAllocs()
	b.SetBytes(int64(cfg.NumLayers * (len(path) - 1) * w * 3 * 4)) // root row is already in its destination slot
	b.ResetTimer()
	b.StopTimer()
	for i := 0; i < b.N; i++ {
		for l := range c.K {
			c.K[l] = c.K[l][:(prefix+tree)*w]
			c.Kraw[l] = c.Kraw[l][:(prefix+tree)*w]
			c.V[l] = c.V[l][:(prefix+tree)*w]
		}
		c.pos = c.pos[:prefix+tree]
		c.lineage.ids = c.lineage.ids[:prefix+tree]
		for dst, rel := range path {
			c.pos[prefix+rel] = prefix + dst
		}
		b.StartTimer()
		if err := PruneAndCompactTreeKV(c, prefix, path, tree); err != nil {
			b.Fatal(err)
		}
		b.StopTimer()
	}
}

func BenchmarkKVTreeCompactionSteady(b *testing.B) {
	// Same realistic geometry and bytes as BenchmarkKVTreeCompaction. This variant keeps
	// the benchmark clock running across iterations so per-iteration timer bookkeeping
	// cannot perturb the warm-memory measurement; conservative slice/position reset work
	// remains inside the timed region.
	cfg := cfgV(4096, 32, 32, 8, 128, 11008)
	const prefix, tree = 128, 32
	c := treeCompactionCache(cfg, prefix+tree)
	path := []int{0, 5, 17, 31}
	w := c.kvStride()
	b.ReportAllocs()
	b.SetBytes(int64(cfg.NumLayers * (len(path) - 1) * w * 3 * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for l := range c.K {
			c.K[l] = c.K[l][:(prefix+tree)*w]
			c.Kraw[l] = c.Kraw[l][:(prefix+tree)*w]
			c.V[l] = c.V[l][:(prefix+tree)*w]
		}
		c.pos = c.pos[:prefix+tree]
		c.lineage.ids = c.lineage.ids[:prefix+tree]
		for dst, rel := range path {
			c.pos[prefix+rel] = prefix + dst
		}
		if err := PruneAndCompactTreeKV(c, prefix, path, tree); err != nil {
			b.Fatal(err)
		}
	}
}

func appendPagedTreeRow(s *PagedKV, cfg Config, p int) {
	w := cfg.NumKVHeads * cfg.HeadDim
	k, r, v := make([][]float32, cfg.NumLayers), make([][]float32, cfg.NumLayers), make([][]float32, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		k[l], r[l], v[l] = make([]float32, w), make([]float32, w), make([]float32, w)
		for x := 0; x < w; x++ {
			k[l][x] = float32(100000*l + 1000*p + x)
			r[l][x] = float32(200000*l + 1000*p + x)
			v[l][x] = float32(300000*l + 1000*p + x)
		}
	}
	s.AppendRaw(k, r, v)
}

func assertPagedRows(t *testing.T, s *PagedKV, cfg Config, oldPos []int) {
	t.Helper()
	w := cfg.NumKVHeads * cfg.HeadDim
	for l := 0; l < cfg.NumLayers; l++ {
		for plane, got := range map[string][]float32{"K": s.GatherK(l), "Kraw": s.GatherKraw(l), "V": s.GatherV(l)} {
			for p, old := range oldPos {
				for x := 0; x < w; x++ {
					base := 100000 * l
					if plane == "Kraw" {
						base = 200000 * l
					}
					if plane == "V" {
						base = 300000 * l
					}
					want := float32(base + 1000*old + x)
					if got[p*w+x] != want {
						t.Fatalf("%s layer %d row %d[%d]=%v want %v", plane, l, p, x, got[p*w+x], want)
					}
				}
			}
		}
	}
}
