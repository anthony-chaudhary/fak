package model

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestKVCacheToPagedSnapshotRows(t *testing.T) {
	cfg := Config{NumLayers: 4, NumKVHeads: 2, HeadDim: 8}
	cache := snapshotRowsCache(cfg, 128)
	pool := NewPagedKVPoolWithRaw(cfg, 16)
	seq, err := KVCacheToPaged(pool, cache)
	if err != nil {
		t.Fatal(err)
	}
	defer seq.Free()
	want := cache.Clone()
	for l := range cache.K {
		cache.K[l][0] = 11
		cache.Kraw[l][0] = 22
		cache.V[l][0] = 33
	}
	got := seq.ToKVCache(cfg)
	if got.Len() != want.Len() {
		t.Fatalf("snapshot length = %d, want %d", got.Len(), want.Len())
	}
	for l := range want.K {
		for p, rows := range [][][]float32{{got.K[l], want.K[l]}, {got.Kraw[l], want.Kraw[l]}, {got.V[l], want.V[l]}} {
			if len(rows[0]) != len(rows[1]) {
				t.Fatalf("layer %d plane %d length mismatch", l, p)
			}
			for i := range rows[0] {
				if math.Float32bits(rows[0][i]) != math.Float32bits(rows[1][i]) {
					t.Fatalf("layer %d plane %d row value %d changed after source mutation", l, p, i)
				}
			}
		}
	}
	seq.Free()
	if pool.PhysicalBlocks() != 0 {
		t.Fatal("snapshot pages remain owned after Free")
	}

	// A late short row must still release pages from earlier successful appends.
	short := snapshotRowsCache(cfg, 17)
	short.Kraw[3] = short.Kraw[3][:16*short.kvStride()]
	if partial, err := KVCacheToPaged(pool, short); err == nil || partial != nil {
		t.Fatalf("short-row snapshot returned sequence=%v, error=%v", partial != nil, err)
	}
	if pool.PhysicalBlocks() != 0 {
		t.Fatal("failed snapshot leaked partial pages")
	}
	empty, err := KVCacheToPaged(pool, NewKVCache(cfg))
	if err != nil || empty == nil || empty.Len() != 0 || pool.PhysicalBlocks() != 0 {
		t.Fatal("empty snapshot changed ownership or failed")
	}
	empty.Free()

	// Pool backing is warmed above. The remaining sequence/table allocations
	// may grow with page count, but not with every layer/plane/token row.
	allocs := testing.AllocsPerRun(10, func() {
		s, err := KVCacheToPaged(pool, cache)
		if err != nil {
			panic(err)
		}
		s.Free()
	})
	if allocs > 32 {
		t.Fatalf("warm snapshot allocations = %.0f, want <=32 without per-row payload allocations", allocs)
	}
}

// BenchmarkKVCacheToPagedSnapshotRows measures temporary bytes/op after required
// page backing is warm. Compare this identical fixture against the pinned old
// conversion; it is not end-to-end receiver latency or device performance.
func BenchmarkKVCacheToPagedSnapshotRows(b *testing.B) {
	cfg := Config{NumLayers: 4, NumKVHeads: 2, HeadDim: 8}
	cache := snapshotRowsCache(cfg, 128)
	pool := NewPagedKVPoolWithRaw(cfg, 16)
	warm, err := KVCacheToPaged(pool, cache)
	if err != nil {
		b.Fatal(err)
	}
	warm.Free()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		seq, err := KVCacheToPaged(pool, cache)
		if err != nil {
			b.Fatal(err)
		}
		seq.Free()
	}
}

func snapshotRowsCache(cfg Config, tokens int) *KVCache {
	c := NewKVCache(cfg)
	c.pos = make([]int, tokens)
	for i := range c.pos {
		c.pos[i] = i
	}
	for l := range c.K {
		planes := []*[]float32{&c.K[l], &c.Kraw[l], &c.V[l]}
		for p, plane := range planes {
			*plane = make([]float32, tokens*c.kvStride())
			for i := range *plane {
				(*plane)[i] = float32(1 + l*10000 + p*3000 + i)
			}
			if len(*plane) > 1 {
				(*plane)[0] = math.Float32frombits(0x80000000)
				(*plane)[1] = math.Float32frombits(0x7fc00001)
			}
		}
	}
	return c
}
