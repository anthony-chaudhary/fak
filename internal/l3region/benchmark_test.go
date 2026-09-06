package l3region

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

var (
	benchSinkRef      abi.Ref
	benchSinkBytes    []byte
	benchSinkKeys     []string
	benchSinkDecision L3Decision
	benchSinkInt      int
	benchSinkErr      error
)

func distinctBenchPayload(seq int, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(((i + seq) * 31) + 7)
	}
	return b
}

// TestBenchmarkSanity verifies that the benchmarked paths execute cleanly before
// running benchmarks.
func TestBenchmarkSanity(t *testing.T) {
	ctx := context.Background()
	be := New(NewL3Store())
	data := payload(PageBytes * 2)

	ref, err := be.Put(ctx, data)
	if err != nil {
		t.Fatalf("sanity Put: %v", err)
	}
	resolved, err := be.Resolve(ctx, ref)
	if err != nil || len(resolved) != len(data) {
		t.Fatalf("sanity Resolve: %v", err)
	}
	keys, err := be.PageKeys(ref)
	if err != nil || len(keys) != 2 {
		t.Fatalf("sanity PageKeys: %v, len=%d", err, len(keys))
	}

	chain := RegionPrefixChainKeys(data)
	if len(chain) != 2 {
		t.Fatalf("sanity RegionPrefixChainKeys: len=%d", len(chain))
	}
	if match := PrefixChainMatchLen(chain, chain); match != 2 {
		t.Fatalf("sanity PrefixChainMatchLen: %d", match)
	}

	gate := NewL3PromotionGate().WithMode(L3PromotionEnforce)
	if d := gate.Admit(DurabilityDurable); !d.Admit {
		t.Fatalf("sanity Admit(Durable): %+v", d)
	}
	if _, d, err := be.PutGated(ctx, data, DurabilityTurn, gate); d.Admit || err != nil {
		t.Fatalf("sanity PutGated(Turn): admit=%v, err=%v", d.Admit, err)
	}
}

// BenchmarkPut measures region allocation and chunking/msetting across representative
// payload sizes: single-page (1KB), multi-page (16KB = 4 pages), and large (64KB = 16 pages).
func BenchmarkPut(b *testing.B) {
	ctx := context.Background()
	sizes := []struct {
		name string
		size int
	}{
		{"SinglePage_1KB", 1024},
		{"MultiPage_16KB", PageBytes * 4},
		{"Large_64KB", PageBytes * 16},
	}
	for _, tc := range sizes {
		b.Run(tc.name, func(b *testing.B) {
			be := New(NewL3Store())
			data := payload(tc.size)
			b.SetBytes(int64(tc.size))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkRef, benchSinkErr = be.Put(ctx, data)
			}
		})
	}
}

// BenchmarkPut_Dedup measures repeated Put of an already-resident region payload,
// measuring chunking, hashing, manifest update, and store dedup idempotency.
func BenchmarkPut_Dedup(b *testing.B) {
	ctx := context.Background()
	be := New(NewL3Store())
	data := payload(PageBytes * 4)
	if _, err := be.Put(ctx, data); err != nil {
		b.Fatalf("seed Put: %v", err)
	}

	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkRef, benchSinkErr = be.Put(ctx, data)
	}
}

// BenchmarkPut_Distinct measures continuous insertion of distinct regions,
// exercising unique digest generation, page chunking, store mutation, and manifest indexing.
func BenchmarkPut_Distinct(b *testing.B) {
	ctx := context.Background()
	const size = PageBytes * 2
	const poolSize = 512
	payloads := make([][]byte, poolSize)
	for i := range payloads {
		payloads[i] = distinctBenchPayload(i, size)
	}
	be := New(NewL3Store())

	b.SetBytes(int64(size))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkRef, benchSinkErr = be.Put(ctx, payloads[i%poolSize])
	}
}

// BenchmarkResolve measures region lookup and bit-exact materialization, including
// manifest lookup, store Mget, per-page sha256 verification, reassembly, and whole-region
// digest verification.
func BenchmarkResolve(b *testing.B) {
	ctx := context.Background()
	sizes := []struct {
		name string
		size int
	}{
		{"SinglePage_1KB", 1024},
		{"MultiPage_16KB", PageBytes * 4},
		{"Large_64KB", PageBytes * 16},
	}
	for _, tc := range sizes {
		b.Run(tc.name, func(b *testing.B) {
			be := New(NewL3Store())
			data := payload(tc.size)
			ref, err := be.Put(ctx, data)
			if err != nil {
				b.Fatalf("setup Put: %v", err)
			}
			b.SetBytes(int64(tc.size))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkBytes, benchSinkErr = be.Resolve(ctx, ref)
			}
		})
	}
}

// BenchmarkResolve_Inline measures resolving inline Ref payloads which bypass
// store and manifest lookups.
func BenchmarkResolve_Inline(b *testing.B) {
	ctx := context.Background()
	be := New(NewL3Store())
	inline := abi.Ref{
		Kind:   abi.RefInline,
		Inline: []byte("inline-arguments-data-for-l3-region-resolver"),
		Len:    44,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes, benchSinkErr = be.Resolve(ctx, inline)
	}
}

// BenchmarkResolve_Parallel measures concurrent region resolution across goroutines.
func BenchmarkResolve_Parallel(b *testing.B) {
	ctx := context.Background()
	be := New(NewL3Store())
	data := payload(PageBytes * 4)
	ref, err := be.Put(ctx, data)
	if err != nil {
		b.Fatalf("setup Put: %v", err)
	}

	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			res, err := be.Resolve(ctx, ref)
			if err != nil || len(res) != len(data) {
				b.Fatalf("parallel Resolve: %v", err)
			}
		}
	})
}

// BenchmarkPageKeys measures manifest lookup returning the ordered page-key set.
func BenchmarkPageKeys(b *testing.B) {
	ctx := context.Background()
	be := New(NewL3Store())
	data := payload(PageBytes * 8)
	ref, err := be.Put(ctx, data)
	if err != nil {
		b.Fatalf("setup Put: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkKeys, benchSinkErr = be.PageKeys(ref)
	}
}

// BenchmarkL3Store_Mset measures batch insertion of chunked pages into the L3Store.
func BenchmarkL3Store_Mset(b *testing.B) {
	pages, _ := chunk(payload(PageBytes * 4))
	store := NewL3Store()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.Mset(pages)
	}
}

// BenchmarkL3Store_Mget measures batch retrieval of resident pages by key.
func BenchmarkL3Store_Mget(b *testing.B) {
	store := NewL3Store()
	pages, keys := chunk(payload(PageBytes * 4))
	store.Mset(pages)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, missing, ok := store.Mget(keys)
		if !ok || missing != "" {
			b.Fatalf("Mget failed")
		}
		benchSinkBytes = out[0]
	}
}

// BenchmarkL3Store_Mdel measures idempotent batch invalidation of page keys from the store.
func BenchmarkL3Store_Mdel(b *testing.B) {
	pages, keys := chunk(payload(PageBytes * 4))
	store := NewL3Store()
	store.Mset(pages)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkInt = store.Mdel(keys)
	}
}

// BenchmarkL3Store_MsetMdel measures a full store insertion and eviction cycle.
func BenchmarkL3Store_MsetMdel(b *testing.B) {
	pages, keys := chunk(payload(PageBytes * 4))
	store := NewL3Store()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.Mset(pages)
		benchSinkInt = store.Mdel(keys)
	}
}

// BenchmarkRegionPrefixChainKeys measures generation of deterministic prefix-hash chains
// across region boundaries.
func BenchmarkRegionPrefixChainKeys(b *testing.B) {
	sizes := []struct {
		name string
		size int
	}{
		{"4Pages_16KB", PageBytes * 4},
		{"16Pages_64KB", PageBytes * 16},
	}
	for _, tc := range sizes {
		b.Run(tc.name, func(b *testing.B) {
			data := payload(tc.size)
			b.SetBytes(int64(tc.size))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkKeys = RegionPrefixChainKeys(data)
			}
		})
	}
}

// BenchmarkPrefixChainMatchLen measures prefix hit length comparison between two chains.
func BenchmarkPrefixChainMatchLen(b *testing.B) {
	dataA := payload(PageBytes * 16)
	dataB := payload(PageBytes * 16)
	dataB[PageBytes*8] ^= 0xFF
	chainA := RegionPrefixChainKeys(dataA)
	chainB := RegionPrefixChainKeys(dataB)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkInt = PrefixChainMatchLen(chainA, chainB)
	}
}

// BenchmarkPromotionGate_Admit measures admission decision latency across durability classes.
func BenchmarkPromotionGate_Admit(b *testing.B) {
	gate := NewL3PromotionGate().WithMode(L3PromotionEnforce)
	classes := []struct {
		name  string
		class string
	}{
		{"Durable", DurabilityDurable},
		{"Bounded", DurabilityBounded},
		{"Turn", DurabilityTurn},
		{"Unknown", "unregistered_class"},
	}
	for _, tc := range classes {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkDecision = gate.Admit(tc.class)
			}
		})
	}
}

// BenchmarkPutGated measures G6 durability-tiered region admission throughput under both
// admitted (durable) and denied (turn) conditions.
func BenchmarkPutGated(b *testing.B) {
	ctx := context.Background()
	data := payload(PageBytes * 4)

	b.Run("Admitted_Durable", func(b *testing.B) {
		be := New(NewL3Store())
		gate := NewL3PromotionGate().WithMode(L3PromotionEnforce)
		b.SetBytes(int64(len(data)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkRef, benchSinkDecision, benchSinkErr = be.PutGated(ctx, data, DurabilityDurable, gate)
		}
	})

	b.Run("Denied_Turn", func(b *testing.B) {
		be := New(NewL3Store())
		gate := NewL3PromotionGate().WithMode(L3PromotionEnforce)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkRef, benchSinkDecision, benchSinkErr = be.PutGated(ctx, data, DurabilityTurn, gate)
		}
	})
}
