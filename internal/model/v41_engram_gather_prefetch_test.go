package model

import "testing"

type v41EngramTruncatingReadahead struct {
	*v41EngramBatchSource
}

func (s v41EngramTruncatingReadahead) ReadRows(start, count int, dst []byte) (int, error) {
	n, err := s.v41EngramBatchSource.ReadRows(start, count, dst)
	if count > 1 && n > 0 {
		n--
	}
	return n, err
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=1s lane=default
func TestV41EngramGatherHonorsPrefetchRows(t *testing.T) {
	t.Parallel()
	src := newV41EngramBatchSource(8, V41EngramPackedRowBytes, 0)
	cache, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{
		TableRows: src.rows, RowBytes: src.rowBytes, BudgetBytes: int64(4 * src.rowBytes), PrefetchRows: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	ids := []uint32{0, 2, 2}
	got, err := GatherV41EngramRows([]*V41EngramRowCache{cache}, ids, 1)
	if err != nil {
		t.Fatal(err)
	}
	v41EngramBatchCheckRows(t, got, ids, []*v41EngramBatchSource{src}, 1)
	stats := cache.Stats()
	if stats.Misses != 1 || stats.Hits != 2 || stats.Prefetches != 2 || len(src.reads) != 2 {
		t.Fatalf("misses=%d hits=%d prefetches=%d reads=%d, want 1/2/2/2", stats.Misses, stats.Hits, stats.Prefetches, len(src.reads))
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=1s lane=default
func TestV41EngramTruncatedReadaheadNeverPoisonsGather(t *testing.T) {
	t.Parallel()
	inner := newV41EngramBatchSource(8, V41EngramPackedRowBytes, 0)
	cache, err := NewV41EngramRowCache(v41EngramTruncatingReadahead{inner}, V41EngramRowCacheOptions{
		TableRows: inner.rows, RowBytes: inner.rowBytes, BudgetBytes: int64(4 * inner.rowBytes), PrefetchRows: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	ids := []uint32{0, 2}
	got, err := GatherV41EngramRows([]*V41EngramRowCache{cache}, ids, 1)
	if err != nil {
		t.Fatal(err)
	}
	v41EngramBatchCheckRows(t, got, ids, []*v41EngramBatchSource{inner}, 1)
}
