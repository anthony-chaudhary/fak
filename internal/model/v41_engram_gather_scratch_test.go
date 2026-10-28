package model

import (
	"errors"
	"io"
	"testing"
)

type v41GatherScratchSource struct {
	starts []int
	counts []int
	bases  []*byte
	failAt int
}

func (*v41GatherScratchSource) RowBytes() int { return V41EngramPackedRowBytes }

func (s *v41GatherScratchSource) ReadRows(start, count int, dst []byte) (int, error) {
	s.starts = append(s.starts, start)
	s.counts = append(s.counts, count)
	s.bases = append(s.bases, &dst[0])
	for row := 0; row < count; row++ {
		for j := 0; j < s.RowBytes(); j++ {
			dst[row*s.RowBytes()+j] = byte(start + row)
		}
	}
	if s.failAt > 0 && len(s.starts) == s.failAt {
		return count * s.RowBytes(), io.ErrUnexpectedEOF
	}
	return count * s.RowBytes(), nil
}

func TestV41EngramGatherReusesReadScratch(t *testing.T) {
	for _, contiguous := range []bool{false, true} {
		t.Run(map[bool]string{false: "independent", true: "panel"}[contiguous], func(t *testing.T) {
			src := &v41GatherScratchSource{}
			cache, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{
				TableRows: 16, RowBytes: src.RowBytes(), BudgetBytes: int64(2 * src.RowBytes()),
			})
			if err != nil {
				t.Fatal(err)
			}
			ids := []uint32{9, 2, 1, 8, 5, 2}
			out, panel, err := cache.gatherKnownRowsOwned(ids, contiguous)
			if err != nil {
				t.Fatal(err)
			}
			if len(src.starts) != 3 || src.starts[0] != 1 || src.starts[1] != 5 || src.starts[2] != 8 ||
				src.counts[0] != 2 || src.counts[1] != 1 || src.counts[2] != 2 {
				t.Fatalf("read ranges changed: starts=%v counts=%v", src.starts, src.counts)
			}
			if src.bases[0] != src.bases[1] || src.bases[0] != src.bases[2] {
				t.Fatal("coalesced reads did not reuse one scratch slab")
			}
			for i, row := range out {
				if len(row) != src.RowBytes() {
					t.Fatalf("row %d has width %d", i, len(row))
				}
				for _, b := range row {
					if b != byte(ids[i]) {
						t.Fatalf("row %d was overwritten by later scratch reuse", i)
					}
				}
			}
			out[1][0] = 255
			if out[5][0] != 2 {
				t.Fatal("duplicate result rows alias")
			}
			cached, ok := cache.inner.Get(0, 2)
			if !ok || cached[0] != 2 {
				t.Fatal("returned result aliases resident cache")
			}
			if contiguous != (len(panel) == len(ids)*src.RowBytes()) {
				t.Fatal("output panel ownership changed")
			}
			stats := cache.Stats()
			if stats.Misses != 5 || stats.Hits != 1 || stats.RowsServed != 6 || stats.BytesRead != int64(5*src.RowBytes()) {
				t.Fatalf("traffic semantics changed: %+v", stats)
			}
		})
	}
}

func TestV41EngramGatherScratchFailureDoesNotPublishPanel(t *testing.T) {
	src := &v41GatherScratchSource{failAt: 2}
	cache, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{
		TableRows: 16, RowBytes: src.RowBytes(), BudgetBytes: int64(2 * src.RowBytes()),
	})
	if err != nil {
		t.Fatal(err)
	}
	out, panel, err := cache.gatherKnownRowsOwned([]uint32{1, 2, 5, 8}, true)
	if !errors.Is(err, io.ErrUnexpectedEOF) || out != nil || panel != nil {
		t.Fatalf("partial read published output: out=%v panel=%v err=%v", out, panel, err)
	}
	if len(src.bases) != 2 || src.bases[0] != src.bases[1] {
		t.Fatal("failing read did not use the same bounded scratch")
	}
}
