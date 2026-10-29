package model

import (
	"errors"
	"testing"
	"unsafe"
)

// v41GatherScratchFake records the identity and capacity of each ReadRows
// destination so tests can witness whether a gather reuses one bounded
// source-read slab across coalesced miss spans instead of allocating a fresh
// buffer per span.
type v41GatherScratchFake struct {
	rowBytes int
	starts   []int
	counts   []int
	dstPtrs  []uintptr
	dstLens  []int
	dstCaps  []int
	failAt   int // 1-based index of the ReadRows call to fail; 0 = never
}

func (s *v41GatherScratchFake) RowBytes() int { return s.rowBytes }

func (s *v41GatherScratchFake) ReadRows(start, count int, dst []byte) (int, error) {
	s.starts = append(s.starts, start)
	s.counts = append(s.counts, count)
	if len(dst) > 0 {
		s.dstPtrs = append(s.dstPtrs, uintptr(unsafe.Pointer(unsafe.SliceData(dst))))
	} else {
		s.dstPtrs = append(s.dstPtrs, 0)
	}
	s.dstLens = append(s.dstLens, len(dst))
	s.dstCaps = append(s.dstCaps, cap(dst))
	if s.failAt > 0 && len(s.starts) == s.failAt {
		return 0, errors.New("synthetic Engram source failure")
	}
	n := 0
	for r := 0; r < count; r++ {
		for b := 0; b < s.rowBytes; b++ {
			dst[n] = byte(start + r)
			n++
		}
	}
	return n, nil
}

func v41GatherScratchCache(t *testing.T, src V41EngramRowSource, tableRows, rowBytes, budget int) *V41EngramRowCache {
	t.Helper()
	c, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{
		TableRows: tableRows, RowBytes: rowBytes, PrefetchRows: 0, BudgetBytes: int64(budget),
	})
	if err != nil {
		t.Fatalf("NewV41EngramRowCache: %v", err)
	}
	return c
}

// [SW-VERIFIED] A gather whose sorted misses form three coalesced spans of 2, 1
// and 2 rows reads through one reused source-read slab: every ReadRows
// destination is backed by the same array with the full slab capacity, while
// the issued ranges, payload bytes and stats stay unchanged from the
// per-miss allocation behavior.
func TestV41EngramGatherScratchReusedAcrossSpans(t *testing.T) {
	const rowBytes = 16
	src := &v41GatherScratchFake{rowBytes: rowBytes}
	c := v41GatherScratchCache(t, src, 64, rowBytes, rowBytes*8) // maxRows = 8
	rows := []uint32{0, 1, 3, 5, 6}                              // sorted missing spans: [0,1] [3] [5,6]

	out, err := c.gatherKnownRows(rows)
	if err != nil {
		t.Fatalf("gatherKnownRows: %v", err)
	}
	if len(out) != 5 {
		t.Fatalf("len(out)=%d want 5", len(out))
	}
	wantStarts := []int{0, 3, 5}
	wantCounts := []int{2, 1, 2}
	if len(src.starts) != 3 || len(src.counts) != 3 {
		t.Fatalf("ReadRows calls=%d want 3 (starts=%v counts=%v)", len(src.starts), src.starts, src.counts)
	}
	for i := range wantStarts {
		if src.starts[i] != wantStarts[i] || src.counts[i] != wantCounts[i] {
			t.Fatalf("span %d = (%d,%d) want (%d,%d)", i, src.starts[i], src.counts[i], wantStarts[i], wantCounts[i])
		}
	}
	// The largest planned span is two rows; every destination must be a view of
	// one slab sized for it, not a fresh exact-size allocation per span.
	slabSize := 2 * rowBytes
	for i := 0; i < 3; i++ {
		if src.dstCaps[i] != slabSize {
			t.Fatalf("span %d dst cap=%d want %d (per-span allocation still in place)", i, src.dstCaps[i], slabSize)
		}
	}
	if src.dstPtrs[0] == 0 || src.dstPtrs[0] != src.dstPtrs[1] || src.dstPtrs[0] != src.dstPtrs[2] {
		t.Fatalf("read destinations not backed by one reused slab: ptrs=%v", src.dstPtrs)
	}
	wantLens := []int{2 * rowBytes, rowBytes, 2 * rowBytes}
	for i, want := range wantLens {
		if src.dstLens[i] != want {
			t.Fatalf("span %d dst len=%d want %d", i, src.dstLens[i], want)
		}
	}
	for p, row := range rows {
		for b := 0; b < rowBytes; b++ {
			if out[p][b] != byte(row) {
				t.Fatalf("out[%d][%d]=%d want %d", p, b, out[p][b], byte(row))
			}
		}
	}
	st := c.Stats()
	if st.Misses != 5 || st.RowsServed != 5 || st.BytesRead != 80 || st.BytesServed != 80 ||
		st.UsefulBytes != 80 || st.Hits != 0 || st.Prefetches != 0 {
		t.Fatalf("stats changed: %+v", st)
	}
	allocs := testing.AllocsPerRun(25, func() {
		c.invalidateRows()
		if _, err := c.gatherKnownRows(rows); err != nil {
			panic(err)
		}
	})
	t.Logf("[SW-VERIFIED] gather allocs/op over 2/1/2 spans, fresh miss each run = %.0f", allocs)
}

// [SW-VERIFIED] Duplicate demand positions and the retained cache copy stay
// independent of the shared source-read slab: mutating one output cannot reach
// its duplicate or the cached payload.
func TestV41EngramGatherScratchIndependentOutputs(t *testing.T) {
	const rowBytes = 16
	src := &v41GatherScratchFake{rowBytes: rowBytes}
	c := v41GatherScratchCache(t, src, 64, rowBytes, rowBytes*8)
	rows := []uint32{7, 7, 8} // one span of two rows; row 7 demanded twice

	out, err := c.gatherKnownRows(rows)
	if err != nil {
		t.Fatalf("gatherKnownRows: %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("len(out)=%d want 3", len(out))
	}
	if unsafe.SliceData(out[0]) == unsafe.SliceData(out[1]) {
		t.Fatalf("duplicate demand positions share one output buffer")
	}
	for b := 0; b < rowBytes; b++ {
		if out[0][b] != 7 || out[1][b] != 7 || out[2][b] != 8 {
			t.Fatalf("payload mismatch: %v %v %v", out[0][:4], out[1][:4], out[2][:4])
		}
	}
	out[0][0] ^= 0xFF
	if out[1][0] != 7 {
		t.Fatalf("mutating out[0] reached duplicate out[1]: %d", out[1][0])
	}
	cached, ok := c.inner.Get(0, 7)
	if !ok {
		t.Fatalf("row 7 not retained after gather")
	}
	if cached[0] != 7 {
		t.Fatalf("mutating out[0] reached cached payload: %d", cached[0])
	}
}

// [SW-VERIFIED] A source failure mid-gather returns an error with no partial
// output, after issuing exactly the spans up to the failing read.
func TestV41EngramGatherScratchNoPartialOutputOnFailure(t *testing.T) {
	const rowBytes = 16
	src := &v41GatherScratchFake{rowBytes: rowBytes, failAt: 2}
	c := v41GatherScratchCache(t, src, 64, rowBytes, rowBytes*8)
	rows := []uint32{0, 1, 3, 5, 6}

	out, err := c.gatherKnownRows(rows)
	if err == nil {
		t.Fatalf("gatherKnownRows succeeded; want synthetic source failure")
	}
	if out != nil {
		t.Fatalf("partial output on failure: %d rows", len(out))
	}
	if len(src.starts) != 2 || src.starts[0] != 0 || src.starts[1] != 3 {
		t.Fatalf("issued spans before failure = %v want [0 3]", src.starts)
	}
}

// [SW-VERIFIED] A legal row wider than the coalescing ceiling keeps its
// singleton reads working through the same bounded slab path.
func TestV41EngramGatherScratchOversizedSingleton(t *testing.T) {
	const rowBytes = 70000 // exceeds the 64 KiB batch ceiling -> maxRows clamps to 1
	src := &v41GatherScratchFake{rowBytes: rowBytes}
	c := v41GatherScratchCache(t, src, 4, rowBytes, rowBytes) // budget >= one row
	rows := []uint32{1, 3}                                    // two non-consecutive singleton spans

	out, err := c.gatherKnownRows(rows)
	if err != nil {
		t.Fatalf("gatherKnownRows: %v", err)
	}
	if len(src.starts) != 2 || src.counts[0] != 1 || src.counts[1] != 1 {
		t.Fatalf("singleton spans = starts %v counts %v, want [1 1]", src.starts, src.counts)
	}
	if src.dstLens[0] != rowBytes || src.dstLens[1] != rowBytes {
		t.Fatalf("dst lens %v want %d", src.dstLens, rowBytes)
	}
	if out[0][0] != 1 || out[1][0] != 3 {
		t.Fatalf("payload mismatch: %d %d", out[0][0], out[1][0])
	}
}
