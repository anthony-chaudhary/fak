package model

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
)

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=1s lane=default
func TestV41EngramBatchExactRows(t *testing.T) {
	t.Parallel()
	for _, rowBytes := range []int{V41EngramPackedRowBytes, V41EngramF32RowBytes} {
		t.Run(fmt.Sprintf("cold_and_resident_%d", rowBytes), func(t *testing.T) {
			ids := []uint32{5, 2, 5, 2, 3, 2, 3, 2, 6, 5, 6, 5}
			sources := []*v41EngramBatchSource{
				newV41EngramBatchSource(8, rowBytes, 0),
				newV41EngramBatchSource(8, rowBytes, 71),
			}
			caches := []*V41EngramRowCache{
				v41EngramBatchCache(t, sources[0], 4*rowBytes),
				v41EngramBatchCache(t, sources[1], 4*rowBytes),
			}
			cold, err := GatherV41EngramRows(caches, ids, 2)
			if err != nil {
				t.Fatal(err)
			}
			v41EngramBatchCheckRows(t, cold, ids, sources, 2)
			for layer, src := range sources {
				v41EngramBatchCheckReads(t, src, []uint32{5, 2, 3, 2, 6, 5}, 4*rowBytes)
				if len(src.reads) != 2 {
					t.Fatalf("layer %d cold ReadRows calls=%d, want 2 adjacent ranges", layer, len(src.reads))
				}
			}
			resident, err := GatherV41EngramRows(caches, ids, 2)
			if err != nil {
				t.Fatal(err)
			}
			v41EngramBatchCheckRows(t, resident, ids, sources, 2)
			for layer, src := range sources {
				if len(src.reads) != 2 {
					t.Fatalf("layer %d resident gather issued %d additional reads", layer, len(src.reads)-2)
				}
			}
		})
	}

	for _, tc := range []struct {
		name                string
		rowBytes, tableRows int
		budgetRows          int
		ids                 []uint32
		wantCalls           int
	}{
		{"eviction", V41EngramPackedRowBytes, 8, 2, []uint32{5, 2, 3, 2, 6, 5}, 2},
		{"packed_64KiB", V41EngramPackedRowBytes, 260, 260, v41EngramBatchIDs(260), 2},
		{"f32_64KiB", V41EngramF32RowBytes, 65, 65, v41EngramBatchIDs(65), 2},
		{"wide_singletons", 64*1024 + 1, 2, 2, []uint32{1, 0, 1}, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := newV41EngramBatchSource(tc.tableRows, tc.rowBytes, 0)
			budget := tc.budgetRows * tc.rowBytes
			cache := v41EngramBatchCache(t, src, budget)
			got, err := GatherV41EngramRows([]*V41EngramRowCache{cache}, tc.ids, 1)
			if err != nil {
				t.Fatal(err)
			}
			v41EngramBatchCheckRows(t, got, tc.ids, []*v41EngramBatchSource{src}, 1)
			v41EngramBatchCheckReads(t, src, tc.ids, budget)
			v41EngramBatchCheckResidency(t, cache, budget)
			if len(src.reads) != tc.wantCalls {
				t.Fatalf("ReadRows calls=%d, want %d bounded adjacent ranges", len(src.reads), tc.wantCalls)
			}
			if tc.budgetRows < tc.tableRows {
				replayed, err := GatherV41EngramRows([]*V41EngramRowCache{cache}, tc.ids, 1)
				if err != nil {
					t.Fatal(err)
				}
				v41EngramBatchCheckRows(t, replayed, tc.ids, []*v41EngramBatchSource{src}, 1)
				v41EngramBatchCheckRows(t, got, tc.ids, []*v41EngramBatchSource{src}, 1)
				v41EngramBatchCheckReadBounds(t, src, budget)
				v41EngramBatchCheckResidency(t, cache, budget)
			}
		})
	}

	t.Run("reject_before_io", func(t *testing.T) {
		for _, ids := range [][]uint32{{1, 8}, {1, ^uint32(0)}} {
			src := newV41EngramBatchSource(8, V41EngramPackedRowBytes, 0)
			cache := v41EngramBatchCache(t, src, 4*src.rowBytes)
			if _, err := GatherV41EngramRows([]*V41EngramRowCache{cache}, ids, 1); err == nil {
				t.Fatal("out-of-bounds gather was accepted")
			}
			if len(src.reads) != 0 {
				t.Fatalf("invalid gather issued %d source reads before refusal", len(src.reads))
			}
		}
	})

	t.Run("source_failures", func(t *testing.T) {
		sentinel := errors.New("injected Engram batch read failure")
		for _, tc := range []struct {
			name, countMode string
			short           bool
			err, want       error
		}{
			{"short_success", "", true, nil, io.ErrUnexpectedEOF},
			{"sentinel", "", false, sentinel, sentinel},
			{"negative_count", "negative", false, nil, nil},
			{"oversized_count", "oversized", false, nil, nil},
		} {
			t.Run(tc.name, func(t *testing.T) {
				src := newV41EngramBatchSource(8, V41EngramPackedRowBytes, 0)
				src.short, src.err, src.countMode = tc.short, tc.err, tc.countMode
				cache := v41EngramBatchCache(t, src, 4*src.rowBytes)
				ids := []uint32{2, 3}
				got, err := GatherV41EngramRows([]*V41EngramRowCache{cache}, ids, 1)
				if err == nil || (tc.want != nil && !errors.Is(err, tc.want)) {
					t.Fatalf("source failure=%v, want errors.Is(%v)", err, tc.want)
				}
				if got != nil {
					t.Fatal("failed gather returned usable output")
				}
				if len(src.reads) != 1 {
					t.Fatalf("source failure ReadRows calls=%d, want 1", len(src.reads))
				}
				if stats := cache.inner.Stats(); stats.ResidentBytes != 0 || stats.ResidentEntries != 0 {
					t.Fatalf("failed read cached bytes=%d entries=%d", stats.ResidentBytes, stats.ResidentEntries)
				}
				src.short, src.err, src.countMode = false, nil, ""
				got, err = GatherV41EngramRows([]*V41EngramRowCache{cache}, ids, 1)
				if err != nil {
					t.Fatalf("retry after source failure: %v", err)
				}
				v41EngramBatchCheckRows(t, got, ids, []*v41EngramBatchSource{src}, 1)
				if len(src.reads) != 2 || src.reads[1].start != 2 || src.reads[1].count != 2 {
					t.Fatal("retry did not perform a fresh complete adjacent read")
				}
				v41EngramBatchCheckResidency(t, cache, 4*src.rowBytes)
			})
		}
	})
}

type v41EngramBatchRead struct {
	start, count, bytes int
}

type v41EngramBatchSource struct {
	rows, rowBytes int
	data           []byte
	reads          []v41EngramBatchRead
	short          bool
	err            error
	countMode      string
}

func newV41EngramBatchSource(rows, rowBytes, bias int) *v41EngramBatchSource {
	src := &v41EngramBatchSource{rows: rows, rowBytes: rowBytes, data: make([]byte, rows*rowBytes)}
	for row := range rows {
		for col := range rowBytes {
			src.data[row*rowBytes+col] = byte(bias + row*31 + col)
		}
	}
	return src
}

func (s *v41EngramBatchSource) RowBytes() int { return s.rowBytes }

func (s *v41EngramBatchSource) ReadRows(start, count int, dst []byte) (int, error) {
	s.reads = append(s.reads, v41EngramBatchRead{start, count, len(dst)})
	if s.err != nil {
		return 0, s.err
	}
	if start < 0 || count <= 0 || count > s.rows || start > s.rows-count {
		return 0, io.EOF
	}
	n := copy(dst, s.data[start*s.rowBytes:(start+count)*s.rowBytes])
	if s.countMode == "negative" {
		return -1, nil
	}
	if s.countMode == "oversized" {
		return len(dst) + 1, nil
	}
	if s.short && n > 0 {
		n--
	}
	return n, nil
}

func v41EngramBatchCache(t *testing.T, src *v41EngramBatchSource, budget int) *V41EngramRowCache {
	t.Helper()
	cache, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{
		TableRows: src.rows, RowBytes: src.rowBytes, BudgetBytes: int64(budget),
	})
	if err != nil {
		t.Fatal(err)
	}
	return cache
}

func v41EngramBatchIDs(rows int) []uint32 {
	ids := make([]uint32, rows)
	for i := range ids {
		ids[i] = uint32(i)
	}
	return ids
}

func v41EngramBatchCheckRows(t *testing.T, got [][]byte, ids []uint32, sources []*v41EngramBatchSource, columns int) {
	t.Helper()
	if len(got) != len(ids) {
		t.Fatalf("gathered rows=%d, want %d", len(got), len(ids))
	}
	for i, id := range ids {
		src := sources[(i/columns)%len(sources)]
		start := int(id) * src.rowBytes
		if !bytes.Equal(got[i], src.data[start:start+src.rowBytes]) {
			t.Fatalf("gathered output position %d differs from layer/row %d/%d", i, (i/columns)%len(sources), id)
		}
	}
}

func v41EngramBatchCheckReads(t *testing.T, src *v41EngramBatchSource, ids []uint32, budget int) {
	t.Helper()
	v41EngramBatchCheckReadBounds(t, src, budget)
	want := make(map[int]bool, len(ids))
	for _, id := range ids {
		want[int(id)] = true
	}
	seen := make(map[int]bool, len(want))
	for _, read := range src.reads {
		for row := read.start; row < read.start+read.count; row++ {
			if !want[row] || seen[row] {
				t.Fatalf("source read row %d was unrequested or read more than once", row)
			}
			seen[row] = true
		}
	}
	if len(seen) != len(want) {
		t.Fatalf("source read %d distinct rows, want %d", len(seen), len(want))
	}
}

func v41EngramBatchCheckReadBounds(t *testing.T, src *v41EngramBatchSource, budget int) {
	t.Helper()
	for _, read := range src.reads {
		if read.count <= 0 || read.start < 0 || read.start > src.rows-read.count {
			t.Fatalf("source read range start=%d count=%d exceeds %d rows", read.start, read.count, src.rows)
		}
		if read.bytes != read.count*src.rowBytes || read.bytes > budget || (read.bytes > 64*1024 && (src.rowBytes <= 64*1024 || read.count != 1)) {
			t.Fatalf("read bytes=%d count=%d exceeds row geometry, budget=%d, or 64KiB", read.bytes, read.count, budget)
		}
	}
}

func v41EngramBatchCheckResidency(t *testing.T, cache *V41EngramRowCache, budget int) {
	t.Helper()
	stats := cache.inner.Stats()
	if stats.ResidentBytes > int64(budget) || stats.ResidentEntries > budget/cache.opts.RowBytes {
		t.Fatalf("resident bytes=%d entries=%d exceed budget=%d rowBytes=%d", stats.ResidentBytes, stats.ResidentEntries, budget, cache.opts.RowBytes)
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=1ms lane=default
func TestV41EngramBatchDemandOrderSurvivors(t *testing.T) {
	t.Parallel()
	ids := []uint32{5, 2, 3, 2, 6, 5}
	batchSource := newV41EngramBatchSource(8, V41EngramPackedRowBytes, 0)
	serialSource := newV41EngramBatchSource(8, V41EngramPackedRowBytes, 0)
	budget := 2 * V41EngramPackedRowBytes
	batch := v41EngramBatchCache(t, batchSource, budget)
	serial := v41EngramBatchCache(t, serialSource, budget)
	got, err := GatherV41EngramRows([]*V41EngramRowCache{batch}, ids, 1)
	if err != nil {
		t.Fatal(err)
	}
	v41EngramBatchCheckRows(t, got, ids, []*v41EngramBatchSource{batchSource}, 1)
	for _, id := range ids {
		if _, err := serial.Row(int(id)); err != nil {
			t.Fatal(err)
		}
	}
	for row := 0; row < batchSource.rows; row++ {
		_, gotResident := batch.inner.Get(0, row)
		_, wantResident := serial.inner.Get(0, row)
		if gotResident != wantResident {
			t.Fatalf("row %d batch residency=%t differs from serial demanded-order residency=%t", row, gotResident, wantResident)
		}
	}
	v41EngramBatchCheckResidency(t, batch, budget)
}

// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41EngramBatchAttachedSession(t *testing.T) {
	t.Parallel()
	m, _ := v41IncrementalEngramFixture(t, false)
	oracle, _ := v41IncrementalEngramFixture(t, false)
	stage := m.v41EngramStageFor()
	observed := make([]*v41EngramBatchObservedSource, len(stage.caches))
	for i, cache := range stage.caches {
		observed[i] = &v41EngramBatchObservedSource{src: cache.src}
		cache.src = observed[i]
	}
	session := m.NewSession()
	t.Cleanup(session.Close)
	var history []int
	for index, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
		before := make([]int, len(observed))
		for layer, src := range observed {
			before[layer] = len(src.reads)
			src.gather, src.prepared = false, false
		}
		var got []float32
		if index == 1 {
			got = session.Step(ids[0])
		} else {
			got = session.Prefill(ids)
		}
		history = append(history, ids...)
		want := v41IncrementalEngramCold(t, oracle, history)
		for _, result := range []struct {
			name   string
			logits []float32
			vocab  int
		}{{"attached", got, m.Cfg.VocabSize}, {"cold", want, oracle.Cfg.VocabSize}} {
			if result.vocab <= 0 || len(result.logits) != result.vocab {
				t.Fatalf("%s session operation %d logits length=%d, configured vocabulary=%d", result.name, index, len(result.logits), result.vocab)
			}
			for token, value := range result.logits {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatalf("%s session operation %d token %d logit is not finite", result.name, index, token)
				}
			}
		}
		assertV41RowsClose(t, "attached batch session", got, want, 1e-6)
		for layer, src := range observed {
			if len(src.reads) <= before[layer] {
				t.Fatalf("session operation %d did not read layer %d source", index, layer)
			}
			if !src.gather || !src.prepared {
				t.Fatalf("session operation %d source did not observe prepared injection through gather", index)
			}
			for _, read := range src.reads[before[layer]:] {
				cache := stage.caches[layer]
				if read.bytes > int(cache.opts.BudgetBytes) || read.bytes > 64*1024 || read.bytes != read.count*src.RowBytes() {
					t.Fatalf("attached layer %d read start=%d count=%d bytes=%d exceeds bounded geometry", layer, read.start, read.count, read.bytes)
				}
			}
		}
	}
}

type v41EngramBatchObservedSource struct {
	src              V41EngramRowSource
	reads            []v41EngramBatchRead
	gather, prepared bool
}

func (s *v41EngramBatchObservedSource) RowBytes() int { return s.src.RowBytes() }

func (s *v41EngramBatchObservedSource) ReadRows(start, count int, dst []byte) (int, error) {
	s.reads = append(s.reads, v41EngramBatchRead{start, count, len(dst)})
	var pcs [32]uintptr
	frames := runtime.CallersFrames(pcs[:runtime.Callers(2, pcs[:])])
	for {
		frame, more := frames.Next()
		s.gather = s.gather || strings.HasSuffix(frame.Function, ".GatherV41EngramRows")
		s.prepared = s.prepared || strings.HasSuffix(frame.Function, ".v41EngramInjectPrepared")
		if !more {
			break
		}
	}
	return s.src.ReadRows(start, count, dst)
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41EngramSharedCacheConcurrency(t *testing.T) {
	t.Parallel()
	t.Run("row_gather_stats", func(t *testing.T) {
		const rows = 32
		packed := streamTestPackedRows(rows)
		src := &v41EngramBatchSafeSource{src: &v41EngramMemorySource{packed: packed, rows: rows}}
		budget := int64(4 * V41EngramPackedRowBytes)
		cache, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{TableRows: rows, RowBytes: src.RowBytes(), BudgetBytes: budget})
		if err != nil {
			t.Fatal(err)
		}
		rowWorker := func(offset int) func() error {
			return func() error {
				for i := range 64 {
					row := (i + offset) % rows
					got, err := cache.Row(row)
					if err != nil {
						return err
					}
					if !bytes.Equal(got, packed[row*src.RowBytes():(row+1)*src.RowBytes()]) {
						return fmt.Errorf("concurrent Row(%d) returned wrong bytes", row)
					}
				}
				return nil
			}
		}
		v41EngramBatchRunConcurrent(t, rowWorker(0), rowWorker(13), func() error {
			for i := range 64 {
				row := uint32(i % (rows - 2))
				ids := []uint32{row + 2, row, row + 1, row}
				got, err := GatherV41EngramRows([]*V41EngramRowCache{cache}, ids, 1)
				if err != nil {
					return err
				}
				if len(got) != len(ids) {
					return fmt.Errorf("concurrent gather returned %d rows, want %d", len(got), len(ids))
				}
				for j, id := range ids {
					start := int(id) * src.RowBytes()
					if !bytes.Equal(got[j], packed[start:start+src.RowBytes()]) {
						return fmt.Errorf("concurrent gather output %d has wrong bytes", j)
					}
				}
			}
			return nil
		}, func() error {
			for range 64 {
				stats := cache.Stats()
				if stats.RowsServed < 0 || stats.Hits < 0 || stats.Misses < 0 || stats.BytesRead < 0 {
					return errors.New("concurrent cache Stats returned a negative counter")
				}
			}
			return nil
		})
		if src.reads.Load() == 0 {
			t.Fatal("concurrent cache operations did not read the source")
		}
		v41EngramBatchCheckResidency(t, cache, int(budget))
	})

	t.Run("model_warm_and_session", func(t *testing.T) {
		m, _ := v41IncrementalEngramFixture(t, false)
		oracle, _ := v41IncrementalEngramFixture(t, false)
		stage := m.v41EngramStageFor()
		sources := make([]*v41EngramBatchSafeSource, len(stage.caches))
		for i, cache := range stage.caches {
			sources[i] = &v41EngramBatchSafeSource{src: cache.src}
			cache.src = sources[i]
		}
		session := m.NewSession()
		t.Cleanup(session.Close)
		turns := [][]int{{1, 3, 2}, {4}, {5}, {6}}
		logits := make([][]float32, len(turns))
		v41EngramBatchRunConcurrent(t, func() error {
			for i, ids := range turns {
				if i == 0 {
					logits[i] = session.Prefill(ids)
				} else {
					logits[i] = session.Step(ids[0])
				}
			}
			return nil
		}, func() error {
			for i := range 64 {
				if _, err := m.WarmV41EngramPrefix([]int{1, 3, 2, 4 + i%4}, nil); err != nil {
					return err
				}
			}
			return nil
		})
		var history []int
		for i, ids := range turns {
			history = append(history, ids...)
			assertV41RowsClose(t, "concurrently warmed session", logits[i], v41IncrementalEngramCold(t, oracle, history), 1e-6)
		}
		for layer, src := range sources {
			if src.reads.Load() == 0 {
				t.Fatalf("concurrent warm/session did not read layer %d source", layer)
			}
			v41EngramBatchCheckResidency(t, stage.caches[layer], int(stage.caches[layer].opts.BudgetBytes))
		}
	})
}

type v41EngramBatchSafeSource struct {
	src   V41EngramRowSource
	reads atomic.Int64
}

func (s *v41EngramBatchSafeSource) RowBytes() int { return s.src.RowBytes() }

func (s *v41EngramBatchSafeSource) ReadRows(start, count int, dst []byte) (int, error) {
	s.reads.Add(1)
	return s.src.ReadRows(start, count, dst)
}

func v41EngramBatchRunConcurrent(t *testing.T, workers ...func() error) {
	t.Helper()
	start := make(chan struct{})
	errs := make(chan error, len(workers))
	for _, worker := range workers {
		go func() {
			<-start
			err := func() (err error) {
				defer func() {
					if recovered := recover(); recovered != nil {
						err = fmt.Errorf("concurrent cache operation panicked: %v", recovered)
					}
				}()
				return worker()
			}()
			errs <- err
		}()
	}
	close(start)
	for range workers {
		if err := <-errs; err != nil {
			t.Error(err)
		}
	}
}

// All APIs exercised here exist on the parent; the parent must fail the
// same-storage assertion at runtime, rather than fail to compile the witness.
// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=100ms lane=default
func TestV41EngramOwnedPanel(t *testing.T) {
	for _, rowBytes := range []int{v41EngramQ2KEncodedRowBytes, V41EngramPackedRowBytes, V41EngramF32RowBytes} {
		t.Run(fmt.Sprint(rowBytes), func(t *testing.T) {
			src := newV41EngramBatchSource(8, rowBytes, 0)
			cache := v41EngramBatchCache(t, src, 2*rowBytes)
			stage := &v41EngramStage{caches: []*V41EngramRowCache{cache}, rowBytes: rowBytes}
			geometry := v41EngramGeometry{stage: stage, cols: 1, dim: 256, qNorm: []float32{1}, kNorm: []float32{1}}
			var m Model
			ids := []uint32{2, 0, 2}
			gathered, err := m.v41EngramGatherRows(1, len(ids), ids, 1e-6, geometry)
			if err != nil {
				t.Fatal(err)
			}
			v41EngramBatchCheckRows(t, gathered.rows, ids, []*v41EngramBatchSource{src}, 1)
			panel, ok := gathered.stored()
			if !ok || len(panel) != len(ids)*rowBytes || &panel[0] != &gathered.rows[0][0] {
				t.Fatal("resident stored rows made a second encoded-panel copy")
			}
			for i, row := range gathered.rows {
				if cap(row) != len(row) || &row[0] != &panel[i*rowBytes] {
					t.Fatalf("row %d is not a width-capped view of its own panel slot", i)
				}
			}
			if allocs := testing.AllocsPerRun(16, func() {
				encoded, valid := gathered.stored()
				if !valid || len(encoded) != len(ids)*rowBytes {
					panic("invalid resident panel")
				}
				runtime.KeepAlive(encoded)
			}); allocs != 0 {
				t.Fatalf("stored resident panel allocations=%g, want zero", allocs)
			}
			before := append([]byte(nil), panel...)
			grown := append(gathered.rows[0], 99)
			grown[0] ^= 0xff
			if !bytes.Equal(panel, before) {
				t.Fatal("appending to one row mutated the owned panel")
			}
			gathered.rows[0][0] ^= 0xff
			resident, exists := cache.inner.Get(0, 2)
			if !exists || resident[0] != before[0] || gathered.rows[2][0] != before[2*rowBytes] {
				t.Fatal("duplicate row slots or cached payload alias one another")
			}
			gathered.rows[0][0] ^= 0xff
			cache.invalidateRows()
			if _, err := m.v41EngramGatherRows(1, 3, []uint32{7, 6, 5}, 1e-6, geometry); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(panel, before) {
				t.Fatal("a later gather or eviction overwrote retained encoded rows")
			}
			// A reordered private fixture must concatenate in its new row order.
			reordered := *gathered
			reordered.rows = append([][]byte(nil), gathered.rows...)
			reordered.rows[0], reordered.rows[1] = reordered.rows[1], reordered.rows[0]
			fallback, ok := reordered.stored()
			if !ok || !bytes.Equal(fallback, bytes.Join(reordered.rows, nil)) || &fallback[0] == &reordered.rows[0][0] {
				t.Fatal("noncontiguous fixture did not preserve concatenation semantics")
			}
			publicRows, err := GatherV41EngramRows([]*V41EngramRowCache{cache}, ids, 1)
			if err != nil {
				t.Fatal(err)
			}
			legacy := &v41EngramGatheredRows{stage: stage, rows: publicRows, dim: 256}
			fallback, ok = legacy.stored()
			if !ok || !bytes.Equal(fallback, before) || &fallback[0] == &publicRows[0][0] {
				t.Fatal("public gather fixture lost its independent concatenated payload")
			}
			empty, err := m.v41EngramGatherRows(1, 0, nil, 1e-6, geometry)
			if err != nil {
				t.Fatal(err)
			}
			if encoded, valid := empty.stored(); !valid || encoded == nil || len(encoded) != 0 {
				t.Fatal("empty gather lost its valid empty stored payload")
			}
			reads := len(src.reads)
			for _, invalid := range []struct{ positions, cols int }{{-1, 1}, {0, 0}, {2, math.MaxInt}} {
				bad := geometry
				bad.cols = invalid.cols
				if _, err := m.v41EngramGatherRows(1, invalid.positions, nil, 1e-6, bad); err == nil {
					t.Fatal("invalid or overflowing panel geometry was accepted")
				}
			}
			if _, err := m.v41EngramGatherRows(1, 1, []uint32{8}, 1e-6, geometry); err == nil {
				t.Fatal("out-of-range resident row was accepted")
			}
			if len(src.reads) != reads {
				t.Fatal("invalid resident panel geometry issued source I/O")
			}
		})
	}
}
