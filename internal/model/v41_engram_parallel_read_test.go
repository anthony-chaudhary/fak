package model

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"reflect"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type v41ParallelReadRecord struct {
	Offset int64
	Bytes  int
}

type v41ParallelReadProbe struct {
	data                                       *bytes.Reader
	observe, eligible, accept                  bool // fixed before the first observed read
	countOnly                                  bool // benchmark records only a bounded counter, not an unbounded trace
	readCalls                                  atomic.Int64
	staging                                    int
	closing                                    chan struct{}
	entered                                    chan int
	blocked                                    <-chan struct{}
	delay                                      time.Duration
	faults                                     map[int]error
	panicRow, exitRow                          int
	mu                                         sync.Mutex
	reads                                      []v41ParallelReadRecord
	active, activeBytes, maximum, maximumBytes atomic.Int64
	holds, releases                            atomic.Int64
	releasedEarly                              atomic.Bool
}

func (r *v41ParallelReadProbe) V41EngramLookAheadReadPolicy(width int) (int, bool) {
	return r.staging, r.eligible && width == V41EngramPackedRowBytes
}
func (r *v41ParallelReadProbe) V41EngramHoldSourceReads() (func(), <-chan struct{}, bool) {
	r.holds.Add(1)
	if !r.accept {
		return nil, nil, false
	}
	var once sync.Once
	return func() {
		once.Do(func() {
			if r.active.Load() != 0 {
				r.releasedEarly.Store(true)
			}
			r.releases.Add(1)
		})
	}, r.closing, true
}
func v41ParallelMaximum(dst *atomic.Int64, n int64) {
	for old := dst.Load(); n > old; old = dst.Load() {
		if dst.CompareAndSwap(old, n) {
			return
		}
	}
}
func (r *v41ParallelReadProbe) ReadAt(dst []byte, off int64) (int, error) {
	if !r.observe {
		return r.data.ReadAt(dst, off)
	}
	v41ParallelMaximum(&r.maximum, r.active.Add(1))
	v41ParallelMaximum(&r.maximumBytes, r.activeBytes.Add(int64(len(dst))))
	defer r.active.Add(-1)
	defer r.activeBytes.Add(-int64(len(dst)))
	r.readCalls.Add(1)
	if !r.countOnly {
		r.mu.Lock()
		r.reads = append(r.reads, v41ParallelReadRecord{off, len(dst)})
		r.mu.Unlock()
	}
	row := int(off / V41EngramPackedRowBytes)
	if r.entered != nil {
		r.entered <- row
	}
	if r.blocked != nil {
		<-r.blocked
	}
	if r.delay != 0 {
		time.Sleep(r.delay)
	}
	if row == r.panicRow {
		panic("parallel-read-panic")
	}
	if row == r.exitRow {
		runtime.Goexit()
	}
	if err := r.faults[row]; err != nil {
		return 0, err
	}
	return r.data.ReadAt(dst, off)
}
func (r *v41ParallelReadProbe) records() []v41ParallelReadRecord {
	r.mu.Lock()
	defer r.mu.Unlock()
	x := append([]v41ParallelReadRecord(nil), r.reads...)
	sort.Slice(x, func(i, j int) bool { return x[i].Offset < x[j].Offset })
	return x
}
func v41ParallelReadSource(tb testing.TB, rows int) (V41EngramRowSource, *v41ParallelReadProbe, []byte) {
	tb.Helper()
	data := v41EngramTestPackedRows(rows)
	r := &v41ParallelReadProbe{data: bytes.NewReader(data), eligible: true, accept: true, closing: make(chan struct{}), panicRow: -1, exitRow: -1}
	shard := V41EngramShard{ID: "parallel-read-fixture", ExpectedID: "parallel-read-fixture", Rows: rows, Size: int64(len(data))}
	binding := V41EngramArtifactBinding{Shard: shard.ID, Rows: rows, Size: shard.Size, RowBytes: V41EngramPackedRowBytes, Digest: v41EngramTestDigest(data)}
	source, err := NewV41VerifiedPackedEngramRowSource(r, shard, binding)
	if err != nil {
		tb.Fatal(err)
	}
	r.observe = true // binding acquisition is not counted as demand I/O
	return source, r, data
}
func v41ParallelReadCache(tb testing.TB, workers, capacity int) (*V41EngramRowCache, *v41ParallelReadProbe, []byte) {
	tb.Helper()
	source, r, data := v41ParallelReadSource(tb, 64)
	c, err := NewV41EngramRowCache(source, V41EngramRowCacheOptions{TableRows: 64, RowBytes: V41EngramPackedRowBytes, BudgetBytes: int64(capacity * V41EngramPackedRowBytes), ReadWorkers: workers})
	if err != nil {
		tb.Fatal(err)
	}
	return c, r, data
}

type v41ParallelReadResult struct {
	rows       [][]byte
	err        error
	panicValue any
}

func v41ParallelReadStart(c *V41EngramRowCache, ids []uint32) <-chan v41ParallelReadResult {
	done := make(chan v41ParallelReadResult, 1)
	go func() {
		var result v41ParallelReadResult
		defer func() { result.panicValue = recover(); done <- result }()
		result.rows, result.err = c.gatherKnownRows(ids)
	}()
	return done
}
func v41ParallelReadWait(t *testing.T, done <-chan v41ParallelReadResult) v41ParallelReadResult {
	t.Helper()
	select {
	case result := <-done:
		return result
	case <-time.After(3 * time.Second):
		t.Fatal("parallel gather did not join")
		return v41ParallelReadResult{}
	}
}
func v41ParallelReadEntered(t *testing.T, r *v41ParallelReadProbe, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		select {
		case <-r.entered:
		case <-time.After(3 * time.Second):
			t.Fatal("required concurrent reads did not start")
		}
	}
}
func v41ParallelReadBarrier(t *testing.T, r *v41ParallelReadProbe) func() {
	t.Helper()
	barrier := make(chan struct{})
	var once sync.Once
	r.blocked = barrier
	r.entered = make(chan int, 64)
	unblock := func() { once.Do(func() { close(barrier) }) }
	t.Cleanup(unblock)
	return unblock
}

// Actual source/gather/attachment calls, not an isolated scheduler witness.
// CPU-only contract source; no device, SSD or throughput result is implied.
// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=5s lane=default
func TestV41EngramParallelReads(t *testing.T) {
	// Do not t.Parallel: the production-wide additional-read credit is shared.
	t.Run("same spans bytes order and immutable output", func(t *testing.T) {
		ids := []uint32{7, 1, 3, 5, 7, 9, 11}
		serial, sr, _ := v41ParallelReadCache(t, 0, 4)
		want, err := serial.gatherKnownRows(ids)
		if err != nil {
			t.Fatal(err)
		}
		parallel, pr, data := v41ParallelReadCache(t, 4, 4)
		unblock := v41ParallelReadBarrier(t, pr)
		done := v41ParallelReadStart(parallel, ids)
		v41ParallelReadEntered(t, pr, 4)
		if pr.holds.Load() != 1 || pr.releases.Load() != 0 {
			t.Fatal("reads are not held until joined")
		}
		if pr.maximum.Load() != 4 || pr.maximumBytes.Load() > int64(4*V41EngramPackedRowBytes) {
			t.Fatal("worker/wave budget violated")
		}
		unblock()
		got := v41ParallelReadWait(t, done)
		if got.err != nil || got.panicValue != nil || !reflect.DeepEqual(got.rows, want) {
			t.Fatalf("parallel gather differs: %v %v", got.err, got.panicValue)
		}
		if !reflect.DeepEqual(sr.records(), pr.records()) || serial.Stats() != parallel.Stats() || !reflect.DeepEqual(serial.inner.order, parallel.inner.order) {
			t.Fatal("successful spans/stats/demand-order residency changed")
		}
		if pr.active.Load() != 0 || pr.releases.Load() != 1 || pr.releasedEarly.Load() {
			t.Fatal("source released before all reads joined")
		}
		got.rows[0][0] ^= 0xff
		if !bytes.Equal(got.rows[4], data[7*V41EngramPackedRowBytes:8*V41EngramPackedRowBytes]) {
			t.Fatal("duplicate output rows alias")
		}
		before := append([]byte(nil), got.rows[4]...)
		if _, err := parallel.gatherKnownRows([]uint32{13, 15, 17, 19}); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(before, got.rows[4]) {
			t.Fatal("later wave/cache reuse changed retained borrower")
		}
	})
	t.Run("serial defaults eligibility and staging fallback", func(t *testing.T) {
		for _, mode := range []string{"default", "one", "ineligible", "staging", "one-span", "one-row-budget", "hit"} {
			t.Run(mode, func(t *testing.T) {
				workers, capacity := 4, 4
				if mode == "default" {
					workers = 0
				}
				if mode == "one" {
					workers = 1
				}
				if mode == "one-row-budget" {
					capacity = 1
				}
				c, r, _ := v41ParallelReadCache(t, workers, capacity)
				ids := []uint32{1, 3, 5, 7}
				if mode == "ineligible" {
					r.eligible = false
				}
				if mode == "staging" {
					r.staging = 1 << 20
				}
				if mode == "one-span" {
					ids = []uint32{1, 2, 3}
				}
				if mode == "hit" {
					for _, id := range ids {
						if _, err := c.Row(int(id)); err != nil {
							t.Fatal(err)
						}
					}
				}
				if _, err := c.gatherKnownRows(ids); err != nil {
					t.Fatal(err)
				}
				if r.holds.Load() != 0 || r.maximum.Load() > 1 {
					t.Fatal("ordinary or ineligible source selected concurrent I/O")
				}
			})
		}
	})
	t.Run("custom unowned reader stays serial", func(t *testing.T) {
		data := v41EngramTestPackedRows(64)
		shard := V41EngramShard{ID: "unowned", ExpectedID: "unowned", Rows: 64, Size: int64(len(data))}
		binding := V41EngramArtifactBinding{Shard: shard.ID, Rows: 64, Size: shard.Size, RowBytes: V41EngramPackedRowBytes, Digest: v41EngramTestDigest(data)}
		src, err := NewV41VerifiedPackedEngramRowSource(bytes.NewReader(data), shard, binding)
		if err != nil {
			t.Fatal(err)
		}
		cache, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{TableRows: 64, RowBytes: V41EngramPackedRowBytes, BudgetBytes: 4 * V41EngramPackedRowBytes, ReadWorkers: 4})
		if err != nil {
			t.Fatal(err)
		}
		ids := []uint32{1, 3, 5, 7}
		out, err := cache.gatherKnownRows(ids)
		if err != nil {
			t.Fatal(err)
		}
		for i, row := range ids {
			if !bytes.Equal(out[i], data[int(row)*V41EngramPackedRowBytes:(int(row)+1)*V41EngramPackedRowBytes]) {
				t.Fatal("unowned serial fallback changed bytes")
			}
		}
	})
	t.Run("failed hold and invalid constructor are premutation", func(t *testing.T) {
		c, r, _ := v41ParallelReadCache(t, 4, 4)
		r.accept = false
		if out, err := c.gatherKnownRows([]uint32{1, 3}); err == nil || out != nil || len(r.records()) != 0 || len(c.inner.entries) != 0 {
			t.Fatal("refused hold issued I/O or published rows")
		}
		for _, workers := range []int{-1, 5} {
			m := &Model{}
			if err := m.AttachV41Engram(V41EngramAttachSpec{ReadWorkers: workers}, V41EngramLayout{}, nil, 1024); !errors.Is(err, ErrV41NativeUnsupported) || m.Cfg.DeepSeekV41 != nil || m.V41EngramAttached() {
				t.Fatal("invalid workers mutated model")
			}
		}
	})
	t.Run("closing error panic and goexit join before release", func(t *testing.T) {
		for _, mode := range []string{"closing", "ordered-error", "later-wave-error", "panic", "goexit"} {
			t.Run(mode, func(t *testing.T) {
				c, r, _ := v41ParallelReadCache(t, 4, 4)
				unblock := v41ParallelReadBarrier(t, r)
				first, later := errors.New("first sorted failure"), errors.New("later sorted failure")
				switch mode {
				case "ordered-error":
					r.faults = map[int]error{1: first, 3: later}
				case "later-wave-error":
					r.faults = map[int]error{9: first}
				case "panic":
					r.panicRow = 1
				case "goexit":
					r.exitRow = 1
				}
				done := v41ParallelReadStart(c, []uint32{1, 3, 5, 7, 9, 11})
				v41ParallelReadEntered(t, r, 4)
				if mode == "closing" {
					close(r.closing)
				}
				select {
				case <-done:
					t.Fatal("gather returned before held reads joined")
				default:
				}
				unblock()
				result := v41ParallelReadWait(t, done)
				if mode == "panic" {
					if result.panicValue != "parallel-read-panic" {
						t.Fatalf("panic not propagated: %v", result.panicValue)
					}
				} else if result.err == nil || result.panicValue != nil {
					t.Fatalf("missing ordered refusal: %v %v", result.err, result.panicValue)
				}
				if mode == "ordered-error" || mode == "later-wave-error" {
					if !errors.Is(result.err, first) {
						t.Fatalf("wrong first failure: %v", result.err)
					}
				}
				if result.rows != nil || len(c.inner.entries) != 0 || r.active.Load() != 0 || r.releases.Load() != 1 || r.releasedEarly.Load() {
					t.Fatal("partial publication or unjoined owner release")
				}
				if mode != "later-wave-error" && len(r.records()) != 4 {
					t.Fatal("started a new wave after failure/close")
				}
			})
		}
	})
	t.Run("occupied group falls back without borrowing other owner", func(t *testing.T) {
		c, r, _ := v41ParallelReadCache(t, 4, 4)
		unblock := v41ParallelReadBarrier(t, r)
		done := v41ParallelReadStart(c, []uint32{1, 3, 5, 7})
		v41ParallelReadEntered(t, r, 4)
		other, probe, _ := v41ParallelReadCache(t, 4, 4)
		if _, err := other.gatherKnownRows([]uint32{9, 11}); err != nil {
			t.Fatal(err)
		}
		if probe.holds.Load() != 0 || probe.maximum.Load() != 1 || r.releases.Load() != 0 {
			t.Fatal("occupied group did not preserve serial fallback and other source hold")
		}
		unblock()
		result := v41ParallelReadWait(t, done)
		if result.err != nil {
			t.Fatal(result.err)
		}
	})
	t.Run("typed attachment reaches native forward", func(t *testing.T) {
		var logits [][][]float32
		for _, workers := range []int{0, 4} {
			m, layout := v41ReducedEngramModel(t)
			t.Cleanup(func() { m.CloseWeights() })
			src, reader, _ := v41ParallelReadSource(t, int(layout.Rows[0]))
			d := m.Cfg.DeepSeekV41
			spec := V41EngramAttachSpec{LayerIDs: d.EngramLayerIDs, MaxNgramSize: d.EngramMaxNgramSize, VocabSize: d.EngramVocabSize, NHeads: d.EngramNHeads, HeadDim: d.EngramHeadDim, PadTokenID: d.EngramPadTokenID, CompressedVocabSize: d.EngramCompressedVocabSize, ReadWorkers: workers}
			if err := m.AttachV41Engram(spec, layout, []V41EngramRowSource{src}, 4*V41EngramPackedRowBytes); err != nil {
				t.Fatal(err)
			}
			cache := m.v41EngramStageFor().caches[0]
			if cache.opts.ReadWorkers != workers {
				t.Fatal("typed selection was not wired")
			}
			got := m.Forward([]int{1, 3, 5})
			if got == nil || cache.Stats().RowsServed == 0 {
				t.Fatal("native Forward did not gather")
			}
			if workers == 4 && (reader.holds.Load() == 0 || reader.holds.Load() != reader.releases.Load()) {
				t.Fatal("actual Forward missed selected joined-read arm")
			}
			if workers == 0 && reader.holds.Load() != 0 {
				t.Fatal("default Forward changed to parallel")
			}
			logits = append(logits, got.Logits)
		}
		if !reflect.DeepEqual(logits[0], logits[1]) {
			t.Fatal("parallel reads changed native reduced-model logits")
		}
	})
}

// Identical cold miss trace, actual gather and fixed synthetic ReaderAt delays.
// Includes scheduler/copy/cache overhead; NOT an SSD/device/whole-model result.
// fak-test:runtime medium est=5s lane=optin
func BenchmarkV41EngramParallelReads(b *testing.B) {
	for _, capacity := range []int{4, 16} {
		for _, delay := range []time.Duration{0, 100 * time.Microsecond} {
			for _, workers := range []int{0, 4} {
				b.Run(fmt.Sprintf("rows=%d/delay=%s/workers=%d", capacity, delay, workers), func(b *testing.B) {
					c, r, _ := v41ParallelReadCache(b, workers, capacity)
					r.delay = delay
					r.countOnly = true
					ids := make([]uint32, 24)
					for i := range ids {
						ids[i] = uint32(2*i + 1)
					}
					b.SetBytes(int64(len(ids) * V41EngramPackedRowBytes))
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						c.invalidateRows()
						if _, err := c.gatherKnownRows(ids); err != nil {
							b.Fatal(err)
						}
					}
					b.StopTimer()
					st := c.Stats()
					b.ReportMetric(float64(st.BytesRead)/float64(b.N), "read-B/op")
					b.ReportMetric(float64(st.Hits)/float64(b.N), "hit/op")
					b.ReportMetric(float64(r.readCalls.Load())/float64(b.N), "reads/op")
					b.ReportMetric(float64(r.maximum.Load()), "max-active")
				})
			}
		}
	}
}

var _ io.ReaderAt = (*v41ParallelReadProbe)(nil)
