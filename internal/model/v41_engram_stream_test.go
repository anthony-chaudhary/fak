package model

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"
)

func TestV41EngramStreamPackedParityAndAtomicRetry(t *testing.T) {
	layout := streamTestLayout()
	packed := streamTestPackedRows(int(layout.Rows[0]))
	path := filepath.Join(t.TempDir(), "engram.packed264")
	if err := os.WriteFile(path, packed, 0o600); err != nil {
		t.Fatal(err)
	}

	tokens := []int{1, 3, 2, 0, 3, 1}
	mask := []bool{true, true, false, true, true, true}
	wantIDs := v41EngramReference(layout, tokens, mask)
	wantRows := streamTestRows(packed, wantIDs)
	for _, chunk := range []int{1, 2, len(tokens)} {
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = f.Close() })
		src, err := NewV41PackedEngramRowSource(f, V41EngramShard{
			ID: "layer-1", ExpectedID: "layer-1", Size: int64(len(packed)), Rows: int(layout.Rows[0]),
		})
		if err != nil {
			t.Fatal(err)
		}
		caches := make([]*V41EngramSetCache, len(layout.Rows))
		for i := range caches {
			caches[i], err = NewV41EngramSetCache(src, V41EngramSetCacheOptions{
				TableRows: int(layout.Rows[i]), Sets: 2, Ways: 2,
			})
			if err != nil {
				t.Fatal(err)
			}
		}
		hash, err := NewV41EngramHashState(layout)
		if err != nil {
			t.Fatal(err)
		}
		stream, err := NewV41EngramStream(hash, caches)
		if err != nil {
			t.Fatal(err)
		}

		var got [][]byte
		for start := 0; start < len(tokens); start += chunk {
			end := min(start+chunk, len(tokens))
			if err := stream.Prefetch(tokens[start:end], mask[start:end]); err != nil {
				t.Fatalf("chunk=%d prefetch: %v", chunk, err)
			}
			part, err := stream.Consume(tokens[start:end], mask[start:end])
			if err != nil {
				t.Fatalf("chunk=%d consume: %v", chunk, err)
			}
			got = append(got, part...)
		}
		if !reflect.DeepEqual(got, wantRows) {
			t.Fatalf("chunk=%d streamed rows differ from in-memory hash oracle", chunk)
		}
		cache := caches[0]
		stats := cache.Stats()
		if stats.Prefetches == 0 || stats.Reads == 0 || stats.BytesRead == 0 {
			t.Fatalf("chunk=%d incomplete accounting: %+v", chunk, stats)
		}

		// A returned row must not alias the cache entry.
		var first [V41EngramPackedRowBytes]byte
		if _, err := cache.ReadRows(0, 1, first[:]); err != nil {
			t.Fatal(err)
		}
		original := first[17]
		first[17] ^= 0xff
		if _, err := cache.ReadRows(0, 1, first[:]); err != nil {
			t.Fatal(err)
		}
		if first[17] != original {
			t.Fatal("caller mutation changed cached row")
		}
	}

	// Invalid prefetch/consume calls leave the live hash state retryable.
	stream := streamTestMemoryStream(t, layout, packed)
	if err := stream.Prefetch([]int{len(layout.TokenMap)}, nil); err == nil {
		t.Fatal("expected invalid prefetch to fail")
	}
	if _, err := stream.Consume([]int{len(layout.TokenMap)}, nil); err == nil {
		t.Fatal("expected invalid consume to fail")
	}
	got, err := stream.Consume(tokens[:2], mask[:2])
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, wantRows[:len(got)]) {
		t.Fatal("failed prefetch/consume advanced live hash state")
	}

	failedStream := streamTestFailOnceStream(t, layout, packed)
	if _, err := failedStream.Consume(tokens[:2], mask[:2]); err == nil {
		t.Fatal("expected backing read failure")
	}
	got, err = failedStream.Consume(tokens[:2], mask[:2])
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, wantRows[:len(got)]) {
		t.Fatal("failed row gather advanced live hash state before retry")
	}
}

func TestV41EngramStreamTypedFailuresAndBoundedCollisions(t *testing.T) {
	packed := streamTestPackedRows(8)
	tests := []struct {
		name string
		make func() (V41EngramRowSource, error)
		read bool
		kind V41EngramStreamErrorKind
	}{
		{"nil reader", func() (V41EngramRowSource, error) {
			return NewV41PackedEngramRowSource(nil, V41EngramShard{ID: "s", ExpectedID: "s", Size: int64(len(packed)), Rows: 8})
		}, false, V41EngramStreamGeometry},
		{"mismatched shard", func() (V41EngramRowSource, error) {
			return NewV41PackedEngramRowSource(bytes.NewReader(packed), V41EngramShard{ID: "actual", ExpectedID: "wanted", Size: int64(len(packed)), Rows: 8})
		}, false, V41EngramStreamShardMismatch},
		{"out of bounds", func() (V41EngramRowSource, error) {
			return NewV41PackedEngramRowSource(bytes.NewReader(packed), V41EngramShard{ID: "s", ExpectedID: "s", Size: int64(len(packed)), Rows: 8})
		}, true, V41EngramStreamOutOfBounds},
		{"short read", func() (V41EngramRowSource, error) {
			return NewV41PackedEngramRowSource(bytes.NewReader(packed[:len(packed)-1]), V41EngramShard{ID: "s", ExpectedID: "s", Size: int64(len(packed)), Rows: 8})
		}, true, V41EngramStreamShortRead},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src, err := tt.make()
			if err == nil && tt.read {
				row := 8
				if tt.kind == V41EngramStreamShortRead {
					row = 7
				}
				_, err = src.ReadRows(row, 1, make([]byte, V41EngramPackedRowBytes))
			}
			var typed *V41EngramStreamError
			if !errors.As(err, &typed) || typed.Kind != tt.kind {
				t.Fatalf("error=%v, want typed kind %q", err, tt.kind)
			}
		})
	}

	src, _ := NewV41PackedEngramRowSource(bytes.NewReader(packed), V41EngramShard{
		ID: "s", ExpectedID: "s", Size: int64(len(packed)), Rows: 8,
	})
	cache, err := NewV41EngramSetCache(src, V41EngramSetCacheOptions{TableRows: 8, Sets: 1, Ways: 2})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewV41EngramSetCache(src, V41EngramSetCacheOptions{
		TableRows: 8, Sets: 1, Ways: 2, BudgetBytes: 2*V41EngramPackedRowBytes - 1,
	}); err == nil {
		t.Fatal("expected undersized cache budget to fail")
	}
	buf := make([]byte, V41EngramPackedRowBytes)
	for _, row := range []int{0, 1, 0, 2, 0} {
		if _, err := cache.ReadRows(row, 1, buf); err != nil {
			t.Fatal(err)
		}
	}
	stats := cache.Stats()
	if stats.Hits != 1 || stats.Misses != 4 || stats.Reads != 4 || stats.Evictions != 2 {
		t.Fatalf("unexpected one-set/two-way collision accounting: %+v", stats)
	}
	retained := 0
	for i := range cache.sets {
		for _, slot := range cache.sets[i].slots {
			retained += len(slot.data)
		}
	}
	if retained > 2*V41EngramPackedRowBytes {
		t.Fatalf("cache retained %d bytes beyond configured %d-byte residency", retained, 2*V41EngramPackedRowBytes)
	}
}

func TestV41EngramStreamSameSetMissesOverlap(t *testing.T) {
	reader := newStreamTestBarrierReader(streamTestPackedRows(4), 2)
	defer func() {
		select {
		case <-reader.release:
		default:
			close(reader.release)
		}
	}()
	src, err := NewV41PackedEngramRowSource(reader, V41EngramShard{
		ID: "s", ExpectedID: "s", Size: 4 * V41EngramPackedRowBytes, Rows: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	cache, err := NewV41EngramSetCache(src, V41EngramSetCacheOptions{TableRows: 4, Sets: 1, Ways: 2})
	if err != nil {
		t.Fatal(err)
	}
	errCh := make(chan error, 2)
	for _, row := range []int{0, 1} {
		go func(row int) {
			_, err := cache.ReadRows(row, 1, make([]byte, V41EngramPackedRowBytes))
			errCh <- err
		}(row)
	}
	select {
	case <-reader.bothEntered:
		close(reader.release)
	case <-time.After(2 * time.Second):
		t.Fatal("same-set cache miss held the set lock during ReaderAt.ReadAt")
	}
	for range 2 {
		if err := <-errCh; err != nil {
			t.Fatal(err)
		}
	}
}

// BenchmarkV41EngramStreamPrefetchLatencyDelta measures local-host consume
// latency through the OS page cache; it is not physical storage or HW evidence.
func BenchmarkV41EngramStreamPrefetchLatencyDelta(b *testing.B) {
	layout := streamTestLayout()
	packed := streamTestPackedRows(int(layout.Rows[0]))
	path := filepath.Join(b.TempDir(), "engram.packed264")
	if err := os.WriteFile(path, packed, 0o600); err != nil {
		b.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = f.Close() })
	tokens := []int{1, 3, 2, 0, 3, 1}
	measure := func(prefetch bool) (consume, setup time.Duration) {
		for range b.N {
			stream := streamTestReaderStream(b, layout, f, int64(len(packed)))
			if prefetch {
				start := time.Now()
				if err := stream.Prefetch(tokens, nil); err != nil {
					b.Fatal(err)
				}
				setup += time.Since(start)
			}
			start := time.Now()
			if _, err := stream.Consume(tokens, nil); err != nil {
				b.Fatal(err)
			}
			consume += time.Since(start)
		}
		return consume, setup
	}
	cold, _ := measure(false)
	prefetched, prefetchCost := measure(true)
	b.ReportMetric(float64(cold.Nanoseconds())/float64(b.N), "local-file-cold-ns/op")
	b.ReportMetric(float64(prefetched.Nanoseconds())/float64(b.N), "local-file-prefetched-ns/op")
	b.ReportMetric(float64(prefetchCost.Nanoseconds())/float64(b.N), "prefetch-cost-ns/op")
	b.ReportMetric(float64(cold.Nanoseconds()-prefetched.Nanoseconds())/float64(b.N), "consume-delta-ns/op")
}

func streamTestLayout() V41EngramLayout {
	return V41EngramLayout{
		TokenMap: []uint32{0, 1, 2, 3}, CompressedVocab: 4, PadID: 0,
		Rows: []uint32{12, 12}, Multipliers: [][]uint64{{17, 13}, {19, 15}}, Primes: [][]uint32{{5, 7}, {5, 7}},
		MaxNgramSize: 2, HeadsPerNgram: 2,
	}
}

func streamTestPackedRows(rows int) []byte {
	out := make([]byte, rows*V41EngramPackedRowBytes)
	for row := range rows {
		for col := range V41EngramPackedRowBytes {
			out[row*V41EngramPackedRowBytes+col] = byte(row*31 + col)
		}
	}
	return out
}

func streamTestRows(packed []byte, ids []uint32) [][]byte {
	out := make([][]byte, len(ids))
	for i, id := range ids {
		start := int(id) * V41EngramPackedRowBytes
		out[i] = append([]byte(nil), packed[start:start+V41EngramPackedRowBytes]...)
	}
	return out
}

func streamTestMemoryStream(tb testing.TB, layout V41EngramLayout, packed []byte) *V41EngramStream {
	tb.Helper()
	return streamTestReaderStream(tb, layout, bytes.NewReader(packed), int64(len(packed)))
}

func streamTestReaderStream(tb testing.TB, layout V41EngramLayout, reader io.ReaderAt, size int64) *V41EngramStream {
	tb.Helper()
	src, err := NewV41PackedEngramRowSource(reader, V41EngramShard{
		ID: "reader", ExpectedID: "reader", Size: size, Rows: int(layout.Rows[0]),
	})
	if err != nil {
		tb.Fatal(err)
	}
	caches := make([]*V41EngramSetCache, len(layout.Rows))
	for i := range caches {
		caches[i], err = NewV41EngramSetCache(src, V41EngramSetCacheOptions{TableRows: int(layout.Rows[i]), Sets: 2, Ways: 2})
		if err != nil {
			tb.Fatal(err)
		}
	}
	hash, err := NewV41EngramHashState(layout)
	if err != nil {
		tb.Fatal(err)
	}
	stream, err := NewV41EngramStream(hash, caches)
	if err != nil {
		tb.Fatal(err)
	}
	return stream
}

func streamTestFailOnceStream(tb testing.TB, layout V41EngramLayout, packed []byte) *V41EngramStream {
	tb.Helper()
	base, err := NewV41PackedEngramRowSource(bytes.NewReader(packed), V41EngramShard{
		ID: "flaky", ExpectedID: "flaky", Size: int64(len(packed)), Rows: int(layout.Rows[0]),
	})
	if err != nil {
		tb.Fatal(err)
	}
	src := &streamTestFailOnceSource{src: base}
	caches := make([]*V41EngramSetCache, len(layout.Rows))
	for i := range caches {
		caches[i], err = NewV41EngramSetCache(src, V41EngramSetCacheOptions{TableRows: int(layout.Rows[i]), Sets: 2, Ways: 2})
		if err != nil {
			tb.Fatal(err)
		}
	}
	hash, err := NewV41EngramHashState(layout)
	if err != nil {
		tb.Fatal(err)
	}
	stream, err := NewV41EngramStream(hash, caches)
	if err != nil {
		tb.Fatal(err)
	}
	return stream
}

type streamTestFailOnceSource struct {
	mu     sync.Mutex
	failed bool
	src    V41EngramRowSource
}

func (s *streamTestFailOnceSource) RowBytes() int { return s.src.RowBytes() }

func (s *streamTestFailOnceSource) ReadRows(start, count int, dst []byte) (int, error) {
	s.mu.Lock()
	if !s.failed {
		s.failed = true
		s.mu.Unlock()
		return 0, &V41EngramStreamError{Kind: V41EngramStreamShortRead, Row: start, Err: io.ErrUnexpectedEOF}
	}
	s.mu.Unlock()
	return s.src.ReadRows(start, count, dst)
}

type streamTestBarrierReader struct {
	data        []byte
	want        int
	mu          sync.Mutex
	entered     int
	bothEntered chan struct{}
	release     chan struct{}
}

func newStreamTestBarrierReader(data []byte, want int) *streamTestBarrierReader {
	return &streamTestBarrierReader{data: data, want: want, bothEntered: make(chan struct{}), release: make(chan struct{})}
}

func (r *streamTestBarrierReader) ReadAt(p []byte, off int64) (int, error) {
	r.mu.Lock()
	r.entered++
	if r.entered == r.want {
		close(r.bothEntered)
	}
	r.mu.Unlock()
	<-r.release
	if off < 0 || off >= int64(len(r.data)) {
		return 0, fmt.Errorf("offset %d: %w", off, io.EOF)
	}
	n := copy(p, r.data[off:])
	if n != len(p) {
		return n, io.EOF
	}
	return n, nil
}
