package model

import (
	"bytes"
	"errors"
	"io"
	"os"
	"strings"
	"sync"
	"testing"
	"unsafe"
)

type issue9073ReaderAt struct {
	data  []byte
	reads int
}

func (r *issue9073ReaderAt) ReadAt(p []byte, off int64) (int, error) {
	r.reads++
	if off < 0 || off >= int64(len(r.data)) {
		return 0, io.EOF
	}
	n := copy(p, r.data[off:])
	if n != len(p) {
		return n, io.EOF
	}
	return n, nil
}

func TestLazyQ4KMaterializesExactRangeWithoutResidentCopy(t *testing.T) {
	cfg := Config{HiddenSize: 256}
	b := NewQuantBuilder(cfg, false)
	payload := make([]byte, q4kBlockBytes)
	for i := range payload {
		payload[i] = byte(i*17 + 3)
	}
	blob := append([]byte("prefix"), payload...)
	name := "model.layers.0.mlp.down_proj.weight"
	if err := b.AddLazyQ4K(name, []int{1, 256}, LazyQ4KRange{Reader: bytes.NewReader(blob), Offset: 6, Bytes: len(payload)}); err != nil {
		t.Fatal(err)
	}
	qt := b.m.q4kw[name]
	if qt == nil || len(qt.raw) != 0 || qt.lazy == nil {
		t.Fatalf("lazy tensor = %+v", qt)
	}
	got, err := qt.materializeRaw()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("materialized payload mismatch")
	}
	if len(qt.raw) != 0 {
		t.Fatalf("materialization retained %d bytes, want ephemeral", len(qt.raw))
	}
}

func TestIssue9073InvalidMappedViewFallsBackToByteIdenticalReadAt(t *testing.T) {
	payload := make([]byte, q4kBlockBytes)
	for i := range payload {
		payload[i] = byte(i*23 + 9)
	}
	prefix := []byte("reader-prefix")
	reader := &issue9073ReaderAt{data: append(append([]byte(nil), prefix...), payload...)}
	qt := &q4kTensor{out: 1, in: qkK, nblk: 1, lazy: &LazyQ4KRange{
		Reader: reader, Offset: int64(len(prefix)), Bytes: len(payload),
		MappedSpan: makePageAlignedResidentBytes(os.Getpagesize()), MappedOffset: os.Getpagesize() - 32,
	}}
	if _, _, ok := qt.mappedRaw(); ok {
		t.Fatal("out-of-bounds mapped view was selected")
	}
	got, err := qt.materializeRaw()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatal("ReadAt fallback changed Q4_K payload bytes")
	}
	if reader.reads != 1 {
		t.Fatalf("fallback ReadAt calls=%d, want 1", reader.reads)
	}
}

type closeWitness struct{ closes int }

func (c *closeWitness) Close() error { c.closes++; return nil }

func TestCloseWeightsOwnsLazyCheckpointLifetime(t *testing.T) {
	m := &Model{}
	w := &closeWitness{}
	m.SetWeightCloser(w)
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if w.closes != 1 {
		t.Fatalf("checkpoint close count = %d, want 1", w.closes)
	}
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if w.closes != 1 {
		t.Fatalf("checkpoint close count after double close = %d, want 1", w.closes)
	}
}

func TestIssue9073CloseWeightsReleasesBorrowingQ4KBeforeCheckpoint(t *testing.T) {
	old := releaseModelQ4KHandles
	defer func() { releaseModelQ4KHandles = old }()
	events := make([]string, 0, 2)
	releaseModelQ4KHandles = func(*Model) { events = append(events, "release") }
	m := &Model{}
	m.SetWeightCloser(closeFunc(func() error {
		events = append(events, "unmap")
		return nil
	}))
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if got, want := strings.Join(events, ","), "release,unmap"; got != want {
		t.Fatalf("teardown order = %q, want %q", got, want)
	}
}

type closeFunc func() error

func (f closeFunc) Close() error { return f() }

type failingReaderAt struct{}

func (failingReaderAt) ReadAt([]byte, int64) (int, error) { return 0, errors.New("read failed") }

func TestCloseWeightsConcurrentCallsCloseOnce(t *testing.T) {
	m := &Model{}
	w := &closeWitness{}
	m.SetWeightCloser(w)
	var wg sync.WaitGroup
	for range 32 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := m.CloseWeights(); err != nil {
				t.Errorf("CloseWeights: %v", err)
			}
		}()
	}
	wg.Wait()
	if w.closes != 1 {
		t.Fatalf("concurrent checkpoint close count = %d, want 1", w.closes)
	}
}

func TestLazyQ4KMaterializationFailsClosed(t *testing.T) {
	qt := &q4kTensor{out: 1, in: 256, nblk: 1, lazy: &LazyQ4KRange{Reader: failingReaderAt{}, Bytes: q4kBlockBytes}}
	if _, err := qt.materializeRaw(); err == nil {
		t.Fatal("materialize succeeded after checkpoint read failure")
	}
}

func TestLazyQ4KCPUFallbackFailsClosed(t *testing.T) {
	qt := &q4kTensor{
		out:  1,
		in:   qkK,
		nblk: 1,
		lazy: &LazyQ4KRange{Reader: bytes.NewReader(make([]byte, q4kBlockBytes)), Bytes: q4kBlockBytes},
	}
	defer func() {
		got := recover()
		if got == nil {
			t.Fatal("CPU GEMV accepted a lazy Q4_K tensor")
		}
		if msg, ok := got.(string); !ok || msg == "" {
			t.Fatalf("CPU fallback panic = %#v, want a clear diagnostic", got)
		}
	}()
	q4kMatRows(qt, make([]float32, qkK))
}

func TestLazyQ4KMaterializationRejectsShortRead(t *testing.T) {
	qt := &q4kTensor{
		out:  1,
		in:   qkK,
		nblk: 1,
		lazy: &LazyQ4KRange{Reader: bytes.NewReader(make([]byte, q4kBlockBytes-1)), Bytes: q4kBlockBytes},
	}
	if _, err := qt.materializeRaw(); err == nil {
		t.Fatal("materialize accepted a truncated checkpoint range")
	}
}

// chunkedProbeReaderAt records the size of every ReadAt so a test can witness that a
// large lazy tensor is streamed through bounded windows rather than one whole-payload read.
type chunkedProbeReaderAt struct {
	data    []byte
	reads   int
	maxRead int
}

func (r *chunkedProbeReaderAt) ReadAt(p []byte, off int64) (int, error) {
	r.reads++
	if len(p) > r.maxRead {
		r.maxRead = len(p)
	}
	if off < 0 || off >= int64(len(r.data)) {
		return 0, io.EOF
	}
	n := copy(p, r.data[off:])
	if n != len(p) {
		return n, io.EOF
	}
	return n, nil
}

// TestLazyQ4KChunkedMaterializeBoundedWindowAndByteIdentical pins the bounded-window
// invariant for the non-mmap path: a payload larger than q4kMaterializeWindowBytes is
// materialized byte-identically, while no single ReadAt is larger than the window, so the
// peak transient host allocation beyond the page-aligned output is bounded by the window
// rather than the tensor size.
func TestLazyQ4KChunkedMaterializeBoundedWindowAndByteIdentical(t *testing.T) {
	payloadLen := q4kMaterializeWindowBytes + q4kBlockBytes*3

	want := make([]byte, payloadLen)
	for i := range want {
		want[i] = byte(i*31 + 7)
	}
	r := &chunkedProbeReaderAt{data: append([]byte("prefix"), want...)}
	qt := &q4kTensor{out: 1, in: qkK, nblk: 1, lazy: &LazyQ4KRange{
		Reader: r, Offset: int64(len("prefix")), Bytes: payloadLen,
	}}
	got, err := qt.materializeRaw()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("chunked materialization changed payload bytes")
	}
	if r.maxRead > q4kMaterializeWindowBytes {
		t.Fatalf("largest ReadAt = %d bytes, want <= window %d", r.maxRead, q4kMaterializeWindowBytes)
	}
	if r.reads < 2 {
		t.Fatalf("ReadAt calls = %d, want multiple bounded windows", r.reads)
	}
	if uintptr(unsafe.Pointer(&got[0]))%uintptr(os.Getpagesize()) != 0 {
		t.Fatal("chunked output is not page-aligned resident bytes")
	}
}

// TestLazyQ4KMappedSpanZeroCopyByteIdentical pins the mmap seam: a validated mapped view is
// returned directly, allocating nothing and performing zero ReadAt calls, and its bytes equal
// the historical ReadAt payload.
func TestLazyQ4KMappedSpanZeroCopyByteIdentical(t *testing.T) {
	const offset = 32
	payload := make([]byte, q4kBlockBytes)
	for i := range payload {
		payload[i] = byte(i*13 + 5)
	}
	span := makePageAlignedResidentBytes(os.Getpagesize())
	copy(span[offset:], payload)
	r := &chunkedProbeReaderAt{data: append([]byte("fallback"), payload...)}
	qt := &q4kTensor{out: 1, in: qkK, nblk: 1, lazy: &LazyQ4KRange{
		Reader: r, Offset: int64(len("fallback")), Bytes: len(payload),
		MappedSpan: span, MappedOffset: offset,
	}}
	got, err := qt.materializeRaw()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatal("mapped zero-copy view is not byte-identical to the ReadAt payload")
	}
	if &got[0] != &span[offset] {
		t.Fatal("materializeRaw did not return the mapped span view")
	}
	if r.reads != 0 {
		t.Fatalf("mapped path performed %d ReadAt calls, want zero", r.reads)
	}
}

// TestLazyQ4KMaterializeKeepsRawMemoization pins that a second call still returns the cached
// resident copy without touching the reader again.
func TestLazyQ4KMaterializeKeepsRawMemoization(t *testing.T) {
	payload := make([]byte, q4kBlockBytes)
	for i := range payload {
		payload[i] = byte(i*3 + 11)
	}
	r := &chunkedProbeReaderAt{data: append([]byte(nil), payload...)}
	qt := &q4kTensor{out: 1, in: qkK, nblk: 1, raw: payload, lazy: &LazyQ4KRange{
		Reader: r, Bytes: len(payload),
	}}
	got, err := qt.materializeRaw()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatal("memoized raw mismatch")
	}
	if r.reads != 0 {
		t.Fatalf("memoized path performed %d ReadAt calls, want zero", r.reads)
	}
}

// --- #13202 lazy k-quant materialization (Q2_K/Q3_K/Q5_K/Q6_K) ---

// kQuantLazyRange builds a fake checkpoint blob holding nblk Q2_K/Q6_K super-blocks and
// returns the payload plus a range that points at it with an unaligned prefix.
func kQuantLazyLazyPayload(kind kQuantKind, out, in int) []byte {
	nblk := in / kind.blockWeights()
	payload := make([]byte, out*nblk*kind.blockBytes())
	for i := range payload {
		payload[i] = byte(i*29 + 13)
	}
	return payload
}

func TestLazyKQuantHoldsDescriptorWithoutResidentCopy(t *testing.T) {
	for _, kind := range []kQuantKind{kindQ2K, kindQ6K} {
		cfg := Config{HiddenSize: 256}
		b := NewQuantBuilder(cfg, false)
		payload := kQuantLazyLazyPayload(kind, 1, 256)
		blob := append([]byte("prefix"), payload...)
		name := "model.layers.0.mlp.down_proj.weight"
		if err := b.AddLazyKQuant(name, []int{1, 256}, kind, LazyQ4KRange{Reader: bytes.NewReader(blob), Offset: 6, Bytes: len(payload)}); err != nil {
			t.Fatalf("%s: %v", kind, err)
		}
		qt := b.m.kqw[name]
		if qt == nil || len(qt.raw) != 0 || qt.lazy == nil {
			t.Fatalf("%s: lazy tensor = %+v", kind, qt)
		}
		got, err := qt.materializeRaw()
		if err != nil {
			t.Fatalf("%s: %v", kind, err)
		}
		if !bytes.Equal(got, payload) {
			t.Fatalf("%s: materialized payload mismatch", kind)
		}
		if len(qt.raw) != 0 {
			t.Fatalf("%s: materialization retained %d bytes, want ephemeral", kind, len(qt.raw))
		}
	}
}

func TestLazyKQuantAddRejectsShapeAndByteMismatch(t *testing.T) {
	cfg := Config{HiddenSize: 256}
	b := NewQuantBuilder(cfg, false)
	name := "model.layers.0.mlp.down_proj.weight"
	expectPanic := func(label string, fn func()) {
		defer func() {
			if recover() == nil {
				t.Fatalf("%s: expected panic, got none", label)
			}
		}()
		fn()
	}
	expectPanic("unaligned reduction dim", func() {
		_ = b.AddLazyKQuant(name, []int{1, 250}, kindQ6K, LazyQ4KRange{Reader: bytes.NewReader(nil), Bytes: q6kBlockBytes})
	})
	expectPanic("payload size mismatch", func() {
		_ = b.AddLazyKQuant(name, []int{1, 256}, kindQ6K, LazyQ4KRange{Reader: bytes.NewReader(nil), Bytes: q6kBlockBytes - 1})
	})
}

func TestLazyKQuantChunkedMaterializeBoundedWindowAndByteIdentical(t *testing.T) {
	payloadLen := q4kMaterializeWindowBytes + q6kBlockBytes*3
	want := make([]byte, payloadLen)
	for i := range want {
		want[i] = byte(i*37 + 11)
	}
	r := &chunkedProbeReaderAt{data: append([]byte("prefix"), want...)}
	for _, kind := range []kQuantKind{kindQ2K, kindQ3K, kindQ5K, kindQ6K} {
		r.reads, r.maxRead = 0, 0
		qt := &kQuantTensor{
			out:  1,
			in:   kind.blockWeights(),
			nblk: 1,
			kind: kind,
			lazy: &LazyQ4KRange{Reader: r, Offset: int64(len("prefix")), Bytes: payloadLen},
		}
		got, err := qt.materializeRaw()
		if err != nil {
			t.Fatalf("%s: %v", kind, err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("%s: chunked materialization changed payload bytes", kind)
		}
		if r.maxRead > q4kMaterializeWindowBytes {
			t.Fatalf("%s: largest ReadAt = %d bytes, want <= window %d", kind, r.maxRead, q4kMaterializeWindowBytes)
		}
		if r.reads < 2 {
			t.Fatalf("%s: ReadAt calls = %d, want multiple bounded windows", kind, r.reads)
		}
		if uintptr(unsafe.Pointer(&got[0]))%uintptr(os.Getpagesize()) != 0 {
			t.Fatalf("%s: chunked output is not page-aligned resident bytes", kind)
		}
	}
}

func TestLazyKQuantMappedSpanZeroCopyByteIdentical(t *testing.T) {
	const offset = 32
	for _, kind := range []kQuantKind{kindQ2K, kindQ6K} {
		payload := kQuantLazyLazyPayload(kind, 1, 256)
		span := makePageAlignedResidentBytes(os.Getpagesize())
		copy(span[offset:], payload)
		r := &chunkedProbeReaderAt{data: append([]byte("fallback"), payload...)}
		qt := &kQuantTensor{
			out: 1, in: 256, nblk: 1, kind: kind,
			lazy: &LazyQ4KRange{
				Reader: r, Offset: int64(len("fallback")), Bytes: len(payload),
				MappedSpan: span, MappedOffset: offset,
			},
		}
		got, err := qt.materializeRaw()
		if err != nil {
			t.Fatalf("%s: %v", kind, err)
		}
		if !bytes.Equal(got, payload) {
			t.Fatalf("%s: mapped zero-copy view is not byte-identical to the ReadAt payload", kind)
		}
		if &got[0] != &span[offset] {
			t.Fatalf("%s: materializeRaw did not return the mapped span view", kind)
		}
		if r.reads != 0 {
			t.Fatalf("%s: mapped path performed %d ReadAt calls, want zero", kind, r.reads)
		}
	}
}

func TestLazyKQuantMaterializationFailsClosed(t *testing.T) {
	for _, kind := range []kQuantKind{kindQ2K, kindQ6K} {
		qt := &kQuantTensor{out: 1, in: 256, nblk: 1, kind: kind, lazy: &LazyQ4KRange{Reader: failingReaderAt{}, Bytes: kind.blockBytes()}}
		if _, err := qt.materializeRaw(); err == nil {
			t.Fatalf("%s: materialize succeeded after checkpoint read failure", kind)
		}
	}
}

func TestLazyKQuantMaterializationRejectsShortRead(t *testing.T) {
	blk := kindQ6K.blockBytes()
	qt := &kQuantTensor{out: 1, in: 256, nblk: 1, kind: kindQ6K, lazy: &LazyQ4KRange{Reader: bytes.NewReader(make([]byte, blk-1)), Bytes: blk}}
	if _, err := qt.materializeRaw(); err == nil {
		t.Fatal("materialize accepted a truncated checkpoint range")
	}
}

func TestLazyKQuantCPUFallbackFailsClosed(t *testing.T) {
	for _, kind := range []kQuantKind{kindQ2K, kindQ6K} {
		qt := &kQuantTensor{
			out: 1, in: 256, nblk: 1, kind: kind,
			lazy: &LazyQ4KRange{Reader: bytes.NewReader(make([]byte, kind.blockBytes())), Bytes: kind.blockBytes()},
		}
		func() {
			defer func() {
				got := recover()
				if got == nil {
					t.Fatalf("%s: CPU GEMV accepted a lazy k-quant tensor", kind)
				}
				if msg, ok := got.(string); !ok || msg == "" {
					t.Fatalf("%s: CPU fallback panic = %#v, want a clear diagnostic", kind, got)
				}
			}()
			x := make([]float32, 256)
			y := make([]float32, 1)
			kQuantMatRowsRange(qt, x, y, 0, 1)
		}()
	}
}

func TestLazyKQuantMaterializeKeepsRawMemoization(t *testing.T) {
	payload := kQuantLazyLazyPayload(kindQ6K, 1, 256)
	r := &chunkedProbeReaderAt{data: append([]byte(nil), payload...)}
	qt := &kQuantTensor{out: 1, in: 256, nblk: 1, kind: kindQ6K, raw: payload, lazy: &LazyQ4KRange{Reader: r, Bytes: len(payload)}}
	got, err := qt.materializeRaw()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatal("memoized raw mismatch")
	}
	if r.reads != 0 {
		t.Fatalf("memoized path performed %d ReadAt calls, want zero", r.reads)
	}
}
