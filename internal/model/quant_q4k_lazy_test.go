package model

import (
	"bytes"
	"errors"
	"io"
	"math"
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

// TestLazyKQuantCPUCGEMVMaterializesOnDemand is the #13216 RED->GREEN witness: the CPU GEMV over
// a checkpoint-backed (lazy) k-quant tensor must MATERIALIZE the bounded range and compute, not
// panic. Before the fix this path hit requireRawCPU's #13202 panic; after it, ensureRawCPU faults
// the range into qt.raw and the GEMV produces exactly the resident-tensor result. Byte-identity
// against the same payload held resident is the no-silent-zeros property.
func TestLazyKQuantCPUCGEMVMaterializesOnDemand(t *testing.T) {
	for _, kind := range []kQuantKind{kindQ2K, kindQ3K, kindQ5K, kindQ6K} {
		payload := kQuantLazyLazyPayload(kind, 4, 256)
		resident := &kQuantTensor{out: 4, in: 256, nblk: 1, kind: kind, raw: payload}
		reader := &chunkedProbeReaderAt{data: append([]byte(nil), payload...)}
		lazy := &kQuantTensor{
			out: 4, in: 256, nblk: 1, kind: kind,
			lazy: &LazyQ4KRange{Reader: reader, Bytes: len(payload)},
		}
		x := make([]float32, 256)
		for i := range x {
			x[i] = float32(i%13) * 0.25
		}
		want := kQuantMatRows(resident, x)
		got := func() []float32 {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("%s: CPU GEMV panicked on a materializable lazy tensor: %v", kind, r)
				}
			}()
			return kQuantMatRows(lazy, x)
		}()
		if len(lazy.raw) == 0 {
			t.Fatalf("%s: CPU GEMV did not memoize the materialized range into raw", kind)
		}
		if reader.reads == 0 {
			t.Fatalf("%s: CPU GEMV produced a result without reading the checkpoint", kind)
		}
		for o := range want {
			if math.Float32bits(got[o]) != math.Float32bits(want[o]) {
				t.Fatalf("%s: row %d = %v, want byte-identical resident result %v (silent-zeros guard)", kind, o, got[o], want[o])
			}
		}
	}
}

// TestLazyKQuantCPUGEMVStillFailsClosedOnUnmaterializable is the retained fail-closed half of the
// #13202 contract: a lazy tensor whose range genuinely cannot be read (reader error) must still
// panic with a named diagnostic from the CPU entry point — ensureRawCPU materializes or refuses,
// it never silently reads nil raw and emits zeros.
func TestLazyKQuantCPUGEMVStillFailsClosedOnUnmaterializable(t *testing.T) {
	for _, kind := range []kQuantKind{kindQ2K, kindQ6K} {
		qt := &kQuantTensor{
			out: 1, in: 256, nblk: 1, kind: kind,
			lazy: &LazyQ4KRange{Reader: failingReaderAt{}, Bytes: kind.blockBytes()},
		}
		msg := recoverContains(func() {
			kQuantMatRowsRange(qt, make([]float32, 256), make([]float32, 1), 0, 1)
		})
		if msg == "" {
			t.Fatalf("%s: CPU GEMV accepted a lazy tensor whose range cannot be read", kind)
		}
		if !strings.Contains(msg, "materialization failed") && !strings.Contains(msg, "#13202") {
			t.Fatalf("%s: fail-closed panic = %q, want a named materialization/guard diagnostic", kind, msg)
		}
		if len(qt.raw) != 0 {
			t.Fatalf("%s: failed materialization left %d resident bytes, want none", kind, len(qt.raw))
		}
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

// lazyKQuantSeamReaderAt counts payload reads so a test can witness that an exported
// per-type lazy entry point stored a descriptor WITHOUT touching the checkpoint bytes.
type lazyKQuantSeamReaderAt struct {
	data  []byte
	reads int
}

func (r *lazyKQuantSeamReaderAt) ReadAt(p []byte, off int64) (int, error) {
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

// TestAddLazyKQuantPerTypeExportsReachThePrimitive is the RED->GREEN witness for the
// exported per-type lazy k-quant seam: AddLazyKQuant takes the unexported kQuantKind, so
// it is unreachable outside package model. Each AddLazyKQuantQxK wrapper must delegate to
// the primitive with the matching kind. This test drives each exported method with a
// correctly-sized range, asserts KQuantLazy(name) is true and that NO payload byte was read
// (the descriptor is held, not materialized), and asserts the fail-closed direction: a wrong
// byte count panics through the exported method exactly as it does through the primitive.
func TestAddLazyKQuantPerTypeExportsReachThePrimitive(t *testing.T) {
	cases := []struct {
		name  string
		add   func(b *QuantBuilder, canon string, shape []int, src LazyQ4KRange) error
		bytes int
	}{
		{"Q2_K", (*QuantBuilder).AddLazyKQuantQ2K, 84},
		{"Q3_K", (*QuantBuilder).AddLazyKQuantQ3K, 110},
		{"Q5_K", (*QuantBuilder).AddLazyKQuantQ5K, 176},
		{"Q6_K", (*QuantBuilder).AddLazyKQuantQ6K, 210},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{HiddenSize: 256}
			b := NewQuantBuilder(cfg, false)
			name := "model.layers.0.mlp.down_proj.weight"
			payload := make([]byte, tc.bytes)
			for i := range payload {
				payload[i] = byte(i*41 + 7)
			}
			reader := &lazyKQuantSeamReaderAt{data: payload}
			rangeSrc := LazyQ4KRange{Reader: reader, Bytes: len(payload)}
			if err := tc.add(b, name, []int{1, 256}, rangeSrc); err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			if !b.m.KQuantLazy(name) {
				t.Fatalf("%s: KQuantLazy(%q) = false, want true", tc.name, name)
			}
			if reader.reads != 0 {
				t.Fatalf("%s: store read %d payload bytes, want 0 (descriptor only)", tc.name, reader.reads)
			}

			// Fail-closed: a payload-size mismatch panics in AddLazyKQuant, and the
			// exported wrapper must reach that same panic rather than swallow it.
			func() {
				defer func() {
					if recover() == nil {
						t.Fatalf("%s: wrong byte count did not panic through the exported method", tc.name)
					}
				}()
				b2 := NewQuantBuilder(cfg, false)
				_ = tc.add(b2, name, []int{1, 256}, LazyQ4KRange{Reader: reader, Bytes: len(payload) - 1})
			}()
		})
	}
}

// TestLazyKQuantCPUGEMVHonorsDenseResidentBound is the #13253 RED->GREEN witness that the
// declared bounded streamed-dense working set has a genuine runtime consumer. Before the fix
// WithStreamedDenseQ4KWorkingSet only recorded the budget; a lazy dense k-quant then
// materialized + MEMOIZED its whole payload into qt.raw with nothing to stop retained dense
// bytes from exceeding the declaration. After it, a bound attached to the model turns memoized
// materialization into a fail-closed retention ceiling: the first payload fitting the bound
// materializes and computes byte-identically to the resident tensor, and the next charge past
// the bound panics by name instead of silently growing host anon-RSS.
func TestLazyKQuantCPUGEMVHonorsDenseResidentBound(t *testing.T) {
	kind := kindQ6K
	payload := kQuantLazyLazyPayload(kind, 4, 256)
	payloadLen := int64(len(payload))

	newLazy := func() (*Model, *kQuantTensor) {
		m := &Model{}
		m.denseResidentBoundBytes = payloadLen // exactly one payload's worth
		ledger := m.denseLedger()
		if ledger == nil {
			t.Fatal("declared bound produced a nil ledger")
		}
		reader := &chunkedProbeReaderAt{data: append([]byte(nil), payload...)}
		return m, &kQuantTensor{
			out: 4, in: 256, nblk: 1, kind: kind,
			lazy:       &LazyQ4KRange{Reader: reader, Bytes: len(payload)},
			denseBound: ledger,
		}
	}

	x := make([]float32, 256)
	for i := range x {
		x[i] = float32(i%13) * 0.25
	}
	resident := &kQuantTensor{out: 4, in: 256, nblk: 1, kind: kind, raw: payload}
	want := kQuantMatRows(resident, x)

	// Sub-case GREEN/default-off: an UNBOUNDED model (no declaration) materializes two lazy
	// tensors and computes byte-identically to the resident result — no new behavior.
	t.Run("unbounded default-off", func(t *testing.T) {
		unbounded := &Model{}
		if unbounded.denseLedger() != nil {
			t.Fatal("bound 0 allocated a ledger, want none")
		}
		for i := 0; i < 2; i++ {
			reader := &chunkedProbeReaderAt{data: append([]byte(nil), payload...)}
			lazy := &kQuantTensor{
				out: 4, in: 256, nblk: 1, kind: kind,
				lazy:       &LazyQ4KRange{Reader: reader, Bytes: len(payload)},
				denseBound: unbounded.denseLedger(), // nil: default-off
			}
			got := kQuantMatRows(lazy, x)
			if len(lazy.raw) == 0 {
				t.Fatalf("tensor %d: unbounded model did not memoize", i)
			}
			for o := range want {
				if math.Float32bits(got[o]) != math.Float32bits(want[o]) {
					t.Fatalf("tensor %d row %d = %v, want %v", i, o, got[o], want[o])
				}
			}
		}
	})

	// Sub-case GREEN/bounded: the first payload fits exactly; the second exceeds the ceiling
	// and materialization panics by name with #13253, never silently retaining past the bound.
	t.Run("bounded refuses over-budget retention", func(t *testing.T) {
		_, first := newLazy()
		firstGot := kQuantMatRows(first, x)
		if len(first.raw) == 0 {
			t.Fatal("first lazy tensor did not materialize under the bound")
		}
		for o := range want {
			if math.Float32bits(firstGot[o]) != math.Float32bits(want[o]) {
				t.Fatalf("first row %d = %v, want byte-identical resident result %v", o, firstGot[o], want[o])
			}
		}
		if got := first.denseBound.retained; got != payloadLen {
			t.Fatalf("retained after first = %d, want %d", got, payloadLen)
		}

		// Second tensor shares the SAME ledger; its payload would double retained past bound.
		second := &kQuantTensor{
			out: 4, in: 256, nblk: 1, kind: kind,
			lazy:       &LazyQ4KRange{Reader: &chunkedProbeReaderAt{data: append([]byte(nil), payload...)}, Bytes: len(payload)},
			denseBound: first.denseBound,
		}
		msg := recoverContains(func() { kQuantMatRows(second, x) })
		if msg == "" {
			t.Fatal("second materialization past the bound did not panic")
		}
		if !strings.Contains(msg, "dense resident bound") || !strings.Contains(msg, "#13253") {
			t.Fatalf("bound panic = %q, want it to name the bound and #13253", msg)
		}
		// RED-proof: the refused charge must not retain, and retained must never exceed bound.
		if len(second.raw) != 0 {
			t.Fatalf("refused second materialization retained %d bytes, want none", len(second.raw))
		}
		if first.denseBound.retained > first.denseBound.bound {
			t.Fatalf("retained %d exceeded bound %d", first.denseBound.retained, first.denseBound.bound)
		}
		if first.denseBound.retained != payloadLen {
			t.Fatalf("refused charge moved retained to %d, want it unchanged at %d", first.denseBound.retained, payloadLen)
		}
	})
}

// TestSetDenseResidentBoundRefusesNegativeAsUnbounded pins the option contract: a negative
// declared budget is not silently accepted as a real ceiling — SetDenseResidentBound clamps it
// to 0 (unbounded) and never creates a ledger, matching WithStreamedDenseQ4KWorkingSet's
// refusal of a negative budget at resolution.
func TestSetDenseResidentBoundRefusesNegativeAsUnbounded(t *testing.T) {
	b := NewQuantBuilder(Config{HiddenSize: 256}, false)
	b.SetDenseResidentBound(-4096)
	if b.m.denseResidentBoundBytes != 0 {
		t.Fatalf("negative bound stored as %d, want 0", b.m.denseResidentBoundBytes)
	}
	if b.m.denseLedger() != nil {
		t.Fatal("negative bound allocated a ledger, want none")
	}
}
