package model

import (
	"bytes"
	"math/rand"
	"runtime"
	"runtime/debug"
	"sync"
	"testing"
	"time"
)

// raceEnabled reports whether the test binary was built with -race. The race detector injects
// scheduling points that defeat GOMAXPROCS(1) serialization, so the ordering test must skip
// under it rather than assert a dispatch order the runtime no longer preserves.
func raceEnabled() bool {
	bi, ok := debug.ReadBuildInfo()
	if !ok {
		return false
	}
	for _, s := range bi.Settings {
		if s.Key == "-race" && s.Value == "true" {
			return true
		}
	}
	return false
}

// expert_one_copy_fill_test.go — independent adversarial witnesses for the checkpoint expert
// tier's one-copy read-span identity accessor and its Prefetch dedupe / ascending-dispatch /
// bounded-concurrency contract.
//
// These tests are written from the contract alone. They probe the edges a straightforward
// implementation may omit: e == -1 and e == Experts, duplicate names in one Prefetch call,
// dispatch ordering under a serialized reader, and the concurrency ceiling with every name
// on a distinct expert (so nothing is coalesced). Nothing here reads the implementation.

// newOneCopyBlob returns deterministic non-repeating bytes, so a wrong byte range can never
// coincidentally equal the right one.
func newOneCopyBlob(n int, seed int64) []byte {
	blob := make([]byte, n)
	rand.New(rand.NewSource(seed)).Read(blob)
	return blob
}

// orderReaderAt is an io.ReaderAt that records every request offset in arrival order under a
// mutex, holds each read briefly, and tracks peak overlap.
type orderReaderAt struct {
	r        *bytes.Reader
	mu       sync.Mutex
	offsets  []int64
	inflight int
	peak     int
	sleep    time.Duration
}

func (o *orderReaderAt) ReadAt(p []byte, off int64) (int, error) {
	o.mu.Lock()
	o.offsets = append(o.offsets, off)
	o.inflight++
	if o.inflight > o.peak {
		o.peak = o.inflight
	}
	o.mu.Unlock()
	time.Sleep(o.sleep)
	n, err := o.r.ReadAt(p, off)
	o.mu.Lock()
	o.inflight--
	o.mu.Unlock()
	return n, err
}

// immediateReaderAt records each offset the instant ReadAt is entered and never yields before
// recording. Combined with GOMAXPROCS(1) it serializes the read goroutines so their recording
// order reflects the tier's dispatch order rather than scheduler luck. (Under GOMAXPROCS>1 a
// mutex-holding variant still reordered, because sync.Mutex handoff is not FIFO.)
type immediateReaderAt struct {
	r       *bytes.Reader
	mu      sync.Mutex
	offsets []int64
}

func (o *immediateReaderAt) ReadAt(p []byte, off int64) (int, error) {
	o.mu.Lock()
	o.offsets = append(o.offsets, off)
	o.mu.Unlock()
	return o.r.ReadAt(p, off)
}

func (o *orderReaderAt) recorded() ([]int64, int) {
	o.mu.Lock()
	defer o.mu.Unlock()
	cp := append([]int64(nil), o.offsets...)
	return cp, o.peak
}

// oneCopyGateTier builds one fused gate projection with `experts` experts and a reader we
// control, so every expert name maps to a distinct byte range and nothing can coalesce.
func oneCopyGateTier(t *testing.T, experts int, sleep time.Duration, seed int64) (*ExpertCheckpointTier, *orderReaderAt, []byte, int64) {
	t.Helper()
	const H = 256
	stride := int64(H * (H / qkK) * q4kBlockBytes)
	blob := newOneCopyBlob(experts*int(stride), seed)
	src := &orderReaderAt{r: bytes.NewReader(blob), sleep: sleep}
	fused := []FusedExpertTensor{{
		Name: "blk.0.ffn_gate_exps.weight", Layer: 0, Proj: "gate_proj",
		Quant: ExpertCheckpointQ4K, Offset: 0, Experts: experts, Rows: H, Cols: H,
	}}
	tier := NewExpertCheckpointTier(0)
	if err := tier.AddShard(src, int64(len(blob)), fused); err != nil {
		t.Fatalf("AddShard: %v", err)
	}
	return tier, src, blob, stride
}

// oneCopyFusedTierBeforeReader builds the full three-projection layout (gate/up/down, 4 experts)
// behind a caller-supplied reader, matching the fused offsets AddShard expects. It returns the
// tier and the blob so byte tests can slice it.
func oneCopyFusedTierBeforeReader(t *testing.T, r *immediateReaderAt) (*ExpertCheckpointTier, []byte, int64) {
	t.Helper()
	const H, E = 256, 4
	stride := int64(H * (H / qkK) * q4kBlockBytes)
	blob := newOneCopyBlob(3*E*int(stride), 11)
	r.r = bytes.NewReader(blob)
	var fused []FusedExpertTensor
	for i, proj := range []string{"gate", "up", "down"} {
		fused = append(fused, FusedExpertTensor{
			Name: "blk.0.ffn_" + proj + "_exps.weight", Layer: 0, Proj: proj + "_proj",
			Quant: ExpertCheckpointQ4K, Offset: int64(i) * E * stride, Experts: E, Rows: H, Cols: H,
		})
	}
	tier := NewExpertCheckpointTier(0)
	if err := tier.AddShard(r, int64(len(blob)), fused); err != nil {
		t.Fatalf("AddShard: %v", err)
	}
	return tier, blob, stride
}

// TestExpertReadSpanIdentity pins the exact byte range accessor. It must be the fused slab's
// offset plus e strides, with size equal to one stride, and it must move no payload bytes.
// fak-test:runtime fast est=300ms lane=default
func TestExpertReadSpanIdentity(t *testing.T) {
	fix := newGGUFExpertFixture(t)
	d := fix.descs[0] // Q4_K geometry, 4 experts, stride 864, Offset 100.
	wantStride := int64(fixtureStride(d))
	if wantStride != 864 {
		t.Fatalf("fixture stride = %d, want 864; this test's arithmetic depends on it", wantStride)
	}

	rec := &expertRangeRecorder{r: bytes.NewReader(fix.file)}
	src, err := newGGUFExpertSource(rec, int64(len(fix.file)), fix.descs)
	if err != nil {
		t.Fatalf("newGGUFExpertSource: %v", err)
	}
	if len(rec.calls) != 0 {
		t.Fatalf("construction moved %d payload reads, want 0", len(rec.calls))
	}

	for e := 0; e < d.Experts; e++ {
		off, size, ok := src.expertReadSpan(d.Name, e)
		if !ok {
			t.Fatalf("expertReadSpan(%s, %d) declined a valid expert", d.Name, e)
		}
		if wantOff := d.Offset + int64(e)*wantStride; off != wantOff {
			t.Fatalf("expertReadSpan(%s, %d) off = %d, want Offset+e*stride = %d", d.Name, e, off, wantOff)
		}
		if size != wantStride {
			t.Fatalf("expertReadSpan(%s, %d) size = %d, want stride %d", d.Name, e, size, wantStride)
		}
	}
	// The accessor is a directory lookup; the fixture reader must not have seen a single read.
	if len(rec.calls) != 0 {
		t.Fatalf("expertReadSpan moved %d payload reads, want 0", len(rec.calls))
	}

	// Cross-check against the resident slice the streamed read is supposed to reproduce.
	off, size, ok := src.expertReadSpan(d.Name, 2)
	if !ok {
		t.Fatal("expertReadSpan(down, 2) declined")
	}
	want := fix.residentExpert(d, 2)
	if !bytes.Equal(fix.file[off:off+size], want) {
		t.Fatal("expertReadSpan range does not select the resident expert's bytes")
	}
}

// TestExpertReadSpanRejectsUnknownAndOutOfRange probes the ok=false boundaries: an unknown
// tensor, e = -1, and e == Experts (the first out-of-range index).
// fak-test:runtime fast est=300ms lane=default
func TestExpertReadSpanRejectsUnknownAndOutOfRange(t *testing.T) {
	fix := newGGUFExpertFixture(t)
	src, err := newGGUFExpertSource(bytes.NewReader(fix.file), int64(len(fix.file)), fix.descs)
	if err != nil {
		t.Fatalf("newGGUFExpertSource: %v", err)
	}
	d := fix.descs[0]

	cases := []struct {
		name   string
		tens   string
		expert int
	}{
		{"unknown tensor", "blk.9.ffn_gate_exps.weight", 0},
		{"empty tensor", "", 0},
		{"negative expert", d.Name, -1},
		{"expert at Experts", d.Name, d.Experts},
		{"expert far past Experts", d.Name, d.Experts + 1000},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			off, size, ok := src.expertReadSpan(tc.tens, tc.expert)
			if ok {
				t.Fatalf("expertReadSpan(%q, %d) = (%d, %d, true), want ok=false", tc.tens, tc.expert, off, size)
			}
			if off != 0 || size != 0 {
				t.Fatalf("declined span returned (%d, %d), want (0, 0)", off, size)
			}
		})
	}

	// The last valid expert must still be accepted, so the boundary is exclusive, not off-by-one.
	if _, _, ok := src.expertReadSpan(d.Name, d.Experts-1); !ok {
		t.Fatalf("expertReadSpan(%s, %d) declined the last valid expert", d.Name, d.Experts-1)
	}
}

// TestExpertCheckpointPrefetchDedupesRepeatedName repeats one name three times in a single
// call: exactly one read must start and the return value must count it once.
// fak-test:runtime fast est=300ms lane=default
func TestExpertCheckpointPrefetchDedupesRepeatedName(t *testing.T) {
	tier, src, _, _ := oneCopyGateTier(t, 1, 5*time.Millisecond, 3)
	name := expertName(0, 0, "gate_proj.weight")

	got := tier.Prefetch([]string{name, name, name})
	if got != 1 {
		t.Fatalf("Prefetch of one name repeated 3x returned %d, want 1", got)
	}
	waitPrefetchSettled(t, tier)

	offsets, _ := src.recorded()
	if len(offsets) != 1 {
		t.Fatalf("repeated Prefetch started %d reads (%v), want exactly 1", len(offsets), offsets)
	}
	if st := tier.Stats(); st.Prefetched != 1 {
		t.Fatalf("Stats().Prefetched = %d, want 1", st.Prefetched)
	}

	// And the one read must actually serve a later demand.
	if _, err := tier.fault(name); err != nil {
		t.Fatalf("fault after a deduped prefetch: %v", err)
	}
}

// TestExpertCheckpointPrefetchDispatchesAscendingOffsets feeds names in reverse offset order
// in a single Prefetch call and asserts the reader observed strictly ascending byte offsets.
//
// Global order is only observable when the work is serialized, so runtime.GOMAXPROCS is pinned
// to 1 and the reader records without yielding: the read goroutines then run in the order the
// tier dispatched them. The reversed input makes the assertion non-trivial — an unsorted tier
// would record the reversed offsets and fail immediately.
// fak-test:runtime fast est=300ms lane=default
func TestExpertCheckpointPrefetchDispatchesAscendingOffsets(t *testing.T) {
	if raceEnabled() {
		t.Skip("the race detector injects scheduling points that defeat GOMAXPROCS(1) serialization")
	}
	old := runtime.GOMAXPROCS(1)
	t.Cleanup(func() { runtime.GOMAXPROCS(old) })

	src := &immediateReaderAt{}
	tier, _, _ := oneCopyFusedTierBeforeReader(t, src)

	// Reverse of the ascending-offset order: down e3..e0, up e3..e0, gate e3..e0.
	var names []string
	for _, proj := range []string{"down", "up", "gate"} {
		for e := 3; e >= 0; e-- {
			names = append(names, expertName(0, e, proj+"_proj.weight"))
		}
	}
	if got := tier.Prefetch(names); got != len(names) {
		t.Fatalf("Prefetch started %d reads, want %d", got, len(names))
	}
	waitPrefetchSettled(t, tier)

	src.mu.Lock()
	offsets := append([]int64(nil), src.offsets...)
	src.mu.Unlock()
	if len(offsets) != len(names) {
		t.Fatalf("recorded %d reads, want %d", len(offsets), len(names))
	}
	for i := 1; i < len(offsets); i++ {
		if offsets[i] < offsets[i-1] {
			t.Fatalf("dispatch order not ascending at %d: offset %d followed %d (all %v)",
				i, offsets[i], offsets[i-1], offsets)
		}
	}
	// The first dispatch must be the globally smallest span, which the reversed input makes
	// non-trivial: gate expert 0 at offset 0.
	if offsets[0] != 0 {
		t.Fatalf("first dispatched offset = %d, want 0 (gate expert 0); reversed input was not sorted", offsets[0])
	}
}

// TestExpertCheckpointPrefetchBoundsConcurrency launches 32 names on 32 distinct experts and
// asserts the peak in-flight count never exceeds 8, while proving the bound is not silently
// one by requiring genuine overlap.
// fak-test:runtime medium est=1s lane=default
func TestExpertCheckpointPrefetchBoundsConcurrency(t *testing.T) {
	const n = 32
	tier, src, _, _ := oneCopyGateTier(t, n, 15*time.Millisecond, 5)

	names := make([]string, 0, n)
	for e := 0; e < n; e++ {
		names = append(names, expertName(0, e, "gate_proj.weight"))
	}
	if got := tier.Prefetch(names); got != n {
		t.Fatalf("Prefetch started %d reads, want %d", got, n)
	}
	waitPrefetchSettled(t, tier)

	offsets, peak := src.recorded()
	if len(offsets) != n {
		t.Fatalf("recorded %d reads, want %d", len(offsets), n)
	}
	if peak > 8 {
		t.Fatalf("peak in-flight reads = %d, want <= 8", peak)
	}
	if peak < 2 {
		t.Fatalf("peak in-flight reads = %d; the bound must permit overlap, not serialize every read", peak)
	}
}

// TestExpertCheckpointPrefetchPreservesBytesAfterDedupe feeds duplicates and fresh names in
// one call, then faults each distinct name and compares the served bytes with the checkpoint
// blob. This guards that dedupe and offset-sorting did not corrupt the one-shot handoff.
// fak-test:runtime fast est=300ms lane=default
func TestExpertCheckpointPrefetchPreservesBytesAfterDedupe(t *testing.T) {
	tier, _, blob, stride := oneCopyGateTier(t, 4, 5*time.Millisecond, 9)

	names := []string{
		expertName(0, 2, "gate_proj.weight"),
		expertName(0, 0, "gate_proj.weight"),
		expertName(0, 2, "gate_proj.weight"), // duplicate
		expertName(0, 3, "gate_proj.weight"),
		expertName(0, 1, "gate_proj.weight"),
		expertName(0, 0, "gate_proj.weight"), // duplicate
	}
	if got := tier.Prefetch(names); got != 4 {
		t.Fatalf("Prefetch returned %d, want 4 distinct reads", got)
	}
	waitPrefetchSettled(t, tier)

	for e := 0; e < 4; e++ {
		name := expertName(0, e, "gate_proj.weight")
		w, err := tier.fault(name)
		if err != nil {
			t.Fatalf("fault %s: %v", name, err)
		}
		start := int64(e) * stride
		if w.q4 == nil {
			t.Fatalf("fault %s returned a nil q4 weight", name)
		}
		if !bytes.Equal(w.q4.raw, blob[start:start+stride]) {
			t.Fatalf("fault %s served bytes that differ from the checkpoint slice [%d,%d)", name, start, start+stride)
		}
	}
}
