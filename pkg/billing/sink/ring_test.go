package sink

import (
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/billing"
)

func ptr(v int64) *int64 { return &v }

func rec(id string) billing.UsageRecord {
	return billing.UsageRecord{
		SchemaVersion:  billing.SchemaVersionV1,
		EventID:        id,
		Model:          "test-model",
		RequestedAt:    time.Date(2026, 11, 4, 12, 0, 0, 0, time.UTC),
		Classification: billing.UsageRecorded,
		PromptTokens:   ptr(10),
	}
}

// chanBackend is a deterministic test Backend: each successful Append
// records the batch's event IDs and pushes its length onto appends, so
// tests can wait for delivery without sleeping.
type chanBackend struct {
	mu      sync.Mutex
	got     [][]string
	appends chan int
	fail    int
	failAll bool
	failErr error
	closed  bool
}

func newChanBackend() *chanBackend {
	return &chanBackend{
		appends: make(chan int, 128),
		failErr: errors.New("backend down"),
	}
}

func (b *chanBackend) Append(batch []billing.UsageRecord) error {
	b.mu.Lock()
	if b.failAll || b.fail > 0 {
		if !b.failAll {
			b.fail--
		}
		b.mu.Unlock()
		return b.failErr
	}
	ids := make([]string, len(batch))
	for i := range batch {
		ids[i] = batch[i].EventID
	}
	b.got = append(b.got, ids)
	b.mu.Unlock()
	b.appends <- len(batch)
	return nil
}

func (b *chanBackend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.closed = true
	return nil
}

func (b *chanBackend) allIDs() []string {
	b.mu.Lock()
	defer b.mu.Unlock()
	var out []string
	for _, g := range b.got {
		out = append(out, g...)
	}
	return out
}

func (b *chanBackend) wasClosed() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.closed
}

// holdBackend blocks every Append until releaseAll is called, pinning
// the ring in a known state. It signals entered (buffered) when the
// first Append begins.
type holdBackend struct {
	entered     chan struct{}
	release     chan struct{}
	releaseOnce sync.Once
	mu          sync.Mutex
	ids         []string
	closed      bool
}

func newHoldBackend() *holdBackend {
	return &holdBackend{
		entered: make(chan struct{}, 1),
		release: make(chan struct{}),
	}
}

// releaseAll unblocks all pending and future Appends. Idempotent, safe
// to defer so a failing test cannot strand the flusher behind a
// permanently blocked backend.
func (h *holdBackend) releaseAll() {
	h.releaseOnce.Do(func() { close(h.release) })
}

func (h *holdBackend) Append(batch []billing.UsageRecord) error {
	select {
	case h.entered <- struct{}{}:
	default:
	}
	<-h.release
	h.mu.Lock()
	for i := range batch {
		h.ids = append(h.ids, batch[i].EventID)
	}
	h.mu.Unlock()
	return nil
}

func (h *holdBackend) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.closed = true
	return nil
}

func expectIDs(t *testing.T, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d ids %v, want %d %v", len(got), got, len(want), want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("id %d: got %q, want %q (full: %v)", i, got[i], want[i], got)
		}
	}
}

func TestNewRingValidation(t *testing.T) {
	bad := []struct {
		name string
		opts RingOptions
	}{
		{"zero capacity", RingOptions{Capacity: 0, BatchSize: 1, OverflowPolicy: DropOldest}},
		{"negative capacity", RingOptions{Capacity: -1, BatchSize: 1, OverflowPolicy: DropOldest}},
		{"zero batch", RingOptions{Capacity: 4, BatchSize: 0, OverflowPolicy: DropOldest}},
		{"batch exceeds capacity", RingOptions{Capacity: 4, BatchSize: 8, OverflowPolicy: DropOldest}},
		{"negative interval", RingOptions{Capacity: 4, BatchSize: 4, FlushInterval: -time.Millisecond, OverflowPolicy: DropOldest}},
		{"unknown policy", RingOptions{Capacity: 4, BatchSize: 4, OverflowPolicy: OverflowPolicy(99)}},
	}
	for _, tc := range bad {
		if _, err := NewRing(newChanBackend(), tc.opts); !errors.Is(err, ErrInvalidOptions) {
			t.Errorf("%s: got %v, want ErrInvalidOptions", tc.name, err)
		}
	}
	if _, err := NewRing(nil, RingOptions{Capacity: 1, BatchSize: 1, OverflowPolicy: DropOldest}); !errors.Is(err, ErrInvalidOptions) {
		t.Errorf("nil backend: got %v, want ErrInvalidOptions", err)
	}
}

// TestDropOldestCountsOverflow pins the ring full behind a blocked
// first Append, then verifies the overflow policy discards the oldest
// QUEUED record and counts it. Capacity 4, batch 2: writes 1-2 are
// popped into the blocked Append; writes 3-6 refill the ring; write 7
// overflows.
func TestDropOldestCountsOverflow(t *testing.T) {
	hb := newHoldBackend()
	r, err := NewRing(hb, RingOptions{Capacity: 4, BatchSize: 2, OverflowPolicy: DropOldest})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer hb.releaseAll() // registered last, runs first: never strand the flusher behind a blocked backend

	for _, id := range []string{"1", "2"} {
		if err := r.Write(rec(id)); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}
	<-hb.entered // the flusher popped [1,2] and is blocked inside Append

	for _, id := range []string{"3", "4", "5", "6"} {
		if err := r.Write(rec(id)); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}
	if got := r.Pending(); got != 6 {
		t.Fatalf("Pending = %d, want 6 (4 queued + 2 in flight)", got)
	}
	if err := r.Write(rec("7")); err != nil {
		t.Fatalf("write 7 under DropOldest: %v", err)
	}
	if got := r.Dropped(); got != 1 {
		t.Fatalf("Dropped = %d, want 1 (record 3 discarded)", got)
	}
	if got := r.Pending(); got != 6 {
		t.Fatalf("Pending after overflow = %d, want 6 (drop and enqueue cancel out)", got)
	}

	hb.releaseAll()
	if err := r.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	expectIDs(t, hb.ids, []string{"1", "2", "4", "5", "6", "7"})
	if got := r.Delivered(); got != 6 {
		t.Fatalf("Delivered = %d, want 6", got)
	}
}

// TestRejectNewRefusesAndCounts pins the ring full behind a blocked
// first Append, then verifies the overflow policy refuses the new
// record with ErrFull and counts it.
func TestRejectNewRefusesAndCounts(t *testing.T) {
	hb := newHoldBackend()
	r, err := NewRing(hb, RingOptions{Capacity: 4, BatchSize: 2, OverflowPolicy: RejectNew})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer hb.releaseAll() // registered last, runs first: never strand the flusher behind a blocked backend

	for _, id := range []string{"1", "2"} {
		if err := r.Write(rec(id)); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}
	<-hb.entered // the flusher popped [1,2] and is blocked inside Append

	for _, id := range []string{"3", "4", "5", "6"} {
		if err := r.Write(rec(id)); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}
	if err := r.Write(rec("7")); !errors.Is(err, ErrFull) {
		t.Fatalf("write 7 under RejectNew: got %v, want ErrFull", err)
	}
	if got := r.Rejected(); got != 1 {
		t.Fatalf("Rejected = %d, want 1", got)
	}
	if got := r.Pending(); got != 6 {
		t.Fatalf("Pending = %d, want 6 (4 queued + 2 in flight; refused record not enqueued)", got)
	}

	hb.releaseAll()
	if err := r.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	expectIDs(t, hb.ids, []string{"1", "2", "3", "4", "5", "6"})

	// Once drained, the same record is accepted.
	if err := r.Write(rec("7")); err != nil {
		t.Fatalf("write 7 after drain: %v", err)
	}
	if err := r.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	expectIDs(t, hb.ids, []string{"1", "2", "3", "4", "5", "6", "7"})
}

// TestFlushIsABarricade checks batching (a partial batch is held until
// asked) and that Flush delivers everything written before it returns.
func TestFlushIsABarricade(t *testing.T) {
	b := newChanBackend()
	r, err := NewRing(b, RingOptions{Capacity: 8, BatchSize: 4, OverflowPolicy: DropOldest})
	if err != nil {
		t.Fatal(err)
	}

	for _, id := range []string{"a", "b", "c"} {
		if err := r.Write(rec(id)); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}
	if got := r.Pending(); got != 3 {
		t.Fatalf("Pending = %d, want 3 (partial batch must be held, not flushed)", got)
	}
	if err := r.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if n := <-b.appends; n != 3 {
		t.Fatalf("first append carried %d records, want 3", n)
	}
	expectIDs(t, b.allIDs(), []string{"a", "b", "c"})
	if got := r.Pending(); got != 0 {
		t.Fatalf("Pending after Flush = %d, want 0", got)
	}

	for _, id := range []string{"d", "e", "f", "g"} {
		if err := r.Write(rec(id)); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}
	if n := <-b.appends; n != 4 {
		t.Fatalf("second append carried %d records, want 4", n)
	}
	expectIDs(t, b.allIDs(), []string{"a", "b", "c", "d", "e", "f", "g"})
	if got := r.Delivered(); got != 7 {
		t.Fatalf("Delivered = %d, want 7", got)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

// TestFlushAfterFailedAppendRetries checks that a failed batch stays
// held, is counted, and is retried before any newer record.
func TestFlushAfterFailedAppendRetries(t *testing.T) {
	b := newChanBackend()
	b.fail = 1
	r, err := NewRing(b, RingOptions{Capacity: 4, BatchSize: 4, OverflowPolicy: DropOldest})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	for _, id := range []string{"a", "b"} {
		if err := r.Write(rec(id)); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}
	if err := r.Flush(); !errors.Is(err, b.failErr) {
		t.Fatalf("first Flush: got %v, want the backend failure", err)
	}
	if got := r.Failures(); got != 1 {
		t.Fatalf("Failures = %d, want 1", got)
	}
	if got := r.Pending(); got != 2 {
		t.Fatalf("Pending after failed Flush = %d, want 2 (failed batch stays held)", got)
	}

	if err := r.Write(rec("c")); err != nil {
		t.Fatalf("write c: %v", err)
	}
	if err := r.Flush(); err != nil {
		t.Fatalf("second Flush: %v", err)
	}
	expectIDs(t, b.allIDs(), []string{"a", "b", "c"})
	if got := r.Delivered(); got != 3 {
		t.Fatalf("Delivered = %d, want 3", got)
	}
}

// TestCloseDrainsStopsAndIdempotent checks the Close lifecycle: final
// drain, backend close, flusher stop, idempotence, and typed refusal
// after close.
func TestCloseDrainsStopsAndIdempotent(t *testing.T) {
	b := newChanBackend()
	r, err := NewRing(b, RingOptions{Capacity: 4, BatchSize: 4, OverflowPolicy: DropOldest})
	if err != nil {
		t.Fatal(err)
	}

	for _, id := range []string{"a", "b"} {
		if err := r.Write(rec(id)); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}
	if err := r.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	expectIDs(t, b.allIDs(), []string{"a", "b"})
	if !b.wasClosed() {
		t.Fatal("backend Close was not called")
	}
	if got := r.Pending(); got != 0 {
		t.Fatalf("Pending after Close = %d, want 0", got)
	}
	select {
	case <-r.stopped:
	default:
		t.Fatal("flusher goroutine did not stop after Close")
	}

	if err := r.Close(); err != nil {
		t.Fatalf("second Close: got %v, want nil (idempotent)", err)
	}
	if err := r.Write(rec("c")); !errors.Is(err, ErrClosed) {
		t.Fatalf("Write after Close: got %v, want ErrClosed", err)
	}
	if err := r.Flush(); !errors.Is(err, ErrClosed) {
		t.Fatalf("Flush after Close: got %v, want ErrClosed", err)
	}
}

// TestCloseSurfacesFailureAndKeepsResidue checks that a persistently
// failing backend surfaces its error through Close and leaves
// undelivered records visible through Pending rather than silently
// losing them.
func TestCloseSurfacesFailureAndKeepsResidue(t *testing.T) {
	b := newChanBackend()
	b.failAll = true
	r, err := NewRing(b, RingOptions{Capacity: 4, BatchSize: 4, OverflowPolicy: DropOldest})
	if err != nil {
		t.Fatal(err)
	}

	for _, id := range []string{"a", "b"} {
		if err := r.Write(rec(id)); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}
	err = r.Close()
	if !errors.Is(err, b.failErr) {
		t.Fatalf("Close: got %v, want the backend failure", err)
	}
	if got := r.Pending(); got != 2 {
		t.Fatalf("Pending after failed Close = %d, want 2 (residue must stay visible)", got)
	}
	if got := r.Failures(); got < 1 {
		t.Fatalf("Failures = %d, want >= 1", got)
	}
	if got := r.Delivered(); got != 0 {
		t.Fatalf("Delivered = %d, want 0", got)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("second Close: got %v, want nil (idempotent)", err)
	}
	if err := r.Write(rec("c")); !errors.Is(err, ErrClosed) {
		t.Fatalf("Write after Close: got %v, want ErrClosed", err)
	}
}

// TestFlushIntervalDeliversPartialBatch checks the timer path: with the
// ring below batch size, the flush interval alone drives delivery.
func TestFlushIntervalDeliversPartialBatch(t *testing.T) {
	b := newChanBackend()
	r, err := NewRing(b, RingOptions{Capacity: 8, BatchSize: 8, FlushInterval: 50 * time.Millisecond, OverflowPolicy: DropOldest})
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Write(rec("a")); err != nil {
		t.Fatalf("write: %v", err)
	}
	if n := <-b.appends; n != 1 {
		t.Fatalf("timer-driven append carried %d records, want 1", n)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

// TestConcurrentWritesNoLoss hammers the ring from many goroutines and
// checks lossless, order-insensitive delivery with zero overflow.
func TestConcurrentWritesNoLoss(t *testing.T) {
	const (
		goroutines = 8
		perG       = 1000
	)
	b := newChanBackend()
	// Capacity exceeds the total number of records written below, so
	// overflow is impossible regardless of scheduling: the no-drop
	// assertion is deterministic, not probabilistic.
	r, err := NewRing(b, RingOptions{Capacity: 8192, BatchSize: 1024, OverflowPolicy: DropOldest})
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < perG; i++ {
				id := fmt.Sprintf("g%d-%04d", g, i)
				if err := r.Write(rec(id)); err != nil {
					t.Errorf("write %s: %v", id, err)
					return
				}
			}
		}(g)
	}
	wg.Wait()
	if err := r.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	want := goroutines * perG
	if got := r.Delivered(); got != uint64(want) {
		t.Fatalf("Delivered = %d, want %d", got, want)
	}
	if got := r.Dropped(); got != 0 {
		t.Fatalf("Dropped = %d, want 0", got)
	}
	seen := make(map[string]int, want)
	for _, id := range b.allIDs() {
		seen[id]++
	}
	if len(seen) != want {
		t.Fatalf("distinct delivered ids = %d, want %d", len(seen), want)
	}
	for g := 0; g < goroutines; g++ {
		for i := 0; i < perG; i++ {
			id := fmt.Sprintf("g%d-%04d", g, i)
			if seen[id] != 1 {
				t.Fatalf("id %s delivered %d times, want 1", id, seen[id])
			}
		}
	}
}

func TestMemoryBackendDirect(t *testing.T) {
	m := NewMemory()
	if err := m.Append([]billing.UsageRecord{rec("a"), rec("b")}); err != nil {
		t.Fatalf("Append: %v", err)
	}
	recs := m.Records()
	expectIDs(t, idsOf(recs), []string{"a", "b"})

	// Records returns a copy: mutating it must not affect the backend.
	recs[0].EventID = "mutated"
	if got := idsOf(m.Records()); len(got) != 2 || got[0] != "a" {
		t.Fatalf("Records copy semantics broken: %v", got)
	}

	if err := m.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := m.Append([]billing.UsageRecord{rec("c")}); !errors.Is(err, ErrClosed) {
		t.Fatalf("Append after Close: got %v, want ErrClosed", err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("second Close: got %v, want nil (idempotent)", err)
	}
}

func idsOf(recs []billing.UsageRecord) []string {
	out := make([]string, len(recs))
	for i := range recs {
		out[i] = recs[i].EventID
	}
	return out
}
