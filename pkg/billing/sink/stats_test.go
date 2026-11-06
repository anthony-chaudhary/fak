package sink

import (
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

// assertIdentity checks the Stats accounting identity on a snapshot:
// every accepted record sits in exactly one bucket - delivered,
// dropped, rejected, or still queued.
func assertIdentity(t *testing.T, s Stats) {
	t.Helper()
	want := s.Delivered + s.Dropped + s.Rejected + uint64(s.Queued)
	if s.Accepted != want {
		t.Fatalf("accounting identity broken: Accepted = %d, want Delivered+Dropped+Rejected+Queued = %d (%+v)", s.Accepted, want, s)
	}
}

// TestStatsCountsQueuedAndAccepted checks the basic counters with the
// timer disabled: below batch size nothing is delivered, so every
// accepted record is still queued.
func TestStatsCountsQueuedAndAccepted(t *testing.T) {
	m := NewMemory()
	r, err := NewRing(m, RingOptions{Capacity: 8, BatchSize: 4, OverflowPolicy: DropOldest})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	const n = 3
	for i := 0; i < n; i++ {
		if err := r.Write(rec(fmt.Sprintf("w%d", i))); err != nil {
			t.Fatalf("write %d: %v", i, err)
		}
	}
	s := r.Stats()
	if s.Queued != n {
		t.Fatalf("Queued = %d, want %d (partial batch must stay queued)", s.Queued, n)
	}
	if s.Accepted != uint64(n) {
		t.Fatalf("Accepted = %d, want %d", s.Accepted, n)
	}
	if s.Delivered != 0 || s.Dropped != 0 || s.Rejected != 0 || s.FlushFailures != 0 {
		t.Fatalf("unexpected counters: %+v", s)
	}
	if got := len(m.Records()); got != 0 {
		t.Fatalf("backend received %d records, want 0 (timer disabled, below batch size)", got)
	}
	assertIdentity(t, s)
}

// TestStatsAccountingIdentity drives every loss and delivery path -
// overflow drops, overflow rejections, forced flushes, and a
// persistently failing backend - and checks the accounting identity
// after each transition.
func TestStatsAccountingIdentity(t *testing.T) {
	// DropOldest: two overflow drops behind a pinned in-flight batch.
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
	<-hb.entered // [1,2] pinned in flight
	for _, id := range []string{"3", "4", "5", "6"} {
		if err := r.Write(rec(id)); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}
	// The ring is full: writes 7 and 8 each discard the oldest queued
	// record to make room.
	for _, id := range []string{"7", "8"} {
		if err := r.Write(rec(id)); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}
	s := r.Stats()
	if s.Dropped != 2 {
		t.Fatalf("Dropped = %d, want 2 (records 3 and 4 discarded)", s.Dropped)
	}
	if s.Queued != 6 {
		t.Fatalf("Queued = %d, want 6 (4 in the ring + 2 in flight)", s.Queued)
	}
	assertIdentity(t, s)

	hb.releaseAll()
	if err := r.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	s = r.Stats()
	if s.Delivered != 6 {
		t.Fatalf("Delivered = %d, want 6", s.Delivered)
	}
	if s.Queued != 0 {
		t.Fatalf("Queued after drain = %d, want 0", s.Queued)
	}
	assertIdentity(t, s)

	// RejectNew: an overflow refusal is counted in both Accepted and
	// Rejected, and the refused record never enters any bucket.
	hb2 := newHoldBackend()
	r2, err := NewRing(hb2, RingOptions{Capacity: 4, BatchSize: 2, OverflowPolicy: RejectNew})
	if err != nil {
		t.Fatal(err)
	}
	defer r2.Close()
	defer hb2.releaseAll()

	for _, id := range []string{"1", "2"} {
		if err := r2.Write(rec(id)); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}
	<-hb2.entered // [1,2] pinned in flight
	for _, id := range []string{"3", "4", "5", "6"} {
		if err := r2.Write(rec(id)); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}
	if err := r2.Write(rec("7")); !errors.Is(err, ErrFull) {
		t.Fatalf("write 7 under RejectNew: got %v, want ErrFull", err)
	}
	s = r2.Stats()
	if s.Rejected != 1 {
		t.Fatalf("Rejected = %d, want 1", s.Rejected)
	}
	if s.Accepted != 7 {
		t.Fatalf("Accepted = %d, want 7 (six enqueued + one refused)", s.Accepted)
	}
	if s.Queued != 6 {
		t.Fatalf("Queued = %d, want 6 (refused record not enqueued)", s.Queued)
	}
	assertIdentity(t, s)

	hb2.releaseAll()
	if err := r2.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	s = r2.Stats()
	if s.Delivered != 6 {
		t.Fatalf("Delivered = %d, want 6", s.Delivered)
	}
	assertIdentity(t, s)

	// Failing backend: the failed batch stays held, so it keeps
	// counting as queued, and no delivery credit ever lands.
	fb := newChanBackend()
	fb.failAll = true
	r3, err := NewRing(fb, RingOptions{Capacity: 4, BatchSize: 4, OverflowPolicy: DropOldest})
	if err != nil {
		t.Fatal(err)
	}
	defer r3.Close()

	for _, id := range []string{"a", "b"} {
		if err := r3.Write(rec(id)); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}
	if err := r3.Flush(); !errors.Is(err, fb.failErr) {
		t.Fatalf("first Flush: got %v, want the backend failure", err)
	}
	s = r3.Stats()
	if s.Delivered != 0 {
		t.Fatalf("Delivered = %d, want 0 (every append fails)", s.Delivered)
	}
	if s.Queued != 2 {
		t.Fatalf("Queued after failed Flush = %d, want 2 (failed batch stays held)", s.Queued)
	}
	assertIdentity(t, s)

	// Below batch size the ring cannot auto-drain, so these writes
	// stay queued deterministically alongside the held batch.
	for _, id := range []string{"c", "d", "e"} {
		if err := r3.Write(rec(id)); err != nil {
			t.Fatalf("write %s: %v", id, err)
		}
	}
	s = r3.Stats()
	if s.Queued != 5 {
		t.Fatalf("Queued = %d, want 5 (2 held + 3 in the ring)", s.Queued)
	}
	assertIdentity(t, s)
}

// TestStatsOldestQueuedAge checks the age lifecycle: zero when empty,
// growing with wall-clock waiting, and surviving the ring-to-held
// handoff with the original enqueue time intact.
func TestStatsOldestQueuedAge(t *testing.T) {
	m := NewMemory()
	r, err := NewRing(m, RingOptions{Capacity: 4, BatchSize: 4, OverflowPolicy: DropOldest})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	if got := r.Stats().OldestQueuedAge; got != 0 {
		t.Fatalf("OldestQueuedAge on an empty ring = %v, want 0", got)
	}

	if err := r.Write(rec("a")); err != nil {
		t.Fatalf("write: %v", err)
	}
	time.Sleep(15 * time.Millisecond)
	if got := r.Stats().OldestQueuedAge; got < 10*time.Millisecond {
		t.Fatalf("OldestQueuedAge after a 15ms wait = %v, want >= 10ms", got)
	}

	// A failing backend moves the record ring-to-held on the forced
	// flush; the age must keep counting from the original enqueue,
	// not restart at the handoff.
	fb := newChanBackend()
	fb.failAll = true
	rf, err := NewRing(fb, RingOptions{Capacity: 4, BatchSize: 4, OverflowPolicy: DropOldest})
	if err != nil {
		t.Fatal(err)
	}
	if err := rf.Write(rec("b")); err != nil {
		t.Fatalf("write: %v", err)
	}
	time.Sleep(15 * time.Millisecond)
	if err := rf.Flush(); !errors.Is(err, fb.failErr) {
		t.Fatalf("Flush: got %v, want the backend failure", err)
	}
	s := rf.Stats()
	if s.Queued != 1 {
		t.Fatalf("Queued after failed Flush = %d, want 1 (failed batch stays held)", s.Queued)
	}
	if s.OldestQueuedAge < 10*time.Millisecond {
		t.Fatalf("OldestQueuedAge after the ring-to-held handoff = %v, want >= 10ms (the original enqueue time must be carried)", s.OldestQueuedAge)
	}
	if err := rf.Close(); !errors.Is(err, fb.failErr) {
		t.Fatalf("Close: got %v, want the backend failure", err)
	}
}

// TestStatsLastFlushSuccess checks the success stamp: zero before any
// delivery, set within the observed window after a successful Flush,
// and untouched when every append fails.
func TestStatsLastFlushSuccess(t *testing.T) {
	m := NewMemory()
	r, err := NewRing(m, RingOptions{Capacity: 4, BatchSize: 4, OverflowPolicy: DropOldest})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	if ts := r.Stats().LastFlushSuccess; !ts.IsZero() {
		t.Fatalf("LastFlushSuccess before any delivery = %v, want zero", ts)
	}

	if err := r.Write(rec("a")); err != nil {
		t.Fatalf("write: %v", err)
	}
	before := time.Now()
	if err := r.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	after := time.Now()
	ts := r.Stats().LastFlushSuccess
	if ts.IsZero() {
		t.Fatal("LastFlushSuccess after a successful Flush = zero, want set")
	}
	// Closed window: the flush can complete within the same clock tick
	// as the before sample, so strict inequality on the lower bound
	// would flake.
	if ts.Before(before) || ts.After(after) {
		t.Fatalf("LastFlushSuccess = %v, want within [%v, %v]", ts, before, after)
	}

	fb := newChanBackend()
	fb.failAll = true
	rf, err := NewRing(fb, RingOptions{Capacity: 4, BatchSize: 4, OverflowPolicy: DropOldest})
	if err != nil {
		t.Fatal(err)
	}
	defer rf.Close()
	if err := rf.Write(rec("a")); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := rf.Flush(); !errors.Is(err, fb.failErr) {
		t.Fatalf("Flush: got %v, want the backend failure", err)
	}
	if ts := rf.Stats().LastFlushSuccess; !ts.IsZero() {
		t.Fatalf("LastFlushSuccess after all-failed appends = %v, want zero", ts)
	}
}

// TestStatsConcurrentWritersReadersClose hammers the ring from several
// writers while reader goroutines sample Stats on every snapshot and
// check the accounting identity; meant to run under -race.
func TestStatsConcurrentWritersReadersClose(t *testing.T) {
	const (
		writers   = 4
		perWriter = 500
		readers   = 4
		samples   = 300
	)
	b := newChanBackend()
	// Capacity exceeds the total number of records written below, so
	// overflow is impossible regardless of scheduling: the exact-total
	// assertions are deterministic, not probabilistic.
	r, err := NewRing(b, RingOptions{Capacity: 8192, BatchSize: 1024, OverflowPolicy: DropOldest})
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < perWriter; i++ {
				id := fmt.Sprintf("w%d-%03d", w, i)
				if err := r.Write(rec(id)); err != nil {
					t.Errorf("write %s: %v", id, err)
					return
				}
			}
		}(w)
	}
	for rd := 0; rd < readers; rd++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < samples; i++ {
				assertIdentity(t, r.Stats())
			}
		}()
	}
	wg.Wait()
	if err := r.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	want := uint64(writers * perWriter)
	if got := r.Delivered(); got != want {
		t.Fatalf("Delivered = %d, want %d", got, want)
	}
	if got := r.Stats().Accepted; got != want {
		t.Fatalf("Accepted = %d, want %d", got, want)
	}
	assertIdentity(t, r.Stats())
}
