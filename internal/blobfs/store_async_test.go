package blobfs

import (
	"bytes"
	"context"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestPutAsyncReturnsBeforeFsyncThenDurable is the core non-blocking contract:
// PutAsync returns while the fsync+rename is still pending (the worker is gated
// before writeBlob, so nothing is on disk yet), and the blob becomes durably
// readable once Flush drains the backlog.
func TestPutAsyncReturnsBeforeFsyncThenDurable(t *testing.T) {
	ctx := context.Background()
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	release := make(chan struct{})
	entered := make(chan struct{}, 1)
	s.beforeWrite = func(string) {
		select {
		case entered <- struct{}{}:
		default:
		}
		<-release
	}

	want := payload(4096, 'a') // > InlineMax → async disk write
	r, err := s.PutAsync(ctx, want)
	if err != nil {
		t.Fatalf("PutAsync: %v", err)
	}

	<-entered // the worker is inside the write, gated BEFORE writeBlob
	if _, err := os.Stat(s.pathFor(r.Digest)); err == nil {
		t.Fatalf("blob is on disk before the fsync/drain — PutAsync did not defer the write")
	}
	if cnt, _, _ := s.Resident(); cnt != 0 {
		t.Fatalf("blob indexed before drain: %d resident, want 0", cnt)
	}

	close(release) // let the deferred fsync+rename run
	if err := s.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	got, err := s.Resolve(ctx, r)
	if err != nil {
		t.Fatalf("Resolve after Flush: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("resolved bytes differ from the async-stored payload")
	}
	if _, err := os.Stat(s.pathFor(r.Digest)); err != nil {
		t.Fatalf("blob not durable on disk after Flush: %v", err)
	}
}

// TestPutAsyncCallerMayReuseSlice proves PutAsync copies into an owned buffer: the
// caller mutates its slice the instant PutAsync returns, yet the stored blob still
// resolves to the ORIGINAL bytes after Flush.
func TestPutAsyncCallerMayReuseSlice(t *testing.T) {
	ctx := context.Background()
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	buf := payload(4096, 'o')
	want := append([]byte(nil), buf...)
	r, err := s.PutAsync(ctx, buf)
	if err != nil {
		t.Fatalf("PutAsync: %v", err)
	}
	for i := range buf { // reuse the caller slice immediately
		buf[i] = 'Z'
	}
	if err := s.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	got, err := s.Resolve(ctx, r)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("stored blob observed the caller's post-Put mutation — bytes were not owned")
	}
}

// TestPutAsyncCoalescesConcurrentSameDigest proves the in-flight set coalesces a
// concurrent same-digest Put onto the pending write: exactly ONE disk write happens,
// even though two Puts of identical bytes are issued while the first is still in
// flight (gated, provably not yet indexed).
func TestPutAsyncCoalescesConcurrentSameDigest(t *testing.T) {
	ctx := context.Background()
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	release := make(chan struct{})
	entered := make(chan struct{}, 1)
	s.beforeWrite = func(string) {
		select {
		case entered <- struct{}{}:
		default:
		}
		<-release
	}

	b := payload(4096, 'c')
	r1, err := s.PutAsync(ctx, b)
	if err != nil {
		t.Fatalf("PutAsync 1: %v", err)
	}
	<-entered // first write is in flight (dequeued, gated) — NOT yet indexed
	if cnt, _, _ := s.Resident(); cnt != 0 {
		t.Fatalf("first write indexed before drain: %d resident, want 0", cnt)
	}

	// Same digest while the first is in flight → must coalesce (no second job).
	r2, err := s.PutAsync(ctx, b)
	if err != nil {
		t.Fatalf("PutAsync 2: %v", err)
	}
	if r2.Digest != r1.Digest {
		t.Fatalf("same bytes produced different digests: %q vs %q", r2.Digest, r1.Digest)
	}
	if _, hits, _ := s.Stats(); hits != 1 {
		t.Fatalf("coalesce should count 1 dedup hit, got %d", hits)
	}

	close(release)
	if err := s.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if got := atomic.LoadInt64(&s.bgWrites); got != 1 {
		t.Fatalf("coalesce failed: %d background disk writes, want exactly 1", got)
	}
	if cnt, _, _ := s.Resident(); cnt != 1 {
		t.Fatalf("coalesce failed: %d resident blobs, want 1", cnt)
	}
	got, err := s.Resolve(ctx, r1)
	if err != nil || !bytes.Equal(got, b) {
		t.Fatalf("coalesced blob did not resolve: err=%v equal=%v", err, bytes.Equal(got, b))
	}
}

// TestCloseDrainsAndJoins proves Close drains every pending async write and joins the
// worker goroutine (no leak): all blobs are durable after Close, the worker has
// exited, and a second Close is a no-op.
func TestCloseDrainsAndJoins(t *testing.T) {
	ctx := context.Background()
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	release := make(chan struct{})
	s.beforeWrite = func(string) { <-release }

	const n = 5
	refs := make([]abi.Ref, n)
	bodies := make([][]byte, n)
	for i := 0; i < n; i++ {
		body := payload(4096, byte(i)) // distinct fill → distinct digest, no coalescing
		r, err := s.PutAsync(ctx, body)
		if err != nil {
			t.Fatalf("PutAsync %d: %v", i, err)
		}
		refs[i], bodies[i] = r, body
	}

	close(release) // unblock all pending writes
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if cnt, _, _ := s.Resident(); cnt != n {
		t.Fatalf("Close did not drain: %d resident, want %d", cnt, n)
	}
	if got := atomic.LoadInt64(&s.bgWrites); got != n {
		t.Fatalf("Close did not drain: %d background writes, want %d", got, n)
	}
	if s.workerLive.Load() {
		t.Fatalf("Close did not join the worker goroutine (still live)")
	}
	for i := range refs {
		got, err := s.Resolve(ctx, refs[i])
		if err != nil || !bytes.Equal(got, bodies[i]) {
			t.Fatalf("blob %d not durable after Close: err=%v", i, err)
		}
	}
	if err := s.Close(); err != nil { // idempotent
		t.Fatalf("second Close: %v", err)
	}
}

// TestAsyncQueueIsBounded proves the pending-write queue does not grow without
// limit: with the worker stalled, only queue-capacity + 1 (the one the worker
// dequeued) PutAsync calls can complete their enqueue; the rest block on
// backpressure until the worker drains. All writes still land once released.
func TestAsyncQueueIsBounded(t *testing.T) {
	ctx := context.Background()
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// Shrink the bound BEFORE the worker starts (no PutAsync issued yet).
	const cap2 = 2
	s.asyncJobs = make(chan asyncJob, cap2)
	s.asyncCap = cap2
	defer s.Close()

	release := make(chan struct{})
	s.beforeWrite = func(string) { <-release }

	if cap(s.asyncJobs) != cap2 {
		t.Fatalf("queue cap = %d, want the bounded %d", cap(s.asyncJobs), cap2)
	}

	const total = 6
	done := make(chan struct{}, total)
	for i := 0; i < total; i++ {
		go func(i int) {
			if _, err := s.PutAsync(ctx, payload(4096, byte(i))); err != nil {
				t.Errorf("PutAsync %d: %v", i, err)
			}
			done <- struct{}{}
		}(i)
	}

	// Worker dequeues 1 (then gates) + buffer holds cap2 ⇒ exactly cap2+1 enqueues
	// can complete while the worker is stalled; the remaining goroutines block.
	const admitted = cap2 + 1
	for i := 0; i < admitted; i++ {
		<-done
	}
	time.Sleep(100 * time.Millisecond) // let any (erroneous) extra completion surface
	if extra := len(done); extra != 0 {
		t.Fatalf("queue admitted more than cap+1 while the worker was stalled (extra=%d) — not bounded", extra)
	}

	close(release) // drain the backpressured writes
	for i := 0; i < total-admitted; i++ {
		<-done
	}
	if err := s.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if cnt, _, _ := s.Resident(); cnt != total {
		t.Fatalf("not all backpressured writes landed: %d resident, want %d", cnt, total)
	}
	if got := atomic.LoadInt64(&s.bgWrites); got != total {
		t.Fatalf("background writes = %d, want %d", got, total)
	}
}
