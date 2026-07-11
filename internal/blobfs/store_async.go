package blobfs

// Non-blocking durable writes. store.go's commit()/writeBlob() persist a blob by
// fsync'ing a temp file and atomically renaming it into place — crash-safe, but it
// BLOCKS the Put caller until that rename commit point, and two concurrent
// same-digest Puts each race a full temp-write+fsync+rename (duplicate disk work
// under the same content address). PutAsync is the additive non-blocking path:
//
//  1. it copies the caller's bytes into an OWNED buffer synchronously, so the
//     caller may reuse (or free) its slice the instant PutAsync returns;
//  2. it enqueues the fsync+rename onto a bounded background worker and returns the
//     addressable Ref immediately — the caller never waits on disk;
//  3. an in-flight-digest set (inflight) makes a concurrent same-digest Put COALESCE
//     onto the pending write (a dedup hit) instead of enqueuing a second job.
//
// Durability is deferred, not dropped: a blob written through PutAsync is readable
// once Flush (drain the current backlog) or Close (drain + join the worker) has
// returned. The synchronous Put/commit/writeBlob path is untouched.

import (
	"context"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
)

// PutAsync stores b durably WITHOUT blocking the caller on fsync. Small payloads
// (<= InlineMax) ride inline on the Ref exactly as Put does — already durable in the
// Ref itself, no worker needed. For a larger payload PutAsync copies b into a
// store-owned buffer (the caller may reuse b at once), records the digest as
// in-flight so a concurrent same-digest PutAsync coalesces onto this write rather
// than racing a duplicate, enqueues the fsync+rename onto the bounded background
// worker, and returns the Ref immediately. The blob is guaranteed durable and
// resolvable only after a subsequent Flush or Close; any deferred write error
// surfaces there, not here.
func (s *Store) PutAsync(ctx context.Context, b []byte) (abi.Ref, error) {
	r, inline := blob.PreparePut(b)
	if inline {
		return r, nil
	}
	// Own the bytes synchronously so the caller's slice is free to reuse the moment
	// this returns — the whole point of the non-blocking path.
	owned := append([]byte(nil), b...)

	s.mu.Lock()
	atomic.AddInt64(&s.puts, 1)
	if _, ok := s.index[r.Digest]; ok {
		atomic.AddInt64(&s.hits, 1) // already durable on disk — content dedup
		s.mu.Unlock()
		return r, nil
	}
	if _, pending := s.inflight[r.Digest]; pending {
		atomic.AddInt64(&s.hits, 1) // coalesce onto the pending write — one disk write, not two
		s.mu.Unlock()
		return r, nil
	}
	s.inflight[r.Digest] = struct{}{}
	s.mu.Unlock()

	s.startAsync()
	if s.enqueue(asyncJob{digest: r.Digest, buf: owned}) {
		return r, nil
	}

	// The async pipeline was already Closed (lifecycle end). Fall back to a
	// synchronous durable commit so the Ref we hand back is honest — the bytes are
	// on disk before we return — rather than silently dropping the write.
	s.mu.Lock()
	delete(s.inflight, r.Digest)
	s.mu.Unlock()
	if err := s.writeBlob(r.Digest, owned); err != nil {
		return abi.Ref{}, err
	}
	s.mu.Lock()
	s.indexResidentLocked(r.Digest, int64(len(owned)))
	s.mu.Unlock()
	return r, nil
}

// Flush blocks until every async write enqueued before this call has reached disk,
// WITHOUT stopping the worker (the store stays usable for more PutAsync). It works
// by pushing a barrier fence through the single FIFO worker: when the worker reaches
// the fence, every earlier job has already drained. It returns the first deferred
// write error observed so far, if any.
func (s *Store) Flush() error {
	s.asyncMu.Lock()
	started, closed := s.asyncStarted, s.asyncClosed
	s.asyncMu.Unlock()
	if !started || closed {
		// Nothing running (never went async), or Close already drained everything.
		return s.asyncErr()
	}
	done := make(chan struct{})
	if s.enqueue(asyncJob{barrier: done}) {
		<-done
	}
	return s.asyncErr()
}

// Close drains all pending async writes and JOINS the background worker, so no
// goroutine is left running after it returns. It is idempotent and safe to call on a
// store that never used PutAsync (no worker was spawned — nothing to join). It
// returns the first deferred write error observed across the store's life, if any.
// Callers must not race PutAsync with Close; a PutAsync that loses the race falls
// back to a synchronous durable commit rather than a lost write.
func (s *Store) Close() error {
	s.closeOnce.Do(func() {
		s.asyncMu.Lock()
		s.asyncClosed = true
		started := s.asyncStarted
		if started {
			close(s.asyncJobs) // worker drains the buffered backlog, then range ends
		}
		s.asyncMu.Unlock()
		if started {
			s.asyncWG.Wait() // join: no goroutine leak
		}
	})
	return s.asyncErr()
}

// startAsync spawns the single background worker exactly once, on the first
// PutAsync. A store that never goes async never spawns it. If Close already ran, the
// worker is not spawned (a lost-race PutAsync falls back to a synchronous commit).
func (s *Store) startAsync() {
	s.asyncOnce.Do(func() {
		s.asyncMu.Lock()
		defer s.asyncMu.Unlock()
		if s.asyncClosed {
			return
		}
		s.asyncStarted = true
		s.asyncWG.Add(1)
		go s.asyncWorker()
	})
}

// enqueue sends a job onto the bounded queue, serialized against Close so a send
// never races the channel close (send-on-closed would panic). It reports false when
// the pipeline is already Closed, leaving the caller to fall back. A full queue
// blocks here (backpressure) until the worker frees a slot — the queue is bounded,
// so the backlog cannot grow without limit; the worker receives without asyncMu, so
// a blocked send cannot deadlock the drain.
func (s *Store) enqueue(job asyncJob) bool {
	s.asyncMu.Lock()
	defer s.asyncMu.Unlock()
	if s.asyncClosed {
		return false
	}
	s.asyncJobs <- job
	return true
}

// asyncWorker is the single FIFO durable-write goroutine. It performs each enqueued
// fsync+rename, folds the result into the resident index under s.mu, and clears the
// digest from the in-flight set so a later same-digest Put dedups against the now-
// resident blob. It exits when Close closes the queue and the buffered backlog has
// drained — the join point.
func (s *Store) asyncWorker() {
	s.workerLive.Store(true)
	defer func() {
		s.workerLive.Store(false)
		s.asyncWG.Done()
	}()
	for job := range s.asyncJobs {
		if job.barrier != nil {
			close(job.barrier) // Flush fence: every earlier job has drained (FIFO)
			continue
		}
		if s.beforeWrite != nil {
			s.beforeWrite(job.digest) // test seam: gate/observe the write
		}
		atomic.AddInt64(&s.bgWrites, 1)
		err := s.writeBlob(job.digest, job.buf)

		s.mu.Lock()
		if err != nil {
			s.recordAsyncErr(err)
		} else {
			s.indexResidentLocked(job.digest, int64(len(job.buf)))
		}
		delete(s.inflight, job.digest) // clear AFTER indexing: coalesce → dedup handoff
		s.mu.Unlock()
	}
}

// indexResidentLocked folds a freshly-committed digest of size n into the resident
// index, order, and byte footprint, then runs the byte-budget GC. It is idempotent:
// a digest already indexed (a concurrent commit of the same content, or a dedup hit)
// is left alone so it counts exactly once. Caller holds s.mu. This is the shared
// index-update tail of both the synchronous commit() and the async worker.
func (s *Store) indexResidentLocked(d string, n int64) {
	if _, ok := s.index[d]; ok {
		return
	}
	s.index[d] = n
	s.bytes += n
	if s.pins[d] == 0 {
		s.orderIdx[d] = s.order.PushFront(d)
	}
	s.evictLocked()
}

// recordAsyncErr latches the first deferred write error so Flush/Close can surface
// it (a caller that never waited on disk cannot receive the error inline).
func (s *Store) recordAsyncErr(err error) {
	s.asyncErrMu.Lock()
	if s.firstAsyncErr == nil {
		s.firstAsyncErr = err
	}
	s.asyncErrMu.Unlock()
}

// asyncErr returns the first deferred write error observed, or nil.
func (s *Store) asyncErr() error {
	s.asyncErrMu.Lock()
	defer s.asyncErrMu.Unlock()
	return s.firstAsyncErr
}
