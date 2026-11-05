package sink

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/billing"
)

// Ring is the bounded-ring + background-flusher RecordSink: the
// out-of-process record-sink adapter. Producers call Write from the
// request path; a single flusher goroutine owns every Backend call, in
// batches, in FIFO order.
//
// Memory is bounded regardless of traffic: the ring never grows past
// RingOptions.Capacity records, and at most one in-flight batch of
// BatchSize records may additionally be held by the flusher while a
// backend call is in progress or being retried. Overflow applies the
// configured OverflowPolicy with a typed counter.
//
// Write, Flush, and Close are safe for concurrent use; Close stops the
// flusher goroutine.
type Ring struct {
	b    Backend
	opts RingOptions

	mu       sync.Mutex
	buf      []billing.UsageRecord // fixed-size ring storage
	head     int                   // index of the oldest queued record
	tail     int                   // index of the next free slot
	count    int                   // queued records
	closed   bool
	flushNow bool     // an explicit Flush is pending
	waiters  []waiter // pending Flush/Close completions

	// held is the in-flight prefix: records accepted but not yet
	// delivered - mid-Append or awaiting retry after a failed Append.
	// It is always delivered before any newer record. Guarded by mu;
	// written only by the flusher goroutine.
	held []billing.UsageRecord

	dropped   atomic.Uint64
	rejected  atomic.Uint64
	failures  atomic.Uint64
	delivered atomic.Uint64

	dataSig chan struct{} // edge-triggered "data arrived" wake-up
	closeCh chan struct{} // closed by Close
	stopped chan struct{} // closed by the flusher on exit
}

// waiter completes one Flush or Close call with the drain result.
type waiter struct{ done chan error }

// Compile-time port conformance.
var _ RecordSink = (*Ring)(nil)

// NewRing starts a Ring over b with the given options and launches its
// flusher goroutine. It returns a wrapped ErrInvalidOptions when opts
// fail validation or b is nil.
func NewRing(b Backend, opts RingOptions) (*Ring, error) {
	if b == nil {
		return nil, fmt.Errorf("%w: nil backend", ErrInvalidOptions)
	}
	if opts.Capacity < 1 {
		return nil, fmt.Errorf("%w: capacity %d", ErrInvalidOptions, opts.Capacity)
	}
	if opts.BatchSize < 1 || opts.BatchSize > opts.Capacity {
		return nil, fmt.Errorf("%w: batch size %d for capacity %d", ErrInvalidOptions, opts.BatchSize, opts.Capacity)
	}
	if opts.FlushInterval < 0 {
		return nil, fmt.Errorf("%w: negative flush interval", ErrInvalidOptions)
	}
	if opts.OverflowPolicy != DropOldest && opts.OverflowPolicy != RejectNew {
		return nil, fmt.Errorf("%w: unknown overflow policy %v", ErrInvalidOptions, opts.OverflowPolicy)
	}
	r := &Ring{
		b:       b,
		opts:    opts,
		buf:     make([]billing.UsageRecord, opts.Capacity),
		dataSig: make(chan struct{}, 1),
		closeCh: make(chan struct{}),
		stopped: make(chan struct{}),
	}
	go r.flusher()
	return r, nil
}

// Write enqueues rec for asynchronous delivery. It never blocks on the
// backend: at full capacity the configured OverflowPolicy applies -
// DropOldest discards the oldest queued record (counted by Dropped) and
// accepts rec; RejectNew refuses rec with ErrFull (counted by
// Rejected). Write returns ErrClosed after Close.
//
// Write does not validate rec; producers that need the closed v1
// contract call UsageRecord.Validate before writing.
func (r *Ring) Write(rec billing.UsageRecord) error {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return ErrClosed
	}
	switch r.opts.OverflowPolicy {
	case DropOldest:
		if r.count == cap(r.buf) {
			r.advanceHead(1)
			r.dropped.Add(1)
		}
	case RejectNew:
		if r.count == cap(r.buf) {
			r.mu.Unlock()
			r.rejected.Add(1)
			return ErrFull
		}
	}
	r.buf[r.tail] = rec
	r.tail++
	if r.tail == cap(r.buf) {
		r.tail = 0
	}
	r.count++
	r.mu.Unlock()

	r.signalData()
	return nil
}

// signalData wakes the flusher if data may need delivering. It is
// edge-triggered and non-blocking: at most one wake-up is outstanding.
func (r *Ring) signalData() {
	select {
	case r.dataSig <- struct{}{}:
	default:
	}
}

// Flush forces delivery of everything currently queued and blocks until
// the flusher has appended it. Every record written before Flush
// returns is appended before it returns; records written concurrently
// after the forced drain began may still be pending. It returns the
// last append error of the forced drain, or ErrClosed once Close has
// been called.
func (r *Ring) Flush() error {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return ErrClosed
	}
	w := waiter{done: make(chan error, 1)}
	r.waiters = append(r.waiters, w)
	r.flushNow = true
	r.mu.Unlock()

	r.signalData()
	return <-w.done
}

// Close drains the ring, closes the backend, and stops the flusher
// goroutine. It blocks until the final drain completes and the flusher
// has exited, and returns the joined error of the final drain and the
// backend close, if any. Records left undelivered by a failing backend
// remain visible through Pending. Close is idempotent: later calls
// return nil.
func (r *Ring) Close() error {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return nil
	}
	r.closed = true
	w := waiter{done: make(chan error, 1)}
	r.waiters = append(r.waiters, w)
	r.mu.Unlock()

	close(r.closeCh)
	err := <-w.done
	<-r.stopped
	return err
}

// Dropped reports how many records DropOldest has discarded to make
// room for newer ones.
func (r *Ring) Dropped() uint64 { return r.dropped.Load() }

// Rejected reports how many Writes RejectNew has refused with ErrFull.
func (r *Ring) Rejected() uint64 { return r.rejected.Load() }

// Failures reports how many backend Append calls have failed. Failed
// batches stay held and are retried before any newer record.
func (r *Ring) Failures() uint64 { return r.failures.Load() }

// Delivered reports how many records the backend has accepted.
func (r *Ring) Delivered() uint64 { return r.delivered.Load() }

// Pending reports how many records have been accepted but not yet
// delivered: those still queued in the ring plus any batch the flusher
// currently holds in flight or is retrying after a failed append.
func (r *Ring) Pending() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.count + len(r.held)
}

// advanceHead drops n queued records from the front. Callers hold mu.
func (r *Ring) advanceHead(n int) {
	r.head += n
	r.head %= cap(r.buf)
	r.count -= n
}

// popFront copies the first n queued records (n <= count), in queue
// order, into a fresh slice and removes them from the ring. Callers
// hold mu.
func (r *Ring) popFront(n int) []billing.UsageRecord {
	out := make([]billing.UsageRecord, n)
	pos := r.head
	for i := 0; i < n; i++ {
		out[i] = r.buf[pos]
		pos++
		if pos == cap(r.buf) {
			pos = 0
		}
	}
	r.advanceHead(n)
	return out
}

// flusher is the sole goroutine that calls the backend. It delivers a
// batch when the ring holds BatchSize records, when the flush interval
// elapses with anything queued, when an explicit Flush asks for it, and
// on Close; otherwise it waits between wake-ups.
func (r *Ring) flusher() {
	defer close(r.stopped)

	var timerC <-chan time.Time
	var timer *time.Timer
	if r.opts.FlushInterval > 0 {
		timer = time.NewTimer(r.opts.FlushInterval)
		defer timer.Stop()
		timerC = timer.C
	}

	for {
		select {
		case <-r.closeCh:
			r.mu.Lock()
			waiters := r.waiters
			r.waiters = nil
			r.flushNow = false
			r.mu.Unlock()
			err := r.drainAll()
			if cerr := r.b.Close(); cerr != nil {
				err = joinErrors(err, cerr)
			}
			complete(waiters, err)
			return

		case <-timerC:
			r.drainPass(true)
			r.resetTimer(timer)

		case <-r.dataSig:
			r.drainPass(false)
		}
	}
}

// drainPass delivers whatever the current state warrants and answers
// any pending Flush requests taken with the state. force=true (timer
// expiry) delivers a partial batch; force=false (data wake-up) delivers
// only when an explicit Flush is pending or the ring holds a full
// batch.
func (r *Ring) drainPass(force bool) {
	r.mu.Lock()
	forced := r.flushNow
	r.flushNow = false
	waiters := r.waiters
	r.waiters = nil
	// Once Close has begun, every pass must drain fully: a stale data
	// wake-up may reach here before the terminal close branch, and it
	// must answer pending waiters with the real drain result, not nil.
	closed := r.closed
	worthIt := force || forced || closed || r.count >= r.opts.BatchSize
	r.mu.Unlock()

	var err error
	if worthIt {
		err = r.drainAll()
	}
	complete(waiters, err)
}

// drainAll appends every queued record to the backend in FIFO order, in
// chunks of at most BatchSize records. A chunk is popped from the ring
// before its Append and kept in held until the Append succeeds, so a
// failed chunk is neither lost nor reordered: it is retried before any
// newer record. It returns the last append error, if any.
func (r *Ring) drainAll() error {
	var lastErr error
	for {
		r.mu.Lock()
		var chunk []billing.UsageRecord
		if len(r.held) > 0 {
			n := len(r.held)
			if n > r.opts.BatchSize {
				n = r.opts.BatchSize
			}
			chunk = r.held[:n]
		} else if r.count > 0 {
			n := r.count
			if n > r.opts.BatchSize {
				n = r.opts.BatchSize
			}
			chunk = r.popFront(n)
			// Claim the chunk as in-flight under the lock so Pending
			// counts it while its Append is in progress.
			r.held = append(r.held, chunk...)
		}
		r.mu.Unlock()

		if chunk == nil {
			break
		}

		if err := r.b.Append(chunk); err != nil {
			r.failures.Add(1)
			lastErr = err
			// The chunk stays at the front of held for retry.
			break
		}
		r.mu.Lock()
		r.held = r.held[len(chunk):]
		r.mu.Unlock()
		r.delivered.Add(uint64(len(chunk)))
	}
	return lastErr
}

// resetTimer re-arms the flush-interval timer after a tick.
func (r *Ring) resetTimer(t *time.Timer) {
	if !t.Stop() {
		select {
		case <-t.C:
		default:
		}
	}
	t.Reset(r.opts.FlushInterval)
}

// complete answers every pending waiter with err.
func complete(waiters []waiter, err error) {
	for _, w := range waiters {
		w.done <- err
	}
}

// joinErrors combines two errors, preserving either one alone.
func joinErrors(a, b error) error {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	return errors.Join(a, b)
}
