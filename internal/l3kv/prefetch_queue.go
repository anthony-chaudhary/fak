package l3kv

import (
	"context"
	"errors"
	"sync"
	"time"
)

// PrefetchQueue is the read side of AsyncStore: queued on-disk KV states are
// read ahead on separate I/O workers while other streams run, so a later
// restore finds its span already in memory instead of paying the disk read at
// admission time. Adapted from TensorFold's lane-host fetch contract
// (zig/src/core/lane_host_fetch.zig + lane_host_admit.zig:13, docs/api.md:661
// @9e8c368b0b4c2c7892a0c9f0f302d7c164016209, Apache-2.0): held reads wait with
// a bounded timeout and fall back to admission restore or fresh prefill;
// concurrent-read permits are capped; a cancelled waiter does not cancel the
// read — it keeps its permit until the I/O returns; Stop cancels reads.
const (
	// DefaultPrefetchTimeout bounds how long a held read waits before the
	// caller falls back to admission restore or fresh prefill.
	DefaultPrefetchTimeout = 30 * time.Second
	// DefaultPrefetchMaxConcurrentReads caps in-flight disk reads. Zero
	// disables read-ahead entirely.
	DefaultPrefetchMaxConcurrentReads = 2
)

var (
	// ErrPrefetchDisabled reports that read-ahead is off (zero permits); the
	// demand path still works through Restore's direct read.
	ErrPrefetchDisabled = errors.New("l3kv: prefetch disabled")
	// ErrPrefetchStopped reports that the queue was stopped; queued and
	// in-flight reads were cancelled.
	ErrPrefetchStopped = errors.New("l3kv: prefetch stopped")
	// ErrPrefetchTimeout reports that a held read outlived its bound; the
	// caller falls back to admission restore or fresh prefill. The read
	// itself may still complete and stage afterward.
	ErrPrefetchTimeout = errors.New("l3kv: prefetch read timed out")
)

// PrefetchConfig tunes a PrefetchQueue. Timeout <= 0 takes
// DefaultPrefetchTimeout; MaxConcurrentReads <= 0 disables read-ahead
// entirely (negative values are refused as invalid and treated as zero);
// MaxStagedBytes <= 0 means no resident-memory cap. Use
// DefaultPrefetchConfig for the default-on settings.
type PrefetchConfig struct {
	Timeout            time.Duration
	MaxConcurrentReads int
	MaxStagedBytes     int64
}

// DefaultPrefetchConfig is the default-on read-ahead setting: the 30 s hold
// bound and 2 concurrent-read permits.
var DefaultPrefetchConfig = PrefetchConfig{
	Timeout:            DefaultPrefetchTimeout,
	MaxConcurrentReads: DefaultPrefetchMaxConcurrentReads,
}

func (c PrefetchConfig) normalized() PrefetchConfig {
	if c.Timeout <= 0 {
		c.Timeout = DefaultPrefetchTimeout
	}
	if c.MaxConcurrentReads < 0 {
		c.MaxConcurrentReads = 0
	}
	return c
}

// PrefetchStats is a point-in-time snapshot of queue pressure.
type PrefetchStats struct {
	Queued        int   // entries accepted but not yet settled
	InFlight      int   // reads currently inside Store.Get (held permits)
	StagedEntries int   // settled hits whose payload is held in memory
	StagedBytes   int64 // resident bytes held for later restore
	DroppedFull   int   // enqueues skipped because the job channel was full
	DroppedCap    int   // completed reads dropped by the memory cap
	Completed     int   // settled clean reads (hits plus misses)
	Faults        int   // settled I/O or integrity faults
}

type prefetchOutcomeKind int

const (
	outcomeHit     prefetchOutcomeKind = iota // payload staged under the key
	outcomeMiss                               // clean store miss
	outcomeFault                              // store returned an error
	outcomeStopped                            // cancelled by Stop
	outcomeSkipped                            // never read (queue full); read directly
)

type prefetchOutcome struct {
	kind    prefetchOutcomeKind
	payload []byte
	err     error
}

type prefetchEntry struct {
	outcome prefetchOutcome
	settled bool
	done    chan struct{}
}

// PrefetchQueue performs bounded, cancellable read-ahead over a Store.
// It is safe for concurrent use.
type PrefetchQueue struct {
	store Store
	cfg   PrefetchConfig

	jobs chan string
	stop chan struct{}
	wg   sync.WaitGroup

	mu             sync.Mutex
	stopped        bool
	workersStarted bool
	entries        map[string]*prefetchEntry
	inFlight       int
	stagedBytes    int64
	stats          PrefetchStats
}

// NewPrefetchQueue builds a queue over store. Pass DefaultPrefetchConfig for
// the default-on settings; a zero-permit config never starts any workers.
func NewPrefetchQueue(store Store, cfg PrefetchConfig) *PrefetchQueue {
	cfg = cfg.normalized()
	return &PrefetchQueue{
		store:   store,
		cfg:     cfg,
		jobs:    make(chan string, 1024),
		stop:    make(chan struct{}),
		entries: make(map[string]*prefetchEntry),
	}
}

// Enqueue schedules a read-ahead of key's durable state. It is best-effort:
// a duplicate or a full job channel skips silently (the demand path always
// remains available through Restore). It returns ErrPrefetchDisabled when
// read-ahead is off and ErrPrefetchStopped after Stop.
func (q *PrefetchQueue) Enqueue(key string) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.stopped {
		return ErrPrefetchStopped
	}
	if q.cfg.MaxConcurrentReads == 0 {
		return ErrPrefetchDisabled
	}
	if _, ok := q.entries[key]; ok {
		return nil
	}
	q.entries[key] = &prefetchEntry{done: make(chan struct{})}
	q.stats.Queued++
	q.startWorkersLocked()
	select {
	case q.jobs <- key:
	default:
		// Channel full: settle as skipped so Restore reads directly.
		q.settleLocked(q.entries[key], prefetchOutcome{kind: outcomeSkipped})
		delete(q.entries, key)
		q.stats.DroppedFull++
	}
	return nil
}

// Restore returns the read-ahead payload for key. A staged hit is returned
// immediately; an unsettled read is held at most cfg.Timeout (or the caller's
// context deadline, whichever is earlier) and then refused with
// ErrPrefetchTimeout so the caller falls back to admission restore or fresh
// prefill — the read keeps running and its permit is kept until the I/O
// returns. A key that was never read ahead is read directly from the store.
func (q *PrefetchQueue) Restore(ctx context.Context, key string) ([]byte, bool, error) {
	q.mu.Lock()
	entry, ok := q.entries[key]
	if !ok || entry.settled {
		var outcome prefetchOutcome
		if ok {
			outcome = entry.outcome
			q.releaseSettledLocked(entry)
			delete(q.entries, key)
		}
		q.mu.Unlock()
		if ok && outcome.kind != outcomeSkipped {
			switch outcome.kind {
			case outcomeHit:
				return outcome.payload, true, nil
			case outcomeMiss:
				return nil, false, nil
			}
			return nil, false, outcome.err // fault or stopped
		}
		return q.getDirect(ctx, key)
	}
	q.mu.Unlock()

	timer := time.NewTimer(q.holdFor(ctx))
	defer timer.Stop()
	select {
	case <-entry.done:
	case <-timer.C:
		if err := ctx.Err(); err != nil {
			return nil, false, err
		}
		return nil, false, ErrPrefetchTimeout
	case <-ctx.Done():
		return nil, false, ctx.Err()
	}

	q.mu.Lock()
	entry, ok = q.entries[key]
	if !ok || entry.settled {
		var outcome prefetchOutcome
		if ok {
			outcome = entry.outcome
			q.releaseSettledLocked(entry)
			delete(q.entries, key)
		}
		q.mu.Unlock()
		if !ok {
			// Another waiter consumed the result first; read directly.
			return q.getDirect(ctx, key)
		}
		switch outcome.kind {
		case outcomeHit:
			return outcome.payload, true, nil
		case outcomeMiss:
			return nil, false, nil
		}
		return nil, false, outcome.err
	}
	q.mu.Unlock()
	return q.getDirect(ctx, key)
}

// holdFor is the remaining hold budget: the configured timeout shortened by
// the caller's deadline when it lands sooner.
func (q *PrefetchQueue) holdFor(ctx context.Context) time.Duration {
	hold := q.cfg.Timeout
	if d, ok := ctx.Deadline(); ok {
		if sooner := time.Until(d); sooner < hold {
			if sooner <= 0 {
				return 0
			}
			hold = sooner
		}
	}
	return hold
}

// Stop cancels queued and in-flight reads. In-flight reads keep their permit
// until the underlying I/O returns; queued entries settle as
// ErrPrefetchStopped, which Restore reports to waiters until they consume it,
// and staged hit payloads are released. Stop is idempotent.
//
// The stopped flag is raised under q.mu before Wait: Enqueue only starts a
// worker (wg.Add) after observing stopped == false under that same lock, so
// every Add is sequenced before this Wait and none can start after it.
func (q *PrefetchQueue) Stop() {
	q.mu.Lock()
	if q.stopped {
		q.mu.Unlock()
		return
	}
	q.stopped = true
	close(q.stop)
	q.mu.Unlock()
	q.wg.Wait()
	q.mu.Lock()
	defer q.mu.Unlock()
	for key, entry := range q.entries {
		switch {
		case !entry.settled:
			q.settleLocked(entry, prefetchOutcome{kind: outcomeStopped, err: ErrPrefetchStopped})
		case entry.outcome.kind == outcomeHit:
			q.releaseSettledLocked(entry)
			delete(q.entries, key)
		}
	}
}

// Stats snapshots current queue pressure.
func (q *PrefetchQueue) Stats() PrefetchStats {
	q.mu.Lock()
	defer q.mu.Unlock()
	s := q.stats
	s.InFlight = q.inFlight
	s.StagedBytes = q.stagedBytes
	return s
}

// getDirect is the fallback demand read: no read-ahead happened (or was
// skipped), so pay the disk read now.
func (q *PrefetchQueue) getDirect(ctx context.Context, key string) ([]byte, bool, error) {
	if q.store == nil {
		return nil, false, nil
	}
	return q.store.Get(ctx, key)
}

func (q *PrefetchQueue) startWorkersLocked() {
	if q.cfg.MaxConcurrentReads == 0 || q.workersStarted {
		return
	}
	q.workersStarted = true
	for i := 0; i < q.cfg.MaxConcurrentReads; i++ {
		q.wg.Add(1)
		go q.worker()
	}
}

// worker holds one concurrent-read permit for its whole life: it runs a Get to
// completion before taking the next job, which is what bounds in-flight reads
// to MaxConcurrentReads and what makes a cancelled waiter keep its permit
// until the I/O returns. A job taken after Stop has raised the flag is
// cancelled without I/O; Stop's drain settles its entry as stopped.
func (q *PrefetchQueue) worker() {
	defer q.wg.Done()
	for {
		select {
		case <-q.stop:
			return
		case key := <-q.jobs:
			q.mu.Lock()
			if q.stopped {
				q.mu.Unlock()
				return
			}
			q.mu.Unlock()
			q.runRead(key)
		}
	}
}

func (q *PrefetchQueue) runRead(key string) {
	q.mu.Lock()
	entry, ok := q.entries[key]
	if !ok || entry.settled {
		q.mu.Unlock()
		return
	}
	q.inFlight++
	q.mu.Unlock()

	var payload []byte
	var found bool
	var err error
	if q.store != nil {
		payload, found, err = q.store.Get(context.Background(), key)
	}

	q.mu.Lock()
	defer q.mu.Unlock()
	q.inFlight--
	switch {
	case !ok || entry.settled:
		// Cancelled or consumed elsewhere; nothing to settle.
	case err != nil:
		q.settleLocked(entry, prefetchOutcome{kind: outcomeFault, err: err})
		q.stats.Faults++
	case found && q.overStagedCap(int64(len(payload))):
		// Memory cap: drop the payload rather than hold it; the reader sees a
		// clean miss and falls back to admission restore or fresh prefill.
		q.settleLocked(entry, prefetchOutcome{kind: outcomeMiss})
		q.stats.DroppedCap++
	case found:
		q.stagedBytes += int64(len(payload))
		q.stats.StagedEntries++
		q.settleLocked(entry, prefetchOutcome{kind: outcomeHit, payload: payload})
		q.stats.Completed++
	default:
		q.settleLocked(entry, prefetchOutcome{kind: outcomeMiss})
		q.stats.Completed++
	}
}

func (q *PrefetchQueue) overStagedCap(additional int64) bool {
	return q.cfg.MaxStagedBytes > 0 && q.stagedBytes+additional > q.cfg.MaxStagedBytes
}

// releaseSettledLocked returns a staged payload to the free pool. Callers hold
// q.mu and have removed the entry from consideration.
func (q *PrefetchQueue) releaseSettledLocked(entry *prefetchEntry) {
	if entry.outcome.kind == outcomeHit {
		q.stagedBytes -= int64(len(entry.outcome.payload))
		q.stats.StagedEntries--
	}
}

// settleLocked records the first outcome for entry and wakes waiters. Callers
// hold q.mu.
func (q *PrefetchQueue) settleLocked(entry *prefetchEntry, outcome prefetchOutcome) {
	if entry.settled {
		return
	}
	entry.outcome = outcome
	entry.settled = true
	q.stats.Queued--
	close(entry.done)
}
