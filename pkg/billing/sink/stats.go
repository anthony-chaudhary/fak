package sink

import (
	"time"
)

// Stats is a point-in-time observability snapshot of a Ring: enough for
// an operator or exporter to see how much usage is still on its way to
// the durable destination, how much was lost to overflow, and whether
// delivery is making progress at all.
//
// Accounting identity: Accepted = Delivered + Dropped + Rejected + Queued
// holds in every snapshot, not just at quiescence. Accepted counts every
// record handed to Write - accepted into the ring or refused by the
// overflow policy - and each such record ends up in exactly one bucket:
// delivered, dropped, rejected, or still waiting (Queued). Records
// refused because the sink was closed are not counted. Every transition
// updates its counter under the same lock that moves the record between
// buckets, so a snapshot can never show a record in two buckets or in
// none. Queued includes the in-flight prefix the flusher holds mid-
// Append or is retrying after a failed append: those records are
// accepted but not yet delivered, exactly like queued ones.
type Stats struct {
	// Queued is the number of records currently accepted but not yet
	// delivered: those still in the ring plus any batch the flusher
	// holds in flight or is retrying.
	Queued int

	// Accepted is the cumulative number of records handed to Write:
	// those accepted into the ring plus those refused by RejectNew.
	// It is the denominator of the accounting identity above.
	Accepted uint64

	// Delivered is the cumulative number of records the backend has
	// accepted.
	Delivered uint64

	// Dropped is the cumulative number of records DropOldest has
	// discarded to make room for newer ones.
	Dropped uint64

	// Rejected is the cumulative number of Writes RejectNew has refused
	// with ErrFull.
	Rejected uint64

	// FlushFailures is the cumulative number of failed backend Append
	// calls. Failed batches stay held and are retried before any newer
	// record.
	FlushFailures uint64

	// OldestQueuedAge is how long the oldest not-yet-delivered record
	// has waited since it was ENQUEUED, measured on the wall clock;
	// zero when nothing is queued. It measures waiting in this ring,
	// not time since the request itself. A non-monotonic clock can
	// skew it; a negative age reads back as zero.
	OldestQueuedAge time.Duration

	// LastFlushSuccess is the wall-clock time of the last successful
	// backend delivery; zero until the first success. A stale value
	// alongside a growing Queued is the signal that delivery has
	// stopped moving.
	LastFlushSuccess time.Time
}

// Stats reports a point-in-time snapshot of the ring's backlog, loss,
// and delivery health. It is safe for concurrent use and never blocks
// on the flusher or the backend.
//
// Consistency: the whole snapshot - queue state, ages, and the
// cumulative counters - is captured in one lock acquisition. Every
// transition credits its counter under the same lock that moves the
// record between buckets, so the accounting identity documented on
// Stats holds in every returned value, not just at quiescence.
func (r *Ring) Stats() Stats {
	r.mu.Lock()
	defer r.mu.Unlock()
	var oldest time.Time
	switch {
	case len(r.held) > 0:
		oldest = r.heldAt[0]
		// The wall clock can step backwards between enqueues, so take
		// the minimum rather than trusting FIFO order alone.
		if r.count > 0 && r.enqAt[r.head].Before(oldest) {
			oldest = r.enqAt[r.head]
		}
	case r.count > 0:
		oldest = r.enqAt[r.head]
	}
	queued := r.count + len(r.held)
	lastOK := r.lastFlushOK
	var age time.Duration
	if !oldest.IsZero() {
		age = time.Since(oldest)
		if age < 0 {
			age = 0
		}
	}

	return Stats{
		Queued:           queued,
		Accepted:         r.accepted.Load(),
		Delivered:        r.delivered.Load(),
		Dropped:          r.dropped.Load(),
		Rejected:         r.rejected.Load(),
		FlushFailures:    r.failures.Load(),
		OldestQueuedAge:  age,
		LastFlushSuccess: lastOK,
	}
}
