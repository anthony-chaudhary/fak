package sink

import (
	"errors"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/billing"
)

// RecordSink is the stable out-of-process record-sink port: the slot a
// serving engine fills with per-request usage records and a separate
// consuming authority reads from across a process boundary, without
// either side linking against the other.
//
// Implementations deliver asynchronously: Write must not block on the
// durable destination, and the behavior at bounded capacity must be
// explicit and counted, never a silent drop.
type RecordSink interface {
	// Write enqueues rec for delivery. It returns nil when rec was
	// accepted into the sink's bounded queue, ErrFull when the sink's
	// overflow policy refused it, and ErrClosed after Close.
	Write(rec billing.UsageRecord) error

	// Flush forces delivery of everything currently queued and returns
	// once it has been appended to the destination, or with the append
	// error. Every record written before Flush returns is appended
	// before it returns.
	Flush() error

	// Close drains the queue, closes the destination, and stops the
	// sink's background delivery. It is idempotent: later calls return
	// nil.
	Close() error
}

// Sentinel errors. Callers match these with errors.Is; implementations
// wrap them with context.
var (
	// ErrClosed reports an operation on a sink whose Close has
	// completed.
	ErrClosed = errors.New("sink: closed")

	// ErrFull reports a Write refused by the RejectNew overflow policy.
	ErrFull = errors.New("sink: ring full")

	// ErrInvalidOptions reports a RingOptions value that fails
	// validation.
	ErrInvalidOptions = errors.New("sink: invalid options")

	// ErrMalformedLine reports a JSONL line that is not exactly one
	// decodable record because of its shape: an empty interior line, or
	// a missing final newline. A line whose content fails strict
	// decoding surfaces the billing.ErrMalformedPayload sentinel
	// instead.
	ErrMalformedLine = errors.New("sink: malformed JSONL line")
)

// OverflowPolicy names what a bounded sink does when a Write arrives at
// full capacity. The choice is explicit and typed: neither policy is a
// default, and both count every record they discard or refuse through a
// typed counter, so overflow is never silent.
type OverflowPolicy int

const (
	// DropOldest discards the oldest queued record to make room for the
	// new one. Discarded records are counted by Ring.Dropped.
	DropOldest OverflowPolicy = iota

	// RejectNew refuses the new record with ErrFull. Refused records
	// are counted by Ring.Rejected.
	RejectNew
)

// String renders the policy in its closed-vocabulary form.
func (p OverflowPolicy) String() string {
	switch p {
	case DropOldest:
		return "drop_oldest"
	case RejectNew:
		return "reject_new"
	default:
		return "unknown"
	}
}

// RingOptions fixes a Ring's behavior explicitly. NewRing validates
// every field and returns ErrInvalidOptions rather than substituting a
// hidden default, so the configured behavior is always the shipped
// behavior.
type RingOptions struct {
	// Capacity is the maximum number of records the ring holds at
	// once. Must be >= 1.
	Capacity int

	// BatchSize is the maximum number of records handed to one Backend
	// Append call. Must be in [1, Capacity].
	BatchSize int

	// FlushInterval bounds how long a record may wait in the ring
	// before the background flusher delivers it, independent of the
	// batch filling. Zero disables the timer: records are delivered
	// only when the ring holds BatchSize records, on an explicit
	// Flush, or on Close.
	FlushInterval time.Duration

	// OverflowPolicy decides what a full ring does with a new record.
	// Must be DropOldest or RejectNew.
	OverflowPolicy OverflowPolicy
}

// Backend is the synchronous, blocking half of an out-of-process sink:
// it moves already-enqueued records to a durable destination outside
// this process. A file today; a socket or broker is the documented
// follow-on seam - implement Backend and pass it to NewRing, nothing
// else changes.
//
// A Backend appends whole batches in order and is called only from the
// ring's flusher goroutine, never from a caller's goroutine. That
// single-writer rule is what keeps batched appends ordered and safe
// without caller-side locking. The batch slice is valid only during the
// Append call; a Backend that wants to keep records copies them first.
//
// Implementations treat Append as fallible and return the transport
// error unchanged; the ring counts the failure and retries the batch on
// the next delivery attempt.
type Backend interface {
	Append(batch []billing.UsageRecord) error
	Close() error
}
