package model

// This file adds a PURE IN-MEMORY receiver-granted credit ledger for cross-node
// KV-transfer backpressure. It is CPU-witnessable: there is NO RDMA, socket, or
// clock in the credit logic — only plain ints and maps behind one mutex — so its
// whole behaviour is exercised deterministically in a unit test.
//
// WHY a receiver-granted credit at all. The existing KV transfer plane
// (kv_transport_registry.go) only SELECTS a backend; it has no flow control
// between a producer of KV blocks (a fast prefill node) and a consumer (a slower
// decode node). fak's other backpressure primitives are producer-side: they
// coalesce or drop at the SOURCE (see internal/loaddebounce). None lets the
// RECEIVER meter the producer. Under disaggregated prefill/decode a prefill node
// can emit KV faster than a decode node ingests it, overrunning the decode node's
// KV-ingest buffers. The fix, borrowed clean-room from Mooncake's `tent` runtime
// (Apache-2.0, no bytes vendored — receiver_credit.cpp tryReserve/rollback +
// cumulative applyUpdate), is to let the CONSUMER grant credit the producer must
// hold before it transfers. The receiver sets the pace per QoS class instead of
// dropping or blocking blindly at the source.
//
// The ledger is keyed per (receiver, sender, qos) tuple so one slow class on one
// receiver throttles only that class from that sender, never the whole fabric.
//
// Two-phase, idempotent by design:
//   - Grant(cumulative) publishes the receiver's CUMULATIVE, monotonic-non-
//     decreasing credit for a tuple. Grants are deduped by (epoch, seq): a
//     replayed or reordered grant that is not strictly newer than the last
//     applied one is ignored, and a lower cumulative value never lowers the
//     ceiling. So a grant may be retransmitted safely.
//   - TryReserve(n) reserves n against the available credit and returns whether
//     it succeeded; a reserve past the granted ceiling is refused (this is the
//     backpressure). Rollback(n) returns a failed/aborted reservation exactly,
//     restoring available credit. Consume(n) finalizes a reservation, spending
//     the credit permanently.
//
// Wiring is deliberately deferred: this primitive plus its test is the shippable
// unit. The cross-node KV transport plane (kv_transport_registry.go) or the
// disaggregation admission path adopts it later; wiring now would risk the shared
// build.

import "sync"

// CreditKey identifies one flow-control tuple. A receiver grants credit to a
// sender for a single QoS class; each distinct tuple has its own tally so
// throttling one class from one sender never touches another.
type CreditKey struct {
	Receiver string
	Sender   string
	QoS      string
}

// creditTally is the per-tuple bookkeeping. Available credit is
// Granted - Reserved - Consumed and is never allowed below zero. GrantEpoch and
// GrantSeq are the dedup coordinates of the last APPLIED grant: a later grant is
// accepted only if its (epoch, seq) is strictly greater, which makes a replayed
// grant a no-op.
type creditTally struct {
	Granted    int64
	Reserved   int64
	Consumed   int64
	GrantEpoch uint64
	GrantSeq   uint64
	// hasGrant distinguishes "no grant ever applied" from "granted zero" so the
	// very first grant at (epoch 0, seq 0) is not swallowed by the dedup test.
	hasGrant bool
}

// CreditLedger is a set of per-tuple tallies behind one mutex. It holds no
// resources, no clock, and no network — every method mutates plain ints, so the
// ledger is safe to share across the producer/consumer goroutines of a transfer
// plane and fully deterministic under test.
type CreditLedger struct {
	mu      sync.Mutex
	tallies map[CreditKey]*creditTally
}

// NewCreditLedger returns an empty ledger. Tuples are created lazily on first
// use so an unbounded key space costs nothing until it is actually granted or
// reserved against.
func NewCreditLedger() *CreditLedger {
	return &CreditLedger{tallies: map[CreditKey]*creditTally{}}
}

func (l *CreditLedger) tallyLocked(key CreditKey) *creditTally {
	t := l.tallies[key]
	if t == nil {
		t = &creditTally{}
		l.tallies[key] = t
	}
	return t
}

// Grant publishes a receiver's CUMULATIVE credit ceiling for a tuple and reports
// whether it changed state. cumulative is the total credit ever granted (not a
// delta), so retransmitting the same or an older grant is safe:
//
//   - the (epoch, seq) pair must be STRICTLY newer than the last applied grant,
//     else the grant is a replayed/reordered duplicate and is ignored;
//   - even when newer, cumulative may only RAISE the ceiling — a lower cumulative
//     value (a stale reading that slipped past dedup) never lowers Granted, so
//     the ceiling is monotonic-non-decreasing.
//
// A negative cumulative is clamped to the current ceiling (never lowers it).
// Returns true iff the ceiling advanced.
func (l *CreditLedger) Grant(key CreditKey, epoch, seq uint64, cumulative int64) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	t := l.tallyLocked(key)
	if t.hasGrant && !newerGrant(epoch, seq, t.GrantEpoch, t.GrantSeq) {
		return false
	}
	t.hasGrant = true
	t.GrantEpoch = epoch
	t.GrantSeq = seq
	if cumulative <= t.Granted {
		return false
	}
	t.Granted = cumulative
	return true
}

// newerGrant reports whether (epoch, seq) is strictly newer than
// (lastEpoch, lastSeq) under lexicographic order (epoch dominates, seq breaks
// ties). This is the whole dedup rule for grants.
func newerGrant(epoch, seq, lastEpoch, lastSeq uint64) bool {
	if epoch != lastEpoch {
		return epoch > lastEpoch
	}
	return seq > lastSeq
}

// Available returns the credit a sender may still reserve for a tuple:
// Granted - Reserved - Consumed, never below zero.
func (l *CreditLedger) Available(key CreditKey) int64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	t := l.tallies[key]
	if t == nil {
		return 0
	}
	return availableLocked(t)
}

func availableLocked(t *creditTally) int64 {
	avail := t.Granted - t.Reserved - t.Consumed
	if avail < 0 {
		return 0
	}
	return avail
}

// TryReserve attempts to reserve n credits for a tuple. It succeeds and holds the
// reservation only when n fits within the currently available credit; otherwise
// it reserves nothing and returns false. That refusal IS the backpressure: a
// producer that cannot reserve must wait for the receiver to grant more. n<=0
// reserves nothing and succeeds trivially.
func (l *CreditLedger) TryReserve(key CreditKey, n int64) bool {
	if n <= 0 {
		return true
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	t := l.tallyLocked(key)
	if availableLocked(t) < n {
		return false
	}
	t.Reserved += n
	return true
}

// Rollback returns an aborted reservation of n credits to the available pool,
// restoring Available exactly. It is clamped so a double rollback or an oversized
// n can never drive Reserved negative. n<=0 is a no-op.
func (l *CreditLedger) Rollback(key CreditKey, n int64) {
	if n <= 0 {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	t := l.tallyLocked(key)
	if n > t.Reserved {
		n = t.Reserved
	}
	t.Reserved -= n
}

// Consume finalizes a held reservation of n credits: the credit is spent, so it
// moves from Reserved to Consumed and does NOT return to Available. It is clamped
// to the outstanding reservation so it can never spend more than was reserved.
// n<=0 is a no-op. Returns the amount actually consumed.
func (l *CreditLedger) Consume(key CreditKey, n int64) int64 {
	if n <= 0 {
		return 0
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	t := l.tallyLocked(key)
	if n > t.Reserved {
		n = t.Reserved
	}
	t.Reserved -= n
	t.Consumed += n
	return n
}

// Snapshot returns a copy of a tuple's counters for observability and testing. A
// missing tuple reads as all-zero. It carries no pointer, so the caller cannot
// mutate ledger state through it.
type CreditSnapshot struct {
	Granted    int64
	Reserved   int64
	Consumed   int64
	Available  int64
	GrantEpoch uint64
	GrantSeq   uint64
}

// Snapshot returns the counters for one tuple.
func (l *CreditLedger) Snapshot(key CreditKey) CreditSnapshot {
	l.mu.Lock()
	defer l.mu.Unlock()
	t := l.tallies[key]
	if t == nil {
		return CreditSnapshot{}
	}
	return CreditSnapshot{
		Granted:    t.Granted,
		Reserved:   t.Reserved,
		Consumed:   t.Consumed,
		Available:  availableLocked(t),
		GrantEpoch: t.GrantEpoch,
		GrantSeq:   t.GrantSeq,
	}
}
