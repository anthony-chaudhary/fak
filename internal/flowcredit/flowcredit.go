// Package flowcredit is the receiver-granted credit ledger for cross-node KV
// block transfer backpressure (#5293, epic #5289 mooncake-study).
//
// The receiver of KV blocks advertises cumulative credit per
// (receiver, sender, class) lane; the sender must reserve credit before it
// may transmit and can never hold more than the receiver has granted. When
// the window is exhausted the sender backpressures (TryReserve refuses) until
// the receiver grants more, so a fast prefill node cannot overrun a slow
// decode node's KV-ingest buffers — the CONSUMER meters the PRODUCER per QoS
// class, distinct from fak's producer-side coalescing/drop primitives
// (internal/loaddebounce).
//
// Grants are cumulative and monotonic non-decreasing, deduplicated by
// sequence number, so a delayed, replayed, or reordered grant can never
// widen the window twice or narrow it. Reservation is two-phase: TryReserve
// takes units all-or-nothing before a transfer, Rollback returns units for a
// transfer that never happened. The invariant reserved <= granted holds by
// construction after every operation; a receiver reflects consumed (freed)
// buffer capacity back to the sender by raising the cumulative grant.
//
// Clean-room borrow of Mooncake's tent receiver-credit runtime (Apache-2.0,
// no bytes vendored): tryReserve/rollbackReservation over a cumulative
// applyUpdate. Pure in-memory and deterministic — no goroutines, timers, or
// wall clock; callers own all transport I/O. Stdlib-only (sync), imports
// nothing internal, off the hot path.
package flowcredit

import "sync"

// Lane identifies one flow-controlled transfer lane: one receiver session
// ingesting KV blocks from one sender peer at one QoS class. Credit is
// scoped per lane — a grant on one lane never widens another, so a receiver
// can pace bulk prefill traffic independently of latency-sensitive classes.
type Lane struct {
	Receiver string // receiver session id (the consumer of KV blocks)
	Sender   string // sender peer id (the producer)
	Class    string // QoS class the receiver meters independently
}

// Snapshot is a point-in-time view of one lane's ledger state.
type Snapshot struct {
	Granted   uint64 // cumulative units the receiver has granted the sender
	Reserved  uint64 // cumulative units the sender holds (reservations minus rollbacks)
	Available uint64 // Granted - Reserved: units the sender may still reserve
	LastSeq   uint64 // highest grant sequence applied (the dedupe watermark)
}

// Ledger meters senders with receiver-granted cumulative credit, one window
// per Lane. The zero unit is whatever the transport meters (KV blocks here);
// the ledger only conserves units. Safe for concurrent use; every method is
// a single short critical section, so behavior under any interleaving is a
// serialization of atomic operations and the reserved <= granted invariant
// holds at every observable point.
// Invariant: flow credit operations are fail-closed and reserve-safe.
// Senders cannot transmit without cumulative credit granted by the receiver,
// and reservations are strictly bounded by unreserved capacity (reserved <= granted).
// Guard: TryReserve refuses atomically whenever available credit is insufficient.
type Ledger struct {
	mu    sync.Mutex
	lanes map[Lane]*laneState
}

type laneState struct {
	granted  uint64 // cumulative credit granted by the receiver (monotonic)
	reserved uint64 // cumulative credit held by the sender; never exceeds granted
	lastSeq  uint64 // highest grant sequence applied
}

// NewLedger returns an empty ledger. Every lane starts with zero credit:
// until the receiver's first grant arrives, TryReserve refuses and the
// sender transmits nothing (fail-closed).
func NewLedger() *Ledger {
	return &Ledger{lanes: make(map[Lane]*laneState)}
}

// Grant applies the receiver's cumulative credit advertisement for a lane.
// A grant is applied only when seq is strictly newer than the lane's
// watermark (replays and reordered-behind duplicates are deduped), and the
// window only ever widens: a newer message carrying a lower cumulative value
// advances the watermark but never narrows granted (monotonic
// non-decreasing). Returns true iff the window widened.
func (g *Ledger) Grant(l Lane, seq, cumulative uint64) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	s := g.lane(l)
	if seq <= s.lastSeq {
		return false // replay or reordered-behind grant: deduped, never re-applied
	}
	s.lastSeq = seq
	if cumulative <= s.granted {
		return false // monotonic: a stale/lower cumulative never narrows the window
	}
	s.granted = cumulative
	return true
}

// TryReserve takes n units of credit for the sender, all-or-nothing: it
// returns true and moves n from available to reserved only when the whole
// amount fits inside the receiver's granted window. When it refuses, the
// lane is unchanged — this is the sender's backpressure signal: queue or
// block, and retry after the next Grant widens the window. n of 0 is a
// trivially successful no-op.
func (g *Ledger) TryReserve(l Lane, n uint64) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	s := g.lane(l)
	if n > s.granted-s.reserved { // reserved <= granted makes the subtraction safe
		return false
	}
	s.reserved += n
	return true
}

// Rollback returns up to n previously reserved units to the lane's available
// window — the second phase for a transfer that was reserved but never
// transmitted (send failed, peer vanished, request cancelled). The amount is
// clamped to what the sender actually holds, so over-rollback can never mint
// credit the receiver did not grant. Returns the units actually restored.
func (g *Ledger) Rollback(l Lane, n uint64) uint64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	s := g.lane(l)
	if n > s.reserved {
		n = s.reserved // clamp: never restore more than is held
	}
	s.reserved -= n
	return n
}

// View reports the lane's current window without changing it.
func (g *Ledger) View(l Lane) Snapshot {
	g.mu.Lock()
	defer g.mu.Unlock()
	s := g.lane(l)
	return Snapshot{
		Granted:   s.granted,
		Reserved:  s.reserved,
		Available: s.granted - s.reserved,
		LastSeq:   s.lastSeq,
	}
}

// lane returns the lane's state, creating the zero-credit window on first
// touch. Callers hold g.mu.
func (g *Ledger) lane(l Lane) *laneState {
	if g.lanes == nil {
		g.lanes = make(map[Lane]*laneState)
	}
	s, ok := g.lanes[l]
	if !ok {
		s = &laneState{}
		g.lanes[l] = s
	}
	return s
}
