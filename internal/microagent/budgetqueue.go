package microagent

import (
	"container/heap"
	"context"
	"errors"
	"sync"
)

// BudgetQueue is the cooperative TOKEN layer for a microagent host (#2021,
// epic #2000 M21): it bounds the SUM of token costs reserved by in-flight
// admits to a configured budget, per host. It is the token-budget analogue of
// the SLOT-layer Scheduler (slotsched.go): slots bound concurrency (calls ≤
// K), the budget bounds tokens (Σ reserved ≤ budget). Admit reserves a cost up
// front and PARKS callers that would exceed the budget — blocked on a channel,
// consuming no CPU (no busy-wait, no polling loop) — then wakes them in
// priority-then-FIFO order as capacity frees, strictly in order: the head
// waiter is never skipped to admit a smaller admit out of order, even when
// that leaves capacity idle (no queue jump).
//
// The park queue is itself BOUNDED (MaxQueue): under sustained
// over-subscription an unbounded park queue just moves the overload from the
// provider into the host's own memory, and the enrolling caller learns nothing
// until the process dies. At the bound, Admit sheds immediately with
// ErrBackpressure instead of parking — that typed refusal IS the backpressure
// signal to the enrolling caller (dispatch/wave, M31), and Pressure is the
// poll-before-enqueue readout of the same state. Three layers, in order:
//
//	fits now                     -> admitted
//	over budget, room to park    -> parked  (fair drain, no busy-wait)
//	over budget, park queue full -> shed    (ErrBackpressure, no busy-wait)
//
// Generation intent: gen/next near-term foundation (#2021). Like the rest of
// this package it is opt-in — nothing in the default fak serve / fak guard /
// dispatch path constructs a BudgetQueue. Closing evidence for the generation
// frame:
//
//   - Promotion evidence: budgetqueue_test.go witnesses the acceptance clause
//     under the bounded regime —
//     TestBudgetQueueSustainedOversubscriptionShedsThenDrains holds Σ reserved
//     ≤ budget and parked ≤ MaxQueue while 50 admits contend for a budget that
//     fits one, sheds the surplus as typed ErrBackpressure rather than growing,
//     and still drains every parked admit when capacity returns. Promote once a
//     real enroller (dispatch/wave, M31) reads Pressure to stop enqueuing and
//     the #2033 density measurement shows provider budget — not local slots —
//     is the binding limit.
//   - Demotion / retirement criteria: retire the shed layer if the enroller
//     gains end-to-end flow control upstream (nothing ever reaches a full park
//     queue, so ErrBackpressure is dead code), or if the provider grows a
//     native token-fair admission the host can defer to instead of reserving
//     locally.
//   - Invalidating assumption: the cost passed to Admit is a usable up-front
//     estimate of the call's token footprint. If real usage routinely overruns
//     the reservation (long tool-call fan-out, streaming continuations,
//     retries billed to the same admit), Σ reserved stops tracking Σ spent and
//     the budget stops binding the provider — the reservation would then have
//     to be reconciled against measured usage on release, not trusted at
//     admit.
type BudgetQueue struct {
	mu       sync.Mutex
	budget   int64            // total token budget. Set once, read freely.
	maxQueue int              // bound on parked admits. Set once, read freely.
	used     int64            // Σ costs currently reserved (running + just-granted).
	seq      uint64           // monotonic arrival counter for FIFO tiebreak among equals.
	waiters  budgetWaiterHeap // parked, runnable-but-waiting admits.
	closed   bool
}

// DefaultTokenBudget is the budget used when NewBudgetQueue is given a
// non-positive value.
const DefaultTokenBudget int64 = 128000

// DefaultMaxParkedAdmits bounds parked admits when no depth is configured. It
// is sized to the host's agent scale (doc.go: one session-table entry per
// agent, limit 8192) rather than to the budget: parking is cheap (one waiter
// struct and a blocked goroutine), so the bound exists to keep overload
// BOUNDED and visible, not to keep it small. An enroller that wants to shed
// earlier reads Pressure and stops before it reaches this.
const DefaultMaxParkedAdmits = 1024

// ErrBudgetQueueClosed is returned by Admit once Close has run: a parked
// caller is woken with it, and every later Admit refuses with it.
var ErrBudgetQueueClosed = errors.New("microagent: budget queue is closed")

// ErrCostExceedsBudget is returned by Admit when the requested cost exceeds
// the TOTAL budget: it can never fit, so it is refused immediately rather than
// parked forever (parking would deadlock).
var ErrCostExceedsBudget = errors.New("microagent: admit cost exceeds total budget")

// ErrNonPositiveCost is returned by Admit for a zero or negative cost — a
// caller bug, not a free pass; refusing keeps the accounting honest.
var ErrNonPositiveCost = errors.New("microagent: admit cost must be positive")

// ErrBackpressure is the backpressure signal to the enrolling caller: the
// budget is spent AND the park queue is at MaxQueue, so this admit is shed
// immediately rather than parked. It is retryable — unlike
// ErrCostExceedsBudget (never fits) and ErrBudgetQueueClosed (terminal), it
// says "not now, stop enqueuing", so an enroller should back off and re-offer
// the work rather than drop it.
var ErrBackpressure = errors.New("microagent: budget spent and park queue full — stop enqueuing (backpressure)")

// NewBudgetQueue builds a token budget queue bounding Σ reserved costs to
// budget, parking up to DefaultMaxParkedAdmits over-budget admits. A
// non-positive budget selects DefaultTokenBudget.
func NewBudgetQueue(budget int64) *BudgetQueue {
	return NewBudgetQueueDepth(budget, DefaultMaxParkedAdmits)
}

// NewBudgetQueueDepth is NewBudgetQueue with an explicit bound on parked
// admits: past maxQueue parked, Admit sheds with ErrBackpressure instead of
// growing the queue. A non-positive maxQueue selects DefaultMaxParkedAdmits.
func NewBudgetQueueDepth(budget int64, maxQueue int) *BudgetQueue {
	if budget <= 0 {
		budget = DefaultTokenBudget
	}
	if maxQueue <= 0 {
		maxQueue = DefaultMaxParkedAdmits
	}
	return &BudgetQueue{budget: budget, maxQueue: maxQueue}
}

// Admit blocks until cost tokens can be reserved within the budget, the
// context is cancelled, or the queue is closed. On success it returns a
// release function the caller MUST call exactly once when the reservation is
// done (extra calls are no-ops; defer it). On error it returns a no-op release
// and a non-nil error (ctx.Err() on cancellation, ErrNonPositiveCost /
// ErrCostExceedsBudget on an impossible cost, ErrBudgetQueueClosed after
// Close, ErrBackpressure when the budget is spent and the park queue is
// already at MaxQueue). Every refusal path reserves nothing and parks nothing.
//
// priority is the fairness hook: higher values are granted first, ties broken
// FIFO by arrival, so equal-priority callers never starve.
func (q *BudgetQueue) Admit(ctx context.Context, cost int64, priority int) (release func(), err error) {
	if err := ctx.Err(); err != nil {
		return noopRelease, err
	}
	if cost <= 0 {
		return noopRelease, ErrNonPositiveCost
	}
	if cost > q.budget {
		// Can NEVER fit — refuse now rather than park forever.
		return noopRelease, ErrCostExceedsBudget
	}
	q.mu.Lock()
	if q.closed {
		q.mu.Unlock()
		return noopRelease, ErrBudgetQueueClosed
	}
	// Fast path: the cost fits AND nobody is ahead in line. The waiters guard
	// keeps a newcomer from jumping the queue when it is not empty.
	if q.used+cost <= q.budget && q.waiters.Len() == 0 {
		q.used += cost
		q.mu.Unlock()
		return q.releaseOnce(cost), nil
	}
	// Shed path: the park queue is at its bound. Refuse NOW — parking would
	// grow the queue without limit, and spinning would burn CPU; both hide the
	// overload from the caller that can actually stop producing it.
	if q.waiters.Len() >= q.maxQueue {
		q.mu.Unlock()
		return noopRelease, ErrBackpressure
	}
	// Slow path: park. The waiter blocks on its own channel — no CPU until a
	// release (or Close) signals it.
	q.seq++
	w := &budgetWaiter{parkedWaiter: parkedWaiter{priority: priority, seq: q.seq, ready: make(chan struct{}), index: -1}, cost: cost}
	heap.Push(&q.waiters, w)
	q.mu.Unlock()

	select {
	case <-w.ready:
		// Woken under the lock: granted a reservation, or drained by Close.
		if w.granted {
			return q.releaseOnce(cost), nil
		}
		return noopRelease, ErrBudgetQueueClosed
	case <-ctx.Done():
		q.mu.Lock()
		if w.granted {
			// Raced a release that transferred capacity to us just as ctx
			// expired; the queue counts the cost as ours, so hand it back.
			q.mu.Unlock()
			q.release(cost)
			return noopRelease, ctx.Err()
		}
		if w.index >= 0 { // still parked — withdraw from the queue.
			heap.Remove(&q.waiters, w.index)
		}
		q.mu.Unlock()
		return noopRelease, ctx.Err()
	}
}

// releaseOnce wraps release in a sync.Once so a caller's deferred release is
// idempotent even if also called explicitly.
func (q *BudgetQueue) releaseOnce(cost int64) func() {
	var once sync.Once
	return func() { once.Do(func() { q.release(cost) }) }
}

// release returns cost tokens to the budget, then drains the parked queue in
// strict priority-then-FIFO order: each head waiter that fits is granted its
// reservation directly (used stays accounted, capacity is transferred rather
// than freed-then-reacquired). The drain STOPS at the first head waiter that
// does not fit — capacity is left idle rather than skipping past it to admit a
// smaller waiter out of order (no queue jump).
func (q *BudgetQueue) release(cost int64) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.used -= cost
	for q.waiters.Len() > 0 && q.used+q.waiters[0].cost <= q.budget {
		w := heap.Pop(&q.waiters).(*budgetWaiter)
		w.granted = true
		q.used += w.cost
		close(w.ready)
	}
}

// Close drains every parked waiter (each Admit returns ErrBudgetQueueClosed)
// and refuses all future Admit. It does NOT cancel in-flight holders; their
// release() still runs correctly (it just returns tokens, finding no waiters).
// Idempotent.
func (q *BudgetQueue) Close() {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return
	}
	q.closed = true
	for q.waiters.Len() > 0 {
		w := heap.Pop(&q.waiters).(*budgetWaiter) // granted stays false → ErrBudgetQueueClosed
		close(w.ready)
	}
}

// Budget reports the configured total token budget (immutable).
func (q *BudgetQueue) Budget() int64 { return q.budget }

// Used reports how many tokens are reserved right now (gauge).
func (q *BudgetQueue) Used() int64 {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.used
}

// MaxQueue reports the configured bound on parked admits (immutable).
func (q *BudgetQueue) MaxQueue() int { return q.maxQueue }

// Waiting reports how many admits are parked right now (gauge).
func (q *BudgetQueue) Waiting() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.waiters.Len()
}

// BudgetPressure is one consistent snapshot of the queue's admission state —
// the readout an enrolling caller (dispatch/wave, M31) polls to decide whether
// to keep enqueuing, and the one an operator reads to see why work is not
// moving. Taken under the lock, so Used and Waiting are the same instant, not
// two racing gauges.
type BudgetPressure struct {
	Budget   int64 // total token budget.
	Used     int64 // Σ tokens reserved right now.
	Waiting  int   // admits parked right now.
	MaxQueue int   // bound on parked admits.
	// Shed reports that the park queue is FULL: the next over-budget Admit is
	// refused with ErrBackpressure. This is the hard stop-enqueuing signal. A
	// caller that wants to back off EARLIER — before it starts earning
	// refusals — sets its own watermark on Waiting/MaxQueue; the queue does
	// not pick that policy for it.
	Shed bool
}

// Pressure returns a consistent snapshot of budget and queue occupancy.
func (q *BudgetQueue) Pressure() BudgetPressure {
	q.mu.Lock()
	defer q.mu.Unlock()
	return BudgetPressure{
		Budget:   q.budget,
		Used:     q.used,
		Waiting:  q.waiters.Len(),
		MaxQueue: q.maxQueue,
		Shed:     q.waiters.Len() >= q.maxQueue,
	}
}

// budgetWaiter is one parked Admit: the shared parked-caller bookkeeping plus the
// reservation this admit is waiting to make.
type budgetWaiter struct {
	parkedWaiter
	cost int64
}

// budgetWaiterHeap is the budget queue's parked-admit queue. It uses the same
// priority/FIFO ordering as the slot scheduler's waiterHeap (see waiterHeapOf).
type budgetWaiterHeap = waiterHeapOf[*budgetWaiter]
