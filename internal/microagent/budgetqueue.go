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
type BudgetQueue struct {
	mu      sync.Mutex
	budget  int64            // total token budget. Set once, read freely.
	used    int64            // Σ costs currently reserved (running + just-granted).
	seq     uint64           // monotonic arrival counter for FIFO tiebreak among equals.
	waiters budgetWaiterHeap // parked, runnable-but-waiting admits.
	closed  bool
}

// DefaultTokenBudget is the budget used when NewBudgetQueue is given a
// non-positive value.
const DefaultTokenBudget int64 = 128000

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

// NewBudgetQueue builds a token budget queue bounding Σ reserved costs to
// budget. A non-positive budget selects DefaultTokenBudget.
func NewBudgetQueue(budget int64) *BudgetQueue {
	if budget <= 0 {
		budget = DefaultTokenBudget
	}
	return &BudgetQueue{budget: budget}
}

// Admit blocks until cost tokens can be reserved within the budget, the
// context is cancelled, or the queue is closed. On success it returns a
// release function the caller MUST call exactly once when the reservation is
// done (extra calls are no-ops; defer it). On error it returns a no-op release
// and a non-nil error (ctx.Err() on cancellation, ErrNonPositiveCost /
// ErrCostExceedsBudget on an impossible cost, ErrBudgetQueueClosed after
// Close).
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
	// Slow path: park. The waiter blocks on its own channel — no CPU until a
	// release (or Close) signals it.
	q.seq++
	w := &budgetWaiter{priority: priority, seq: q.seq, cost: cost, ready: make(chan struct{}), index: -1}
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

// Waiting reports how many admits are parked right now (gauge).
func (q *BudgetQueue) Waiting() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.waiters.Len()
}

// budgetWaiter is one parked Admit. ready is closed under q.mu to wake it;
// granted distinguishes a reservation grant from a Close drain. index is its
// position in the heap (-1 once popped/removed), maintained by
// budgetWaiterHeap.Swap.
type budgetWaiter struct {
	priority int
	seq      uint64
	cost     int64
	ready    chan struct{}
	granted  bool
	index    int
}

// budgetWaiterHeap orders parked admits by priority (higher first), ties
// broken by arrival sequence (lower first = FIFO), so capacity always goes to
// the most urgent waiter and equal-priority waiters never starve.
type budgetWaiterHeap []*budgetWaiter

func (h budgetWaiterHeap) Len() int { return len(h) }

func (h budgetWaiterHeap) Less(i, j int) bool {
	if h[i].priority != h[j].priority {
		return h[i].priority > h[j].priority
	}
	return h[i].seq < h[j].seq
}

func (h budgetWaiterHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].index = i
	h[j].index = j
}

func (h *budgetWaiterHeap) Push(x any) {
	w := x.(*budgetWaiter)
	w.index = len(*h)
	*h = append(*h, w)
}

func (h *budgetWaiterHeap) Pop() any {
	old := *h
	n := len(old)
	w := old[n-1]
	old[n-1] = nil // don't hold the popped waiter alive via the backing array.
	w.index = -1
	*h = old[:n-1]
	return w
}
