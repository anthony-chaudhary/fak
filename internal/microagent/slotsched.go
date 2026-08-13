package microagent

import (
	"container/heap"
	"context"
	"errors"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// Scheduler is the cooperative SLOT layer for a microagent host (#2006, epic
// #2000 M6): it bounds the number of model calls in flight at once to a
// configured K, per host (or per seat — run one Scheduler per seat), no matter
// how many goroutine agents are runnable. A host can hold thousands of agents;
// the provider caps concurrency and TPM, so all of them cannot have a live
// provider call at the same instant. Acquire admits up to K callers and PARKS
// the rest as runnable-but-waiting — blocked on a channel, consuming no CPU
// (no busy-wait, no polling loop) — then wakes them in priority-then-FIFO order
// as slots free.
//
// It is deliberately distinct from the two layers the issue names:
//
//   - the INFERENCE scheduler (internal/modelengine/nativesched.go) batches work
//     inside one model server;
//   - the DISPATCH router assigns whole agents to lanes/hosts.
//
// This is neither: it is the per-host admission gate a hosted agent steps
// through around ONE model call. It also composes cleanly ABOVE the token-aware
// admission control (M19, internal/gateway/admission.go): that is the BUDGET
// layer (Σ running tokens ≤ TokenBudget); this is the SLOT layer (concurrent
// calls ≤ K). Slots bound concurrency; the budget bounds tokens. A call passes
// the slot gate here, then the budget gate there — neither subsumes the other.
//
// Generation intent: gen/second-next architectural OPTION (#2002). Like the rest
// of this package, nothing in the default fak serve / fak guard / dispatch path
// constructs a Scheduler — it is opt-in behind SchedulingGateway. Closing
// evidence for the generation frame:
//
//   - Promotion evidence: slotsched_test.go witnesses both acceptance criteria —
//     under a burst of N≫K microagents the concurrent-call gauge never exceeds K
//     and every agent completes, and parked agents make zero progress (stable
//     waiting count, woken only by a Release) so they cannot be busy-waiting.
//   - Demotion / retirement criteria: retire the slot layer if the #2033
//     density benchmark shows per-host concurrency is never the binding limit
//     (the token budget alone (M19) always binds first, so the slot gate is dead
//     weight), or if the provider grows a native fair concurrency admission the
//     host can defer to instead of bounding locally.
//   - Invalidating assumption: Acquire assumes one model call == one slot for
//     its whole duration (a call holds exactly one unit of provider concurrency).
//     Streaming, tool-call fan-out within a turn, or speculative/parallel decode
//     that issues several concurrent provider requests per logical turn would
//     break the one-call-one-slot accounting; the slot would then have to be a
//     weighted reservation (like M19's token footprint), not a unit.
type Scheduler struct {
	mu       sync.Mutex
	limit    int        // K: max concurrent in-flight slots. Set once, read freely.
	inflight int        // slots currently held (running + just-granted).
	seq      uint64     // monotonic arrival counter for FIFO tiebreak among equals.
	waiters  waiterHeap // parked, runnable-but-waiting callers.
	closed   bool
}

// DefaultSlots is the K used when NewScheduler is given a non-positive value —
// matched to DefaultWorkers so the slot gate does not bind tighter than the
// worker pool by default.
const DefaultSlots = 8

// ErrSchedulerClosed is returned by Acquire once Close has run: a parked caller
// is woken with it, and every later Acquire refuses with it.
var ErrSchedulerClosed = errors.New("microagent: slot scheduler is closed")

// noopRelease is the release function returned on an Acquire that did not take a
// slot (error paths), so callers can defer release() unconditionally.
func noopRelease() {}

// NewScheduler builds a slot scheduler bounding concurrent in-flight calls to k.
// A non-positive k selects DefaultSlots.
func NewScheduler(k int) *Scheduler {
	if k <= 0 {
		k = DefaultSlots
	}
	return &Scheduler{limit: k}
}

// Acquire blocks until a slot is free (bounded to K), the context is cancelled,
// or the scheduler is closed. On success it returns a release function the
// caller MUST call exactly once when the model call completes (extra calls are
// no-ops; defer it). On error it returns a no-op release and a non-nil error
// (ctx.Err() on cancellation, ErrSchedulerClosed after Close).
//
// priority is the fairness hook: higher values are granted first, ties broken
// FIFO by arrival, so equal-priority callers never starve. The host sources it
// from the session Table (session.State.Priority) — see WithPriority — but the
// scheduler stays decoupled by taking a plain int.
func (s *Scheduler) Acquire(ctx context.Context, priority int) (release func(), err error) {
	if err := ctx.Err(); err != nil {
		return noopRelease, err
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return noopRelease, ErrSchedulerClosed
	}
	// Fast path: a free slot AND nobody ahead in line. The invariant "inflight <
	// limit ⇒ no waiters" (Release grants until slots or waiters run out) means
	// this is the normal uncontended case; the waiters guard keeps a newcomer
	// from jumping the queue when it is not.
	if s.inflight < s.limit && s.waiters.Len() == 0 {
		s.inflight++
		s.mu.Unlock()
		return s.releaseOnce(), nil
	}
	// Slow path: park. The waiter blocks on its own channel — no CPU until a
	// Release (or Close) signals it.
	s.seq++
	w := &waiter{parkedWaiter{priority: priority, seq: s.seq, ready: make(chan struct{}), index: -1}}
	heap.Push(&s.waiters, w)
	s.mu.Unlock()

	select {
	case <-w.ready:
		// Woken under the lock: granted a slot, or drained by Close.
		if w.granted {
			return s.releaseOnce(), nil
		}
		return noopRelease, ErrSchedulerClosed
	case <-ctx.Done():
		s.mu.Lock()
		if w.granted {
			// Raced a Release that transferred a slot to us just as ctx expired;
			// the scheduler counts the slot as ours, so hand it straight back.
			s.mu.Unlock()
			s.release()
			return noopRelease, ctx.Err()
		}
		if w.index >= 0 { // still parked — withdraw from the queue.
			heap.Remove(&s.waiters, w.index)
		}
		s.mu.Unlock()
		return noopRelease, ctx.Err()
	}
}

// releaseOnce wraps release in a sync.Once so a caller's deferred release is
// idempotent even if also called explicitly.
func (s *Scheduler) releaseOnce() func() {
	var once sync.Once
	return func() { once.Do(s.release) }
}

// release returns one slot. It hands the freed slot DIRECTLY to the next waiter
// (highest priority, FIFO among equals) if any — inflight stays put, the slot is
// transferred rather than freed-then-reacquired, so a concurrent burst cannot
// slip an extra call past K. With no waiter it decrements inflight.
func (s *Scheduler) release() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.waiters.Len() > 0 {
		w := heap.Pop(&s.waiters).(*waiter)
		w.granted = true
		close(w.ready)
		return
	}
	s.inflight--
}

// Close drains every parked waiter (each Acquire returns ErrSchedulerClosed) and
// refuses all future Acquire. It does NOT cancel in-flight holders; their
// release() still runs correctly (it just decrements, finding no waiters).
// Idempotent.
func (s *Scheduler) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.closed = true
	for s.waiters.Len() > 0 {
		w := heap.Pop(&s.waiters).(*waiter) // granted stays false → ErrSchedulerClosed
		close(w.ready)
	}
}

// Limit reports K, the configured maximum concurrent in-flight calls.
// TryAcquire admits immediately or returns false without parking. It is the
// capacity-fallback seam for seat-aware routing: callers can probe another
// independently bounded seat rather than queue behind a busy affinity target.
func (s *Scheduler) TryAcquire() (release func(), ok bool) {
	if s == nil {
		return noopRelease, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.inflight >= s.limit || s.waiters.Len() > 0 {
		return noopRelease, false
	}
	s.inflight++
	return s.releaseOnce(), true
}

func (s *Scheduler) Limit() int { return s.limit }

// Inflight reports how many slots are held right now (gauge).
func (s *Scheduler) Inflight() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inflight
}

// Waiting reports how many callers are parked right now (gauge).
func (s *Scheduler) Waiting() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.waiters.Len()
}

// parkedWaiter is the bookkeeping every parked caller in this package carries,
// whatever it is parked ON: ready is closed under the owning mutex to wake it;
// granted distinguishes a real grant from a Close drain; index is its position in
// the heap (-1 once popped/removed), maintained by waiterHeapOf.Swap. The slot
// scheduler's waiter and the budget queue's budgetWaiter both embed it, which is
// what lets them share one heap implementation.
type parkedWaiter struct {
	priority int
	seq      uint64
	ready    chan struct{}
	granted  bool
	index    int
}

// node exposes the shared bookkeeping to waiterHeapOf, which cannot reach an
// embedded field through a type parameter.
func (w *parkedWaiter) node() *parkedWaiter { return w }

// waiterNode is what waiterHeapOf orders: anything carrying a parkedWaiter.
type waiterNode interface{ node() *parkedWaiter }

// waiterHeapOf orders parked callers by priority (higher first), ties broken by
// arrival sequence (lower first = FIFO), so capacity always goes to the most urgent
// waiter and equal-priority waiters never starve. One implementation serves both
// queues; they differ only in what else their element carries (the budget queue's
// element also carries the admit's cost).
type waiterHeapOf[T waiterNode] []T

func (h waiterHeapOf[T]) Len() int { return len(h) }

func (h waiterHeapOf[T]) Less(i, j int) bool {
	a, b := h[i].node(), h[j].node()
	if a.priority != b.priority {
		return a.priority > b.priority
	}
	return a.seq < b.seq
}

func (h waiterHeapOf[T]) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].node().index = i
	h[j].node().index = j
}

func (h *waiterHeapOf[T]) Push(x any) {
	w := x.(T)
	w.node().index = len(*h)
	*h = append(*h, w)
}

func (h *waiterHeapOf[T]) Pop() any {
	old := *h
	n := len(old)
	w := old[n-1]
	var zero T
	old[n-1] = zero // don't hold the popped waiter alive via the backing array.
	w.node().index = -1
	*h = old[:n-1]
	return w
}

// waiter is one parked Acquire.
type waiter struct {
	parkedWaiter
}

// waiterHeap is the slot scheduler's parked-caller queue.
type waiterHeap = waiterHeapOf[*waiter]

// priorityKey carries a per-call slot priority through a context.
type priorityKey struct{}

// WithPriority tags ctx with the slot priority SchedulingGateway uses for the
// model call made under it. This is the seam the host uses to forward a
// session's priority (session.State.Priority) into the slot scheduler WITHOUT
// the scheduler importing internal/session. Higher = scheduled first.
func WithPriority(ctx context.Context, priority int) context.Context {
	return context.WithValue(ctx, priorityKey{}, priority)
}

// PriorityFromContext reads a WithPriority tag, defaulting to 0 (all-equal ⇒
// strict FIFO). It is SchedulingGateway's default priority hook.
func PriorityFromContext(ctx context.Context) int {
	if p, ok := ctx.Value(priorityKey{}).(int); ok {
		return p
	}
	return 0
}

// SchedulingGateway wraps the ONE shared gateway with the cooperative slot
// scheduler: every Complete first Acquires a slot, so no more than K model calls
// are ever in flight at once across all hosted microagents — the surplus park in
// the scheduler (no busy-wait) and drain as slots free. It satisfies
// agent.Planner (== Gateway), so it drops straight into NewHost in place of the
// raw gateway. This is the concrete SLOT layer for a host; it composes above any
// token-budget admission (M19) the wrapped gateway already enforces.
type SchedulingGateway struct {
	gw    Gateway
	sched *Scheduler
	prio  func(context.Context) int
}

// NewSchedulingGateway wraps gw so every model call goes through sched. Priority
// defaults to PriorityFromContext; override it with SetPriorityFunc.
func NewSchedulingGateway(gw Gateway, sched *Scheduler) *SchedulingGateway {
	return &SchedulingGateway{gw: gw, sched: sched, prio: PriorityFromContext}
}

// SetPriorityFunc overrides how the per-call slot priority is derived (e.g. read
// it straight off a *session.Table by a trace id carried in ctx). A nil fn
// restores the default. Set it before serving traffic.
func (g *SchedulingGateway) SetPriorityFunc(fn func(context.Context) int) {
	if fn == nil {
		fn = PriorityFromContext
	}
	g.prio = fn
}

// Model reports the wrapped gateway's model id (provenance passthrough).
func (g *SchedulingGateway) Model() string { return g.gw.Model() }

// Complete acquires a slot, runs the wrapped Complete, then releases. A slot is
// held for exactly the duration of the underlying model call. If the slot cannot
// be acquired (ctx cancelled or scheduler closed) it returns that error and
// never dials the gateway.
func (g *SchedulingGateway) Complete(ctx context.Context, msgs []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	release, err := g.sched.Acquire(ctx, g.prio(ctx))
	if err != nil {
		return nil, err
	}
	defer release()
	return g.gw.Complete(ctx, msgs, tools, opts...)
}

// Scheduler exposes the underlying slot scheduler (observability gauges, Close).
func (g *SchedulingGateway) Scheduler() *Scheduler { return g.sched }
