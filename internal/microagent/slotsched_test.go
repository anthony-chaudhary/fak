package microagent_test

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/microagent"
)

// waitForSlot polls cond until it holds or the deadline expires, so state
// assertions ("all N parked", "gauge settled") never race a background
// goroutine on a loaded CI box. It parks the test goroutine, never busy-spins
// the thing under test.
func waitForSlot(t *testing.T, cond func() bool, timeout time.Duration, what string) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timed out after %s waiting for %s", timeout, what)
}

// TestSchedulerBoundsInflightUnderBurst is the first acceptance witness (#2006):
// under a burst of N≫K acquirers, the number of slots held at once never exceeds
// K, and every acquirer eventually completes. The peak gauge is sampled from
// inside the critical section (between Acquire and release) so a breach of K
// would be observed directly.
func TestSchedulerBoundsInflightUnderBurst(t *testing.T) {
	t.Parallel()
	const (
		k = 8
		n = 300
	)
	s := microagent.NewScheduler(k)

	var live atomic.Int64 // slots held right now (sampled inside the gate).
	var peak atomic.Int64 // high-water mark of live.
	var done atomic.Int64 // acquirers that ran to completion.

	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			release, err := s.Acquire(context.Background(), 0)
			if err != nil {
				t.Errorf("Acquire: %v", err)
				return
			}
			defer release()
			cur := live.Add(1)
			for {
				old := peak.Load()
				if cur <= old || peak.CompareAndSwap(old, cur) {
					break
				}
			}
			// Hold the slot briefly so contention is real, not serialized.
			time.Sleep(200 * time.Microsecond)
			live.Add(-1)
			done.Add(1)
		}()
	}
	wg.Wait()

	if got := peak.Load(); got > k {
		t.Fatalf("in-flight peak %d exceeded K=%d", got, k)
	}
	if got := done.Load(); got != n {
		t.Fatalf("only %d/%d acquirers completed", got, n)
	}
	if inf := s.Inflight(); inf != 0 {
		t.Fatalf("scheduler leaked slots: Inflight()=%d, want 0", inf)
	}
	if w := s.Waiting(); w != 0 {
		t.Fatalf("scheduler leaked waiters: Waiting()=%d, want 0", w)
	}
}

// concGateway is a stub agent.Planner whose Complete asserts, from inside the
// call, that no more than max calls are ever concurrently in flight — the
// integration form of acceptance criterion 1 through SchedulingGateway.
type concGateway struct {
	max    int
	live   atomic.Int64
	calls  atomic.Int64
	breach atomic.Bool
	hold   time.Duration
}

func (g *concGateway) Model() string { return "conc" }

func (g *concGateway) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	g.calls.Add(1)
	if cur := g.live.Add(1); cur > int64(g.max) {
		g.breach.Store(true)
	}
	if g.hold > 0 {
		time.Sleep(g.hold)
	}
	g.live.Add(-1)
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}, nil
}

// TestSchedulingGatewayBoundsConcurrentCalls witnesses that wrapping the ONE
// shared gateway in a SchedulingGateway holds the provider's concurrent-call
// count to K, no matter how many goroutine agents fire Complete at once.
func TestSchedulingGatewayBoundsConcurrentCalls(t *testing.T) {
	t.Parallel()
	const (
		k = 4
		n = 120
	)
	inner := &concGateway{max: k, hold: 150 * time.Microsecond}
	sched := microagent.NewScheduler(k)
	gw := microagent.NewSchedulingGateway(inner, sched)

	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			if _, err := gw.Complete(context.Background(), nil, nil); err != nil {
				t.Errorf("Complete: %v", err)
			}
		}()
	}
	wg.Wait()

	if inner.breach.Load() {
		t.Fatalf("provider saw more than K=%d concurrent calls", k)
	}
	if got := inner.calls.Load(); got != n {
		t.Fatalf("gateway made %d/%d calls", got, n)
	}
	if inf := sched.Inflight(); inf != 0 {
		t.Fatalf("slots leaked after drain: Inflight()=%d", inf)
	}
}

// TestSchedulerParkedWaitersDoNotBusyWait is the second acceptance witness:
// callers that cannot get a slot PARK — they make zero progress and their count
// stays stable until a Release, and each Release wakes exactly one. A busy-wait
// or spurious wakeup would show up as progress with no Release, or a parked
// count that drifts on its own.
func TestSchedulerParkedWaitersDoNotBusyWait(t *testing.T) {
	t.Parallel()
	const (
		k = 2
		m = 6 // surplus waiters beyond K.
	)
	s := microagent.NewScheduler(k)

	// Fill every slot and keep them held.
	holders := make([]func(), 0, k)
	for i := 0; i < k; i++ {
		release, err := s.Acquire(context.Background(), 0)
		if err != nil {
			t.Fatalf("Acquire holder %d: %v", i, err)
		}
		holders = append(holders, release)
	}

	var progressed atomic.Int64
	waiterRelease := make(chan func(), m)
	for i := 0; i < m; i++ {
		go func() {
			release, err := s.Acquire(context.Background(), 0)
			if err != nil {
				t.Errorf("waiter Acquire: %v", err)
				return
			}
			progressed.Add(1)
			waiterRelease <- release
		}()
	}

	waitForSlot(t, func() bool { return s.Waiting() == m }, 2*time.Second, "all waiters parked")

	// No Release has happened: parked count must be stable and progress zero.
	// This window is the negative witness — a busy-wait would advance here.
	time.Sleep(50 * time.Millisecond)
	if w := s.Waiting(); w != m {
		t.Fatalf("parked count drifted without a Release: Waiting()=%d, want %d", w, m)
	}
	if p := progressed.Load(); p != 0 {
		t.Fatalf("a parked waiter progressed with no slot freed: %d", p)
	}
	if inf := s.Inflight(); inf != k {
		t.Fatalf("Inflight()=%d while all slots held, want %d", inf, k)
	}

	// Release one held slot: exactly one waiter must wake, then the system must
	// go quiet again (no further progress until the next Release).
	holders[0]()
	waitForSlot(t, func() bool { return progressed.Load() == 1 }, 2*time.Second, "one waiter woken")
	time.Sleep(50 * time.Millisecond)
	if p := progressed.Load(); p != 1 {
		t.Fatalf("more than one waiter woke from a single Release: %d", p)
	}
	if w := s.Waiting(); w != m-1 {
		t.Fatalf("Waiting()=%d after one Release, want %d", w, m-1)
	}

	// Drain the rest: release the remaining held slot and every woken waiter's
	// slot; all m waiters must complete and the scheduler must settle to empty.
	holders[1]()
	for i := 0; i < m; i++ {
		select {
		case release := <-waiterRelease:
			release()
		case <-time.After(2 * time.Second):
			t.Fatalf("only %d/%d waiters drained", i, m)
		}
	}
	waitForSlot(t, func() bool { return s.Inflight() == 0 && s.Waiting() == 0 }, 2*time.Second, "scheduler drained")
	if p := progressed.Load(); p != m {
		t.Fatalf("only %d/%d waiters ever progressed", p, m)
	}
}

// TestSchedulerPriorityOrderNoStarvation checks the fairness hook: with one
// slot held, parked waiters are woken highest-priority-first, ties broken FIFO
// by arrival — so no equal-priority waiter is passed over indefinitely. Waiters
// are parked one at a time (each after the prior one is observed enqueued) to
// pin a deterministic arrival order for the FIFO tiebreak.
func TestSchedulerPriorityOrderNoStarvation(t *testing.T) {
	t.Parallel()
	s := microagent.NewScheduler(1)

	held, err := s.Acquire(context.Background(), 0)
	if err != nil {
		t.Fatalf("Acquire the one slot: %v", err)
	}

	// (label, priority) in arrival order. Two priority-5 entries witness the
	// FIFO tiebreak; the low-priority "lo" witnesses it still gets served (no
	// starvation), just last.
	type wp struct {
		label string
		prio  int
	}
	arrivals := []wp{
		{"lo", 1},
		{"hi-a", 5},
		{"hi-b", 5},
		{"mid", 3},
	}
	woke := make(chan string, len(arrivals))
	for i, a := range arrivals {
		go func() {
			release, err := s.Acquire(context.Background(), a.prio)
			if err != nil {
				t.Errorf("waiter %s Acquire: %v", a.label, err)
				return
			}
			woke <- a.label
			release()
		}()
		// Ensure this waiter is enqueued before the next, fixing arrival order.
		want := i + 1
		waitForSlot(t, func() bool { return s.Waiting() == want }, 2*time.Second, "waiter enqueued in order")
	}

	// Free the slot; the released chain then hands off waiter-to-waiter.
	held()

	got := make([]string, 0, len(arrivals))
	for range arrivals {
		select {
		case label := <-woke:
			got = append(got, label)
		case <-time.After(2 * time.Second):
			t.Fatalf("only %d/%d waiters woke", len(got), len(arrivals))
		}
	}
	want := []string{"hi-a", "hi-b", "mid", "lo"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("wake order = %v, want %v", got, want)
		}
	}
}

// TestSchedulerCloseDrainsWaiters checks Close wakes every parked waiter with
// ErrSchedulerClosed and refuses all later Acquire, and is idempotent.
func TestSchedulerCloseDrainsWaiters(t *testing.T) {
	t.Parallel()
	s := microagent.NewScheduler(1)

	held, err := s.Acquire(context.Background(), 0)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	defer held()

	const m = 4
	errs := make(chan error, m)
	for i := 0; i < m; i++ {
		go func() {
			_, err := s.Acquire(context.Background(), 0)
			errs <- err
		}()
	}
	waitForSlot(t, func() bool { return s.Waiting() == m }, 2*time.Second, "waiters parked")

	s.Close()
	s.Close() // idempotent.

	for i := 0; i < m; i++ {
		select {
		case err := <-errs:
			if !errors.Is(err, microagent.ErrSchedulerClosed) {
				t.Fatalf("parked waiter got %v, want ErrSchedulerClosed", err)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("only %d/%d parked waiters drained on Close", i, m)
		}
	}

	// Every later Acquire refuses.
	if _, err := s.Acquire(context.Background(), 0); !errors.Is(err, microagent.ErrSchedulerClosed) {
		t.Fatalf("post-Close Acquire got %v, want ErrSchedulerClosed", err)
	}
}

// TestSchedulerContextCancelWithdraws checks a parked waiter whose context is
// cancelled returns ctx.Err() and is removed from the queue, freeing no slot it
// never held and leaving the held slots intact.
func TestSchedulerContextCancelWithdraws(t *testing.T) {
	t.Parallel()
	s := microagent.NewScheduler(1)

	held, err := s.Acquire(context.Background(), 0)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	defer held()

	ctx, cancel := context.WithCancel(context.Background())
	res := make(chan error, 1)
	go func() {
		_, err := s.Acquire(ctx, 0)
		res <- err
	}()
	waitForSlot(t, func() bool { return s.Waiting() == 1 }, 2*time.Second, "waiter parked")

	cancel()
	select {
	case err := <-res:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled waiter got %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancelled waiter never returned")
	}
	waitForSlot(t, func() bool { return s.Waiting() == 0 }, 2*time.Second, "waiter withdrawn from queue")
	if inf := s.Inflight(); inf != 1 {
		t.Fatalf("Inflight()=%d after a cancel, want 1 (held slot intact)", inf)
	}
}

// TestSchedulerAlreadyCancelledCtx checks Acquire with an already-cancelled
// context refuses immediately and takes no slot.
func TestSchedulerAlreadyCancelledCtx(t *testing.T) {
	t.Parallel()
	s := microagent.NewScheduler(2)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	release, err := s.Acquire(ctx, 0)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	release() // must be a safe no-op.
	if inf := s.Inflight(); inf != 0 {
		t.Fatalf("Inflight()=%d after a refused Acquire, want 0", inf)
	}
}

// TestSchedulerReleaseIdempotent checks the returned release is safe to call
// more than once (deferred + explicit): the extra calls free no phantom slot.
func TestSchedulerReleaseIdempotent(t *testing.T) {
	t.Parallel()
	s := microagent.NewScheduler(1)
	release, err := s.Acquire(context.Background(), 0)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	release()
	release() // second call is a no-op, not a second decrement.
	if inf := s.Inflight(); inf != 0 {
		t.Fatalf("Inflight()=%d after double release, want 0", inf)
	}
	// The slot is genuinely free again (limit still enforced at 1).
	r2, err := s.Acquire(context.Background(), 0)
	if err != nil {
		t.Fatalf("re-Acquire after double release: %v", err)
	}
	r2()
}

// TestNewSchedulerDefaultSlots checks a non-positive K selects DefaultSlots and
// a positive K is honored.
func TestNewSchedulerDefaultSlots(t *testing.T) {
	t.Parallel()
	for _, k := range []int{0, -1} {
		if got := microagent.NewScheduler(k).Limit(); got != microagent.DefaultSlots {
			t.Fatalf("NewScheduler(%d).Limit() = %d, want DefaultSlots=%d", k, got, microagent.DefaultSlots)
		}
	}
	if got := microagent.NewScheduler(3).Limit(); got != 3 {
		t.Fatalf("NewScheduler(3).Limit() = %d, want 3", got)
	}
}

// TestSchedulingGatewayPriorityFunc checks Model passthrough, the WithPriority /
// PriorityFromContext default seam, and that SetPriorityFunc overrides the hook
// (and a nil restores the default).
func TestSchedulingGatewayPriorityFunc(t *testing.T) {
	t.Parallel()
	inner := &concGateway{max: 8}
	gw := microagent.NewSchedulingGateway(inner, microagent.NewScheduler(8))
	if gw.Model() != "conc" {
		t.Fatalf("Model()=%q, want passthrough %q", gw.Model(), "conc")
	}

	// Default hook reads WithPriority off the context.
	if got := microagent.PriorityFromContext(microagent.WithPriority(context.Background(), 7)); got != 7 {
		t.Fatalf("PriorityFromContext = %d, want 7", got)
	}
	if got := microagent.PriorityFromContext(context.Background()); got != 0 {
		t.Fatalf("PriorityFromContext default = %d, want 0", got)
	}

	// Override the hook; a call should route the override's value in.
	var seen atomic.Int64
	gw.SetPriorityFunc(func(context.Context) int {
		seen.Store(42)
		return 42
	})
	if _, err := gw.Complete(context.Background(), nil, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if seen.Load() != 42 {
		t.Fatal("SetPriorityFunc override was not consulted")
	}

	// A nil restores the default (must not panic on the next call).
	gw.SetPriorityFunc(nil)
	if _, err := gw.Complete(context.Background(), nil, nil); err != nil {
		t.Fatalf("Complete after nil SetPriorityFunc: %v", err)
	}
}

// TestSchedulingGatewayRefusesWhenClosed checks a Complete on a closed
// scheduler returns the closed error and never dials the wrapped gateway.
func TestSchedulingGatewayRefusesWhenClosed(t *testing.T) {
	t.Parallel()
	inner := &concGateway{max: 1}
	sched := microagent.NewScheduler(1)
	gw := microagent.NewSchedulingGateway(inner, sched)
	sched.Close()

	if _, err := gw.Complete(context.Background(), nil, nil); !errors.Is(err, microagent.ErrSchedulerClosed) {
		t.Fatalf("Complete on closed scheduler got %v, want ErrSchedulerClosed", err)
	}
	if got := inner.calls.Load(); got != 0 {
		t.Fatalf("wrapped gateway was dialed %d times despite closed slot gate", got)
	}
	if gw.Scheduler() != sched {
		t.Fatal("Scheduler() did not expose the underlying scheduler")
	}
}
