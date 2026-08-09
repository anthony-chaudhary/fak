package microagent_test

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

// The BudgetQueue is the TOKEN layer for a microagent host (#2021, epic #2000
// M21): it bounds the sum of tokens reserved by in-flight admits to a configured
// budget, PARKING admits that would exceed it rather than 429-storming the
// provider, and draining the parked queue fairly (priority-then-FIFO, strictly
// in order) as capacity frees. It is the token-budget analogue of the SLOT-layer
// Scheduler (slotsched.go): slots bound concurrency (calls ≤ K), the budget
// bounds tokens (Σ reserved ≤ budget). These tests are the acceptance witnesses.

// TestBudgetQueueNeverExceedsBudget is the first acceptance witness: under a
// burst of N≫capacity admits, the sum of tokens held at once never exceeds the
// budget, and every admit eventually completes. The peak is sampled from inside
// the reservation (between Admit and release) so a breach would be seen directly.
func TestBudgetQueueNeverExceedsBudget(t *testing.T) {
	t.Parallel()
	const (
		budget = 100
		cost   = 10 // at most budget/cost = 10 admits held at once.
		n      = 300
	)
	q := microagent.NewBudgetQueue(budget)

	var live atomic.Int64 // Σ tokens held right now (sampled inside the gate).
	var peak atomic.Int64 // high-water mark of live.
	var done atomic.Int64 // admits that ran to completion.

	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			release, err := q.Admit(context.Background(), cost, 0)
			if err != nil {
				t.Errorf("Admit: %v", err)
				return
			}
			defer release()
			cur := live.Add(cost)
			for {
				old := peak.Load()
				if cur <= old || peak.CompareAndSwap(old, cur) {
					break
				}
			}
			time.Sleep(200 * time.Microsecond) // hold so contention is real.
			live.Add(-cost)
			done.Add(1)
		}()
	}
	wg.Wait()

	if got := peak.Load(); got > budget {
		t.Fatalf("held-token peak %d exceeded budget=%d", got, budget)
	}
	if got := done.Load(); got != n {
		t.Fatalf("only %d/%d admits completed", got, n)
	}
	if u := q.Used(); u != 0 {
		t.Fatalf("queue leaked tokens: Used()=%d, want 0", u)
	}
	if w := q.Waiting(); w != 0 {
		t.Fatalf("queue leaked waiters: Waiting()=%d, want 0", w)
	}
}

// TestBudgetQueueDrainsPriorityThenFIFO witnesses fair drain order: with the
// budget saturated and three admits parked, freeing capacity must wake them
// highest-priority-first, ties broken by arrival (FIFO). Each parked admit
// reserves the whole budget, so exactly one is granted at a time and the grant
// order is observed directly.
func TestBudgetQueueDrainsPriorityThenFIFO(t *testing.T) {
	t.Parallel()
	const budget = 10
	q := microagent.NewBudgetQueue(budget)

	holder, err := q.Admit(context.Background(), budget, 0) // saturate.
	if err != nil {
		t.Fatalf("holder Admit: %v", err)
	}

	grants := make(chan string, 3)
	rel := make(chan struct{}) // gates each woken admit's release.
	park := func(name string, prio int) {
		go func() {
			release, err := q.Admit(context.Background(), budget, prio)
			if err != nil {
				t.Errorf("%s Admit: %v", name, err)
				return
			}
			grants <- name
			<-rel // hold the whole budget until told to release.
			release()
		}()
	}
	// Force arrival order w1 -> w2 -> w3 by waiting for each to park.
	park("w1", 0)
	waitForSlot(t, func() bool { return q.Waiting() == 1 }, 2*time.Second, "w1 parked")
	park("w2", 5)
	waitForSlot(t, func() bool { return q.Waiting() == 2 }, 2*time.Second, "w2 parked")
	park("w3", 0)
	waitForSlot(t, func() bool { return q.Waiting() == 3 }, 2*time.Second, "w3 parked")

	holder() // free the budget: w2 (prio 5) first, then w1 (prio 0, earlier), then w3.
	for i, exp := range []string{"w2", "w1", "w3"} {
		select {
		case got := <-grants:
			if got != exp {
				t.Fatalf("grant %d: got %s, want %s (priority-then-FIFO)", i, got, exp)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("grant %d: timed out waiting for %s", i, exp)
		}
		rel <- struct{}{} // release the granted admit so the next can be admitted.
	}
	waitForSlot(t, func() bool { return q.Waiting() == 0 && q.Used() == 0 }, 2*time.Second, "queue drained")
}

// TestBudgetQueueParkedWaitersDoNotBusyWait is the no-busy-wait witness: admits
// that cannot be reserved PARK — they make zero progress and their count stays
// stable until capacity frees, and each freeing wakes exactly one. A busy-wait
// or spurious wake would show as progress with no release, or a parked count
// that drifts on its own.
func TestBudgetQueueParkedWaitersDoNotBusyWait(t *testing.T) {
	t.Parallel()
	const (
		budget = 10
		m      = 5 // surplus admits, each reserving the whole budget.
	)
	q := microagent.NewBudgetQueue(budget)

	holder, err := q.Admit(context.Background(), budget, 0)
	if err != nil {
		t.Fatalf("holder Admit: %v", err)
	}

	var progressed atomic.Int64
	waiterRelease := make(chan func(), m)
	for i := 0; i < m; i++ {
		go func() {
			release, err := q.Admit(context.Background(), budget, 0)
			if err != nil {
				t.Errorf("waiter Admit: %v", err)
				return
			}
			progressed.Add(1)
			waiterRelease <- release
		}()
	}

	waitForSlot(t, func() bool { return q.Waiting() == m }, 2*time.Second, "all waiters parked")

	// Negative witness: no capacity freed, so parked count is stable, progress zero.
	time.Sleep(50 * time.Millisecond)
	if w := q.Waiting(); w != m {
		t.Fatalf("parked count drifted with no capacity freed: Waiting()=%d, want %d", w, m)
	}
	if p := progressed.Load(); p != 0 {
		t.Fatalf("a parked admit progressed with no capacity freed: %d", p)
	}
	if u := q.Used(); u != budget {
		t.Fatalf("Used()=%d while budget fully held, want %d", u, budget)
	}

	holder() // free the whole budget: exactly one admit (cost==budget) may wake.
	waitForSlot(t, func() bool { return progressed.Load() == 1 }, 2*time.Second, "one waiter woken")
	time.Sleep(50 * time.Millisecond)
	if p := progressed.Load(); p != 1 {
		t.Fatalf("more than one admit woke from a single freeing: %d", p)
	}
	if w := q.Waiting(); w != m-1 {
		t.Fatalf("Waiting()=%d after one freeing, want %d", w, m-1)
	}

	// Drain the rest: each release admits exactly one more, until empty.
	for i := 0; i < m; i++ {
		select {
		case release := <-waiterRelease:
			release()
		case <-time.After(2 * time.Second):
			t.Fatalf("waiter %d never completed", i)
		}
	}
	waitForSlot(t, func() bool { return q.Waiting() == 0 && q.Used() == 0 }, 2*time.Second, "queue settled empty")
}

// TestBudgetQueueRejectsCostExceedingBudget: an admit whose cost can never fit
// the whole budget must be refused immediately (not parked forever) — otherwise
// it would deadlock. It must consume no budget and park no waiter.
func TestBudgetQueueRejectsCostExceedingBudget(t *testing.T) {
	t.Parallel()
	q := microagent.NewBudgetQueue(10)
	if _, err := q.Admit(context.Background(), 11, 0); !errors.Is(err, microagent.ErrCostExceedsBudget) {
		t.Fatalf("Admit(cost>budget): err=%v, want ErrCostExceedsBudget", err)
	}
	if w := q.Waiting(); w != 0 {
		t.Fatalf("over-budget admit parked a waiter: Waiting()=%d", w)
	}
	if u := q.Used(); u != 0 {
		t.Fatalf("over-budget admit consumed budget: Used()=%d", u)
	}
}

// TestBudgetQueueRejectsNonPositiveCost: a zero/negative cost is a caller bug,
// not a free pass; it is refused so accounting stays honest.
func TestBudgetQueueRejectsNonPositiveCost(t *testing.T) {
	t.Parallel()
	q := microagent.NewBudgetQueue(10)
	for _, c := range []int64{0, -5} {
		if _, err := q.Admit(context.Background(), c, 0); !errors.Is(err, microagent.ErrNonPositiveCost) {
			t.Fatalf("Admit(cost=%d): err=%v, want ErrNonPositiveCost", c, err)
		}
	}
}

// TestBudgetQueueContextCancelWithdrawsWaiter: a parked admit whose context is
// cancelled withdraws from the queue, returns ctx.Err(), and never later
// consumes budget when capacity frees.
func TestBudgetQueueContextCancelWithdrawsWaiter(t *testing.T) {
	t.Parallel()
	const budget = 10
	q := microagent.NewBudgetQueue(budget)

	holder, err := q.Admit(context.Background(), budget, 0)
	if err != nil {
		t.Fatalf("holder Admit: %v", err)
	}
	defer holder()

	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() {
		_, err := q.Admit(ctx, budget, 0)
		errc <- err
	}()
	waitForSlot(t, func() bool { return q.Waiting() == 1 }, 2*time.Second, "waiter parked")

	cancel()
	select {
	case err := <-errc:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled Admit: err=%v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancelled Admit did not return")
	}
	waitForSlot(t, func() bool { return q.Waiting() == 0 }, 2*time.Second, "waiter withdrawn")
	if u := q.Used(); u != budget {
		t.Fatalf("Used()=%d, want %d (holder still holds; withdrawn waiter took nothing)", u, budget)
	}
}

// TestBudgetQueueCloseDrainsWaiters: Close wakes every parked admit with
// ErrBudgetQueueClosed and refuses all future admits.
func TestBudgetQueueCloseDrainsWaiters(t *testing.T) {
	t.Parallel()
	const (
		budget = 10
		m      = 4
	)
	q := microagent.NewBudgetQueue(budget)

	holder, err := q.Admit(context.Background(), budget, 0)
	if err != nil {
		t.Fatalf("holder Admit: %v", err)
	}
	defer holder()

	errc := make(chan error, m)
	for i := 0; i < m; i++ {
		go func() {
			_, err := q.Admit(context.Background(), budget, 0)
			errc <- err
		}()
	}
	waitForSlot(t, func() bool { return q.Waiting() == m }, 2*time.Second, "all waiters parked")

	q.Close()
	for i := 0; i < m; i++ {
		select {
		case err := <-errc:
			if !errors.Is(err, microagent.ErrBudgetQueueClosed) {
				t.Fatalf("waiter %d: err=%v, want ErrBudgetQueueClosed", i, err)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("waiter %d not drained by Close", i)
		}
	}
	if _, err := q.Admit(context.Background(), 1, 0); !errors.Is(err, microagent.ErrBudgetQueueClosed) {
		t.Fatalf("Admit after Close: err=%v, want ErrBudgetQueueClosed", err)
	}
}

// TestBudgetQueueDrainsInOrderWithoutQueueJump witnesses the fairness rule that
// makes drain "in order": a newcomer whose cost would fit the free capacity must
// still PARK behind an already-waiting admit (no queue jump), and when capacity
// frees the head is granted even if that leaves capacity idle rather than
// admitting a smaller admit out of order.
func TestBudgetQueueDrainsInOrderWithoutQueueJump(t *testing.T) {
	t.Parallel()
	const budget = 10
	q := microagent.NewBudgetQueue(budget)

	a, err := q.Admit(context.Background(), 4, 0) // used = 4, 6 free.
	if err != nil {
		t.Fatalf("A Admit: %v", err)
	}

	grants := make(chan string, 2)
	relW1 := make(chan struct{})
	go func() {
		r, err := q.Admit(context.Background(), 8, 0) // needs 8; 4+8 > 10 -> parks.
		if err != nil {
			t.Errorf("w1 Admit: %v", err)
			return
		}
		grants <- "w1"
		<-relW1
		r()
	}()
	waitForSlot(t, func() bool { return q.Waiting() == 1 }, 2*time.Second, "w1 parked")

	relW2 := make(chan struct{})
	go func() {
		r, err := q.Admit(context.Background(), 4, 0) // 4 would fit the 6 free, but must not jump w1.
		if err != nil {
			t.Errorf("w2 Admit: %v", err)
			return
		}
		grants <- "w2"
		<-relW2
		r()
	}()
	waitForSlot(t, func() bool { return q.Waiting() == 2 }, 2*time.Second, "w2 parked behind w1")

	if u := q.Used(); u != 4 {
		t.Fatalf("Used()=%d before release, want 4 (no waiter admitted while A holds)", u)
	}

	a() // free A's 4 -> used 0; drain must grant w1 (head), then STOP (w2 no longer fits behind it).
	select {
	case got := <-grants:
		if got != "w1" {
			t.Fatalf("first grant=%s, want w1 (strictly in order)", got)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("w1 not granted after A released")
	}
	waitForSlot(t, func() bool { return q.Used() == 8 }, 2*time.Second, "w1 holds 8")
	time.Sleep(50 * time.Millisecond)
	if w := q.Waiting(); w != 1 {
		t.Fatalf("Waiting()=%d, want 1 (w2 still parked; capacity left idle, not reordered)", w)
	}
	select {
	case got := <-grants:
		t.Fatalf("w2 granted (%s) out of order while w1 held the budget", got)
	default:
	}

	close(relW1) // w1 releases 8 -> used 0; now w2 fits.
	select {
	case got := <-grants:
		if got != "w2" {
			t.Fatalf("second grant=%s, want w2", got)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("w2 not granted after w1 released")
	}
	close(relW2)
	waitForSlot(t, func() bool { return q.Waiting() == 0 && q.Used() == 0 }, 2*time.Second, "queue settled empty")
}

// TestBudgetQueueShedsWithBackpressureWhenParkQueueFull witnesses the BOUND on
// the park queue (#2021 scope 1) and the backpressure signal it raises (scope
// 2): once the budget is spent AND MaxQueue admits are parked, the next Admit
// is SHED immediately with ErrBackpressure — it does not park (which would let
// the queue grow without limit), does not spin, and reserves nothing.
func TestBudgetQueueShedsWithBackpressureWhenParkQueueFull(t *testing.T) {
	t.Parallel()
	const (
		budget   = 10
		maxQueue = 2
	)
	q := microagent.NewBudgetQueueDepth(budget, maxQueue)

	holder, err := q.Admit(context.Background(), budget, 0) // spend the whole budget.
	if err != nil {
		t.Fatalf("holder Admit: %v", err)
	}

	var wg sync.WaitGroup // joined before the test returns, so no late t.Errorf.
	wg.Add(maxQueue)
	for i := 0; i < maxQueue; i++ {
		go func() {
			defer wg.Done()
			release, err := q.Admit(context.Background(), budget, 0)
			if err != nil {
				t.Errorf("parked Admit: %v", err)
				return
			}
			release()
		}()
	}
	waitForSlot(t, func() bool { return q.Waiting() == maxQueue }, 2*time.Second, "park queue at its bound")

	// The bound binds: this admit is refused, not queued.
	start := time.Now()
	if _, err := q.Admit(context.Background(), budget, 0); !errors.Is(err, microagent.ErrBackpressure) {
		t.Fatalf("Admit past the park bound: err=%v, want ErrBackpressure", err)
	}
	if el := time.Since(start); el > time.Second {
		t.Fatalf("shed blocked for %s — it parked instead of refusing", el)
	}
	if w := q.Waiting(); w != maxQueue {
		t.Fatalf("Waiting()=%d after a shed, want %d (a shed admit must not park)", w, maxQueue)
	}
	if u := q.Used(); u != budget {
		t.Fatalf("Used()=%d after a shed, want %d (a shed admit must reserve nothing)", u, budget)
	}

	holder() // capacity returns: the parked admits still drain normally.
	wg.Wait()
	waitForSlot(t, func() bool { return q.Waiting() == 0 && q.Used() == 0 }, 2*time.Second, "queue settled empty")
}

// TestBudgetQueuePressureReadoutTracksShedState witnesses the poll-before-enqueue
// half of the backpressure signal: Pressure reports one consistent snapshot, and
// its Shed flag goes false -> true -> false as the park queue fills and drains,
// so an enrolling caller can stop enqueuing without first earning a refusal. It
// also pins the default bound applied when no depth is configured.
func TestBudgetQueuePressureReadoutTracksShedState(t *testing.T) {
	t.Parallel()
	if got := microagent.NewBudgetQueue(0).MaxQueue(); got != microagent.DefaultMaxParkedAdmits {
		t.Fatalf("NewBudgetQueue MaxQueue()=%d, want DefaultMaxParkedAdmits=%d", got, microagent.DefaultMaxParkedAdmits)
	}
	if got := microagent.NewBudgetQueueDepth(10, -1).MaxQueue(); got != microagent.DefaultMaxParkedAdmits {
		t.Fatalf("non-positive depth MaxQueue()=%d, want DefaultMaxParkedAdmits=%d", got, microagent.DefaultMaxParkedAdmits)
	}

	const (
		budget   = 10
		maxQueue = 2
	)
	q := microagent.NewBudgetQueueDepth(budget, maxQueue)

	want := microagent.BudgetPressure{Budget: budget, MaxQueue: maxQueue}
	if got := q.Pressure(); got != want {
		t.Fatalf("idle Pressure()=%+v, want %+v", got, want)
	}

	holder, err := q.Admit(context.Background(), budget, 0)
	if err != nil {
		t.Fatalf("holder Admit: %v", err)
	}
	// Budget spent but the queue has room: NOT shed — upstream may still enqueue.
	want = microagent.BudgetPressure{Budget: budget, Used: budget, MaxQueue: maxQueue}
	if got := q.Pressure(); got != want {
		t.Fatalf("saturated-budget Pressure()=%+v, want %+v (room to park is not backpressure)", got, want)
	}

	var wg sync.WaitGroup
	wg.Add(maxQueue)
	for i := 0; i < maxQueue; i++ {
		go func() {
			defer wg.Done()
			release, err := q.Admit(context.Background(), budget, 0)
			if err != nil {
				t.Errorf("parked Admit: %v", err)
				return
			}
			release()
		}()
	}
	waitForSlot(t, func() bool { return q.Pressure().Shed }, 2*time.Second, "pressure reports shed")

	want = microagent.BudgetPressure{Budget: budget, Used: budget, Waiting: maxQueue, MaxQueue: maxQueue, Shed: true}
	if got := q.Pressure(); got != want {
		t.Fatalf("full-queue Pressure()=%+v, want %+v", got, want)
	}

	holder()
	wg.Wait()
	waitForSlot(t, func() bool { return !q.Pressure().Shed }, 2*time.Second, "pressure clears after drain")
	want = microagent.BudgetPressure{Budget: budget, MaxQueue: maxQueue}
	if got := q.Pressure(); got != want {
		t.Fatalf("drained Pressure()=%+v, want %+v", got, want)
	}
}

// TestBudgetQueueSustainedOversubscriptionShedsThenDrains is the #2021 acceptance
// witness under the BOUNDED regime: with 50 admits contending for a budget that
// fits exactly one, no request is ever reserved above budget, the park queue
// never grows past MaxQueue (the surplus is shed as typed ErrBackpressure, not
// queued and not spun on), and every parked admit still drains when capacity
// returns. The settle is deterministic — the whole budget is held while the
// burst arrives, so exactly MaxQueue park and the rest are shed.
func TestBudgetQueueSustainedOversubscriptionShedsThenDrains(t *testing.T) {
	t.Parallel()
	const (
		budget   = 10
		maxQueue = 3
		n        = 50 // sustained over-subscription.
	)
	q := microagent.NewBudgetQueueDepth(budget, maxQueue)

	holder, err := q.Admit(context.Background(), budget, 0) // budget spent up front.
	if err != nil {
		t.Fatalf("holder Admit: %v", err)
	}

	var admitted, shed atomic.Int64
	var peak atomic.Int64 // high-water Σ reserved, sampled from inside the gate.
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			release, err := q.Admit(context.Background(), budget, 0)
			if err != nil {
				if !errors.Is(err, microagent.ErrBackpressure) {
					t.Errorf("Admit: err=%v, want nil or ErrBackpressure", err)
				}
				shed.Add(1)
				return
			}
			defer release()
			cur := q.Used()
			for {
				old := peak.Load()
				if cur <= old || peak.CompareAndSwap(old, cur) {
					break
				}
			}
			admitted.Add(1)
		}()
	}
	waitForSlot(t, func() bool { return shed.Load() == n-maxQueue && q.Waiting() == maxQueue },
		10*time.Second, "surplus shed and park queue at its bound")

	// Over-subscription is bounded, not absorbed: budget held flat, queue capped.
	if u := q.Used(); u != budget {
		t.Fatalf("Used()=%d under sustained over-subscription, want %d (nothing reserved above budget)", u, budget)
	}
	if a := admitted.Load(); a != 0 {
		t.Fatalf("%d admits granted while the whole budget was held", a)
	}
	if p := q.Pressure(); !p.Shed || p.Waiting != maxQueue {
		t.Fatalf("Pressure()=%+v, want Shed=true Waiting=%d", p, maxQueue)
	}

	holder() // capacity returns: the parked admits drain, none is dropped.
	wg.Wait()

	if got := peak.Load(); got > budget {
		t.Fatalf("reserved-token peak %d exceeded budget=%d", got, budget)
	}
	if a := admitted.Load(); a != maxQueue {
		t.Fatalf("%d parked admits drained, want %d (a parked admit must never be dropped)", a, maxQueue)
	}
	if s := shed.Load(); s != n-maxQueue {
		t.Fatalf("%d admits shed, want %d", s, n-maxQueue)
	}
	if u, w := q.Used(), q.Waiting(); u != 0 || w != 0 {
		t.Fatalf("queue did not settle empty: Used()=%d Waiting()=%d", u, w)
	}
}
